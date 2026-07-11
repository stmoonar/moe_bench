# 22 EP 第三轮计划:microbench 归因后的口径修复与优化路线

> **导入说明(2026-07-11)**:本文来自并行工作区(SYNC07101059_2)的 EP 主线,
> 原编号 docs/19(与本分支 TP 系列的 19 冲突,重编为 22,内容未改)。它是 EP
> 主线的当前任务清单;microbench 套件与结果已一并导入
> `microbench/`、`microbench/results/20260710_071848/`。
> **TP 分支读者**:§0 的平台事实表(push/pull/copy engine/干扰/comm SM)是
> 两条线共享的依据,TP 侧的映射与任务见 docs/23。

**日期**:2026-07-10
**依据**:microbench 全套实测(`microbench/results/20260710_071848/`,分析报告
`mb3_report_sched205us.md`)。本文取代 docs/11 成为**当前任务清单**;docs/11 保留作历史。
**执行者须知**:本文写给低上下文成本的执行模型,每个任务给出:改哪个文件、怎么改、
怎么验收、预期收益、风险回退。开工前先读 HANDOFF.md §4(踩坑索引),改 kernel 前
必读 docs/08(远端原子禁用)与 docs/05(group::store 行映射)。

---

## 0. Microbench 结论摘要(所有任务的依据,勿重测已有结论)

形状:E=64/topk8/hidden4096/inter3072,bf16 EP 4卡(9,11,13,15)。
token 数一律指**全 rank 总数**(每卡 = /4)。

**收益分解**(schedule 按 graph 口径 205µs 计入):

| tokens | serial | tkfused(fair) | 加速 | 计算收益占比 | 通信+重叠占比 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 2912µs | 2181µs | 1.34× | 32% | 68% |
| 8192 | 10942µs | 8116µs | 1.35× | 46% | 54% |

- TK 计算链比 vLLM triton 快 1.2~1.3×(但 vLLM 用的是**未调优**默认 config,见 P3);
- 只重叠不换算子的上限 1.36~1.37×;TK 算子+全重叠上限 **1.58×@2048 / 1.63×@8192**;
- 当前距上限的差距 = 融合损失:L0 暴露 17~25%、L1 暴露 26~37%,8192 时合计 ~1.7ms。

**平台事实**(mb6/mb7,做方案选择时直接引用,不要再猜):

| 事实 | 数值 |
|---|---|
| SM push(写远端,强路径) | 50.9GB/s,**4 个 SM 打满**,4 卡并发**零退化**(49.8/卡) |
| SM pull(读远端,弱路径) | 相邻对 50、跨对 29GB/s,要 16 SM 打满,4 卡并发**退化到 23.5/卡** |
| copy engine(memcpyPeer) | 54~56GB/s,零 SM 占用(并发行为未测,见 T14) |
| NCCL AG/RS busbw | ~30GB/s;NCCL 小包 24µs vs TK barrier 9~11µs / 信号 RTT 2.1~2.9µs |
| 8KB 散布行 vs 顺序 | 无差别(行粒度不是瓶颈,方向才是) |
| 通信/计算干扰 | ≈0(both≈max(comm,comp);编排=纯资源划分问题,intra-SM 无额外魔法) |
| comm SM 让渡代价(k=16) | GEMM 慢 8~16%(8192 时 +330µs);k=4 只慢 ~4% |
| 最优 comm SM 数 | ≤1024 tokens→8,2048→12,≥5120→16(pull 路径下;mb5) |
| 小 token 地板 | ≤1024 tokens 时 padding 率高(128 时 7.1×),tk_l0 恒为 ~708µs;128 时 e2e 与 serial 打平 |

---

## 1. P0 口径修复(先做,否则报数不可信;全部是小改动)

### P1 修 schedule 计时漏计(tk_scheme.py)

- **问题**:`tk_scheme.py` run() 里 GPU schedule 重建只在 `combine_mode == "prered"`
  时执行并计时(约第 681 行),而默认 combine 是 `prered_push` → 默认路径 e2e
  (HANDOFF 的 1963µs/1.48×)**没含 schedule 成本**。
- **改法**:该行条件
  `self.combine_mode == "prered"` → `self.combine_mode in ("prered", "prered_push")`。
  注意:`_sched_out` 在 setup 里已按 prered_push 加入 `push_expected_l1/recv_from`,
  builder 兼容,无需其它改动。
- **验收**:`CUDA_VISIBLE_DEVICES=... python -m moe_bench.tools.run_tkfused 64` 对拍
  仍过(rel_err ~4e-3);`bench_shape_4096` 的 tkfused e2e 应变为 **~2170µs**
  (1963 + graph sched ~205),speedup ~1.34×。
- **收尾**:更新 HANDOFF §2 的默认数字为 fair 口径,注明"旧 1.48× 未含 sched"。

### P2 mb4 补测 graph 口径的 sched(microbench/mb4_fusion.py)

- **问题**:mb4 的 `sched` 计时是 eager(~800µs),真实 run() 是 CUDA graph
  (~205µs,docs/14),mb3 目前靠 `--sched-ms 0.205` 人工覆盖。
- **改法**:mb4 worker 里在现有 `sched`(改名 `sched_eager`)旁增加 `sched_graph`:
  先 eager 跑一次 `_build_schedules_gpu` 预热,再
  `g = torch.cuda.CUDAGraph(); with torch.cuda.graph(g): _build_schedules_gpu(...)`,
  计时函数 = 两个 all_gather(eager)+ `g.replay()`(与 run() 完全同构)。
  mb3_ratio 默认改用 `sched_graph_ms`(存在时),`--sched-ms` 保留为覆盖项。
- **验收**:重跑 mb4+mb3,`sched_graph` 应在 200~260µs 区间;报告不再需要手工覆盖。

### P3 vLLM triton 调优后复测计算收益(基线公平性)

- **问题**:mb1 日志显示 vLLM fused_moe 在用默认 config
  (`Config file not found ... E=16,N=3072,device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell.json`),
  即 triton 未调优 → "TK 计算快 1.2~1.3×"可能部分是没调优造成的。
- **做法**:一键脚本 `bash moe_bench/microbench/tune_vllm_moe.sh`(约 0.5~2h)。
  原理:`benchmark_moe.py --tune` 的 `--model` 只读 config.json 推形状、不加载
  权重,脚本会生成一个只含 config.json 的假 Mixtral 目录
  (num_local_experts=16、intermediate=3072、hidden=4096、bf16、--tp-size 1;
  **不能填 64**——EP 掩码后 kernel 只见 16 个本地专家,json 文件名里的 E 就是它)。
  - **topk 密度匹配**:运行时 M 个 gathered token × topk8 只有 1/4 命中本地,
    实际展开行 = M×2;config 查表键不含 topk,所以脚本默认用 **topk=2** 调优,
    每专家行数与真实负载一致。`tune_vllm_moe.sh 8` 可另调一份标准口径,
    用 mb1 实测选更快的那份留下。
  - `--batch-size` 传我们 bench 的 M 档(128~8192);若该版脚本不支持多值,
    查 --help 后逐档跑。产物 json 脚本会自动拷入 vllm 的 configs/ 目录。
- **验收**:重跑 mb1 + mb3:若 vllm_compute 明显变快,更新"计算收益占比"结论并写回
  本文档 §0;若不变,结论加固。**两种结果都要记录。**
- **注意**:调优只影响 vLLM 基线,不改我们的 kernel;serial e2e 也会变快,mb4 需同跑。

### P4 文档口径统一(AGENTS/HANDOFF)

- AGENTS.md 写"当前用 FP8",实际当前实现是 bf16(fp8 是后续,见 T15)→ 改为
  "当前 bf16,FP8 版在计划中(microbench 已支持 MB_PRECISION=fp8)"。
- 报数规范写入 HANDOFF §5:①一律 fair 口径(含 sched);②一律 token sweep
  (至少 2048/8192 两点),禁止单点结论;③microbench 复跑用
  `bash moe_bench/microbench/run_all.sh`,结果目录整个归档。

---

## 2. 优化任务(按数据支撑强度排序)

### T10 dispatch push 化 v2(主攻,数据支撑最强)

- **依据**:pull 是弱路径(跨对 29GB/s、4 卡并发 23.5GB/s、要 16 个 comm SM);
  push 是强路径(50.9GB/s、4 SM 打满、并发零退化)。L0 融合损失 8192 时 966µs,
  基本都是 pull 暴露 + 16 SM 让渡。
- **目标**:dispatch 数据面从 pull 改为 push(基于已验证的 push3 骨架,docs/09/10),
  同时 comm SM 从 16 降到 4~8。
- **已知坑与解法**(docs/10/11 已归因,不要重新踩):
  1. push3 在 NE=256 退化 → **目的块重排**:push 按目的 (dst_dev, dst 行块) 分组发送,
     让目的行块尽早齐活,而非按源 token 序;
  2. 信号数 ∝ 行块数 → **水位聚合**:沿用 T6-v1 的 per-card 水位选举
     (`preredpush` 的 election 模式镜像到 layer0),不要 per-block 信号;
  3. 远端原子仍然禁用(docs/08):完成检测只用"本地 atom.acq_rel 选举 +
     单写者 st.release.sys"。
- **步骤**:
  1. 在 `kernels/tk/tk_moe.cu` 的 dpush3 基础上做 `dpush4` namespace(目的块重排 +
     水位信号),schedule 侧在 `tk_scheme.py::_build_schedules` 增加按目的分组的
     push 顺序表(host golden),再镜像进 `_build_schedules_gpu`;
  2. 先做 push-only debug 入口对账 gathered 内容(照抄 `validate_push3.py` 方法论,
     30 迭代 × NE∈{64,128,256});
  3. 全链路对拍 `run_tkfused`,再 `time_dispatch` 隔离计时 + mb5 扫 k∈{4,8,12};
  4. 赢了改默认 `TK_DISPATCH=push4`,输了记录归因文档后冻结。
- **预期收益**(mb5/mb6/mb7 推算):8192 时 L0_fused 4947→~4100µs
  (让渡回收 ~300 + pull 暴露消除 ~500);2048 时 L0 1272→~1030µs。
  e2e 目标:2048 → ~1750µs(fair ~1.49×),8192 → ~7100µs(fair ~1.50×)。
- **回退**:环境开关 `TK_DISPATCH`,默认不切换直到全档位赢。

### T11 comm SM 数自适应(快赢,半天工作量)

- **依据**:mb5 最优 k 随 token 变化(≤1024→8,2048→12,≥5120→16);当前写死 16。
- **改法**:`tk_scheme.py` setup 里
  `self.num_comm_sms = 8 if num_tokens*world <= 1024 else (12 if <= 2048 else 16)`
  (阈值用全 rank 总数;T10 落地后此表要按 push 路径重扫 mb5 更新)。
- **验收**:mb5 重跑确认各档 e2e 取到各自最优;bench 512/rank 点 e2e 1976→~1940µs。
- **预期**:小 token -4~6%,2048 点 -2%。零风险(纯参数)。

### T12 L1 combine 融合损失压缩(第二大矿)

- **依据**:L1 融合损失 26~37%(8192 时 747µs),而 combine 跨卡字节只有
  ~18MB@8192(强路径只需 ~360µs)→ 大头不是带宽,是**串行化/尾巴**。
- **第一步是归因不是改代码**:写 `tools/time_layer1_push.py`(镜像 docs/12 的方法),
  把 `moe_gemm_prered_push_fused + moe_final_reduce_push` 拆成:纯 GEMM /
  GEMM+prered 散射(不推)/ +推流 / +final_reduce,定位 747µs 在哪段。
- **候选修复**(按归因结果选):
  a. 推流启动过晚 → 行块粒度 edge-trigger 提前(GEMM 写完一个 row block 即推该块
     相关 partial,而非等 job 齐);
  b. final_reduce_push 等水位的尾巴 → 按 src_dev 分片水位(细化等待粒度);
  c. prered 散射本身慢(docs/13 测过 +77µs 量级)→ 暂不动。
- **预期**:8192 时 L1_fused 2773→~2100µs;e2e 再省 ~600µs。
- **验收**:`validate_prered_push.py` 30 迭代 × 3 NE 全过 + mb4 L1pen 下降。

### T13 小 token 路径(decode 侧;若产品只关心大 batch 可降级为 P2 优先级)

- **依据**:≤1024 总 token 时 padding 率 2~7×,tk_l0 有 708µs 地板,128 时不赢 serial。
- **改法**(依次尝试,一个见效即停):
  1. 小 token 时用 `TK_ROW_BLOCK=64` 的预编译 .so(T5 已有编译开关,build.py 按
     row_block 缓存;setup 按 num_tokens 选择加载哪个);
  2. GEMM 跳过全 padding 的行块(schedule 已知每块 real 数,给 kernel 传 skip 表);
- **验收**:mb4 在 128/512 总 token 档 tkfused ≥ serial(当前 1.00×/1.09×)。
- **注意** docs/16 的教训:padding 行是"满效率算的",减 padding 可能掉进小 tile
  低效区 —— 必须全档位扫,只在小 token 档启用。

### T14 copy engine 数据面(远期探索,先补一个 microbench)

- **依据**:memcpyPeer 54~56GB/s 零 SM 占用,但**并发争用行为未测**。
- **第一步**:mb7 增加 `concurrent_memcpy` 用例(4 卡同时 memcpyPeerAsync ring,
  照抄现有 concurrent_ring 的写法换成 `ext.memcpy_peer`)。
- **若并发不退化**:原型 = dispatch 数据面用 batched memcpyPeerAsync(每 src 卡一条),
  SM 只做水位信号;这会把 comm SM 降到 ~0。**只有 mb7 数据支持才开工。**

### T15 FP8(既定方向,依赖 fp8 版 TK scheme)

- 通信字节减半(dispatch 100MB→50MB@8192)+ 计算翻倍吞吐;microbench 已支持
  `MB_PRECISION=fp8`(vLLM/NCCL 侧即开即用),TK 侧接入注意事项见
  `microbench/README.md`「切 FP8」一节(保持 entry 签名兼容则 microbench 零改动)。

---

## 3. 统一验证流程(每个任务收尾必做)

```bash
source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
cd /data/cinnzhang_vllm_td_test/xxy
nvidia-smi   # 确认卡空闲,优先 9,11,13,15
# 1) 正确性对拍(改 kernel/schedule 后必跑,三档 NE)
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.run_tkfused 64
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.run_tkfused 128
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.run_tkfused 256
# 2) 协议类改动加隔离裁决(validate_push3 / validate_prered_push 同款方法)
# 3) 性能回归:microbench 一键(token sweep 全档)
bash moe_bench/microbench/run_all.sh
python moe_bench/microbench/mb3_ratio.py --results <结果目录>   # P2 完成后不再需要 --sched-ms
# 4) 与上一次结果目录 diff 关键行(mb4 的 e2e/L0pen/L1pen,mb3 的收益分解)
```

报数规则:fair 口径(含 sched)、写明 token 档、附结果目录路径;每个任务完成后
提交 + 在 docs/ 落一篇归因/结果文档 + 更新 HANDOFF。

## 4. 执行顺序

```
P1 → P2 → P4(合计 ~1 天,先把口径钉死)
P3(可与下面并行;结果决定"计算收益"结论是否要改写)
T11(半天快赢)→ T10(主攻,3~5 天)→ T12(归因 1 天 + 修复 2~3 天)
T13(按产品需要)→ T14(先补 mb7 用例再决定)→ T15(等 fp8 scheme)
```

全部落地后的量化目标(bound 推算):**2048 总 token fair ~1.5×,8192 fair ~1.5~1.6×**
(理论上限 1.58×/1.63×)。

## 5. 明确不要做的事

- 不做 stream overlap 降级方案(AGENTS.md 红线:不许把 GEMM 拆 chunk 与通信分 stream);
- 不用远端原子做完成检测(docs/08,会丢增量);
- 不做单点 NE/token 的结论(docs/10 §7);
- 不重复造 microbench 已有的测量(§0 的平台事实直接引用);
- microbench 代码不要为单次实验魔改口径 —— 加环境变量开关并写回 README。
