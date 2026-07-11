# HANDOFF — TK 通算融合 MoE 进度交接

> 新 session 从这里接手。读完本文 + 最新一篇 docs/ 即可继续。

**最后更新**:2026-07-11(本轮:**TP 版 tile overlap(tktp scheme)落地,分支 tp_test**,
docs/19。kernel 新增 tpdisp(AG去重拉取⊕gate+up GEMM)+ preredpush TP 入口;layer1 复用
T6-v1 push+水位协议(TP 下 top-k 预归约完全本地、跨卡退化为稠密 RS)。本地已验证:调度表
host/GPU 逐元素一致 + 不变量全过(NE×分布×rank 全档),CPU 数据流模拟对拍 rel 3.7e-7。
**尚未上机**:一键脚本 tools/run_tp_all.sh(编译→裁决→对拍→bench→打包zip),等实测结果回流)

> **【EP 上一轮 2026-07-09】** T6-v0 预归约 + T4 gate+up + T3 schedule GPU化(公平口径)
+ T7 dispatch 去重 + T5 ROW_BLOCK=64,五项落地;NE=256 e2e 7131→5830µs 公平口径超 serial
7063 的 1.21×,见 docs/13~16。

## 1. 项目一句话

在 moe_bench 里用 ThunderKittens 实现通算融合的 EP MoE 层(dispatch⊕GEMM、GEMM⊕combine
单 kernel 融合,不做 stream overlap 降级方案),对比 vLLM serial baseline。
平台:16×RTX Pro 5000(sm120)PCIe,用 4 卡组(优先 9,11,13,15),无 NVLink、
**远端原子高并发不可靠**(docs/08)、无 multimem。

## 2. 当前状态(全部已提交,分支 tk_dev)

> **【仓库结构重构 2026-07-11】** ThunderKittens 已改为**正式 git submodule**
> (`.gitmodules`,钉在上游 `02e9acbd`,TK 核心零改动)。原先压在 TK 本地提交里的
> 自研代码 `ThunderKittens/tileoverlap/` 已整体迁至 **`kernels/tileoverlap/`**,
> 由 moe_bench 自己追踪(TK 内留有 `tileoverlap-archive` 分支存档旧提交,仅本地)。
> 路径已同步修正:各 `Makefile` include `../../../ThunderKittens/kernels/common.mk`、
> benchmark 脚本 sys.path 指向 `../../../ThunderKittens/kernels/parallel`、
> `kernels/tk/build.py` 的 `_COMMON` 指向 `kernels/tileoverlap/common/sm120_common.cuh`。
> 新 clone 后需 `git submodule update --init`。**服务器侧
> (`/data/cinnzhang_vllm_td_test/xxy/moe_bench`)尚未做同样的目录调整**,下次上
> 服务器时需同步(迁移目录 + git pull 本仓库),否则两边结构不一致。

> **【默认形状已改 2026-07-09】** 默认 shape 现为 **E=64, TOP_K=8, hidden=4096,
> gate_up=6144(intermediate=3072)**,512 token/rank, bf16 EP world=4(E_local=16,
> 每专家 ~256 token,padding 少)。config.py / configs/tk_ep_bf16.yaml / 各 tool 默认
> 已同步。此 shape 下 **tkfused 1963µs vs serial 2905µs = 1.48×**(公平口径,combine
> 预归约 push 化 T6-v1 默认,对拍 rel_err 4.43e-3 ok;`bench_shape_4096` / `run_tkfused`)。
> 下面 §2 里 NE=256/hidden=7168 的历史数字是旧 shape 的记录,保留备查。

- **正确性**:tkfused 全链路对拍 reference_moe 通过(bf16, EP, 4 卡, rel_err ~4.4e-3)。
- **性能(新默认 shape E=64/hidden=4096)**:tkfused **1963µs vs serial 2905µs**(快 1.48×,
  T6-v1 combine push 化默认;v0 prered 为 2331µs)。
- **combine 三条路径**:`TK_COMBINE=prered_push`(默认,T6-v1,docs/18,边算边推)|
  `prered`(v0,barrier+pull,docs/13)| `pull`(旧 moe_gemm_combine_fused)。
- **性能(旧 shape NE=256/hidden=7168,历史记录)**:tkfused 5832µs vs serial 7063µs(1.21×)。
- **combine 两条路径**:`TK_COMBINE=prered`(默认,T6-v0,docs/13,全档位优于 pull)|
  `pull`(旧 moe_gemm_combine_fused,保留)。
- **T4 gate+up 合并**:`TK_FUSE_GATEUP=1`(默认,pull dispatch)。
- **T3 schedule GPU 化**:`TK_GPU_SCHED=1`(默认,pull+prered 路径,CUDA graph,docs/14)。
- **dispatch 三条路径**:`TK_DISPATCH=pull`(默认,已对拍)| `push3`(正确,NE≤128 更快、
  NE=256 更慢,docs/10)| `push`/`push2`(冻结,docs/08)。
- **代码**:`tk_scheme.py`(scheme + host schedule)、`kernels/tk/tk_moe.cu`(gg/disp/
  dpush/dpush3/comb 五个 namespace)、`kernels/tk/sm120_common.cuh`(pcie_sync +
  grouped_gemm_sm120 模板)。
- **工具**:`tools/run_tkfused.py`(对拍)、`time_dispatch.py`(dispatch-only 隔离计时)、
  `time_layer1.py`(T1 layer1 5 点隔离归因)、`ncu_gemm_probe.py`(单卡 ncu)、
  `validate_push3.py` / `verify_push3_schedule.py`(协议裁决)、
  `reconcile_prereduce.py`(T6-v0 预归约表对账,docs/13)、
  `validate_prered.py`(T6-v0 全链路正确性:prered vs pull combine,30 迭代 × 3 NE)、
  `verify_schedule_gpu.py`(T3 GPU schedule vs host golden element-wise,docs/14)、
  `time_schedule.py`(T3 schedule 重算成本:all_gather/eager/graph 分解)、
  `validate_prered_push.py`(T6-v1 push combine 正确性,30 迭代 × 3 NE,docs/18)、
  `analyze_overlap.py`(通算重叠 + 融合损失分析,docs/17/18)、
  `bench_shape_4096.py`(默认 shape tkfused vs serial)。
  ⚠️ `time_layer1.py` / `ncu_gemm_probe.py` 及 `tk_moe.cu` 的 T1 debug 入口
  (`gg::entry_nb`、`comb::combine_only_entry`)在服务器侧,**尚未同步入本仓库提交**。

## 3. 下一步:按 docs/11 路线图执行

**docs/11_评审_layer1归因与下一步优化路线图.md 是当前的任务清单**,按优先级:

| # | 任务 | 状态 |
|---|---|---|
| T1 | ncu 归因 layer1 GEMM 为何只有 71 TFLOP/s(fence/SM让渡/L2) | **✅ 完成(docs/12):假设推翻,慢在 combine gather 94~99%,非 fusion** |
| T2 | combine epilogue 换 push3 式选举信号(预期 layer1 −1ms) | **❄️ 冻结(T1 止损:fence 仅 1.2~2.8%,收益 ≤0.05ms)** |
| T3 | schedule GPU 化并计入 run()(公平性,报数前必须) | **✅ 完成(docs/14):默认路径 6 表 GPU 化 + CUDA graph(~205µs),计入 run();对拍 host golden 全等(NE×{balanced,skewed});e2e NE=256 计入后 5832µs 仍 < serial(1.21×),星号已去** |
| T4 | gate+up 合并一次 GEMM(up 的 1.5ms 藏进 dispatch) | **✅ 完成(docs/13 §6):NE=256 −204µs、NE=64 −478µs,默认开;T5 后 up 才完全隐藏** |
| T5 | ROW_BLOCK=64(NE=256 两层 GEMM 各省一半行) | **✅ 完成(docs/16):编译期 TK_ROW_BLOCK 开关,全档正确;但预期被推翻——padding 本以满效率算,减 padding 后 npl=4096 落小 tile 低效区(143→75 TFLOP/s),W2 wash。RB=64 仅 NE=256 净赢 ~260µs,NE≤128 变慢。默认保持 128。大收益需 8warp×8行/fp8 计算** |
| T6 | combine 预归约 + push 化(A' 镜像,layer1 通信 4×) | **✅ v0+v1 完成:v0 预归约(docs/13)+ v1 push 化(docs/18,边算边推消零重叠)。默认 prered_push;layer1 融合损失 283→192µs,e2e(默认shape)2331→1963µs(1.48× serial)** |
| T7 | dispatch (token,dst) 去重 + fp8 传输 | **去重✅(docs/15):v0 落地默认关(TK_DEDUP=1);NE≤128 大赢(e2e NE=64 −0.47ms),NE=256 无收益因 dispatch 实为 GEMM-bound(账本修正);与 T5 协同后 NE=256 生效。fp8 未做** |
| T8 | push3 目的块重排 + 水位信号(NE=256 翻盘后设默认) | 未开始 |
| T9 | 杂项:AGENTS.md 口径、skewed 测试、comm_sms 扫参、probe 增强 | 未开始 |

核心评审结论(细节在 docs/11 §0,**T1 实测修正见 docs/12**):
1. **layer1 是最大的矿,但慢的不是 GEMM 而是 combine gather**:T1 实测推翻 §0.1(a)——
   W2 GEMM 带 fence epilogue 仍跑 ~140 TFLOP/s,"71 TFLOP/s" 是伪指标;真正的 94~99%
   在 combine 每 token 拉 8 条散布 ~14KB peer 行(58.7MB @ ~32GB/s,PCIe 弱路径顶)。
   → T2 冻结,**T6(combine 预归约+push)是 layer1 唯一主攻**;
2. dispatch pull 已贴 20GB/s 带宽顶,剩余杠杆全在减字节(去重 2.2×、fp8 2×)和换强路径;
3. push3 在 NE=256 退化的补充归因:push 按源 token 序,目的块就绪全部后置 → 重排解决;
   信号数 ∝ 行块数 → 水位聚合解决;
4. 总账目标:全部落地后 e2e ~2.4ms vs serial 7ms。

## 4. 踩坑索引(改代码前必读)

- **远端原子会丢增量**(高并发散射 red.add,probe 弱压力测不出)→ docs/08。
  跨卡完成检测只用"本地 atom.acq_rel 选举 + 单写者 st.release.sys"两阶段协议。
- **group::store 行映射置换**(warpgroup 交织,改 CONSUMER_WARPS 数必重推)→ docs/05。
- **新方案必须全档位扫 NE∈{64,128,256}**,单点结论会误导 → docs/10 §7。
- **协议正确性靠隔离裁决**(xx-only debug 入口 + 30 迭代 + 双重对账),
  不靠"跑跑看对不对" → docs/08、docs/10。
- 内存序 cumulativity 链仅本机验证过,换卡先跑 validate_push3 类裁决。

## 5. 环境与运行速查

```bash
source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
cd /data/cinnzhang_vllm_td_test/xxy          # 必须在上级目录跑 -m moe_bench.*
nvidia-smi                                    # 跑前确认卡空闲
# 对拍(全链路 vs reference_moe)
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.run_tkfused 64
# benchmark(bf16 EP;--distributed 必带,--scheme 单值,分别跑 tkfused / serial)
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.bench --distributed --scheme tkfused \
    --mode ep --precision bf16 --world-size 4 --no-verify --num-tokens 512
# dispatch-only 隔离计时
CUDA_VISIBLE_DEVICES=9,11,13,15 TK_DISPATCH=push3 python -m moe_bench.tools.time_dispatch 256 10 50
```

**默认路径 = pull dispatch + prered combine + fused gate+up + GPU schedule(计入 run)。**
环境开关速查:
- `TK_COMBINE` = `prered_push`(默认,T6-v1)| `prered`(v0)| `pull`(旧 combine)
- `TK_FUSE_GATEUP` = `1`(默认,T4)| `0`
- `TK_GPU_SCHED` = `1`(默认,T3,CUDA graph)| `0`(schedule 回 setup 不计时)
- `TK_DEDUP` = `0`(默认)| `1`(T7,NE≤128 赢、NE=256 无收益)
- `TK_ROW_BLOCK` = `128`(默认)| `64`(T5,仅 NE=256 净赢 ~260µs,NE≤128 变慢)
- `TK_DISPATCH` = `pull`(默认)| `push3`/`push2`/`push`(实验/冻结)

## 6. 文档索引

| 文档 | 内容 |
|---|---|
| docs/01~06 | Phase 0~4:probe、grouped GEMM、dispatch⊕GEMM、combine、scheme 接入 |
| docs/05 | group::store 行映射 bug(最深的坑) |
| docs/07 | 第一轮评审:公平性 P1~P5 + 优化方向 #1~#6 |
| docs/08 | push 冻结:远端原子丢增量归因 + push2 |
| docs/09/10 | push3 规划 / 实现验证实测(NE 交叉点) |
| **docs/11** | **第二轮评审 + 当前任务清单(T1~T9,做事看这篇)** |
| **docs/12** | **T1 归因实测:layer1 慢在 combine gather(94~99%),非 fusion;T2 冻结→T6** |
| **docs/13** | **T6-v0:combine 预归约 host 表设计(按 expert 卡重分组,等价)+ 双向对账工具 + kernel 落地 + T4 gate+up 合并** |
| **docs/14** | **T3:schedule GPU 化(单 argsort ring-order + 稠密 job 空间)+ CUDA graph 捕获,计入 run() 公平口径** |
| **docs/15** | **T7:dispatch 去重(稠密 staging,gathered 逐字节等价);揭示 NE=256 dispatch 是 GEMM-bound(账本修正)** |
| **docs/16** | **T5:ROW_BLOCK=64 编译期开关;揭示 padding 非纯浪费(满效率算),小 tile 效率折损抵消,默认保持 128** |
| **docs/17** | **通算重叠与融合损失分析:layer0 融合 +30%、layer1 +56% 且通信零重叠(T6-v1 目标)** |
| **docs/18** | **T6-v1:combine 预归约 push 化(边算边推 + 水位选举,消 barrier+零重叠);layer1 融合损失 283→192µs,默认** |
| **docs/19** | **TP 版 tile overlap(tktp,分支 tp_test):AG⊕gate+up GEMM + 本地prered⊕稠密RS push;复用矩阵/调度表/账/上机风险清单/一键脚本** |
| experience/ | 12 篇相关工作与平台经验(01 总览、12 SM120/PCIe 适配最常用) |
