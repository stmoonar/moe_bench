# HANDOFF — TK 通算融合 MoE 进度交接

> 新 session 从这里接手。读完本文 + 最新一篇 docs/ 即可继续。

**最后更新**:2026-07-09(push3 落地 + 全量评审 + 路线图 + T1 归因实测 docs/12 +
**T6-v0 host 预归约表 + 对账落地,见 docs/13**)

## 1. 项目一句话

在 moe_bench 里用 ThunderKittens 实现通算融合的 EP MoE 层(dispatch⊕GEMM、GEMM⊕combine
单 kernel 融合,不做 stream overlap 降级方案),对比 vLLM serial baseline。
平台:16×RTX Pro 5000(sm120)PCIe,用 4 卡组(优先 9,11,13,15),无 NVLink、
**远端原子高并发不可靠**(docs/08)、无 multimem。

## 2. 当前状态(全部已提交,分支 tk_dev)

- **正确性**:tkfused 全链路对拍 reference_moe 通过(bf16, EP, 4 卡, rel_err ~4.4e-3)。
- **性能**(NE=256, 512 token/rank, bf16):tkfused **5832µs vs serial 7063µs**(快 1.21×,
  **schedule 已 GPU 化并计入 run(),公平口径,星号已去**;T3 前不计时口径为 5603µs;
  NE=64 3462µs)。
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
  `time_schedule.py`(T3 schedule 重算成本:all_gather/eager/graph 分解)。
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
| T5 | ROW_BLOCK=64(NE=256 两层 GEMM 各省一半行) | 未开始(**下一主攻**:T7 揭示 NE=256 dispatch 也 GEMM-bound,减 padding 直接砍两层 GEMM 且让 T7 去重在 NE=256 生效,docs/15) |
| T6 | combine 预归约 + push 化(A' 镜像,layer1 通信 4×) | **⬆️ 主攻,v0 ✅ 落地达标:NE=256 e2e 7131→5775µs(首超 serial 7063),NE∈{64,128,256} 对拍全过 rel~7e-3,prered 设默认(docs/13 §5);下一步 v1 push 化** |
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
# 对拍
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.run_tkfused 64
# benchmark(bf16 EP)
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.bench --scheme tkfused,serial \
    --mode ep --precision bf16 --world-size 4 --no-verify
# dispatch-only 隔离计时
CUDA_VISIBLE_DEVICES=9,11,13,15 TK_DISPATCH=push3 python -m moe_bench.tools.time_dispatch 256 10 50
```

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
| experience/ | 12 篇相关工作与平台经验(01 总览、12 SM120/PCIe 适配最常用) |
