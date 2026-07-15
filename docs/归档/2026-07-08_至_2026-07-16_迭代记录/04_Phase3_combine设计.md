# Phase 3 设计：layer1 GEMM(W2) ⊕ combine（SM120 + PCIe，路线 B'）

**日期**：2026-07-08

## 1. combine 是什么

EP MoE 的 layer1 = `y_expert = h @ W2`（每 expert），随后 **combine**：把每个 expert 的输出行
按 token 送回**源卡**，并对该 token 的 top-k 个 expert 输出做**加权求和**（权重=topk_weight）。

- W2：`(I=2048) → (H=7168)`，输入 `h (padded_local_tokens, I)`，输出 `expert_outputs
  (padded_local_tokens, H)`。**这就是 `grouped_gemm_sm120`（01/common）换个维度**，无需新 GEMM。
- combine 是 dispatch 的**逆向 scatter-reduce**：dispatch 把源卡 token 拉到 expert 卡；combine
  把 expert 卡的输出行送回源卡并 top-k 加权累加。

## 2. 路线 B'（源卡 pull + FP32 本地归约），为什么选它

experience/12 §6 的两条路线：
- **A'（推 + 目标卡归约）**：expert 卡把行推到源卡的 per-source 独立 buffer，源卡本地加。
- **B'（源卡 pull + 本地归约）**：源卡 comm SM 从 top-k 个 expert 卡**拉**自己 token 的输出行，
  FP32 本地加权累加。**全 pull、无远端原子、信号单向，和 layer0 dispatch 同构** → 选 B'。

（注：Phase 0 probe 发现本机远端原子实际可用，A' 也可行；但 B' 与 layer0 协议一致、复用
`pcie_sync`，工程上最省，且 pull 归约天然易上 FP32，故先做 B'。）

## 3. 索引：combine_indices = dispatch pull 索引的逆映射

- dispatch（在 expert 卡 e_rank 视角）：`pull_dispatch_indices[slot] = (src_dev, src_token)`。
- combine（在源卡 r 视角）：对我的 token t 路由到的第 k 个 expert E（在 e_rank 的某 slot），
  `combine_indices[t*topk + k] = (e_rank, remote_slot)`，`combine_weights[t,k] = topk_weight[t][k]`。

benchmark 里 routing 全卡已知（broadcast），所以**每张卡都能重放所有 expert 卡的 schedule**，
从中筛出 `src_dev == 本卡` 的条目，反查出自己的 combine 索引。生产路径再考虑上 GPU 向量化
（`argsort`+`cumsum` 的逆），v1 先 Python 顶上。

## 4. 分两步实现（本 doc 对应的 Phase 3a/3b）

### Phase 3a：combine kernel 单元验证（非融合，本次）

先把 combine 原语单独打通、对拍 torch，**不依赖真实 W1/W2**：
- 直接构造随机 `expert_outputs`（TKParallelTensor，IPC 共享，供 peer 读）。
- combine kernel：每 block 管一个源 token，256 线程各管 H/256 列；对 top-k 个 expert：
  从 `expert_outputs[e_rank][remote_slot, col]` **P2P 直读** bf16，`acc_fp32 += w * v`；
  top-k 完存 `output[t, col]`（fp32→bf16）。
- 对拍 torch：`out[t] = Σ_k w[t,k] * expert_outputs_full[e_rank(t,k)][slot(t,k)]`。

非融合锚点用 **P2P 直读**（probe 确认远端 load 可用），不走 TMA、不需要 per-col-block 信号——
因为 combine 在所有 expert 卡 W2 GEMM 完成后（一次 `device_barrier`）才跑，数据全就绪。
这是正确性锚点，最简。

### Phase 3b：融合 W2 GEMM ⊕ combine（后续）

把 combine 融进 W2 grouped GEMM：expert 卡 consumer 存完一个列块 → 信号 warp 本地 red 计数 →
`pcie_sync::signal_slot` 广播「该 (slot 段, 列块) 就绪」给各源卡；源卡 comm SM
`pcie_sync::wait_slot` 等就绪后按列块 pull + FP32 累加。协议与 layer0 同构，复用
`pcie_sync`。对拍 Phase 3a 锚点。

## 5. 精度与后续

- v1 全程 bf16（用户选定「先 bf16 打通」）；combine 累加用 **FP32**（top-k=8，收窄误差）。
- fp8：等 bf16 全链路 + scheme 接入打通后再上（dispatch 传 fp8e4m3、SM120 原生 fp8 mma）。

## 6. 验收锚点

- Phase 3a：combine kernel vs torch combine，max diff < 1e-2（bf16 输出，fp32 累加）。
- Phase 3b：融合版 vs 3a 锚点，max diff ~0（同 kernel 逻辑，仅加信号）。
- Phase 4：完整两层 scheme vs `reference_moe`，走 moe_bench 的 fp8/bf16 容差。

## 7. Phase 3a 结果（已完成）

**kernel**：`03_moe_gemm_combine/moe_gemm_combine_sm120.cu`
**验证**：2 卡（8,9）与 4 卡（9,11,13,15），DeepSeek-V3 配置 E=256 top8 H=7168。

| | max diff | mean diff |
|---|---|---|
| 2 卡 | 0.0039 | ~1e-8 |
| 4 卡 | ≤0.0078 | ~1e-8 |

- **combine 原语正确**：源卡 P2P pull top-k expert 输出行 + FP32 加权累加，对拍 torch
  几乎逐字节一致（mean ~1e-8，max 是单元素 bf16 舍入）。
- **schedule 逆映射正确**：`build_combine_schedule` 重放每张 expert 卡的 dispatch 写位
  （复用 02 的 ring/顺序），筛出 `src_dev==本卡` 的条目反查 (e_rank, slot)——与 02 的
  dispatch 完全对偶。
- 首次编译零错误、零 spill。
- 性能：512 源 token（真实规模）1.2~1.8 ms，等效 pull 带宽 ~32~49 GB/s（接近 CE 峰值
  56GB/s；rank 间不均是拓扑/pull 不对称，与 probe [C2] 一致）。非融合锚点用 P2P 直读、
  逐 token 一个 block，已相当接近带宽上限——说明 combine 也是 comm-bound。

**这是非融合锚点**：combine 在所有 expert 卡 W2 完成后（一次 device_barrier）才跑，数据全就绪，
故直接 P2P 读、无需 per-col-block 信号。Phase 3b 会把它融进 W2 GEMM、加 slot 就绪信号，
对拍这个锚点。

## 8. Phase 3b 结果（已完成）：融合 W2 GEMM ⊕ combine

**kernel**：`03_moe_gemm_combine/moe_gemm_combine_fused_sm120.cu`（模块 `_Cf`）
**验证**：2 卡（8,9）与 4 卡（9,11,13,15），E=256 top8 H=7168 I=2048。

| | max diff | mean diff |
|---|---|---|
| 2 卡 | 0.00012 | ~3e-10 |
| 4 卡 | ≤0.00012 | ~3e-10 |

融合形态：
- expert 卡 comp SM 跑 W2 grouped GEMM（复用 `grouped_gemm_sm120` + 新增 output epilogue）；
  每写完一个 (row_block, col_block) tile，epilogue 本地 `atom.acq_rel.gpu` 计数该 row_block
  完成的列块数；满 col_blocks 时，**每个 consumer 线程 `__threadfence_system()`** 刷自己的
  输出条带，`group::sync` 汇合，单线程 `pcie_sync::signal_slot` 向所有源卡广播
  「(本 expert 卡, row_block) 就绪」（单调 seq，PCIe-safe）。
- 源卡 combine block（grid 尾部，每 block 一个源 token）先 `pcie_sync::wait_slot` 等
  (e_rank, slot/128) 就绪，再 P2P pull 行 + FP32 加权累加（与 3a 相同）。

两个正确性关键点（都在踩坑后修复，见 docs/05）：
1. **group::store 行映射**：consumer 加载 A 要用 store 的 warpgroup 交织 `local_warpid`。
2. **system fence**：跨卡读的输出必须每线程 `__threadfence_system()` 后再发就绪信号。

死锁规避：comp（GEMM）block 是最低 blockIdx 且 persistent，常驻产生信号；combine block
（在 wait_slot 自旋）排在剩余 SM 上——与 02 的 dispatch 同构，`num_comm_sms >= 1`。

性能（4 卡，num_src=512 真实规模）：~3.5 ms、W2 GEMM ~71 TFLOP/s。融合已把 combine 的
跨卡 pull 藏在 GEMM 之后（comm-bound）。绝对时延后续靠 FP8 + 源端 pack 降。
