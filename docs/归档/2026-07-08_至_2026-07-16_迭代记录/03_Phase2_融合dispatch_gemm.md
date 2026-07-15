# Phase 2 融合 dispatch+GEMM kernel：正确性与性能

**日期**：2026-07-08
**kernel**：`tileoverlap/02_moe_dispatch_gemm/moe_dispatch_gemm_sm120.cu`
**卡组**：2 卡用 `8,9`（PIX 同 switch）；4 卡用 `9,11,13,15`

## 结论：2/4 卡正确性全通，性能 comm-bound（符合 PCIe 预期）

### 正确性（DeepSeek-V3 配置：E=256 top8, H=7168, I=2048）

| | Gathered tokens（dispatch pull） | Outputs（融合 GEMM） |
|---|---|---|
| 2 卡 | **max diff 0.0（字节精确）** | max diff 0.098 < 0.1 |
| 4 卡 | **max diff 0.0（字节精确）** | max diff 0.089~0.095 < 0.1 |

- **跨卡 TMA pull 字节精确**：dispatch 从 peer 卡拉 token 向量、落本地、本地 red 计数的
  整条 pull 路径完全正确，验证了「pull + 本地原子」在 PCIe 上的正确形态。
- 融合 GEMM 输出误差与 Phase 1 单卡同量级（bf16 舍入），说明 producer 自旋等灯 + grouped
  GEMM 融合无逻辑错误。

### 性能：dispatch 主导，时间 ∝ 1/comm_sms

4 卡 seq=4096 各 rank 平均：

| comm_sms | 时延（rank0） |
|---|---|
| 1 | 40.4 ms |
| 2 | 24.4 ms |
| 4 | 14.6 ms |
| 8 | 5.4 ms |
| 16 | 3.2 ms |

时延几乎与 comm_sms 成反比 → **完全 dispatch-bound**，GEMM 计算被完全隐藏（符合 experience/12
篇「PCIe 上 overlap 藏的是计算，E2E 下限 = T_comm」）。这也印证 Phase 0 probe 的发现：
**SM 驱动 pull 带宽弱**，逐 token TMA pull 是瓶颈，comm SM 越多并发拉取越快。

**含义**：
1. `num_comm_sms` 在本机应往大给（16 甚至更多），与 experience/12 篇「PCIe 上 2~8 起步」的
   预估不同——因为本机 SM-pull 带宽比预期更弱，需要更多 block 才能把带宽时延积填满。
2. 绝对时延偏高（4096 seq、16 comm_sms 仍要 3 ms）。头号优化杠杆是 experience/12 §4.1 的
   **FP8 dispatch**（传输量减半）和 §4.3 的**源端 pack + 大块 push**（push 带宽 51GB/s 远好于
   pull）。这些是后续迭代项，v1 先保证正确性打通。
3. 逐 token pull 的延迟受限（probe [D] RTT 2.17us）需要**深流水**：当前 dispatch 每 block
   6 个 token（TOKENS_PER_BLOCK）、每 token 一个 outstanding TMA，并发度可能不够，是后续
   调优点（PLAN 风险 R3）。

## 运维坑：改 NUM_GPUS 必须先 make clean

`make NUM_GPUS=4` 若 `.so` 比源文件新则**不重编**，仍用上次 `TK_NUM_DEVICES` 的值。
症状：`RuntimeError: weights first dim must be local expert count`（C++ 侧
`num_local_experts=256/旧world` 与 python 传入的权重专家数不符）。
**规矩**：每次切换 NUM_GPUS 前先 `make clean`。

## 与 09 篇三件套的对齐（待补全）

- `T_comp_alone`：Phase 1 的 216~234 TFLOP/s。
- `T_comm_alone`：可用 NCCL all-gather 折算，或把 GEMM 短路单测 dispatch。
- overlap 效率：本机 comm-bound，T_total 逼近 T_comm 即算成功。当前 T_total 基本等于
  dispatch 时间，说明 GEMM 已被完全隐藏，overlap 在「藏计算」意义上已达成。
