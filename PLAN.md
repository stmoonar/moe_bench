# 通算融合 MoE 实现计划

## 1. 项目目标

在 RTX Pro 5000（sm120 架构）+ PCIe 互连 + 4 卡环境下，**从零实现**一套 COMET 风格的 MoE 层通信计算融合算子，并接入 moe_bench 框架，与串行 baseline（AllGather → fused_experts → ReduceScatter）对比正确性与性能。

核心思想：将 MoE 层的跨卡通信（dispatch/combine）与专家计算（Grouped GEMM）在 **tile 粒度**上重叠，用 kernel 内的信号机制（"红绿灯" flag）表达依赖，用 SM 级线程块分工隔离通信对计算的干扰，从而隐藏通信延迟。

## 2. 硬件环境

| 维度 | 规格 |
|---|---|
| GPU | RTX Pro 5000，sm120 架构（消费级 Blackwell） |
| 互连 | PCIe（卡间无 NVLink），连续两卡共享同一 PCIe Switch |
| 卡数 | 使用 4 卡测试 |
| 推荐卡组合 | 9,11,13,15 > 8,10,12,14 > 1,3,5,7 > 0,2,4,6（同 Switch 优先） |
| CUDA | 需 12.8+（编译 `sm_120`） |

**关键约束**：PCIe 带宽远低于 NVLink（~25-60 GB/s vs 数百 GB/s），且跨卡 RMW 原子大概率不可用。所有通信设计必须适配这一前提。

## 3. 技术规格

| 参数 | 值 |
|---|---|
| 精度 | FP8（w8a8，per-block 量化，block_shape=[128,128]） |
| token 数 | 512 per rank（全局 2048） |
| 模型 | DeepSeek V3.2 规格（hidden=7168, inter=2048, experts=256, topk=8） |
| 并行模式 | EP（专家并行），world_size=4 |
| baseline | SerialNaive（AllGather + fused_experts + ReduceScatter 串行） |

## 4. 实现约束

- **不提供 COMET/flux 源码**。Agent 依据 `references/` 下的原理文档自行设计和实现。
- 必须接入 moe_bench 的 `DistributedScheme` 接口（见 [ADDING_IMPLEMENTATIONS.md](ADDING_IMPLEMENTATIONS.md)）。
- 必须通过 moe_bench 的正确性校验（`verify_output`，与纯 torch 参考比对）。
- 计时方式为 eager 模式（分布式下禁用 CUDA Graph），需保证 `run()` 无副作用、分配稳定。
- 每次测试前检查 GPU 是否空闲。

## 5. 实现里程碑

### M0：环境探测与基础验证
- 探测 PCIe P2P 能力（`cudaDeviceCanAccessPeer`、`cudaIpcOpenMemHandle` 跨进程映射）
- 探测跨卡原子支持（`cudaDevP2PAttrNativeAtomicSupported`）
- 实测 P2P DMA / SM 远端写 / SM 远端读带宽
- 在 sm120 上跑通最小 CUTLASS grouped GEMM（bf16，不带融合）
- 确认 NVSHMEM 在纯 PCIe 上是否可用（备选 cudaIPC）
- **产出**：确定通信方案档位（A/B/C）与 GEMM 路线

### M1：单卡组件
- Grouped GEMM + gather_A/scatter_D（FP8）
- 路由元数据 kernel（逆映射、排序、cumsum 表、device 侧 workspace 生成）
- Reducer 单卡版（gather + topk 加权求和，无 ring）
- 每步与 torch 参考对拍
- **产出**：所有计算组件单独正确

### M2：通信层
- 对称缓冲分配（cudaIPC 或 NVSHMEM）+ stream 上的 group barrier
- Ring1D AllGather（push 模式 + 点灯）
- Ring ReduceScatter（push 模式）
- 独立验证通信正确性与带宽
- **产出**：跨卡数据搬运正确，带宽达标

### M3：融合集成
- Layer0：per-tile 到达等待接入 GEMM prologue
- Layer1：N-split 分层计数 + 双流水平融合（GEMM + Reducer）
- 4 卡全链路对拍
- **产出**：通算融合算子端到端正确

### M4：调优与 benchmark
- 扫参建表（tile 形状、n_split、RS_BLOCKS、sm_margin）
- 接入 moe_bench，与 serial baseline 对比
- 输出性能报告
- **产出**：性能数据，相对 serial baseline 的耗时降低达到 PCIe 参照档
  （16.7%–33.3%）

## 6. 成功标准

1. **正确性**：moe_bench `verify` 列全部 `ok`（FP8 容差 atol=3.5e-2, rtol=3.5e-2）
2. **性能**：单层延迟低于 serial baseline（PCIe 参照档耗时降低 16.7%–33.3%）
3. **鲁棒性**：边界场景通过（某专家 0 行、极端倾斜路由、token 数不能整除等）

## 7. 参考文档索引

| 文档 | 内容 |
|---|---|
| [00_项目总览.md](references/00_项目总览.md) | 项目定位、硬件约束、技术规格、成功标准 |
| [01_背景知识.md](references/01_背景知识.md) | MoE 层结构、分布式执行、GPU 执行模型、跨卡通信原语、红绿灯模式 |
| [02_通算融合原理.md](references/02_通算融合原理.md) | shared tensor 分析、垂直/水平融合、分解维度、layer0/layer1 机制、自适应负载 |
| [03_Grouped_GEMM与实现要点.md](references/03_Grouped_GEMM与实现要点.md) | CUTLASS grouped GEMM、tile、gather/scatter、侵入点、实现顺序 |
| [04_sm120_PCIe适配.md](references/04_sm120_PCIe适配.md) | 硬件差异、通信方案选择、协议修正、GEMM 路线、常见坑 |
| [05_benchmark接入契约.md](references/05_benchmark接入契约.md) | moe_bench 架构、DistributedScheme 接口、数据结构、验证方法、运行方式 |

## 8. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 消费卡 P2P 被驱动禁用 | 通信方案退化为 host staging | M0 先探测，准备 B/C 档备选 |
| CUTLASS grouped GEMM × sm120 组合不成熟 | 计算层无法构建 | 尝试 CuTe collective 与 sm89 兼容路径两条路线 |
| PCIe 带宽过低 | 重叠收益缩水 | 加大 n_c、加粗通信粒度，参考 L20 档预期 |
| 跨卡原子不可用 | ring 握手不能用 RMW | 改用 release-store + acquire-load ring-token |
| FP8 精度问题 | 正确性不通过 | 确保累加用 fp32，参考实现携带相同量化误差 |
