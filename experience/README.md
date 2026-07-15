# GPU 通信-计算重叠算子开发经验库

**整理日期**：2026-07-08
**来源**：Flux、Comet、FlashOverlap、DeepEP、DeepSeek-V3/DualPipe、TileLink/Triton-distributed、ThunderKittens、FlashDMoE、TokenWeave、ISO、Syncopate(AutoOverlap)、CommFuse、MSCCL++ 等论文与开源仓库的深度研读（论文全文 + 源码逐文件阅读）。

**性质**：本目录只沉淀**思想、经验、因果与数字**，不含可复制粘贴的代码；出现的伪代码均为思想级示意。目标读者：算子开发工程师、以及用于指导 AI agent 做算子开发的知识库。

## 目录

| 文档 | 内容 | 什么时候读 |
|---|---|---|
| [01_相关工作总览.md](01_相关工作总览.md) | 13 个相关工作的定位、核心机制、关键数字、适用边界、一页对比表 | 进入领域第一篇 |
| [02_依赖分解与数据流.md](02_依赖分解与数据流.md) | shared tensor 依赖分析、沿哪个维度分解、token 重排、readiness 粒度选择 | 设计任何 overlap 算子前 |
| [03_SM间调度与资源分配.md](03_SM间调度与资源分配.md) | 通信 SM 预算、block specialization、tile swizzle、尾延迟消除、wave 对齐 | 决定资源划分与调度顺序时 |
| [04_SM内组织.md](04_SM内组织.md) | warp specialization、pipeline 模板、寄存器/smem 预算、persistent kernel | 写 kernel 内部结构时 |
| [05_同步与channel.md](05_同步与channel.md) | signal 设计、memory ordering、计数语义、channel/QP 组织、复位策略 | 设计同步协议时（最容易出错的部分） |
| [06_tile粒度与wave.md](06_tile粒度与wave.md) | 计算/通信/信号三种粒度解耦、wave 检测与分组、粒度选择数字 | 调粒度参数时 |
| [07_通信底座与拓扑.md](07_通信底座与拓扑.md) | NCCL/NVSHMEM/MSCCL++/copy engine/multimem 选型、push-pull、IB+NVLink 两跳转发 | 选通信机制与做拓扑适配时 |
| [08_算子路线图.md](08_算子路线图.md) | AG+GEMM、GEMM+RS、MoE dispatch/combine、AllReduce+Norm 各算子的推荐路线与决策树 | 针对具体算子动手前 |
| [09_调参与benchmark方法论.md](09_调参与benchmark方法论.md) | 必测 baseline、干扰系数、autotune 体系（含分布式 autotune 的特殊点）、sweep 空间 | 建性能验证体系时 |
| [10_坑与反模式清单.md](10_坑与反模式清单.md) | 死锁面、内存序、false sharing、one-wave、隐式同步等 40+ 条具体的坑 | 全程对照，code review 清单 |
| [11_ThunderKittens多GPU与MoE融合实战.md](11_ThunderKittens多GPU与MoE融合实战.md) | TK 多 GPU 基础设施（pgl/multimem/TKParallelTensor）、仓库 4 个融合 kernel 逐个拆解、MoE TP/EP tile 粒度 overlap 的 TK 实施方案与 TK 特有坑 | 用 TK 动手实现时（本篇例外地含真实 API 名与文件路径） |
| [12_SM120与PCIe拓扑适配.md](12_SM120与PCIe拓扑适配.md) | 实际硬件（RTX PRO 5000 / SM120 + PCIe）上 11 篇的勘误：multimem/跨卡原子/wgmma 失效清单与替代、st 版信号协议、PCIe 通信账重算、SM120 GEMM 模板改造 | 与 11 篇对照读，动手前先跑 §1/§6 的平台微基准 |
| [13_ThunderKittens融合Kernel性能分析.md](13_ThunderKittens融合Kernel性能分析.md) | TK 的 CUDA Event、torch trace、nsys/NCU 与设备端 TKProfiler 分层用法；融合 kernel 内 comm/comp 拆账、multi-rank replay 风险及 SM120 移植边界 | 融合 kernel 已能运行、需要定位等待/通信/计算瓶颈时 |

## 十条最重要的经验（速览）

1. **依赖粒度决定重叠上限**：把"等整个 tensor"降到"等一个 tile/chunk/rank 分片"，是所有工作的共同起点；粒度选错方向（如 MoE layer0 沿 N 切）会导致完全无法提前放行。
2. **通信不是免费并发**：NCCL 式 16+ SM 的通信 kernel 会让并发 GEMM 慢 15~20%；打满带宽实际只需 2~20 个 SM（multimem 2~8 个、跨节点 all-to-all 20 个），甚至 0 个（copy engine / NIC 后台）。
3. **重叠时序的控制手段首选 tile 遍历顺序（swizzle），而不是新调度器**：rank 偏移环形推进、本地数据先算、无需通信的块最后算，一个下标公式解决大半问题。
4. **计算 tile、通信 chunk、信号粒度是三个独立的量**，解耦后各自向最优调；信号通常应粗于 tile（wave/chunk 级），per-tile 信号只在必要时用。
5. **wave 是判断有没有重叠空间的第一指标**：tile 数不足一个 wave 时基本无空间；切分导致总 wave 数增加时细粒度反而变慢（切分要 wave 对齐）。
6. **同步协议要"少、准、可验证"**：计数语义（等第 k 个事件）优于布尔 flag；数据依赖用 release/acquire，纯调度用 relaxed；复位策略（消费即复位 / 递增 target）设计进协议而不是靠额外 memset。
7. **确定性要验证而不是假设**：FlashOverlap 用"10 次实测完成序全部一致"作为方案准入条件——对 GPU 调度行为的任何假设都应有类似的验证机制。
8. **拓扑决定策略**：NVSwitch/PCIe/IB 的最优 push-pull、swizzle、channel 数完全不同；跨节点要用"IB 只发同 GPU-index 的 rank + 节点内 NVLink 扇出"的两跳转发消除带宽不对称。
9. **分层实施**：stream+signal 低侵入方案 → prologue/epilogue 融合 → block/warp specialization → persistent kernel + device 调度器。每一级都是上一级的对照组，用测量决定是否继续下沉。
10. **收益天花板 = min(T_compute, T_comm) 且受通信占比约束**：NVLink 机器 TP 通信占比一般 15~25%，PCIe/跨节点 35~48%；预期收益（端到端 1.1~1.7×）要先算账再立项。
