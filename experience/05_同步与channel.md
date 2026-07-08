# 05 同步与 channel

> 同步协议是 overlap 算子最容易写错的部分：错了不是慢，是 hang 或 silent corruption。本篇是协议设计的经验汇总。

## 1. 信号的语义分级：数据依赖与执行依赖分开

| 用途 | 语义 | 说明 |
|---|---|---|
| 数据依赖（"这个 tile 的数据可以读了"） | 写侧 release（fence + 归约加/写值），读侧 acquire 自旋 | 保证信号可见时其之前的数据写也可见 |
| 执行依赖（"轮到你了"，不携带数据保证） | relaxed 原子 | 快，但**误用在数据依赖上=难查的偶发竞态** |
| 卡内 producer-consumer | 硬件 barrier（mbarrier），可睡眠、硬件仲裁 | 不要用 global flag 自旋做卡内流水 |
| 跨卡 | release 归约写 + relaxed/acquire 自旋读，系统 scope | scope 用错 = 偶发不可见 |

- **计数语义优于布尔 flag**：signal = 计数 +1，wait = 等"到达数 ≥ 期望值"。天然支持"等第 k 个 chunk"、多次 signal 排队、多消费者各等各的次序（MSCCL++ semaphore 的设计，值得作为默认选择）。
- **数据即信号（LL packet）**：小消息把数据与 flag 打包成 8/16B 原子可见单元，一次 store 完成传输+通知，免独立同步往返；代价是有效带宽减半，阈值大约在 KB 级消息以下适用。
- 编译器/软件流水环境下，wait 与后续 load 之间要有**显式依赖**（wait 返回 token，load 消费 token），防止调度器把异步 load 提前越过 wait——这是所有"编译出来的"细粒度同步的正确性关键。

## 2. 信号复位：设计进协议，不要事后 memset

三种已验证的复位策略，按优先级：

1. **消费即复位**：等待用 CAS（期望值命中即原子换成 0），复位摊进正常路径，无需额外 kernel。要求严格一写一读。
2. **递增目标值**：跨迭代不清零，每次调用把期望值 +1，信号缓冲只增不清。与 CUDA graph 或异常重入混用时要小心一致性。
3. **显式复位 + 双 barrier 夹住**：必须 memset 时（如多写者计数器），复位前后各一道全局 barrier，防止"A 已开始写信号时 B 还在清零"的竞态；复位可放独立 stream 与计算并行。

补充：flag 复用前必须归零且**各卡对齐退出**——用独立的小收尾 kernel + 全卡 barrier，不要在主 kernel 里顺手做（多付一次全网同步）。

## 3. 信号缓冲的物理布局

- **每个 flag 独占一条 cache line（128B 对齐 + padding）**：自旋读与原子写混在同一 cache line 上的 false sharing 代价巨大；shape 小于 tile 时相邻 tile 的 flag 尤其容易挤进同一行（Flux 代码里专门注释"不要删 padding"）。
- 信号缓冲与数据缓冲分离，按 channel 分片，避免所有 channel 竞争同一个原子计数器。
- 信号本身放对称内存（各 rank 同偏移），让 host 写值原语、copy engine、SM、NIC 四种执行体说同一套协议。

## 4. 谁来发信号、谁来等（执行体矩阵）

| 生产侧 | 机制 | 典型场景 |
|---|---|---|
| host/stream | stream 级写值原语（拷贝完成后置位） | copy engine 路径的 AG |
| GEMM epilogue | tile store 落地后 thread 0 一次原子加 | FlashOverlap/Flux/TileLink |
| 通信原语自带 | "数据+信号"一体的 put（一条原语完成 push+notify） | NVSHMEM/MSCCL++ push 路径 |
| NIC/proxy | RDMA 完成后由代理递增远端计数 | 跨节点 0-SM 路径 |

| 消费侧 | 机制 | 典型场景 |
|---|---|---|
| kernel prologue | producer warp acquire 自旋 | AG+GEMM 的 tile 等待 |
| 独立 wait kernel | 1 线程 CAS 自旋，卡住同 stream 后续 kernel | 触发 NCCL 等黑盒通信 |
| stream 等值原语 | host 侧等值 | copy engine 链路的串接 |

## 5. channel / QP 的组织经验

- channel = 通信并行度的单位，同时决定 SM 占用与链路利用率。经验组织法（DeepEP）：**2 个 SM 构成 1 个 channel**（一收一发），10 个 channel 打满 IB+NVLink；token/数据按 channel 静态切段。
- **数据通道与控制通道分离**：每 channel 两条 RC QP，一条走数据 put，一条走 head/ack 回写——控制消息不被大数据流阻塞。
- 低延迟场景 QP 按目标实体分（per-expert 一条 QP）才能并行打满 NIC。
- channel 数的调法：先与链路数/目标 rank 数对齐，再按 SM 预算回退；小消息少 channel 求延迟，大消息多 channel 求带宽，但多 channel 会抬高 SM/L2 占用与 metadata 开销。
- 热点打散：目标遍历顺序、warp→目标映射统一加"(自身 id + channel id) mod N"式轮转，避免所有 channel 同一时刻怼同一个目标（incast）。

## 6. 环形缓冲与流控（跨设备队列协议)

生产级跨设备数据队列（DeepEP normal 路径）的协议要点，可作为设计模板：

- head/tail 双指针：生产者推 tail（数据落地后），消费者推 head（用完后）；双方都缓存对端指针减少远端读。
- **发送窗口 ≤ 接收缓冲的一半**：保证懒惰 head 回写模式下发送方永远有空间，这是防死锁的静态约束。
- 乱序完成 + 有序 release：多个生产 warp 乱序写完，由协调 warp 按滑动窗口把连续完成的前缀批量发布。
- head 回写取**所有消费者的最小值**、按 chunk 粒度懒惰回写（减少控制流量）。
- 简化替代：轮次×双缓冲的静态错位（FlashDMoE）——实现简单，但规模变化时会溢出，需按硬件容量预算；capacity 上限 + 允许丢弃（MoE 语义容忍）是最后的兜底。

## 7. 与黑盒通信库（NCCL 类）共存的同步纪律

- 触发一次黑盒 collective 的标准结构：wait kernel 在通信 stream 上自旋计数 → 满足后同 stream 的 collective 自然启动。阻塞纯靠 stream 内顺序，通信库零修改。
- **所有 rank 必须以相同顺序、相同分段发射 collective**：分段配置（每段 tile 数、段数）必须全 rank 一致（配置文件共享），某 rank 计数永远到不了阈值 = 全体挂死。
- 通信 stream 要高优先级；多 stream 混入同一个通信 group 可能引入隐式全局同步，用 timeline 工具验证真实并发。

## 8. 死锁面检查单（协议评审用）

- [ ] 每个 wait 的期望值，在所有执行路径（包括空转 CTA、grid < SM 数、expert 无 token）下都能被凑齐？（"没拿到任务也要补 arrive"）
- [ ] signal 的发出顺序与 wait 方的遍历顺序一致？（乱序到达 + 按序等待 = 卡死）
- [ ] 所有卡的 persistent grid 同时可驻留？（占用率、cluster 限制逐卡验证）
- [ ] 两个互相等待的 kernel（计算/通信）是否可能一个占满 SM 另一个未调度？（用"调度就绪 flag"或静态 CTA 配额约定）
- [ ] 发送窗口与接收缓冲的静态约束是否满足（窗口 ≤ 缓冲/2 或双缓冲错位）？
- [ ] 复位与下一轮写入之间有 barrier 或 CAS 语义隔离？
- [ ] 自旋有超时报警/陷阱兜底？慢 rank/死 rank 能被屏蔽而不炸全局？
