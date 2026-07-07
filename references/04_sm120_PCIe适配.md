# sm120 / PCIe 适配指南

## 1. 目标硬件画像

| 维度 | 源环境（A100 sm80） | 目标（RTX Pro 5000 sm120） | 影响 |
|---|---|---|---|
| 互连 | NVLink（数百 GB/s，全互连） | **PCIe**（~25-60 GB/s，可能过 root complex） | 通信慢一个量级；可隐藏的通信占比下降 |
| P2P | 原生支持，NVSwitch all-to-all | ⚠️ PCIe P2P 是否可用取决于主板/驱动/卡（消费卡常被限制） | 全部跨卡直接访存的前提，第一件要验证的事 |
| 跨卡原子 | 支持 | ⚠️ `cudaDevP2PAttrNativeAtomicSupported` 大概率 false | 跨卡握手只能用 store/load（ring-token），不能 RMW |
| Tensor core | mma.sync (sm80) | sm120 有自己的第 5 代 tensor core，**非** wgmma（sm90）**非** tcgen05（sm100）；血统更接近 Ada 的 warp 级 MMA ⚠️ | MMA/mainloop 需用 CUTLASS 的 sm120 支持或 sm80 风格模板重编 |
| CUTLASS | 2.x grouped GEMM 成熟 | ⚠️ CUTLASS 3.8+/4.0 有 sm120 collective；**grouped GEMM × sm120 组合成熟度未知** | 迁移第一风险，先跑最小 grouped GEMM 实验 |
| CUDA | 12.4 够用 | 需 **12.8+**（编译 `sm_120`/`sm_120a`） | 环境准备 |
| NVSHMEM | 单机 NVLink 无障碍 | ⚠️ 纯 PCIe transport 支持情况未知 | 备选 cudaIPC |
| SM 数 | 108 | 视卡型，运行时 `cudaDevAttrMultiProcessorCount` 查询 | 所有 SM 相关常数重新调 |
| 显存 | 40/80 GB HBM | 16-32 GB GDDR7 | 对称缓冲 + 权重的预算更紧 |

> ⚠️ 标注的结论基于公开资料推断，**动手前必须在目标机器实测验证**。

---

## 2. 概念层：什么原样保留

以下设计**与互连/架构无关**，照原理文档复刻即可：

- 路由元数据契约（splits / scatter_index / gather_index / output_vec_scale）与生成算法
- layer0 沿 M 切 + 按 (expert, 源rank) 排序 + per-rank 累积表定位依赖
- layer1 沿 N 切 n_split 片 + 分层计数
- "红绿灯" flag 协议（数据先落、灯后亮；acquire 自旋消费）
- **消费侧只读本地**的缓冲结构（layer0 GEMM 读本地 input_buffer/本地 flag；layer1 reducer 本地读、远端写）
- device 侧生成 problem/调度表、唯一 D2H 同步点（M_this_ep / splits_cpu）
- 双流 + event 的 host 编排骨架、水平融合（sm_margin + 高优先级流 + 少量通信 CTA）
- fp32 累加、gate 权重经逐行 scale 进 combine 的数值方案
- 调优表（形状 → tile 形状/n_split/RS_BLOCKS/sm_margin）的组织方式

---

## 3. 通信层：必须替换/验证的部分

### 3.1 第一步：探测目标机器的 P2P 能力

写一个探测程序依次确认（结果决定后面走哪条路）：

1. `cudaDeviceCanAccessPeer(i, j)` 每对卡
2. `cudaDeviceGetP2PAttribute(cudaDevP2PAttrNativeAtomicSupported, ...)`
3. `cudaIpcOpenMemHandle` 跨进程映射 + kernel 内对映射指针 ld/st 的正确性
4. 实测三种搬运的带宽：`cudaMemcpyAsync` P2P、SM 远端写（st）、SM 远端读（ld）——**期待远端读显著慢于远端写**
5. ⚠️ NVSHMEM：能否在纯 PCIe 上 `nvshmem_init`（UID 模式）+ `nvshmem_ptr` 返回可解引用指针

### 3.2 三档方案（按探测结果选）

| 档位 | 条件 | 做法 |
|---|---|---|
| **A：P2P 可用**（期望路径） | 1/3 通过 | 保持原结构：对称缓冲用 cudaIPC（NVSHMEM 可选）；AG 用 `cudaMemcpyAsync` P2P DMA + `cuStreamWriteValue` 点灯；RS 用 push 模式远端写。**Ring1D 固定替代 All2All**（无 NVSwitch）；双 NUMA 机器用 Ring2D |
| **B：P2P 不可用，但需要 kernel 级细粒度** | 3 失败 | AG/RS 的跨卡段改 **host-staging 两跳**（GPU→pinned host→GPU 的两次 async DMA，仍按段流水+点灯，"远程点灯"改为"数据落本地后本地点灯"）；带宽减半，重叠结构不变 |
| **C：退守** | 上述皆不理想 | 通信直接用 NCCL send/recv 按段做 ring，flag 由 host 在段完成回调处写；牺牲部分细粒度，保住 layer0 的 tile 等待与 layer1 的 N-split 交付 |

### 3.3 PCIe 专属的协议修正（无论哪档）

- **禁用 pull/read 模式**：数据流动只用远端写（push）或 DMA，绝不让 kernel 做远端读。
- **跨卡握手无 RMW**：ring 的 tile 握手（layer1）和 group barrier 全部用 release-store + acquire-load 的 ring-token；计数类逻辑挪到各卡本地做。
- **加粗握手粒度**：PCIe 上 system-scope flag 传播延迟高，逐 tile 握手可能吃不消——把 layer1 ring 的握手单位从 tile 加大到"每 stage 每片一次"（或把 reducer 的 M-tile 调大），profile 决定。
- **点灯位置**：生产者向消费者本地 flag 写（一次跨卡 store），消费者永远轮询本地——避免轮询流量上 PCIe。
- **n_c 加大**：通信 CTA 数（RS_BLOCKS）与 AG 的并发段数要按 PCIe 带宽重新 profile，预期比 A100 上更大。
- **对称缓冲显存预算**：`input_buffer(max_ntokens×K) + reduce_buffer(ntokens×N) + gemm_out(M_this_ep×N)`，按 16-32 GB 卡核算 max_ntokens 上限。

---

## 4. 计算层：GEMM 路径怎么建

1. **先做实验**：用 CUTLASS 4.x 在 sm120 上实例化一个最小 **bf16 grouped GEMM**（不带任何融合）。两条候选路线，按可编译性/性能择优：
   - ⚠️ CUTLASS sm120 CuTe collective（若 grouped/array 变体可用）
   - CUTLASS 2.x 风格模板以 `sm_89` 兼容路径编译到 sm120（消费 Blackwell 与 Ada 的 warp-MMA 血统相近，PTX JIT 可运行，但吃不到新 tensor core 峰值 ⚠️）
2. 无论哪条，都要确认支持/可加装：**GatherA / ScatterD 行级间接寻址**、grouped ProblemVisitor 或等价 tile 调度器、epilogue 尾部插入自定义回调（计数/置 flag）、mainloop 前插入等待段。
3. **tile 形状/stage 数重新筛**：sm120 的 shared memory / 寄存器 / L2 与 A100 不同，源环境的 128×128×32×3stage 未必最优甚至未必放得下；按 smem 预算过滤 + profile。
4. **所有硬件常数运行时化**：SM 数、reducer 线程组织、RS_BLOCKS——全部改成查询+配置，不写死。

---

## 5. 构建与工程

- CUDA 12.8+；`-gencode arch=compute_120,code=sm_120`（用到 Blackwell 专属指令时需 `sm_120a` ⚠️）
- `TORCH_CUDA_ARCH_LIST=12.0` 编 torch 扩展；pybind + torch custom class 常规做法
- 不需要复刻编译期代码生成系统：**手写 2~4 组模板实例 + 运行时查表**足够
- 若走 NVSHMEM：pip 的 `nvidia-nvshmem-cu12`（≥3.x）+ UID bootstrap；确认其 license/内核模块在消费平台可用 ⚠️
- NCCL 仅作控制面/退守数据面，任一新版都支持 Blackwell

---

## 6. 验证方法

### 6.1 正确性验证（与 moe_bench 相同思路）

1. 固定 seed 生成输入/权重/路由，纯 torch 参考（逐 expert 循环、fp32 累加）
2. 每一步独立对拍：AG 后全量 buffer 相等 → layer0 输出（expert 排序行）相等 → reducer 单卡版相等 → 全链路输出 `assert_close`（FP8: atol=3.5e-2, rtol=3.5e-2）
3. 分布式验证利用"每 rank 输出 = 其 token 分片过全量专家的结果"这一性质，可用单卡参考核对多卡输出
4. 边界：某专家 0 行、极端倾斜路由（全部 token 同一专家）、num_tokens 不能整除、EP 边界专家

### 6.2 性能预期

- 与"AllGather → grouped GEMM → ReduceScatter 串行"的 baseline 对比（同输入同验证）
- PCIe 参照档：单层 **1.2x-1.5x** 即达到论文 L20 水平；M 小时（decode 场景）收益应更明显
- 若通信时间 > 计算时间（PCIe 上大 M 时可能发生）：收益封顶于计算时长，重点转向"吃满 PCIe 带宽"（加粗段、加大 n_c）而非更细的重叠

---

## 7. 常见坑清单

- ⚠️ **消费卡 P2P 被驱动禁用**：`cudaMemcpyAsync(cudaMemcpyDefault)` 会静默走 host 中转——带宽骤降但不报错，务必用带宽数字验证真 P2P
- **flag 忘记每次 forward 清零** → 第二次调用直接读到旧灯，结果错但不崩——把"清零完成"纳入 prepare event 的依赖
- **点灯与数据的顺序**：DMA 路径必须同一 stream 上先 memcpy 后 write_value；kernel 路径必须 fence + release store。少一个 fence 在 NVLink 上往往侥幸通过，在 PCIe 上必现
- **grouped GEMM 的空 problem**（splits=0）要能跳过；tile 计数的"该 problem tile 总数"按实际 M_i 算
- **reducer 与 GEMM 抢 SM**：sm_margin 忘加或加错 → reducer CTA 排队到 GEMM 结束才启动，"重叠"静默退化为串行——用 nsight/torch profiler 确认两 kernel 时间线真的并行
- **高优先级 stream 忘记设置** → 同上退化
- **bf16/fp8 直接累加**（漏 fp32）→ 大 K 下精度崩
- **EP 时 routing_idx 忘记重基/过滤非本 rank 行** → 越界读 gemm_out
- **显存**：max_ntokens 按最大档预分配对称缓冲，OOM 要在构造期报而不是 forward 中段

---

## 8. sm80 vs sm90 路径对照（为何 sm120 选 sm80 形态）

| 维度 | sm80（V2，推荐参考） | sm90（V3，勿参考） |
|---|---|---|
| kernel 形态 | 非持久 grouped GEMM，多 block/SM，warp 切换隐藏延迟 | persistent warp-specialized（producer/consumer warpgroup + mbarrier） |
| 数据加载 | cp.async | TMA（B）+ cp.async（gather-A）+ 运行时 tensormap |
| MMA | mma.sync | wgmma（warpgroup） |
| layer0 等待 | 每 tile 在 kernel 内 ballot+自旋 | producer warp 按 group 等待 |
| layer1 融合 | 两 kernel 双流水平融合 | 单 kernel 内 CTA specialization（末尾 N 个 CTA 做通信） |
| 依赖硬件 | cp.async/mma.sync/ballot/atomic_ref —— **sm120 全都有** | cluster/CGA、multicast、PDL、`sm_90a` —— sm120 形态不同或缺失 |

sm120（消费级 Blackwell）没有 wgmma，也不是数据中心 Blackwell 的 tcgen05 体系；其可靠的公共子集恰好是 sm80 形态所依赖的那组特性。因此**复现走 sm80 形态**，MMA 用 CUTLASS 对 sm120 可用的 collective/模板生成。
