# 12 SM120（RTX PRO 5000）+ PCIe 拓扑适配经验

> 我们的实际硬件不是 8×H100 NVLink，而是 **RTX PRO 5000 Blackwell（SM120）+ PCIe 互连**。11 篇里相当一部分机制在这个组合上直接失效。本篇逐条清点"什么没了、用什么替代、账怎么重新算"，是 11 篇在本机器上的勘误与增补。凡标注【实测确认】的条目，动手前必须先在真机上跑微基准验证——本篇部分内容来自架构文档与公开资料的推断，不是在这台机器上验证过的事实。

**一句话结论**：方案依然可行，但形态变了——**从"multimem/NVLS 广播+在网归约"退到"纯 unicast pull + 本地归约"，从"红操作计数信号"退到"每 producer 独占 slot 的 st 写信号"，GEMM 模板从 wgmma 换成 warp 级 mma.sync**。TK 的 pgl/TMA/mbarrier 这层地基在 SM120 上是编译门控齐全的，但仓库里**没有任何 SM120 的 GEMM 或 parallel kernel 示例**——GEMM 流水线要自己搭，这是最大的额外工作量。

---

## 1. 硬件画像与两条红线

RTX PRO 5000 Blackwell：GB202 裁剪版，**110 SM**、48GB GDDR7（ECC）、显存带宽 ~1.3TB/s、PCIe Gen5 ×16（单向理论 64GB/s，实测经验值 50~55GB/s）。bf16 tensor 算力在数百 TFLOPS 量级（以实测为准，别抄 H100 的账）。

两条改变一切的红线：

- **红线 1：没有 NVSwitch → 没有 NVLS/multimem**。多播对象（`cuMulticastCreate`）只在 NVSwitch 系统上受支持。TK 里所有走 `mc_ptr` 的东西全部失效。
- **红线 2：PCIe 上没有跨卡原子**。`cudaDevAttrP2PNativeAtomicSupported` 在 PCIe 平台为 0——对 peer 内存做 `red`/`atom`（包括 TMA 的 `cp.reduce.async.bulk`）不受支持，结果不可靠。【实测确认：写个小 kernel 对 peer 内存 atomicAdd，对拍】
- 前置条件（红线 0）：**P2P 本身要先确认**。RTX PRO Blackwell 系列官方恢复了专业卡的 PCIe P2P（Ada 一代专业卡没有），但它依赖大 BAR + IOMMU/ACS 配置——ACS 开着会把流量强制绕行 root complex 甚至禁掉 P2P，虚拟化/云环境是重灾区。跨 CPU socket（`nvidia-smi topo -m` 显示 SYS）的对可能不支持或奇慢。**第一步永远是 `p2pBandwidthLatencyTest` + `nvidia-smi topo -m`**。如果 P2P 不可用，TK 的 pgl 整条路（IPC 映射 peer 显存）都不成立，只能退 NCCL/copy-engine + signal 的低侵入路线（08 篇级 1），本篇后面的内容全部作废。

## 2. TK 能力清单：SM120+PCIe 下的存亡表

| TK 机制 | 状态 | 说明/替代 |
|---|---|---|
| `pgl` 的 unicast 视图 `G.A[i]`（VMM+IPC） | ✅ | 需 P2P；这是活下来的主力数据面 |
| `tma::load_async(smem, G.A[peer])` 远端拉 | ✅ | 普通 load 语义，PCIe P2P 可走；注意延迟账（§4） |
| `tma::store_async(G.A[peer])` 远端推 | ✅ | 普通 store 语义 |
| `tma::store_add_async` 到**远端** | ❌ 红线 2 | 改"推到目标卡 per-source 独立 buffer + 目标卡本地加"或"目标卡 pull + 本地加" |
| `tma::store_add_async` 到**本地** | ✅ | 本地原子没问题 |
| multicast TMA store（一写八卡） | ❌ 红线 1 | 循环 unicast store ×N；或让各卡自己来拉 |
| `multimem ld_reduce/st/red`、`group::all_reduce` | ❌ 红线 1 | pull 各卡分片到本地 + 寄存器/smem 归约 |
| `signal(barrier, idx, dst_dev, val)`（red.release.sys 到远端） | ❌ 红线 2 | **st 版信号**：每 producer 独占一个 slot，`st.release.sys` 写单调递增序号；consumer 逐 slot 等（§3） |
| `signal_all` / `barrier_all` | ❌ 红线 1+2 | 自写：对 N-1 卡各做一次 st 版 signal + 本地等齐 |
| `wait(barrier, ...)` 本地自旋 | ✅ | 方向本来就对：轮询**本地**副本。绝不能改成自旋读远端——每次 poll 一个 PCIe 往返（~1.5~2μs）还占带宽 |
| mbarrier/semaphore、块内流水 | ✅ | SM120 门控齐全（`sync.cuh`） |
| `warpgroup::mma`（wgmma） | ❌ | SM90 专属（`mma.cuh:10`）。用 warp 级 `mma_AB`（mma.sync m16n8k16，SM120 路径齐全，含 FP8 m16n8k32） |
| `warpgroup::increase/decrease_registers`（setmaxnreg） | ❌ | sm_90a 专属。寄存器均分，consumer 的累加器规模要重新预算 |
| tcgen05 / `*_b200.cu` kernel | ❌ | SM100 专属 |
| cluster / DSMEM | ⚠️ 别依赖 | 仓库 parallel kernel 全是 CLUSTER_SIZE=1，保持即可 |
| `TKParallelTensor(multicast=True)` | ❌ 红线 1 | 全部 `multicast=False`；barrier 用 `pgl<gl<int,...>, N, false>`（`barrier_t` 别名是 MULTICAST=true 的，不能直接用） |
| smem 99KB（`MAX_SHARED_MEMORY`，util.cuh:71） | ⚠️ | H100 kernel 的 227KB 布局全部塞不下，tile/stage 重排（§5） |

## 3. 信号协议重写：从"红操作计数"到"slot + 序号"

11 篇的核心纪律"计数语义 + `wait ==` 精确匹配"依赖跨卡原子加，PCIe 上整个作废。替代协议（MSCCL++ 在 PCIe 上的同款思路）：

```text
barrier 布局：[信号点][producer_slot]，每个 (信号点, producer) 一个独占 int
producer：完成第 k 个单元 → st.release.sys 把 k 写进 consumer 卡上属于自己的 slot
consumer：对自己依赖的每个 producer slot 本地自旋等 值 >= 目标序号
```

要点与坑：

1. **单写者原则**：每个 slot 只有一个写者，就不需要原子——这是整个协议的根。producer 数量因此必须编译期/启动期定死（和 11 篇"精确计数"同源的纪律，只是形式变了）。
2. **序号单调递增，天然免复位**：flag 用 0/1 布尔就要复位（又是一轮跨卡同步）；用单调序号则 epoch 之间只需 consumer 记住自己的 target 递增。11 篇里"epilogue kernel 复位 + barrier_all"可以简化成"每层 target += 单元数"，只在 buffer 复用边界做一次全卡对齐。
3. **等待成本从 O(1) 变 O(P)**：consumer 要扫 P 个 slot。把 P 压小的手段就是信号粗化（06 篇）——按行块/列块聚合，别 per-tile；或者让一个 comm warp 专职扫 slot、聚合成块内 mbarrier 再放行 GEMM producer（把 O(P) 藏进专职 warp）。
4. **st.release.sys 跨 PCIe 的顺序保证**：同一 producer 先写数据（TMA store 完成 + `store_async_wait`）再 release 写信号，PCIe 保序足够；11 篇 §1.4 的"relaxed 自旋灰色地带"警告在 PCIe 上同样成立且更该防——出偶发错先升级成 acquire 读。
5. moe_dispatch_gemm 的 layer0 信号**本来就是本地的**（pull 进本地后 `red.release.gpu` 本地加），红线 2 不影响它——**pull 模式在 PCIe 上是天然正确的形态**，这也是为什么整体方案要向 pull 倾斜。本地 red 保留原样即可，不用 slot 化。

## 4. 通信账重算：PCIe 是 comm-bound 的世界

以 dispatch 为例粗算（seq=8192、top-8、H=7168、bf16、8 卡 EP、跨卡率 7/8）：

```text
每卡收发 ≈ 8192×8×7168×2B×(7/8)/8 ≈ 103MB → @55GB/s ≈ 1.9ms
GEMM0 每卡 ≈ 2×8192×8/8×7168×2048 ≈ 240GFLOP → @~200TFLOPS ≈ 1.2ms
→ comm(1.9) > comp(1.2)：comm-bound。完美 overlap 后 E2E≈1.9ms，对比串行 3.1ms ≈ 1.63×
```

（NVLink 上同样的 dispatch 是 ~0.23ms，comp-bound，overlap 藏的是通信；PCIe 上反过来，**overlap 藏的是计算**，E2E 下限 = T_comm。）这带来四个结构性变化：

1. **收益上限更高、也更值得做**（01 篇"PCIe 通信占比 35~48%"在 MoE dispatch 上会到 60%+），但**天花板锁死在 T_comm**——想再快只能减字节：**FP8 传输 dispatch**（7168×1B=7KB/token，通信直接减半）在 PCIe 上的优先级比 NVLink 高得多；SM120 的 mma.sync 有 e4m3 路径（warp.cuh:140），甚至可以 FP8 直接进 GEMM 不转回 bf16。
2. **comm SM 预算倒转**：打满 55GB/s 只要 1~2 个 SM 发 TMA bulk（NVLink 要 28~54 个的经验作废）；sweep 范围从 2~8 起步。省下的 SM 全给 GEMM——反正是 comm-bound，计算侧几乎不受干扰。
3. **延迟主导小消息**：PCIe 往返 ~1.5~2μs。逐 token 拉 14KB（bf16）是延迟受限的——必须靠深流水（每 warp 多个 outstanding TMA + 多 warp）填满带宽时延积（55GB/s × 2μs ≈ 110KB 在途才满带宽）。比 NVLink 更值得做的优化：**源端打包**——token 在源卡先按 (目标卡, expert) 排成连续段（一个本地 pack kernel 或融进上一层 epilogue），跨卡就变成少数大块传输，甚至可以交给 copy engine（0 SM）跑，kernel 只管信号和消费。02 篇的"token 重排"在 PCIe 上从优化项升级为必做项。
4. **拓扑调度权重变大**：NVSwitch 无阻塞，ring 错峰只是锦上添花；PCIe 上全员同时收发会挤爆 root port/switch 上行口。`nvidia-smi topo -m` 先画清楚哪些对是 PIX/PXB（同 switch，带宽独立）哪些是 PHB/SYS（过根桥/跨 socket，共享瓶颈），dispatch 的 ring 顺序按"同 switch 优先、跨根桥的对错开时间片"排。moe_dispatch_gemm 的 `(local_rank+i)%N` ring 起点保留，但相位要按拓扑组重排。

## 5. SM120 kernel 模板改造

仓库没有 SM120 GEMM 参考（`kernels/gemm/` 只有 h100/b200），要点自己搭：

1. **计算主体**：warp 级 `mma_AB`（rt 寄存器 tile，mma.sync m16n8k16）替代 wgmma。仍然可以保留 producer/consumer warp specialization（mbarrier + TMA 在 SM120 都在），但没有 setmaxnreg——所有 warp 均分 64K 寄存器/SM，consumer 累加器要收窄：110 SM 上合理起点是 **128×128×64 tile、每 warp 算 128×32 或 64×64 条带、384 线程/block**，而不是 H100 的 128×256。
2. **smem 99KB**：128×64 A（16KB）+ 64×128 B（16KB）= 32KB/stage → **3 stage 流水**（H100 的 4 stage×48KB 布局塞不下）。输出 tile 复用最后一个 stage 的空间（原型的 overlap 技巧照抄）。PCIe 场景 GEMM 反正不是瓶颈，先求正确再调这里。
3. **persistent grid = 110**（不是 132）；自旋死锁红线（自旋 block 数 < 110、给短命 block 留 SM）同 11 篇。
4. **占用率**：SM120 每 SM 1536 线程，384 线程/block 可 3 block/SM 或 1 block/SM 大 smem 独占——PCIe 场景倾向后者（GEMM 干扰不敏感，simple 优先）。
5. dispatch/comm block 结构不变：pull + 本地信号那套（moe_dispatch_gemm 原样）在 SM120 上语义全部成立，只需把 token 拉取深度加大（§4.3）。

## 6. 方案重构：MoE TP/EP 在 SM120+PCIe 上的形态

对照 11 篇 §4 逐条修订：

- **Layer0（dispatch⊕GEMM(W1)）**：架构照抄 moe_dispatch_gemm（pull + 本地 red + 自旋==128，全部 PCIe-safe），三处改动：GEMM 换 SM120 模板（§5）；加源端 pack（§4.3）把逐 token 远端读变成分段大块读；dispatch 传输考虑 FP8。TP 维的"multicast 推给 TP 组"不存在了——TP 组内各卡独立拉，或源端 pack 后循环 unicast 推 TP_size 份（PCIe 读放大和写放大代价相同，选实现简单的拉）。
- **Layer1（GEMM(W2)⊕combine）**：11 篇路线 A（epilogue 远端 store_add 直推）**整体出局**（红线 2）。PCIe 版路线：
  - **A'（推 + 目标卡归约）**：epilogue 乘 gate 权重后把行推到**源卡上按 (expert来源) 分槽的独立 buffer**（每写者独占区域，无原子），st 版信号；源卡一个轻量 reduce kernel（或下一层 prologue）把 top-k 槽本地加起来。多占 top_k 倍 buffer，换来无原子+大块写。
  - **B'（源卡 pull + 本地归约，更推荐先做）**：comp SM 每列块完成 → st 版信号广播给各源卡；源卡的 comm SM 按 combine 索引从 top-k 个 expert 卡**拉**自己 token 的部分输出行，FP32 本地加权累加。全 pull、无原子、信号单向，和 layer0 同构，协议只写一次。缺点是 gather 读是行粒度远端读——同样用"expert 卡先按目标卡 pack"消解。
  - TP row-parallel 的部分和归约并进同一次 pull-reduce（拉 TP_size×top_k 份加一起），不再依赖 store_add。
- **验证顺序**（替换 11 篇 §4.4 第 1 步，那个 `make run` 在 SM120 上编译不过）：
  1. 平台微基准：`nvidia-smi topo -m`、p2pBandwidthLatencyTest、自写 peer-atomic 探测、SM 驱动 TMA 跨卡带宽 vs copy engine 带宽。【本篇一切结论以此为准】
  2. `ARCH=SM120` 跑 `tests/` 确认 TK 基础层（TMA/mbarrier/mma/pgl）在本卡 OK。
  3. SM120 单卡 grouped GEMM 对拍 torch。
  4. dispatch 通信微基准：逐 token pull vs 源端 pack + 分段 pull，测出 pack 收益和所需流水深度。
  5. 拼 layer0 → 非重叠 layer1 锚点 → 融合 layer1（路线 B'），每步对拍 + 09 篇三件套。

## 7. 速查：11 篇结论在本机器上的存废

| 11 篇结论 | SM120+PCIe 状态 |
|---|---|
| moe_dispatch_gemm 可直接跑通当 baseline | ❌ 编译不过（wgmma/setmaxnreg/227KB smem/132 grid），但**架构照抄、GEMM 主体重写** |
| multicast TMA / signal_all / multimem 三件套 | ❌ 全灭，unicast + st 信号 + pull 归约替代 |
| `wait ==` 精确计数纪律 | ✅ 精神保留，形式变为"slot 数精确可数 + 序号 ≥ target" |
| 自旋 block < SM 数死锁红线 | ✅ 不变（110） |
| comm/comp 任务序一致 | ✅ 不变 |
| num_comm_sms 28~54 | ❌ 改 2~8 起步 |
| 层尾 barrier_all 防串扰 | ✅ 需要但要自写（st 版全对全）；序号化信号可减少对齐次数 |
| bf16 归约精度/确定性 | ✅ 不变；pull 本地归约反而更容易上 FP32 |
| 收益预期 | 从"藏通信"变"藏计算"，E2E 下限 = T_comm；FP8 传输成为头号杠杆 |
