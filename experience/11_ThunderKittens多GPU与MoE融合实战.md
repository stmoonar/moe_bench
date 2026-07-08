# 11 ThunderKittens 多 GPU 编程与 MoE tile 粒度融合实战

> 本篇与 01~10 篇不同：不是跨工作的思想综述，而是针对 ThunderKittens（下称 TK）这一具体框架的**开发经验文档**——多 GPU 基础设施怎么用、仓库里 4 个通信-计算融合 kernel 的机制逐个拆解、以及"MoE 层 TP/EP 的 tile 粒度通信计算 overlap（GEMM kernel 不拆开，在内部融入通信）"的可行性结论与实施方案。文中出现的 API 名与文件路径都是仓库里真实存在的，可直接对照源码阅读。

> ⚠️ **本篇的机制分析以仓库 kernel 的目标硬件（8×H100 NVLink/NVSwitch）为背景。我们的实际机器是 RTX PRO 5000（SM120）+ PCIe——multimem/NVLS、跨卡原子、wgmma 等关键机制在该组合上不可用，方案形态需按 [12_SM120与PCIe拓扑适配.md](12_SM120与PCIe拓扑适配.md) 修订，两篇必须对照着读。**

**先给结论**：这个方案**可行，且 TK 仓库已经自带了一半的答案**——`kernels/parallel/moe_dispatch_gemm/` 就是"MoE dispatch 通信融入 grouped GEMM、GEMM 不拆开"的完整原型（DeepSeek-V3 配置，8×H100）。缺的另一半是 layer1（expert GEMM + combine）和 TP 维度，可以用同仓库的 `gemm_rs` / `gemm_ar` 模式拼出来。TK 的定位恰好是做这件事的最短路径：它把跨卡数据面（multicast TMA、远端 TMA、TMA 原子加、multimem 在网归约）和控制面（跨卡计数 barrier）都做成了和单卡 tile 原语同一风格的 device 函数，融合通信只是在标准 GEMM 模板里"多写几行"。

---

## 1. TK 多 GPU 基础设施地图

### 1.1 Host 侧：TKParallelTensor（`include/pyutils/parallel_tensor.cuh`）

- 运行模型：**torchrun 每 GPU 一进程**（8 进程/8 卡）。进程间通过 `KittensBroker`（Unix socket）交换 IPC handle。
- `TKParallelTensor(shape, dtype, local_rank, local_world_size, multicast)`：
  - 自分配路径：用 CUDA VMM（`cuMemCreate` 系）分配 + fd 交换，每个进程拿到**全部 8 卡副本的 VA 指针**（`raw_ptrs_[8]`）；`multicast=True` 时额外创建 NVLS multicast 对象，绑定所有卡后映射出一个 `mc_ptr`（写一次 = 写 8 卡，读 = 在 NVSwitch 里归约）。
  - 包装已有 torch tensor 的路径走 legacy cudaIpc，**不支持 multicast**——需要多播的 buffer 必须让 TK 自己分配。
  - 限制：contiguous、dim≤4、每进程只应有一个 broker（建第二个 world 会打警告且不安全）。
- `kittens::py::parallel_tensor_to_pgl<PGL>(t)` / `tensor_to_gl<GL>(t)` 在 entrypoint 里把 tensor 转成 device 侧布局；`kittens::py::launch_kernel<config, globals, kernel>(G)` 发射。
- 实测注意：multicast 对象有最小 2MB 粒度（`moe_dispatch_gemm/benchmark.py` 里 barrier 特意开成 `(2, N)` 形状，注释原话 "2 MB is the minimum requirement for multicast object"）。

### 1.2 Device 侧类型：pgl（`include/types/system/pgl.cuh`）

```
pgl<GL, NUM_DEVICES, MULTICAST, TMA_Types...>
  ├── gls[NUM_DEVICES]   // 各卡副本的 unicast 视图，G.A[i] 即第 i 卡
  ├── mc_ptr             // NVLS 多播虚地址，mc_ptr_at(coord) 取元素指针
  └── tma_descs          // 建在 mc_ptr 上的 multicast TMA descriptor
```

- **同一套 tile API 对 pgl 泛化**：`tma::load_async(smem, G.A[peer], coord, sem)` 就是从 peer 卡拉数据（NVLink P2P，读远端像读本地）；`tma::store_async(G.A_pgl, smem, coord)` 走 multicast descriptor，**一条 TMA 指令把一个 tile 同时写进 8 张卡**；`tma::store_add_async(G.C[peer], smem, coord)` 是对远端卡的 TMA reduce-add（bf16 支持）。
- `barrier_t<N>` = `pgl<gl<int,...>, N, true>`，即"每卡一份、带多播指针的 int 数组"，是所有跨卡信号的载体。
- `NUM_DEVICES` 是编译期常量（仓库 kernel 全部写死 8，改卡数要改代码重编）。

### 1.3 multimem 在网计算（`include/common/multimem.cuh`）

`multimem<T>::ld_reduce / st / red` 三件套，T 覆盖 int/uint/float/float2/bf16/bf16_2/half/half_2：

- `ld_reduce`：对多播地址做一次 load，NVSwitch 返回 8 卡数据的归约结果（bf16 用 `acc::f32` 累加）。
- `st`：写多播地址 = 广播到 8 卡。
- `red`：远端原子归约（release.sys 语义）。
- 每个操作有 WEAK / STRONG（acquire/release.sys）两档内存序。
- group 级封装：`group<N>::all_reduce<ROWS, COLS, op>(pgl, idx)`（`include/ops/group/memory/tile/parallel_global_to_global.cuh`）——N 个 warp 分行对一个 tile 就地做 ld_reduce+st，即 **tile 粒度的在网 AllReduce**，无 smem 中转。

### 1.4 跨卡同步原语（`include/ops/thread/util/sync.cuh:292-329`）

```
signal(barrier, idx, dst_dev, val)   // red.release.sys.add 到指定卡的计数器
signal_all(barrier, idx, val)        // multimem.red：一条指令给所有卡的计数器 +val
wait(barrier, idx, dev, expected)    // ld.relaxed.sys 自旋，直到 == expected（精确匹配！）
barrier_all(barrier, idx, dev)       // signal_all(+1) + wait(==N) + 本地 -N 复位
```

三个必须刻进脑子的语义细节：

1. **`wait` 是 `== expected` 精确匹配，不是 `>=`**。信号方数量必须编译期/运行期精确可数，多一个少一个都挂死（见 §3.1 ag_gemm 的 worker 计数公式）。
2. 数据依赖信号用 **release 红操作**发送（保证数据先于信号可见），但等待侧是 **relaxed 自旋**。TK 全仓库都这么用且实测可靠；但严格内存模型下"generic proxy 的 relaxed load"与"async proxy（TMA）的后续读"之间的可见性是灰色地带——如果你的 kernel 出现**偶发、低概率**的数据错配，第一个怀疑对象就是这里（升级成 acquire load 或补 fence 试试）。
3. 本地信号可以降 scope：`moe_dispatch_gemm` 里 dispatch SM 把 token 拉到**本地**显存后用 `red.release.gpu`（device scope）而非 sys scope 发信号——写和读都在本卡，不需要跨卡序。scope 能降就降，sys 红操作比 gpu 贵。

---

## 2. 仓库 4 个融合 kernel 的机制拆解（`kernels/parallel/`）

四个 kernel 覆盖了通信融入 GEMM 的全部三种数据流向（consumer 等上游 / producer 推下游 / 双向归约），共享同一个骨架（§3）。GEMM 本体在四个 kernel 里**完全一致**：128×256×64 tile、4 级 smem 流水、2 consumer warpgroup（232 寄存器）+ 1 producer warpgroup（40 寄存器，warp0 管 load、warp1 管 store）、SUPER_M=12 的 super-grouping swizzle、132 block persistent grid。通信是"贴"上去的，GEMM 没有为通信拆开或改小。

### 2.1 AG+GEMM（`ag_gemm/ag_gemm_h100.cu`）——consumer 等上游，广播型

- **comm SM**（grid 尾部 `num_comm_sms` 个 block）：每 warp 管一个 64KB chunk（256 行×128 列 bf16），把本卡 shard 从本地 TMA load 进 smem，再用 **multicast TMA store 一次广播到 8 卡**；每完成一行块的最后一个列 chunk，`signal_all(barrier[row], +1)`。
- **comp SM producer**：任务序 = 先算本地 shard 全部 tile（零等待，SUPER_M swizzle），再按"每个 peer 的第 r 行"轮转消费远端；远端 tile 前 `wait(barrier[row], == num_comm_workers_per_stage)`。
- **计数语义的教科书细节**：期望值 `num_comm_workers_per_stage = min(num_comm_sms × NUM_CHUNKS, num_iters/2)`——精确等于"会对该行发信号的 comm worker 数"。这就是 `wait ==` 语义逼出来的纪律：**设计信号协议时先把'谁会发、发几次'算成闭式表达式**。
- **信号粒度**：256 行一个计数器（比 tile 粗），验证了 06 篇"信号粗于 tile"的经验。
- **epilogue kernel**（第二次 launch）：复位 barrier + `barrier_all` 保证 8 卡一起退出——否则下一个 iteration 的快卡会污染慢卡还在用的 barrier。

### 2.2 GEMM+RS（`gemm_rs/gemm_rs_h100.cu`）——producer 推下游，最省事的一档

- **没有 comm SM，没有任何信号**。storer warp 算出 tile 属于哪张卡的 shard，直接 `tma::store_add_async(G.C[目标卡], tile)` 把部分和**原子加进目标卡显存**。
- 时序控制全靠一个下标公式：`dev_task_offset = ((dev+1) × num_blocks/8) % num_blocks`，各卡从不同 shard 起算、环形推进、**本卡 shard 最后算**——8 卡的写流量在时间上错开，NVLink 无热点（03 篇"swizzle 优先于新调度器"的最纯实例）。
- 完成语义：第二个 1-block kernel 做 `barrier_all`。代价：bf16 原子加求和顺序非确定（训练可复现性问题，08 篇已提）。

### 2.3 GEMM+AR（`gemm_ar/gemm_ar_h100.cu`）——per-tile 信号 + 在网归约

- comp SM 算完 tile 写**本地** C，然后 `signal(barrier, {row,col}, task_id % 8, +1)`——每个 tile 静态指定一张"归约负责卡"。
- 负责卡的 comm SM 按**与 comp 相同的任务序**遍历自己负责的 tile，`wait(==8)`（8 卡都算完了这个 tile），然后 `group::all_reduce`：multimem ld_reduce 读出 8 卡之和、multimem st 广播回 8 卡，就地完成、无 smem 中转。
- 源码注释两次强调 "**ordering must match with comp SM signaling**"——comm 与 comp 的 task 遍历序必须一致，否则 comm 会在没就绪的 tile 上长时间空转，堵住已就绪的 tile。
- 这是**唯一 per-tile 信号**的 kernel：因为 AR 的消费粒度就是 tile，粗化会推迟归约。粒度跟着消费方走（06 篇）。

### 2.4 MoE dispatch + grouped GEMM（`moe_dispatch_gemm/moe_dispatch_gemm_h100.cu`）——你要的 layer0 原型

**一次 kernel launch 里**：`blockIdx < num_comp_sms` 的 block 跑 grouped GEMM，其余 block 跑 dispatch。

- **Host 侧预计算**（`benchmark.py`）：routing 结果整理成 `pull_dispatch_indices (num_padded_local_tokens, 2)` = 每个"我卡要的 token"的 (源卡, 源下标)，按 expert 排序、每 expert 128 对齐 padding、源卡顺序从 `(local_rank+i)%8` 开始形成 ring（8 卡拉取流量错开）；外加 `padded_tokens_per_expert` 前缀和定位。
- **dispatch block**：每 block 16 个 token、每线程管 1 个 token：从 `G.pre_tokens[src_dev]` 用 TMA **拉**整条 7168 维 token 向量进 smem → TMA store 到本地 `post_tokens` → `red.release.gpu.add` 给本地 `barrier[token_idx/128]` +1。纯 pull 模式，写本地所以信号只需 gpu scope。
- **GEMM producer**：按 expert 顺序遍历（cumsum 出每个 expert 的 row block 区间——grouped GEMM 不为路由切小 tile），每个 row block 先 `ld.relaxed.gpu` 自旋到计数 **==128**（该 128-token 行块的 token 全部到位），然后进入完全标准的 4 级流水 mma。**GEMM 结构零改动，"通信融入"物化为 producer 的一个自旋等待**。
- **grid 超额的调度技巧**：dispatch block 数 = `ceil(tokens/16) + num_comp_sms`，可以远超 132。能行的前提：comp block（会无限自旋）只占前 `num_comp_sms` 个，**剩下的 SM 留给短命的 dispatch block 排队消化**。推论=死锁红线：自旋型 block 的数量必须 < SM 数，且给非自旋 block 至少留 1 个 SM。
- **数字**：DeepSeek-V3 配置（H=7168, I=2048, 256 expert, top-8），seq 8K~128K，`num_comm_sms` sweep 28~54（132 里拿出 21%~41% 做通信——MoE dispatch 数据量大，比 ag_gemm 的通信预算高一个档位）。
- dispatch 顺序与 GEMM 消费顺序天然对齐（indices 就是按 expert 排的），这是 02 篇"重排让依赖变成前缀"的落地。

### 2.5 其他可参考

- `all_to_all.cu` / `all_gather.cu` / `reduce_scatter.cu` / `all_reduce.cu`：纯通信 kernel，是各数据面原语的最小用例；`all_reduce_educational.cu` 有注释版讲解。
- `ring_attn` / `ulysses_attn`：通信融入 attention 的例子（P2P 环 / all-to-all 换维）。
- `gemm_ar_h100_lcsc.cu`：AR 的 local-compute-store-communicate 变体，可对照两种 AR 组织方式的取舍。
- fp8 变体（`ag_gemm_fp8`, `gemm_rs_fp8`）是 B200 的，H100 上不用看。

---

## 3. 提炼：TK 融合 kernel 的标准骨架

```text
launch 1（主 kernel，persistent，NUM_BLOCKS=132）:
  if (blockIdx < num_comp_sms):   comp 角色 —— 标准 TK GEMM 模板
      producer warpgroup(40 reg):
          warp0 loader: [若消费远端数据: wait(barrier, == 精确计数)] → tma::load_async 流水
          warp1 storer: 等 outputs_arrived → 写出（本地 store / 远端 store_add / multicast store）
                        [若下游要信号: signal / signal_all]
      consumer warpgroup ×2 (232 reg): wait → mma → arrive，完全不知道多 GPU 的存在
  else:                            comm 角色（可选，push 型融合可以没有）
      按与 comp 一致的任务序遍历: [wait 信号] → 数据面操作(拉/广播/ld_reduce) → [发信号]

launch 2（epilogue，几乎免费）:
  复位 barrier + barrier_all（8 卡对齐退出）
```

四个自由度，按你的算子填空：

| 自由度 | 选项（TK 原语） |
|---|---|
| 数据面 | 拉：`load_async(G.X[peer])` ／ 推+归约：`store_add_async(G.X[peer])` ／ 广播：multicast `store_async(pgl)` ／ 在网归约：`multimem ld_reduce/st` |
| 控制面 | 无信号（纯 push）／ 本地 gpu-scope 计数 ／ 跨卡 `signal`（点对点）／ `signal_all`（多播） |
| 信号粒度 | per-tile（消费即 tile，如 AR）／ 行块级（128~256 行，如 AG、MoE）／ 无 |
| 时序 | swizzle：本地先算、ring 偏移错开、通信侧与计算侧同序 |

---

## 4. MoE TP/EP tile 粒度 overlap 的实施方案

目标形态（单机 8 卡 NVLink，EP=8 或 TP×EP 混合）：

```text
[gating/routing] → dispatch ⊕ groupedGEMM(W1) [+激活] → groupedGEMM(W2) ⊕ combine
                   └────── layer0，已有原型 ──────┘   └── layer1，需自己写 ──┘
```

### 4.1 Layer0：dispatch ⊕ GEMM(W1) —— 直接继承 moe_dispatch_gemm

原型已可跑，生产化要补三件事：

1. **routing 上 GPU**：原型的 `pull_dispatch_indices` 在 host 用 Python 算。生产版需要一个轻量 metadata kernel（或融进前一层 epilogue）：gating → 每卡广播自己 token 的 top-k 选择（小数据，`signal_all`/multicast 即可）→ 各卡本地 sort/cumsum 出 pull indices。**避免 CPU 同步**：`num_padded_local_tokens` 不要回传 host——buffer 按上限分配，kernel 内读 device 端计数（05 篇 CPU-sync-free 经验）。
2. **gated FFN**：W1/W3 双矩阵（SiLU(xW1)⊙xW3）可以共享同一次 dispatch 和同一个 A tile 流水，B 侧两条 tile 流交替，或激活融进 epilogue。
3. **TP 维（若 TP×EP 混合）**：W1 按 I 维列切（column-parallel）时 layer0 **不产生额外通信**——每 TP rank 用同样的 dispatch 结果算自己那份列。dispatch 的目标从"expert 所在卡"变成"expert 所在 TP 组"，token 要送到组内每张卡：把 `pre_tokens` 做成 multicast pgl，dispatch 改用 multicast store 推（一次写整个 TP 组），或各 rank 独立拉（NVLink 读放大 TP_size 倍，TP≤2 可接受，TP≥4 建议 multicast 推）。

### 4.2 Layer1：GEMM(W2) ⊕ combine —— 用 gemm_rs / gemm_ar 模式拼装

combine = "expert 输出按 token 回源卡 + top-k 加权求和"，本质是**带 scatter 的 ReduceScatter**。两条路线（对应 08 篇 §5，现在有了 TK 具体化）：

**路线 A：epilogue 直推（gemm_rs 式，先做这个）**
- storer warp 在寄存器里给 C tile 乘 gate 权重（consumer 存 smem 前乘更顺），然后**按行**推：C tile 的 128 行属于同一 expert 但源卡各异，tile 级 TMA store 用不了，退化成 per-row 的 `sv_bf<COL_BLOCK>` 级 `store_add_async(G.out[src_dev], row_sv, {src_token_idx, col_block})`。行长 256×2B=512B/列块——**够长，NVLink 上行粒度写不致命**，但比 tile store 差；把 COL_BLOCK 开大（如整个 I 维一次写）可改善。
- 需要 `combine_indices (num_padded_local_tokens,) → (src_dev, src_token_idx)`：就是 layer0 pull indices 的逆映射，metadata kernel 顺手生成。
- top-k 跨 expert（可能跨卡）加到同一行：`store_add_async` 天然处理；源卡输出 buffer 初值放 shared expert 结果或清零。
- 完成语义：per-源卡计数信号（每推完一行 `signal(barrier_out, {row_blk}, src_dev, +1)`，源卡等 top-k×行数的精确计数），或者不发信号、层尾 `barrier_all`（先用后者，简单且够用，除非下一层还要 tile 级消费）。

**路线 B：comm SM gather（gemm_ar/Comet 式，路线 A 的写出成为瓶颈时再上）**
- comp SM 零改动，每 tile 写本地 + 本地 gpu-scope 信号；comm SM 等某列块全部 tile 就绪后按 token gather、FP32 加权累加、按目标卡**聚合成大消息**写出。写效率高、GEMM 完全不动，代价是多一次本地读 + comm SM 预算。

**TP 维的 combine**：expert 内 W2 行切（row-parallel）时输出是 partial sum，combine 兼做 TP 归约——`store_add_async` 让 TP 组内多卡加到同一目标行即可，一步到位；精度敏感就先 TP 组内 `multimem ld_reduce` 归约（fp32 累加）再单卡推。**注意**：所有原子加路径求和顺序非确定，训练场景要么接受、要么用 fp32 combine buffer 收窄误差。

### 4.3 信号与 barrier 设计清单

- barrier 全部用 `barrier_t<8>`（int 计数器 pgl，multicast=True 建）；布局按信号粒度：layer0 每 128-token 行块 1 个、layer1 每列块或每源卡行块 1 个。
- **每个计数器先写下闭式的期望值公式**（谁发、发几次），再写代码——`wait ==` 语义下这是纪律不是建议。
- 复位：双 buffer ping-pong（barrier 开 `(2, N)`，奇偶层交替用）省掉 epilogue 复位 kernel；或照抄原型的 epilogue kernel + `barrier_all`。多 iteration 场景**必须**有 8 卡对齐点，否则快卡下一轮污染慢卡的 barrier——这是最常见的偶发翻车源。
- layer0→layer1 之间不需要跨卡同步（W2 的输入就是本卡 W1 的输出），两次 launch 或同 kernel 两阶段皆可；先两次 launch，profile 后再决定要不要合。

### 4.4 实施顺序（每步都有正确性锚点）

1. 8 卡机器上先跑通原型：`export ARCH=SM90; cd kernels/parallel/moe_dispatch_gemm; make run`，拿到 baseline 和 num_comm_sms sweep 曲线，确认环境（NVSwitch/NVLS、fabric、IPC）没坑。
2. 单独写**非重叠版 layer1**（GEMM 完 → 独立 combine kernel），对拍 torch 参考实现——这是后面所有融合版的正确性锚点。
3. 融合路线 A，对拍步骤 2；测 09 篇三件套（T_comp_alone / T_comm_alone / T_total，overlap 效率目标 ≥0.8，计算干扰 ≤1.05）。
4. routing 上 GPU、去 CPU 同步；再考虑 TP 维、路线 B、两 GEMM 合 kernel。

---

## 5. TK 特有的坑（补充 10 篇通用清单）

1. **`wait` 精确匹配挂死**：信号次数算错 ±1 = 无限自旋。ag_gemm 的 `min(num_comm_sms×NUM_CHUNKS, num_iters/2)` 是范例——期望值必须是闭式可推的。
2. **自旋 block 占满 SM 死锁**：会自旋等待的 persistent block 数 < SM 数（H100=132），给非自旋 block 留出口；`num_comm_sms ≥ 1` 是 moe_dispatch_gemm 不死锁的隐含前提。
3. **comm/comp 任务序不一致**：gemm_ar 源码两处注释警告。两侧遍历公式要么共享代码，要么加静态断言级的对拍测试。
4. **忘记层尾 `barrier_all`**：单次运行正确、循环 benchmark 偶发错——快卡提前进入下一轮改写了慢卡还在读的 buffer/barrier。
5. **multicast 只支持 TK 自分配**：包装现有 torch tensor 的 TKParallelTensor 走 legacy IPC，`multicast=True` 会直接 CHECK 失败；multicast 对象最小 2MB。
6. **编译期假设**：NUM_DEVICES 写死；K 必须是 `PIPELINE_STAGES×RED_BLOCK=256` 的倍数（producer 里 outputs_finished 的等待位置依赖此假设，源码有注释）；M 侧 expert padding 到 ROW_BLOCK=128。padding 率高时（小 batch 多 expert）grouped GEMM 有效算力打折——先按 06 篇算 wave 账。
7. **relaxed 自旋 + async proxy 的灰色地带**：见 §1.4；偶发数据错配先怀疑内存序，再怀疑别的。
8. **bf16 归约精度**：multimem ld_reduce 内部 f32 累加但存回 bf16；store_add_async 逐次 bf16 舍入。top-k=8 的 combine 建议 fp32 buffer。
9. **硬件门槛**：multicast/NVLS 需要 NVSwitch（H100 8 卡 HGX OK；PCIe 互联机器 mc_ptr 路径全废，退 unicast `gls[i]` + `signal` 点对点，架构上 pgl 接口不变）。`ARCH=SM90`，Blackwell 用对应 `_b200.cu` 风格重写 mma 部分。
10. **每进程一个 broker**：不要在同一进程里对两个不同 world_size 建 TKParallelTensor。

---

## 6. 与 01~10 篇的对照索引

- 骨架的 block specialization、comm SM 预算 → 03 篇；moe_dispatch_gemm 的 28~54 sweep 是该篇"通信 SM 预算要 sweep"的实测例。
- producer/consumer warpgroup、寄存器 40/232、4 级流水 → 04 篇模板的 TK 官方实现。
- `wait ==` 计数语义、release 红/relaxed 自旋、复位策略 → 05 篇；TK 把"计数语义优于布尔 flag"做成了唯一选项。
- 信号粗于 tile（AG/MoE 行块级）vs per-tile（AR）→ 06 篇粒度解耦。
- multimem/NVLS、push-pull 选择 → 07 篇；TK 是"multimem 数据面 + IPC unicast 数据面"双修的参考实现。
- MoE layer0/layer1 路线 → 08 篇 §4/§5，本篇 §4 是其 TK 具体化。
