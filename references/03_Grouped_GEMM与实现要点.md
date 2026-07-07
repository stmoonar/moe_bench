# Grouped GEMM 与实现要点

## 1. CUTLASS 与 tile 化 GEMM

### 1.1 tile 是基本单位

CUTLASS 的 GEMM 把输出矩阵切成 `BLK_M × BLK_N` 的 tile，每个 threadblock 负责一个（或多个）tile：沿 K 维分段流水（Ampere：`cp.async` 把 A/B 片段异步搬进 shared memory，`mma.sync` tensor core 计算，多级 software pipeline），最后 epilogue 把累加器写回 global memory。

**tile 是 GPU 上兼顾计算效率与调度粒度的天然单位**——通算融合的一切细粒度依赖都对齐到 tile。

### 1.2 Grouped GEMM

MoE 的 per-expert GEMM 形状各异（每个专家分到的行数 `M_e = splits[e]` 由路由决定，运行时才知道）。Grouped GEMM 用**一次 kernel 发射**算完全部 E 个 problem：

- host（或 device）侧准备数组：`problem_sizes[e]`、`ptr_A/B/C/D[e]`、`lda/ldb/ldd[e]`。
- kernel 内有一个 **ProblemVisitor / tile scheduler**：把「(problem, tile)」的二维空间线性化，threadblock 按全局 tile 序号领任务（`while (next_tile())` 持久循环），跨 problem 连续工作，不留发射间隙。
- **调度表可以完全由 device 侧生成**：一个小 kernel 读取 `splits`，就地算出所有 problem 尺寸、指针、tile 总数、以及**自定义的 tile 执行顺序**。这样 host 不需要知道每个专家多少行（避免 device→host 同步）。

---

## 2. GEMM 的两个"侵入点"

要在不破坏 GEMM 本体效率的前提下融合通信，可动的手术位置只有两处：

1. **prologue / mainloop 之前**：threadblock 领到 tile 后、加载数据前，插入"等待依赖数据到达"的自旋（layer0 用）。
2. **epilogue / 写回之后**：tile 写回本地后，做**计数/置 flag**通知消费者（layer1 用）；或直接把结果写到别处（间接寻址 scatter）。

### 2.1 两个访存改造

- **gather_A**：A 操作数迭代器带一个行索引数组，读的是"全量输入缓冲里散落的行"而非连续行（token 按 expert 重排后源行不连续）。
- **scatter_D**：输出迭代器带行索引数组，写到重排后的目标行。

CUTLASS 2.x 的 `DefaultGemm` 模板原生支持 `GatherA` / `ScatterD` 布尔开关（`PredicatedTileIteratorGather` 等组件）。**gather/scatter 只是给迭代器的行偏移查一张 index 表，不改 GEMM 主循环**。

### 2.2 epilogue 与数值

- 累加恒用 **fp32**，写回时转 bf16/fp16/fp8。
- epilogue 可挂逐行缩放（per-row scale）——layer1 用它承载 **topk gate 权重**（`output_vec_scale`，每个 token-assignment 行乘自己的权重，随后求和即完成 combine 的加权）。

---

## 3. FP8 特定要点

### 3.1 量化方式

本项目使用 FP8 w8a8 per-block 量化，block_shape=[128,128]：

- 权重 W：per-(block_n, block_k) tile 缩放因子
- 激活 A：per-block_k 列缩放因子（per_act_token_quant=False）
- block matmul：`out[m,n] += sum_k(A_q[m,k] * W_q[n,k] * a_s[m,k_blk] * w_s[n_blk,k_blk])`

### 3.2 精度注意

- 累加必须用 fp32，否则大 K 下精度崩。
- 参考实现需要携带**相同的量化误差**才能与 kernel 在紧容差内可比——即参考实现也要对激活做 block 量化，再做逐 tile 反量化的 matmul。
- 容差：FP8 atol=3.5e-2, rtol=3.5e-2（moe_bench 默认）。

### 3.3 在融合路径中的位置

- layer0 的输入 `hidden_states` 原始为 bf16，需要在 GEMM prologue 或之前做 block 量化到 FP8。
- layer0 和 layer1 之间的激活（SiLU-mul 输出）也需要量化到 FP8 再喂给 layer1 的 GEMM。
- 权重预量化为 FP8 + scales，在 setup 阶段完成。

---

## 4. sm120 上 Grouped GEMM 的构建路线

### 4.1 两条候选路线

1. **CUTLASS sm120 CuTe collective**（若 grouped/array 变体可用）——主打 block-scaled 窄精度，可能原生支持 FP8 grouped GEMM。
2. **CUTLASS 2.x 风格模板以 sm89 兼容路径编译到 sm120**——消费 Blackwell 与 Ada 的 warp-MMA 血统相近，PTX JIT 可运行，但吃不到新 tensor core 峰值。

按可编译性/性能择优，需实验验证。

### 4.2 必须确认支持的能力

无论哪条路线，都要确认支持/可加装：

- **GatherA / ScatterD 行级间接寻址**
- **grouped ProblemVisitor 或等价 tile 调度器**
- **epilogue 尾部插入自定义回调**（计数/置 flag）
- **mainloop 前插入等待段**（自旋轮询 flag）

### 4.3 tile 形状与 stage 数

sm120 的 shared memory / 寄存器 / L2 与 A100 不同，源环境的 128×128×32×3stage 未必最优甚至未必放得下。需按 smem 预算过滤 + profile。

### 4.4 硬件常数运行时化

SM 数、reducer 线程组织（768 worker / 128 线程组 / 1024 列 tile）、RS_BLOCKS——全部改成查询+配置，不写死。

---

## 5. 从头实现的组件清单与顺序

按依赖顺序，每步都可独立验证：

1. **控制面**：torch.distributed 进程组桥接；对称缓冲分配（NVSHMEM UID 或 cudaIPC）+ stream 上的 group barrier。验证：跨卡写入互见。
2. **非融合 grouped GEMM**（先 bf16，含 gather_A/scatter_D）：先不做任何通信，输入本地造，和 torch 参考比对。这一步同时验证 sm120 上 grouped GEMM 的可用性与性能基线。然后切换到 FP8。
3. **路由元数据 kernel**：逆映射、(expert,rank) 排序、cumsum 表、device 侧 workspace/problem 生成。纯功能，好测。
4. **AllGather + 点灯**（ring push，PCIe 版）：先独立测（灯全亮后数据正确），再接 GEMM 的 per-tile 等待 → **layer0 融合完成**。
5. **layer1 GEMM 的 N-split + 分层计数**：flag 时序可单测（记录每片点灯时间）。
6. **Reducer kernel**：先单卡版（gather+加权求和，无 ring），对拍；再加 ring RS（2 卡→4 卡）；最后双流水平融合 + sm_margin。
7. **调优表 + moe_bench 接入**：与 serial baseline 对比正确性和性能。

**风险最高、应最早验证的三件事**：
- sm120 的 grouped GEMM 模板可用性（第 2 步）
- PCIe P2P/IPC 可用性（第 1 步）
- PCIe 上 ring RS 的实际带宽（第 6 步）

---

## 6. 调优体系（最小可用版）

1. 定义 hparams = {tile 形状 BLK_M/N/K、stage 数、layer1 的 n_split、RS_BLOCKS、sm_margin}。
2. 手工预设 2~4 组候选（如 128×128×32 / 128×64×32 / 64×128×32，stages 3~4——sm120 的 smem/寄存器预算需实测校准）。
3. 提供 `profiling(输入形状)`：逐候选实跑计时，把 (形状→最优组合) 存成 json/表；运行时查表，不命中用默认。
4. 论文的 adaptive workload assignment 对应的就是把 `RS_BLOCKS / sm_margin / n_split` 也纳入这张表按 M 档位选择。

数值 correctness 基准：与纯 torch 参考（fp32 累加的逐 expert 循环）比对，FP8 容差 `atol=3.5e-2, rtol=3.5e-2`。
