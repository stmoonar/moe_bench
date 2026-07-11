---
title: "通算融合算子（三）：ThunderKittens 多卡编程与可融合的 GEMM 模板"
description: "tile 抽象、TMA 异步拷贝、跨卡 pgl 视图，以及一个 130 行的 persistent grouped GEMM——它留了两个钩子，通信就从这两个钩子里长进去。"
pubDate: 2026-07-11
tags: ["GPU", "CUDA", "ThunderKittens", "GEMM", "通算融合"]
series: "从零写通算融合算子"
order: 3
---

> 前两篇讲清了"为什么"和"平台约束"。这一篇搭地基：用 ThunderKittens（TK）
> 写一个高性能 grouped GEMM，并且在正确的位置留出**两个钩子**——后面所有
> 融合都只是往钩子里填代码。本篇代码来自 `kernels/tk/sm120_common.cuh`。

## 1. TK 十分钟：你需要知道的全部抽象

ThunderKittens 是斯坦福出的 CUDA 头文件库，核心思想：**以 tile（16×16 的
倍数的小矩阵）为单位组织一切**。你需要的概念只有四个：

```cpp
// ① 数据类型: 寄存器 tile / 共享内存 tile / 全局内存布局
rt_bf<16, 128>            // register tile: 一个 warp 持有的 16×128 bf16
st_bf<128, 64>            // shared tile: smem 里的 128×64
sv_bf<4096>               // shared vector: smem 里的一行
gl<bf16, 1, 1, -1, 4096>  // global layout: 全局显存的 (?, 4096) 张量视图

// ② warp/group 级操作(集体操作, 一个 warp 或一组 warp 一起干)
warp::load(reg, smem_subtile);       // smem -> 寄存器
warp::mma_AB(acc, a, b, acc);        // tensor core 乘加
group<8>::store(gl, acc, {r, c});    // 8 个 warp 合写一个 128×128 输出块

// ③ TMA: 硬件异步拷贝引擎(不占用计算线程)
tma::load_async(smem_tile, gl, {row, col}, semaphore);  // 发起后立即返回
tma::store_async(gl, smem_tile, {row, col});
tma::store_async_wait();             // 等本线程发起的 store 全部落地

// ④ mbarrier 信号量: 和 TMA 配套的完成通知
init_semaphore(sem, 0, 1);
tma::expect_bytes(sem, nbytes);      // 声明"这次到达要等多少字节"
wait(sem, phase);                    // 等 TMA 完成(phase 0/1 交替)
```

TMA + mbarrier 是整个高性能 kernel 的心脏：**数据搬运交给硬件引擎，
计算线程只负责"发起"和"等待"**，两者天然重叠。

## 2. 多卡：pgl 把 peer 显存变成一个数组下标

多卡的地基是 CUDA IPC：每个进程（一卡一进程）把自己的显存 buffer 映射进
所有进程的地址空间。TK 把这包装成两层：

```python
# Python 侧: TKParallelTensor —— 分配一块所有卡都能看到的对称 buffer
pre_tokens = TKParallelTensor((512, 4096), dtype=torch.bfloat16,
                              local_rank=rank, local_world_size=4,
                              multicast=False)   # PCIe 平台永远 False(红线 1)
```

```cpp
// device 侧: pgl (parallel global layout) —— 所有卡副本的统一视图
using pre_tokens_pgl = pgl<gl<bf16, 1, 1, -1, 4096>, 4, false>;
//                                                   ^卡数  ^multicast 关

G.pre_tokens[2]           // 第 2 张卡上的副本, 当成普通 gl 用
tma::load_async(smem, G.pre_tokens[peer], {row, 0}, sem);  // 直接读远端!
```

`G.pre_tokens[peer]` 的读写就是 PCIe P2P 事务——**跨卡通信在代码上和本地
访存长得一模一样**，区别只在带宽和延迟（第二篇的表）。这就是"通信融进
kernel"在工程上可行的原因：它不需要任何库调用，就是几行访存。

## 3. persistent kernel + block specialization：融合的骨架

普通 kernel 一个 block 算一块就退出；**persistent kernel** 启动恰好
"SM 数量"个 block，每个 block 循环领取任务直到做完。好处：block 常驻，
可以自旋等信号、可以扮演不同角色：

```cpp
__global__ void kernel(const globals G) {
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, gate, blockIdx.x, G.num_comp_sms);  // 计算角色
    else
        dispatch(G, blockIdx.x - G.num_comp_sms);                 // 通信角色
}
// 发射: kernel<<<110 /*SM数*/, NUM_THREADS, smem>>>(G);
```

一条铁律（死锁红线）：**会无限自旋的 block 数量必须小于 SM 数**，永远给
非自旋 block 留出口。计算 block 自旋等数据，如果它们占满了所有 SM，
负责送数据的 block 永远上不了车——整卡挂死。

## 4. GEMM 模板走读：128 行，两个钩子

下面是我们 SM120 grouped GEMM 的骨架（`grouped_gemm_sm120`），去掉细节后
的完整结构。**grouped** 的意思是：输入行按 expert 分组，每组乘不同的权重
矩阵——这正是 MoE 需要的形状。

```cpp
struct gemm_config {
    static constexpr int ROW_BLOCK = 128;   // 每个任务块 128 行(也是 expert 的对齐单位)
    static constexpr int COL_BLOCK = 128;   // 输出列块
    static constexpr int RED_BLOCK = 64;    // K 维每级流水的深度
    static constexpr int PIPELINE_STAGES = 3;  // 3 × 32KB = 96KB ≤ 99KB smem
    static constexpr int CONSUMER_WARPS = 8;   // 8 个计算 warp, 各管 16 行
    static constexpr int NUM_THREADS = 9 * 32; // +1 个 producer warp = 288 线程
};

template <typename Globals, typename Gate, typename Epilogue>
__device__ void grouped_gemm_sm120(const Globals &G, const Gate &gate,
                                   const Epilogue &epilogue,
                                   int sm_idx, int num_sms) {
    // smem: 3 级流水缓冲(A tile + B tile) × 2 组 mbarrier
    ...
    if (warp_id == CONSUMER_WARPS) {
        // ---------- producer warp: 只负责发起 TMA 加载 ----------
        for (每个 expert e) for (task_id = sm_idx; ...; task_id += num_sms) {
            int row_idx = ..., col_idx = ...;

            gate(row_idx);   // ★钩子 1: 加载这个行块前, 先等它的数据到齐

            for (int red = 0; red < K / RED_BLOCK; red++) {
                wait(inputs_finished[stage], phase);        // 等流水位空闲
                tma::expect_bytes(inputs_arrived[stage], ...);
                tma::load_async(A_tile, G.activations, {row_idx, red}, ...);
                tma::load_async(B_tile, G.weights, {e, red, col_idx}, ...);
            }
        }
    } else {
        // ---------- consumer warps: 只负责 tensor core 计算 ----------
        for (同样的任务遍历) {
            rt_fl<16, COL_BLOCK> acc;  warp::zero(acc);
            for (int red = 0; red < K / RED_BLOCK; red++) {
                wait(inputs_arrived[stage], phase);          // 等数据到 smem
                warp::mma_AB(acc, a_reg, b_reg, acc);        // 算
                warp::arrive(inputs_finished[stage]);        // 归还流水位
            }
            consumers::store(G.outputs, acc, {row_idx, col_idx});
            consumers::sync(0);
            epilogue(row_idx, col_idx);  // ★钩子 2: 这块输出已落地, 通知下游
        }
    }
}
```

几个值得停下来看的设计：

**① producer/consumer 分工。**一个 warp 专职发 TMA（几乎不耗算力），
八个 warp 专职 mma。三级 smem 流水让"搬第 N+1 块"和"算第 N 块"重叠——
这是 kernel **内部**的"通算重叠"，结构上和跨卡重叠同构。

**② `gate(row_idx)`：融合的入口。**纯计算时传空实现；融合时传一个自旋
等待。整个 GEMM 对"数据从哪来"一无所知——通信侧只要保证"计数器到位"，
GEMM 就正确。第五篇的两个融合 kernel 都基于这一个模板，**GEMM 本体零改动**。

```cpp
struct dispatch_gate {   // 融合版 gate: 等这 128 行的到达计数打满
    __device__ void operator()(int row_idx) const {
        int v;
        do { v = relaxed_load(&counter[row_idx]); __nanosleep(32); }
        while (v != ROW_BLOCK);
    }
};
```

**③ `epilogue(row, col)`：向下游融合的出口。**layer1 用它在每块输出落地时
给等待方发信号（做法见第四篇的"选举"）。

**④ 一个真实的坑：`group::store` 的行映射。**8 个 warp 合写 128×128 输出时，
TK 内部会做 warpgroup 交织置换——warp w 实际写的是第
`w/4 + (w%4)*2` 个 16 行条带，不是第 w 个。计算时加载 A 就必须加载**同一个
置换后的条带**，否则除 warp0 外全部错位。这是我们整个项目最深的坑之一，
症状是"结果几乎对但少数行错"。用不同 warp 数改模板时必须重推这个映射。

## 5. 编译与工程细节

- **卡数是编译期常量**（`pgl<..., NUM_DEVICES, ...>`），每个 world size
  编译一个 .so，按 `(world, hidden, row_block)` 缓存；
- smem 只有 99KB（消费级 Blackwell），H100 的 4 级 227KB 流水塞不下，
  所以是 3 级 × 32KB——**tile/流水配置永远跟着 smem 预算走**；
- SM120 没有 H100 的 wgmma/setmaxnreg，计算主体是 warp 级
  `mma.sync m16n8k16`，寄存器全 warp 均分。

到此为止，我们有了：能读写 peer 显存的 pgl、能自旋等信号的 persistent
block、留好两个钩子的 GEMM。还缺最后一块地基，也是最容易翻车的一块——
**怎么在 PCIe 上正确地发信号**。下一篇专讲同步协议。
