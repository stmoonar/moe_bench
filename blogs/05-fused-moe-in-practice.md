---
title: "通算融合算子（五）：实战——两个 kernel 写完一个 TP MoE 融合层"
description: "调度表怎么设计（一张表干两层的活）、AllGather⊕GEMM 与 GEMM⊕combine 的完整代码走读、全员收尾的 dispenser，以及三道正确性锚点。"
pubDate: 2026-07-11
tags: ["GPU", "CUDA", "MoE", "ThunderKittens", "通算融合"]
series: "从零写通算融合算子"
order: 5
---

> 组装时刻。本篇按真实代码走读一个完整的 TP MoE 融合层：
> `kernels/tk/tk_moe.cu` 的 `tpdisp` / `preredpush` 两个命名空间 +
> `tk_tp_scheme.py` 的调度表。读完你应该能自己写出第三个。

## 1. 问题设定

4 卡 TP：每卡持有**全部 64 个 expert** 的 intermediate 维 1/4 切片
（gate/up 每 expert 4096→1536，down 768→4096），每卡 512 个 token，
每 token 路由到 top-8 个 expert。要算的东西：

```text
layer0:  y = [silu(x@W_gate) * (x@W_up)]     x 来自所有 4 张卡 → 需要 AllGather
layer1:  out[t] = Σ_k w(t,k) · (y_{t,k} @ W_down)   4 卡各算部分和 → 需要 ReduceScatter
```

融合形态（每层一个 kernel）：

```text
kernel 1:  AllGather ⊕ (gate+up GEMM)          —— 消费者等上游模式
kernel 2:  (down GEMM) ⊕ top-k 归约 ⊕ 推回源卡 —— 生产者推下游模式
```

## 2. 灵魂在 host 侧：调度表

融合 kernel 的复杂度大头不在 device 代码，在**预计算的索引表**。核心是
把"路由"这个动态问题变成 GEMM 喜欢的静态布局：

**gathered 布局**：把 4 卡 × 512 token × 8 份 expert 命中，按 expert 排序
铺进一个 16384 行的大缓冲，每个 expert 的行数向上对齐到 128（GEMM 的
ROW_BLOCK）。于是 grouped GEMM 只需要每 expert 的行区间——这就是
`padded_tokens_per_expert` 前缀和。

**一张表干两层的活**——`tp_slots (2048, 8)`：第 j 行 = 全局第 j 个 token，
第 k 列 = 它第 k 个 expert 命中被放进 gathered 的哪一行（槽位）：

- layer0 用它**散播**：token 拉到本地后，复制到它的 8 个槽位；
- layer1 用它**回收**：把这 8 个槽位的 GEMM 输出行加权求和。

两层共享同一张表，正确性天然自洽（EP 模式下这是两张要跨卡对账的表，
TP 因为布局纯本地而免掉了这个麻烦）。

配套的三张顺序表，每张都对应一条已验证的性能教训（第六篇细讲）：

| 表 | 内容 | 为什么需要 |
|---|---|---|
| `pull_order` | token 按其最小槽位排序 | **到达序对齐消费序**：GEMM 按 expert 顺序吃数据，通信就按 expert 顺序送。顺序错了，所有行块都会等到最后 |
| `job_order` | combine job 按最大槽位排序 | **就绪序**：job 依赖它最晚完成的槽位，按就绪序领任务才能边算边推 |
| `slack` | 每 128 行块的 padding 数 | 计数器种子：padding 行没人写，预置后真实到达恰好顶满 |

一条工程纪律：**每张表写两个实现**——host 上的朴素循环版（golden）和
GPU 上的向量化版（单次 argsort + scatter，可被 CUDA graph 捕获、每轮计入
计时以对齐公平口径）。两版逐元素对拍 + 不变量检查（槽位是双射、区间正确、
slack 恒等式）构成第一道正确性锚点。

## 3. kernel 1：AllGather ⊕ gate/up GEMM（`tpdisp`）

一次 launch，110 个 block 分两个角色：

```cpp
__global__ void kernel(const globals G) {
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, dispatch_gate{G}, blockIdx.x, G.num_comp_sms);
    else
        dispatch(G, blockIdx.x - G.num_comp_sms);   // 通信角色
}
```

通信角色：每个线程负责一个**唯一** token——跨卡拉一次，本地散播 8 份
（top-8 的 8 个 expert 都在本卡，所以每个 token 恰好被用 8 次，
"去重拉取"在 TP 是天然形态而不是优化）：

```cpp
__device__ void dispatch(const globals &G, int sm_idx) {
    // smem: 12 个 token 槽(96KB/8KB) + 每槽一个 mbarrier
    const int i = sm_idx * TOKENS_PER_BLOCK + lane_id;
    if (i >= G.s_max) return;
    const int d = G.pull_order[{i}];             // ★ 按 min-slot 序拉取
    const int src_dev = d / G.num_tokens, src_tok = d % G.num_tokens;

    tma::load_async(token[lane], G.pre_tokens[src_dev], {src_tok, 0}, sem);
    wait(sem, 0);                                 // 跨卡拉取(PCIe P2P)
    for (int k = 0; k < 8; k++)                   // 本地散播到 8 个槽位
        tma::store_async(G.activations, token[lane], {G.tp_slots[{d,k}], 0});
    tma::store_async_wait();
    for (int k = 0; k < 8; k++)                   // 给 8 个行块计数器 +1
        red_release_gpu_add(&G.barrier[me][{G.tp_slots[{d,k}] / 128}], 1);
}
```

GEMM 那侧只有一个 gate（上一篇讲过的自旋），等行块计数器 == 128。
时序上的效果：`pull_order` 让 expert 0 的 256 个 token（来自 4 个源，
链路并发）最先到达 → expert 0 的行块最先解锁 → GEMM 紧贴着数据到达的
前沿推进。**AllGather 完全消失在 GEMM 的背影里**（实测暴露仅 ~250µs，
其中约一半是首个 expert 数据的固有头部延迟）。

## 4. kernel 2：down GEMM ⊕ top-k 归约 ⊕ 推回（`preredpush`）

layer1 是生产者推下游模式，三件事融合在一个 kernel：

```text
① down GEMM: 16384 行 × (768 → 4096), 每块输出落地时 epilogue 发本地信号
② prered job: 等某 token 的 8 个槽位行都算完 → FP32 加权求和成一行
③ push: 把这行推到该 token 源卡的 staging 平面 + 选举水位(第四篇的协议)
```

早期版本给每个 job 开一个 block（2048 个短命 block 在 16 个通信 SM 上
排 128 波），慢得离谱。现在是**常驻 dispenser + 全员收尾**：

```cpp
__global__ void gemm_push_kernel_tp(const globals G,
                                    const int *job_order, int *job_next) {
    if (blockIdx.x < G.num_comp_sms) {
        grouped_gemm_sm120(G, no_gate{}, signal_epilogue{G}, ...);
        // 汇合必须用空闲的 named barrier —— __syncthreads 是 bar0,
        // 会和 GEMM 内部 consumer group 的 bar0@256 混计数 → UB(见第四篇)
        asm volatile("bar.sync 2, %0;" :: "n"(NUM_THREADS));
    }
    __shared__ int s_j;
    while (true) {                                  // 原子 dispenser
        if (threadIdx.x == 0) s_j = atomicAdd(job_next, 1);
        __syncthreads();
        if (s_j >= G.num_jobs) break;
        push_job(G, job_order[s_j]);                // ★ 按就绪序领 job
        __syncthreads();
    }
}
```

设计意图拆开看：

- **GEMM 期间**：通信 block 从 dispenser 按 `job_order`（就绪序）领 job，
  最早就绪的 partial 行边算边推——推流藏在 GEMM 下面；
- **GEMM 结束后**：每个算完自己任务的计算 block **立即加入领 job**
  （"全员收尾"）——尾部排空速度从 24 个 SM 变成全部 110 个；
- dispenser 保证每个 job 恰好被领一次，第四篇的单写者/选举协议不被破坏。

`push_job` 内部就是第四篇 §3 展示过的完整链：等槽位行块的本地信号 →
smem 里 FP32 加权求和 → TMA 推到源卡 staging → 本地选举 → 当选者发水位。
最后一个小 kernel（`final_reduce`）在源卡上等齐 4 个水位、把 4 个平面
逐行相加——这就是 ReduceScatter 的"scatter+sum"，只是它在 GEMM 还没
全部结束时就开始收货了。

## 5. Python 侧：把碎片粘起来

scheme 层（`tk_tp_scheme.py`）负责每次迭代的编排，几处关键：

```python
def run(self):
    # ① 调度表 GPU 重建(argsort 链, CUDA graph 回放, ~250µs) —— 计入计时!
    #    serial 基线每轮也要付路由元数据的钱, 不计入就是作弊(公平口径)
    dist.all_gather_into_tensor(self._all_topk, self._topk_ids_local)
    self._sched_graph.replay()

    # ② 保护性 barrier: 覆盖 pre_tokens 前确认没有 peer 还在读上一轮
    self._l0_seq += 1
    tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)
    self.pre_tokens.data_.copy_(self.problem.hidden_states)
    self._l0_seq += 1
    tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)

    tk.moe_tp_dispatch_gemm(...)          # ③ kernel 1
    torch.mul(F.silu(gu[:, :I]), gu[:, I:], out=self.act)   # ④ SiLU(torch 够快)
    self.job_next.zero_()                 # ⑤ dispenser 归零(本地, 同 stream 安全)
    tk.moe_tp_gemm_prered_push(...)       # ⑥ kernel 2
    tk.moe_final_reduce_push(...)         # ⑦ 源卡收尾
    return self.combine_out
```

所有 buffer 在 setup 里一次分配、每轮复用（allocation-stable，CUDA graph
的前提）；所有序号（`_l0_seq/_l1_seq`）单调递增贯穿整个进程生命周期。

## 6. 三道正确性锚点（顺序不能反）

1. **表裁决**：host golden vs GPU builder 逐元素相等 + 不变量
   （无 GPU 也能跑，我们做成了 CPU 预检，几秒钟拦住一类整轮报废的错误）；
2. **CPU 数据流模拟**：用真实的表在 CPU 上模拟 散播→GEMM→归约→推送→合并
   全流程，对拍朴素参考实现（fp32 误差 1e-7 级）——验证**语义**，
   与 CUDA 无关；
3. **端到端对拍**：多种 expert 数 × balanced/skewed 路由分布，bf16 相对
   误差 ~4e-3 为正常水位。

先 1 后 2 后 3：上机后一旦出错，前两道已经排除了"表逻辑错"和"语义错"，
归因范围直接缩小到"协议/环境"。

最后一篇：这套东西从比基线慢 38% 优化到快 20% 的完整过程——每一步的
归因方法比结论更值钱。
