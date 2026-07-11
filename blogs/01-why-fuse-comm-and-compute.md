---
title: "通算融合算子（一）：为什么要把通信写进 GEMM kernel 里"
description: "从 MoE 层的串行瓶颈讲起：什么是通信计算重叠、收益天花板怎么算、四种融合模式的地图，以及为什么我们不用多 stream 方案。"
pubDate: 2026-07-11
tags: ["GPU", "CUDA", "MoE", "通算融合", "分布式"]
series: "从零写通算融合算子"
order: 1
---

> 本系列写给**会一点 GPU 编程**的读者：你知道 kernel/block/thread，写过简单的
> CUDA kernel，但没碰过多卡通信和高性能 GEMM。读完六篇，你能看懂并动手写出
> "把跨卡通信融合进 GEMM kernel"的算子。系列的所有代码和数字都来自一个真实
> 项目：在 4×RTX PRO 5000（PCIe 互连）上实现的 MoE 通算融合层，
> 最终比 vLLM 的串行基线快 1.16~1.20×。

## 1. 问题：一个 MoE 层的时间都花在哪

MoE（Mixture of Experts）层的分布式执行是最典型的"通信夹着计算"结构。
以 TP（张量并行）模式为例，4 张卡，每张卡拿到 512 个 token，held 全部 64 个
expert 的 1/4 权重切片：

```text
串行执行（vLLM baseline 的真实数据流）:

AllGather ──→ expert GEMM (gate/up) ──→ SiLU ──→ GEMM (down) ──→ ReduceScatter
  ~0.4ms          ~1.0ms                 0.1ms      ~0.5ms           ~0.4ms
[通信]           [计算]                 [计算]      [计算]           [通信]
```

三个阶段互相**串行等待**：GEMM 必须等 AllGather 拿齐所有卡的 token 才能开始，
ReduceScatter 必须等 GEMM 全部算完才能开始。在我们的机器上这套串行流程跑
2625µs——其中约 25% 的时间 GPU 的计算单元在干等通信。

**通算融合（communication-computation fusion）**要做的事：让通信和计算在
时间上重叠。理想情况下总时间从 `T_comm + T_comp` 变成 `max(T_comm, T_comp)`。

## 2. 先算账：收益天花板在立项前就能算出来

这是整个领域最重要的一条纪律：**重叠的收益上限 = min(T_comp, T_comm)，
且实际能兑现多少由通信占比决定**。动手前先把账算清：

```text
完美重叠后的时间  =  max(T_comp, T_comm)
收益率           =  (T_comp + T_comm) / max(T_comp, T_comm)
```

两个真实例子（同一台机器、同一个 MoE 形状）：

| 并行模式 | 通信占比 | 完美重叠上限 | 我们实测 |
|---|---|---|---|
| EP（专家并行，dispatch 按路由发 token） | ~40%+ | ~1.6× | 1.34× |
| TP（张量并行，稠密 AllGather/ReduceScatter） | ~25% | ~1.35× | 1.16× |

TP 的天花板**结构性地**低于 EP——不是实现不够好，是通信占比摆在那里。
如果你的场景通信占比 <10%，收益上限只有 ~1.05×，这活不值得干。
**先测通信占比，再决定要不要写融合算子。**

## 3. 两条技术路线，我们选难的那条

### 路线 A：多 stream 重叠（我们不用）

把 GEMM 拆成几个 chunk，通信放另一个 stream，用 event 编排依赖。侵入小，
但有个致命问题：**GEMM 拆小后每块的计算效率下降**（tile 利用率、L2 命中、
wave 数都变差），常常"重叠赚的不够拆分亏的"。

### 路线 B：kernel 内融合（本系列的主题）

GEMM kernel 保持完整不拆，把通信"贴"进去。核心观察是：一个高性能 GEMM
kernel 内部本来就有等待点——加载数据的 warp 在等数据、写出结果的 warp
在等计算完成。融合就是把这些等待点接到**跨卡数据的就绪信号**上：

```text
融合 kernel 的样子（伪代码）:

if (blockIdx.x < num_comp_blocks) {
    // 计算角色: 完整的 GEMM, 只多一行——
    gate(row_block);        // <-- 自旋等待"这块数据到齐了"的信号
    ... 标准 GEMM 流水线 ...
    epilogue(tile);         // <-- 算完一块立刻通知需要它的人
} else {
    // 通信角色: 搬数据 + 发信号
    pull_or_push(data); signal(counter);
}
```

一次 kernel launch，一部分 block 算，一部分 block 搬运。数据到一块、算一块；
算完一块、发走一块。GEMM 的 tile 尺寸、流水线深度全部保持最优。

## 4. 四种数据流向：融合模式的地图

所有通算融合算子都落在四个模式里，按"谁等谁"分类：

| 模式 | 方向 | 典型算子 | 融合点 |
|---|---|---|---|
| **AG + GEMM** | 消费者等上游 | TP 的 AllGather→GEMM | GEMM **prologue**：等输入分片到达 |
| **GEMM + RS** | 生产者推下游 | GEMM→ReduceScatter | GEMM **epilogue**：算完就推给目标卡 |
| **dispatch + GEMM** | 带路由的 AG | MoE layer0 | prologue + 按路由重排 |
| **GEMM + combine** | 带归约的 RS | MoE layer1 | epilogue + top-k 加权归约 |

记住这张图。后面第五篇写的两个实战 kernel，正是最后两行。

## 5. 三个立刻要建立的心智模型

**① 依赖粒度决定重叠上限。**"等整个 tensor"改成"等一个 128 行的块"，
是所有融合方案的起点。但粒度的**方向**必须对：如果你的 GEMM 按 expert 顺序
消费数据，通信就要按 expert 顺序送达——送达序对齐消费序。这个方向错了，
细粒度信号一点用都没有（我们在这上面翻过车，第六篇有完整复盘）。

**② 通信不是免费的并发。**通信占用 SM（除非走 copy engine），占多了 GEMM
变慢。在我们的 PCIe 机器上，16 个通信 SM 会让 GEMM 慢 8~16%。通信 SM 数量
是要扫参的（我们的最优值是 24/110——而且这个值只有实测才知道）。

**③ wave 是判断有没有重叠空间的第一指标。**GEMM 的任务块数不足一个
wave（一次能同时驻留的 block 数）时，基本没有重叠空间——所有数据必须在
开跑前就绪。token 数太小时融合不划算，直接回退串行。

## 6. 系列大纲

1. **为什么要通算融合**（本篇）：问题、账、模式地图
2. **认识你的平台**：PCIe 拓扑、强弱路径、跨卡原子红线、microbench 方法
3. **ThunderKittens 多卡编程**：tile 抽象、TMA、可挂钩子的 persistent GEMM 模板
4. **PCIe 上的同步协议**：单写者 slot、选举、水位——最容易翻车的部分
5. **实战：MoE 融合层**：两个 kernel 的完整走读（调度表 + layer0 + layer1）
6. **调优、归因与踩坑**：从慢 38% 到快 20% 的完整迭代史

> 系列基于的真实代码在 `moe_bench` 仓库（kernel：`kernels/tk/`，调度：
> `tk_tp_scheme.py`），实验平台为 4×RTX PRO 5000 Blackwell（SM120，PCIe Gen5）。
