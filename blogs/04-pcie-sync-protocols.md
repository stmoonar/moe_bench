---
title: "通算融合算子（四）：PCIe 上的跨卡同步协议——单写者、选举与水位"
description: "跨卡原子不可用之后怎么办：slot+单调序号的信号原语、本地选举+单写者发布的两阶段协议、chunk 水位，以及内存序链和死锁审计清单。"
pubDate: 2026-07-11
tags: ["GPU", "CUDA", "同步", "内存模型", "通算融合"]
series: "从零写通算融合算子"
order: 4
---

> 通算融合 kernel 的数据面（第三篇）其实很简单——难的全在控制面：
> **A 卡怎么可靠地告诉 B 卡"你要的数据我写完了"**。这是全系列技术含量
> 最高、也最容易写出"偶发错误"的一篇。代码来自
> `kernels/tileoverlap/common/sm120_common.cuh` 的 `pcie_sync` 命名空间。

## 1. 直觉方案为什么不行

最自然的想法：B 卡放一个计数器，A 卡每写完一份数据就
`atomicAdd(counter_on_B, 1)`，B 自旋等计数到位。NVLink 机器上这就是标准
做法。但第二篇的红线 2 说了：**PCIe 上对 peer 显存的原子操作高并发时会
丢增量**——计数器永远到不了目标值，B 卡自旋到天荒地老。更恶劣的是弱压力
下测不出来，只有真实负载才触发。

所以 PCIe 上的同步协议必须重造。替代原则一句话：

> **每个信号槽位只有一个写者，就不需要原子。**

## 2. 原语层：slot + 单调序号

三个函数构成全部原语（真实代码，逐行能读懂）：

```cpp
// 单写者跨卡信号: 把序号 seq 写进 dst 卡的 (row, col) 槽位。
// 前提: 全系统只有一个写者会碰这个槽位 —— 普通 store 就够, 无需原子。
template <typename BAR>
__device__ void signal_slot(const BAR &bar, int dst_dev,
                            int row, int col, int seq) {
    asm volatile("st.release.sys.global.s32 [%0], %1;"
                 :: "l"(&bar[dst_dev][{row, col}]), "r"(seq) : "memory");
}

// 等待: 自旋轮询【本地】槽位, 直到值 >= seq。
// 绝不自旋读远端 —— 每次 poll 一个 PCIe 往返(~2µs)还占带宽。
template <typename BAR>
__device__ void wait_slot(const BAR &bar, int my_dev,
                          int row, int col, int seq) {
    int val;
    do {
        asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
                     : "=r"(val) : "l"(&bar[my_dev][{row, col}]) : "memory");
        if (val < seq) __nanosleep(64);
    } while (val < seq);
}
```

两个设计点比代码本身重要：

**① 序号单调递增，永不复位。**如果槽位存布尔（0/1），每轮迭代后要清零——
清零本身又需要一轮跨卡同步，而且"快卡已进入下一轮、慢卡还在读上一轮"的
串扰是多卡程序**最常见的偶发翻车源**。改成单调序号后：第 k 轮等 `>= k`，
旧值天然无害，复位问题直接消失。

**② 写者数量必须编译期/启动期精确可数。**"单写者"不是自然形成的，是
**设计**出来的：给每个 (信号点 × 生产者) 分配独占槽位，消费者逐槽等待。
生产者有几个、各发几次信号，必须能写成闭式表达式——写不出来说明协议
本身没想清楚。

全卡 barrier 也用同一套原语拼出来（每卡在其它每张卡上有独占槽位）：

```cpp
// 每卡把 seq 写到所有卡上属于自己的槽位, 然后等本地槽位集齐所有卡的 seq
if (threadIdx.x < N) {
    signal_slot(bar, threadIdx.x, /*row=*/1, /*col=*/my_dev, seq);  // 广播到达
    while (local_load(bar[my_dev][{1, threadIdx.x}]) < seq) __nanosleep(64);
}
```

实测这个 barrier 9~11µs，而 NCCL 的最小集合操作要 24µs——自研协议的
延迟优势就在这。

## 3. 协议层：本地选举 + 单写者发布

真实场景马上遇到问题：一份"完成"往往由**很多线程共同构成**。比如
"我给 3 号卡的 512 行都推完了"——512 个线程各推一行，谁来发信号？

答案是两阶段：**本地原子选举（本卡原子是可靠的）→ 当选者做单写者发布**。
这是我们 layer1 combine 的真实代码：

```cpp
// 每个线程推完自己那行之后:
tma::store_async(G.staging[dst], row, {dst_row, 0});
tma::store_async_wait();                       // ① 我的跨卡写已提交

int old;
asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"   // ② 本地计数
             : "=r"(old) : "l"(&G.local_cnt[dst]) : "memory");

if (old + 1 == G.expected[dst]) {              // ③ 我是最后一个 → 当选
    __threadfence_system();                    // ④ 系统级 fence
    pcie_sync::signal_slot(G.barrier, dst, 2 + my_dev, 0, G.seq);  // ⑤ 发布
}
```

注意 ①→⑤ 的**内存序链**，一环不能少：

```text
各线程的跨卡数据写
  → (store_async_wait: 本线程的写已提交)
  → atom.acq_rel 本地计数     (release: 把"我写完了"挂到计数器上)
  → 当选者的 acq_rel          (acquire: 接住所有前序线程的写)
  → __threadfence_system      (提升到系统作用域)
  → st.release.sys 信号        (信号可见 ⇒ 数据必可见)
  → 消费者 ld.acquire.sys      (接住一切)
```

这条链在我们机器上经受了每轮上万次信号 × 几十轮迭代 × 多种负载的检验。
但要诚实：**等待侧若用 relaxed load（更快），与 TMA 异步代理的可见性在
严格内存模型下是灰色地带**——如果你的 kernel 出现低概率数据错配，第一
嫌疑人就是内存序，先把等待升级成 acquire 再排查别的。

## 4. 粒度层：信号要粗于 tile——chunk 水位

给每个 tile 发一个信号太贵（信号数 ∝ tile 数，等待方要扫的槽位也 ∝ tile 数）。
经验规则：**信号粒度跟着消费方的最小放行单位走，通常粗于计算 tile**。

我们的做法叫 **chunk 水位**：把每个生产者的输出流按 64 行切 chunk，
每个 (消费者, chunk) 一个槽位；chunk 内 64 行全部提交后，选举出的线程把
序号写进对应槽位。消费者按到达顺序逐 chunk 放行——生产推进到哪，水位
涨到哪。信号量从"每行一个"降到"每 64 行一个"，等待开销可忽略。

## 5. 计数器类的本地信号：gate 的另一半

上面是跨卡方向。还有一类更简单的：数据已经**拉到本卡**了，只需通知本卡的
GEMM。本地原子是可靠的，直接用：

```cpp
// 通信 block: 把 token 写进本地 gathered 缓冲后
asm volatile("red.release.gpu.global.add.s32 [%0], %1;"       // gpu scope 就够
             :: "l"(&counter[slot / ROW_BLOCK]), "r"(1) : "memory");

// GEMM 的 gate: 自旋到这 128 行全部到位
while (relaxed_load(&counter[row_block]) != ROW_BLOCK) __nanosleep(32);
```

细节：**scope 能降就降**（`gpu` 比 `sys` 便宜——写和读都在本卡就不需要
系统作用域）；padding 行没人会写，所以计数器要**预置 padding 数量的种子**，
让真实到达恰好把它顶到 ROW_BLOCK。

## 6. 死锁与 barrier 资源审计清单

发信号的代码写对了，还要保证**有人活着去发**。每写一个融合 kernel，
过一遍这张清单：

1. **自旋 block 数 < SM 数**，给做实事的 block 留出口（第三篇的红线）；
2. **等待有供给**：每个 wait 的信号源画出来——它跑在哪个卡的哪种 block 上？
   那种 block 保证常驻吗？我们的 push kernel 里，scatter 块只等**远端**
   水位（由 peer 的常驻 push 块供给）、自己 shard 直读本地——刻意消掉了
   "等自己人"的自依赖环；
3. **通信侧与计算侧的任务遍历序一致**，否则等待方会在没就绪的任务上
   空转，堵住已就绪的；
4. **named barrier 资源盘点**。这是我们用一整轮实验换来的教训：
   `__syncthreads()` 就是硬件 barrier 0 按"全 block 线程数"计数，而 TK 的
   consumer group 在 GEMM 内部用 barrier 0/1 按 256 线程计数。让算完 GEMM
   的 block 转去干别的活时，如果用 `__syncthreads()` 汇合——两种计数并发
   落在同一个 barrier 上，是未定义行为，Blackwell 上直接
   `illegal instruction`。修法：汇合用没被占用的 barrier id
   （`bar.sync 2, 288`）。**persistent block 换角色前，先盘点两侧用了哪些
   barrier id。**

## 7. 协议正确性怎么验

一句话纪律：**靠隔离裁决，不靠"跑跑看结果对不对"**。

- 给每个协议做独立的 debug 入口（只跑 dispatch、只跑 combine），
  30 次迭代 × 多种负载分布，逐字节对账中间缓冲；
- 期望值表（谁发几次信号）在 host 上用朴素循环重算一遍，与 kernel 用的
  表逐元素对比；
- 端到端对拍放在最后——它只能告诉你"错了"，不能告诉你"哪错了"。

地基到此完备。下一篇把这四篇的所有零件组装成两个真实的 MoE 融合 kernel。
