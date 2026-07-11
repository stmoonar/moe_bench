---
title: "通算融合算子（二）：先认识你的平台，再写一行代码"
description: "PCIe 拓扑怎么读、push 为什么比 pull 快一倍、跨卡原子为什么不能用——用 microbench 把平台事实钉死，方案选型就不会靠猜。"
pubDate: 2026-07-11
tags: ["GPU", "PCIe", "microbenchmark", "通算融合"]
series: "从零写通算融合算子"
order: 2
---

> 上一篇算了收益账。这一篇解决"方案在**这台机器**上怎么选"——同一个算法在
> NVLink 和 PCIe 机器上的最优形态完全不同。所有数字来自我们在
> 4×RTX PRO 5000（PCIe Gen5）上的实测。

## 1. 第一件事：`nvidia-smi topo -m`

```text
        GPU9   GPU11  GPU13  GPU15
GPU9     X     PIX    NODE   NODE
GPU11    PIX    X     NODE   NODE
GPU13    NODE  NODE    X     PIX
GPU15    NODE  NODE   PIX     X
```

读法：`PIX` = 同一个 PCIe switch 下（带宽最好），`NODE` = 过 CPU 的
PCIe host bridge（共享上行带宽），`SYS` = 跨 CPU socket（最差，甚至可能
不支持 P2P）。我们的 16 卡机上，相邻编号的卡成对挂在同一 switch 下，
所以测试固定用 9,11,13,15 这样的组合（两对 PIX）。

拓扑决定两件事：**选哪几张卡**，以及**通信的时序要不要错开**（同 switch
的对带宽独立，跨 host bridge 的流量会互相挤）。

## 2. 三条红线：PCIe 平台上什么东西直接不能用

写多卡 kernel 前必须知道的三条：

**红线 1：没有 NVSwitch → 没有 multimem/组播。**H100 NVL 机器上"一条指令
写 8 张卡"、"交换机在网归约"这类魔法（`multimem.ld_reduce/st/red`）在 PCIe
上全部不存在。所有跨卡数据面退化为**逐卡 unicast**。

**红线 2：跨卡原子不可靠。**对 peer 显存做 `atomicAdd`/`red.add`，在 PCIe
平台上（`p2pNativeAtomicSupported == 0`）**高并发时会丢增量**——而且弱压力
的探针测不出来，只有真实负载的散射写才会触发。我们在这上面损失过整整一个
优化方向（EP 的 push dispatch v1 因此冻结）。结论：**跨卡完成通知永远不用
远端原子**，用什么替代见第四篇。

**红线 3：P2P 本身要先确认。**`p2pBandwidthLatencyTest` 先跑一遍；ACS/IOMMU
配置不对时 P2P 会被强制绕行甚至禁用，虚拟化环境是重灾区。P2P 不可用的话，
本系列的整条技术路线（IPC 映射 peer 显存）都不成立。

## 3. 强弱路径：push 和 pull 差一倍，这决定数据面方向

PCIe 上"我写你的显存"（push）和"我读你的显存"（pull）走的硬件路径不对称。
我们的 microbench 实测（这张表值一台机器的钱）：

| 数据面 | 带宽 | 需要的 SM 数 | 4 卡并发行为 |
|---|---|---|---|
| SM push（写远端） | **50.9 GB/s** | **4 个打满** | 零退化（49.8/卡） |
| SM pull（读远端，同 switch 对） | 50 GB/s | 16 个 | — |
| SM pull（跨对） | 29 GB/s | 16 个 | **退化到 23.5/卡** |
| copy engine（`cudaMemcpyPeer`） | 54~56 GB/s | **0 个** | 未测并发 |
| NCCL AllGather/ReduceScatter | busbw ~30 GB/s | ~16+ | — |

三个结论：

1. **push 是强路径**：带宽高、省 SM、并发不退化。数据面设计优先考虑
   "生产者推"而不是"消费者拉"。
2. 但**这条规则只在 comm-bound 时重要**。我们后来在 TP 场景实测：AG 只有
   12.6MB，完全藏在 GEMM 下面，pull 和 push 的差异根本不在关键路径上——
   push 版反而因为多一跳本地搬运更慢（第六篇有完整故事）。
   **先判 bound 类型，再选数据面。**
3. 8KB 的行粒度散布读写和顺序读写**没有测出差别**——行粒度不是瓶颈，
   （送达的）方向才是。

## 4. 干扰系数：通信和计算抢什么

融合 kernel 里通信 block 和计算 block 同时跑，它们互相拖累吗？实测：

- **带宽干扰 ≈ 0**：`T(同时跑) ≈ max(T_comm, T_comp)`，PCIe 流量和 HBM
  流量基本不打架。编排问题纯粹是**SM 怎么分**的问题。
- **SM 让渡代价**：把 16 个 SM 划给通信，GEMM 慢 8~16%；只划 4 个则只慢 ~4%。
  这就是为什么"通信要用几个 SM"必须扫参——它是拿 GEMM 时间换通信时间。

## 5. microbench 方法论：每一级方案的验收三件套

任何融合方案，测这三个数，缺一不可：

```text
T_comp_alone   计算单独跑（GEMM-only，同样的 SM 数）
T_comm_alone   通信单独跑
T_total        融合跑

overlap 效率 = (T_comp + T_comm - T_total) / min(T_comp, T_comm)   目标 ≥ 0.8
计算干扰    = T_comp@融合 / T_comp_alone                            目标 ≤ 1.1
```

我们把这套方法做成了常驻工具（`tools/time_tp_stages.py`）：每一层融合 kernel
旁边都放一个"纯 GEMM 对照"，`融合耗时 − GEMM-alone = 暴露的通信/协议成本`，
每轮实验自动输出。第六篇会展示它怎么把优化方向一个个钉出来。

## 6. 平台探测清单（拿去照抄）

上新机器时按顺序跑：

1. `nvidia-smi topo -m` → 选卡组、记下哪些对共享带宽；
2. `p2pBandwidthLatencyTest` → P2P 可用性 + 各对带宽矩阵；
3. 自写 peer-atomic 探针（**高并发散射写**，弱压力测不出丢增量）；
4. push/pull/copy-engine 三种数据面的带宽-SM 数曲线；
5. NCCL 小包延迟 vs 自研信号 RTT（我们测得：NCCL 小包 24µs，
   跨卡 store 信号 RTT 2~3µs——这就是自研同步协议的收益空间）；
6. 通信+计算并发的干扰系数。

这些数字构成你做一切设计决策的"平台事实表"。**动手写 kernel 之前，
先把表填满**——我们项目里这张表推翻过两次"理所当然"的设计直觉。

下一篇进入代码：用 ThunderKittens 搭出一个能挂通信钩子的 GEMM 模板。
