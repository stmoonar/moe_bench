# Microbench 结果汇总：20260710_071848

## 1. 测试目的

这组 microbench 用来拆解 `tkfused` 相对 vLLM serial 的收益来源：TK 计算 kernel、
PCIe 通信、通信与计算的融合/重叠；并测量融合损失、通信 SM 数、inter/intra-SM
编排和 PCIe 搬运方案。

> 全文的 `tokens` 均为 4 个 rank 的总 token 数，每 rank 数量为其 1/4。时间为
> 30 次迭代中位数、跨 rank 取最慢值。

## 2. 核心结论

1. **公平口径下，主工作点达到 1.34~1.35×。** 将默认路径漏计的 graph
   schedule 按 205 us 加回后，2048 总 token（每 rank 512）为
   `2912 / 2181 us = 1.34×`，8192 总 token 为 `10942 / 8116 us = 1.35×`。
2. **收益不是单纯来自 TK GEMM 更快。** 2048 token 的总收益中，计算收益占
   32%，通信方案与重叠占 68%；8192 token 分别占 46% 和 54%。
3. **TK 纯计算链在本次未调优 vLLM 基线上快约 1.21~1.29×（1024 token 以上）。**
   但 vLLM 缺少当前 GPU/shape 的 tuning config，因此需要调优后复测。
4. **主要性能缺口是融合损失。** 8192 token 时 L0/L1 分别比纯 GEMM 多
   966/747 us，合计约 1.71 ms；公平 e2e 距全重叠理论上限 1.42 ms。
5. **PCIe push 明显优于跨 PCIe 对的 pull。** push 用 4 个 block 即达到约
   50.8 GB/s，4 卡并发仍约 49.8 GB/s/卡；pull 需 16 个 block 才饱和，跨对
   只有约 29 GB/s，4 卡并发最慢卡降至约 23.5 GB/s。
6. **inter/intra-SM 没有额外的重叠收益。** `both ≈ max(comm, comp)`，额外干扰
   接近 0；问题本质是用多少 SM/warp 换通信带宽。

## 3. 环境与复现信息

| 项目 | 值 |
|---|---|
| GPU | 4 × RTX PRO 5000 72GB Blackwell，物理卡 `9,11,13,15` |
| 测试前状态 | 目标卡显存占用 0 MiB、利用率 0%；物理卡 2 有负载但未被选用 |
| MoE shape | E=64，topk=8，hidden=4096，intermediate=3072，EP world=4 |
| 精度 | **bf16** |
| token sweep | 128、512、1024、2048、5120、6648、8192（总 token） |
| TK 路径 | pull dispatch + prered_push + fused gate/up + GPU schedule，ROW_BLOCK=128 |
| 软件 | Python 3.12.3，PyTorch 2.10.0+cu128，CUDA runtime 12.8，vLLM 0.1.0 |
| 代码快照 | `913d5e1fc91e32401401c26a9e1168cc14716c25`，运行时 dirty |

dirty 项包括 `ThunderKittens` 子模块修改、`microbench/` 与
`tools/time_serial.py` 未跟踪，因此 commit 哈希不能单独复现本批数据。

## 4. 总体性能与收益分解（MB3）

采用 `mb3_report_sched205us.md` 的公平口径：
`fair tkfused = 原始 tk_e2e + 0.205 ms graph schedule`。原始 `mb3_report.md`
使用约 0.8 ms eager schedule，不代表实际 graph 路径。

| 总 tokens | serial | tkfused fair | 加速 | 计算收益占比 | 通信+重叠占比 | TK+全重叠上限 | 距上限 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1309 us | 1519 us | 0.86× | 64%¹ | 36%¹ | 0.97× | 175 us |
| 512 | 1449 us | 1536 us | 0.94× | 72%¹ | 28%¹ | 1.08× | 194 us |
| 1024 | 1814 us | 1525 us | 1.19× | 41% | 59% | 1.35× | 185 us |
| 2048 | 2912 us | 2181 us | **1.34×** | **32%** | **68%** | **1.58×** | 337 us |
| 5120 | 6924 us | 5121 us | 1.35× | 39% | 61% | 1.60× | 789 us |
| 6648 | 9238 us | 6815 us | 1.36× | 41% | 59% | 1.57× | 922 us |
| 8192 | 10942 us | 8116 us | **1.35×** | **46%** | **54%** | **1.63×** | 1421 us |

¹ 128/512 token 的总收益为负，百分比表示退化来源的拆分。serial 通信占比从
14% 上升并稳定在约 29~31%。1024 token 是公平口径由慢转快的交叉点。

## 5. 计算 kernel（MB1）

| 总 tokens | padding | vLLM compute | TK chain | TK/vLLM | vLLM/TK 有效 TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 7.11× | 1178 us | 1108 us | 1.06× | 18.5 / 19.6 |
| 512 | 2.00× | 1249 us | 1107 us | 1.13× | 61.9 / 69.9 |
| 1024 | 1.00× | 1427 us | 1104 us | **1.29×** | 108.3 / 140.1 |
| 2048 | 1.00× | 2136 us | 1695 us | 1.26× | 144.8 / 182.4 |
| 5120 | 1.00× | 5068 us | 4159 us | 1.22× | 152.5 / 185.9 |
| 6648 | 1.08× | 6887 us | 5699 us | 1.21× | 145.9 / 176.3 |
| 8192 | 1.00× | 8007 us | 6514 us | 1.23× | 154.5 / 189.9 |

小 token 的 TK 地板约 1.10 ms：128 token 实际 assignment 只有 288 行，却计算
2048 个 padded 行。1024 token 后 padding 基本消失，TK 链达到约 176~190
有效 TFLOP/s。日志提示 vLLM 未找到当前 GPU/shape 的 MoE config，使用默认配置，
所以“TK 计算快 1.2~1.3×”仍是暂定结论。

## 6. 通信 kernel（MB2）

| 总 tokens | serial 通信 | NCCL AG / RS busbw | TK push | push 带宽 | TK reduce | reduce 带宽 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 177 us | 12.1 / 12.1 GB/s | 45 us | 36.5 GB/s | 19 us | 15.0 GB/s |
| 512 | 323 us | 23.3 / 22.7 GB/s | 137 us | 45.8 GB/s | 54 us | 20.8 GB/s |
| 1024 | 522 us | 26.7 / 26.6 GB/s | 263 us | 47.9 GB/s | 102 us | 22.1 GB/s |
| 2048 | 904 us | 29.5 / 29.5 GB/s | 512 us | 49.1 GB/s | 197 us | 23.0 GB/s |
| 5120 | 2108 us | 30.8 / 30.7 GB/s | 1264 us | 49.8 GB/s | 479 us | 23.6 GB/s |
| 6648 | 2693 us | 31.2 / 31.1 GB/s | 1637 us | 49.9 GB/s | 623 us | 23.6 GB/s |
| 8192 | 3278 us | 31.4 / 31.4 GB/s | 2020 us | 49.8 GB/s | 770 us | 23.5 GB/s |

TK barrier 为 9~12 us，NCCL 小包 all-reduce 约 24~25 us。NCCL AG 与 TK
dispatch 的语义字节不同，不能只按耗时横比。8192 token 时 TK push/reduce 分别
搬 100.7/18.1 MB。

## 7. 融合损失与正确性（MB4）

| 总 tokens | L0 GEMM → fused | L0 损失 | L1 GEMM → fused | L1 损失 | 相对误差 |
|---:|---:|---:|---:|---:|---:|
| 128 | 708 → 870 us | +162 us（23%） | 363 → 403 us | +40 us（11%） | 3.44e-3 |
| 512 | 707 → 881 us | +174 us（25%） | 362 → 410 us | +47 us（13%） | 3.44e-3 |
| 1024 | 708 → 859 us | +151 us（21%） | 363 → 463 us | +100 us（27%） | 3.43e-3 |
| 2048 | 1005 → 1260 us | **+255 us（25%）** | 532 → 722 us | **+190 us（36%）** | 3.44e-3 |
| 5120 | 2500 → 2986 us | +486 us（19%） | 1292 → 1752 us | +460 us（36%） | 3.43e-3 |
| 6648 | 3462 → 4056 us | +595 us（17%） | 1771 → 2394 us | +622 us（35%） | 3.43e-3 |
| 8192 | 3962 → 4928 us | **+966 us（24%）** | 2016 → 2763 us | **+747 us（37%）** | 3.43e-3 |

正确性误差稳定在约 `3.43e-3`。MB4 的 `sched_ms=0.775~0.834 ms` 是 eager
测量，正式总体报数使用第 4 节的 205 us graph schedule。

## 8. 通信 SM 数（MB5）

| 总 tokens | 本次最优 k | 最优 e2e | k=16 e2e | 节省 |
|---:|---:|---:|---:|---:|
| 128 | 2 | 1197 us | 1316 us | 120 us |
| 512 | 8 | 1241 us | 1321 us | 80 us |
| 1024 | 12 | 1269 us | 1323 us | 54 us |
| 2048 | 12 | 2026 us | 2065 us | 39 us |
| 5120 | 16 | 4978 us | 4978 us | 0 us |
| 6648 | 16 | 6668 us | 6668 us | 0 us |
| 8192 | 16 | 8034 us | 8034 us | 0 us |

8192 token 的 k=4/8/12/16 e2e 为 15.19/9.43/8.05/8.03 ms；增至 24/32 又因
让渡计算资源退化到 8.15/8.46 ms。几十微秒差异接近约 5% 抖动，最优阈值需复测。

## 9. inter/intra-SM（MB6）

| 方向 | 配置 | 带宽 | compute slowdown | both − ideal |
|---|---:|---:|---:|---:|
| pull/inter | 4 blocks | 16.8 GB/s | 1.04× | 0.9 us |
| pull/inter | 8 blocks | 33.4 GB/s | 1.08× | 1.0 us |
| pull/inter | 16 blocks | **50.6 GB/s** | 1.17× | 0.6 us |
| push/inter | 1 block | 22.4 GB/s | 1.01× | 4.0 us |
| push/inter | 2 blocks | 43.2 GB/s | 1.02× | 1.1 us |
| push/inter | 4 blocks | **50.8 GB/s** | 1.04× | 0.6 us |

所有配置的 `both_ms` 都接近 `max(comm_only, compute_only)`，最大额外干扰约
5.9 us。intra-SM 虽能达到 50.8 GB/s，但计算 slowdown 为 1.14~1.33×，没有
表现出额外优势。此处计算代理是 fp32 FMA，真实 GEMM 让渡以 MB5 为准。

## 10. PCIe 方案与拓扑（MB7）

逻辑 pair 映射为 `0→1 = GPU 9→11`、`0→2 = GPU 9→13`、
`0→3 = GPU 9→15`。64 MiB 结果：

| pair | memcpyPeer | SM pull | SM push | row pull 顺序/散布 | row push 散布 |
|---|---:|---:|---:|---:|---:|
| 0→1 | **56.1** | 50.8 | 51.0 | 50.9 / 50.8 | 51.0 GB/s |
| 0→2 | **55.9** | 29.0 | 51.0 | 25.1 / 25.2 | 51.0 GB/s |
| 0→3 | **55.9** | 29.1 | 51.0 | 25.3 / 25.3 | 51.0 GB/s |

方向和拓扑决定性能，8 KiB 行是否散布不是瓶颈。16 MiB sweep 中 push 在
1/2/4 blocks 为 20.1/38.5/50.1 GB/s，pull 在 1/4/8/16 blocks 为
4.2/16.5/32.6/49.7 GB/s。

4 卡并发 ring 中，push 为 49.8~50.0 GB/s/卡、聚合 199.4 GB/s；pull 最慢卡
约 23.5 GB/s、聚合 94.0 GB/s。信号 RTT 在 0→1 上为 2.14 us，跨 pair 为
2.87 us。copy engine 单 pair 达 54~56 GB/s，但本批未测其 4 卡并发。

## 11. 优化优先级

1. 修复 prered_push 的 schedule 漏计，让 MB4 直接测 `sched_graph`。
2. 优先将 dispatch 数据面从 pull 改成 push，减少通信暴露并归还 SM。
3. 当前 pull 路径按负载自适应 comm SM；push 落地后重新 sweep。
4. 归因 L1 combine 尾部：8192 token 的 747 us 损失大于数据面传输时间。
5. 处理小 token padding 地板，但要防止减少 padding 后落入低效小 tile。
6. 调优 vLLM 后重跑 MB1/MB3/MB4，再确定最终计算收益比例。
7. 补测 4 卡并发 memcpyPeer，再决定是否原型化 copy-engine 数据面。

## 12. 口径与数据质量

- 本批为 **bf16**，不能外推为 FP8 性能。
- vLLM 使用未调优默认 MoE config，计算侧对比可能偏向 TK。
- 原始 e2e 漏计 prered_push schedule；本汇总统一用 205 us graph 修正。
- 日志开头有 `No module named 'triton.language.target_info'` 报错，但测试继续完成、
  JSON 全部保存，vLLM compute/serial 也有有效输出。复测时应确认没有改变 kernel。
- `nvidia-smi.txt` 的实际驱动是 580.105.08，而环境变量
  `CUDA_DRIVER_VERSION=570.86.10`；复现以 `nvidia-smi` 为准。
- MB2 的 NCCL/TK 语义字节不同；MB6/MB7 probe 用 runtime P2P，TK 正式实现用
  VMM+IPC。适合比较趋势，不是逐字节等价替换。

## 13. 原始文件索引

| 文件 | 内容 |
|---|---|
| `mb1_compute.json` | vLLM、TK、cuBLAS 计算 kernel |
| `mb2_comm.json` | NCCL 与 TK 通信时间、字节和带宽 |
| `mb3_ratio_sched205us.json` / `mb3_report_sched205us.md` | 公平口径收益分解 |
| `mb3_ratio.json` / `mb3_report.md` | eager schedule 诊断口径 |
| `mb4_fusion.json` | e2e、融合损失、输出交叉校验 |
| `mb5_sm_sweep.json` | 通信 SM 数、让渡和干扰 |
| `mb6_inter_intra.json` | inter/intra-SM 合成探针 |
| `mb7_pcie_schemes.json` | PCIe 方向、拓扑、并发和 RTT |
| `env*.txt`、`nvidia-smi.txt`、`clocks.txt`、`git_commit.txt` | 环境快照 |
