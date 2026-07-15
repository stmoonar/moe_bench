# 23 TP 第三轮计划：基于 microbench 平台事实的路线修订

> **依据**：EP 主线导入的 microbench 实测（docs/22 §0、
> `microbench/results/20260710_071848/RESULTS_SUMMARY.md`）。这些平台事实
> 直接改写 TP 的设计判断——特别是 **我们 TP dispatch 现在用的 pull 是弱路径**。
> 承接 docs/20（第二轮修复 pull_order + dispenser，实测在途）。

## 0. 对 TP 有决定意义的平台事实（引用 docs/22 §0，勿重测）

| 事实 | 数值 | 对 TP 的含义 |
|---|---|---|
| SM pull（弱路径） | 跨 PCIe 对 29GB/s，**4 卡并发退化到 23.5GB/s/卡**，要 16 SM 才饱和 | 我们 L0 AG 是 4 卡并发 pull：12.6MB ≈ **536µs** 且占 16 comm SM |
| SM push（强路径） | 50.9GB/s，**4 个 SM 打满**，4 卡并发零退化（49.8/卡） | push 化后同样 12.6MB ≈ **253µs**，只要 4 SM |
| comm SM 让渡代价 | k=16 → GEMM 慢 8~16%；k=4 只慢 ~4% | 16→4 可拿回 L0 GEMM ~120-240µs |
| 干扰 ≈ 0 | both ≈ max(comm, comp) | 编排=纯资源划分；dispenser/全员收尾方向正确 |
| copy engine | 54~56GB/s 零 SM，**并发未测**（docs/22 T14） | TP 的 AG 是稠密整块搬运，copy engine 天然契合——等 mb7 并发用例 |
| vLLM triton 未调优 | TK 计算链快 1.2~1.3× 可能部分虚高 | **TP serial 基线同样未调优**（E=64/N=768 config 缺失），调优后 serial 会变快 |
| NCCL AG/RS | busbw ~30GB/s；小包 24µs vs TK barrier 9~11µs | serial TP 的 AG+RS ~700-900µs，是我们要吃掉的部分 |
| EP 口径修正 | EP 默认路径漏计 sched，fair 后 1963→~2170µs（1.34×） | TP scheme 的 sched 无条件计入，**口径本来就 fair**，与 EP 对比时用 fair 数字 |

## 1. 当前状态与在途

第二轮修复（docs/20：pull_order + 全员 dispenser）**尚未拿到实测**（第三轮
上机毁于设备坑，docs/21）。第四轮回流将带回：修复后的 e2e、comm_sms sweep、
time_tp_stages 分阶段归因（L0/L1 相对 GEMM-alone 的暴露量）。

**用 mb 事实重算第二轮的预期**：pull 并发 23.5GB/s（比 docs/20 §4 假设的更差）
→ L0 AG ≈ 536µs，即使 pull_order 完美流水，L0 ≈ max(GEMM ~1.5ms, AG 536µs)
但 16 comm SM 的让渡使 GEMM 变慢 8~16% → L0 ≈ 1.65~1.75ms。
修正后预期 e2e ≈ **2.8~3.0ms**（原估 2.7-2.9），仍难赢 serial 2624µs。
**结论：TP 要赢，dispatch push 化不是可选项，是必选项。**

## 2. 任务清单（按数据支撑排序）

### TP-T1 dispatch push 化（主攻；比 EP 的 T10 简单得多）

- **依据**：§0 前两行。TP 的 AG 是**稠密、路由无关**的 shard 广播——push 形态
  天然成立，没有 EP push3 的目的块重排/信号爆炸问题。
- **设计**：
  1. 新增 peer 可写的 `ag_staging (world, T, H)` pgl；源卡 s 把自己的 shard
     push 到每个 peer 的平面 [s]（单写者、无原子；数据面 = tpdisp 现有
     TMA 行推的镜像，或整块大段 TMA）；
  2. 完成信号：per-source 水位（`st.release.sys` 写 seq 到 barrier 行 2+s，
     复用 pcie_sync 与 T6-v1 选举模式；每源卡只需 1 个信号）；
  3. scatter 块改从**本地** ag_staging 读（等对应源的水位）→ 散到 topk 个
     gathered slot + 计数，pull_order 改为 gate 在"源水位 + 本地散播"上；
     GEMM gate 逻辑不变（counter == ROW_BLOCK）。
  4. comm SM 降到 4~8（push 4 个 SM 打满；scatter 是本地 1.3TB/s 搬运）。
- **正确性锚点**：CPU 预检加 ag_staging 语义模拟；上机 verify_tp_schedule +
  run_tktp 三档 NE + skewed。
- **预期**：L0 通信 536→253µs 且完全可藏；GEMM 让渡 16%→4%；
  L0 ≈ 1.55ms，e2e 目标 **~2.4-2.6ms**（serial 2624 的 1.0~1.1×）。
- **回退**：`TK_TP_DISPATCH=pull|push` 开关，默认 pull 直到全档赢。

### TP-T2 comm SM 自适应（快赢，跟随 T1）

- pull 路径按 mb5 表（总 token ≤1024→8、2048→12、≥5120→16）；push 落地后
  重扫 k∈{2,4,8}。`TK_COMM_SMS` 保留为覆盖项。

### TP-T3 基线公平性：vLLM triton 调优（与 EP 的 P3 联动）

- TP 形状的 config key 是 **E=64, N=768**（全部 expert 本地、inter 分片），
  与 EP 的 E=16/N=3072 不同——EP 跑 `tune_vllm_moe.sh` 的产物**救不了 TP**。
  需要另生成 TP 形状的调优 json（脚本参数：num_local_experts=64、
  intermediate=768、topk 密度=8，即真实口径）。
- 调优后 serial 变快多少，直接改写 TP 的收益上限——**在报最终数字前必须做**。

### TP-T4 copy engine AG（远期，等 mb7 并发用例）

- TP 的 AG 是稠密整块 shard 搬运，是 copy engine 的理想形态（EP 的路由散布
  反而不是）。若 docs/22 T14 的并发 memcpy 用例证明不退化：dispatch 数据面
  换 `memcpyPeerAsync` ring，comm SM ≈ 0，只留 scatter+水位。
  **只有 mb7 数据支持才开工。**

## 3. 报数规范（对齐 docs/22 P4）

- 一律 fair 口径（TP 已默认满足：sched 无条件计入 run()）；
- token sweep 至少 512/2048 每卡两点（脚本已含 256/512/1024）+ NE 三档；
- 与 EP 对比时用 EP 的 **fair** 数字（~2170µs/1.34×，非 1963/1.48×）；
- 每轮回流结果目录整包归档进 `tp_test_results/`。

## 4. 顺序

```
第四轮回流（在途）：验证 docs/20 修复 + time_tp_stages 归因
  → TP-T1 push 化（3~4 天，含 CPU 预检扩展）
  → TP-T2 comm SM 重扫（半天）
  → TP-T3 triton 调优复测（0.5~2h 机器时间，可与 T1 并行）
  → 终数报告（fair、全档、引用结果目录）
```

量化目标：TP e2e **~2.4-2.6ms vs serial 2624µs**。诚实预期：TP 通信占比
（~25-30%）决定了天花板 ≈1.2×，远低于 EP 的 1.5-1.6×；TP 版本的价值在
证明融合框架的通用性与大 NE 档（serial 随 NE 恶化：NE=256 时 5292µs）。
