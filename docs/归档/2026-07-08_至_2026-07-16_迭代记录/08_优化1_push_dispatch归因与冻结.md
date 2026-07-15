# 08 优化#1 push dispatch 归因：PCIe 远端原子在高并发下丢增量（冻结）

**日期**：2026-07-08
**结论**：源端 push dispatch 方案**冻结**。根因是 PCIe 远端原子 `red.release.sys.add` 在
高并发散射写下**永久丢失增量**（~5~7%），不是索引/映射 bug，修复需重构计数协议。按评审
docs/07 的止损条件（"若涉及远端原子可见性/跨 CTA 协议，冻结 push 转 #3"）执行。

## 背景

评审 docs/07 优化#1：把 dispatch 从 pull（SM 拉，本机弱路径 ~20GB/s）反转为源端 push
（TMA 推，强路径 ~51GB/s）+ 远端 `red.release.sys` 计数。Phase 0 probe [B] 曾实测远端原子
"可用"，据此认为 experience/12 的"红线 2"在本机不成立。

## 现象

- push 正确性：32 专家对拍 reference_moe 通过（rel_err 4.3e-3）。
- **专家数 ≥64 时挂死**：GEMM producer 的 gate 自旋等某 row block 计数器 `==128`，永远等不到。
- 挂死在 `moe_dispatch_push` kernel 内（融合的 push+GEMM），前置 3 个跨卡 barrier 均通过。

## 决定性定位（两步 instrumentation，非盲调）

**第一步——host 侧源端直方图 vs 期望**（不跑挂死 kernel）：
把每卡 `(dst_rank, dst_block)` push 直方图 all-gather，与目标卡按 slack 算的 `expected` 逐 block 比。
结果：`mismatches=0, overflow=0`——**源端逻辑完全正确**。每个 block 应收/逻辑实发一致，
无越界。排除"`dst_slot`/padded-cumsum 映射错位"这一类。

**第二步——device 侧 push-only + counter dump**（加 `moe_dispatch_push_only` debug 入口，
只跑 push blocks 不跑 gate，不会挂）：seed 计数器 → push-only → barrier + 300ms settle →
读回目标卡 counter 逐 block 对比。结果（NE=64, 每 block 应=128）：

```
sum=3817~3908 vs exp_sum=4096   （短 ~5~7%）
under_blk 大量（112~127），over_blk 恒为空 []
```

- **只少不多**：守恒下若是映射错位，必然有 block 超收（>128）。**over 恒为空**排除映射/重复计数。
- **300ms + 额外 barrier 后仍短**：排除"增量还在 PCIe 在途"。增量是**永久丢失**。

## 根因

并发 `red.release.sys.add` 到 peer 显存，在**大量 push block 跨 4 卡向散射地址**同时红加时，
**丢增量**。probe [B] 之所以显示"可用"，是因为它只用 8 block×128 线程打**单个**目标地址、
并发度低；真实 dispatch 是数百 block 向数十个散射 counter 高并发红加，硬件/fabric 在这个压力下
不保证原子累加不丢。**这正是 experience/12"红线 2"的真面目，被 probe 的弱压力测试掩盖了。**

## 处置

1. **push 冻结**：`dpush` namespace 与 `moe_dispatch_push*` 保留在源码里（含 push-only debug
   入口），但 scheme 默认走 **pull**（`TK_DISPATCH=pull` 已是正确、已接入、已对拍的路径）。
   push 仅作实验/换机器（若某平台远端原子在高并发下可靠）时启用。
2. **probe 需增强**：Phase 0 probe [B] 应补一档"高并发散射红加"压力测试（N 卡 × 数百 block ×
   多目标地址，校验总和），否则会再次得出乐观的假结论。
3. **dispatch 提速改走无远端原子的路线**（后续，若要做）：
   - 源端 push 数据（强路径）+ **完成信号用 slot 序号**（每 (src,dst) 独占 slot，单写者，
     `st.release.sys` 写序号，目标卡按 slot 数精确等）——即 combine 已验证的 B' 同款协议，
     无远端原子。代价是目标卡要扫 slot 聚合，协议更重。
   - 或源端 pack 后用 copy engine 搬（0 SM、无原子），kernel 只等信号。

## 影响面

- 正确性无影响：主线一直用 pull，push 从未设为默认交付路径。
- 收益：#1 的预期 ~6× dispatch 提速暂时拿不到；转 #3（schedule 上 GPU，公平性必需）和 benchmark
  覆盖，这些低风险、确定有价值。

## 补充：无远端原子的 push2 重构（实现、验证、实测更慢，同样搁置）

按评审止损后的决定，实现了 push 的无远端原子变体 **push2**：
`push_data`（TMA 推数据，**无 red.add**）→ `pcie_barrier_all`（已验证的 slot+seq 原子-free
barrier）→ 普通 `grouped_gemm`（gate 投影）。单写者完成信号本就要拆分 push 与 GEMM，故直接
用「推 → barrier → GEMM」三段，只复用已验证原语，零新风险。

### 正确性 ✅
- 64 专家（push 会挂的规模）：对拍 reference_moe passed=True，rel_err 4.26e-3。
- 256 专家：30 迭代 stress 无挂死、稳定（push 在此必挂）。

### 性能 ❌ 反而更慢
4 卡 EP bf16 512 token，**dispatch-only** 隔离计时：

| dispatch 模式 | dispatch-only | 端到端层 |
|---|---|---|
| pull（融合 dispatch⊕gate GEMM） | **2.00 ms** | 7133 µs |
| push2（push→barrier→GEMM） | 2.65 ms | 7823 µs |

push2 比 pull 慢 ~30%。原因：
1. **丢了 layer0 的通算 overlap**——pull 把 dispatch 融进 gate GEMM 的 producer 自旋，
   push2 拆成三段串行（推、barrier、GEMM），多一次 barrier + 一次独立 kernel launch。
2. **512 token 数据量太小**：每卡 dispatch ~57MB，20→51GB/s 的带宽差（~1.8ms vs 1.1ms 理论）
   压不过固定开销（barrier RTT、launch、GEMM 不再被隐藏）。push 带宽的收益要在**大 token**
   档才显现，而本 benchmark 单点 512。

### 结论
push2 **正确但更慢**，与 push（更快但错）都搁置。**pull 仍是默认且最优**。
`moe_push_data` / `TK_DISPATCH=push2` 代码保留供大 token 档或换机器时再评估。

**教训**：#1 的 ~6× 预估是基于"dispatch 纯 comm-bound 且打满带宽"的账，但**pull 版本本就
把 dispatch 和 gate GEMM 融合overlap 了**，dispatch 并非纯串行通信——预估把已有的 overlap
收益重复计入了。真正要提速 dispatch，得在**保持融合 overlap 的同时**换更快的数据面，
而不是拆开。这需要 fused push+gate（producer 边等 push 边算），但完成信号又不能用远端原子
（丢增量）——本质回到"融合 kernel 里如何无原子地精确知道 128 个 token 到齐"的难题，
非平凡，优先级低于 #3/benchmark 覆盖。
