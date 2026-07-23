# 路由不均衡的劣化归因与 padding 粒度（RB64 负结果）

日期：2026-07-23。卡组 4-7，主配置形状（E=64/T=512/topk8/fp8），P2 push 默认
路径。背景：balanced 是对我们最友好的完美对齐口径（256 行/expert 整除
ROW_BLOCK），真实业务路由更接近 uniform/skewed，需要量化不均衡敏感性。

## 1. 实测矩阵（2026-07-23，iters=50）

| e2e (µs) | RB128（默认） | RB64 | serial |
|---|---|---|---|
| balanced | **1618** | 1867 (+15.4%) | 2062 |
| uniform | **1988** | 2146 (+7.9%) | 2167 |
| 不均衡劣化 | +370 (+22.9%) | +279 (+14.9%) | +105 (+5.1%) |

- uniform 下我们对 serial 的领先从 21.5% 缩到 **8.3%**（RB128）。
- 复现：`run_tktp 64 --dist uniform --precision fp8 --no-verify --iters 50`
  （`--dist`/`--skew-alpha`/`--active` 旋钮见 tools/run_tktp.py；serial 加
  `--scheme serial`，同种子链 → 两边路由逐 bit 相同，直接可比）。

## 2. 劣化归因：ROW_BLOCK=128 的 padding 税

每 expert 行数补齐到 ROW_BLOCK 整数倍。uniform 下每 expert ≈256±16 行，
而 256 恰好卡在 128 的块边界：**约一半 expert 越到 257+ 行 → 3 块**，平均
行块 2.0→~2.5，M 维 GEMM 有效功 +25%；且 L0 GEMM、L1 GEMM（act 按
expert 补齐布局）、scatter/推流全按 padded 行走。账验证：balanced 下
GEMM 相关 ≈1500µs ×25% ≈ +375 ≈ 实测 +370。**是粒度税，不是调度失灵**
（dispenser 对不均衡块数自适应良好）。

serial 只付 +5.1%：vLLM triton 的 M 块粒度是 64。**BLOCK_M=64/BLOCK_N=128
从 NCU grid 反解实锤**（docs/08 的 report）：grid = cdiv(topk·M + E·(BM−1),
BM) × cdiv(N, BN) = cdiv(16384+64·63, 64)×{12, 32} = 3828/10208，与 w13/w2
两颗 kernel 完全吻合（最坏情况分配的多余块 early-exit，near-free）。粒度
差一倍 → padding 税差 ~2.5 倍，与 22.9% vs 5.1% 吻合。

## 3. RB64 A/B：负结果

`TK_ROW_BLOCK=64`（T5 管道，fp8 config 已吃宏，零代码改动；编译竞态修复
见 §5）。正确性门过（rel_err 4.28e-2 与 RB128 同水位 → WG=4 store 交织
在 fp8 路径同样成立）。性能判负：

- **B tile 重载税 +249µs（balanced 实测）**：行块数 ×2，每任务重拉整条
  权重列块（COL_BLOCK=128 × K），L0/L1 权重流量近似翻倍。balanced 下
  padding 两边均为零，+249 即纯 B 重载税。
- padding 敏感度确实降了（劣化 370→279µs，机制兑现），但底座抬高把收益
  全吞：uniform 档 2146 **比 RB128 的 1988 还慢 158µs**，对 serial 领先
  几乎归零（2146 vs 2167）。
- **结论：本形状整体换 RB64 判负**。小块的收益只该付给余数行。

## 4. 后续方向：两级 tile（未立项）

主体行块保 128（不付 B 重载税），每 expert 余数行（1-127）单独走 64 或
32 的尾块模板。预期 padding 税 25%→<5%，balanced 零损失。落点：blk_expert
表构建挂异构任务描述 + GEMM 模板加 64-M 实例 + dispenser 按任务型分发。
在瓶颈排序中与本地 scatter/sched 竞争优先级；若报数口径改双分布
（balanced + uniform），优先级应上调。

## 5. 附带修复与口径说明

- **并发编译竞态**（RB64 首跑暴露）：冷缓存下 mp.spawn 4 worker 同时 nvcc
  写同一 .so → ImportError/file too short。修复：build.py 排他 flock +
  tmp+atomic rename（提交 2d66561）。半成品 .so 需手动 rm 后重跑。
- **verify FAIL 口径**：我们 rel_err 4.28e-2（w1 交织后二次量化 + act 源端
  量化的叠加噪声）vs serial 1.67e-2（单次量化），元素容差 0.035 卡在中间
  → 我们 ~2% 元素越线、serial 名义上也 FAIL（个位数元素）。所有方案变体
  （lane/push/warp/RB64）rel_err 钉在 4.27-4.28e-2 同水位 = 协议无关，
  纯量化口径。处置悬案（放容差或消 w1 二次量化）仍挂 HANDOFF。
- 建议：正式报数从单 balanced 口径改为 **balanced + uniform 双口径**，
  balanced 是我们的最优对齐上界，uniform 更接近真实负载。
