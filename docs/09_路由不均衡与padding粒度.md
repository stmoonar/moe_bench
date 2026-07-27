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

## 4. 两级 tile（2026-07-27 立项推进，分阶段）

主体行块保 128（不付 B 重载税——余数行本来就要多一个任务读一遍 B，
两级不新增 B 读），每 expert 余数行（1-127）单独走 64 尾块任务。预期
padding 税 25%→~5-8%（uniform e2e −250~300µs），balanced 零损失（无尾块）。

**Phase 0（概念验证，零 kernel 代码）**：64 行任务的单位经济学。
`verify_fp8_gemm` 已把 `row_block` 通到 CLI（第 6 个位置参数，走
TK_ROW_BLOCK 编译宏，.so 按 rb 分开缓存）：

```bash
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 4096 1536 20 64
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 20 64
```

判据：RB64（任务数 ×2、CONSUMER_WARPS=4、160 线程）总时间对比 RB128
同形状——**若 ≈ 持平或仅慢 B 重载份额，则 64 任务成本 ≈ 半个 128 任务，
两级 tile 上界兑现，立项**；若明显更慢（4 consumer warp 吃不满 tensor），
收益打折重估。注意该对比含 2× B 读（悲观界），两级 tile 尾块不付此税。

**Phase 0 实测定案（2026-07-27，GPU0，boost，含 P3 交织代码的 build，
量级结论不受影响）**：

| 形状 | RB128 fp8 | RB64 fp8 | Δ | raw Δ |
|---|---:|---:|---:|---:|
| L0 | 654.4µs | 749.4µs | +14.5% | +19.8% |
| L1 | 359.7µs | 436.5µs | +21.3% | +21.4% |

→ **64 行任务单位成本 ≈ 0.57（L0）/ 0.61（L1）× 128 任务**（含 2× B 读
悲观成分）。判定：**立项成立但收益打折**——uniform 税 22.9% → 预计
~13-15%（e2e −150~180µs），低于本节原 <5% 乐观估计（那假设 c64≈0.5 且
有 32 尾块）；L0-only 变体（裁决点 5 前者）预计 −~100µs。附带发现：
RB64 的 160 线程口径**没有 168 寄存器帽**（gg8 178 reg 零 spill、tppr8
184——ptxas 上限 65536/160=409），尾块模板的寄存器预算天然宽裕。

**Phase 1a 实现与首测（2026-07-27，GPU0）**：dispenser 尾块任务落地
（`blk_rows` 表 + 前半条带 warp 完整路径 / 后半"信号伴走"，布局零改动，
balanced 零开销双路结构，死锁审计零新增等待点）。tail=32 探针
（每 expert 2 满块+1 尾块）：**正确性 full/2level 双双 1.68e-03 OK**；
性能 v1 = L0 −6.7% / L1 −5.0%（预期 −14/−13），反推尾块成本 ~0.80×
而非 0.57×。**归因：v1 的 producer 仍给尾块装满 128 行 A tile**（stage
喂料 24KB 不减），尾块 stage 被 TMA 托底。**v2 = A64 装载**：利用 TK st
布局等价性（swizzle_bytes=128 单 panel → st<128,128> 前 64 行与
st<64,128> 逐字节相同），producer 对尾块只装 8KB A tile（stage 24K→16K，
expect_bytes 同步），consumer 零改动；编译期 gl_has_tma trait 检测 gl 是
否带 A_tail_tile 描述符（TK get_tma 非 SFINAE，requires 检测恒真，须对
gl 模板参数列表偏特化匹配），未接入的 fused kernel 自动回退全量装载。

**v2 实测（GPU0）**：L0 −8.9% / L1 −6.7%（v1 −6.7/−5.0），raw −8.3/−8.8。
尾块成本 ~0.73-0.80，仍距 0.57。**v3 = 尾块恒等 strip 映射**：残差主因
是 active warp 沿用 store 交织映射选出 {0,4,1,5} → 按 warp_id%4 聚在
SMSP {0,0,1,1}，**半数 tensor 单元整个尾块任务闲置**（RB64 实验 4 个
warp 恰好铺满 4 个 SMSP，故能到 0.57）。尾块 store 是单 warp 版不受
group store 交织约束 → 改恒等映射（warp 0-3 → 条带 0-3，SMSP 0-3 全
铺满），满块路径不变。

**v3 实测定案（2026-07-27，GPU0）**：tail=32 探针 **L0 −12.2%（尾块
成本 0.63）/ L1 −9.9%（0.70）**，raw −10.1/−9.9；正确性 full/2level
全部 1.68e-03，默认路径无回退（659.3）。距理论 −14.3/−13 已收 85%/76%，
残差为每任务固定开销量级。**Phase 1a 引擎机制收敛，进 Phase 1b**
（调度表按每 expert 余数发尾块 + fused 两层接入 + uniform e2e 报数）。

## 6. Phase 1b 定案（2026-07-27，卡组 0-3）：uniform e2e −45µs，正确性逐位等价

接入方式：blk_rows ≡ ROW_BLOCK − slack，直接复用调度表 slack（golden/
tpsched 已对拍、每迭代重建）→ 调度链零改动；两层 fused 挂 A_tail_tile
描述符启用 A64；开关 `TK_TWO_LEVEL`（默认 1）。

- **正确性（最强判据）**：balanced rel_err 与基线逐位一致
  （0.042817506939172745）；uniform 下 two_level on/off 四 rank 的
  rel_err/max_abs_err **全部逐位相同** → 尾块路径与满块逐比特等价在
  真实路由坐实。balanced e2e 1530.9 零回退。
- **性能**：uniform e2e **1903.1（on） vs 1946.0（off）= −43~48µs**，
  劣化 415→372µs（对 balanced 1531）。低于探针折算 ~90µs 的原因：
  ① 真实 uniform 尾块占比仅 ~20%（~32/160 块，探针形状 33%）；
  ② L0 的 GEMM 省时部分被 AG 重叠窗口的通信约束吃掉（GEMM 变快后该
  窗口 comm-bound，省时不全兑现 e2e）。
- 残余方向（未立项）：uniform 口径下重扫 `TK_COMM_SMS`（GEMM 变快后
  让渡平衡点移动）；64 粒度 act 布局（省 scatter/推流 padding 流量，
  裁决点 5 后者）。正式报数建议 balanced+uniform 双口径、TWO_LEVEL
  默认开。

**Phase 1（立项后）设计裁决点**（按依赖序）：

1. 任务描述加 M 型别（128/64）：blk_expert 表旁挂 blk_rows 或高位编码；
   tpsched/torch 版/host golden 三层同步 + preflight 对拍扩展。
2. producer：尾任务 A tile 用 64 行 TMA 描述（第二个 st 类型 + 描述符），
   B 路径零改动。
3. consumer：尾任务 warps 4-7 跳过 compute/store 但**照常 arrive 全部
   信号**（inputs_finished/task_done 计数不变——死锁审计点）。
4. store/epilogue：半高 store 路径（plain 直接半高；GLU 配对是列维
   [gate32|up32]，行高减半不影响）。
5. **最大裁决点——act/padded 布局粒度**：若下游布局仍按 128 补齐，只省
   L0 的 QMMA（收益减半）；若整链改 64 粒度补齐，L1 GEMM/scatter/推流
   同步受益，但所有 per-row-block 信号与调度表要过一遍死锁/正确性审计。
   建议先做前者（L0-only，改动面小）拿一半收益，再评估后者。

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
