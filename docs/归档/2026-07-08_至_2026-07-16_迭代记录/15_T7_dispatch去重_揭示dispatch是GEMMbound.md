# 15 T7:dispatch (token,dst) 去重 —— NE≤128 大赢,NE=256 揭示 dispatch 是 GEMM-bound

**日期**:2026-07-09
**任务**:docs/11 §T7 / docs/07 P3 —— dispatch 逐 assignment 拉取无去重,同一源 token
路由到本卡多个 expert 时被重复跨卡拉。
**状态**:✅ 落地(v0),**默认关**(`TK_DEDUP=1` 开)。正确(gathered 逐字节等价)。
**关键发现**:实测推翻 docs/11 账本"dispatch ~2005µs comm-bound"——**NE=256 的 dispatch 是
GEMM-bound**(gate GEMM ~1.7ms > 被掩盖的通信),去重 7.2× 减通信量却零 e2e 收益;
NE≤128 通信裸露,去重把 dispatch-only 从 2.0ms 砍到 ~1.1ms(e2e NE=64 −0.47ms)。

---

## 1. 去重因子(实测,4 卡 balanced)

`disp_idx` 按 (src_dev, src_tok) 分组去重:

| NE | padded_local | valid_pulls | unique | dedup |
|---|---|---|---|---|
| 64  | 4096 | 4096 | 736 | 5.6× |
| 128 | 4096 | 4096 | 624 | 6.6× |
| 256 | 8192 | 4096 | 568 | **7.2×** |

远大于 docs/11 §0.3 的 ~2.2× 估计。

## 2. 实现(v0,namespace ddisp)

**稠密 staging 布局**:staging_row(src_dev,src_tok) = src_dev·num_tokens + src_tok,
容量 S_max = world·num_tokens = 2048(routing 无关常量,与 T6 partials 同布局)。两张表
纯是 disp_idx 的函数(`_derive_staging`,host golden + GPU builder T3 共用,继承其正确性
与 graph 可捕获性):
- `slot_to_staging` (npl,):gathered slot → staging 行(padding=-1);
- `staging_needed` (S_max,):1 iff 某本地 slot 引用该稠密行(unique 拉取表)。

两 kernel(同 stream,kernel 边界=发布 barrier,零新协议):
- `pull_unique_kernel`:每个 needed 稠密行跨卡 TMA 拉一次到 LOCAL staging(唯一跨卡流量);
- `kernel`(scatter⊕gate GEMM):每 slot 从 `staging[slot_to_staging[slot]]` 本地拷到
  gathered,再 **红加同款 per-row-block 计数**(padding slot 也 +1,语义与 pull 逐字节一致),
  与 gate GEMM 融合。staging 是纯本地 28MB tensor(无 peer 访问,无 pgl)。

**计数正确性**:kernel 2 与原 pull 逐 slot 结构相同,red.add 一模一样(含 -1 padding),
每 row block 仍恰好到 128,release 序不变 → gathered 逐字节等价。

## 3. 性能(实测,4 卡 512 tok/rank)

dispatch-only(tools/time_dispatch,gate-only):

| NE | dedup OFF | dedup ON |
|---|---|---|
| 64  | 2058µs | **1087µs** |
| 128 | 2018µs | **1238µs** |
| 256 | 2005µs | 2029µs(**无收益**) |

e2e:NE=64 3465→**2997µs**(−468);NE=256 5839→6041µs(**略负**,失去 fusion overlap +
scatter 往返)。

## 4. 为什么 NE=256 无收益:dispatch 是 GEMM-bound(账本修正)

单测 gate GEMM(npl=8192, H=7168, inter=2048, 满 SM)= **1705µs**;只给 94 SM(16 让 comm)
更长。dispatch-only 2005µs ≈ 这个 GEMM。所以 **NE=256 dispatch 的瓶颈是 gate GEMM 计算,
不是跨卡 pull**——pull 43MB 本就被 GEMM 掩盖住了。去重把 pull 降到 ~8MB 毫无用处,反而
v0 的两 kernel 串行 + 本地 scatter 往返略增开销。

docs/11 §0.1 把 dispatch 2005µs 记为"comm-bound ~2.15ms @ 20GB/s"是**巧合数值接近**的误判;
真实是 compute-bound。这也解释了 docs/12 combine 尾才是真正 comm-bound(弱路径 gather),
两段瓶颈性质不同。

NE≤128 npl 小(4096)、gate GEMM 短,通信裸露 → 去重直接砍掉裸露的 pull。

## 5. 结论与后续

- **默认关**(prod 点 NE=256 无收益);`TK_DEDUP=1` 在 NE≤128 大赢,已验证正确可用。
- **与 T5 协同**:T5(ROW_BLOCK=64)把 NE=256 的 padding/npl 减半 → gate GEMM ~减半 →
  通信重新裸露 → 届时去重在 NE=256 也会赢。T7 是 T5 的前置储备。
- **T7 的 fp8 第二步**(docs/11 §T7)独立于去重,减字节对 comm-bound 的 combine 更有用;
  dispatch 既是 compute-bound,fp8 传输对 NE=256 dispatch 同样帮助有限,但 fp8 **计算**
  (SM120 原生 mma)会直接砍 GEMM——那是另一条线(T7 v2 / 精度切换)。

## 6. 正确性裁决

- `verify_schedule_gpu.py`(加入 slot_to_staging/staging_needed):host golden vs GPU builder
  逐元素全等,NE∈{64,128,256}×{balanced,skewed} 全过;
- run_tkfused 全链路 NE=64/128 rel_err 4.42/4.34e-3 ok(与 pull 一致,去重不改结果);
- 默认(dedup off)NE=256 e2e 5830µs 无回归。

## 7. 复现

```bash
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.time_dispatch 64 10 50       # dedup off
CUDA_VISIBLE_DEVICES=9,11,13,15 TK_DEDUP=1 ... (scheme 默认已 off, tool 读 sch.dedup_dispatch)
CUDA_VISIBLE_DEVICES=9,11,13,15 TK_DEDUP=1 python -m moe_bench.tools.run_tkfused 64
```
（注:time_dispatch 依 `sch.dedup_dispatch`;去重开关走 TK_DEDUP,scheme 默认 0。）
