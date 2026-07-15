# 33 TP 第十三轮:L1 v2 首测回退归因(推送粒度病)与 16-token 组批修复

> 第十三轮承接 docs/32。首测(tp_run_20260712_083157)**48/48 正确性全绿,
> 但性能大幅回退**:默认档 2675 vs L1 v1 回退档 2119(+556µs)。协议对,
> 粒度错。

## 1. 首测归因

| 项 | v1(08l) | v2(08) | 结论 |
|---|---:|---:|---|
| L1_fused | 691 | **1294** | +603µs,全部回退在此 |
| sched | 272 | **233** | 删 job_order 兑现 −35µs ✓(意外之喜) |
| L0_fused | (干扰) | 1168 | 与第十一轮一致,未受影响 ✓ |
| final_red | 20 | 14 | 略好 ✓ |

- 04l(TK_L1=v1)= 2119 与第十一轮 2125 一致 → 其余栈完好,问题隔离在 L1 v2;
- 三档正确性门全绿 → 列扫聚合信号/选举/watermark 协议正确;
- 本轮两处双峰(t1024 med 10945/min 4930、rb64_rep med 6969)均为环境干扰
  (rep/对照稳定),按 docs/27 纪律取稳定值。

**根因:job 粒度 = 1 token × 1KB**。每个 job 一次 TMA push + 一次
`store_async_wait` 串行化——**每块同时只有 1 个 1KB 写在飞**,PCIe 写延迟
被暴露 16384 次(v1 是 2048 次 × 8KB)。粗账:每 job ~4-5µs 串行延迟,
GEMM 窗口内 24 个 comm 块只能消化 ~3000 个,剩下 ~13000 个在 GEMM 结束后
排空 ≈ +500µs——与实测 +603 吻合。违反了 experience/02 §3 自己写下的
判断标准:**工作/信号粒度应是"能触发一次高效通信的最小单位"**。

## 2. 修复:16-token 组批推送(GRP=16)

关键洞察:`j = src_dev*T + src_tok` 的 job 编号**天然目的卡优先**——连续
16 个 j 同目的卡、目的行连续(T % 16 == 0,全档成立)。于是:

- push job 改为 **(chunk c, 16-token 组 g)**:一个 job 归约 16 个 token 的
  512 列段(16KB smem,GRP×CVEC=1024 个 float4 单元全块摊),然后
  **背靠背发 16 个 TMA store,最后只等一次** `store_async_wait`——
  单块吞吐 = 16 个 1KB 写并发在飞,延迟摊薄 16 倍;
- job 总数 16384 → **1024**((world*T/16)×8 chunk),原子/信号量开销同步
  缩水;选举 atomicAdd 一次加 nreal(组内非空 token 数),expected×NCHUNKS
  口径不变;
- N 维分解的全部收益保留:chunk-major 序、列扫聚合信号、从 GEMM ~1/8
  进度起流推。

## 3. 新增归因探针:grouped_gemm_cm

首测无法区分"+603 里有多少是列外层换序拖慢了 GEMM 本身(L2 复用变化)"。
新增 `grouped_gemm_cm`(COL_MAJOR dispenser 纯算版,无 gate 无推送),
time_tp_stages 每轮同时报:

- `L1 exposure`(vs 行外层 plain 参考,历史可比);
- `L1 exp(vs cm)`(vs 列外层参考,v2 的真实协议暴露);
- `cm-rm delta`(列外层换序本身的 GEMM 代价,若显著则考虑 swizzle 折中)。

## 4. 预期与裁决

- L1_fused 预期 1294 → **~600-700**(推送延迟摊薄后基本贴 GEMM);
  e2e 预期 ~2000-2100,若 cm-rm delta ≈ 0 则应逼近 2000;
- 若仍慢于 v1:看 `cm-rm delta`(GEMM 换序代价)与 `L1 exp(vs cm)`
  (残余协议成本)哪个背锅,分别对应 swizzle 折中 / GRP 或 chunk 再调;
- 06/07 各档、正确性门、A/B(04l)照旧。
