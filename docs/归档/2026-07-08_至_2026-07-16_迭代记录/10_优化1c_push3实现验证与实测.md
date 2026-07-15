# 10 优化#1c push3（slot 信号 dispatch⊕GEMM 融合）实现、验证与实测

**日期**：2026-07-09
**状态**：**已实现、正确、条件达标**（NE≤128 快于 pull；NE=256 慢于 pull）→ 作为可选
`TK_DISPATCH=push3` 保留，**默认仍 pull**（生产档 NE=256）。
**前置**：docs/09（规划）、docs/08（push/push2 冻结）、experience/12 §3（slot+序号协议）

## 0. 一句话结论

push3 解掉了 docs/08 的核心难题——**在保持 dispatch⊕gate GEMM 融合 overlap 的同时，
用强路径 TMA push 数据面，且完成检测零远端原子**。正确性完胜 push（NE=64/256 全过），
性能出现**专家数相关的交叉**：低/中专家数（≤128）显著快于 pull，高专家数（256）反而慢。
生产 benchmark 单点是 NE=256，故 push3 达不到"设为默认"的门槛，但作为一个正确且在
一大片工况下更快的方案保留。docs/08 的 push2 是"处处更慢"，push3 是"半数工况更快"——
这是实质进展。

## 1. 实现要点（与 docs/09 §3 规划一致）

- **host 侧**（`tk_scheme.py::_build_schedules`）：复用现有 replay 循环，增产
  `push_cnt_idx`（每 outgoing assignment 的本地扁平计数器下标 = `dst_blk_offset[dst]+slot//128`）、
  `push_expected`（各计数器满额值 = bincount）、`gate_expected`（本卡作为目的卡，
  每本地行块从各源卡应收 token 数）、`dst_blk_offset`。与 `push_idx/push_src` 同 index、
  同 `valid` 掩码过滤——避免 device 侧计数器错位。
- **barrier 扩容**：`barrier_l0` (2, C) → (2+world, C)。行 0/1（pull 计数器 / pcie_barrier_all）
  字节不变，无回归。行 2+s = 源卡 s 的 push3 完成信号，列 = 目的卡本地行块。
- **kernel**（`tk_moe.cu::dpush3`）：
  - push 块：TMA push → `store_async_wait` → `atom.acq_rel.gpu` 本地计数 →
    满额线程 `__threadfence_system()` + `pcie_sync::signal_slot`（`st.release.sys`，单写者）。
  - `push3_gate`：producer 每行块本地自旋，只等 `gate_expected>0` 的源卡（O(N)=4 个本地 load）。
  - `grouped_gemm_sm120` 模板/consumer/producer **零改动**——通信仍是贴在标准 GEMM 上的 Gate functor。
  - 另出 `moe_dispatch_push3_only` debug 入口（只跑 push 块，用于 §3.4 R1 裁决）。

## 2. R1（内存序，docs/09 §3.4 唯一非平凡风险）裁决——**通过**

用 push3-only + 双重对账（`tools/validate_push3.py`）：
- (a) **数据面**：push3-only 后 `gathered[:npl]` 与 pull 版 `gathered` **逐字节 `torch.equal`**；
- (b) **信号**：目的卡 `barrier[2+s][rb] == seq` 当且仅当 `gate_expected[rb][s]>0`，否则 `<seq`。

结果：**NE=64 与 NE=256 各 30 迭代，total_failures=0**。"信号先于数据到"的偶发竞争
**未出现**——`store_async_wait`（bulk 写提交）→ `atom.acq_rel.gpu`（release）→ 满额线程
`threadfence_system` → `st.release.sys` 的链条在本平台成立，与 `combine_signal_epilogue`
同构、同样可靠。补救手段（重读远端字节 / 单 warp 顺序发）**未触发，无需启用**。

host 侧 schedule 对账（`tools/verify_push3_schedule.py`，docs/09 §5.1 锚点）：
`push_expected[s][dst_blk_offset[d]+rb] == gate_expected[d][rb][s]`（源=汇），且行块行和
`== 128 - slack[d][rb]`（与已验证 pull slack seed 一致）。NE=64/256 mismatches=0。

## 3. 融合正确性（docs/09 §5.5）——**通过**

- **NE=64 融合对拍 reference_moe**：`verify=ok, rel_err=4.27e-3`（与 pull 完全一致），
  3 次重跑稳定。
- **NE=256 stress 30 迭代**：无挂死、无错（push 在此必挂）。
  （NE=256 full-dense reference 会 OOM 14GiB——这是参考实现的显存上限，非 push3 问题；
  且 §2 已证 gathered==pull 逐字节，其后 up/silu/combine 与 dispatch 模式无关，
  256 正确性由 NE=64 对拍 + 256 gathered 对账**传递成立**。）

## 4. 实测（4 卡组 9,11,13,15，跑前查空闲；同 docs/08 口径）

**dispatch-only 隔离**（layer0 dispatch⊕gate GEMM，跨卡 barrier 锁步，max-over-ranks 中位）：

| 专家数 NE | pull | push3 | Δ |
|---|---|---|---|
| 64  | 2064 µs | **1589 µs** | **−23%** |
| 128 | 2017 µs | **1831 µs** | **−9%** |
| 256 | 2005 µs | 2779 µs | **+38%** |

**端到端整层**：

| 专家数 NE | pull | push3 | Δ |
|---|---|---|---|
| 64  | 5081 µs | **4607 µs** | **−9%** |
| 256 | 7124 µs | 7936 µs | **+11%** |

**成功判据对照**（docs/09 §4："dispatch-only < 2.00 ms"）：
- NE=64：1589 µs < 2000 ✅　NE=128：1831 µs < 2000 ✅
- NE=256（**生产档**）：2779 µs > 2000 ❌

## 5. 交叉点归因

push3 相对 pull 的开销随**专家数（≈本卡行块数）线性增长**，收益（push 带宽）基本恒定：
- **收益**：数据面从 SM pull ~20GB/s 换成 TMA push ~51GB/s，每卡 dispatch ~57MB，恒定收益。
- **开销**：完成信号是 per-(src, dst 行块) 的**远端小 store**。NE=64 每卡 16 行块 → ~64 个
  信号 + 每行块 4 个本地 gate wait；NE=256 每卡 64 行块 → ~256 个信号 + 256×4 gate wait。
  信号/等待流量 4×，且 NE=256 每专家 64 真 token padding 到 128（50% 浪费），GEMM
  被拉长、gate 尾部占比升高。开销在 256 档吃掉了带宽收益并反超。

即：**push3 在"行块少、数据面占主导"时赢，在"行块多、信号/gate 协议占主导"时输**。
交叉大约在 NE=128~256 之间。

## 6. 处置（docs/09 §5.7）

- **保留为可选路径**：`TK_DISPATCH=push3` 接入 `tk_scheme.py::run()`，正确、已对拍。
  低/中专家数（≤128）或大 token 档（数据面占比更高）下它是更快的选择。
- **默认仍 pull**：生产 benchmark 单点 NE=256，pull 更快，不改默认。
- **不追 NE=256 调优**（止损纪律，同 docs/08）：可能的杠杆是 `num_comm_sms` 调参、
  信号粒度改回 per-(src,expert)（256 档多数 expert 恰 1 行块，差别小）、或每 (dst,rb)
  集中单 warp 发信号减少远端 store 数——但这些属边际优化，优先级低于下面两项**正交叠加**：
  - **(token,dst) 去重**（docs/07 P3）：push 天然适合，流量 4096→~1800，数据面再省 ~2.2×，
    会把交叉点右移（NE=256 也可能翻盘）。
  - **FP8 传输**（docs/07 #2）：字节减半，与 push3 正交。

## 7. 教训

- docs/08 的难题（"融合 kernel 里无远端原子地精确知道 128 token 到齐"）**是可解的**：
  把"多写者计数"在 host 侧变成"单写者定值信号"（Triton-distributed 同款两阶段协议），
  本地 `atom.acq_rel.gpu` 选举唯一信号者，跨卡只用 `st.release.sys`。这条协议在本仓库
  layer1 combine 早已验证，push3 只是搬到 dispatch 输入侧。
- **收益是工况相关的**，不能只看单点。push2 单点（256）更慢就冻结，可能错过了它在其他
  档位的价值——这次 push3 全档位扫描才暴露出交叉点。以后新方案默认扫 NE∈{64,128,256}。
- push3-only + 逐字节对账 + 信号对账的 debug 模式（docs/08 传下来）再次是关键——
  R1 风险不是靠"跑跑看对不对"排除的，是靠隔离协议、30 迭代抓偶发、双重对账裁决的。

## 附：复现命令

```bash
source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
cd /data/cinnzhang_vllm_td_test/xxy
# host schedule 对账
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.verify_push3_schedule
# R1 协议裁决（push3-only, 30 迭代）
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.validate_push3 64 30
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.validate_push3 256 30
# 融合对拍
CUDA_VISIBLE_DEVICES=9,11,13,15 TK_DISPATCH=push3 python -m moe_bench.tools.run_tkfused 64
# dispatch-only 隔离计时
CUDA_VISIBLE_DEVICES=9,11,13,15 TK_DISPATCH=push3 python -m moe_bench.tools.time_dispatch 256 10 50
```
