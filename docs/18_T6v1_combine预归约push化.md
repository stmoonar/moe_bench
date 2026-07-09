# 18 T6-v1:combine 预归约 push 化(边算边推,消 barrier + 零重叠)

**日期**:2026-07-09
**任务**:docs/11 §3 / docs/17 §4 —— 把 T6-v0 的"expert 本地写 partial → 全卡 barrier →
源卡 pull" 反转为"expert 边算边 TMA-push partial 到源卡 staging + 水位选举信号,源卡等
水位后归约",让 combine 传输藏进 W2 GEMM,消除 docs/17 实测的 ~0% 重叠。
**状态**:✅ 落地并设为默认(`TK_COMBINE=prered_push`)。NE∈{64,128,256} 30 迭代对拍全过
(rel~7e-3);e2e(默认 shape E=64/hidden=4096)**2331→1963µs(−368µs,−16%),vs serial
2905µs = 1.48×**。

---

## 1. 设计(镜像已验证的 dpush3 push+选举)

- **数据面**:job block 算完 (s,t) 的 partial 行(FP32 加权和,写 smem `sv_bf<H>`),
  立即 `tma::store_async` push 到源卡 s 的 `combine_staging[s]` 行 `my_rank*T + t`
  (强路径 51GB/s)。staging 是 peer-writable TKParallelTensor (world*T, H),平面 [d]
  只被 expert 卡 d 写(单写者,无原子)。
- **边算边推**:job block 与 W2 GEMM 同 grid;GEMM 写完一个 row block → 本地信号(barrier_l1
  row 1)→ 命中该块的 job 立即算+推。推流贯穿整个 W2 GEMM,把跨卡传输藏进 GEMM。
- **水位选举**(= dpush3 R1 同款):每卡对每个源卡 s 一个本地计数器 `local_cnt[s]`;push 后
  `atom.acq_rel.gpu` +1,达到 host 预算 `push_expected_l1[s]`(= 本卡推给 s 的行数)的
  那个线程是**唯一选举者**:`fence.sys` + `st.release.sys` 写源卡 s 的
  `barrier_l1[s][2+my_rank][0] = seq`。无远端原子。
- **源卡归约**(`final_reduce_push`):每 token 一个 block,只等 `recv_from[d]>0` 的卡的
  水位(`barrier_l1[my][2+d]`,≤world 个本地自旋),然后按 `final_contrib[t][d]` 加
  `combine_staging[my][d*T+t]`。**无全卡 barrier**,HOL 消失。

**内存序链**(= push3 R1,docs/09 §3.4):`store_async_wait`(远端 bulk 写提交)→
本地 acq_rel 计数 → 选举者 acquire 读到满额 → 选举者 `fence.sys` → `st.release.sys` 水位 →
源卡 `ld.acquire.sys` 等水位 → 读 staging。单写者、单 src→dst 路径 posted 写不重排。

**信号布局**(barrier_l1,加宽到 `(2+world, bar_cols)`):

| row | 写者 | 读者 | 语义 |
|---|---|---|---|
| 0 | 本地 GEMM epilogue 选举(atom.acq_rel.gpu) | — | W2 col-block 计数(每迭代 reset) |
| 1 | 本地 st.release.gpu(seq) | job block slot gate | 本地 W2 完成信号(同 v0) |
| 2+d | expert 卡 d 的 fence.sys + st.release.sys(seq) | 源卡 final_reduce_push(ld.acquire.sys) | 跨卡水位:"d 推完了给我的所有行" |

rows 0/1 与 bar_cols 不变 → pull-combine(comb)与 v0 prered 逐字节不变。

## 2. host 表(纯 prered 表的函数,GPU builder 同步)

`_derive_combine_push`(host + `_build_schedules_gpu` 共用,capture-safe):
- `push_expected_l1 (world,)`:本卡推给源卡 s 的行数 = `has_hit.view(world,T).sum(1)`
  (has_hit = 该 job 有本地命中);
- `recv_from (world,)`:作为源卡,`(final_contrib[:,d].sum()>0)` —— 只等真会来的卡的水位
  (push_expected 的对称补,杜绝死等)。

## 3. 正确性

- **validate_prered_push.py**(以 pull combine 为 golden,30 迭代):NE∈{64,128,256} 全过
  total_failures=0,max_rel 6.9e-3 / 7.0e-3 / 3.7e-3(与 v0 同量级,一次额外 bf16 舍入);
- run_tkfused 全链路 vs reference_moe:rel_err **4.43e-3 ok**;
- 关键坑已规避:local_cnt 每迭代 same-stream `.zero_()`(不在 kernel 内 reset,避免在途
  远端 add 竞争);seq 单调免复位;空 job 不推不计数(push_expected 对齐);staging 平面
  单写者 + 求和按 final_contrib 门控(未写的行不读)。

## 4. 性能(默认 shape E=64/hidden=4096,4 卡,512 tok/rank)

| | e2e (min) | vs serial 2905 |
|---|---|---|
| v0 prered(barrier+pull) | 2331µs | 1.25× |
| **v1 prered_push(边算边推)** | **1963µs** | **1.48×** |

融合损失(tools/analyze_overlap,layer1):

| | 纯 W2 GEMM | 融合 | 融合损失 | 重叠 |
|---|---|---|---|---|
| v0(docs/17) | 508µs | 794µs | +283µs(+56%) | ~0% |
| **v1** | 523µs | **715µs** | **+192µs(+37%)** | 传输大部分藏进 GEMM |

layer1 融合损失 283→192µs(−90µs);e2e −368µs(除 layer1 外还省掉 v0 的全卡
`pcie_device_barrier` 串行点 + 流水启动更早)。残留 +192µs 是尾部 push 无法与 GEMM 重叠
(源卡 final_reduce 必须等最后一个水位)+ prered 散射本身。

## 5. 后续

- 残留尾部:combine 与下一算子/下一层流水(persistent kernel,docs/11 §T9-6)可进一步消。
- v0(`TK_COMBINE=prered`)保留为无新协议的保底;`pull` 保留。
- 复现:`CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.validate_prered_push`
  和 `... bench_shape_4096`(默认已 v1)。
