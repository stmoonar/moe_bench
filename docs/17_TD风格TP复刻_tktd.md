# 17. TD 风格 TP 复刻(tktd): 用 TK/PK 原语按 Triton-distributed 逻辑重组 L0/L1

> 状态: 已实现, CPU preflight 全绿, **未上机**(开发机无 CUDA)。
> runbook 见 §7。定位是**归因 A/B 探针**, 不是性能路线(§5 的账)。

## 1. 动机

tdtp(Triton-distributed 原版移植)与 tktp 的 e2e 差距混着两个变量:

1. **引擎差异**: triton `tl.dot` vs 手写 TK gg8(docs/08 NCU 对比);
2. **调度差异**: TD 的"粗粒度段 gate + 双 stream chunk 追赶" vs tktp 的
   "per-token/per-行块细粒度 gate + 单 kernel 融合"。

tktd 把 TD 的调度策略搬到 TK 引擎上(数据面原语与 tktp 完全同源), 于是:

- `tdtp − tktd` ≈ 引擎差异;
- `tktd − tktp` ≈ 调度差异(段粒度 + chunk 化 + comm SM 不转岗)。

## 2. 结构对照(TD 机制 → TK/PK 等价物)

| TD(tp_moe c4) | tktd 实现 | tktp 对照 |
|---|---|---|
| NVSHMEM symmetric workspace | `TKParallelTensor`(IPC 对称缓冲) | 同 |
| producer stream 上的 AG(per-rank putmem + per-segment signal) | `moe_td_ag_producer`: 独立 kernel/stream, TMA 行 push(朴素 token 序) + **per-src 段到达 flag**(计数选举, `st.release.sys` 定值) | push 在融合 kernel 内, **per-token** flag, 消费序 push_order |
| consumer GEMM `dl.wait(segment_start..end)` | `moe_td_ag_gemm` 的 `seg_gate`: 行块真实行覆盖的 src 区间 `[blk_lo, blk_hi]` 逐段等本地 ready flag(**有界**自旋) | per-行块计数 gate(slack 种子) |
| threadblock swizzle(本 rank tile 先算) | `l0_row_perm`: 按(最晚就绪段 = 区间内 ring 距离最大值, 块 id)排 | pull_order 消费序 + TK_LOCAL_FIRST(docs/14, 逐 expert 精化版) |
| GEMM 内 sorted_token_ids 行间接(gather 不落地) | scatter 落地 gathered(TK TMA 要求 A 行连续; 归 producer kernel 的 scatter lanes, **散完整个 src 分片**才发 ready) | scatter 同款但 per-token 放行 |
| L1 GEMM per-N-chunk `gemm_done_flag` | dispenser 新增 **CHUNKED** 编译期模板参数(chunk-major 任务序) + `chunk_signal_epilogue`(per-tile 计数, 满 `nblk×chunk_cols` 发 `chunk_done[c]=seq`) | 行块就绪信号(signal_epilogue) |
| reduce_stream 上 reduce_topk + RS 逐 chunk 追赶 | `moe_td_reduce_rs`: 独立 kernel/stream, 等 chunk flag → top-k 加权归约该列窗 → **向量 st** 直推源卡 staging(与 TD 同为向量 store 数据面) | push_job 融合在 GEMM kernel 内, TMA 整行 8KB |
| 最终 RS 收端归约 | 复用 `moe_final_reduce_push`(watermark 协议不变) | 同 |

不采用的 TD 细节: NVSHMEM(PCIe 上 IPC+flag 等价且已有)、copy-engine AG
(docs/04 判负)、无界 `dl.wait`(违反红线 2, 全部换有界自旋)。

## 3. 调度表(每 run 重建, 计入耗时)

共享表(padded/tp_slots/prered_w/slack/blk_expert/slot_job/slot_w)复用
tk_tp_scheme 的 host golden + GPU builder(canonical 单段布局)。TD 专用表:

- `blk_lo/blk_hi (nblk,)`: 行块真实行的 src 区间。canonical 布局 expert 内
  按 (src_dev, src_tok) 升序 → 区间连续, min/max 即可;
- `l0_row_perm (nblk,)`: `argsort(stage*nblk + blk)`, stage = 区间内
  `(d - rank) % world` 的最大值;
- `pull_order (S,)`: `argsort(ring_dist(src)*S + j)` —— 本 rank 分片免等待
  先散(TD 的 rank_start 语义)。

host golden(纯循环) vs GPU 向量化版在 setup 逐元素对拍(不过直接 raise);
`tools/preflight_td_cpu.py` 在无 GPU 机器上裁决同一套表 + 不变量(区间
正确/双射/stage 沿 perm 非降/stage0 块只含本 rank 行/pull_order 前 T 项为
本 rank token) + CHUNKED 任务映射(双射 + 与嵌套循环枚举逐项相同) + RS
数据流仿真(chunk 列窗索引算术对拍直接参考)。2026-07-28 本机全绿。

sched 阶段 tktd 走 torch 向量化 builder(未接 tpsched 融合 kernel, 慢
~120µs); A/B 对照 tktp 时要么看去掉 sched 的分阶段数字, 要么给 tktp 设
`TK_SCHED_FUSED=0` 对齐口径。

## 4. run() 的流水与 stream 编排

```text
sched(graph) → barrier → quant(tok) → barrier → 清计数器 → ev0
  comm stream: wait(ev0) → moe_td_ag_producer(push lanes + scatter lanes)
  main  stream: moe_td_ag_gemm(seg_gate, grid = sm - TKTD_COMM_SMS)
main: quant(act)
  comm stream: moe_td_reduce_rs(grid = TKTD_RS_SMS, 等 chunk flags) → ev_end
  main  stream: moe_td_gemm_nchunk(grid = sm - TKTD_RS_SMS, chunk-major)
main: moe_final_reduce_push → wait(ev_end) → return
```

- 双 barrier 与 tktp 同协议: 覆写 pre_tokens/staging 前全 rank 汇合;
- 所有跨 kernel flag(段到达/段 ready/chunk_done/watermark)都是单调 seq 值,
  免清零; 计数器在 ev0 之前 main stream 清零, 消费者经 event/flag 链有序;
- SM 账: L0 = (sm−comm) GEMM ∥ comm producer; L1 = (sm−rs) GEMM ∥ rs。
  两对 kernel 网格之和 == sm, 与启动顺序无关都能全部驻留(无饿死)。
  final_reduce(grid = min(T, sm−2))在 RS 未退场时部分块排队 —— RS 不等
  final, 无环, 只是尾部串行化。

## 5. 预期与已判负形态的关系(上机前的账)

刻意复刻了两个本平台已判负的调度特征, **预期 tktd 慢于 tktp**:

1. **L0 粗粒度段 gate**: expert-major 布局下行块的行来自各 rank 交错,
   多数块的区间 `[lo,hi]` 覆盖多个甚至全部 rank → 排序靠后的 tile 等整个
   AG(docs/03 ring-by-source 教训的段粒度版)。stage0(纯本 rank)块的占比
   决定能藏多少 —— balanced T=512 下每 expert 512 行/卡均分, 纯本 rank
   块少; 预期 L0 exposure 明显大于 tktp 的 ~94µs。
2. **L1 N 维分解**: 即 docs/04 的 Comet-N(判负: PCIe push 耗 SM, 通信与
   GEMM 在 SM 维度零和)。且 RS 数据面是向量 st 小消息(chunk_elems×2B/行),
   nchunks 越大消息越小。
3. **comm SM 不转岗**: producer/RS kernel 干完不帮 GEMM(TD 形态), tktp
   的 comm 块会转岗领 GEMM task。

这些"税"正是要测量的对象——归因读法见 §1。若 tktd 意外不慢, 优先怀疑
tktp 的细粒度信号开销在当前形状下未摊销, 再查 nchunks/comm_sms 配比。

## 6. 死锁审计(红线 1, 逐等待点)

| 等待点 | 生产者 | 有界性 | 生产者永不到达的情形 |
|---|---|---|---|
| push_lane_td `guarded_wait`(TMA load 本地 mbarrier) | 本 lane 自己的本地 DMA | guarded_wait(时钟超时 trap) | 本地 DMA 不会消失; kernel trap 连带 |
| scatter_lane_td 段到达自旋(本地内存 acquire.sys) | 对端 rank 的 push 计数选举 lane | PCIE_SPIN_GUARD(~32s trap) | 对端崩溃/被 kill → trap → CUDA error → `_fail_fast_exit` |
| scatter_lane_td `guarded_wait`(TMA load 本地) | 本地 DMA | guarded_wait | 同上 |
| seg_gate 段 ready 自旋(本地, acquire.gpu) | 本卡 producer kernel 的 scatter 计数选举 | PCIE_SPIN_GUARD | producer 卡死(上游对端死)→ 本 gate trap; producer trap → 同 context 连带 |
| L1 RS chunk_done 自旋(本地) | 本卡 L1 GEMM 的 chunk_signal_epilogue 计数选举 | PCIE_SPIN_GUARD | GEMM trap → 同 context 连带; GEMM 卡死(不可能, no_gate 纯计算)→ trap 兜底 |
| final_reduce `wait_slot` watermark | 各 rank RS kernel 的 done_blocks 计数选举 | 既有有界 wait_slot | 对端 RS 死 → trap |
| pcie_device_barrier | 全 rank | 既有有界 | 对端死 → trap |
| dispenser 内部 mbarrier(inputs/task) | 同 kernel 生产者 | 裸 wait(红线允许: trap 连带杀死) | — |

信号生产者不可跳过: push_done/scattered_cnt/chunk_cnt/done_blocks 的计数
域都是确定的满额(T / T / nblk×chunk_cols / gridDim), 领取都走原子计数器
且映射是双射(preflight 断言); two-level 尾块的 epilogue 计数与满块完全
一致(伴走 warp 同节奏, 同 tppr8)。跨 rank 依赖链 barrier → push → scatter
→ L0 GEMM → quant → L1 GEMM → RS → watermark → final, 全部前向, 无环。
跨迭代覆写窗口由双 barrier 收口(同 tktp)。

## 7. 上机 runbook(红线 3: 首测单步隔离)

```bash
# 步 0(必须先跑, 默认就是这一步): 正确性门, 单步隔离, verify 强制开
python -m moe_bench.tools.run_tktd
# 步 1: 门 + 计时 + 同口径对照
python -m moe_bench.tools.run_tktd --perf --with tktp,serial
# 步 2: 门 + nchunks 扫描(TD 默认 4; 消息大小与 overlap 粒度的 tradeoff)
python -m moe_bench.tools.run_tktd --sweep --nchunks-list 2,4,8,16
# 不均匀路由判决档(memory: 判决口径以不均匀路由为准)
python -m moe_bench.tools.run_tktd --perf --dist uniform
```

先决: 卡 0-3 空闲(nvidia-smi), /workspace 下执行。首跑会触发 nvcc 重编
(tk_moe.cu 变更, ~分钟级)。异常处置(挂死/取证)按 docs/04、docs/06。

旋钮: `TKTD_COMM_SMS`(24) / `TKTD_PUSH_SMS`(4) / `TKTD_RS_SMS`(24) /
`TKTD_NCHUNKS`(4) / `TKTD_L0_NOGATE`(归因探针, 数值错) /
`TKTD_TWO_LEVEL`(1) / `TKTD_GPU_SCHED`(1)。

## 8. 改动面

- `sm120_common.cuh`: dispenser 加 `CHUNKED` 模板参数 + `chunk_cols` 尾参
  (默认 false/0, `if constexpr` 剪支, 既有 kernel 逐指令不变);
- `tk_moe.cu`: 新增 `tdag`(producer + seg-gate GEMM)、`tdrs`(chunk GEMM +
  reduce-RS)两个 namespace 与 4 个绑定; 既有 kernel 零改动;
- `tk_td_scheme.py`(新, `--scheme tktd`)、`schemes.py`(注册)、
  `tools/run_tktd.py`(驱动)、`tools/preflight_td_cpu.py`(CPU 裁决)。
