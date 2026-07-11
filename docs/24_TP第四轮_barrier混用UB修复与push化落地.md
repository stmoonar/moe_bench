# 24 TP 第四轮：barrier 混用 UB 修复 + dispatch push 化落地（TP-T1）

> 第四轮（tp_run_20260711_133751）归因与本轮改动。承接 docs/20（调度修复）、
> docs/23（计划）。同轮完成 EP 口径修复（docs/22 P1）。

## 1. 第四轮结果与归因

**现象**：CPU 预检、verify_tp_schedule（表裁决）全过；所有 tktp 运行时步骤
死于 `cudaErrorIllegalInstruction`（首个 kernel 执行即挂）；serial 全过
（2624µs 复现稳定）。

**根因（代码审出，非猜测）**：`gemm_push_kernel_tp` 里 comp block 跑完
GEMM 后用 `__syncthreads()` 做全块汇合——但 `__syncthreads` 是**硬件
barrier 0 按 288 线程计数**，而 `grouped_gemm_sm120` 内部 consumer group
一直在用 `bar.sync 0/1` **按 256 线程计数**（producer warp 不参与）。
producer warp 的 lane 1..31 在 GEMM 函数里**立刻退出**（只有 lane 0 干活），
直接撞进我的 `__syncthreads` → 同一个硬件 barrier 上并发存在两种到达计数
= UB，Blackwell 上表现为 illegal instruction。EP 的 job 块从不跑 GEMM，
所以老 kernel 无此问题——**“角色混跑”的常驻块设计引入了新的 barrier
资源冲突面**。

**修复**：汇合改用**专用命名 barrier**（`bar.sync 2, NUM_THREADS`），
与 GEMM 占用的 0/1 隔离；汇合完成后 barrier 0 空闲，push_job 内部的
`__syncthreads` 恢复安全。

**经验（追加到踩坑清单）**：persistent kernel 里让计算角色收工后转任何
其它角色，必须先盘点两侧用到的 named barrier（`group<N>::sync(id)` 的 id
与线程数），汇合点用未占用的 id + 全块计数；`__syncthreads` 不是中性的，
它就是 bar 0。

## 2. 本轮落地：TP-T1 dispatch push 化（docs/23 主攻项）

依据 docs/22 §0：pull 弱路径（4 卡并发 23.5GB/s、16 SM），push 强路径
（50.9GB/s、4 SM、并发零退化）。TP 的 AG 稠密且路由无关，push 形态天然成立：

- **canonical 布局（前置）**：tp_slots 的 expert 内序从 per-rank ring 改为
  (src_dev, src_tok, kpos)（全卡一致）。push 的水位位置映射要求生产/消费
  两侧对 push_order 逐字节一致，per-rank 布局做不到；“本卡优先”反正已被
  docs/20 证明无用。GPU builder 的 key 简化为 `eid * N + n`。
- **push_order (world, T)**：每源卡自己的 token 按 min-slot 排序 = 消费序。
  host/GPU 双实现；裁决加了三条不变量（每行置换、按 min-slot 有序、
  跨 rank canonical——verify_tp_schedule 用 broadcast 对比 rank0）。
- **新 kernel `tppdisp`（moe_tp_dispatch_push_gemm）**，单 launch 三角色全常驻：
  - comp：grouped GEMM，gate 不变（row-block 计数 == ROW_BLOCK，slack 种子）；
  - push（默认 4 块）：把本卡 shard 按 push_order TMA 推到每个 peer 的
    ag_staging 平面 [me]（单写者、无原子），每 CHUNK=64 行按目的卡做
    本地 acq_rel 选举 → `st.release.sys` 水位（barrier_l0 行 2+s 列 chunk，
    seq 单调免复位）——完整镜像 preredpush 已验证的内存序链；
  - scatter（默认 comm−push 块）：按 (pos, src) 到达序消费，远端源等
    chunk 水位、**本卡 shard 直读 pre_tokens（消掉自依赖）**，行拷到
    topk 个 gathered slot + red.release.gpu 计数。
  - 死锁审计：scatter 只 spin 远端水位（由 peer 的常驻 push 块供给）；
    push 块无任何 spin；全部块常驻（grid = SM 数）。
- scheme：`TK_TP_DISPATCH=pull(默认)|push`、`TK_TP_PUSH_SMS`（默认 4）；
  ag_staging (world*T, H) 仅 push 模式分配。
- 脚本：push 正确性（NE64/256+skewed）、push bench、(push_sms, comm_sms)
  组合扫描（4/8, 4/12, 4/16, 2/8）、push 版分阶段归因（step 08p）。

## 3. 同轮完成的口径修复（EP，docs/22 P1）

`tk_scheme.py` run() 的 GPU schedule 重建条件 `== "prered"` →
`in ("prered", "prered_push")`：默认 combine 是 prered_push，此前默认路径
e2e **漏计 ~205µs 的 schedule 成本**。fair 口径：EP 默认 shape
1963→**~2170µs（1.34×）**，与 microbench 的公平口径一致；HANDOFF 已改。

## 4. 第五轮期望读数

1. pull 路径（docs/20 修复首次真正执行）：e2e、comm_sms sweep 形态、
   08 分阶段（L0/L1 相对 GEMM-alone 的暴露量）；
2. push 路径：对拍三档 + e2e + (p,c) 扫描 + 08p 分阶段——对数 docs/23 §2
   的预期（L0 ≈ 1.55ms、e2e ~2.4-2.6ms）；
3. 若 push 全档正确且全档快 → 默认切 push，下一步 TP-T3（triton 调优基线）。
