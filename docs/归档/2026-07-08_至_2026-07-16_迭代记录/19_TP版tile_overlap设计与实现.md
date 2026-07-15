# 19 TP 版 tile overlap 设计与实现（tktp scheme，分支 tp_test）

> 基于 experience/08（AG+GEMM、GEMM+RS 路线）、11 §4（MoE TP/EP 的 TK 实施方案）、
> 12 §6（SM120+PCIe 形态修订）落地的 **TP 模式通算融合 MoE**。与已验证的 EP 版
> （tkfused，docs/01~18）共享 kernel 基建，本文只写 TP 特有的部分。
> **状态：本地静态验证完成，尚未上机编译/运行**（见 §6 风险清单）。

## 0. TP 与 EP 的结构差异决定了方案形态

TP 切分的是 **intermediate 维**（每卡持有全部 E 个 expert 的 1/world 薄片），
EP 切分的是 expert。这带来三个决定性差异：

| | EP（tkfused） | TP（tktp） |
|---|---|---|
| layer0 通信 | 按路由把 token 送到 expert 所在卡（数据量依赖路由，(token,dst) 有重复 → T7 去重是"优化"） | **稠密 AllGather**：每卡需要全部 world×T 个 token；token 的全部 top-k expert 都在本卡 → 每个唯一 token 恰好被本卡用 topk 次，**去重是天然形态不是优化** |
| layer1 top-k 归约 | 跨卡（token 的 8 个 expert 分布在多卡）→ 预归约只能按 expert 卡分组（T6） | **完全本地**（8 个 expert 全在本卡）→ 预归约一步到位，每 (src,tok) 一行 |
| layer1 跨卡通信 | 稀疏 gather/push（依赖路由） | **稠密 ReduceScatter**：每卡给每个源卡推恰好 T 行 partial |
| 调度表跨卡一致性 | dispatch slot 必须与 expert 卡对齐（ring 回放） | **纯本地**：gathered 布局各卡自选，跨卡只走稠密 (src_dev, src_tok) 索引 |

结论：TP 的通信协议比 EP 更简单、更规则（全稠密、路由无关），
在 PCIe 上是 experience/08 §1/§2 的教科书场景。

## 1. 数据流（bf16，默认 shape E=64/topk8/H=4096/I_shard=768，T=512/卡）

```text
1. AllGather ⊕ gate+up GEMM      moe_tp_dispatch_gemm（单 launch 融合）
     comm block：每唯一 (src_dev,src_tok) 拉一次（ring 序，本卡 shard 先）
       → TMA 散播到它的 topk 个 expert-sorted gathered slot（本地）
       → 每 slot 给所在 row block 计数器 red.release.gpu +1
     comp block：grouped GEMM，gate 自旋等 row block 计数 == ROW_BLOCK
       （padding 用 slack 预置种子，dpush 的老技巧，但这里纯本地）
2. act = silu(gate)*up            torch（up 的计算藏在 AG 通信下，T4 同款）
3. W2 GEMM ⊕ prered ⊕ push       moe_tp_gemm_prered_push（复用 preredpush kernel）
     job j=(s,t)：等 8 个 slot 的 W2 row block 本地完成信号
       → FP32 加权和成一行 partial → TMA 推到源卡 s 的 staging 平面 [my_rank]
       → 本地选举计数到 T 后 st.release.sys 水位信号（T6-v1 协议原样）
4. final reduce                   moe_final_reduce_push（复用）
     源卡每 token 等 world 个水位 → 累加 world 个平面的行
```

正确性恒等式（本地 CPU 模拟已验证，见 §5）：

```
out[t] = Σ_d Σ_k w(t,k)·(act_d(t,e_k) @ W2_d[e_k])     d 为 I 维分片
       = Σ_k w(t,k)·(act(t,e_k) @ W2[e_k])              与 serial TP 参考一致
```

## 2. 复用矩阵（kernel 侧只加了 ~200 行）

| 组件 | 来源 | 变更 |
|---|---|---|
| `grouped_gemm_sm120` 模板 | sm120_common.cuh | 零改动（expert_offset=0、num_local_experts=E 本来就是参数）|
| TP dispatch（`tpdisp::`）| 新增，disp::/ddisp:: 的杂交 | pull-once-scatter-topk + slack 种子 + reset-to-slack |
| layer1 GEMM⊕prered⊕push | `preredpush::` 原 kernel | 只加 TP 入口 `gemm_push_entry_tp`（expert 几何 + expert_out 用普通 tensor）|
| final reduce | `moe_final_reduce_push` | 零改动（final_contrib/recv_from 全 1）|
| pcie 同步原语 | pcie_sync | 零改动 |
| GPU 调度重建（T3 公平口径）| 新 `_build_tp_schedules_gpu` | 单 argsort，比 EP 简化（无 trash-row：全部 assignment 都是本地的）|

## 3. 调度表（tk_tp_scheme.py）

**一张表干两层的活**：`tp_slots (world*T, topk)`，行 j = src_dev*T + src_tok，
列 = kpos，值 = gathered slot。layer0 用它散播 token，layer1 用它回读 W2 输出
（EP 里这是两张要跨卡对账的表；TP 里布局纯本地，天然自洽）。

- 槽位序：expert 内按 (ring, src_tok, kpos)，ring = (src_dev − rank) % world
  —— 本卡 shard 排最前，GEMM 首波只等本地拷贝（experience/03 swizzle 经验）。
- `slack (nblk,)`：ROW_BLOCK − 块内真实 token 数。dispatch 计数器的预置种子，
  kernel 尾部 reset-to-slack（不是清零），Python 侧只在 setup 播一次。
- layer1 表全是路由无关常量：`push_expected_l1 = T`、`recv_from/final_contrib 全 1`、
  `prered_dst = (j//T, j%T)`——setup 设一次，run() 里不重建。
- run() 每迭代重建 padded/tp_slots/prered_w/slack（all_gather 路由 + CUDA graph
  回放 argsort builder），对齐 serial 每次 run 付路由元数据的公平口径（docs/07 P1、docs/14）。

## 4. 通信/计算账（预估，待实测校准）

每卡每迭代（bf16、T=512、H=4096）：
- L0 AG 拉入：3/4 × 2048 token × 8KB ≈ **12.6MB**（弱路径 ~20GB/s → ~630µs，
  与 gate+up GEMM ~1.5ms 重叠 → 应完全隐藏）
- L1 RS 推出：512 × 3 卡 × 8KB ≈ 12.6MB（强路径 ~50GB/s → ~250µs，
  边算边推藏在 W2 GEMM ~0.7ms 下）
- GEMM 总量与 EP 完全同 FLOPs（16384 行薄片 vs 4096 行全片），公平可比。
- EP 对照（docs/18）：tkfused 1963µs vs serial 2905µs。TP serial 的 AG/RS 是
  NCCL 稠密集合通信（比 EP 的路由 gather 高效），预期 serial 更快、
  融合收益比率略小；tktp 目标 = 明显快于 TP serial。

## 5. 已做验证（本地，无 GPU）

1. **调度表裁决**（scratchpad test_tp_sched.py = tools/verify_tp_schedule.py 同款）：
   host golden vs GPU builder 逐元素相等 + 5 项不变量（槽位双射、expert 区间、
   slack 恒等式 slack+real==ROW_BLOCK、ring 序），NE∈{64,128,256}×{balanced,skewed}×4 rank 全过。
2. **数据流语义模拟**（test_tp_dataflow.py）：用真实调度表在 CPU 上完整模拟
   dispatch scatter → grouped GEMM → prered → push 平面 → final reduce，
   对拍朴素 TP MoE 参考，fp32 rel_err 3.7e-7。

## 6. 上机风险清单（一键脚本按此排查）

- [ ] `tpdisp::` 新 kernel 的 nvcc 编译（designated initializer 顺序已静态核对）。
- [ ] 每 comm 线程 1 拉 + 8 散 + 8 计数的 TMA 序（store_async_wait 后 red，
  与 disp:: 同构，但扇出 8 倍，注意首次跑对拍是否有行错位）。
- [ ] slack 种子 + reset-to-slack 的迭代间恒等（若 verify 首迭代对、后续错 → 查这里）。
- [ ] NE=256 时 P=32768：gathered/expert_out 各 256MB，显存 ~1GB 级，48GB 无虞
  但注意 4 进程同机总量。
- [ ] 水位/选举协议直接复用 T6-v1，若 layer1 偶发错 → 先跑 EP 的
  validate_prered_push 确认平台，再怀疑 TP 入口。

## 7. 运行方式

```bash
# 一键（自动挑空闲 4 卡组、编译、裁决、对拍、bench、打包 zip）
bash moe_bench/tools/run_tp_all.sh          # QUICK=1 只跑核心步骤
# 手动
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.verify_tp_schedule
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.run_tktp 64
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.run_tktp 64 --scheme serial --no-verify --iters 50
```

环境开关：`TK_COMM_SMS`（默认 16，experience/12 建议 PCIe 上从 2~8 起 sweep，
脚本内含 4/8/16/24 sweep）、`TK_GPU_SCHED`（默认 1）、`TK_ROW_BLOCK`（须与编译一致）。
