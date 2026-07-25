# fp8 GEMM 主循环 P1：K-tile 128 改造与寄存器预算事故

日期：2026-07-25。背景：NCU 锁频对比（docs/08 §5）gg8 924µs/60.9% tensor vs
CUTLASS 87c 664µs/84.9%（同 MMA atom SM120_16x8x32_TN fp8 fp32acc、同 8
consumer warp、同 3 stage、同 99KB smem 预算、重标定同为每 128 K 一次）。
差距定位为软件流水结构欠账，P1 进行改造。

## 1. CUTLASS 87c 主循环的结构要点（我们原先的差距）

| 维度 | 旧 gg8 | CUTLASS 87c |
|---|---|---|
| K-tile / stage | 64（量化块的一半） | 128（==量化块，static_assert 强制） |
| consumer warp 几何 | 1×8（每 warp 16×128） | 4×2（每 warp 32×64，B 复用 ×2） |
| LDSM / warp / K-block | 36 | 24 |
| LDSM→MMA | load→use 串行，每 64-K 被 mbarrier wait 打断 | 寄存器双缓冲：kk+1 的 LDSM 先于 kk 的 QMMA 发射 |
| stage 边界 | wait 在头部，无法跨 stage 预取 | consumer_wait 放在末尾 QMMA 之前，下一 stage LDSM 与末尾 QMMA+rescale 重叠 |
| scale 供给 | consumer 3 个 LDG（L2，手动预取） | producer cp.async 进 smem，主循环零 gmem |

修正 docs/08 §5 的表述：两边重标定**频率相同**（每 128 K），+31% 指令
来自 64 深 stage 的 2 倍 mbarrier 事件 + LDSM 冗余 + sub 清零 + dispenser，
**以及更要命的 load→use 串行在每 64-K 边界暴露 LDSM 延迟**（每 SMSP 只有
2 个 consumer warp，相位被同一 mbarrier 锁定，stall 互相关联）。triton 恰
好也是 BLOCK_K=128 + 编译器软流水——这正解释 CUTLASS > triton > TK 排序。

## 2. 事故：全宽 B 双缓冲触发 ptxas spill（首测 4.7× 回退）

P1 第一版把 consumer 改成全宽寄存器双缓冲（`a_reg[2]` + `b_reg[2]`，
b_reg = rt_fp8e4m3<128,32> = 32 regs ×2）。寄存器峰值 ≈ acc 64 + sub 64
+ b 64 + a 8 + misc ≈ 215+，超过 ptxas 舒适区（288 线程硬上限 224）——
**ptxas 进入 spill 模式，只分 168 regs，把 acc/sub 两个累加器（128 regs）
扔进 local memory**（它认为累加器"冷"，每 stage 只在 rescale 时用一次；
实际上每 stage 都要写+读，local 流量每 K-block 每 warp ≈32KB，4× 于
tensor 周期预算）。

首测数据（rel_err 1.68e-3 正确，纯性能事故）：

| 信号 | 旧版 | P1 全宽双缓冲 | 判读 |
|---|---|---|---|
| raw 探针（无 rescale） | 311 TFLOP/s | **325 TFLOP/s** | 流水结构本身是对的（raw 无 acc，寄存器够） |
| fp8 主路径 | ~283 | **69.5** | rescale 每 stage 对 spill 的 acc/sub 做 LDL/STL |
| cuobjdump gg8::kernel | — | REG:168 **STACK:520**（raw 64） | acc+sub 512B/thread ≈ 520B stack，实锤 |
| NCU L0 Duration | 924µs | 4300µs | 4.7× 回退 |
| NCU Memory Throughput | 46.8% | **90.6%**（DRAM 仅 11%） | local/L1TEX 管道打满，不是 DRAM |
| NCU 指令数 | 208.5M | 332M（+59%） | 每 FFMA 变 LDL+FFMA+STL 三连 |
| IPC | 1.26 | 0.42 | 内存管道饱和挤占发射 |

**识别信号清单**（下次再遇同类问题直接对照）：cuobjdump 看 `STACK`/`LOCAL`
暴涨；NCU 的 SOL `Memory Throughput` 高而 `DRAM Throughput` 低（说明是
local/smem 而非 HBM）；IPC 崩 + 指令数大涨；raw（无累加器变体）正常而
带累加器路径塌。

## 3. 修复：A 双缓冲、B 单缓冲 + wait 提前（最终落地版）

1×8 warp 几何下每 warp 必须覆盖全 128 列（GLU gate/up 寄存器内配对的硬
约束），全宽 B 双缓冲的 64 regs 预算不存在。折中：

- **A 双缓冲**（rt<16,32> = 4 regs ×2，便宜）；
- **B 单缓冲**：kk+1 的 B LDSM 紧跟 kk 的 16 条 QMMA 之后发射——WAR 由
  程序序保证安全（in-order issue，QMMA 发射在前），LDSM ~30cyc 延迟由
  QMMA 群掩护；
- **下一 stage 的 arrived wait 仍提前到末尾 QMMA 之前**（CUTLASS 序，
  mbarrier 延迟全遮蔽，这是 CUTLASS 主循环最关键的一手）；
- finished arrive 保持在末尾 QMMA 之后（此时该 stage 的 LDSM 已全部被
  QMMA 消费，smem 可读覆；我方 per-warp arrive 计数，不需要 CUTLASS 的
  named barrier）。

峰值 ≈ 190 regs，回到旧版无 spill 水位。重标定点（每 128 K）与块内 MMA
顺序（K 升序）与旧版完全一致 → **数值逐比特等价**。

## 4. 后续杠杆（按 ROI）

1. **4×2 warp 几何**（CUTLASS 布局）：b frag 32→16 regs，全双缓冲可行，
   LDSM −33%/warp——但 warp 只覆盖 64 列，GLU 的 gate/up 配对跨 warp，
   需要权重交织粒度从 64 列改 32 列（动重量化管道），深改。
2. **scale 走 smem 流水**（P2）：producer cp.async 每 stage 带 128 个
   a_scale + 1 个 w_scale，consumer 零 gmem，与 CUTLASS 对齐。
3. bf16 引擎同构移植（CUTLASS bf16 92.3% 说明同样有 headroom，非主线）。

## 5. 复测口径

单卡：`verify_fp8_gemm 64 256 4096 1536 10`（NCU 同 docs/08 §1 命令）+
`cuobjdump --dump-resource-usage` 确认 STACK 回落；预期 L0 tensor
60.9% → 70%+。全宽双缓冲版数据留档于本文 §2，勿复跑。

## 6. setmaxnreg 路线判负：本平台 TMA 是驱动 syscall

P1 修复版（A 双/B 单）仍慢：NCU 1.36ms vs 旧 924µs，REG:168 + STACK:216
残留 spill。排查链：

1. `regs_per_mp = 65536`（288 线程静态上限 224），168 不是硬顶；
2. CUTLASS 87c 探针也是 **REG:168 STACK:0**——人家结构正好塞下；
3. 试 setmaxnreg（CUTLASS cooperative 同手法，producer dec 64 /
   consumer inc 232）→ ptxas 报 **C7506: 'setmaxnreg' ignored to
   maintain compatibility into 'extern' call**，全数忽略；
4. `__forceinline__` dispenser 无用——SASS 里 `CALL.ABS.NOINC` 的
   目标是 **`__cuda_syscall_cp_async_bulk_tensor_{4d,5d}_tile_unicast`**：
   **本平台（sm_120 工作站卡）的 TMA bulk tensor 拷贝是驱动 syscall
   实现的**（每个 `tma::load_async/store_async` 都是一次 ABI 调用）。

结论：ptxas 对含 ABI call 的 kernel 预留 ~56 regs/thread 调用帧 →
**有效寄存器上限 = 224 − 56 = 168**（分毫不差），且 setmaxnreg 对本
平台任何用 TMA 的 kernel 都是死路。CUTLASS 能在 168 内零 spill，我们
128 宽 tile 需求 ~230 塞不下。唯一的出路是把需求压进 168。

另注：TMA syscall 意味着每次 TMA 调用有 ~8 条指令 + syscall 延迟的
额外开销（CUTLASS 的 2D descriptor 可能走原生路径——后续杠杆之一：
A/B 改 2D TMA descriptor 绕开 4d/5d syscall）。

## 7. P2：COL_BLOCK = 64（把需求压进 168）

acc+sub = 128 regs 是 128 宽 tile（8 warp × 16×128）的数学下限，无解；
COL=64 后 acc/sub 各 32，A/B 全双缓冲（b 16×2）总需求 ~135 ≤ 168，
**任何 cap 理论下都安全**，且恢复了 P1 原设计的全双缓冲主循环。

配套改动：
- `gemm_config_fp8`：COL_BLOCK 128→64，PIPELINE_STAGES 3→4
  （stage = A 16KB + B 8KB = 24KB，4×24 = 96KB ≤ 99KB）；
- store policy 模板化（`operator()<COL>`，bf16=128 / fp8=64 共用）；
- GLU 权重交织 `[gate64|up64]`→`[gate32|up32]`（同一 128 列 scale 块内
  置换，量化/scale 布局零改动；`tk_tp_scheme.py`）；
- `w_scales` 块索引 `col_idx >> 1`（128 列 scale 块含两个 64 列 tile），
  三个 entry 的 w_scales 尺寸检查同步按 SCALE_K 修正；
- 协议侧零改动：`signal_epilogue` 的 col_blocks 在 kernel 内按
  `weights.rows()/cfg::COL_BLOCK` 现算，自动一致；任务数翻倍由
  dispenser 吸收。

代价（记录在案）：任务数 ×2（L0 3072 个），epilogue 信号 ×2，TMA
syscall 次数 ×2（每次 stage 2 个），A tile 的 L2 复用压力略增。
