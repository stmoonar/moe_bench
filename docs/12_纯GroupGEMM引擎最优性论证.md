# 纯 Group GEMM 引擎（gg8）最优性论证

日期：2026-07-27。对象：`gg8::kernel`（`tk.grouped_gemm_fp8`，
`tools/verify_fp8_gemm.py` 所测）——融合 kernel 同一套 `gemm_config_fp8`
引擎的纯 GEMM 形态：`no_gate + noop_epilogue + plain_store_policy`
（tk_moe.cu:62），无 GLU / 路由 scatter / 通信 lane，与 CUTLASS 87c 的
纯 GEMM 口径逐项对齐（同 MMA atom、同量化语义、同重标定频率）。

## 1. 结论

1. **同类实现中最优（已坐实）。** 同类 = 路由感知（per-expert grouped +
   间接调度）+ blockwise FP8（act 1×128 / weight 128×128 + fp32 重标定）+
   **可通算融合**（persistent + dispenser 任务分发，作为融合 kernel 的
   consumer 引擎原样复用）。唯一同类对手 vLLM triton fused_moe 两层皆慢
   （L0 慢 8%、L1 慢 26%，§2.2），且其 tile config 经调优扫描（E3，320
   候选）确认已是最优水位。其余候选不存在：DeepGEMM 无 sm120 路径
   （wgmma/tcgen05 本卡都没有）；cuBLASLt 无 blockwise grouped fp8 口径；
   CUTLASS 87c 非同类（纯 GEMM、无路由/GLU、非 persistent）——是上界
   参照，不是可替换实现。
2. **未达厂商纯 GEMM 上界，不做"已达上限"表述。** 同卡 boost 下
   L0 76.8% / L1 88.2%（§2.1）。
3. **差距有账、杠杆在册。** L0（锁频 110.6µs）= 重标定机制 70% +
   结构性 30%，杠杆：重标定切片交织（头号）、任务边界 store 后置
   （~20µs）；4×2+COL=128 几何已被 E4 判负（§5）。L1 的 11.8% 来源
   （流量 vs 调度）待 E5。便宜旁路已全部实验判负（§4）。

## 2. 数据

### 2.1 gg8 vs CUTLASS：同卡 boost wall-clock（2026-07-27，正式横比，卡号待补）

| 层 | gg8（TFLOP/s） | CUTLASS 87c（TFLOP/s） | gg8/CUTLASS |
|---|---:|---:|---:|
| L0 (K=4096, N=1536) | 659.4µs (312.7) | **506.1µs (407.3)** | 76.8% |
| L1 (K=768, N=4096) | 357.8µs (288.1) | **315.7µs (326.5)** | 88.2% |

时钟缩放互证：锁频→boost，CUTLASS 加速 1.31×（= 时钟比，纯算力受限），
gg8 仅 1.17× → 差距主体是**固定延迟 stall**（不随频率缩放），与 NCU
stall_wait 第一名（docs/11 §3）一致——重标定切片交织正是吃这块的杠杆。

### 2.2 gg8 vs triton：NCU 锁频（kernel replay，triton 为调优水位）

| L0 | Duration | tensor | Duration×tensor |
|---|---:|---:|---:|
| gg8 fp8 | 774.6µs | 71.2% | 551µs |
| gg8 raw（无重标定探针） | 697.7µs | 79.4% | 554µs |
| triton fused_moe | 834µs | 66.7% | 556µs |

L1 锁频：gg8 425.0µs/64.2% vs triton 537µs/50.2%（triton 受两颗 GEMM
共用一个 config 的结构上限，docs/08 §6）。

- **tensor-busy 不变量**：同形状下 Duration×tensor% ≈ 常数——QMMA 总量
  与速率各实现完全相同，差距 100% 是 tensor 空转，"更快的乘法"不存在，
  最优化问题收窄为消空转。
- **平台边界对所有实现同等生效**（docs/04/10）：fp32 累加税（fp8 峰值
  减半）、smem ~100KB、寄存器 168 帽（TMA ABI 调用帧）。CUTLASS 的
  407.3 TFLOP/s（boost）即边界下的实测天花板。
- 正确性门：rel_err 1.68e-03（fp32 反量化参考，输入逐 bit 相同）。

## 3. 结构现状：每一项都是一个已关闭的差距

| 结构点（现状） | 关闭的差距（出处） |
|---|---|
| QMMA SM120_16x8x32 fp8 + fp32 累加 | 与 CUTLASS/triton 同 atom，指令选型无差异（08 §3） |
| K-tile = 128 = 量化块 | 旧 64 深 stage 的 2× mbarrier/LDSM 冗余（10 §1） |
| 重标定每 128 K 一次 | 与 CUTLASS 87c blockwise promotion 同频同数学（10 §1） |
| A/B 全双缓冲 + arrived wait 提前至末尾 QMMA 前 | load→use 串行暴露 LDSM 延迟（10 §3） |
| COL_BLOCK=64，consumer ~156 reg 零 spill | 168 寄存器帽下 128 宽 tile 必 spill；E4 定案帽与 syscall 无因果，COL=64 平台强制（§5 E4、10 §7） |
| 4 stage × 24KB = 96KB ≤ 99KB | sm120 smem 帽下最深流水（08 §6） |
| TMA load `.shared::cta` 原生形态 | `.shared::cluster` 驱动 syscall，109 个 CALL.ABS→0（11 §6-7） |

## 4. 已判负的旁路（没有便宜的翻盘路）

| 旁路 | 结果 | 出处 |
|---|---|---|
| TASK_Q 2→4（producer 跑道） | +0.2~0.6% 变慢，判负 | 11 §5 |
| EPIRED（epilogue 直推归约） | v2 向量原子仍输 108µs，L2 fp32 原子吞吐是硬墙 | docs/04 §2 |
| scale 走 smem 流水 | stall 无 LDG 记分牌 signature，降级观察 | 11 §9 |
| 全宽 B 双缓冲 / setmaxnreg | ptxas spill 4.7× 回退 / C7506 全数忽略 | 10 §2, §6 |
| 4×2 warp 几何 + COL=128 | E4：CALL.ABS=0 后 ptxas 仍 168 帽，COL=128 四个大 kernel 全 spill，判负 | §5 E4、10 §7 |
| L1 引擎优化曾判"关闭" | 重开：CUTLASS 同流量快 11.8%，待 E5 定方向 | 11 §2 |

## 5. 实验状态

> 红线：先确认卡空闲；NCU 严禁与 4 卡任务共存（docs/08 §4）。E4 不占 GPU。

- **E3 triton 调优 ✅ 完成**：调优扫描未找到优于现役 config 的配置，
  triton 数字即其最优水位，结论 1 坐实，e2e 报数无需重报。
- **E4 ✅ 完成（2026-07-27）**：COL=128/stages=3 重编（build.py 无
  `-maxrregcount`，168 非自设），在 CALL.ABS=0 前提下四个大 kernel 仍
  全部被压 **168 reg 且 spill**（gg8::kernel 504B stack/816 spill
  stores，kernel_push 448B，tppr8 504B，kernel_raw 56B；硬上限本是
  224=65536/288 线程）。**定案**：① 168 帽与 ABI call 无因果——去
  syscall 不解除，"224−56 调用帧"只解释数值来源；② **4×2+COL=128
  判死，COL=64 是平台强制最优**；③ 已还原重建（L1 复测 371.6µs、
  rel_err 1.68e-03 证实现役 .so 为 COL=64）。
- **E1（待跑）**：cta 补丁后 NCU 锁频重定基，
  `bash moe_bench/tools/probe_engine.sh <空闲卡>`，§2.2 数字更新，
  rel_err 须仍 1.68e-03。
- **E5（待跑，已升优先级）**：L1 差距来源裁决——tk 与 CUTLASS 各采一次
  `dram__bytes.sum`（L1 形状）。字节相近 → 差在延迟/调度（tk 可修）；
  CUTLASS 更少 → 差在 L2 复用/栅格化顺序。命令（在 /workspace/work 下；
  注意必须带 `--kernel-name-base demangled`，否则 regex 匹配不到命名空间，
  会报 "No kernels were profiled"）：

```bash
CUDA_VISIBLE_DEVICES=<空闲卡> ncu --kernel-name-base demangled --kernel-name 'regex:gg8' \
  --launch-skip 4 --launch-count 1 --metrics dram__bytes.sum \
  python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 10
CUDA_VISIBLE_DEVICES=<空闲卡> ncu --kernel-name-base demangled --kernel-name 'regex:cutlass' \
  --launch-skip 4 --launch-count 1 --metrics dram__bytes.sum \
  moe_bench/tools/cutlass_probe/build/cutlass_fp8_grouped \
  --groups=64 --m=256 --n=4096 --k=768 --iterations=10
```

- **E2（可选）**：CUTLASS 当前卡 NCU 锁频 tensor%，让 §2.2 不变量分解
  整体换到同卡口径。

结果回填本文对应小节；单次实验参数照例记入结果目录与 HANDOFF。
