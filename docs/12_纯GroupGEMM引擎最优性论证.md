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
3. **差距有账，但主杠杆已实验收窄。** L0 锁频差 = 重标定机制 **107µs**
   （fp8−raw 实测）+ 结构性 ~24µs（raw vs CUTLASS 旧锚点）。**重标定
   切片交织（P3）已判负**（源码级交织不优于 ptxas 自身调度，两版实测
   反向 +5~10µs，docs/04 §2）；4×2+COL=128 几何已被 E4 判负。在册剩余：
   任务边界 store 后置（~20µs）；引擎侧重心转向 uniform padding 税
   （两级 tile，docs/09 §4，预期 −150~180µs，量级大于引擎残差）。L1 的
   10.1% 已由 E5 裁决为**延迟/调度**（DRAM 流量两边相同），同受 P3
   判负约束。便宜旁路已全部实验判负（§4）。

## 2. 数据

### 2.1 gg8 vs CUTLASS：同卡 boost wall-clock（2026-07-27，正式横比；gg8 = probe_20260727_040652，GPU0，×3 稳定，cta 补丁后）

| 层 | gg8（TFLOP/s） | CUTLASS 87c（TFLOP/s） | gg8/CUTLASS |
|---|---:|---:|---:|
| L0 (K=4096, N=1536) | 649.1µs (317.6) | **506.1µs (407.3)** | 78.0% |
| L1 (K=768, N=4096) | 351.0µs (293.7) | **315.7µs (326.5)** | 89.9% |

raw（无重标定探针，boost）：L0 580.7µs（重标定代价 +11.8%）、
L1 330.7µs（+6.2%）。时钟缩放互证：锁频→boost，CUTLASS 加速 1.31×
（= 时钟比，纯算力受限），gg8 L0 仅 1.19×（774.6 cta 前锁频→649.1）→
差距主体是**固定延迟 stall**（不随频率缩放），与 NCU stall_wait 第一名
（docs/11 §3）一致——重标定切片交织正是吃这块的杠杆。

### 2.2 gg8 vs triton：NCU 锁频（kernel replay；gg8 = cta 补丁后 probe_20260727_040652，triton = 调优水位）

| L0 | Duration | tensor | Duration×tensor |
|---|---:|---:|---:|
| gg8 fp8 | 795.4µs | 68.7% | 546µs |
| gg8 raw（无重标定探针） | 688.4µs | 80.9% | 557µs |
| triton fused_moe | 834µs | 66.7% | 556µs |

L1 锁频：gg8 fp8 438.4µs/61.7%、raw 390.7µs/69.7% vs triton
537µs/50.2%（triton 受两颗 GEMM 共用一个 config 的结构上限，docs/08 §6）。

cta 补丁的锁频形态：raw 改善（697.7→688.4，syscall 移除兑现在结构侧），
fp8 反而略升（774.6→795.4）——load 瓶颈拆掉后**重标定链成为更裸的关键
路径**（fp8−raw 从 76.9 涨到 107.0µs），交织的账面收益进一步变大。
不变量（L0 546-557µs / L1 270-272µs）跨 session 复现，分解框架有效。

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
| P3 重标定切片交织 | 两版实测反向 +5~10µs：源码级交织不优于 ptxas 调度，只顶穿 168 帽；已 revert | 04 §2 |
| L1 引擎优化曾判"关闭" | 重开：E5 定案差距在延迟/调度（流量相同），杠杆与 L0 同类 | 11 §2、§5 E5 |

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
- **E1 ✅ 完成（2026-07-27，GPU0，probe_20260727_040652）**：boost ×3
  极稳——L0 fp8 649.1µs/raw 580.7（重标定 +11.8%）；L1 fp8 351.0/raw
  330.7（+6.2%）；rel_err 全部 1.68e-03。对比 cta 前基线（L0 646/591，
  L1 353/336）：**raw −1.7%，fp8 持平**——syscall 移除主要惠及 load
  侧，fp8 路径仍被重标定 stall 主导，与 docs/11 §8"收益为正但幅度小"
  预期一致。litmus 补验 **ld5d_cta = 原生 UTMALDG.5D 无 CALL**（cta
  补丁的最后一块拼图，load 全形态原生实锤）。§2.1 已按 ×3 数字刷新；
  NCU 锁频读数已回填 §2.2（L0 795.4/68.7%、raw 688.4/80.9%；L1
  438.4/61.7%、raw 390.7/69.7%；不变量跨 session 复现）。
- **E5 ✅ 完成（2026-07-27，GPU0）**：L1 `dram__bytes.sum` —— tk
  **313.44MB** vs CUTLASS **317.89MB**（差 1.4%，CUTLASS 反而略多）。
  **定案：L1 的 11.8% 差距不在流量，在延迟/调度**——同流量下有效 DRAM
  带宽 tk ~876GB/s vs CUTLASS ~1007GB/s，时间差全是空转；L1 重开后的
  杠杆与 L0 同类（重标定交织、任务边界；短 K 下每任务固定开销占比更高，
  11 §4）。顺带实锤（NCU demangled 签名）：CUTLASS 87c 为 **4×2 warp
  几何**（TiledMMA 4,2,1）+ **SM90_TMA_LOAD 2D 路径** + 384 线程，与
  docs/10 §1 结构对照一致。⚠️ NCU 运行中打印的 wall-clock（fp8
  454.2µs / cutlass "77.5ms"）受 replay/锁频扰动，不作数。

- **E2（可选）**：CUTLASS 当前卡 NCU 锁频 tensor%，让 §2.2 不变量分解
  整体换到同卡口径。

结果回填本文对应小节；单次实验参数照例记入结果目录与 HANDOFF。
