# 纯 Group GEMM 引擎（gg8）最优性论证与证据缺口

日期：2026-07-27（同日更新：CUTLASS 锚点同卡 boost 复测完成，见 §3.1，
主张 1 已按新数据降级）。对象：`gg8::kernel`（`tk.grouped_gemm_fp8`，
`tools/verify_fp8_gemm.py` 所测），即融合 kernel 里同一套 `gemm_config_fp8`
引擎的"无通信角色"形态。本文回答一个问题：**这颗纯 grouped GEMM 是不是
最优实现，证据链还缺哪几块**。数据来源 docs/08/10/11 + §3.1 同卡复测。

## 1. 主张的准确表述（三层，逐层可证伪）

"最优"不做绝对表述，拆成三条，各自对应证据与残余缺口：

1. **同类实现中最优（当前能成立的主张）。** 同类 = 路由感知（grouped 按
   expert 分组 + 间接调度）+ blockwise FP8（act 1×128 / weight 128×128 +
   fp32 重标定）+ **可通算融合**（persistent + dispenser 任务分发 + SM
   转岗，作为融合 kernel 的 consumer 引擎原样复用）。同类里唯一的对手是
   vLLM triton fused_moe，两层锁频均慢于我们（§3.2；其调优水位待 E3 复核）。
   CUTLASS 87c 不在同类内——纯 GEMM、无路由/GLU、非 persistent，是上界
   参照，不是可替换实现。
2. **对厂商纯 GEMM 上界：L0 76.8% / L1 88.2%（同卡 boost，§3.1）——
   两层都未到顶，不做"已达可达上限"的表述。** 此前"L1 反超 CUTLASS 6%"
   （docs/11 §2）经同卡 boost 复测**证伪**：那是跨卡（GPU4 vs 当前卡）+
   锁频锚点的伪象（旧锚点按时钟比折算本身偏慢 ~9%）。锁频→boost 后 L0
   差距从 14% 扩大到 23%，与 docs/11 §3"固定延迟依赖是第一大 stall"互证
   ——时钟升、延迟不缩，空转占比放大。
3. **剩余差距不是黑箱。** L0 锁频口径差 110.6µs = 重标定机制 76.9（70%）+
   结构性 33.7（30%），杠杆在册（重标定切片交织、store 后置、4×2 几何），
   且 boost 下放大的部分正是交织要吃的固定延迟 stall；所有便宜旁路
   （TASK_Q、EPIRED、scale 进 smem）已实验判负关闭（§4）。L1 差 11.8% 的
   来源（流量 vs 调度）待 E5 裁决。

## 2. 为什么说它站在结构最优附近：每个结构点都是一个被关闭的差距

| 结构点（现状） | 关闭的差距（出处） |
|---|---|
| QMMA SM120_16x8x32 fp8 + fp32 累加 | 与 CUTLASS/triton 同 atom，指令选型无差异（08 §3） |
| K-tile = 128 = 量化块 | 旧 64 深 stage 的 2× mbarrier/LDSM 冗余（10 §1） |
| 重标定每 128 K 一次 | 与 CUTLASS 87c blockwise promotion 同频同数学（10 §1） |
| A/B 全双缓冲 + arrived wait 提前至末尾 QMMA 前 | load→use 串行暴露 LDSM 延迟；CUTLASS 主循环最关键一手（10 §3） |
| COL_BLOCK=64，consumer ~156 reg 零 spill | 168 寄存器帽（=224−56 TMA ABI 调用帧，平台强制）下 128 宽 tile 需求 ~230 无解（10 §6-7） |
| 4 stage × 24KB = 96KB ≤ 99KB | sm120 smem ~100KB 帽下最深流水（08 §6） |
| TMA load `.shared::cta` 原生形态 | `.shared::cluster` 走驱动 syscall，.so 109 个 CALL.ABS→0（SASS 已实证，11 §6-7） |

## 3. 证据：横比与不变量

### 3.1 正式横比：同卡 boost 口径（2026-07-27，wall-clock，卡号待补）

| 层 | gg8（TFLOP/s） | CUTLASS 87c（TFLOP/s） | gg8/CUTLASS |
|---|---:|---:|---:|
| L0 | 659.4µs (312.7) | **506.1µs (407.3)** | 76.8% |
| L1 | 357.8µs (288.1) | **315.7µs (326.5)** | 88.2% |

一致性检查（锁频→boost 折算）：CUTLASS L0 664→506µs = 1.31×，恰为时钟比
→ 它纯算力受限、随频率线性；gg8 L0 774.6→659.4 仅 1.17× → 固定延迟
stall 不随频率缩放（与 docs/11 §3 的 stall_wait 第一名互证）。旧 L1 锚点
453µs 按同时钟比折算应 ~345µs，实测 315.7µs → **旧跨卡锚点本身偏慢 ~9%，
"L1 反超"由此产生，已证伪**。

### 3.2 归因数据：NCU 锁频口径（07-23/26，保留用于结构分析，绝对横比以 §3.1 为准）

| L0 (K=4096, N=1536) | Duration | tensor | Duration×tensor |
|---|---:|---:|---:|
| gg8 fp8（现役） | 774.6µs | 71.2% | 551µs |
| gg8 raw（无重标定探针） | 697.7µs | 79.4% | 554µs |
| triton fused_moe（vLLM 兜底 config） | 834µs | 66.7% | 556µs |
| CUTLASS 87c（07-23 GPU4 旧锚点） | 664µs | 84.9% | 564µs |

L1 锁频：gg8 425.0µs/64.2%，triton 537µs/50.2%（两 GEMM 共 config 的
结构受害者）。gg8 raw DRAM 67.7% 曾据此判"近带宽墙、L1 引擎关闭"
（docs/11 §2）——§3.1 中 CUTLASS 用同样的最小流量做到 315.7µs，说明 tk
的 L1 距墙仍有 ~12%，该判断降级为"带宽压力大但未到墙"，差距来源待 E5
裁决。

- **tensor-busy 不变量**（锁频口径内自洽）：同形状下 Duration×tensor% ≈
  551-564µs（L0），四行互证——QMMA 总量与速率四个实现完全相同，差距
  100% 是 tensor 空转，没有任何"更快的乘法"存在。最优化问题被收窄为
  "消空转"，而 gg8 的空转账已列清（§1 第 3 条）。boost 下无 tensor%
  （NCU 必锁频），只能用 §3.1 的时钟比折算做间接归因。
- **平台边界对所有实现同等生效**（docs/04/10）：fp32 累加税（fp8 峰值
  减半）、smem ~100KB（triton 深 stage 上不去的根因）、寄存器 168 帽
  （任何用 TMA 的 kernel）。CUTLASS L0 84.9%（锁频）/ 407.3 TFLOP/s
  （boost）就是这些边界下的实测天花板。
- **没有其它候选**：DeepGEMM 无 sm120 路径（sm90 wgmma / sm100 tcgen05
  在本卡都不存在）；cuBLASLt 无 1×128/128×128 blockwise 反量化语义的
  grouped fp8 接口，不构成同口径对照；triton fused_moe 即 vLLM 生产
  kernel，已在表内。
- 正确性门：rel_err 1.68e-03（fp32 反量化参考，输入逐 bit 相同）。

## 4. 已判负的旁路（支撑"没有便宜的翻盘路"）

| 旁路 | 结果 | 出处 |
|---|---|---|
| TASK_Q 2→4（producer 跑道） | +0.2~0.6% 变慢，判负 | 11 §5 |
| EPIRED（epilogue 直推归约省 268MB 流量） | v2 向量原子仍输 108µs，本机 L2 fp32 原子吞吐 ~1e11/s 是硬墙 | HANDOFF 07-26 / docs/04 §2 |
| scale 走 smem 流水 | stall 无 LDG 记分牌 signature，降级观察 | 11 §9 |
| ~~L1 引擎继续优化~~ | ~~已反超参照且近 DRAM 墙，关闭~~ **07-27 重开**：同卡 boost 复测 CUTLASS 快 11.8%（§3.1），"关闭"依据失效，待 E5 定差距来源 | 11 §2 |
| 全宽 B 双缓冲 / setmaxnreg | ptxas spill 4.7× 回退 / C7506 全数忽略 | 10 §2, §6 |

## 5. 证据缺口 → 补实验清单（按执行顺序，均单卡或纯编译）

> 红线提醒：先确认卡空闲；NCU 严禁与 4 卡任务共存（docs/08 §4）；
> E1 的 probe 脚本自带空闲检查。E4 不占 GPU。

### E4（先跑，纯编译不占卡）：寄存器 168 帽是否随 syscall 消失而解除

cta 补丁后 CALL.ABS=0 已实证，但"帽解除与否"未裁决——它决定
4×2+COL=128 路线（L0 剩余 33.7µs 结构差距）的生死，也决定 §2 中
"COL=64 是平台强制最优解"这句话能不能写死。

```bash
cd /workspace
# litmus 5d cta 形态（docs/11 §8 第 0 步，若还没验过）
nvcc -arch=sm_120a -cubin --ptxas-options=-v -o /tmp/l.cubin moe_bench/tools/tma_litmus.cu
cuobjdump -sass /tmp/l.cubin | awk '/Function :/{f=$3} /CALL|UTMALDG/{print f,$2,$4}' | grep ld5d_cta
# COL=128 压力编译：只读 ptxas 资源，不运行
sed -i 's/COL_BLOCK = 64/COL_BLOCK = 128/; s/PIPELINE_STAGES = 4/PIPELINE_STAGES = 3/' \
  moe_bench/kernels/tileoverlap/common/sm120_common.cuh
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4 2>&1 | tee /tmp/build_col128.log
grep -E 'registers|spill|STACK' /tmp/build_col128.log
git -C moe_bench checkout kernels/tileoverlap/common/sm120_common.cuh          # 还原
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4    # 重建正式 .so
```

判据：gg8 consumer **REG > 168 且零 spill** → 帽已解除，4×2+COL=128 复活；
仍压 168 或 spill → "COL=64 平台强制"定案。⚠️ COL=128 可能撞上
[gate32|up32] 交织/scale 索引的编译错——报错则退而检出 P2 之前的提交整体
编译（`git log -- kernels/tileoverlap/common/sm120_common.cuh` 找 P1v2）。

### E1：cta 补丁后 gg8 锁频重定基（§3.2 归因表的数字更新）

§3.2 的 774.6/71.2 等 NCU 数字测于 cta 补丁**之前**（probe_20260726_080115）；
boost 基线已在 07-27 复测（659.4/357.8，§3.1），**待补的是 NCU 锁频四份
报告**（新的 tensor% / 空转分解要靠它）。

```bash
bash moe_bench/tools/probe_engine.sh <空闲卡>
```

产出：P0 boost 基线（L0/L1 × fp8/raw ×3 遍）+ P3 四份 NCU 锁频报告。
判据：rel_err 必须仍 1.68e-03；CALL.ABS 普查 = 0；新 Duration/tensor%
回填 §3.2。

### E2：CUTLASS 锚点当前卡复测 —— **boost 部分已完成（2026-07-27）**

✅ boost wall-clock 已测：L0 506.1µs/407.3 TFLOP/s、L1 315.7µs/326.5，
已入 §3.1，结论 = L1"反超"证伪、旧跨卡锚点偏慢 ~9%。
**待补（可选）**：NCU 锁频采样（拿 CUTLASS 在当前卡的 tensor%，让 §3.2
的不变量分解整体换到同卡口径；不做则 §3.2 只用于结构定性）：

```bash
cd /workspace/moe_bench
CUDA_VISIBLE_DEVICES=<空闲卡> ncu --replay-mode kernel --kernel-name 'regex:cutlass' \
  --launch-skip 4 --launch-count 1 --section SpeedOfLight --section ComputeWorkloadAnalysis \
  --export cutlass_L0 ./tools/cutlass_probe/build/cutlass_fp8_grouped \
  --groups=64 --m=256 --n=1536 --k=4096 --iterations=10
# L1 同命令换 --n=4096 --k=768, --export cutlass_L1
```

### E3：triton 调优水位（"同类最优"主张的最后一块，也最耗时）

现在赢的是 vLLM **兜底 config** 的 triton（docs/08 §6），不是它的调优
水位——不补这块，"beats triton"会被一句"你没调优"顶回来。

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.tune_triton_moe --dry-run   # 先干跑
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.tune_triton_moe             # 320 候选，最长
# 调优后 e2e 重报（4 卡）：
VLLM_TUNED_CONFIG_FOLDER=<调优结果目录> CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m moe_bench.tools.run_tktp --scheme serial --no-verify
# （可选）调优后单 kernel NCU：恢复探针后按 docs/08 §1 命令采 w13/w2
git -C moe_bench checkout ad59fa3 -- tools/ncu_serial_gemm_probe.py
```

判据：调优 triton 的 L0 NCU duration vs gg8 同卡数字。仍慢 → 主张 1
坐实；反超 → 主张 1 降级为"与调优 triton 同档，且唯一携带通算融合能力"，
同时 e2e 领先（当前 28.2%）按 docs/08 §6 预告重新报数。

### E5（优先级已提升）：L1 差距来源裁决 —— 流量 vs 调度

CUTLASS L1 boost 315.7µs 快我们 11.8%（§3.1），但两边最小 DRAM 流量相同
（≈ A 12.6MB fp8 + W 201MB fp8 + C 134MB bf16 ≈ 350MB）。**双方各采一次
dram__bytes 即可裁决**：字节数相近 → 差在延迟/调度（tk 侧可修）；CUTLASS
字节更少 → 差在 L2 复用/栅格化顺序（要动 tile 调度序）。

```bash
# tk 侧
CUDA_VISIBLE_DEVICES=<空闲卡> ncu --kernel-name 'regex:gg8' --launch-skip 4 --launch-count 1 \
  --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 10
# CUTLASS 侧
CUDA_VISIBLE_DEVICES=<空闲卡> ncu --kernel-name 'regex:cutlass' --launch-skip 4 --launch-count 1 \
  --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./tools/cutlass_probe/build/cutlass_fp8_grouped --groups=64 --m=256 --n=4096 --k=768 --iterations=10
```

判据：另用 nvbandwidth/bandwidthTest 实测峰值带宽（不抄 spec），
`dram__bytes ÷ 峰值带宽` 得时间下界；tk L1 与下界的比值 + 两边字节差
写进 §1 主张 3，并决定 L1 重开后的第一个杠杆方向。

## 6. 回填规则

E1（NCU 部分）跑完后把 §3.2 换成 cta 补丁后的同卡数字并注明产物目录；
E2 的 NCU 部分（可选）补 CUTLASS 当前卡 tensor%；E3 结论回填 §1 主张 1
与 docs/08 §6；E4 结论回填 §2 寄存器行与 docs/10 §7；E5 结论回填 §1
主张 3 与 docs/11 §2。§3.1 是当前唯一的同卡同日横比，正式引用以它为准
（卡号/iterations 等元数据待补齐）。
