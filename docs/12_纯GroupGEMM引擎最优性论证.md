# 纯 Group GEMM 引擎（gg8）最优性论证与证据缺口

日期：2026-07-27。对象：`gg8::kernel`（`tk.grouped_gemm_fp8`，
`tools/verify_fp8_gemm.py` 所测），即融合 kernel 里同一套 `gemm_config_fp8`
引擎的"无通信角色"形态。本文回答一个问题：**这颗纯 grouped GEMM 是不是
最优实现，证据链还缺哪几块**。数据来源 docs/08/10/11，未重复展开过程。

## 1. 主张的准确表述（三层，逐层可证伪）

"最优"不做绝对表述，拆成三条，各自对应证据与残余缺口：

1. **L1 形状（K=768, N=4096）：已达本平台可达最优水位。**
   快于厂商参照 CUTLASS 87c 6%（425 vs 453µs 锁频），且 raw 探针
   DRAM 67.7%、L2 命中 76-80% —— 已贴近带宽墙，tensor 占空比继续抠的
   天花板极低（docs/11 §2）。
2. **L0 形状（K=4096, N=1536）：同类实现中最优，距"纯 GEMM 厂商上界"14%,
   且该上界对本引擎不可达。** 同类 = 路由感知（grouped 按 expert 分组 +
   间接调度）+ blockwise FP8（act 1×128 / weight 128×128 + fp32 重标定）+
   **可通算融合**（persistent + dispenser 任务分发 + SM 转岗，作为融合
   kernel 的 consumer 引擎原样复用）。CUTLASS 87c 与 triton 都不满足
   第三条——它们是对照参考，不是可替换实现；87c 更是连路由/GLU 都没有的
   纯 GEMM，664µs 是上界不是目标线。
3. **剩余差距不是黑箱。** L0 差 110.6µs = 重标定机制 76.9（70%）+ 结构性
   33.7（30%），杠杆在册（重标定切片交织、store 后置、4×2 几何）；所有
   便宜旁路（TASK_Q、EPIRED、scale 进 smem）已实验判负关闭（§4）。

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

## 3. 证据：横比与不变量（NCU 锁频 ~1.67-1.69GHz，kernel replay）

| L0 (K=4096, N=1536) | Duration | tensor | Duration×tensor |
|---|---:|---:|---:|
| **gg8 fp8（现役）** | 774.6µs | 71.2% | 551µs |
| gg8 raw（无重标定探针） | 697.7µs | 79.4% | 554µs |
| triton fused_moe（vLLM 兜底 config） | 834µs | 66.7% | 556µs |
| CUTLASS 87c（07-23 GPU4 锚点） | **664µs** | 84.9% | 564µs |

| L1 (K=768, N=4096) | Duration | tensor | 备注 |
|---|---:|---:|---|
| **gg8 fp8（现役）** | **425.0µs** | 64.2% | raw DRAM 67.7% 近带宽墙 |
| triton fused_moe（兜底 config） | 537µs | 50.2% | 两 GEMM 共 config 的结构受害者 |
| CUTLASS 87c（07-23 GPU4 锚点） | 453µs | 63.3% | 我们快 6% |

- **tensor-busy 不变量**：同形状下 Duration×tensor% ≈ 551-564µs（L0），
  四行互证——QMMA 总量与速率四个实现完全相同，差距 100% 是 tensor 空转，
  没有任何"更快的乘法"存在。最优化问题被收窄为"消空转"，而 gg8 的空转
  账已列清（§1 第 3 条）。
- **平台边界对所有实现同等生效**（docs/04/10）：fp32 累加税（fp8 峰值
  减半）、smem ~100KB（triton 深 stage 上不去的根因）、寄存器 168 帽
  （任何用 TMA 的 kernel）。CUTLASS 84.9% 就是这些边界下的实测天花板。
- **没有其它候选**：DeepGEMM 无 sm120 路径（sm90 wgmma / sm100 tcgen05
  在本卡都不存在）；cuBLASLt 无 1×128/128×128 blockwise 反量化语义的
  grouped fp8 接口，不构成同口径对照；triton fused_moe 即 vLLM 生产
  kernel，已在表内。
- boost 口径（`verify_fp8_gemm`，供 e2e 对账）：L0 646µs / L1 353µs；
  正确性门 rel_err 1.68e-03（fp32 反量化参考，输入逐 bit 相同）。

## 4. 已判负的旁路（支撑"没有便宜的翻盘路"）

| 旁路 | 结果 | 出处 |
|---|---|---|
| TASK_Q 2→4（producer 跑道） | +0.2~0.6% 变慢，判负 | 11 §5 |
| EPIRED（epilogue 直推归约省 268MB 流量） | v2 向量原子仍输 108µs，本机 L2 fp32 原子吞吐 ~1e11/s 是硬墙 | HANDOFF 07-26 / docs/04 §2 |
| scale 走 smem 流水 | stall 无 LDG 记分牌 signature，降级观察 | 11 §9 |
| L1 引擎继续优化 | 已反超参照且近 DRAM 墙，关闭 | 11 §2 |
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

### E1：cta 补丁后 gg8 重定基（§3 表格的正式数字来源）

§3 的 774.6/71.2 等数字全部测于 cta 补丁**之前**（probe_20260726_080115），
补丁预期小幅向好，正式论证不能引用旧口径。

```bash
bash moe_bench/tools/probe_engine.sh <空闲卡>
```

产出：P0 boost 基线（L0/L1 × fp8/raw ×3 遍）+ P3 四份 NCU 锁频报告。
判据：rel_err 必须仍 1.68e-03；CALL.ABS 普查 = 0；新 Duration/tensor%
回填 §3。

### E2：CUTLASS 锚点在当前卡复测（L1"反超"与 L0"86%"的分母）

锚点是 07-23 在 GPU4 测的，跨卡跨日；docs/11 §2 早已要求复测。
工作树已有 `cutlass/` 源码（build.sh 找的就是 `../../cutlass`），只需恢复
slim 时删除的探针：

```bash
cd /workspace/moe_bench
git checkout 9b847be^ -- tools/cutlass_probe
bash tools/cutlass_probe/build.sh
CUDA_VISIBLE_DEVICES=<空闲卡> ./tools/cutlass_probe/build/cutlass_fp8_grouped \
  --groups=64 --m=256 --n=1536 --k=4096 --iterations=100        # L0
CUDA_VISIBLE_DEVICES=<空闲卡> ./tools/cutlass_probe/build/cutlass_fp8_grouped \
  --groups=64 --m=256 --n=4096 --k=768 --iterations=100         # L1
# NCU 同口径（与 E1 同一张卡）：
CUDA_VISIBLE_DEVICES=<空闲卡> ncu --replay-mode kernel --kernel-name 'regex:cutlass' \
  --launch-skip 4 --launch-count 1 --section SpeedOfLight --section ComputeWorkloadAnalysis \
  --export cutlass_L0 ./tools/cutlass_probe/build/cutlass_fp8_grouped \
  --groups=64 --m=256 --n=1536 --k=4096 --iterations=10
```

判据：L1 上 CUTLASS 仍 ≥ 我们（同卡 E1 数字）→ 主张 1 坐实；
L0 用同卡同日数字重算差距百分比替换"86%"。

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

判据：调优 triton 的 L0 NCU duration vs gg8 同卡数字。仍慢 → 主张 2
坐实；反超 → 主张 2 降级为"与调优 triton 同档，且唯一携带通算融合能力"，
同时 e2e 领先（当前 28.2%）按 docs/08 §6 预告重新报数。

### E5（可选，与 E1 同卡顺手）：L1 带宽墙定量，把"近"换成数字

L1 最小 DRAM 流量 ≈ A 12.6MB(fp8) + W 201MB(fp8) + C 134MB(bf16) ≈ 350MB。

```bash
CUDA_VISIBLE_DEVICES=<空闲卡> ncu --kernel-name 'regex:gg8' --launch-skip 4 --launch-count 1 \
  --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 10
```

判据：`实测 dram__bytes ÷ 实测峰值带宽`（峰值用 nvbandwidth/bandwidthTest
实测，不抄 spec）得时间下界，与 L1 duration 的比值写进主张 1 做 roofline
数字。

## 6. 回填规则

E1/E2 跑完后把 §3 两张表整体替换为**同卡同日**数字并注明产物目录；
E3 结论回填 §1 主张 2 与 docs/08 §6；E4 结论回填 §2 寄存器行与
docs/10 §7。在此之前，本文所有数字视为"07-23/07-26 混合口径的暂定值"。
