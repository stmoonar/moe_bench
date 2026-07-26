# 引擎归因探针（2026-07-26）与 TMA .shared::cta 形态修复

日期：2026-07-26。工具：`tools/probe_engine.sh`（一键探针，产物自动打包）+
`tools/tma_litmus.cu`（编译期试金石）。机器：GPU 0，driver 见产物包
`env_nvidia_smi.txt`，NCU 默认 clock-control（锁基频 ~1.69GHz），与 docs/08/10
的 789/664µs 锚点同口径。产物：`tp_test_results/probe_20260726_080115_engine/`。

## 1. 分析框架：tensor-busy 不变量

同一形状下 `Duration × tensor%` 是常数（QMMA 总量与速率两边相同），差距
全部是 tensor 空转，可以按来源二分：

| L0 (K=4096, N=1536) | Duration | tensor | Duration×tensor |
|---|---:|---:|---:|
| gg8 fp8（现役） | 774.6µs | 71.2% | 551µs |
| gg8 raw（无重标定探针） | 697.7µs | 79.4% | **554µs ✓** |
| CUTLASS 87c 锚点（07-23, GPU4） | 664µs | 84.9% | 564µs |

**L0 差距 110.6µs = 重标定机制 76.9µs（70%）+ 结构性 33.7µs（30%）**。
重标定多付 46.8M 条指令（224.7M vs 177.9M, +26%）。

boost 口径基线（`verify_fp8_gemm`, 3 遍稳定）：L0 fp8 646µs / raw 591µs
（+9.4%）；L1 fp8 353µs / raw 336µs（+5.0%）。rel_err 1.68e-03 不变。

## 2. L1 已反超 CUTLASS 参照，引擎优化只做 L0

| L1 (K=768, N=4096) | Duration | tensor | DRAM |
|---|---:|---:|---:|
| gg8 fp8 | **425.0µs** | **64.2%** | 62.6% |
| gg8 raw | 393.9µs | 69.1% | 67.7% |
| CUTLASS 锚点 | 453µs | 63.3% | — |

P2（COL=64）之后 L1 已比 CUTLASS 快 6%（docs/08 的 519µs 是 P1 前旧引擎）。
且 raw 的 DRAM 已到 67.7%（L2 命中 76-80%），**L1 接近带宽墙**，继续抠
tensor 占空比天花板很低。正式报数前建议在当前卡上复测一次 CUTLASS 锚点
（探针在 `fp8_tp` 分支历史）。

## 3. NCU stall 归因：固定延迟依赖是第一大 stall

四份报告（L0/L1 × fp8/raw）的 Warp State 第一名全部是
**stall_wait（固定延迟执行依赖）**：fp8 L0 每发射间隔 5.63 周期里占 1.7
（31%），raw L0 2.3/6.4（36%），L1 raw 2.2/6.7（32.7%）。同时
`No Eligible` 60-66%、发射槽只有 35-40% 忙、每调度器仅 2.25 个活跃 warp——
**是延迟受限（依赖链卡 QMMA），不是发射带宽受限**。这直接支持
"重标定尾巴切片交织"方案：每 stage 边界的
`末 QMMA → FFMA(RAW 等写回) → 清零 → 次 stage QMMA(RAW 等清零)`
串行链是主要空转来源，且发射侧有 60% 空闲容量可以吸收交织进来的指令。

## 4. K 扫描：每任务固定开销是小头

N=4096 固定（8192 任务），K∈{768,1536,3072,4096}，拟合 `时间≈a·stage数+b`：

- fp8：斜率 51.9µs/stage 单位，截距 ≈ 42µs；raw：斜率 46.9，截距 ≈ 55。
- 折合**每任务边界成本 ~0.5-0.7µs**（boost，74.5 任务/SM）。L1 形状占总时
  间 ~12%，L0 形状（3072 任务）合计仅 ~20µs。
- 斜率差 = 重标定每 stage 约 +10.6%，与整体 +9.4% 互相印证。

结论：任务边界重叠（store 后置）值得做但排在交织之后。

## 5. TASK_Q 2→4：判负

一行改动 A/B（脚本 P4 自动改/还原）：L0 646.7→648.2µs（+0.2%）、
L1 352.8→354.9µs（+0.6%），全部轻微变慢。**producer 供给跑道不是瓶颈**，
TASK_Q 保持 2，此路关闭。（这同时弱化了"TMA syscall 卡 producer 发射"
的猜想——syscall 的主要代价在寄存器预留，不在发射开销。）

## 6. 试金石：.shared::cta 载入是原生指令（本轮最大发现）

`tools/tma_litmus.cu` 逐形态编译看 SASS（`sm_120a`，与生产同 nvcc）：

| 形态 | SASS | 判定 |
|---|---|---|
| `ld{2,3,4,5}d_cluster`（TK 现用） | CALL.ABS.NOINC + UTMALDG，REG:24，STACK:8 | **驱动 syscall 包装** |
| `ld2d_cta` / `ld4d_cta` | 单条 UTMALDG，REG:4，STACK:0 | **原生** |
| `st{1,2,4,5}d_cta`（TK store 现用形态） | UTMASTG/UBLKCP，无 CALL | 本来就是原生 |
| `ld1d_cluster` | CALL + UBLKCP | syscall |

现役 .so 里共 **109 个 CALL.ABS 位点，全部来自 load**（store 无 CALL）。

寄存器帽问题**本轮未定论**：press_none（无 TMA 对照，需求 ~190）也被
分配器压进 REG:168/零 spill，压力探针设计不够狠；press_2d/4d（cluster 形态）
REG:168 + STACK:56/64 只复现了带 CALL 必 spill。**真裁决 = cta 补丁后重编
真 kernel 看 ptxas 是否允许 >168**——这决定 4×2 + COL=128 路线的生死。

注意：5d 的 cta 形态（swizzled tile，A/B 加载实际路径）当时没测，litmus
已补 `ld5d_cta`，上机先编它。

## 7. 修复：tileoverlap::tma_cta::load_async

ThunderKittens 是 git 子模块（不可改），在正本
`kernels/tileoverlap/common/sm120_common.cuh` 新增 `tma_cta` 命名空间：
复刻 TK `load_async` 的地址/坐标计算（tile 版 swizzle→5d / 非 swizzle→4d，
vec 版按 `sv_tma_dim1/2` 分片 4d），仅把 `.shared::cluster` 换成
`.shared::cta`。替换 8 个调用点：dispenser producer 2 个（A/B tile）、
push_lane 2 个、scatter_lane 4 个（sv）。`tma::expect_bytes`（原生
mbarrier 指令）与全部 store（本来就是 cta 形态）不动。

语义等价性：sm120 无 thread block cluster，CTA==cluster，两形态的
dst/mbar 操作数（CTA 本地 shared 地址）与 mbarrier complete_tx 完成语义
一致，仅指令编码不同。

死锁审计：无新增/修改等待点。TMA 完成仍喂同一 mbarrier，消费侧
`guarded_wait` / 自旋 guard 全部不变。

## 8. 上机验证 runbook（cta 补丁）

```bash
# 0) litmus 先验 5d cta 形态（编译期, 不占卡）
nvcc -arch=sm_120a -cubin --ptxas-options=-v -o /tmp/l.cubin moe_bench/tools/tma_litmus.cu
cuobjdump -sass /tmp/l.cubin | awk '/Function :/{f=$3} /CALL|UTMALDG/{print f,$2,$4}' | grep ld5d_cta
#    预期: 只有 UTMALDG.5D, 无 CALL。若有 CALL → tile 路径回退 tma::load_async, 只留 vec 的 cta

# 1) 干净重编, 看 ptxas: 预期零 spill; 重点记录各 kernel REG(是否仍被 168 帽住)
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4 2>&1 | tee /tmp/build_cta.log

# 2) syscall 普查: 预期 CALL.ABS 位点 109 → 0
cuobjdump -sass moe_bench/kernels/tk/build/tk_moe_w4_h4096_rb128.so | grep -c 'CALL.ABS'

# 3) 单卡裁决(正确性逐比特 + 性能): rel_err 必须仍为 1.68e-03
CUDA_VISIBLE_DEVICES=<idle> python -m moe_bench.tools.verify_fp8_gemm 64 256 4096 1536 20
CUDA_VISIBLE_DEVICES=<idle> python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 20

# 4) 四卡全链路(通信路径也换了 cta): 先 10 迭代正确性, 再性能
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --no-verify
```

预期信号：正确性逐比特不变；纯 GEMM 收益为正但幅度小（syscall 主要在
producer 单 lane，本就不是瓶颈）；通信 lane 每 token 少 2 次 syscall；
**最重要的产出是第 1 步的 REG 读数**——若帽解除，下一步按
docs/10 §8 的排序推进 4×2 几何 + COL=128。

## 9. 修正后的杠杆排序（引擎侧，只做 L0）

1. **重标定切片交织**（本文 §3 证实，76.9µs 的大头）：按 16 列 base-tile
   粒度把 `FFMA(片j)→清零(片j)→下一 stage QMMA_kk0(片j)` 交错，寄存器
   零成本，每元素运算次序不变 → 逐比特等价。
2. **任务边界 store 后置重叠**（§4 定价 ~20µs/L0）：acc（store 源）与
   sub（下一任务累加器）寄存器不相交，下一任务 stage-0 可提到 store 前。
3. **4×2 warp 几何 + COL=128 + [gate16|up16] 交织**：等 §8 第 1 步的
   REG 裁决；吃剩余 ~34µs 结构性差距（LDSM/单位输出 −40%、任务数减半）。
4. 已关闭：TASK_Q（§5 判负）、scale 进 smem（stall 数据无 LDG 记分牌
   signature，降级观察）、L1 引擎优化（§2 已反超参照且近 DRAM 墙）。
