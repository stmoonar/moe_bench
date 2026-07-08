# Phase 1 单卡 grouped GEMM：编译对拍结果与坑

**日期**：2026-07-08
**卡**：`CUDA_VISIBLE_DEVICES=9`（单卡）
**kernel**：`tileoverlap/01_grouped_gemm/grouped_gemm_sm120.cu` + `common/sm120_common.cuh`

## 结论：首次编译零错误，性能超预期

- 编译：SM120 一次通过，**无 spill**（`8 bytes stack frame, 0 spill`）——比 PLAN §8 的悲观预期好。
- 性能：**TK / torch = 150%~178%**，远超 Phase 1 的 70% 验收线。
  | padded tokens | TK | torch | TK/torch | max diff |
  |---|---|---|---|---|
  | 10240 | 216 TFLOP/s | 122 | 178% | 0.093 |
  | 34304 | 234 TFLOP/s | 156 | 150% | 0.096 |
  | 144896 | — | — | — | 0.1003（略超阈值，见下） |

  216~234 TFLOP/s 是本卡 bf16 grouped GEMM 的一个合理起点（warp mma.sync + 3 级流水）。

## 坑 1（已修复）：tma_swizzle_allocator 的 1024B 对齐会顶破紧贴的 smem 预留

**症状**：kernel 一跑就 `illegal memory access`；compute-sanitizer 定位到 producer warp
（thread 256）的 `tma::load_async` 写 smem 时 **Out-of-range shared or local address**
（`sm120_common.cuh:192`，即最后一个 pipeline stage 的 B tile TMA store）。

**排查**：去掉 `-DNDEBUG` 重编让 `tma.cuh` 的 descriptor 断言生效——**断言全过**，
说明 TMA descriptor 本身合法，问题不在 gmem/smem shape，而在 smem 地址越界。

**根因**：`tma_swizzle_allocator` 在 SM120 上是 `shared_allocator<1024>`
（`util.cuh:301`，"swizzled TMA modes require up to 1024 byte alignments"）。它在
`allocate()` 里把 base 指针**向上对齐到 1024B**。而 kernel 只按
`DYNAMIC_SHARED_MEMORY = 3*32KB = 96KB` 精确预留动态 smem。当 CUDA 返回的动态 smem 起始
地址不是 1024B 对齐时，allocator 把指针前移最多 1023B，导致最后一个 stage 的 tile
写到 96KB 预留区之外 → 越界。

**修复**：launch 时多留 1024B（`DYNAMIC_SHARED_MEMORY + 1024`），把对齐 bump 的空间预留出来。
99KB 上限下 96KB+1KB=97KB 仍然放得下。改动只在 entrypoint 的 `cudaFuncSetAttribute` +
launch 的第三个参数。

**通用教训**：TK 的 `tma_swizzle_allocator` 有 1024B 对齐语义，凡是「按 tile 尺寸精确预留
动态 smem」的 kernel 都要额外预留 1024B 的对齐余量，否则最后一个分配可能越界。02 融合 kernel
的 dispatch 路径也用同一个 allocator，同样要留余量（已在 02 一并修）。

## 坑 2（⚠️ 结论作废，见 docs/05）：144896 token 处 max diff 0.1003

**当时误判**：以为是 bf16 大 K 舍入的正常误差。**实际上是** `group::store` 的
warpgroup 交织行映射与 consumer 输入行不匹配的**真 bug**（详见
[docs/05_关键bug_group_store行映射.md](05_关键bug_group_store行映射.md)）。当时
mean diff ≈ ref mean（0.0099 vs 0.0094）本应是系统性错误的铁证，被我误读成舍入。

修复后 01 三档全部 max diff ~0.0002（含 144896 token），性能不变（150~177%）。
下面的原始记录保留以便追溯，但「bf16 舍入」的解释是错的。

- mean diff 稳定在 0.0099，与通过的两档**完全一致**；只有 max 的单点 outlier 从 0.096
  漂到 0.1003。
- TK 用 `rt_fl`（fp32）累加，torch matmul 也 fp32 累加，差异来自两者 fp32 舍入路径不同 +
  K=7168 深归约下的极值 outlier。**这是 bf16 输入量级的正常误差**，不是算子 bug。
- 144896 padded tokens（~140k）远超我们的真实规模（每 rank 512 token、总 2048、top-8 后
  每卡约万级激活行）。真实规模落在通过的两档区间，无需担心。
- 若要让 benchmark 三档全绿，可把 01 的断言阈值放宽到 0.12（bf16 大 K 合理量级），但**不改
  kernel**。当前保留 0.1 阈值并在此记录。

## T_comp_alone 锚点（供 Phase 2 算 overlap 效率的分母）

- 216~234 TFLOP/s @ H=7168, I=2048。Phase 2 的融合 kernel 计算侧应接近此曲线，
  掉太多说明 dispatch 干扰了计算。
