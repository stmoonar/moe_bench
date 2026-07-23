#!/usr/bin/env bash
# CUTLASS grouped GEMM 单卡探针构建（sm120，RTX PRO 5000）。
#
# 目的: 用 CUTLASS 官方 kernel 给我们的 gg8/gg 引擎立一个"厂商可达水位"参照，
# 形状与主配置 rank0 口径对齐（E=64 组、每组 M=256）:
#   L0: --groups=64 --m=256 --n=1536 --k=4096
#   L1: --groups=64 --m=256 --n=4096 --k=768
#
# 两个二进制:
#   cutlass_fp8_grouped  = examples/87c（原样编译）: sm120 blockwise grouped GEMM,
#       A fp8 RowMajor(act, scale 1x128) x B fp8 ColumnMajor(weight, scale 128x128),
#       bf16 输出, fp32 累加, tile 128x128x128 —— 与我们 fp8 口径完全同构。
#   cutlass_bf16_grouped = grouped_gemm_bf16.cu（example 24 改 bf16+TN）:
#       2.x GemmGrouped + Sm80 mma.sync 16x8x16（sm120 无 bf16 的 3.x grouped 路径,
#       array builder 仅支持 F8F6F4）。同时会附带跑一个 batched GEMM 对照。
#
# 用法（远端容器, 任意目录）:
#   bash /workspace/work/moe_bench/tools/cutlass_probe/build.sh
#   CUDA_VISIBLE_DEVICES=<空闲卡> ./build/cutlass_fp8_grouped --groups=64 --m=256 --n=1536 --k=4096 --iterations=100
#   CUDA_VISIBLE_DEVICES=<空闲卡> ./build/cutlass_bf16_grouped --groups=64 --m=256 --n=1536 --k=4096 --iterations=100
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CUTLASS="$HERE/../../cutlass"
OUT="$HERE/build"
mkdir -p "$OUT"

COMMON_FLAGS=(-std=c++17 -O3 -DNDEBUG --expt-relaxed-constexpr
  -gencode arch=compute_120a,code=sm_120a
  -I"$CUTLASS/include" -I"$CUTLASS/tools/util/include" -I"$CUTLASS/examples/common"
  -lcuda)

echo "== building cutlass_fp8_grouped (87c, sm120 blockwise) =="
nvcc "$CUTLASS/examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu" \
  "${COMMON_FLAGS[@]}" -o "$OUT/cutlass_fp8_grouped"

echo "== building cutlass_bf16_grouped (example24 改 bf16+TN, Sm80 path) =="
nvcc "$HERE/grouped_gemm_bf16.cu" \
  "${COMMON_FLAGS[@]}" -o "$OUT/cutlass_bf16_grouped"

echo "done. run e.g.:"
echo "  CUDA_VISIBLE_DEVICES=<idle> $OUT/cutlass_fp8_grouped  --groups=64 --m=256 --n=1536 --k=4096 --iterations=100"
echo "  CUDA_VISIBLE_DEVICES=<idle> $OUT/cutlass_fp8_grouped  --groups=64 --m=256 --n=4096 --k=768  --iterations=100"
echo "  CUDA_VISIBLE_DEVICES=<idle> $OUT/cutlass_bf16_grouped --groups=64 --m=256 --n=1536 --k=4096 --iterations=100"
echo "  CUDA_VISIBLE_DEVICES=<idle> $OUT/cutlass_bf16_grouped --groups=64 --m=256 --n=4096 --k=768  --iterations=100"
