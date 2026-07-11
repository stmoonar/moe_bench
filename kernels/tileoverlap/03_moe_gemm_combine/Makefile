# Phase 3: fused W2 GEMM + combine (module _Cf) and the standalone combine
# anchor (module _C). Both are built by explicit rules to control the pybind
# module name (common.mk hardwires TORCH_EXTENSION_NAME=_C).
# NUM_GPUS must match both the torchrun world size and the compiled pgl size.
NUM_GPUS ?= 2
ARCH ?= SM120

# Borrow the flag machinery from common.mk without its default build rule by
# setting a harmless SRC/OUT/CMD/CONFIG, then override targets below.
SRC := moe_gemm_combine_fused_sm120.cu
OUT := _Cf$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
ANCHOR_OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := OMP_NUM_THREADS=1 torchrun --nproc_per_node=$(NUM_GPUS) benchmark_fused.py
CONFIG := pytorch
NVCCFLAGS += -DTK_NUM_DEVICES=$(NUM_GPUS)

include ../../../ThunderKittens/kernels/common.mk

# common.mk's $(OUT) rule compiles SRC with its flags (incl. -DTORCH_EXTENSION_NAME=_C).
# We need the fused module named _Cf, so override with an explicit recipe that
# appends -DTORCH_EXTENSION_NAME=_Cf LAST (nvcc takes the last -D).
$(OUT): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -DTORCH_EXTENSION_NAME=_Cf -o $(OUT)

anchor: $(ANCHOR_OUT)
$(ANCHOR_OUT): moe_gemm_combine_sm120.cu
	$(NVCC) moe_gemm_combine_sm120.cu $(NVCCFLAGS) -DTORCH_EXTENSION_NAME=_C -o $(ANCHOR_OUT)

run_anchor: anchor
	OMP_NUM_THREADS=1 torchrun --nproc_per_node=$(NUM_GPUS) benchmark.py

.PHONY: anchor run_anchor
