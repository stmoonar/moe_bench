# NUM_GPUS must match both the torchrun world size and the compiled pgl size.
NUM_GPUS ?= 2

ARCH ?= SM120
NVCCFLAGS += -DTK_NUM_DEVICES=$(NUM_GPUS)
SRC := moe_dispatch_gemm_sm120.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := OMP_NUM_THREADS=1 torchrun --nproc_per_node=$(NUM_GPUS) benchmark.py
CONFIG := pytorch

include ../../../ThunderKittens/kernels/common.mk
