ARCH ?= SM120
SRC := grouped_gemm_sm120.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python3 benchmark.py
CONFIG := pytorch

include ../../../ThunderKittens/kernels/common.mk
