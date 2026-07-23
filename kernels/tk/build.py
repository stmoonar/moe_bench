"""Build & load the combined TK MoE extension (tk_moe.cu) for a given world size.

The extension is compiled per (world_size, hidden) because TK_NUM_DEVICES and the
pgl sizes are compile-time constants. We cache builds under kernels/tk/build/.
"""
import os
import subprocess
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_TK_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "ThunderKittens"))
_COMMON = os.path.abspath(os.path.join(_HERE, "..", "tileoverlap", "common", "sm120_common.cuh"))


def _nvcc_flags():
    import sysconfig
    import pybind11
    from torch.utils.cpp_extension import include_paths, library_paths
    inc = [f"-I{sysconfig.get_path('include')}"]
    inc += [f"-I{pybind11.get_include()}"]
    inc += [f"-I{p}" for p in include_paths()]
    lib = [f"-L{p}" for p in library_paths()]
    return inc, lib


def build_and_load(world_size: int, hidden: int = 4096, module_name: str = "tk_moe",
                   row_block: int = 128):
    build_dir = os.path.join(_HERE, "build")
    os.makedirs(build_dir, exist_ok=True)
    mod = f"{module_name}_w{world_size}_h{hidden}_rb{row_block}"
    so_path = os.path.join(build_dir, f"{mod}.so")

    if not os.path.exists(so_path):
        _build_so(so_path, world_size, hidden, mod, row_block)

    spec = importlib.util.spec_from_file_location(mod, so_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _build_so(so_path: str, world_size: int, hidden: int, mod: str, row_block: int):
    """Compile under an exclusive file lock + atomic rename, so N concurrent
    workers (mp.spawn cold-cache first run) produce exactly one nvcc build and
    never load a half-written .so."""
    import fcntl
    with open(so_path + ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if os.path.exists(so_path):  # another rank built it while we waited
                return
            # copy the shared header next to the source so its #include "sm120_common.cuh" resolves
            import shutil
            shutil.copy(_COMMON, os.path.join(_HERE, "sm120_common.cuh"))
            inc, lib = _nvcc_flags()
            tmp_path = f"{so_path}.tmp.{os.getpid()}"
            cmd = [
                "nvcc", os.path.join(_HERE, "tk_moe.cu"),
                "-std=c++20", "-O3", "--use_fast_math",
                "-lrt", "-lpthread", "-ldl", "-lcuda", "-lcudadevrt", "-lcudart_static",
                "--expt-extended-lambda", "--expt-relaxed-constexpr",
                "-forward-unknown-to-host-compiler", "-Xcompiler=-Wno-psabi",
                "-Xcompiler=-fno-strict-aliasing", "-DNDEBUG", "-lineinfo",
                "-shared", "-fPIC", "-diag-suppress", "3189",
                "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
                "-DTORCH_API_INCLUDE_EXTENSION_H", f"-DTORCH_EXTENSION_NAME={mod}",
                f"-DTK_NUM_DEVICES={world_size}", f"-DTK_HIDDEN={hidden}",
                f"-DTK_ROW_BLOCK={row_block}",
                f"-I{_TK_ROOT}/include", f"-I{_TK_ROOT}/prototype",
                *inc, *lib,
                "-ltorch_python", "-ltorch_cuda", "-ltorch_cpu", "-ltorch", "-lc10_cuda", "-lc10",
                "-DKITTENS_SM120", "-gencode", "arch=compute_120a,code=sm_120a",
                "-o", tmp_path,
            ]
            subprocess.run(cmd, check=True)
            os.replace(tmp_path, so_path)  # atomic: readers see old-or-new, never partial
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


if __name__ == "__main__":
    import torch  # noqa
    ws = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rb = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    m = build_and_load(ws, row_block=rb)
    print("built & loaded:", [x for x in dir(m) if not x.startswith("__")])
