# SPDX-License-Identifier: Apache-2.0
"""编译并加载 microbench 探针扩展(probes.cu,纯 CUDA + torch,无 TK 依赖)。

torch.utils.cpp_extension.load 带增量编译:probes.cu 未变则直接复用
microbench/build/ 下的产物。
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_probes():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    from torch.utils.cpp_extension import load

    build_dir = os.path.join(_HERE, "build")
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="mb_probes",
        sources=[os.path.join(_HERE, "probes.cu")],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=os.environ.get("MB_BUILD_VERBOSE", "0") == "1",
    )


if __name__ == "__main__":
    m = load_probes()
    print("built & loaded:", [x for x in dir(m) if not x.startswith("__")])
