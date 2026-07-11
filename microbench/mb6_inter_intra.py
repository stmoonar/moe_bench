# SPDX-License-Identifier: Apache-2.0
"""MB6 inter-SM vs intra-SM 通算编排对比(合成探针,单进程多卡)。

当前 TK 融合 kernel 是 inter-SM 编排(整 block 专职通信或计算)。本探针用
同一份通信量(跨卡 float4 copy)+ 同一份计算量(FMA 循环,按计算线程数均摊)
对比两种编排:

  inter-SM : grid 里 k 个 block 专职通信,其余 (SM-k) 个 block 专职计算
  intra-SM : 每个 block 前 w 个 warp 通信,其余 (8-w) 个 warp 计算(全部 SM 参与)

每种配置测三个数:
  comm_only    : 只通信(计算量=0)—— 该通信资源配置能达到的 PCIe 带宽
  compute_only : 只计算(通信量=0)—— 该编排下计算侧的损失(让渡/占用 warp)
  both         : 通信+计算同时     —— 干扰后的 makespan

理想 makespan = max(comm_only, compute_only);both 与它的差 = 干扰损失。
动态 smem 占位强制 1 block/SM,保证编排语义干净。计算代理是 fp32 FMA
(SM 数/warp 数敏感,与 tensor-core GEMM 的差异见 README 口径说明;真实
GEMM 的 SM 让渡损失由 mb5 的 grouped_gemm_nb 给出)。

  MB_OVL_BYTES(默认 32MB)/ MB_OVL_COMPUTE_MS(默认 1.0)可调。
  python -m moe_bench.microbench.mb6_inter_intra
"""
from __future__ import annotations

import os

import torch

from . import common
from .build_probes import load_probes

K_SWEEP = [1, 2, 4, 8, 16, 24, 32]
W_SWEEP = [1, 2, 4]
THREADS = 256          # 8 warps / block
SMEM_BYTES = 60 * 1024  # 2 block 需要 120KB > sm120 的 ~100KB → 强制 1 block/SM


def main():
    ext = load_probes()
    ext.enable_p2p()
    ndev = torch.cuda.device_count()
    assert ndev >= 2, "mb6 needs >= 2 visible GPUs"
    sm = int(ext.sm_count(0))
    warmup, iters = common.warmup_iters(), common.bench_iters()

    comm_bytes = int(os.environ.get("MB_OVL_BYTES", str(32 << 20)))
    comm_bytes -= comm_bytes % 16
    target_ms = float(os.environ.get("MB_OVL_COMPUTE_MS", "1.0"))

    # 通信缓冲:dev0 为执行卡;pull = 读 dev1,push = 写 dev1
    n_f32 = comm_bytes // 4
    dst0 = torch.empty(n_f32, dtype=torch.float32, device=0)
    src0 = torch.randn(n_f32, dtype=torch.float32, device=0)
    dst1 = torch.empty(n_f32, dtype=torch.float32, device=1)
    src1 = torch.randn(n_f32, dtype=torch.float32, device=1)
    empty0 = torch.empty(0, dtype=torch.float32, device=0)
    sink = torch.zeros(1, dtype=torch.float32, device=0)

    def run(dst, src, fma, comm_blocks, comm_warps, mode):
        ext.overlap(dst, src, sink, 0, fma, comm_blocks, comm_warps, mode,
                    sm, THREADS, SMEM_BYTES)

    # ---- 标定计算量:compute-only(全 SM 全 warp)≈ target_ms ----
    total_fma = int(2e9)
    t = common.single_dev_median_ms(
        lambda: run(empty0, empty0, total_fma, 0, 0, 0), 0, 3, 10)
    total_fma = max(int(total_fma * target_ms / max(t, 1e-3)), 10**6)
    t_comp = common.single_dev_median_ms(
        lambda: run(empty0, empty0, total_fma, 0, 0, 0), 0, warmup, iters)
    print(f"[mb6] SM={sm}  comm_bytes={comm_bytes>>20}MB  "
          f"compute_only(full)={t_comp*1e3:.0f}us (calibrated)", flush=True)

    rows = [{"kind": "compute_only_full", "ms": t_comp, "sm": sm,
             "total_fma_iters": total_fma, "comm_bytes": comm_bytes}]

    for direction, (d, s) in [("pull", (dst0, src1)), ("push", (dst1, src0))]:
        # ---- inter-SM ----
        for k in K_SWEEP:
            if k >= sm:
                continue
            t_comm = common.single_dev_median_ms(
                lambda: run(d, s, 0, k, 0, 0), 0, warmup, iters)
            t_cpu = common.single_dev_median_ms(
                lambda: run(empty0, empty0, total_fma, k, 0, 0), 0, warmup, iters)
            t_both = common.single_dev_median_ms(
                lambda: run(d, s, total_fma, k, 0, 0), 0, warmup, iters)
            ideal = max(t_comm, t_cpu)
            rows.append({
                "kind": "inter", "direction": direction, "comm_blocks": k,
                "comm_only_ms": t_comm,
                "comm_gbps": comm_bytes / (t_comm * 1e-3) / 1e9,
                "compute_only_ms": t_cpu, "both_ms": t_both,
                "ideal_ms": ideal, "interference_ms": t_both - ideal,
                "compute_slowdown": t_cpu / t_comp,
            })
            print(f"[mb6] inter {direction} k={k:2d}  comm={t_comm*1e3:7.0f}us "
                  f"({rows[-1]['comm_gbps']:5.1f}GB/s)  comp={t_cpu*1e3:7.0f}us  "
                  f"both={t_both*1e3:7.0f}us  ideal={ideal*1e3:7.0f}us", flush=True)

        # ---- intra-SM ----
        for w in W_SWEEP:
            t_comm = common.single_dev_median_ms(
                lambda: run(d, s, 0, 0, w, 1), 0, warmup, iters)
            t_cpu = common.single_dev_median_ms(
                lambda: run(empty0, empty0, total_fma, 0, w, 1), 0, warmup, iters)
            t_both = common.single_dev_median_ms(
                lambda: run(d, s, total_fma, 0, w, 1), 0, warmup, iters)
            ideal = max(t_comm, t_cpu)
            rows.append({
                "kind": "intra", "direction": direction, "comm_warps": w,
                "comm_only_ms": t_comm,
                "comm_gbps": comm_bytes / (t_comm * 1e-3) / 1e9,
                "compute_only_ms": t_cpu, "both_ms": t_both,
                "ideal_ms": ideal, "interference_ms": t_both - ideal,
                "compute_slowdown": t_cpu / t_comp,
            })
            print(f"[mb6] intra {direction} w={w}   comm={t_comm*1e3:7.0f}us "
                  f"({rows[-1]['comm_gbps']:5.1f}GB/s)  comp={t_cpu*1e3:7.0f}us  "
                  f"both={t_both*1e3:7.0f}us  ideal={ideal*1e3:7.0f}us", flush=True)

    meta = common.result_meta({
        "test": "mb6_inter_intra", "comm_bytes": comm_bytes,
        "total_fma_iters": total_fma, "threads_per_block": THREADS,
        "smem_bytes": SMEM_BYTES, "sm": sm,
        "note": "计算代理为 fp32 FMA;真实 GEMM 的 SM 让渡损失见 mb5",
    })
    common.write_json("mb6_inter_intra", meta, rows)


if __name__ == "__main__":
    main()
