# SPDX-License-Identifier: Apache-2.0
"""Single-GPU ncu probe (docs/11 T1): the W2 GEMM shape run as plain grouped_gemm
and as the fused GEMM+fence epilogue (num_source_tokens=0, no combine blocks, no
peer access -> single process is fine). Confirms via counters that the GEMM is
compute-bound, not membar-stalled, and that the fence epilogue adds ~0.

  ncu --set full --kernel-name-base demangled -k 'kernel' \
      python -m moe_bench.tools.ncu_gemm_probe <ne> <which:A|D>
"""
from __future__ import annotations
import sys, torch


def main():
    ne = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    which = sys.argv[2] if len(sys.argv) > 2 else "A"
    world = 4
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    from moe_bench.kernels.tk import build as b
    tk = b.build_and_load(world, hidden=4096)

    H, inter = 4096, 3072
    e_local = ne // world
    # per-expert tokens: 512*world*topk / ne, padded up to ROW_BLOCK=128
    per_real = 512 * world * 8 // ne
    per = ((per_real + 127) // 128) * 128
    npl = e_local * per
    padded = torch.full((ne,), per, dtype=torch.int32, device=dev)
    act = torch.randn(npl, inter, dtype=torch.bfloat16, device=dev) * 0.02
    w2 = torch.randn(e_local, inter, H, dtype=torch.bfloat16, device=dev) * 0.02
    sm = torch.cuda.get_device_properties(dev).multi_processor_count

    if which == "A":
        for _ in range(3):
            tk.grouped_gemm_nb(act, w2, torch.empty(npl, H, dtype=torch.bfloat16, device=dev),
                               padded, 0, sm)
        torch.cuda.synchronize()
        tk.grouped_gemm_nb(act, w2, torch.empty(npl, H, dtype=torch.bfloat16, device=dev),
                           padded, 0, sm)
    else:  # D: fused GEMM + fence epilogue, no combine (single-process safe)
        TK = tk.TKParallelTensor
        expert_out = TK((npl, H), dtype=torch.bfloat16, local_rank=0, local_world_size=1, multicast=False)
        bar = TK((1 + 1, npl // 128 + 32), dtype=torch.int, local_rank=0, local_world_size=1, multicast=False)
        bar.data_.zero_()
        cout = torch.zeros(512, H, dtype=torch.bfloat16, device=dev)
        cidx = torch.full((512 * 8, 2), -1, dtype=torch.int32, device=dev)
        cw = torch.zeros(512 * 8, 1, dtype=torch.float32, device=dev)
        seq = 0
        for _ in range(3):
            seq += 1
            tk.moe_gemm_combine_fused(act, w2, expert_out, padded, cout, cidx, cw, bar,
                                      16, npl, 0, seq)
        torch.cuda.synchronize()
        seq += 1
        tk.moe_gemm_combine_fused(act, w2, expert_out, padded, cout, cidx, cw, bar, 16, npl, 0, seq)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
