# SPDX-License-Identifier: Apache-2.0
"""T1 (docs/11) layer1 GEMM attribution timer.

Decomposes the layer1 W2⊕combine slowdown (71 vs ~160 TFLOP/s, docs/11 §0.1a)
into three additive costs, all measured on the REAL W2 GEMM shape at 4 ranks in
lockstep, so the numbers close: t_E - t_A == (SM-yield) + (fence/signal) + (L2/contention).

  A: plain grouped_gemm @ full SM count (110)      -> ~160 TFLOP/s reference
  B: plain grouped_gemm @ comp SM count (94)       -> SM yield  = t_B - t_A
  D: fused kernel, num_source_tokens=0 (94 GEMM    -> fence/sig = t_D - t_B
     blocks + combine_signal_epilogue, no combine
     blocks -> can't hang)
  E: full fused kernel (94 GEMM + 512 combine)     -> L2/contend = t_E - t_D
                                                       (== the 71 TFLOP/s baseline)

  python -m moe_bench.tools.time_layer1  <num_experts>  [warmup] [iters]
"""
from __future__ import annotations

import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _time_call(fn, warmup, iters):
    """Median ms over `iters` timed runs of `fn`, kept in lockstep across ranks."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    samples = []
    for _ in range(iters):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _worker(rank, world, init_method, ne, warmup, iters, out_list):
    from moe_bench.config import (Distribution, MoEBenchConfig, ParallelMode,
                                  Precision, RoutingConfig)
    from moe_bench.context import DistContext
    from moe_bench.data import make_problem, make_weights
    from moe_bench.tk_scheme import TKFusedEP

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)
    dist.all_reduce(torch.tensor([rank], device=device))

    cfg = MoEBenchConfig(hidden_size=4096, intermediate_size=3072, num_experts=ne, topk=8,
                         parallel_mode=ParallelMode.EP, world_size=world, precision=Precision.BF16,
                         num_tokens=[512], routing=RoutingConfig(distribution=Distribution.BALANCED),
                         distributed=True, use_cuda_graph=False, seed=0, verify=False, device="cuda")
    ctx = DistContext(rank=rank, world_size=world, local_rank=rank, device=device, group=None)
    problem = make_problem(cfg, 512, rank=rank, weights=make_weights(cfg, rank=rank))
    sch = TKFusedEP(); sch.setup(problem, ctx)
    tk = sch.tk

    sm = torch.cuda.get_device_properties(device).multi_processor_count  # 110
    comp = sm - sch.num_comm_sms                                          # 94
    eoff = ctx.rank * cfg.num_local_experts
    npl = sch.num_padded_local
    flop = 2.0 * npl * sch.inter * sch.H  # W2: (npl, inter) @ (inter, H)

    # Pre-fill act with the real dispatched activations so all four points run on
    # identical inputs (A/B/D/E share sch.act, sch.w2, sch.expert_out).
    torch.nn.init.normal_(sch.act, std=0.02)

    def run_A():
        tk.grouped_gemm_nb(sch.act, sch.w2, sch.expert_out.data_, sch.padded, eoff, sm)
    def run_B():
        tk.grouped_gemm_nb(sch.act, sch.w2, sch.expert_out.data_, sch.padded, eoff, comp)
    def run_D():
        sch._l1_seq += 1
        tk.moe_gemm_combine_fused(sch.act, sch.w2, sch.expert_out, sch.padded,
                                  sch.combine_out, sch.comb_idx, sch.combine_w,
                                  sch.barrier_l1, sch.num_comm_sms,
                                  sch.num_padded_local, 0, sch._l1_seq)
    def run_E():
        sch._l1_seq += 1
        tk.moe_gemm_combine_fused(sch.act, sch.w2, sch.expert_out, sch.padded,
                                  sch.combine_out, sch.comb_idx, sch.combine_w,
                                  sch.barrier_l1, sch.num_comm_sms,
                                  sch.num_padded_local, sch.num_tokens, sch._l1_seq)
    # F: combine-only, barrier PRE-satisfied. Run one full fused E OUTSIDE timing
    # to populate expert_out + leave barrier_l1 at a fixed seq S; then time
    # moe_combine_only at that SAME S. combine_only reads expert_out + waits on S
    # but writes NEITHER, so every repeat passes instantly and reads valid data
    # -> pure cross-card gather+reduce cost (no GEMM, no signal-wait). Isolates
    # combine bandwidth from HOL-wait.
    run_E()  # populate + satisfy barrier at sch._l1_seq
    torch.cuda.synchronize(); dist.barrier()
    fseq = sch._l1_seq
    def run_F():
        tk.moe_combine_only(sch.act, sch.w2, sch.expert_out, sch.padded,
                            sch.combine_out, sch.comb_idx, sch.combine_w,
                            sch.barrier_l1, sch.num_comm_sms,
                            sch.num_padded_local, sch.num_tokens, fseq)

    tA = _time_call(run_A, warmup, iters)
    tB = _time_call(run_B, warmup, iters)
    tD = _time_call(run_D, warmup, iters)
    tF = _time_call(run_F, warmup, iters)
    # E last (it advances _l1_seq past fseq; run F before it so fseq stays valid).
    tE = _time_call(run_E, warmup, iters)

    out_list.append({"rank": rank, "npl": npl, "flop": flop,
                     "tA": tA, "tB": tB, "tD": tD, "tE": tE, "tF": tF,
                     "sm": sm, "comp": comp, "comm": sch.num_comm_sms})
    del sch, problem; torch.cuda.empty_cache()
    dist.destroy_process_group()


def _tflops(flop, ms):
    return flop / (ms * 1e-3) / 1e12


def main():
    world = 4
    ne = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mgr = mp.Manager(); out_list = mgr.list()
    mp.spawn(_worker, args=(world, init_method, ne, warmup, iters, out_list),
             nprocs=world, join=True)

    rows = sorted(list(out_list), key=lambda r: r["rank"])
    r0 = rows[0]
    print(f"\n===== layer1 W2⊕combine attribution  NE={ne}  ({r0['sm']} SM, "
          f"{r0['comp']} comp / {r0['comm']} comm)  world={world} =====")
    print(f"{'rank':>4} {'npl':>6} {'A@110':>9} {'B@94':>9} {'D+fence':>9} {'E full':>9} {'F comb':>9}"
          f"  {'TFLOP/s A':>9} {'TFLOP/s E':>9}")
    for r in rows:
        print(f"{r['rank']:>4} {r['npl']:>6} "
              f"{r['tA']*1e3:>8.1f}u {r['tB']*1e3:>8.1f}u {r['tD']*1e3:>8.1f}u {r['tE']*1e3:>8.1f}u {r['tF']*1e3:>8.1f}u"
              f"  {_tflops(r['flop'],r['tA']):>9.1f} {_tflops(r['flop'],r['tE']):>9.1f}")

    # attribution on the max-time rank (the lockstep-critical one)
    rm = max(rows, key=lambda r: r["tE"])
    tA, tB, tD, tE = rm["tA"], rm["tB"], rm["tD"], rm["tE"]
    gap = tE - tA
    sm_yield = tB - tA
    fence = tD - tB
    l2 = tE - tD
    print(f"\n--- attribution (max-tE rank {rm['rank']}, npl={rm['npl']}) ---")
    print(f"  reference GEMM (A@110)     : {tA*1e3:8.1f} us   {_tflops(rm['flop'],tA):6.1f} TFLOP/s")
    print(f"  fused W2⊕combine (E)       : {tE*1e3:8.1f} us   {_tflops(rm['flop'],tE):6.1f} TFLOP/s")
    print(f"  total slowdown  t_E - t_A  : {gap*1e3:8.1f} us  (100%)")
    if gap > 0:
        print(f"    SM yield    (t_B - t_A)  : {sm_yield*1e3:8.1f} us  ({sm_yield/gap*100:5.1f}%)")
        print(f"    fence/sig   (t_D - t_B)  : {fence*1e3:8.1f} us  ({fence/gap*100:5.1f}%)")
        print(f"    L2/contend  (t_E - t_D)  : {l2*1e3:8.1f} us  ({l2/gap*100:5.1f}%)")
    print(f"  closure check: (SMyield+fence+L2)={ (sm_yield+fence+l2)*1e3:.1f}us vs gap={gap*1e3:.1f}us")
    print(f"  combine-only (F, waits pre-satisfied): {rm['tF']*1e3:8.1f} us  "
          f"= pure gather+reduce; HOL-wait ≈ (t_E - t_D - t_F) = {(l2-rm['tF'])*1e3:.1f} us")


if __name__ == "__main__":
    main()
