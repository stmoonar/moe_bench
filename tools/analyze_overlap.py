# SPDX-License-Identifier: Apache-2.0
"""Comm/compute overlap + fusion-penalty analysis for the DEFAULT tkfused path
(pull dispatch⊕gate+up GEMM  +  prered combine), at the default shape.

For each of the two fused stages we measure, at 4 ranks in lockstep on the real
dispatched shapes:

LAYER0 (dispatch ⊕ gate+up GEMM, N=2*inter):
  L0_gemm : pure grouped_gemm on `gathered`  (compute-only reference)
  L0_fused: moe_dispatch_gemm (pull tokens cross-card + gate+up GEMM fused)
  => fusion penalty = L0_fused - L0_gemm ; the cross-card pull is hidden under
     the GEMM iff L0_fused ≈ L0_gemm (well-overlapped).

LAYER1 (prered combine):
  L1_gemm  : pure grouped_gemm W2 on `act` (compute-only reference)
  L1_fused : moe_gemm_prered_fused (W2 GEMM + local pre-reduction) + barrier +
             moe_final_reduce (the full default combine)
  => fusion penalty = L1_fused - L1_gemm.

Also reports pure comm proxies: dispatch cross-card bytes and combine as measured
by their fused-minus-compute gaps.

  python -m moe_bench.tools.analyze_overlap  [num_experts]  [warmup] [iters]
"""
from __future__ import annotations

import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    s = []
    for _ in range(iters):
        dist.barrier()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        s.append(a.elapsed_time(b))
    return statistics.median(s)


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
    eoff = ctx.rank * cfg.num_local_experts
    npl = sch.num_padded_local
    H, inter = sch.H, sch.inter
    fuse = sch.fuse_gateup

    # keep pre_tokens fresh + peers ready (dispatch reads peer pre_tokens)
    def _prep():
        sch._l0_seq += 1; tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
        sch.pre_tokens.data_.copy_(problem.hidden_states)
        sch._l0_seq += 1; tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
    _prep()

    # ---- LAYER0 ----
    gw = sch.w_gateup if fuse else sch.w_gate
    go = sch.gateup_out if fuse else sch.gate_out
    N0 = gw.shape[2]                       # 2*inter if fused else inter
    flop0 = 2.0 * npl * H * N0             # gate(+up): (npl,H)@(H,N0)
    def L0_gemm():                          # pure GEMM on already-gathered tokens
        tk.grouped_gemm(sch.gathered.data_, gw, go, sch.padded, eoff)
    def L0_fused():                         # cross-card pull + gate+up GEMM fused
        tk.moe_dispatch_gemm(sch.pre_tokens, sch.gathered.data_, gw, go,
                             sch.padded, sch.disp_idx, sch.barrier_l0,
                             sch.num_comm_sms, sch.num_padded_local)

    # ---- LAYER1 (prered combine) ----
    torch.nn.init.normal_(sch.act, std=0.02)
    flop1 = 2.0 * npl * inter * H          # W2: (npl,inter)@(inter,H)
    def L1_gemm():                          # pure W2 GEMM
        tk.grouped_gemm(sch.act, sch.w2, sch.expert_out.data_, sch.padded, eoff)
    def L1_fused():                         # W2⊕prered + barrier + final_reduce
        sch._l1_seq += 1
        tk.moe_gemm_prered_fused(sch.act, sch.w2, sch.expert_out, sch.padded,
                                 sch.partials, sch.prered_dst, sch.prered_slots,
                                 sch.prered_w, sch.barrier_l1, sch.num_comm_sms,
                                 sch.num_padded_local, sch.num_tokens, sch.num_jobs, sch._l1_seq)
        sch._l0_seq += 1; tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
        tk.moe_final_reduce(sch.partials, sch.final_contrib, sch.combine_out,
                            sch.barrier_l0, sch.num_tokens)

    def L1_prered():                        # W2⊕prered only (no barrier/final_reduce)
        sch._l1_seq += 1
        tk.moe_gemm_prered_fused(sch.act, sch.w2, sch.expert_out, sch.padded,
                                 sch.partials, sch.prered_dst, sch.prered_slots,
                                 sch.prered_w, sch.barrier_l1, sch.num_comm_sms,
                                 sch.num_padded_local, sch.num_tokens, sch.num_jobs, sch._l1_seq)

    t = {}
    t["L0_gemm"] = _time(L0_gemm, warmup, iters)
    t["L1_prered"] = _time(L1_prered, warmup, iters)
    _prep(); t["L0_fused"] = _time(lambda: (_prep(), L0_fused()), warmup, iters)
    t["L1_gemm"] = _time(L1_gemm, warmup, iters)
    t["L1_fused"] = _time(L1_fused, warmup, iters)

    # standalone comm proxies (no GEMM):
    #  - final_reduce alone = source-side cross-card partial gather (T6-v1 replaces
    #    the source pull with expert-side push; this is the pull cost it competes with)
    def L1_finalreduce():
        sch._l0_seq += 1; tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
        tk.moe_final_reduce(sch.partials, sch.final_contrib, sch.combine_out,
                            sch.barrier_l0, sch.num_tokens)
    # populate partials once so final_reduce reads real data
    L1_fused(); torch.cuda.synchronize(); dist.barrier()
    t["L1_finalreduce"] = _time(L1_finalreduce, warmup, iters)

    out_list.append({"rank": rank, "npl": npl, "N0": N0, "flop0": flop0,
                     "flop1": flop1, "fuse": fuse, **t})
    del sch, problem; torch.cuda.empty_cache()
    dist.destroy_process_group()


def _tf(flop, ms):
    return flop / (ms * 1e-3) / 1e12


def main():
    world = 4
    ne = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    from vllm.utils.network_utils import get_open_port
    im = f"tcp://localhost:{get_open_port()}"
    mgr = mp.Manager(); out_list = mgr.list()
    mp.spawn(_worker, args=(world, im, ne, warmup, iters, out_list), nprocs=world, join=True)

    rows = sorted(out_list, key=lambda r: r["rank"])
    rm = max(rows, key=lambda r: r["L1_fused"])
    print(f"\n===== comm/compute overlap + fusion penalty  NE={ne}  world={world} "
          f"(max-time rank {rm['rank']}, npl={rm['npl']}, fuse_gateup={rm['fuse']}, N0={rm['N0']}) =====")
    for lab, gemm, fused, flop in [
            ("LAYER0 dispatch⊕gate+up", "L0_gemm", "L0_fused", "flop0"),
            ("LAYER1 prered combine  ", "L1_gemm", "L1_fused", "flop1")]:
        g = rm[gemm]*1e3; fu = rm[fused]*1e3; pen = fu - g
        print(f"\n  {lab}")
        print(f"    pure compute (GEMM)   : {g:8.1f} us   {_tf(rm[flop],rm[gemm]):6.1f} TFLOP/s")
        print(f"    fused (comm+compute)  : {fu:8.1f} us   {_tf(rm[flop],rm[fused]):6.1f} TFLOP/s")
        print(f"    fusion penalty        : {pen:8.1f} us   (+{pen/g*100:4.1f}% over pure compute)")
    # layer1 standalone comm (source-side partial gather) and overlap estimate
    fr = rm["L1_finalreduce"]*1e3
    l1pen = (rm["L1_fused"] - rm["L1_gemm"])*1e3
    print(f"\n  LAYER1 comm detail")
    print(f"    W2⊕prered only (no barrier/final)                    : {rm['L1_prered']*1e3:8.1f} us")
    print(f"    final_reduce alone (source cross-card partial gather): {fr:8.1f} us")
    print(f"    layer1 fusion penalty (un-overlapped comm)           : {l1pen:8.1f} us")
    preredpen = (rm["L1_prered"] - rm["L1_gemm"])*1e3
    print(f"      of which W2⊕prered penalty (scatter+signal)        : {preredpen:8.1f} us")
    print(f"      of which barrier + final_reduce                    : {l1pen - preredpen:8.1f} us")
    hid = max(fr - l1pen, 0.0)
    print(f"    => hidden under W2 GEMM ≈ {hid:6.1f} us  ({(hid/fr*100 if fr>0 else 0):4.1f}% of standalone comm)")
    print()


if __name__ == "__main__":
    main()
