# SPDX-License-Identifier: Apache-2.0
"""TP stage attribution (docs/09 methodology): time each stage of the tktp
run() in lockstep across ranks, plus GEMM-alone references, so the exposed
comm / drain cost of each fused stage is directly readable:

  L0 exposure = t(L0 fused) - t(L0 GEMM alone)     (AG + gate stalls)
  L1 exposure = t(L1 fused) - t(L1 GEMM alone)     (prered/push drain)

  python -m moe_bench.tools.time_tp_stages [ne] [iters] [tokens_per_rank]

Per-stage numbers are the MAX across ranks (slowest gates), mean over iters.
cuda.synchronize between stages perturbs overlap slightly but the fused
kernels themselves run unmodified.
"""
from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world, init_method, ne, iters, tokens):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)
    dist.all_reduce(torch.tensor([rank], device=device))

    from moe_bench.config import Distribution, MoEBenchConfig, ParallelMode, Precision, RoutingConfig
    from moe_bench.context import DistContext
    from moe_bench.data import make_problem, make_weights
    from moe_bench.tk_tp_scheme import TKFusedTP
    import torch.nn.functional as F

    cfg = MoEBenchConfig(
        hidden_size=4096, intermediate_size=3072, num_experts=ne, topk=8,
        parallel_mode=ParallelMode.TP, world_size=world, precision=Precision.BF16,
        num_tokens=[tokens], routing=RoutingConfig(distribution=Distribution.BALANCED),
        distributed=True, verify=False, device="cuda")
    ctx = DistContext(rank=rank, world_size=world, local_rank=rank,
                      device=device, group=None)
    weights = make_weights(cfg, rank)
    problem = make_problem(cfg, tokens, rank=rank, weights=weights)
    s = TKFusedTP()
    s.setup(problem, ctx)

    # scratch for GEMM-alone references (same shapes as the fused calls)
    ref_gateup = torch.empty_like(s.gateup_out)
    ref_expout = torch.empty_like(s.expert_out)

    stages = ["sched", "tok_copy", "L0_fused", "silu", "L1_fused", "final_red",
              "L0_gemm_alone", "L1_gemm_alone", "full_run"]
    acc = {k: 0.0 for k in stages}

    def timed(key, fn):
        dist.barrier()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        acc[key] += e0.elapsed_time(e1) * 1000  # us

    for _ in range(3):  # warmup (captures the sched graph too)
        s.run()
    torch.cuda.synchronize()

    for _ in range(iters):
        def st_sched():
            dist.all_gather_into_tensor(s._all_topk, s._topk_ids_local)
            dist.all_gather_into_tensor(s._all_w, s._topk_w_local)
            s._sched_graph.replay()
        timed("sched", st_sched)

        def st_copy():
            s._l0_seq += 1
            s.tk.pcie_device_barrier(s.barrier_l0, s._l0_seq)
            s.pre_tokens.data_.copy_(s.problem.hidden_states)
            s._l0_seq += 1
            s.tk.pcie_device_barrier(s.barrier_l0, s._l0_seq)
        timed("tok_copy", st_copy)

        def st_l0():
            if s.dispatch_mode == "push":
                s._l0_seq += 1
                s.l0_push_cnt.zero_()
                s.tk.moe_tp_dispatch_push_gemm(
                    s.pre_tokens, s.ag_staging, s.gathered, s.w_gateup,
                    s.gateup_out, s.padded, s.tp_slots, s.slack, s.push_order,
                    s.l0_push_cnt, s.barrier_l0, s.num_push_sms,
                    max(s.num_comm_sms - s.num_push_sms, 1),
                    s.num_padded_total, s.num_tokens, s._l0_seq)
            else:
                s.tk.moe_tp_dispatch_gemm(
                    s.pre_tokens, s.gathered, s.w_gateup, s.gateup_out, s.padded,
                    s.tp_slots, s.slack, s.pull_order, s.barrier_l0,
                    s.num_comm_sms, s.num_padded_total, s.num_tokens)
        timed("L0_fused", st_l0)

        timed("silu", lambda: torch.mul(
            F.silu(s.gateup_out[:, :s.inter]), s.gateup_out[:, s.inter:], out=s.act))

        def st_l1():
            s._l1_seq += 1
            s.combine_local_cnt.zero_()
            s.job_next.zero_()
            s.tk.moe_tp_gemm_prered_push(
                s.act, s.w2, s.expert_out, s.padded, s.combine_staging,
                s.prered_dst, s.tp_slots, s.prered_w, s.combine_local_cnt,
                s.push_expected_l1, s.job_order, s.job_next, s.barrier_l1,
                s.num_comm_sms, s.num_padded_total, s.num_tokens, s.num_jobs,
                s._l1_seq)
        timed("L1_fused", st_l1)

        timed("final_red", lambda: s.tk.moe_final_reduce_push(
            s.combine_staging, s.final_contrib, s.recv_from, s.combine_out,
            s.barrier_l1, s.num_tokens, s._l1_seq))

        # references: same GEMMs, no comm/prered (gathered/act already populated)
        timed("L0_gemm_alone", lambda: s.tk.grouped_gemm(
            s.gathered, s.w_gateup, ref_gateup, s.padded, 0))
        timed("L1_gemm_alone", lambda: s.tk.grouped_gemm(
            s.act, s.w2, ref_expout, s.padded, 0))

        timed("full_run", s.run)

    t = torch.tensor([acc[k] / iters for k in stages], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"\n== tktp stage attribution (NE={ne}, T={tokens}, iters={iters}, "
              f"dispatch={s.dispatch_mode}, comm_sms={s.num_comm_sms}, "
              f"push_sms={s.num_push_sms}, max over ranks, us) ==")
        for k, v in zip(stages, t.tolist()):
            print(f"  {k:14} {v:10.1f}")
        l0 = dict(zip(stages, t.tolist()))
        print(f"  -> L0 exposure {l0['L0_fused'] - l0['L0_gemm_alone']:10.1f}")
        print(f"  -> L1 exposure {l0['L1_fused'] - l0['L1_gemm_alone']:10.1f}")
        print(f"  -> stage sum   {sum(l0[k] for k in stages[:6]):10.1f} "
              f"(vs full_run {l0['full_run']:.1f})")
    dist.destroy_process_group()


def main():
    ne = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    world = 4
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mp.spawn(_worker, args=(world, init_method, ne, iters, tokens), nprocs=world, join=True)


if __name__ == "__main__":
    main()
