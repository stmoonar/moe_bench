# SPDX-License-Identifier: Apache-2.0
"""Step-4 protocol validation (docs/09 §5.4 — the R1 decision point).

Reuses TKFusedEP.setup() to build real symmetric buffers + schedules, then:

  1. Reference: run the verified PULL dispatch once, snapshot each card's
     gathered buffer (the ground-truth dispatched layout).
  2. push3-only x30: zero gathered + local_cnt, bump seq, barrier, run ONLY the
     push blocks (no GEMM/gate → cannot hang), barrier, then check on each card:
       (a) gathered checksum == pull reference   (data plane correct)
       (b) barrier[2+s][rb] == seq   iff gate_expected[rb][s] > 0
           and < seq otherwise       (single-writer signal reconciliation)
     30 iters to catch the §3.4 "signal before data" intermittent race.

Run from the PARENT dir (AGENTS.md):
  python -m moe_bench.tools.validate_push3   [num_experts]   [iters]
"""
from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROW_BLOCK = 128


def _worker(rank, world, init_method, num_experts, iters, out_list):
    import dataclasses

    from moe_bench.config import MoEBenchConfig, ParallelMode, Precision
    from moe_bench.context import DistContext
    from moe_bench.data import make_problem, make_weights
    from moe_bench.tk_scheme import TKFusedEP

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)
    dist.all_reduce(torch.tensor([rank], device=device))

    cfg = MoEBenchConfig(hidden_size=4096, intermediate_size=3072, num_experts=num_experts,
                         topk=8, parallel_mode=ParallelMode.EP, world_size=world,
                         precision=Precision.BF16, num_tokens=[512], distributed=True,
                         verify=False, device="cuda", seed=0)
    ctx = DistContext(rank=rank, world_size=world, local_rank=rank, device=device, group=None)
    weights = make_weights(cfg, rank=rank)
    problem = make_problem(cfg, 512, rank=rank, weights=weights)
    sch = TKFusedEP()
    sch.setup(problem, ctx)

    npl = sch.num_padded_local

    # ---- (0) reference: verified PULL dispatch fills gathered ----
    sch.gathered.data_.zero_()
    sch._l0_seq += 1; sch.tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
    sch.pre_tokens.data_.copy_(problem.hidden_states)
    sch._l0_seq += 1; sch.tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
    sch.tk.moe_dispatch_gemm(sch.pre_tokens, sch.gathered.data_, sch.w_gate, sch.gate_out,
                             sch.padded, sch.disp_idx, sch.barrier_l0,
                             sch.num_comm_sms, npl)
    torch.cuda.synchronize()
    ref_gathered = sch.gathered.data_[:npl].clone()

    gate_exp_cpu = sch.gate_expected.cpu()  # (nblk_local, world)
    nblk_local = gate_exp_cpu.shape[0]

    push3_seq = 100  # start well above any pull barrier seq; monotonic per iter
    fails = []
    for it in range(iters):
        push3_seq += 1
        sch.gathered.data_.zero_()
        sch.push_local_cnt.zero_()
        # make sure pre_tokens is fresh + all cards ready before any push
        sch._l0_seq += 1; sch.tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
        sch.pre_tokens.data_.copy_(problem.hidden_states)
        sch._l0_seq += 1; sch.tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)

        sch.tk.moe_dispatch_push3_only(
            sch.pre_tokens, sch.gathered, sch.w_gate, sch.gate_out, sch.padded,
            sch.push_idx, sch.push_src, sch.push_cnt_idx, sch.push_local_cnt,
            sch.push_expected, sch.gate_expected, sch.barrier_l0,
            sch.num_push, push3_seq)
        # all peers' pushes + signals landed
        sch._l0_seq += 1; sch.tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
        torch.cuda.synchronize()

        # (a) data plane: gathered must match the pull reference exactly
        got = sch.gathered.data_[:npl]
        if not torch.equal(got, ref_gathered):
            ndiff = int((got != ref_gathered).any(dim=1).sum().item())
            fails.append(f"iter {it}: gathered MISMATCH ({ndiff}/{npl} rows differ)")
            continue

        # (b) signal reconciliation: barrier[2+s][rb] == seq iff gate_expected>0
        bar = sch.barrier_l0.data_.cpu()  # (2+world, bar_cols)
        sig_bad = 0
        for rb in range(nblk_local):
            for s in range(world):
                v = int(bar[2 + s, rb].item())
                exp_signaled = int(gate_exp_cpu[rb, s].item()) > 0
                if exp_signaled and v != push3_seq:
                    sig_bad += 1
                    if len(fails) < 10:
                        fails.append(f"iter {it}: MISSING signal rb={rb} s={s} val={v} want={push3_seq}")
                elif (not exp_signaled) and v == push3_seq:
                    sig_bad += 1
                    if len(fails) < 10:
                        fails.append(f"iter {it}: SPURIOUS signal rb={rb} s={s} val={v}")
        # (nothing appended if sig_bad==0)

    # gather pass/fail across ranks
    nfail = torch.tensor(len(fails), device=device)
    dist.all_reduce(nfail, op=dist.ReduceOp.SUM)
    if rank == 0:
        out_list.append(int(nfail.item()))
    # print own failures (first few)
    for f in fails[:5]:
        print(f"  [rank {rank}] {f}", flush=True)

    del sch, problem
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def main():
    world = 4
    num_experts = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mgr = mp.Manager()
    out_list = mgr.list()
    mp.spawn(_worker, args=(world, init_method, num_experts, iters, out_list),
             nprocs=world, join=True)
    total_fail = out_list[0] if out_list else -1
    status = "OK" if total_fail == 0 else "FAIL"
    print(f"[push3 validate NE={num_experts} iters={iters}] total_failures={total_fail} -> {status}")


if __name__ == "__main__":
    main()
