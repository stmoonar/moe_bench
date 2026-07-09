# SPDX-License-Identifier: Apache-2.0
"""T6-v0 correctness adjudication (docs/13 §4): the pre-reduction combine must
produce the same layer1 output as the verified PULL combine.

Reuses TKFusedEP.setup() (with TK_COMBINE=prered so the partial buffer exists)
to build real symmetric buffers + both schedules, then per iteration:

  1. golden: run the verified moe_gemm_combine_fused -> combine_out_ref.
  2. prered: run moe_gemm_prered_fused + pcie_device_barrier + moe_final_reduce
     -> combine_out, on the SAME expert_out/act.
  3. compare: combine_out ~= combine_out_ref (rtol — prered does one extra bf16
     rounding: FP32 partial -> bf16 -> FP32 final sum vs combine's single FP32
     accumulate; expected same magnitude as the ~4.3e-3 reference rel_err).

30 iters × NE∈{64,128,256} to catch any cross-kernel publish race. act and
expert_out are produced once by the real layer0+up+silu path so the GEMM in
both combine variants sees identical inputs.

Run from the PARENT dir (AGENTS.md):
  python -m moe_bench.tools.validate_prered   [num_experts]   [iters]
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def _worker(rank, world, init_method, num_experts, iters, out_list):
    os.environ["TK_COMBINE"] = "prered"
    os.environ["TK_FUSE_GATEUP"] = "0"  # this tool drives w_gate/disp_idx directly
    os.environ["TK_GPU_SCHED"] = "0"    # host schedule (tool builds act via w_gate)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)
    dist.all_reduce(torch.tensor([rank], device=device))

    from moe_bench.config import MoEBenchConfig, ParallelMode, Precision
    from moe_bench.context import DistContext
    from moe_bench.data import make_problem, make_weights
    from moe_bench.tk_scheme import TKFusedEP

    cfg = MoEBenchConfig(hidden_size=7168, intermediate_size=2048, num_experts=num_experts,
                         topk=8, parallel_mode=ParallelMode.EP, world_size=world,
                         precision=Precision.BF16, num_tokens=[512], distributed=True,
                         verify=False, device="cuda", seed=0)
    ctx = DistContext(rank=rank, world_size=world, local_rank=rank, device=device, group=None)
    weights = make_weights(cfg, rank=rank)
    problem = make_problem(cfg, 512, rank=rank, weights=weights)
    sch = TKFusedEP()
    sch.setup(problem, ctx)
    tk = sch.tk

    def _produce_act():
        # real layer0 + up + silu to fill gathered/expert_out inputs deterministically
        sch._l0_seq += 1; tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
        sch.pre_tokens.data_.copy_(problem.hidden_states)
        sch._l0_seq += 1; tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
        tk.moe_dispatch_gemm(sch.pre_tokens, sch.gathered.data_, sch.w_gate, sch.gate_out,
                             sch.padded, sch.disp_idx, sch.barrier_l0,
                             sch.num_comm_sms, sch.num_padded_local)
        tk.grouped_gemm(sch.gathered.data_, sch.w_up, sch.up_out, sch.padded,
                        ctx.rank * cfg.num_local_experts)
        torch.mul(F.silu(sch.gate_out), sch.up_out, out=sch.act)

    npl = sch.num_padded_local
    max_rel = 0.0
    fails = []
    for it in range(iters):
        _produce_act()
        # golden: verified pull combine
        sch._l1_seq += 1
        tk.moe_gemm_combine_fused(sch.act, sch.w2, sch.expert_out, sch.padded,
                                  sch.combine_out, sch.comb_idx, sch.combine_w,
                                  sch.barrier_l1, sch.num_comm_sms,
                                  npl, sch.num_tokens, sch._l1_seq)
        torch.cuda.synchronize()
        ref = sch.combine_out.clone()

        # prered path on the SAME expert_out/act (recompute expert_out via prered GEMM)
        sch.combine_out.zero_()
        sch.partials.data_.zero_()
        sch._l1_seq += 1
        tk.moe_gemm_prered_fused(sch.act, sch.w2, sch.expert_out, sch.padded,
                                 sch.partials, sch.prered_dst, sch.prered_slots,
                                 sch.prered_w, sch.barrier_l1, sch.num_comm_sms,
                                 npl, sch.num_tokens, sch.num_jobs, sch._l1_seq)
        sch._l0_seq += 1; tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
        tk.moe_final_reduce(sch.partials, sch.final_contrib, sch.combine_out,
                            sch.barrier_l0, sch.num_tokens)
        torch.cuda.synchronize()
        got = sch.combine_out

        denom = ref.abs().amax().clamp_min(1e-6)
        rel = (got - ref).abs().amax().item() / denom.item()
        max_rel = max(max_rel, rel)
        if rel > 2e-2:  # generous: one extra bf16 rounding over the 4.3e-3 baseline
            nbad = int(((got - ref).abs() > 0.05 * denom).any(dim=1).sum().item())
            fails.append(f"iter {it}: rel_err={rel:.4e} ({nbad}/{sch.num_tokens} rows off)")

    nfail = torch.tensor(len(fails), device=device)
    dist.all_reduce(nfail, op=dist.ReduceOp.SUM)
    mr = torch.tensor(max_rel, device=device)
    dist.all_reduce(mr, op=dist.ReduceOp.MAX)
    if rank == 0:
        out_list.append((int(nfail.item()), float(mr.item())))
    for f in fails[:5]:
        print(f"  [rank {rank}] {f}", flush=True)

    del sch, problem
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def main():
    world = 4
    num_experts = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    from vllm.utils.network_utils import get_open_port
    nes = (num_experts,) if num_experts else (64, 128, 256)
    init_method = f"tcp://localhost:{get_open_port()}"
    mgr = mp.Manager()
    out_list = mgr.list()
    for ne in nes:
        out_list[:] = []
        mp.spawn(_worker, args=(world, init_method, ne, iters, out_list),
                 nprocs=world, join=True)
        total_fail, max_rel = out_list[0] if out_list else (-1, -1.0)
        status = "OK" if total_fail == 0 else "FAIL"
        print(f"[prered validate NE={ne} iters={iters}] total_failures={total_fail} "
              f"max_rel_err={max_rel:.4e} -> {status}")
        init_method = f"tcp://localhost:{get_open_port()}"


if __name__ == "__main__":
    main()
