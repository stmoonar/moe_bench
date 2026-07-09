# SPDX-License-Identifier: Apache-2.0
"""Step-6 dispatch-only isolation timer (docs/09 §5.6, docs/08 口径).

Times ONLY the layer0 dispatch step (dispatch⊕gate GEMM for pull/push3; the
three-stage push→barrier→GEMM for push2) in isolation, ranks kept in lockstep
by a cross-device barrier each iteration. Reports max-over-ranks median us.

  python -m moe_bench.tools.time_dispatch  <num_experts>  [warmup] [iters]

Honors TK_DISPATCH (pull|push2|push3). Forces TK_FUSE_GATEUP=0 so this isolates
the gate-only dispatch (the historical baseline for this timer); the fused
gate+up path (T4) is measured end-to-end via bench.
"""
from __future__ import annotations

import os

os.environ["TK_FUSE_GATEUP"] = "0"  # isolate gate-only dispatch (see docstring)

import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _dispatch_once(sch, tk):
    """Run exactly the scheme's layer0 dispatch, mirroring TKFusedEP.run()."""
    mode = sch.dispatch_mode
    sch._l0_seq += 1
    tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
    sch.pre_tokens.data_.copy_(sch.problem.hidden_states)
    sch._l0_seq += 1
    tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
    if mode == "push3":
        sch.push_local_cnt.zero_()
        sch._l0_seq += 1
        tk.moe_dispatch_push3(sch.pre_tokens, sch.gathered, sch.w_gate, sch.gate_out,
                              sch.padded, sch.push_idx, sch.push_src, sch.push_cnt_idx,
                              sch.push_local_cnt, sch.push_expected, sch.gate_expected,
                              sch.barrier_l0, sch.num_comm_sms, sch.num_padded_local,
                              sch.num_push, sch._l0_seq)
    elif mode == "push2":
        tk.moe_push_data(sch.pre_tokens, sch.gathered, sch.w_gate, sch.gate_out,
                         sch.padded, sch.push_idx, sch.push_src, sch.barrier_l0, sch.num_push)
        sch._l0_seq += 1
        tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
        tk.grouped_gemm(sch.gathered.data_, sch.w_gate, sch.gate_out, sch.padded,
                        sch.ctx.rank * sch.problem.config.num_local_experts)
    else:  # pull
        tk.moe_dispatch_gemm(sch.pre_tokens, sch.gathered.data_, sch.w_gate, sch.gate_out,
                             sch.padded, sch.disp_idx, sch.barrier_l0,
                             sch.num_comm_sms, sch.num_padded_local)


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

    cfg = MoEBenchConfig(hidden_size=7168, intermediate_size=2048, num_experts=ne, topk=8,
                         parallel_mode=ParallelMode.EP, world_size=world, precision=Precision.BF16,
                         num_tokens=[512], routing=RoutingConfig(distribution=Distribution.BALANCED),
                         distributed=True, use_cuda_graph=False, seed=0, verify=False, device="cuda")
    ctx = DistContext(rank=rank, world_size=world, local_rank=rank, device=device, group=None)
    problem = make_problem(cfg, 512, rank=rank, weights=make_weights(cfg, rank=rank))
    sch = TKFusedEP(); sch.setup(problem, ctx)
    tk = sch.tk

    for _ in range(warmup):
        _dispatch_once(sch, tk)
    torch.cuda.synchronize(); dist.barrier()

    samples = []
    for _ in range(iters):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        _dispatch_once(sch, tk)
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end))  # ms

    med = statistics.median(samples)
    t = torch.tensor(med, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        out_list.append(float(t.item()))
    del sch, problem; torch.cuda.empty_cache()
    dist.destroy_process_group()


def main():
    world = 4
    ne = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    mode = os.environ.get("TK_DISPATCH", "pull")
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mgr = mp.Manager(); out_list = mgr.list()
    mp.spawn(_worker, args=(world, init_method, ne, warmup, iters, out_list),
             nprocs=world, join=True)
    print(f"[dispatch-only NE={ne} mode={mode}] median = {out_list[0]*1e3:.1f} us")


if __name__ == "__main__":
    main()
