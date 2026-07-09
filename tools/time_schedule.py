# SPDX-License-Identifier: Apache-2.0
"""T3 timing (docs/14): isolate the GPU schedule-rebuild cost in the real 4-rank
setting — all_gather vs the vectorized builder vs both — max-over-ranks median us.

  python -m moe_bench.tools.time_schedule  [num_experts]  [iters]
"""
from __future__ import annotations

import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world, init_method, ne, iters, out_list):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)
    dist.all_reduce(torch.tensor([rank], device=device))

    from moe_bench.tk_scheme import (_build_schedules, _build_prereduce_schedule,
                                     _build_schedules_gpu)

    T, topk = 512, 8
    e_local = ne // world
    N = world * T * topk
    g = torch.Generator(device=device).manual_seed(0)
    all_ids = torch.topk(torch.rand(world, T, ne, generator=g, device=device),
                         topk, dim=-1).indices.to(torch.int32)
    all_w = torch.rand(world, T, topk, device=device).float()
    tid = all_ids[rank].contiguous()
    tw = all_w[rank].float().contiguous()
    (_, _, _, npl_g, *_) = _build_schedules(tid, T, world, ne, e_local, rank, device)
    (_, _, _, J_g, _) = _build_prereduce_schedule(tid, tw, T, world, ne, e_local, rank, device)
    Jm = max(J_g, 1)

    ar = torch.arange(N, device=device)
    at = torch.empty(world, T, topk, device=device, dtype=tid.dtype)
    aw = torch.empty(world, T, topk, device=device)
    out = {
        "src_dev_grid": ar // (T * topk), "src_tok_grid": (ar // topk) % T,
        "kpos_grid": ar % topk,
        "disp_idx": torch.full((npl_g, 2), -1, dtype=torch.int32, device=device),
        "padded": torch.zeros(ne, dtype=torch.int32, device=device),
        "prered_dst": torch.full((Jm, 2), -1, dtype=torch.int32, device=device),
        "prered_slots": torch.full((Jm, topk), -1, dtype=torch.int32, device=device),
        "prered_w": torch.zeros((Jm, topk), dtype=torch.float32, device=device),
        "final_contrib": torch.zeros(T, world, dtype=torch.int32, device=device),
    }

    def ag():
        dist.all_gather_into_tensor(at, tid)
        dist.all_gather_into_tensor(aw, tw)

    def build():
        _build_schedules_gpu(all_ids, all_w, world, ne, e_local, rank, out)

    def both():
        ag(); _build_schedules_gpu(at, aw, world, ne, e_local, rank, out)

    def tm(fn, n):
        for _ in range(10):
            fn()
        torch.cuda.synchronize(); dist.barrier()
        t = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t) / n * 1e6

    # CUDA-graph capture of the pure-compute builder (no NCCL inside) — kills the
    # ~30-launch eager overhead. Routing is fixed per problem so sizes are stable.
    graph_us = -1.0
    try:
        build(); torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            _build_schedules_gpu(all_ids, all_w, world, ne, e_local, rank, out)
        graph_us = tm(lambda: gph.replay(), iters)
    except Exception as e:
        if rank == 0:
            print(f"  [graph capture failed] {repr(e)[:160]}", flush=True)

    res = torch.tensor([tm(ag, iters), tm(build, iters), tm(both, iters), graph_us],
                       device=device)
    dist.all_reduce(res, op=dist.ReduceOp.MAX)
    if rank == 0:
        out_list.append(res.tolist())
    dist.destroy_process_group()


def main():
    world = 4
    ne = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mgr = mp.Manager()
    out_list = mgr.list()
    mp.spawn(_worker, args=(world, init_method, ne, iters, out_list), nprocs=world, join=True)
    a, b, c, gph = out_list[0]
    print(f"[time_schedule NE={ne}] all_gather×2={a:.0f}us  builder(eager)={b:.0f}us  "
          f"both={c:.0f}us  builder(graph)={gph:.0f}us")


if __name__ == "__main__":
    main()
