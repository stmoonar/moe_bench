"""Minimal broker-under-mp.spawn derisk: does TKParallelTensor's IPC broker
(fixed shm/socket keys, per-local_rank handshake) work inside torch mp.spawn
processes the way the moe_bench harness spawns them? Mirrors distributed._worker.
"""
import os
import sys

sys.path.insert(0, "/data/cinnzhang_vllm_td_test/xxy/moe_bench/kernels/tileoverlap/02_moe_dispatch_gemm")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(local_rank, world_size, init_method):
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=local_rank, world_size=world_size, device_id=device)
    dist.all_reduce(torch.tensor([local_rank], device=device))

    from _C import TKParallelTensor, pcie_device_barrier  # noqa

    t = TKParallelTensor((512, 7168), dtype=torch.bfloat16, local_rank=local_rank,
                         local_world_size=world_size, multicast=False)
    t.data_.copy_(torch.full((512, 7168), float(local_rank), device=device, dtype=torch.bfloat16))
    bar = TKParallelTensor((2, 32), dtype=torch.int, local_rank=local_rank,
                           local_world_size=world_size, multicast=False)
    bar.data_.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    pcie_device_barrier(bar, 1)  # cross-device slot barrier through the kernel
    torch.cuda.synchronize()
    dist.barrier()
    if local_rank == 0:
        print(f"[rank {local_rank}] BROKER OK under mp.spawn, world={world_size}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mp.spawn(_worker, args=(world_size, init_method), nprocs=world_size, join=True)
    print("spawn test done")
