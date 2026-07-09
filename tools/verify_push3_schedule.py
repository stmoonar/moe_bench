# SPDX-License-Identifier: Apache-2.0
"""Step-1 anchor (docs/09 §5.1): validate the push3 expected tables.

Spawns `world` ranks, runs the real `_build_schedules`, all-gathers the per-rank
push3 tables, and cross-checks on rank 0:

  (a) push_expected[s][dst_blk_offset[d] + rb]  ==  gate_expected[d][rb][s]
      (source s's outgoing count into (dst d, row block rb) == dst d's expected
       arrivals from source s into local row block rb)
  (b) sum_s gate_expected[d][rb][s]  ==  ROW_BLOCK - slack[d][rb]
      (real tokens in a row block == the slack-complement; ties push3 to the
       already-verified pull-path slack seed)

Run from the PARENT dir (AGENTS.md):
  python -m moe_bench.tools.verify_push3_schedule
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROW_BLOCK = 128


def _worker(rank, world, init_method, num_tokens, num_experts, topk, seed, out_list):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)

    from moe_bench.tk_scheme import _build_schedules

    e_local = num_experts // world
    g = torch.Generator(device=device).manual_seed(seed)  # SAME seed on all ranks
    all_ids = torch.randint(0, num_experts, (world, num_tokens, topk),
                            generator=g, device=device)
    # distinct top-k per token (mirror data.py routing well enough for counts)
    topk_ids = all_ids[rank].to(torch.int32)

    (disp_idx, comb_idx, padded, num_padded_local, push_idx, push_src, slack,
     push_cnt_idx, push_expected, gate_expected, dst_blk_offset,
     total_dst_blocks) = _build_schedules(
        topk_ids, num_tokens, world, num_experts, e_local, rank, device)

    # all-gather the tables (pad to uniform sizes)
    max_tot = torch.tensor(total_dst_blocks, device=device)
    dist.all_reduce(max_tot, op=dist.ReduceOp.MAX)
    max_tot = int(max_tot.item())
    max_nblk = torch.tensor(gate_expected.shape[0], device=device)
    dist.all_reduce(max_nblk, op=dist.ReduceOp.MAX)
    max_nblk = int(max_nblk.item())

    pe = torch.zeros(max_tot, dtype=torch.int32, device=device)
    pe[:total_dst_blocks] = push_expected
    ge = torch.zeros(max_nblk, world, dtype=torch.int32, device=device)
    ge[:gate_expected.shape[0]] = gate_expected
    sl = torch.zeros(max_nblk, dtype=torch.int32, device=device)
    sl[:slack.shape[0]] = slack
    dbo = dst_blk_offset.to(torch.int32)

    pe_all = [torch.zeros_like(pe) for _ in range(world)]
    ge_all = [torch.zeros_like(ge) for _ in range(world)]
    sl_all = [torch.zeros_like(sl) for _ in range(world)]
    dbo_all = [torch.zeros_like(dbo) for _ in range(world)]
    dist.all_gather(pe_all, pe)
    dist.all_gather(ge_all, ge)
    dist.all_gather(sl_all, sl)
    dist.all_gather(dbo_all, dbo)

    if rank == 0:
        pe_all = [x.cpu() for x in pe_all]
        ge_all = [x.cpu() for x in ge_all]
        sl_all = [x.cpu() for x in sl_all]
        dbo = dbo_all[0].cpu()  # dst_blk_offset is a global geometry, same on all ranks
        tot = int(max_tot)      # total_dst_blocks is global-same across ranks
        # real per-device local row-block count (bound rb by this, NOT the padded
        # max_nblk — beyond it, dbo[d]+rb would spill into the next device's region)
        nblk_dev = [int(dbo[d + 1].item()) - int(dbo[d].item()) if d + 1 < world
                    else tot - int(dbo[d].item()) for d in range(world)]
        errs = []
        # (a) cross-check push_expected[s] vs gate_expected[d]
        mism = 0
        for d in range(world):
            for rb in range(nblk_dev[d]):
                for s in range(world):
                    c = int(dbo[d].item()) + rb
                    src_val = int(pe_all[s][c].item())
                    dst_val = int(ge_all[d][rb, s].item())
                    if src_val != dst_val:
                        mism += 1
                        if len(errs) < 10:
                            errs.append(f"(a) d={d} rb={rb} s={s}: push_exp={src_val} gate_exp={dst_val}")
        # (b) row sums vs slack complement
        mism_b = 0
        for d in range(world):
            for rb in range(nblk_dev[d]):
                real = int(ge_all[d][rb].sum().item())
                slack_real = ROW_BLOCK - int(sl_all[d][rb].item())
                if real != slack_real:
                    mism_b += 1
                    if len(errs) < 20:
                        errs.append(f"(b) d={d} rb={rb}: gate_sum={real} slack_real={slack_real}")
        out_list.append((mism, mism_b, errs))

    dist.destroy_process_group()


def main():
    world = 4
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mgr = mp.Manager()
    out_list = mgr.list()
    # test a few expert counts incl the ones that broke push (64) + prod (256)
    for ne in (64, 256):
        out_list[:] = []
        mp.spawn(_worker, args=(world, init_method, 512, ne, 8, 0, out_list),
                 nprocs=world, join=True)
        mism, mism_b, errs = out_list[0]
        status = "OK" if (mism == 0 and mism_b == 0) else "FAIL"
        print(f"[NE={ne}] cross-check(a) mismatches={mism}  rowsum(b) mismatches={mism_b}  -> {status}")
        for e in errs:
            print("   ", e)
        init_method = f"tcp://localhost:{get_open_port()}"


if __name__ == "__main__":
    main()
