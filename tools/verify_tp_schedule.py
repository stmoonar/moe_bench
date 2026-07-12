# SPDX-License-Identifier: Apache-2.0
"""TP schedule anchor (mirror of verify_schedule_gpu for the TP tables): the
GPU-vectorized `_build_tp_schedules_gpu` must reproduce the host
`_build_tp_schedules` golden element-for-element, AND the tables must satisfy
the TP invariants that the kernels rely on:

  - tp_slots covers every assignment exactly once (bijection onto the real
    slot set): all slots unique, inside [0, P), and inside their expert's real
    region [base_e, base_e + count_e);
  - slack sums to P - N (total padding) with every entry in [0, ROW_BLOCK);
  - per row block: slack[b] + (real slots in b) == ROW_BLOCK — the dispatch
    gate's counter identity (spin until == ROW_BLOCK).

Sweeps NE∈{64,128,256} × {balanced, skewed}. Gate: total_failures == 0.

Run from the PARENT dir (AGENTS.md):
  python -m moe_bench.tools.verify_tp_schedule
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moe_bench.tk_scheme import ROW_BLOCK


def _make_topk(world, T, ne, topk, dist_kind, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    if dist_kind == "skewed":
        hot = max(topk, ne // 8)
        logits = torch.zeros(world, T, ne, device=device)
        logits[..., :hot] += 4.0
        logits += torch.rand(world, T, ne, generator=g, device=device)
    else:
        logits = torch.rand(world, T, ne, generator=g, device=device)
    ids = torch.topk(logits, topk, dim=-1).indices.to(torch.int32)
    gw = torch.Generator(device=device).manual_seed(seed + 777)
    w = torch.rand(world, T, topk, generator=gw, device=device).float()
    return ids, w


def _worker(rank, world, init_method, T, ne, topk, dist_kind, seed, out_list):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)

    from moe_bench.tk_tp_scheme import _build_tp_schedules, _build_tp_schedules_gpu

    all_ids, all_w = _make_topk(world, T, ne, topk, dist_kind, device, seed)
    topk_ids = all_ids[rank].contiguous()
    topk_weights = all_w[rank].contiguous()

    # ---- host golden ----
    (padded_g, slots_g, w_g, slack_g, pull_g, job_g, pusho_g, blk_g, P) = _build_tp_schedules(
        topk_ids, topk_weights, T, world, ne, rank, device)

    # ---- GPU builder (docs/30: single packed all_gather input) ----
    N = world * T * topk
    ar = torch.arange(N, device=device)
    packed_all = torch.stack(
        [all_ids, all_w.contiguous().view(torch.int32)], dim=-1).contiguous()
    out = {
        "src_dev_grid": ar // (T * topk),
        "src_tok_grid": (ar // topk) % T,
        "kpos_grid": ar % topk,
        "padded": torch.zeros(ne, dtype=torch.int32, device=device),
        "tp_slots": torch.full((world * T, topk), -1, dtype=torch.int32, device=device),
        "prered_w": torch.zeros(world * T, topk, dtype=torch.float32, device=device),
        "slack": torch.zeros(P // ROW_BLOCK, dtype=torch.int32, device=device),
        "pull_order": torch.zeros(world * T, dtype=torch.int32, device=device),
        "job_order": torch.zeros(world * T, dtype=torch.int32, device=device),
        "push_order": torch.zeros(world, T, dtype=torch.int32, device=device),
        "blk_expert": torch.zeros(P // ROW_BLOCK, dtype=torch.int32, device=device),
    }
    _build_tp_schedules_gpu(packed_all, world, ne, rank, out)

    fails = []
    for name, got, ref in [("padded", out["padded"], padded_g),
                           ("tp_slots", out["tp_slots"], slots_g),
                           ("prered_w", out["prered_w"], w_g),
                           ("slack", out["slack"], slack_g),
                           ("pull_order", out["pull_order"], pull_g),
                           ("job_order", out["job_order"], job_g),
                           ("push_order", out["push_order"], pusho_g),
                           ("blk_expert", out["blk_expert"], blk_g)]:
        if got.shape != ref.shape:
            fails.append(f"{name} shape {tuple(got.shape)} != host {tuple(ref.shape)}")
        elif not torch.equal(got, ref):
            nbad = int((got != ref).sum().item())
            fails.append(f"{name} MISMATCH ({nbad} elems)")

    # ---- TP invariants (on the host golden; GPU equals it if above passed) ----
    flat = slots_g.reshape(-1).long()
    if int(flat.min()) < 0 or int(flat.max()) >= P:
        fails.append("tp_slots out of [0, P)")
    if flat.unique().numel() != N:
        fails.append(f"tp_slots not a bijection ({flat.unique().numel()} unique != {N})")
    counts = torch.bincount(all_ids.view(-1).long(), minlength=ne)
    base = torch.zeros(ne, dtype=torch.long, device=device)
    base[1:] = torch.cumsum(padded_g.long(), dim=0)[:-1]
    # expert of each assignment vs the slot's expert region
    eid = all_ids.reshape(-1).long()
    slot_of = flat  # same flat n ordering as all_ids.reshape(-1)
    in_region = (slot_of >= base[eid]) & (slot_of < base[eid] + counts[eid])
    if not bool(in_region.all()):
        fails.append(f"{int((~in_region).sum())} slots outside their expert's real region")
    if int(slack_g.sum()) != P - N:
        fails.append(f"slack sum {int(slack_g.sum())} != padding {P - N}")
    if int(slack_g.max()) >= ROW_BLOCK or int(slack_g.min()) < 0:
        fails.append("slack entry out of [0, ROW_BLOCK)")
    # per-row-block counter identity
    real_per_blk = torch.bincount(flat // ROW_BLOCK, minlength=P // ROW_BLOCK)
    if not torch.equal(real_per_blk + slack_g.long(), torch.full_like(real_per_blk, ROW_BLOCK)):
        fails.append("slack[b] + real_in_block != ROW_BLOCK somewhere")
    # docs/30: blk_expert — non-decreasing, and expert e owns exactly
    # padded[e]/ROW_BLOCK consecutive row blocks (the dispenser GEMM's B index)
    blk = blk_g.long()
    if not bool((blk[1:] >= blk[:-1]).all()):
        fails.append("blk_expert not non-decreasing")
    blk_counts = torch.bincount(blk, minlength=ne)
    if not torch.equal(blk_counts, (padded_g.long() // ROW_BLOCK)):
        fails.append("blk_expert counts != padded/ROW_BLOCK")
    # docs/20 orders: both must be permutations of [0, world*T), and sorted by
    # their respective keys (pull: min slot ascending, job: max slot ascending)
    S = world * T
    for name, ordv, key in [("pull_order", pull_g, slots_g.min(dim=1).values),
                            ("job_order", job_g, slots_g.max(dim=1).values)]:
        o = ordv.long()
        if o.unique().numel() != S or int(o.min()) < 0 or int(o.max()) >= S:
            fails.append(f"{name} not a permutation of [0, {S})")
        else:
            k = key[o]
            if not bool((k[1:] > k[:-1]).all()):
                fails.append(f"{name} not sorted by its slot key")
    # push_order (docs/23): per-source permutation sorted by min slot; canonical
    # across ranks is adjudicated by the cross-rank all_reduce below (any rank
    # disagreeing fails its own sortedness/equality checks identically).
    mins2 = slots_g.min(dim=1).values.view(world, T)
    for srow in range(world):
        o = pusho_g[srow].long()
        if o.unique().numel() != T:
            fails.append(f"push_order[{srow}] not a permutation")
        elif not bool((mins2[srow][o][1:] > mins2[srow][o][:-1]).all()):
            fails.append(f"push_order[{srow}] not sorted by min slot")
    # canonical check: all ranks must hold the SAME push_order (byte compare via
    # all_reduce of a hash-like sum of rank0-broadcast diff)
    ref = pusho_g.clone()
    dist.broadcast(ref, src=0)
    if not torch.equal(ref, pusho_g):
        fails.append("push_order differs from rank 0 (not canonical)")

    nfail = torch.tensor(len(fails), device=device)
    dist.all_reduce(nfail, op=dist.ReduceOp.SUM)
    if rank == 0:
        out_list.append(int(nfail.item()))
    for f in fails[:6]:
        print(f"  [rank {rank}] {f}", flush=True)

    dist.destroy_process_group()


def main():
    world = 4
    from vllm.utils.network_utils import get_open_port
    mgr = mp.Manager()
    out_list = mgr.list()
    total = 0
    for dist_kind in ("balanced", "skewed"):
        for ne in (64, 128, 256):
            out_list[:] = []
            init_method = f"tcp://localhost:{get_open_port()}"
            mp.spawn(_worker, args=(world, init_method, 512, ne, 8, dist_kind, 0, out_list),
                     nprocs=world, join=True)
            tf = out_list[0] if out_list else -1
            total += tf if tf > 0 else (0 if tf == 0 else 1)
            status = "OK" if tf == 0 else "FAIL"
            print(f"[tp_schedule NE={ne} {dist_kind}] total_failures={tf} -> {status}")
    print(f"[tp_schedule] TOTAL {'OK' if total == 0 else 'FAIL'}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
