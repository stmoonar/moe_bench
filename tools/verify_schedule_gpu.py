# SPDX-License-Identifier: Apache-2.0
"""T3 anchor (docs/14): the GPU-vectorized schedule builder must reproduce the
host `_build_schedules` / `_build_prereduce_schedule` goldens element-for-element.

Spawns `world` ranks, builds BOTH the host golden and _build_schedules_gpu on the
same gathered routing, and compares the 6 DEFAULT-path tables with torch.equal:
disp_idx, padded, prered_dst, prered_slots, prered_w, final_contrib.

Sweeps NE∈{64,128,256} × {balanced, skewed} — skewed exercises empty/uneven
experts, the highest-risk case (docs/07 P4). Gate: total_failures == 0.

Run from the PARENT dir (AGENTS.md):
  python -m moe_bench.tools.verify_schedule_gpu
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROW_BLOCK = 128


def _make_topk(world, T, ne, topk, dist_kind, device, seed):
    """SAME routing on all ranks (each rank slices its own row). Distinct experts
    per token. balanced = uniform random; skewed = most mass on a few experts."""
    g = torch.Generator(device=device).manual_seed(seed)
    if dist_kind == "skewed":
        # concentrate on the first ~1/8 of experts (many experts get 0 tokens)
        hot = max(topk, ne // 8)
        logits = torch.zeros(world, T, ne, device=device)
        logits[..., :hot] += 4.0
        logits += torch.rand(world, T, ne, generator=g, device=device)
        ids = torch.topk(logits, topk, dim=-1).indices.to(torch.int32)
    else:
        # distinct experts per token via top-k over random logits
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

    from moe_bench.tk_scheme import (_build_schedules, _build_prereduce_schedule,
                                     _build_schedules_gpu)

    e_local = ne // world
    all_ids, all_w = _make_topk(world, T, ne, topk, dist_kind, device, seed)
    topk_ids = all_ids[rank].contiguous()
    topk_weights = all_w[rank].contiguous()

    # ---- host goldens ----
    (disp_g, comb_g, padded_g, npl_g, *_) = _build_schedules(
        topk_ids, T, world, ne, e_local, rank, device)
    (pdst_g, pslots_g, pw_g, J_g, contrib_g) = _build_prereduce_schedule(
        topk_ids, topk_weights, T, world, ne, e_local, rank, device)
    from moe_bench.tk_scheme import _derive_staging
    s2s_g, need_g, s_max = _derive_staging(disp_g, T, world)

    # ---- GPU builder into pre-allocated tables ----
    N = world * T * topk
    dev_grid = (torch.arange(N, device=device) // (T * topk))
    tok_grid = (torch.arange(N, device=device) // topk) % T
    kpos_grid = torch.arange(N, device=device) % topk
    # prered_dst is constant (dense job space j=src_dev*T+src_tok); set once.
    pdst = torch.full((max(J_g, 1), 2), -1, dtype=torch.int32, device=device)
    pdst[:, 0] = (torch.arange(J_g, device=device) // T).to(torch.int32)
    pdst[:, 1] = (torch.arange(J_g, device=device) % T).to(torch.int32)
    out = {
        "src_dev_grid": dev_grid, "src_tok_grid": tok_grid, "kpos_grid": kpos_grid,
        "disp_idx": torch.full((npl_g, 2), -1, dtype=torch.int32, device=device),
        "padded": torch.zeros(ne, dtype=torch.int32, device=device),
        "prered_dst": pdst,
        "prered_slots": torch.full((max(J_g, 1), topk), -1, dtype=torch.int32, device=device),
        "prered_w": torch.zeros((max(J_g, 1), topk), dtype=torch.float32, device=device),
        "final_contrib": torch.zeros(T, world, dtype=torch.int32, device=device),
        "slot_to_staging": torch.full((npl_g,), -1, dtype=torch.int32, device=device),
        "staging_needed": torch.zeros(s_max, dtype=torch.int32, device=device),
    }
    npl_gpu, J_gpu = _build_schedules_gpu(all_ids, all_w, world, ne, e_local, rank, out)

    fails = []
    if npl_gpu != npl_g:
        fails.append(f"num_padded_local {npl_gpu} != host {npl_g}")
    if J_gpu != J_g:
        fails.append(f"num_jobs {J_gpu} != host {J_g}")
    checks = [
        ("disp_idx", out["disp_idx"], disp_g),
        ("padded", out["padded"], padded_g.to(torch.int32)),
        ("prered_dst", out["prered_dst"], pdst_g),
        ("prered_slots", out["prered_slots"], pslots_g),
        ("final_contrib", out["final_contrib"], contrib_g),
        ("slot_to_staging", out["slot_to_staging"], s2s_g),
        ("staging_needed", out["staging_needed"], need_g),
    ]
    for name, got, ref in checks:
        if got.shape != ref.shape:
            fails.append(f"{name} shape {tuple(got.shape)} != host {tuple(ref.shape)}")
        elif not torch.equal(got, ref):
            nbad = int((got != ref).sum().item())
            fails.append(f"{name} MISMATCH ({nbad} elems)")
    # prered_w: same f32 values reordered, exact equality expected
    if out["prered_w"].shape == pw_g.shape and not torch.equal(out["prered_w"], pw_g):
        nbad = int((out["prered_w"] != pw_g).sum().item())
        fails.append(f"prered_w MISMATCH ({nbad} elems)")

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
    init_method = f"tcp://localhost:{get_open_port()}"
    mgr = mp.Manager()
    out_list = mgr.list()
    for dist_kind in ("balanced", "skewed"):
        for ne in (64, 128, 256):
            out_list[:] = []
            mp.spawn(_worker, args=(world, init_method, 512, ne, 8, dist_kind, 0, out_list),
                     nprocs=world, join=True)
            tf = out_list[0] if out_list else -1
            status = "OK" if tf == 0 else "FAIL"
            print(f"[schedule_gpu NE={ne} {dist_kind}] total_failures={tf} -> {status}")
            init_method = f"tcp://localhost:{get_open_port()}"


if __name__ == "__main__":
    main()
