# SPDX-License-Identifier: Apache-2.0
"""T6-v0 anchor (docs/13): reconcile the pre-reduction schedule against the
already-verified combine schedule (`comb_idx` / `combine_w`).

Pre-reduction regroups the combine sum by expert card:

    combine (golden):  out[t] = Σ_k  w(t,k) · E[erank(t,k)][slot(t,k)]
    pre-reduce      :  partial[d][(s,t)] = Σ_{k:erank==d} w · E[d][slot]
                       out[t]            = Σ_d partial[d][(s=rank,t)]

Both must contain the EXACT SAME set of (erank, slot, weight) terms per source
token — just grouped differently. This tool spawns `world` ranks, builds both
schedules, all-gathers every expert card's pre-reduce jobs, and on each source
rank checks term-for-term in both directions:

  (a) coverage: every golden term (t, erank, slot, w) appears exactly once as a
      (slot, w) entry of expert card `erank`'s job for (src=rank, t).
  (b) no extras: every (slot, w) entry of a job targeting (src=rank, t) maps
      back to exactly one golden comb_idx term (bijection).
  (c) contrib mask: final_contrib[t][d] == 1  iff  card d appears in t's experts.

Run from the PARENT dir (AGENTS.md):
  python -m moe_bench.tools.reconcile_prereduce
"""
from __future__ import annotations

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

    from moe_bench.tk_scheme import _build_schedules, _build_prereduce_schedule

    e_local = num_experts // world
    g = torch.Generator(device=device).manual_seed(seed)  # SAME seed on all ranks
    all_ids = torch.randint(0, num_experts, (world, num_tokens, topk),
                            generator=g, device=device)
    gw = torch.Generator(device=device).manual_seed(seed + 777)
    all_weights = torch.rand(world, num_tokens, topk, generator=gw, device=device)
    topk_ids = all_ids[rank].to(torch.int32)
    topk_weights = all_weights[rank].float()

    # golden combine schedule (source-side view for THIS rank)
    (disp_idx, comb_idx, padded, num_padded_local, push_idx, push_src, slack,
     push_cnt_idx, push_expected, gate_expected, dst_blk_offset,
     total_dst_blocks) = _build_schedules(
        topk_ids, num_tokens, world, num_experts, e_local, rank, device)

    # pre-reduce schedule (expert-card view for THIS rank)
    (prered_dst, prered_slots, prered_w, num_jobs, final_contrib) = \
        _build_prereduce_schedule(topk_ids, topk_weights, num_tokens, world,
                                  num_experts, e_local, rank, device)

    # all-gather every expert card's pre-reduce jobs (pad J to a uniform max)
    max_j = torch.tensor(num_jobs, device=device)
    dist.all_reduce(max_j, op=dist.ReduceOp.MAX)
    max_j = int(max_j.item())

    def _pad_rows(t, n, fill):
        out = torch.full((n,) + tuple(t.shape[1:]), fill, dtype=t.dtype, device=device)
        out[:t.shape[0]] = t
        return out

    dst_p = _pad_rows(prered_dst, max_j, -1)
    slots_p = _pad_rows(prered_slots, max_j, -1)
    w_p = _pad_rows(prered_w.to(device), max_j, 0.0)
    njobs = torch.tensor(num_jobs, device=device, dtype=torch.int32)

    dst_all = [torch.zeros_like(dst_p) for _ in range(world)]
    slots_all = [torch.zeros_like(slots_p) for _ in range(world)]
    w_all = [torch.zeros_like(w_p) for _ in range(world)]
    nj_all = [torch.zeros_like(njobs) for _ in range(world)]
    dist.all_gather(dst_all, dst_p)
    dist.all_gather(slots_all, slots_p)
    dist.all_gather(w_all, w_p)
    dist.all_gather(nj_all, njobs)

    # each rank checks its OWN source tokens against the gathered expert jobs.
    dst_all = [x.cpu() for x in dst_all]
    slots_all = [x.cpu() for x in slots_all]
    w_all = [x.cpu() for x in w_all]
    nj_all = [int(x.item()) for x in nj_all]
    comb_cpu = comb_idx.cpu()
    combw_cpu = topk_weights.reshape(-1).cpu()  # (num_tokens*topk,)
    contrib_cpu = final_contrib.cpu()

    # index expert card d's jobs by (src_dev, src_tok) -> multiset of (slot, w)
    job_map = [dict() for _ in range(world)]
    for d in range(world):
        for j in range(nj_all[d]):
            s = int(dst_all[d][j, 0].item()); t = int(dst_all[d][j, 1].item())
            terms = []
            for c in range(topk):
                slot = int(slots_all[d][j, c].item())
                if slot < 0:
                    continue
                terms.append((slot, float(w_all[d][j, c].item())))
            job_map[d][(s, t)] = terms

    fails = []
    # (a) coverage + (c) contrib: walk golden terms for this rank's tokens
    # build golden per-token multiset {(erank, slot, w)} and contributor set
    golden_used = [dict() for _ in range(world)]  # d -> {(rank,t): list of matched idx}
    for t in range(num_tokens):
        golden_cards = set()
        for k in range(topk):
            erank = int(comb_cpu[t * topk + k, 0].item())
            slot = int(comb_cpu[t * topk + k, 1].item())
            if erank < 0 or slot < 0:
                continue
            w = float(combw_cpu[t * topk + k].item())
            golden_cards.add(erank)
            terms = job_map[erank].get((rank, t))
            if terms is None:
                fails.append(f"(a) missing job on card {erank} for (src={rank},t={t}) slot={slot}")
                continue
            # find a matching (slot, w) term, consume it
            used = golden_used[erank].setdefault((rank, t), [False] * len(terms))
            hit = False
            for ci, (jslot, jw) in enumerate(terms):
                if not used[ci] and jslot == slot and abs(jw - w) < 1e-6:
                    used[ci] = True; hit = True; break
            if not hit:
                fails.append(f"(a) no term slot={slot} w={w:.4f} in card {erank} job (src={rank},t={t})")
        # (c) contrib mask
        for d in range(world):
            want = 1 if d in golden_cards else 0
            got = int(contrib_cpu[t, d].item())
            if want != got:
                fails.append(f"(c) contrib[t={t}][d={d}] got={got} want={want}")

    # (b) no extras: every job term targeting (rank, *) must have been consumed
    for d in range(world):
        for (s, t), terms in job_map[d].items():
            if s != rank:
                continue
            used = golden_used[d].get((rank, t), [False] * len(terms))
            for ci, u in enumerate(used):
                if not u:
                    slot, w = terms[ci]
                    fails.append(f"(b) EXTRA term card {d} (src={rank},t={t}) slot={slot} w={w:.4f}")

    nfail = torch.tensor(len(fails), device=device)
    dist.all_reduce(nfail, op=dist.ReduceOp.SUM)
    if rank == 0:
        out_list.append(int(nfail.item()))
    for f in fails[:5]:
        print(f"  [rank {rank}] {f}", flush=True)

    dist.destroy_process_group()


def main():
    world = 4
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mgr = mp.Manager()
    out_list = mgr.list()
    for ne in (64, 128, 256):
        out_list[:] = []
        mp.spawn(_worker, args=(world, init_method, 512, ne, 8, 0, out_list),
                 nprocs=world, join=True)
        total_fail = out_list[0] if out_list else -1
        status = "OK" if total_fail == 0 else "FAIL"
        print(f"[prered reconcile NE={ne}] total_failures={total_fail} -> {status}")
        init_method = f"tcp://localhost:{get_open_port()}"


if __name__ == "__main__":
    main()
