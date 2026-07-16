# SPDX-License-Identifier: Apache-2.0
"""TK fused MoE scheme (bf16, EP) for the distributed benchmark.

Full layer on one rank using the ThunderKittens fused kernels verified in
tileoverlap/{01,02,03}:

  1. dispatch ⊕ gate GEMM   (layer0 fused, moe_dispatch_gemm)   -> gate_out
  2. up GEMM                (grouped_gemm on the gathered tokens) -> up_out
  3. act = silu(gate) * up  (torch)
  4. W2 GEMM ⊕ combine      (layer1 fused, moe_gemm_combine_fused) -> output

Route B' communication (experience/12): pull + local FP32 reduce + slot signals,
no remote atomics, no multimem. See docs/06_Phase4_scheme接入设计.md.

bf16 only for v1 (user choice); fp8 is a follow-up. EP only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import ParallelMode
from .context import DistContext
from .data import MoEProblem
from .schemes import DistributedScheme

import os as _os_rb
# T5 (docs/16): ROW_BLOCK is the tokens-per-tile AND expert padding unit, switchable
# via TK_ROW_BLOCK ∈ {128 (default), 64}. At 64 the per-expert padding halves
# (NE=256: 64 real tokens no longer pad to 128) so both GEMM layers compute ~half
# the rows. Must match the compiled kernel's TK_ROW_BLOCK (build_and_load below).
ROW_BLOCK = int(_os_rb.environ.get("TK_ROW_BLOCK", "128"))


def _build_schedules(topk_ids, num_tokens, world_size, num_experts, num_experts_per_dev,
                     rank, device):
    """From this rank's routing (and all ranks', gathered) produce:
      - pull_dispatch_indices (num_padded_local, 2): (src_dev, src_token) for the
        tokens THIS rank's experts need, expert-sorted, 128-padded, ring-ordered.
      - combine_indices (num_tokens*topk, 2): (e_rank, remote_slot) for THIS rank's
        source tokens' top-k experts.
      - push_indices (num_tokens*topk, 2): (dst_dev, dst_slot) for THIS rank's
        OUTGOING tokens (== combine_indices, the source-side view of the same
        assignment); push_src (num_tokens*topk, 1): local source token index.
      - slack_seed (num_padded_local//ROW_BLOCK,): ROW_BLOCK - (real tokens in
        that row block on THIS card), so push counters start pre-seeded and real
        pushes bring each row block exactly to ROW_BLOCK.
      - push_cnt_idx (num_tokens*topk, 1): push3 (docs/09) flat LOCAL counter idx
        for each outgoing assignment = dst_blk_offset[dst_dev] + dst_slot//ROW_BLOCK.
      - push_expected (total_dst_blocks,): push3 satisfaction count per local
        counter (real tokens this card sends into that (dst,row block)).
      - gate_expected (nblk_local, world): push3, this card AS DST — tokens
        expected from each source card per local row block (gate uses >0 only).
      - dst_blk_offset (world,), total_dst_blocks: push3 counter geometry.
      - padded_tokens_per_expert (num_experts,), num_padded_local.
    All host-side (torch), done in setup (not timed).
    """
    # gather all ranks' topk_ids so every rank can replay every expert card's schedule
    all_topk = torch.empty(world_size, num_tokens, topk_ids.shape[1],
                           device=device, dtype=topk_ids.dtype)
    torch.distributed.all_gather_into_tensor(all_topk, topk_ids.contiguous())
    all_topk_cpu = all_topk.cpu()

    tokens_per_expert = torch.bincount(all_topk.view(-1), minlength=num_experts).to(torch.int32)
    padded = ((tokens_per_expert + ROW_BLOCK - 1) // ROW_BLOCK * ROW_BLOCK)
    padded_cpu = padded.cpu().long()
    real_cpu = tokens_per_expert.cpu().long()

    my_estart = num_experts_per_dev * rank
    my_eend = num_experts_per_dev * (rank + 1)
    num_padded_local = int(padded[my_estart:my_eend].sum())

    # push3 (docs/09) row-block geometry: per-device row-block counts + offsets.
    # A single flat counter array [0, total_dst_blocks) spans EVERY device's
    # gathered row blocks; dst_blk_offset[d] is where device d's blocks start.
    padded_per_dev = padded_cpu.reshape(world_size, num_experts_per_dev).sum(dim=1)
    nblk_per_dev = (padded_per_dev // ROW_BLOCK).long()               # (world,)
    dst_blk_offset = torch.cat([
        torch.zeros(1, dtype=torch.int64, device="cpu"),
        torch.cumsum(nblk_per_dev, dim=0)[:-1]
    ]).tolist()                                                       # (world,)
    total_dst_blocks = int(nblk_per_dev.sum())
    nblk_local = num_padded_local // ROW_BLOCK

    top_k = topk_ids.shape[1]
    disp_idx = torch.full((num_padded_local, 2), -1, dtype=torch.int32, device="cpu")
    comb_idx = torch.full((num_tokens * top_k, 2), -1, dtype=torch.int32, device="cpu")
    push_idx = torch.full((num_tokens * top_k, 2), -1, dtype=torch.int32, device="cpu")
    push_src = torch.full((num_tokens * top_k, 1), -1, dtype=torch.int32, device="cpu")
    # push3: local flat counter index per outgoing assignment (built at the SAME
    # src_tok*top_k+kpos slot as push_idx/push_src, filtered by the same mask).
    push_cnt_idx = torch.full((num_tokens * top_k, 1), -1, dtype=torch.int32, device="cpu")
    # push3: this card AS DST — tokens expected from each source card per local row block.
    gate_expected = torch.zeros(nblk_local, world_size, dtype=torch.int32, device="cpu")

    # replay each expert card's write cursor (ring by source device), exactly the
    # order the dispatch kernel pulls in — so slot ids match on both sides.
    for e_rank in range(world_size):
        estart = num_experts_per_dev * e_rank
        eend = num_experts_per_dev * (e_rank + 1)
        write_pos = torch.cat([
            torch.zeros(1, dtype=torch.int64, device="cpu"),
            torch.cumsum(padded_cpu[estart:eend - 1], dim=0)
        ]).tolist()
        for i in range(world_size):
            src_dev = (i + e_rank) % world_size
            for src_tok in range(num_tokens):
                for kpos, eid in enumerate(all_topk_cpu[src_dev, src_tok].tolist()):
                    if estart <= eid < eend:
                        e = eid - estart
                        slot = write_pos[e]
                        write_pos[e] += 1
                        if e_rank == rank:  # this rank is the expert card -> dispatch entry
                            disp_idx[slot, 0] = src_dev
                            disp_idx[slot, 1] = src_tok
                            # push3: as DST, count expected arrivals per (local row block, source)
                            gate_expected[slot // ROW_BLOCK, src_dev] += 1
                        if src_dev == rank:  # this rank is the source card -> combine + push
                            comb_idx[src_tok * top_k + kpos, 0] = e_rank
                            comb_idx[src_tok * top_k + kpos, 1] = slot
                            push_idx[src_tok * top_k + kpos, 0] = e_rank
                            push_idx[src_tok * top_k + kpos, 1] = slot
                            push_src[src_tok * top_k + kpos, 0] = src_tok
                            # push3: flat local counter for this (dst_dev=e_rank, dst row block)
                            push_cnt_idx[src_tok * top_k + kpos, 0] = \
                                dst_blk_offset[e_rank] + slot // ROW_BLOCK

    # slack seed: for THIS card's experts, each row block's padding = ROW_BLOCK -
    # (real tokens in that block). Blocks fully inside real region → 0; the tail
    # block of each expert carries (padded - real) slack.
    nblk = num_padded_local // ROW_BLOCK
    slack = torch.zeros(nblk, dtype=torch.int32, device="cpu")
    blk = 0
    for e in range(num_experts_per_dev):
        ge = my_estart + e
        real_e = int(real_cpu[ge]); pad_e = int(padded_cpu[ge])
        nblk_e = pad_e // ROW_BLOCK
        for b in range(nblk_e):
            block_start = b * ROW_BLOCK
            real_in_block = max(0, min(ROW_BLOCK, real_e - block_start))
            slack[blk] = ROW_BLOCK - real_in_block
            blk += 1

    # push3 expected values (source side): satisfaction count per local flat
    # counter = number of THIS card's real assignments pointing at it. Built by
    # bincount over the valid push_cnt_idx entries (padding rows are -1 → dropped).
    valid_cnt = push_cnt_idx[:, 0][push_cnt_idx[:, 0] >= 0].to(torch.int64)
    push_expected = torch.bincount(valid_cnt, minlength=total_dst_blocks).to(torch.int32)

    return (disp_idx.to(device), comb_idx.to(device), padded.to(device),
            num_padded_local, push_idx.to(device), push_src.to(device), slack.to(device),
            push_cnt_idx.to(device), push_expected.to(device), gate_expected.to(device),
            torch.tensor(dst_blk_offset, dtype=torch.int32, device=device), total_dst_blocks)


def _derive_staging(disp_idx, num_tokens, world_size, slot_to_staging=None,
                    staging_needed=None):
    """T7 (docs/15): dedup tables, pure functions of disp_idx (so they inherit
    its verified correctness AND the GPU builder's graph-capturability).

    DENSE staging layout: staging_row(src_dev, src_tok) = src_dev*num_tokens +
    src_tok, capacity S_max = world*num_tokens (routing-invariant). Two tables:
      slot_to_staging (num_padded_local,) int32 : gathered slot -> its staging
        row (-1 for padding slots, preserving the count-but-no-data behavior);
      staging_needed  (S_max,) int32 : 1 iff some local slot references that dense
        row (the unique cross-card pull list — the pull kernel skips 0 rows).
    Writes in place when out tensors are given (capture-safe: scatter_, no
    bincount); else allocates. Returns (slot_to_staging, staging_needed, S_max).
    """
    device = disp_idx.device
    P = disp_idx.shape[0]
    S_max = world_size * num_tokens
    if slot_to_staging is None:
        slot_to_staging = torch.full((P,), -1, dtype=torch.int32, device=device)
    if staging_needed is None:
        staging_needed = torch.zeros(S_max, dtype=torch.int32, device=device)
    sd = disp_idx[:, 0].long()
    st = disp_idx[:, 1].long()
    mask = sd >= 0
    row = torch.where(mask, sd * num_tokens + st, torch.full_like(sd, -1))
    slot_to_staging.copy_(row.to(torch.int32))
    staging_needed.zero_()
    # scatter a 1 into every referenced dense row (padding rows map to -1 -> use a
    # trash slot S_max then drop, keeping this fixed-shape / capture-safe).
    row_dst = torch.where(mask, row, torch.full_like(row, S_max))
    tmp = torch.zeros(S_max + 1, dtype=torch.int32, device=device)
    tmp.scatter_(0, row_dst, torch.ones_like(row_dst, dtype=torch.int32))
    staging_needed.copy_(tmp[:S_max])
    return slot_to_staging, staging_needed, S_max


def _build_prereduce_schedule(topk_ids, topk_weights, num_tokens, world_size,
                              num_experts, num_experts_per_dev, rank, device):
    """T6-v0 pre-reduction schedule (docs/13). Regroups the combine sum

        out[t] = Σ_k  w(t,k) · expert_out[e_rank(t,k)][slot(t,k)]

    by EXPERT CARD instead of by (token, expert):

        partial[d][(s,t)] = Σ_{k : e_rank(t,k)==d}  w(t,k) · expert_out[d][slot]
        out[t]            = Σ_d partial[d][(rank,t)]

    Same terms, regrouped — so it must reconcile term-for-term against the
    already-verified `comb_idx` (see tools/reconcile_prereduce.py). The slot
    cursor here is the SAME ring-by-source-device replay as `_build_schedules`,
    so the local slots here == the dispatch slots there. The pre-reduced row for
    (src_dev, src_tok) lives at plane [src_dev][src_tok] of a (world, num_tokens,
    H) partial buffer — a layout shared by v0 (source pulls that row) and v1
    (expert card TMA-pushes it to the source card's staging[d][t]).

    Standalone / not wired into run() or setup(): this is the T6 foundation to
    be adjudicated clean before any kernel change (docs/11 §3).

    Returns, EXPERT-CARD view (this rank as producer d == rank). DENSE job space
    (docs/14): num_jobs = world*num_tokens, job j = src_dev*num_tokens + src_tok
    covers ALL (src_dev, src_tok) pairs (aligned to the partials buffer rows);
    column = kpos. Jobs with no local hit stay all -1 (harmless unread zero row).
    Constant num_jobs makes the GPU rebuild (T3) fixed-shape / graph-capturable.
      prered_dst   (J, 2)     int32 : (src_dev, src_tok) per job (= (j//T, j%T))
      prered_slots (J, TOP_K) int32 : local slot per kpos (-1 where no local hit)
      prered_w     (J, TOP_K) f32   : weight per kpos (0 where no local hit)
      num_jobs J = world*num_tokens
    and SOURCE-CARD view (this rank as consumer s == rank):
      final_contrib (num_tokens, world) int32 : 1 iff card d holds ≥1 of t's experts
    """
    top_k = topk_ids.shape[1]

    all_topk = torch.empty(world_size, num_tokens, top_k, device=device, dtype=topk_ids.dtype)
    torch.distributed.all_gather_into_tensor(all_topk, topk_ids.contiguous())
    all_w = torch.empty(world_size, num_tokens, top_k, device=device, dtype=torch.float32)
    torch.distributed.all_gather_into_tensor(all_w, topk_weights.float().contiguous())
    all_topk_cpu = all_topk.cpu()
    all_w_cpu = all_w.cpu()

    tokens_per_expert = torch.bincount(all_topk.view(-1), minlength=num_experts).to(torch.int32)
    padded_cpu = ((tokens_per_expert + ROW_BLOCK - 1) // ROW_BLOCK * ROW_BLOCK).cpu().long()

    # expert-card side: (src_dev, src_tok) -> per-kpos (local_slot, weight) this
    # card must FP32-reduce into one partial row. DENSE job space (docs/14):
    # job j = src_dev*num_tokens + src_tok covers ALL (src_dev, src_tok) pairs,
    # aligned to the partials buffer row layout; column = kpos (order-independent
    # for a sum). Jobs with no local hit stay all -1 and write a harmless unread
    # zero row. num_jobs = world*num_tokens (constant, routing-independent) —
    # which makes the GPU rebuild (T3) fixed-shape / CUDA-graph capturable.
    num_jobs = world_size * num_tokens
    prered_dst = torch.full((num_jobs, 2), -1, dtype=torch.int32, device="cpu")
    prered_slots = torch.full((num_jobs, top_k), -1, dtype=torch.int32, device="cpu")
    prered_w = torch.zeros((num_jobs, top_k), dtype=torch.float32, device="cpu")
    src_dev_col = torch.arange(num_jobs) // num_tokens
    src_tok_col = torch.arange(num_jobs) % num_tokens
    prered_dst[:, 0] = src_dev_col.to(torch.int32)
    prered_dst[:, 1] = src_tok_col.to(torch.int32)
    final_contrib = torch.zeros(num_tokens, world_size, dtype=torch.int32, device="cpu")

    for e_rank in range(world_size):
        estart = num_experts_per_dev * e_rank
        eend = num_experts_per_dev * (e_rank + 1)
        write_pos = torch.cat([
            torch.zeros(1, dtype=torch.int64, device="cpu"),
            torch.cumsum(padded_cpu[estart:eend - 1], dim=0)
        ]).tolist()
        for i in range(world_size):
            src_dev = (i + e_rank) % world_size
            for src_tok in range(num_tokens):
                for kpos, eid in enumerate(all_topk_cpu[src_dev, src_tok].tolist()):
                    if estart <= eid < eend:
                        e = eid - estart
                        slot = write_pos[e]
                        write_pos[e] += 1
                        if e_rank == rank:  # I am the expert card: record hit at (job, kpos)
                            j = src_dev * num_tokens + src_tok
                            prered_slots[j, kpos] = slot
                            prered_w[j, kpos] = float(all_w_cpu[src_dev, src_tok, kpos])
                        if src_dev == rank:  # I am the source card: mark contributor
                            final_contrib[src_tok, e_rank] = 1

    return (prered_dst.to(device), prered_slots.to(device), prered_w.to(device),
            num_jobs, final_contrib.to(device))


def _derive_combine_push(prered_slots, final_contrib, num_tokens, world_size,
                         push_expected_l1=None, recv_from=None):
    """T6-v1 (docs/18): expert-side combine-push tables, pure functions of the v0
    prered tables (so they inherit correctness AND graph-capturability).

    Dense job space: job j = src_dev*num_tokens + src_tok, grouped by src_dev in
    contiguous blocks of num_tokens. This card (expert card d == rank) pushes a
    partial row for every NON-EMPTY job to source card s = j // num_tokens.

      push_expected_l1 (world,) int32 : rows this card pushes to source card s =
        count of non-empty jobs with src_dev==s (the election target count).
      recv_from (world,) int32 : as SOURCE card, 1 iff expert card d sends me >=1
        row = (final_contrib[:,d].sum()>0) — the symmetric complement, so I only
        wait on cards that will actually signal (no deadlock).
    Capture-safe (no bincount / host sync). Writes in place if out given."""
    device = prered_slots.device
    has_hit = (prered_slots >= 0).any(dim=1).to(torch.int32)          # (J,)
    exp = has_hit.view(world_size, num_tokens).sum(dim=1).to(torch.int32)  # (world,)
    rf = (final_contrib.sum(dim=0) > 0).to(torch.int32)               # (world,)
    if push_expected_l1 is None:
        push_expected_l1 = torch.empty(world_size, dtype=torch.int32, device=device)
    if recv_from is None:
        recv_from = torch.empty(world_size, dtype=torch.int32, device=device)
    push_expected_l1.copy_(exp)
    recv_from.copy_(rf)
    return push_expected_l1, recv_from


def _build_schedules_gpu(all_topk, all_w, world_size, num_experts,
                         num_experts_per_dev, rank, out):
    """T3 (docs/14): GPU-vectorized rebuild of the DEFAULT-path schedule tables,
    element-for-element identical to the host `_build_schedules` /
    `_build_prereduce_schedule` goldens. Called each run() and counted in timing
    (fairness — serial pays its routing metadata per run; docs/07 P1, docs/11 §T3).

    Reproduces the host's ring-order slot cursor with ONE global sort. Host assigns
    a token-expert assignment's slot by its rank in the total order

        (eid, ring_offset, src_tok, kpos),  ring_offset = (src_dev - e_rank) mod world

    Because (src_dev, src_tok, kpos) is a bijection over flattened all_topk, every
    key is UNIQUE — so argsort needs no stability (CUDA argsort is not stable).

    `all_topk` (world, T, TOPK) int, `all_w` (world, T, TOPK) f32 are already
    gathered. `out` is a dict of PRE-ALLOCATED output tensors written in place:
      disp_idx (P,2) int32, padded (num_experts,) int32,
      prered_dst (J,2) int32, prered_slots (J,TOP_K) int32, prered_w (J,TOP_K) f32,
      final_contrib (T, world) int32
    plus constant index grids src_dev_grid/src_tok_grid/kpos_grid (shape N).
    No GPU->host sync; grid/tensor sizes are fixed (routing invariant per problem).
    """
    device = all_topk.device
    T = all_topk.shape[1]
    top_k = all_topk.shape[2]
    e_local = num_experts_per_dev
    N = world_size * T * top_k

    eid = all_topk.reshape(N).long()
    e_rank_of = eid // e_local
    ring = (out["src_dev_grid"] - e_rank_of) % world_size
    key = ((eid * world_size + ring) * T + out["src_tok_grid"]) * top_k + out["kpos_grid"]
    order = torch.argsort(key)
    eid_s = eid[order]

    # per-expert counts (shared cursor across ALL source devices) + padding.
    # scatter_add (not bincount) — bincount does a device sync that breaks CUDA
    # graph capture; scatter_add is capture-safe and equivalent.
    counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    counts.scatter_add_(0, eid, torch.ones_like(eid))
    padded = ((counts + ROW_BLOCK - 1) // ROW_BLOCK * ROW_BLOCK).to(torch.int32)
    out["padded"].copy_(padded)
    grp_start = torch.zeros(num_experts, dtype=torch.long, device=device)
    grp_start[1:] = torch.cumsum(counts, dim=0)[:-1]
    pos_in_expert = torch.arange(N, device=device) - grp_start[eid_s]

    # absolute padded slot for EVERY (sorted) assignment; local ones land in
    # [0, num_padded_local). Non-local ones use a masked el_idx=0 to stay
    # in-bounds and are dropped by the trash-row redirect below — no boolean
    # compaction, so the whole builder is fixed-shape / CUDA-graph capturable.
    is_loc = (eid_s // e_local) == rank
    el_idx = torch.where(is_loc, eid_s - rank * e_local, torch.zeros_like(eid_s))
    padded_l = padded[rank * e_local:(rank + 1) * e_local].long()
    padded_base = torch.zeros(e_local, dtype=torch.long, device=device)
    padded_base[1:] = torch.cumsum(padded_l, dim=0)[:-1]
    slot_abs_sorted = padded_base[el_idx] + pos_in_expert

    # ---- disp_idx: slot -> (src_dev, src_tok) (local assignments only) ----
    # non-local assignments scatter into a trash row P (extra slot), then dropped.
    P = out["disp_idx"].shape[0]
    slot_dst = torch.where(is_loc, slot_abs_sorted, torch.full_like(slot_abs_sorted, P))
    di = torch.full((P + 1, 2), -1, dtype=torch.int32, device=device)
    di[:, 0].scatter_(0, slot_dst, out["src_dev_grid"][order].to(torch.int32))
    di[:, 1].scatter_(0, slot_dst, out["src_tok_grid"][order].to(torch.int32))
    out["disp_idx"].copy_(di[:P])

    # ---- T7 dedup tables (derived from disp_idx) ----
    if "slot_to_staging" in out:
        _derive_staging(out["disp_idx"], T, world_size,
                        out["slot_to_staging"], out["staging_needed"])

    # ---- prered (DENSE job space, column = kpos): the flat cell of prered_slots
    # for assignment n is (src_dev*T + src_tok)*top_k + kpos == n (the original
    # flat index). So scatter each sorted local assignment's slot back to its
    # original position `order[pos]`. Column order within a job is kpos, which is
    # order-independent for the FP32 sum. prered_dst is constant (set by caller).
    slot_by_n = torch.full((N,), -1, dtype=torch.int32, device=device)
    w_by_n = torch.zeros(N, dtype=torch.float32, device=device)
    slot_by_n.scatter_(0, order, torch.where(is_loc, slot_abs_sorted.to(torch.int32),
                                             torch.full_like(eid_s, -1, dtype=torch.int32)))
    w_by_n.scatter_(0, order, torch.where(is_loc, all_w.reshape(N)[order],
                                          torch.zeros(N, device=device)))
    out["prered_slots"].copy_(slot_by_n.view(world_size * T, top_k))
    out["prered_w"].copy_(w_by_n.view(world_size * T, top_k))

    # ---- final_contrib (source view): card d holds >=1 of my token t's experts ----
    card = (all_topk[rank] // e_local).long()  # (T, TOPK)
    out["final_contrib"].zero_()
    out["final_contrib"].scatter_(1, card, torch.ones_like(card, dtype=torch.int32))

    # ---- T6-v1 combine-push tables (derived from prered_slots / final_contrib) ----
    if "push_expected_l1" in out:
        _derive_combine_push(out["prered_slots"], out["final_contrib"], T, world_size,
                             out["push_expected_l1"], out["recv_from"])

    return P, world_size * T


class TKFusedEP(DistributedScheme):
    name = "tkfused"

    def setup(self, problem: MoEProblem, ctx: DistContext) -> None:
        cfg = problem.config
        assert cfg.parallel_mode == ParallelMode.EP, "TKFusedEP is EP-only"
        assert problem.quant_config is None, "TKFusedEP v1 is bf16-only (no fp8 yet)"
        self.ctx = ctx
        self.problem = problem
        H = cfg.hidden_size
        inter = cfg.intermediate_shard
        world = ctx.world_size
        num_tokens = problem.num_tokens
        num_experts = cfg.num_experts
        e_local = cfg.num_local_experts
        device = problem.hidden_states.device
        self.H, self.inter, self.num_tokens, self.top_k = H, inter, num_tokens, cfg.topk

        from importlib.util import spec_from_file_location, module_from_spec
        import os
        _build_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "kernels", "tk", "build.py")
        _spec = spec_from_file_location("_tk_build", _build_py)
        _bmod = module_from_spec(_spec)
        _spec.loader.exec_module(_bmod)
        self.tk = _bmod.build_and_load(world, hidden=H, row_block=ROW_BLOCK)

        # schedules (host-side, not timed)
        (disp_idx, comb_idx, padded, num_padded_local,
         push_idx, push_src, slack,
         push_cnt_idx, push_expected, gate_expected, dst_blk_offset,
         total_dst_blocks) = _build_schedules(
            problem.topk_ids, num_tokens, world, num_experts, e_local, ctx.rank, device)
        self.disp_idx = disp_idx
        self.comb_idx = comb_idx
        self.padded = padded
        self.num_padded_local = num_padded_local
        self.combine_w = problem.topk_weights.reshape(-1, 1).contiguous().float()
        # push schedule: drop -1 padding rows (every real assignment is valid) so
        # num_push is exactly this card's outgoing assignment count.
        valid = push_src[:, 0] >= 0
        self.push_idx = push_idx[valid].contiguous()
        self.push_src = push_src[valid].contiguous()
        self.push_cnt_idx = push_cnt_idx[valid].contiguous()  # push3, same mask/order
        self.num_push = int(valid.sum())
        self.slack = slack  # (num_padded_local//ROW_BLOCK,) int32
        # push3 (docs/09) expected-value tables + geometry
        self.push_expected = push_expected            # (total_dst_blocks,) int32
        self.gate_expected = gate_expected.contiguous()  # (nblk_local, world) int32
        self.dst_blk_offset = dst_blk_offset          # (world,) int32
        self.total_dst_blocks = total_dst_blocks
        # push3 local counters (this card's outgoing counters), zeroed each iter.
        self.push_local_cnt = torch.zeros(total_dst_blocks, dtype=torch.int32, device=device)
        # dispatch mode: "pull" is the verified/default path. "push" (source-side
        # push + remote red.add) is FROZEN — PCIe remote atomics drop increments
        # under high-concurrency scatter on this machine (see docs/08); it only
        # runs correctly at small expert counts. Kept behind TK_DISPATCH=push for
        # experiments / other platforms.
        # "push3" (docs/10) is the correct, atomic-free source-push ⊕ gate GEMM
        # fusion (slot-signal completion): FASTER than pull at NE<=128, SLOWER at
        # NE=256 (crossover — signal/gate protocol cost grows with row-block count).
        # Correct at all NE. Default stays pull because the prod point is NE=256.
        import os as _os
        self.dispatch_mode = _os.environ.get("TK_DISPATCH", "pull")
        # combine mode: "prered" (T6-v0, docs/13) is the DEFAULT — combine
        # pre-reduction. Expert cards FP32-weighted-sum their own hits per
        # (src,tok) into one partial row (moe_gemm_prered_fused), a barrier
        # publishes them, and the source card sums the <=world contributing
        # partial rows (moe_final_reduce). Cuts sent rows 4096 -> ~1800 and the
        # combine gather (docs/12: 94~99% of layer1 tail); zero NEW cross-card
        # protocol (reuses the verified push2 barrier). Adjudicated correct at
        # NE∈{64,128,256} (tools/validate_prered.py, rel_err ~7e-3) and e2e
        # NE=256 7131->5776µs (beats serial). "pull" (moe_gemm_combine_fused,
        # source gathers 8 peer rows/token) kept behind TK_COMBINE=pull.
        # "prered_push" (T6-v1, docs/18, DEFAULT): expert card pushes each partial
        # row to the source card's staging edge-triggered (strong path, streams
        # under the W2 GEMM) + per-card watermark election, no full barrier —
        # fixes v0's ~0% comm overlap (docs/17). Correct at NE∈{64,128,256}
        # (validate_prered_push.py, rel ~7e-3); e2e (default shape) 2331->1963µs
        # (32.4% less time than serial). "prered" (v0) kept behind
        # TK_COMBINE=prered; "pull"
        # (moe_gemm_combine_fused) behind TK_COMBINE=pull.
        self.combine_mode = _os.environ.get("TK_COMBINE", "prered_push")

        # T4 (docs/11 §T4): fuse gate+up into ONE GEMM. w1 is [gate; up] rows
        # (E, 2*inter, H), so w1.T = (E, H, 2*inter) with columns [gate | up].
        # The dispatch⊕GEMM's N doubles (col_blocks derives from weights.cols(),
        # zero kernel change) so the up projection's compute hides under the
        # comm-bound dispatch instead of running as an exposed second GEMM.
        # DEFAULT on (pull dispatch only): NE=256 e2e 5775->5570µs, NE=64
        # 3712->3227µs, correct (run_tkfused rel_err ~4.3e-3). push* dispatch
        # modes fall back to the two-GEMM path. TK_FUSE_GATEUP=0 disables.
        self.fuse_gateup = _os.environ.get("TK_FUSE_GATEUP", "1") == "1"
        # push* dispatch modes push into w_gate and don't support the doubled-N
        # fused GEMM; fall back to the two-GEMM path for them (keeps gate/up
        # weights materialized below and the run() branch consistent).
        if self.dispatch_mode != "pull":
            self.fuse_gateup = False

        # T7 (docs/15): dedup dispatch — pull each unique (src_dev,src_tok) once
        # cross-card into staging, then local scatter to the gathered slots
        # (measured ~7.2× fewer cross-card pulls at NE=256). BUT dispatch at NE=256
        # is GEMM-bound (gate GEMM ~1.7ms > hidden comm), so dedup only helps where
        # comm is exposed: NE≤128 dispatch-only 2.0ms→~1.1ms (e2e NE=64 −0.47ms);
        # NE=256 neutral/slightly negative (lost fusion overlap). Pairs with T5:
        # once ROW_BLOCK=64 halves the padding/GEMM, NE=256 comm re-exposes and
        # dedup wins there too. DEFAULT OFF (prod point is NE=256); TK_DEDUP=1 on.
        self.dedup_dispatch = (_os.environ.get("TK_DEDUP", "0") == "1"
                               and self.dispatch_mode == "pull")
        if self.dedup_dispatch:
            s2s, need, s_max = _derive_staging(disp_idx, num_tokens, world)
            self.slot_to_staging = s2s.contiguous()
            self.staging_needed = need.contiguous()
            self.staging_s_max = s_max

        # T6-v0 pre-reduction schedule (docs/13). Built regardless of combine_mode
        # (host-side, not timed); only consumed when combine_mode == "prered".
        (prered_dst, prered_slots, prered_w, num_jobs, final_contrib) = \
            _build_prereduce_schedule(problem.topk_ids, problem.topk_weights,
                                      num_tokens, world, num_experts, e_local,
                                      ctx.rank, device)
        self.prered_dst = prered_dst.contiguous()
        self.prered_slots = prered_slots.contiguous()
        self.prered_w = prered_w.contiguous()
        self.num_jobs = num_jobs
        self.final_contrib = final_contrib.contiguous()  # (num_tokens, world) int32

        # T6-v1 (docs/18) combine-push tables: rows this card pushes to each
        # source card (election target) + which cards send me (watermark wait
        # mask). Derived from prered_slots / final_contrib; rebuilt in the GPU
        # schedule too. Only consumed when combine_mode == "prered_push".
        self.push_expected_l1, self.recv_from = _derive_combine_push(
            self.prered_slots, self.final_contrib, num_tokens, world)
        self.push_expected_l1 = self.push_expected_l1.contiguous()
        self.recv_from = self.recv_from.contiguous()
        # per-source-card push counter, zeroed each iter (private, same-stream)
        self.combine_local_cnt = torch.zeros(world, dtype=torch.int32, device=device)

        # T3 (docs/14): GPU-vectorized schedule rebuild, counted in run() for a
        # fair vs-serial number (serial pays routing metadata per run; docs/07 P1).
        # Default ON. Grid sizes (num_padded_local, num_jobs) stay host-computed
        # above for allocation (routing fixed per problem); only the index
        # CONTENTS recompute on GPU each run. Covers the DEFAULT path tables only.
        self.gpu_schedule = _os.environ.get("TK_GPU_SCHED", "1") == "1"
        if self.gpu_schedule:
            N = world * num_tokens * self.top_k
            ar = torch.arange(N, device=device)
            self._sched_grids = {
                "src_dev_grid": ar // (num_tokens * self.top_k),
                "src_tok_grid": (ar // self.top_k) % num_tokens,
                "kpos_grid": ar % self.top_k,
            }
            # per-run routing gather buffers (all ranks' topk_ids / weights)
            self._all_topk = torch.empty(world, num_tokens, self.top_k,
                                         device=device, dtype=problem.topk_ids.dtype)
            self._all_w = torch.empty(world, num_tokens, self.top_k,
                                      device=device, dtype=torch.float32)
            self._topk_ids_local = problem.topk_ids.contiguous()
            self._topk_w_local = problem.topk_weights.float().contiguous()
            self._num_experts = num_experts
            self._e_local = e_local
            # prered_dst is a constant of the dense job space (j = src_dev*T +
            # src_tok); the GPU builder does not rewrite it, so set it once here.
            nj = world * num_tokens
            self.prered_dst[:, 0] = (torch.arange(nj, device=device) // num_tokens).to(torch.int32)
            self.prered_dst[:, 1] = (torch.arange(nj, device=device) % num_tokens).to(torch.int32)
            # in-place GPU write targets (fixed shape/address -> CUDA-graph safe)
            self._sched_out = {
                **self._sched_grids,
                "disp_idx": self.disp_idx, "padded": self.padded,
                "prered_dst": self.prered_dst, "prered_slots": self.prered_slots,
                "prered_w": self.prered_w, "final_contrib": self.final_contrib,
            }
            if self.dedup_dispatch:  # rebuild dedup tables in the timed builder too
                self._sched_out["slot_to_staging"] = self.slot_to_staging
                self._sched_out["staging_needed"] = self.staging_needed
            if self.combine_mode == "prered_push":  # rebuild v1 combine-push tables
                self._sched_out["push_expected_l1"] = self.push_expected_l1
                self._sched_out["recv_from"] = self.recv_from
            self._sched_graph = None  # captured lazily on first run()

        # weights: problem.w1 (E_local, 2*inter, H) is [gate; up] stored for x@w.T.
        # grouped_gemm computes x @ W with W as (K, N), so W = w.T:
        #   gate/up:  x(.,H) @ (H, inter) -> (., inter)   => W_gate = w1[:, :inter, :].transpose(1,2)
        #   W2:       act(.,inter) @ (inter, H) -> (., H)  => W2 = w2.transpose(1,2)
        w1 = problem.w1  # (E_local, 2*inter, H)
        self.w2 = problem.w2.transpose(1, 2).contiguous()            # (E_local, inter, H)
        # T4: fused gate+up weight (E_local, H, 2*inter) columns [gate | up]. When
        # fusing, we don't materialize the separate gate/up copies (saves memory).
        if self.fuse_gateup:
            self.w_gateup = w1.transpose(1, 2).contiguous()          # (E_local, H, 2*inter)
        else:
            self.w_gate = w1[:, :inter, :].transpose(1, 2).contiguous()  # (E_local, H, inter)
            self.w_up = w1[:, inter:, :].transpose(1, 2).contiguous()    # (E_local, H, inter)

        # padded expert-output max across cards (for pgl uniform sizing)
        num_padded_max = int(padded.reshape(world, e_local).sum(dim=1).amax())
        self.num_padded_max = num_padded_max

        TK = self.tk.TKParallelTensor
        lr, lws = ctx.local_rank, world
        # symmetric buffers (allocated once, reused)
        self.pre_tokens = TK((num_tokens, H), dtype=torch.bfloat16, local_rank=lr,
                             local_world_size=lws, multicast=False)
        self.expert_out = TK((num_padded_max, H), dtype=torch.bfloat16, local_rank=lr,
                             local_world_size=lws, multicast=False)
        bar_cols = max(num_padded_max // ROW_BLOCK + 1, 32)
        # push3 (docs/09 §3.2) widens barrier_l0 to (2 + world, bar_cols):
        #   row 0        : pull-mode local row-block counter (retained, pull fallback)
        #   row 1        : pcie_barrier_all arrival slots (retained)
        #   row 2 + s    : source card s's push3 completion signal (col = local row block)
        # Rows 0/1 and bar_cols are unchanged, so pull / pcie_barrier_all are byte-identical.
        self.barrier_l0 = TK((2 + world, bar_cols), dtype=torch.int, local_rank=lr,
                             local_world_size=lws, multicast=False)
        # barrier_l1 rows: 0 = GEMM col-block counter (reset each iter), 1 = local
        # W2 completion signal (prered job/push slot gate). T6-v1 (docs/18) adds
        # rows 2+d = expert card d's cross-card combine watermark. Widen to
        # (2+world); pull-combine (comb) uses rows 0/1+d and stays byte-identical.
        self.barrier_l1 = TK((2 + world, bar_cols), dtype=torch.int, local_rank=lr,
                             local_world_size=lws, multicast=False)
        self.barrier_l0.data_.zero_()
        self.barrier_l1.data_.zero_()

        # local workspaces. gathered is peer-writable (push target) so it's a
        # TKParallelTensor sized to num_padded_max for a uniform pgl view; the GEMM
        # reads only its first num_padded_local rows.
        self.gathered = TK((num_padded_max, H), dtype=torch.bfloat16, local_rank=lr,
                           local_world_size=lws, multicast=False)
        self.gathered.data_.zero_()
        self.gate_out = torch.zeros(num_padded_local, inter, device=device, dtype=torch.bfloat16)
        self.up_out = torch.zeros(num_padded_local, inter, device=device, dtype=torch.bfloat16)
        self.act = torch.zeros(num_padded_local, inter, device=device, dtype=torch.bfloat16)
        # T4: fused gate+up output (padded, 2*inter); halves are [gate | up].
        if self.fuse_gateup:
            self.gateup_out = torch.zeros(num_padded_local, 2 * inter, device=device,
                                          dtype=torch.bfloat16)
        self.combine_out = torch.zeros(num_tokens, H, device=device, dtype=torch.bfloat16)

        # T6-v0 (docs/13) partial buffer: peer-readable (world, num_tokens, H)
        # flattened to (world*num_tokens, H). Plane [s*num_tokens + t] holds THIS
        # card's FP32-weighted pre-reduced contribution to source card s's token t.
        # Only allocated/used when combine_mode == "prered".
        if self.combine_mode == "prered":
            self.partials = TK((world * num_tokens, H), dtype=torch.bfloat16,
                               local_rank=lr, local_world_size=lws, multicast=False)
            self.partials.data_.zero_()

        # T6-v1 (docs/18) combine staging: peer-WRITABLE (world*num_tokens, H) bf16.
        # Plane [d] (rows [d*T, d*T+T)) is written ONLY by expert card d (single
        # writer, no atomics); source card sums the contributing planes. Push target.
        if self.combine_mode == "prered_push":
            self.combine_staging = TK((world * num_tokens, H), dtype=torch.bfloat16,
                                      local_rank=lr, local_world_size=lws, multicast=False)
            self.combine_staging.data_.zero_()

        # T7 (docs/15) staging: LOCAL (world*num_tokens, H) bf16 = 28MB. Holds one
        # copy of each pulled unique source token; the scatter then copies each to
        # its gathered slots. Local-only (no peer touches it) -> plain tensor, no pgl.
        if self.dedup_dispatch:
            self.staging = torch.zeros(world * num_tokens, H, device=device,
                                       dtype=torch.bfloat16)

        self.num_comm_sms = 16
        self._l0_seq = 0
        self._l1_seq = 0

    def run(self) -> torch.Tensor:
        tk = self.tk
        # T3 (docs/14): rebuild the DEFAULT-path schedule on GPU each iteration,
        # inside the timed region — the fair analogue of serial's per-run routing
        # metadata (moe_align_block_size) + routing all_gathers. Grid sizes are
        # fixed (routing invariant per problem), so this only rewrites the index
        # CONTENTS of pre-allocated tables; num_padded_local / num_jobs are
        # asserted unchanged. Only active on the default path (pull + prered).
        # P1 (docs/22): include prered_push — the DEFAULT combine — in the timed
        # GPU schedule rebuild. The old `== "prered"` skipped it, so the default
        # path's e2e (1963µs / 32.4% time reduction) never paid the ~205µs
        # sched cost; the fair result is ~2170µs / 25.3% time reduction
        # (microbench mb3_report_sched205us).
        if self.gpu_schedule and self.dispatch_mode == "pull" \
                and self.combine_mode in ("prered", "prered_push"):
            torch.distributed.all_gather_into_tensor(self._all_topk, self._topk_ids_local)
            torch.distributed.all_gather_into_tensor(self._all_w, self._topk_w_local)
            # The pure-compute builder (no NCCL, fixed shapes, fixed tensor
            # addresses) is CUDA-graph captured on first call and replayed after —
            # this cuts the ~30 eager kernel launches from ~660µs to ~180µs. The
            # all_gathers above stay eager (NCCL can't be captured here).
            if self._sched_graph is None:
                _build_schedules_gpu(self._all_topk, self._all_w, self.ctx.world_size,
                                     self._num_experts, self._e_local, self.ctx.rank,
                                     self._sched_out)  # warmup (populates + allocs)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    _build_schedules_gpu(self._all_topk, self._all_w, self.ctx.world_size,
                                         self._num_experts, self._e_local, self.ctx.rank,
                                         self._sched_out)
                self._sched_graph = g
            else:
                self._sched_graph.replay()

        # P2: barrier BEFORE overwriting pre_tokens, so no peer is still pulling
        # last iteration's tokens when we clobber the buffer. The safety of
        # copy->dispatch used to rely on an implicit chain (peer pull ⇔ combine
        # wait ⇔ W2 launched ⇔ this rank's dispatch done, all in stream order);
        # this explicit barrier makes it robust to any future change in that chain
        # (e.g. skipping combine, changing wait granularity). ~10µs on a ~7ms layer.
        self._l0_seq += 1
        tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)
        self.pre_tokens.data_.copy_(self.problem.hidden_states)
        self._l0_seq += 1
        tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)

        if self.dispatch_mode == "push":
            # Source-side push (optimization #1): seed row-block counters with
            # padding slack, then each card pushes its own tokens to expert cards
            # (strong ~51GB/s path) and remotely bumps their counters. gathered is
            # peer-writable. The GEMM producer gate spins on the local counter.
            self.barrier_l0.data_[0, :self.slack.numel()].copy_(self.slack)
            self._l0_seq += 1
            tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)  # all counters seeded before any push
            tk.moe_dispatch_push(self.pre_tokens, self.gathered, self.w_gate, self.gate_out,
                                 self.padded, self.push_idx, self.push_src, self.barrier_l0,
                                 self.num_comm_sms, self.num_padded_local, self.num_push)
        elif self.dispatch_mode == "push2":
            # Atomic-free push (docs/08 refactor): push tokens to peers (strong
            # ~51GB/s path, no remote atomics), one cross-device barrier for
            # completion, then a plain local grouped GEMM for the gate projection.
            # Trades layer0 comm/compute overlap (unneeded — comm-bound) for the
            # push bandwidth win, reusing only verified primitives.
            tk.moe_push_data(self.pre_tokens, self.gathered, self.w_gate, self.gate_out,
                             self.padded, self.push_idx, self.push_src, self.barrier_l0,
                             self.num_push)
            self._l0_seq += 1
            tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)  # all pushes landed
            tk.grouped_gemm(self.gathered.data_, self.w_gate, self.gate_out, self.padded,
                            self.ctx.rank * self.problem.config.num_local_experts)
        elif self.dispatch_mode == "push3":
            # Source-side push ⊕ gate GEMM with SLOT-signal completion (docs/09,
            # optimization #1c): strong ~51GB/s TMA push data plane AND fusion
            # preserved (gate spins locally while pushes stream in), with NO
            # remote atomics — completion is a single-writer st.release.sys seq
            # per (src card, dst row block), elected by a LOCAL atom.acq_rel that
            # reaches the host-precomputed expected count. Zero seed / pre-barrier
            # needed: local_cnt is a private tensor cleared here (same-stream), and
            # the signal seq is monotonic (immune to reset).
            self.push_local_cnt.zero_()
            self._l0_seq += 1
            tk.moe_dispatch_push3(self.pre_tokens, self.gathered, self.w_gate, self.gate_out,
                                  self.padded, self.push_idx, self.push_src, self.push_cnt_idx,
                                  self.push_local_cnt, self.push_expected, self.gate_expected,
                                  self.barrier_l0, self.num_comm_sms, self.num_padded_local,
                                  self.num_push, self._l0_seq)
        else:  # "pull" fallback
            # T4: when fusing gate+up, the dispatch⊕GEMM does BOTH projections in
            # one pass (w_gateup / gateup_out, N doubled). Otherwise gate only.
            gw = self.w_gateup if self.fuse_gateup else self.w_gate
            go = self.gateup_out if self.fuse_gateup else self.gate_out
            if self.dedup_dispatch:
                # T7 (docs/15): pull unique source tokens to staging then local
                # scatter ⊕ gate GEMM — same gathered bytes, ~7× less cross-card.
                tk.moe_dispatch_dedup(self.pre_tokens, self.staging, self.gathered.data_,
                                      gw, go, self.padded, self.slot_to_staging,
                                      self.staging_needed, self.barrier_l0,
                                      self.num_comm_sms, self.num_padded_local,
                                      self.num_tokens)
            else:
                tk.moe_dispatch_gemm(self.pre_tokens, self.gathered.data_, gw, go,
                                     self.padded, self.disp_idx, self.barrier_l0,
                                     self.num_comm_sms, self.num_padded_local)

        if self.fuse_gateup and self.dispatch_mode == "pull":
            # gate = gateup_out[:, :inter], up = gateup_out[:, inter:]; up compute
            # was hidden under the dispatch comm. silu-mul the two halves.
            inter = self.inter
            torch.mul(F.silu(self.gateup_out[:, :inter]),
                      self.gateup_out[:, inter:], out=self.act)
        else:
            # up GEMM on the gathered tokens (local)
            tk.grouped_gemm(self.gathered.data_, self.w_up, self.up_out, self.padded,
                            self.ctx.rank * self.problem.config.num_local_experts)
            # SiLU-mul, in-place into preallocated self.act (allocation-stable)
            torch.mul(F.silu(self.gate_out), self.up_out, out=self.act)

        # layer1: W2 GEMM ⊕ combine
        self._l1_seq += 1
        if self.combine_mode == "prered":
            # T6-v0 (docs/13): W2 GEMM ⊕ LOCAL pre-reduction. Expert cards write
            # expert_out, then job blocks FP32-weighted-sum their own hits per
            # (src,tok) into a peer-readable partial row (barrier_l1: row 0 =
            # per-row-block counter, row 1 = local completion signal). A separate
            # pcie_device_barrier (barrier_l0, verified push2 publish) makes all
            # cards' partials system-visible, then moe_final_reduce sums the
            # <=world contributing partial rows into combine_out. Reset row-0
            # counters happen inside moe_gemm_prered_fused (reset_kernel).
            tk.moe_gemm_prered_fused(self.act, self.w2, self.expert_out, self.padded,
                                     self.partials, self.prered_dst, self.prered_slots,
                                     self.prered_w, self.barrier_l1, self.num_comm_sms,
                                     self.num_padded_local, self.num_tokens,
                                     self.num_jobs, self._l1_seq)
            self._l0_seq += 1
            tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)  # all partials published
            tk.moe_final_reduce(self.partials, self.final_contrib, self.combine_out,
                                self.barrier_l0, self.num_tokens)
        elif self.combine_mode == "prered_push":
            # T6-v1 (docs/18): W2 GEMM ⊕ prered PUSH. Expert card pushes each
            # partial row to the source card's staging edge-triggered (streams
            # under the W2 GEMM) + per-card watermark election (barrier_l1 row
            # 2+d). NO full barrier — final_reduce_push waits only on the <=world
            # cards in recv_from. local_cnt zeroed each iter (same-stream); seq is
            # monotonic (immune to reset). Same seq gates the epilogue signal
            # (row 1), the push election (row 2+d), and the source wait.
            self.combine_local_cnt.zero_()
            tk.moe_gemm_prered_push_fused(self.act, self.w2, self.expert_out, self.padded,
                                          self.combine_staging, self.prered_dst,
                                          self.prered_slots, self.prered_w,
                                          self.combine_local_cnt, self.push_expected_l1,
                                          self.barrier_l1, self.num_comm_sms,
                                          self.num_padded_local, self.num_tokens,
                                          self.num_jobs, self._l1_seq)
            tk.moe_final_reduce_push(self.combine_staging, self.final_contrib,
                                     self.recv_from, self.combine_out, self.barrier_l1,
                                     self.num_tokens, self._l1_seq)
        else:
            tk.moe_gemm_combine_fused(self.act, self.w2, self.expert_out, self.padded,
                                      self.combine_out, self.comb_idx, self.combine_w,
                                      self.barrier_l1, self.num_comm_sms,
                                      self.num_padded_local, self.num_tokens, self._l1_seq)
        return self.combine_out
