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

ROW_BLOCK = 128


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

    Returns, EXPERT-CARD view (this rank as producer d == rank):
      prered_dst   (J, 2)     int32 : (src_dev, src_tok) each partial job targets
      prered_slots (J, TOP_K) int32 : local slots to FP32-reduce (-1 padded)
      prered_w     (J, TOP_K) f32   : weights aligned to slots (0 padded)
      num_jobs J
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

    # expert-card side: (src_dev, src_tok) -> list of (local_slot, weight) this
    # card must FP32-reduce into one partial row. source-card side: contrib mask.
    jobs: dict[tuple[int, int], list[tuple[int, float]]] = {}
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
                        if e_rank == rank:  # I am the expert card: accumulate a partial job
                            w = float(all_w_cpu[src_dev, src_tok, kpos])
                            jobs.setdefault((src_dev, src_tok), []).append((slot, w))
                        if src_dev == rank:  # I am the source card: mark contributor
                            final_contrib[src_tok, e_rank] = 1

    num_jobs = len(jobs)
    prered_dst = torch.full((max(num_jobs, 1), 2), -1, dtype=torch.int32, device="cpu")
    prered_slots = torch.full((max(num_jobs, 1), top_k), -1, dtype=torch.int32, device="cpu")
    prered_w = torch.zeros((max(num_jobs, 1), top_k), dtype=torch.float32, device="cpu")
    for j, (src_dev, src_tok) in enumerate(sorted(jobs.keys())):
        prered_dst[j, 0] = src_dev
        prered_dst[j, 1] = src_tok
        for c, (slot, w) in enumerate(jobs[(src_dev, src_tok)]):
            prered_slots[j, c] = slot
            prered_w[j, c] = w

    return (prered_dst.to(device), prered_slots.to(device), prered_w.to(device),
            num_jobs, final_contrib.to(device))


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
        self.tk = _bmod.build_and_load(world, hidden=H)

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
        self.combine_mode = _os.environ.get("TK_COMBINE", "prered")

        # T6-v0 pre-reduction schedule (docs/13). Built regardless of combine_mode
        # (host-side, not timed); only consumed when combine_mode == "prered".
        (prered_dst, prered_slots, prered_w, num_jobs, final_contrib) = \
            _build_prereduce_schedule(problem.topk_ids, problem.topk_weights,
                                      num_tokens, world, num_experts, e_local,
                                      ctx.rank, device)
        self.prered_dst = prered_dst
        self.prered_slots = prered_slots
        self.prered_w = prered_w
        self.num_jobs = num_jobs
        self.final_contrib = final_contrib.contiguous()  # (num_tokens, world) int32

        # weights: problem.w1 (E_local, 2*inter, H) is [gate; up] stored for x@w.T.
        # grouped_gemm computes x @ W with W as (K, N), so W = w.T:
        #   gate/up:  x(.,H) @ (H, inter) -> (., inter)   => W_gate = w1[:, :inter, :].transpose(1,2)
        #   W2:       act(.,inter) @ (inter, H) -> (., H)  => W2 = w2.transpose(1,2)
        w1 = problem.w1  # (E_local, 2*inter, H)
        self.w_gate = w1[:, :inter, :].transpose(1, 2).contiguous()  # (E_local, H, inter)
        self.w_up = w1[:, inter:, :].transpose(1, 2).contiguous()    # (E_local, H, inter)
        self.w2 = problem.w2.transpose(1, 2).contiguous()            # (E_local, inter, H)

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
        self.barrier_l1 = TK((1 + world, bar_cols), dtype=torch.int, local_rank=lr,
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
        self.combine_out = torch.zeros(num_tokens, H, device=device, dtype=torch.bfloat16)

        # T6-v0 (docs/13) partial buffer: peer-readable (world, num_tokens, H)
        # flattened to (world*num_tokens, H). Plane [s*num_tokens + t] holds THIS
        # card's FP32-weighted pre-reduced contribution to source card s's token t.
        # Only allocated/used when combine_mode == "prered".
        if self.combine_mode == "prered":
            self.partials = TK((world * num_tokens, H), dtype=torch.bfloat16,
                               local_rank=lr, local_world_size=lws, multicast=False)
            self.partials.data_.zero_()

        self.num_comm_sms = 16
        self._l0_seq = 0
        self._l1_seq = 0

    def run(self) -> torch.Tensor:
        tk = self.tk
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
            tk.moe_dispatch_gemm(self.pre_tokens, self.gathered.data_, self.w_gate, self.gate_out,
                                 self.padded, self.disp_idx, self.barrier_l0,
                                 self.num_comm_sms, self.num_padded_local)

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
        else:
            tk.moe_gemm_combine_fused(self.act, self.w2, self.expert_out, self.padded,
                                      self.combine_out, self.comb_idx, self.combine_w,
                                      self.barrier_l1, self.num_comm_sms,
                                      self.num_padded_local, self.num_tokens, self._l1_seq)
        return self.combine_out
