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

    top_k = topk_ids.shape[1]
    disp_idx = torch.full((num_padded_local, 2), -1, dtype=torch.int32, device="cpu")
    comb_idx = torch.full((num_tokens * top_k, 2), -1, dtype=torch.int32, device="cpu")
    push_idx = torch.full((num_tokens * top_k, 2), -1, dtype=torch.int32, device="cpu")
    push_src = torch.full((num_tokens * top_k, 1), -1, dtype=torch.int32, device="cpu")

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
                        if src_dev == rank:  # this rank is the source card -> combine + push
                            comb_idx[src_tok * top_k + kpos, 0] = e_rank
                            comb_idx[src_tok * top_k + kpos, 1] = slot
                            push_idx[src_tok * top_k + kpos, 0] = e_rank
                            push_idx[src_tok * top_k + kpos, 1] = slot
                            push_src[src_tok * top_k + kpos, 0] = src_tok

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

    return (disp_idx.to(device), comb_idx.to(device), padded.to(device),
            num_padded_local, push_idx.to(device), push_src.to(device), slack.to(device))


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
         push_idx, push_src, slack) = _build_schedules(
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
        self.num_push = int(valid.sum())
        self.slack = slack  # (num_padded_local//ROW_BLOCK,) int32
        # dispatch mode: "pull" is the verified/default path. "push" (source-side
        # push + remote red.add) is FROZEN — PCIe remote atomics drop increments
        # under high-concurrency scatter on this machine (see docs/08); it only
        # runs correctly at small expert counts. Kept behind TK_DISPATCH=push for
        # experiments / other platforms.
        import os as _os
        self.dispatch_mode = _os.environ.get("TK_DISPATCH", "pull")

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
        self.barrier_l0 = TK((2, bar_cols), dtype=torch.int, local_rank=lr,
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
        tk.moe_gemm_combine_fused(self.act, self.w2, self.expert_out, self.padded,
                                  self.combine_out, self.comb_idx, self.combine_w,
                                  self.barrier_l1, self.num_comm_sms,
                                  self.num_padded_local, self.num_tokens, self._l1_seq)
        return self.combine_out
