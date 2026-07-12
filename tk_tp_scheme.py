# SPDX-License-Identifier: Apache-2.0
"""TK fused MoE scheme (bf16, TP) for the distributed benchmark.

TP shards the INTERMEDIATE dim (every card holds all E experts, thin), so the
layer's cross-card traffic is dense and routing-independent:

  1. AllGather ⊕ gate+up GEMM   (layer0 fused, moe_tp_dispatch_gemm)
       every unique (src_dev, src_tok) row is pulled ONCE cross-card (ring
       order, own shard first) and scattered to its TOP_K expert-sorted
       gathered slots locally — the T7 dedup insight is the NATURAL TP form,
       since all top-k experts of every token live on this card.
  2. act = silu(gate) * up      (torch, on the [gate | up] halves)
  3. W2 GEMM ⊕ prered-push      (layer1 fused, moe_tp_gemm_prered_push)
       the top-k weighted combine is FULLY LOCAL in TP (T6 prered with all
       TOP_K hits local); the cross-card step degenerates to a dense
       ReduceScatter of (T, H) partial rows, done with the verified T6-v1
       push + watermark protocol (edge-triggered, streams under the GEMM).
  4. moe_final_reduce_push      (source card sums the world partial planes)

All communication is PCIe-safe (experience/12): unicast pull/push, local
red.release.gpu counters, st.release.sys slot signals — no remote atomics,
no multimem. Kernels are shared with the EP scheme (tk_scheme.py); only the
dispatch data plane (pull-once-scatter-TOP_K) and the entry geometry
(expert_offset=0, all experts local) are TP-specific.

bf16 only, TOP_K must equal the kernels' compile-time TOP_K (8).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .config import ParallelMode
from .context import DistContext
from .data import MoEProblem
from .schemes import DistributedScheme
from .tk_scheme import ROW_BLOCK


def _build_tp_schedules(topk_ids, topk_weights, num_tokens, world_size,
                        num_experts, rank, device):
    """Host golden TP schedule (setup only, not timed). Produces:
      - padded (num_experts,) int32: per-expert ROW_BLOCK-padded token counts
        over the FULL world*T batch (identical on every rank);
      - tp_slots (world*T, TOP_K) int32: gathered slot of every assignment,
        row j = src_dev*T + src_tok, column = kpos. This ONE table drives both
        layer0 (dispatch scatters the pulled token row to slots) and layer1
        (prered gathers the same slots' W2 rows) — in TP they are the same map.
      - tp_w (world*T, TOP_K) float32: routing weight per assignment;
      - slack (nblk,) int32: ROW_BLOCK - real tokens per row block (counter seed);
      - pull_order (world*T,) int32: unique tokens sorted by MIN gathered slot
        (expert-major; docs/20 — a ring-by-source order made every row block
        wait for the LAST ring stage, stalling the GEMM behind the whole AG);
      - job_order (world*T,) int32: jobs sorted by MAX slot = readiness order
        for the layer1 dispenser (docs/20);
      - push_order (world, T) int32: source s's tokens sorted by their MIN slot
        — the order card s PUSHES its shard in TP-T1 (docs/23), which must be
        byte-identical on every rank (producer and consumers replay it for the
        chunk-watermark position mapping);
      - num_padded_total: gathered rows (= sum(padded)).

    Slot order within an expert is CANONICAL (src_dev, src_tok, kpos) —
    identical on every rank. docs/23: the push dispatch requires a layout all
    ranks agree on (the old per-rank ring made push_order rank-dependent), and
    "own shard first" was already proven useless in docs/20.
    """
    top_k = topk_ids.shape[1]
    all_topk = torch.empty(world_size, num_tokens, top_k, device=device,
                           dtype=topk_ids.dtype)
    torch.distributed.all_gather_into_tensor(all_topk, topk_ids.contiguous())
    all_w = torch.empty(world_size, num_tokens, top_k, device=device,
                        dtype=torch.float32)
    torch.distributed.all_gather_into_tensor(all_w, topk_weights.float().contiguous())
    all_topk_cpu = all_topk.cpu()
    all_w_cpu = all_w.cpu()

    counts = torch.bincount(all_topk.view(-1), minlength=num_experts).cpu().long()
    padded = (counts + ROW_BLOCK - 1) // ROW_BLOCK * ROW_BLOCK
    num_padded_total = int(padded.sum())

    # NOTE: explicit device="cpu" everywhere — the distributed worker runs under
    # torch.set_default_device(cuda), which would silently move these host-side
    # tables (and turn the fill loop into 16k+ single-element GPU writes).
    write_pos = torch.cat([
        torch.zeros(1, dtype=torch.int64, device="cpu"),
        torch.cumsum(padded[:-1], dim=0)
    ]).tolist()
    S = world_size * num_tokens
    tp_slots = torch.full((S, top_k), -1, dtype=torch.int32, device="cpu")
    tp_w = torch.zeros((S, top_k), dtype=torch.float32, device="cpu")
    for src_dev in range(world_size):  # canonical: src_dev ascending on EVERY rank
        for src_tok in range(num_tokens):
            j = src_dev * num_tokens + src_tok
            for kpos, eid in enumerate(all_topk_cpu[src_dev, src_tok].tolist()):
                slot = write_pos[eid]
                write_pos[eid] += 1
                tp_slots[j, kpos] = slot
                tp_w[j, kpos] = float(all_w_cpu[src_dev, src_tok, kpos])

    nblk = num_padded_total // ROW_BLOCK
    slack = torch.zeros(nblk, dtype=torch.int32, device="cpu")
    blk = 0
    for e in range(num_experts):
        real_e = int(counts[e])
        for b in range(int(padded[e]) // ROW_BLOCK):
            real_in = max(0, min(ROW_BLOCK, real_e - b * ROW_BLOCK))
            slack[blk] = ROW_BLOCK - real_in
            blk += 1

    # docs/20 orders. min/max slots are already unique across tokens (slots are
    # a bijection and each slot belongs to one token), so argsort needs no
    # stability; the +index tie-break just keeps host/GPU byte-identical under
    # any future table change.
    mins = tp_slots.min(dim=1).values.long()
    pull_order = torch.argsort(mins * S + torch.arange(S, device="cpu")).to(torch.int32)
    maxs = tp_slots.max(dim=1).values.long()
    job_order = torch.argsort(maxs * S + torch.arange(S, device="cpu")).to(torch.int32)
    # docs/23 TP-T1: per-source push order = that source's tokens by min slot.
    # Canonical layout makes this identical on all ranks (min slots are unique
    # within a row, argsort stability irrelevant).
    push_order = torch.argsort(mins.view(world_size, num_tokens), dim=1).to(torch.int32)

    # docs/30: row block -> expert id (drives the dispenser GEMM's B-tile index;
    # first expert whose cumulative row-block end exceeds the block index).
    rb_end = torch.cumsum(padded // ROW_BLOCK, dim=0)
    blk_expert = torch.searchsorted(
        rb_end, torch.arange(nblk, device="cpu"), right=True).to(torch.int32)

    return (padded.to(torch.int32).to(device), tp_slots.to(device),
            tp_w.to(device), slack.to(device), pull_order.to(device),
            job_order.to(device), push_order.to(device), blk_expert.to(device),
            num_padded_total)


def _build_tp_schedules_gpu(packed_all, world_size, num_experts, rank, out):
    """GPU-vectorized rebuild of the TP schedule tables, element-for-element
    identical to the host `_build_tp_schedules` golden (tools/verify_tp_schedule
    adjudicates). Called each run() and counted in timing — the fair analogue of
    serial's per-run routing all_gathers + moe_align_block_size (docs/07 P1,
    docs/14). Same single-argsort trick as the EP builder: slot = rank of the
    assignment in the total order (eid, ring, src_tok, kpos), every key unique
    (bijection over flattened all_topk) so stability is not required.

    docs/30 sched-merge: the input is now ONE gathered tensor
    `packed_all (world, T, TOP_K, 2) int32` — [..., 0] = topk ids,
    [..., 1] = the float32 routing weights' BIT PATTERN (view(int32)); one
    NCCL all_gather replaces the former two. Weights are bit-copied into
    prered_w through its int32 view (reinterpret, not a dtype cast).

    In TP every assignment is local, so there is no trash-row redirect; and the
    layer1 tables (push_expected_l1 / recv_from / final_contrib / prered_dst)
    are routing-INDEPENDENT constants (every card holds all experts), set once
    in setup and not rebuilt here.

    Fixed shapes, no host sync, no bincount — CUDA-graph capturable.
    """
    device = packed_all.device
    T = packed_all.shape[1]
    top_k = packed_all.shape[2]
    N = world_size * T * top_k

    eid = packed_all[..., 0].reshape(N).long()
    # canonical order within an expert = (src_dev, src_tok, kpos) = flat index n
    # (docs/23: all ranks must build the SAME layout for the push dispatch).
    # docs/44 sched 瘦身: key 用 int32(值域 E*N ≤ 256*131072 < 2^31, 排序
    # 结果与 int64 逐元素相同, verify 对拍不变), argsort 提速 ~30-40%。
    key = (eid * N + torch.arange(N, device=device)).to(torch.int32)
    order = torch.argsort(key)
    eid_s = eid[order]

    # per-expert counts + padding (scatter_add: capture-safe, unlike bincount)
    counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    counts.scatter_add_(0, eid, torch.ones_like(eid))
    padded = (counts + ROW_BLOCK - 1) // ROW_BLOCK * ROW_BLOCK
    out["padded"].copy_(padded.to(torch.int32))
    grp_start = torch.zeros(num_experts, dtype=torch.long, device=device)
    grp_start[1:] = torch.cumsum(counts, dim=0)[:-1]
    pos_in_expert = torch.arange(N, device=device) - grp_start[eid_s]
    padded_base = torch.zeros(num_experts, dtype=torch.long, device=device)
    padded_base[1:] = torch.cumsum(padded, dim=0)[:-1]
    slot_sorted = padded_base[eid_s] + pos_in_expert

    # tp_slots: scatter each sorted assignment's slot back to its original flat
    # index n = (src_dev*T + src_tok)*top_k + kpos == row/col of the table.
    slot_by_n = torch.empty(N, dtype=torch.int32, device=device)
    slot_by_n.scatter_(0, order, slot_sorted.to(torch.int32))
    out["tp_slots"].copy_(slot_by_n.view(world_size * T, top_k))
    # weights: flat n already IS (j, kpos); bit-copy the float32 pattern
    # through prered_w's int32 view (docs/30 sched-merge packing).
    out["prered_w"].view(torch.int32).copy_(
        packed_all[..., 1].reshape(world_size * T, top_k))

    # slack: only the tail block of each expert carries (padded - real); scatter
    # via a trash slot for empty experts (capture-safe, no boolean compaction).
    nb_total = out["slack"].shape[0]
    tail_blk = (padded_base + padded) // ROW_BLOCK - 1          # (E,)
    slackv = (padded - counts).to(torch.int32)
    idx = torch.where(padded > 0, tail_blk, torch.full_like(tail_blk, nb_total))
    tmp = torch.zeros(nb_total + 1, dtype=torch.int32, device=device)
    tmp.scatter_(0, idx, slackv)
    out["slack"].copy_(tmp[:nb_total])

    # docs/20 + docs/23 orders (see host golden for rationale)
    S = world_size * T
    tpl = out["tp_slots"].long()
    mins = tpl.min(dim=1).values
    out["pull_order"].copy_(
        torch.argsort((mins * S + torch.arange(S, device=device))
                      .to(torch.int32)).to(torch.int32))
    # L1 v2(docs/32)不再消费 job_order(列扫聚合信号取代 max-slot 就绪序),
    # 只有 v1 路径要求重建 —— 与 push_order 同款的按需策略(docs/26)。
    if "job_order" in out:
        out["job_order"].copy_(
            torch.argsort((tpl.max(dim=1).values * S + torch.arange(S, device=device))
                          .to(torch.int32)).to(torch.int32))
    if "push_order" in out:
        out["push_order"].copy_(
            torch.argsort(mins.view(world_size, T), dim=1).to(torch.int32))

    # docs/30: row block -> expert (dispenser GEMM B-tile index). searchsorted
    # keeps shapes fixed (capture-safe), mirrors the host golden formula.
    if "blk_expert" in out:
        nblk = out["blk_expert"].shape[0]
        rb_end = torch.cumsum(padded // ROW_BLOCK, dim=0)
        out["blk_expert"].copy_(torch.searchsorted(
            rb_end, torch.arange(nblk, device=device), right=True).to(torch.int32))


class TKFusedTP(DistributedScheme):
    name = "tktp"

    def setup(self, problem: MoEProblem, ctx: DistContext) -> None:
        cfg = problem.config
        assert cfg.parallel_mode == ParallelMode.TP, "TKFusedTP is TP-only"
        assert cfg.topk == 8, "kernels compile TOP_K=8"
        # docs/39 P2: fp8 = L0 走 fp8 AG + fp8 GEMM(GLU 融合), L1 保持 bf16
        # (act/push/combine 精度不变, 阶段边界干净)。
        self.fp8 = problem.quant_config is not None
        self.l1_fp8 = False  # fp8 分支里按 TK_L1_FP8 重置(docs/42)
        if self.fp8:
            assert cfg.block_shape == [128, 128], "fp8 kernels assume [128,128] blocks"
        self.ctx = ctx
        self.problem = problem
        H = cfg.hidden_size
        inter = cfg.intermediate_shard          # I / world (TP shard)
        world = ctx.world_size
        num_tokens = problem.num_tokens         # per-rank T
        num_experts = cfg.num_experts
        device = problem.hidden_states.device
        self.H, self.inter, self.num_tokens, self.top_k = H, inter, num_tokens, cfg.topk
        # GEMM template constraints (sm120_common.cuh)
        assert H % 64 == 0 and H % 128 == 0, "hidden must be tile-aligned"
        assert (2 * inter) % 128 == 0, "gate_up shard % COL_BLOCK"
        assert inter % 64 == 0, "intermediate shard % RED_BLOCK"

        from importlib.util import spec_from_file_location, module_from_spec
        _build_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "kernels", "tk", "build.py")
        _spec = spec_from_file_location("_tk_build", _build_py)
        _bmod = module_from_spec(_spec)
        _spec.loader.exec_module(_bmod)
        self.tk = _bmod.build_and_load(world, hidden=H, row_block=ROW_BLOCK)

        # ---- host golden schedule (not timed) ----
        (padded, tp_slots, tp_w, slack, pull_order, job_order, push_order,
         blk_expert, num_padded_total) = _build_tp_schedules(
            problem.topk_ids, problem.topk_weights, num_tokens, world,
            num_experts, ctx.rank, device)
        self.padded = padded
        self.tp_slots = tp_slots.contiguous()
        self.prered_w = tp_w.contiguous()
        self.slack = slack.contiguous()
        self.pull_order = pull_order.contiguous()
        self.job_order = job_order.contiguous()
        self.push_order = push_order.contiguous()
        self.blk_expert = blk_expert.contiguous()
        self.num_padded_total = num_padded_total
        self.num_jobs = world * num_tokens
        # layer1 dispenser counter (docs/20), zeroed each iter (same-stream)
        self.job_next = torch.zeros(1, dtype=torch.int32, device=device)
        # docs/30: layer0 v2 = dispenser GEMM (comm blocks join after the AG)
        # + fused SwiGLU store (weights column-interleaved). Independent
        # rollback switches: TK_L0=v1 restores the static-walk kernel wholesale,
        # TK_L0_GLU=0 keeps the dispenser but stores gateup_out + torch silu.
        self.l0_mode = os.environ.get("TK_L0", "v2")
        self.l0_glu = (os.environ.get("TK_L0_GLU", "1") == "1"
                       and self.l0_mode == "v2") or self.fp8  # fp8 kernel 自带 GLU
        # layer0 dispenser task counter, zeroed each iter (same-stream)
        self.gemm_next = torch.zeros(1, dtype=torch.int32, device=device)
        # docs/32~35: layer1 v2 = N 维分解 combine(Comet layer1-N)。三轮实测
        # 后**默认回 v1**(docs/35 负结果):GRP=16 修复粒度病后 v2 仍 786 vs
        # v1 691,小预算 sweep 单调反向 —— 本机 L1 GEMM 是 SM-bound + 后排空
        # 全员并行已近最优,N 维分解买不回调度成本(与 Comet 的 NVLink 结论
        # 是平台差异)。TK_L1=v2 保留可复现。
        self.l1_mode = os.environ.get("TK_L1", "v1")
        self.l1_gemm_next = torch.zeros(1, dtype=torch.int32, device=device)
        # TP-T1 (docs/23): dispatch data plane. "pull" = tpdisp (weak path,
        # 23.5GB/s under 4-way concurrency, 16 comm SMs); "push" = tppdisp
        # (strong path 50.9GB/s, 4 SMs saturate, chunk watermarks). Default
        # pull until push wins all tiers (rollback policy docs/23).
        self.dispatch_mode = os.environ.get("TK_TP_DISPATCH", "pull")
        # per-(dst, chunk) election counters, zeroed each iter; CHUNK=64 must
        # match tppdisp::globals::CHUNK.
        self._nchunks = (num_tokens + 63) // 64
        self.l0_push_cnt = torch.zeros(world * self._nchunks, dtype=torch.int32,
                                       device=device)
        self.num_push_sms = int(os.environ.get("TK_TP_PUSH_SMS", "4"))

        # ---- layer1 constants (routing-independent in TP) ----
        # prered_dst: job j -> (src_dev, src_tok) of the dense job space.
        nj = self.num_jobs
        self.prered_dst = torch.empty(nj, 2, dtype=torch.int32, device=device)
        self.prered_dst[:, 0] = (torch.arange(nj, device=device) // num_tokens).to(torch.int32)
        self.prered_dst[:, 1] = (torch.arange(nj, device=device) % num_tokens).to(torch.int32)
        # every card holds all experts -> every job non-empty, every card
        # contributes to every token: dense expected counts / contrib masks.
        self.push_expected_l1 = torch.full((world,), num_tokens,
                                           dtype=torch.int32, device=device)
        self.recv_from = torch.ones(world, dtype=torch.int32, device=device)
        self.final_contrib = torch.ones(num_tokens, world,
                                        dtype=torch.int32, device=device)
        self.combine_local_cnt = torch.zeros(world, dtype=torch.int32, device=device)

        # ---- GPU schedule rebuild (T3 fairness: counted in run()) ----
        self.gpu_schedule = os.environ.get("TK_GPU_SCHED", "1") == "1"
        if self.gpu_schedule:
            N = world * num_tokens * self.top_k
            ar = torch.arange(N, device=device)
            # docs/30 sched-merge: ids + weight bits ride ONE all_gather.
            # packed[..., 0] = topk ids, packed[..., 1] = float32 bit pattern.
            self._packed_local = torch.empty(num_tokens, self.top_k, 2,
                                             device=device, dtype=torch.int32)
            self._packed_all = torch.empty(world, num_tokens, self.top_k, 2,
                                           device=device, dtype=torch.int32)
            self._topk_ids_local = problem.topk_ids.contiguous()
            self._topk_w_local = problem.topk_weights.float().contiguous()
            self._topk_w_bits = self._topk_w_local.view(torch.int32)
            self._num_experts = num_experts
            self._sched_out = {
                "src_dev_grid": ar // (num_tokens * self.top_k),
                "src_tok_grid": (ar // self.top_k) % num_tokens,
                "kpos_grid": ar % self.top_k,
                "padded": self.padded, "tp_slots": self.tp_slots,
                "prered_w": self.prered_w, "slack": self.slack,
                "pull_order": self.pull_order,
                "blk_expert": self.blk_expert,
            }
            # job_order 只有 L1 v1 消费(v2 的列扫信号取代就绪序, docs/32),
            # 不进 v2 的计时重建 —— 同 push_order 的按需策略。
            if self.l1_mode != "v2":
                self._sched_out["job_order"] = self.job_order
            # push_order is only consumed by the (frozen) push dispatch — keep
            # it out of the timed per-iter rebuild on the pull path (docs/26:
            # sched is the largest single compressible slice, 277µs @ 12%).
            if self.dispatch_mode == "push":
                self._sched_out["push_order"] = self.push_order
            self._sched_graph = None  # captured lazily on first run()

        # ---- weights ----
        if self.fp8:
            # docs/39: fp8 权重先反量化(setup 一次), L1 用 bf16; L0 在 GLU
            # 列交织后的布局上重量化(scale 块与 GEMM tile 对齐, docs/37 §2)。
            qc = problem.quant_config
            FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
            # docs/42 P3: L1 fp8(默认)—— w2 (E,H,inter) 就是 B^T 布局,
            # qc.w2_scale (E,H/128,inter/128) 原样可用, 零转置零重量化。
            self.l1_fp8 = os.environ.get("TK_L1_FP8", "1") == "1"
            # docs/43: 线上字节改走 copy engine(0 SM, 打破 SM 零和);
            # TK 抽象 = TKParallelTensor.raw_ptrs_ + side streams(ce:: 编排)。
            self.l0_ce = os.environ.get("TK_L0_CE", "0") == "1"
            self.l1_ce = os.environ.get("TK_L1_CE", "0") == "1"
            s1 = qc.w1_scale.float()                         # (E, 2I/128, H/128)
            w1_bf = (problem.w1.float() * s1.repeat_interleave(128, 1)
                                            .repeat_interleave(128, 2))
            if self.l1_fp8:
                self.w2_fp8 = problem.w2.contiguous()        # (E, H, inter) fp8
                self.w2_scales = qc.w2_scale.float().contiguous()
            else:
                s2 = qc.w2_scale.float()                     # (E, H/128, I/128)
                w2_bf = (problem.w2.float() * s2.repeat_interleave(128, 1)
                                                .repeat_interleave(128, 2))
                self.w2 = w2_bf.to(torch.bfloat16).transpose(1, 2).contiguous()
                del w2_bf
            # GLU 列交织(按 N 行, 单位 64): [gate64 | up64] per 128-N block
            E = w1_bf.shape[0]
            gate = w1_bf[:, :inter].view(E, inter // 64, 64, H)
            up = w1_bf[:, inter:].view(E, inter // 64, 64, H)
            w1_il = torch.stack([gate, up], dim=2).view(E, 2 * inter, H)
            # 交织后 128(N)×128(K) 重量化 -> scale 块与 B tile 天然对齐
            v = w1_il.view(E, 2 * inter // 128, 128, H // 128, 128)
            amax = v.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-8)
            self.w1_il_scales = (amax / FP8_MAX).view(
                E, 2 * inter // 128, H // 128).contiguous()
            self.w_gateup_fp8 = (v / (amax / FP8_MAX)).to(torch.float8_e4m3fn) \
                                                      .view(E, 2 * inter, H).contiguous()
            del w1_bf, w1_il, v
        else:
            w1 = problem.w1                                     # (E, 2*inter, H), [gate; up]
            self.w_gateup = w1.transpose(1, 2).contiguous()     # (E, H, 2*inter), [gate | up]
            self.w2 = problem.w2.transpose(1, 2).contiguous()   # (E, inter, H)
        if self.l0_glu and not self.fp8:
            # docs/30: column-interleave so every 128-col GEMM tile is
            # [gate64 | up64] of the SAME intermediate columns — the SwiGLU
            # epilogue pairs the halves inside the accumulator. Setup-time
            # permutation, run-time free.
            E = self.w_gateup.shape[0]
            gate = self.w_gateup[:, :, :inter].reshape(E, H, inter // 64, 64)
            up = self.w_gateup[:, :, inter:].reshape(E, H, inter // 64, 64)
            self.w_gateup_il = torch.stack([gate, up], dim=3) \
                                    .reshape(E, H, 2 * inter).contiguous()

        # ---- buffers ----
        TK = self.tk.TKParallelTensor
        lr, lws = ctx.local_rank, world
        P = num_padded_total
        # peer-readable token shard (dispatch pull source)
        if self.fp8:
            # fp8 AG(docs/39): 源端量化后 4KB/token + 128B scales, AG 字节减半
            self.pre_tokens = TK((num_tokens, H), dtype=torch.float8_e4m3fn,
                                 local_rank=lr, local_world_size=lws, multicast=False)
            self.pre_scales = TK((num_tokens, H // 128), dtype=torch.float32,
                                 local_rank=lr, local_world_size=lws, multicast=False)
        else:
            self.pre_tokens = TK((num_tokens, H), dtype=torch.bfloat16, local_rank=lr,
                                 local_world_size=lws, multicast=False)
        bar_cols = max(P // ROW_BLOCK + 1, 32)
        # barrier_l0: row 0 = dispatch row-block counters (slack-seeded), row 1 =
        # pcie_barrier_all slots. barrier_l1: row 0 = W2 col-block counters,
        # row 1 = local W2 completion signal, rows 2+d = card d's combine watermark.
        self.barrier_l0 = TK((2 + world, bar_cols), dtype=torch.int, local_rank=lr,
                             local_world_size=lws, multicast=False)
        self.barrier_l1 = TK((2 + world, bar_cols), dtype=torch.int, local_rank=lr,
                             local_world_size=lws, multicast=False)
        self.barrier_l0.data_.zero_()
        self.barrier_l1.data_.zero_()
        # seed dispatch counters with padding slack (restored by the kernel's
        # reset-to-slack after each iteration; padding slots are never scattered).
        nblk = P // ROW_BLOCK
        self.barrier_l0.data_[0, :nblk].copy_(self.slack)

        # LOCAL workspaces (TP: peers never touch gathered / expert_out).
        # zeros so never-written padding rows stay clean bf16.
        if self.fp8:
            self.gathered = torch.zeros(P, H, device=device, dtype=torch.float8_e4m3fn)
            # padding 行 scale=0 -> 反量化恒 0, GEMM 对 padding 行为与 bf16 一致
            self.gathered_scales = torch.zeros(P, H // 128, device=device,
                                               dtype=torch.float32)
            if self.l1_fp8:  # docs/42 P3: act 量化缓冲(复用 rowgroup kernel)
                self.act_fp8 = torch.zeros(P, inter, device=device,
                                           dtype=torch.float8_e4m3fn)
                self.act_scales = torch.zeros(P, inter // 128, device=device,
                                              dtype=torch.float32)
            # docs/43 CE 缓冲(kernel 入口需要实参, 常驻分配; CE 关闭时不访问)
            S = world * num_tokens
            self.ag_tokens = torch.zeros(S, H, device=device,
                                         dtype=torch.float8_e4m3fn)
            self.ag_scales = torch.zeros(S, H // 128, device=device,
                                         dtype=torch.float32)
            self.ce_flags = torch.zeros(world, dtype=torch.int32, device=device)
            self.out_planes = torch.zeros(S, H, device=device, dtype=torch.bfloat16)
            self.seq_buf = torch.zeros(1, dtype=torch.int32, device=device)
        else:
            self.gathered = torch.zeros(P, H, device=device, dtype=torch.bfloat16)
        self.gateup_out = torch.zeros(P, 2 * inter, device=device, dtype=torch.bfloat16)
        self.act = torch.zeros(P, inter, device=device, dtype=torch.bfloat16)
        self.expert_out = torch.zeros(P, H, device=device, dtype=torch.bfloat16)
        # peer-writable combine staging: plane d (rows [d*T, d*T+T)) is written
        # only by card d (single writer, no atomics).
        self.combine_staging = TK((world * num_tokens, H), dtype=torch.bfloat16,
                                  local_rank=lr, local_world_size=lws, multicast=False)
        self.combine_staging.data_.zero_()
        # TP-T1 push dispatch staging: peer-writable (world*T, H); plane s is
        # written only by source card s (own plane unused — own shard is read
        # straight from pre_tokens, which also breaks any self-dependency).
        if self.dispatch_mode == "push":
            self.ag_staging = TK((world * num_tokens, H), dtype=torch.bfloat16,
                                 local_rank=lr, local_world_size=lws, multicast=False)
            self.ag_staging.data_.zero_()
        self.combine_out = torch.zeros(num_tokens, H, device=device, dtype=torch.bfloat16)

        # docs/25: pull path e2e improves through 24 comm SMs (2366@16 ->
        # 2290@24, round 5) — TP is GEMM-bound but the L0 dispatch queue AND
        # the L1 dispenser both live on comm blocks; 24 is the measured best
        # so far (32/40 swept next round for the knee).
        # docs/30: with v2 the L0 comm blocks convert to GEMM workers after the
        # AG, so L0's knee may move; L1 gets its own budget (TK_COMM_SMS_L1) —
        # its comm blocks stream pushes but never help the GEMM before it ends,
        # so a SMALLER L1 budget may win (push saturates at ~4 SMs, docs/22).
        self.num_comm_sms = int(os.environ.get("TK_COMM_SMS", "24"))
        self.num_comm_sms_l1 = int(os.environ.get("TK_COMM_SMS_L1",
                                                  str(self.num_comm_sms)))
        self._l0_seq = 0
        self._l1_seq = 0

    def run(self) -> torch.Tensor:
        tk = self.tk
        # T3 fairness: rebuild the schedule tables on GPU inside the timed
        # region (serial pays its routing all_gathers + alignment per run).
        # Table shapes/addresses are fixed (routing invariant per problem);
        # the pure-compute builder is CUDA-graph captured after first use.
        if self.gpu_schedule:
            # docs/30 sched-merge: pack ids + weight bits, ONE all_gather
            # (saves an NCCL launch ~40us; pack copies are ~5us device kernels
            # and stay inside the timed region for fairness).
            self._packed_local[..., 0].copy_(self._topk_ids_local)
            self._packed_local[..., 1].copy_(self._topk_w_bits)
            torch.distributed.all_gather_into_tensor(
                self._packed_all.view(-1), self._packed_local.view(-1))
            if self._sched_graph is None:
                _build_tp_schedules_gpu(self._packed_all,
                                        self.ctx.world_size, self._num_experts,
                                        self.ctx.rank, self._sched_out)  # warmup
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    _build_tp_schedules_gpu(self._packed_all,
                                            self.ctx.world_size, self._num_experts,
                                            self.ctx.rank, self._sched_out)
                self._sched_graph = g
            else:
                self._sched_graph.replay()

        # barrier BEFORE overwriting pre_tokens (no peer still pulling last
        # iter's tokens) and AFTER (my tokens visible before peers pull).
        self._l0_seq += 1
        tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)
        if self.fp8:
            # 源端 1×128 group 量化(docs/41: 单 kernel 版, torch 链 ~80µs → ~15µs)
            tk.rowgroup_quant_fp8(self.problem.hidden_states,
                                  self.pre_tokens.data_, self.pre_scales.data_)
        else:
            self.pre_tokens.data_.copy_(self.problem.hidden_states)
        self._l0_seq += 1
        tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)

        # layer0: AllGather-dedup dispatch ⊕ gate+up GEMM (one launch)
        if self.fp8:
            # docs/39 P2: fp8 AG(token 4KB + scales 128B)⊕ fp8 dispenser GEMM
            # ⊕ GLU epilogue(fp32 上 silu*up 直存 bf16 act); L1 保持 bf16。
            # docs/43: TK_L0_CE=1 时线上字节由 copy engine 拉进本地 ag 缓冲
            # (0 SM), comm 块只做本地 scatter(按分片 flag 放行)。
            if self.l0_ce:
                self.ce_flags.zero_()
                tk.ce_ag_pull(self.pre_tokens, self.pre_scales,
                              self.ag_tokens, self.ag_scales, self.ce_flags)
            self.gemm_next.zero_()
            tk.moe_tp_dispatch_gemm_fp8(
                self.pre_tokens, self.pre_scales, self.ag_tokens,
                self.ag_scales, self.ce_flags, self.gathered,
                self.gathered_scales, self.w_gateup_fp8, self.w1_il_scales,
                self.act, self.padded, self.tp_slots, self.slack,
                self.pull_order, self.blk_expert, self.gemm_next,
                self.barrier_l0, self.num_comm_sms, self.num_padded_total,
                self.num_tokens, self.l0_ce)
        elif self.dispatch_mode == "push":
            # TP-T1 (docs/23): strong-path push + chunk watermarks + resident
            # scatter. seq gates the watermark slots (monotonic, no reset).
            self._l0_seq += 1
            self.l0_push_cnt.zero_()
            tk.moe_tp_dispatch_push_gemm(
                self.pre_tokens, self.ag_staging, self.gathered, self.w_gateup,
                self.gateup_out, self.padded, self.tp_slots, self.slack,
                self.push_order, self.l0_push_cnt, self.barrier_l0,
                self.num_push_sms, max(self.num_comm_sms - self.num_push_sms, 1),
                self.num_padded_total, self.num_tokens, self._l0_seq)
        elif self.l0_mode == "v2":
            # docs/30: dispenser GEMM (comm blocks join after the AG drains) +
            # fused SwiGLU store (GLU) or plain store + torch silu (rollback).
            self.gemm_next.zero_()
            l0_out = self.act if self.l0_glu else self.gateup_out
            l0_w = self.w_gateup_il if self.l0_glu else self.w_gateup
            tk.moe_tp_dispatch_gemm_v2(self.pre_tokens, self.gathered, l0_w,
                                       l0_out, self.padded, self.tp_slots,
                                       self.slack, self.pull_order,
                                       self.blk_expert, self.gemm_next,
                                       self.barrier_l0, self.num_comm_sms,
                                       self.num_padded_total, self.num_tokens,
                                       self.l0_glu)
        else:
            tk.moe_tp_dispatch_gemm(self.pre_tokens, self.gathered, self.w_gateup,
                                    self.gateup_out, self.padded, self.tp_slots,
                                    self.slack, self.pull_order, self.barrier_l0,
                                    self.num_comm_sms, self.num_padded_total,
                                    self.num_tokens)

        # silu(gate) * up on the halves — skipped when the GLU store already
        # produced act inside the L0 GEMM epilogue (docs/30).
        if not self.l0_glu:
            inter = self.inter
            torch.mul(F.silu(self.gateup_out[:, :inter]),
                      self.gateup_out[:, inter:], out=self.act)

        # layer1: W2 GEMM ⊕ local top-k prered ⊕ push (dense ReduceScatter),
        # then the source-side reduce over the world partial planes.
        self._l1_seq += 1
        self.combine_local_cnt.zero_()
        self.job_next.zero_()
        if self.fp8 and self.l1_fp8:
            # docs/42 P3: act 量化(单 kernel)+ fp8 W2 GEMM ⊕ v1 push/排空。
            # docs/43: TK_L1_CE=1 时归约直写本地 out_planes, 线上搬运与
            # watermark 由 ce::rs_push(copy engine)完成, final_red 零改动。
            tk.rowgroup_quant_fp8(self.act, self.act_fp8, self.act_scales)
            if self.l1_ce:
                tk.ce_rs_fence()  # 等上一迭代 CE 读完 out_planes(docs/43)
            self.l1_gemm_next.zero_()
            tk.moe_tp_gemm_prered_push_fp8(
                self.act_fp8, self.act_scales, self.w2_fp8, self.w2_scales,
                self.expert_out, self.out_planes, self.padded,
                self.combine_staging, self.prered_dst, self.tp_slots,
                self.prered_w, self.combine_local_cnt, self.push_expected_l1,
                self.blk_expert, self.l1_gemm_next, self.job_order,
                self.job_next, self.barrier_l1, self.num_comm_sms_l1,
                self.num_padded_total, self.num_tokens, self.num_jobs,
                self._l1_seq, self.l1_ce)
            if self.l1_ce:
                self.seq_buf.fill_(self._l1_seq)
                tk.ce_rs_push(self.out_planes, self.combine_staging,
                              self.barrier_l1, self.seq_buf, self.num_tokens)
        elif self.l1_mode == "v2":
            self.l1_gemm_next.zero_()
            tk.moe_tp_gemm_prered_push_v2(
                self.act, self.w2, self.expert_out, self.padded,
                self.combine_staging, self.prered_dst, self.tp_slots,
                self.prered_w, self.combine_local_cnt, self.push_expected_l1,
                self.blk_expert, self.l1_gemm_next, self.job_next,
                self.barrier_l1, self.num_comm_sms_l1, self.num_padded_total,
                self.num_tokens, self.num_jobs, self._l1_seq)
        else:
            tk.moe_tp_gemm_prered_push(self.act, self.w2, self.expert_out, self.padded,
                                       self.combine_staging, self.prered_dst,
                                       self.tp_slots, self.prered_w,
                                       self.combine_local_cnt, self.push_expected_l1,
                                       self.job_order, self.job_next,
                                       self.barrier_l1, self.num_comm_sms_l1,
                                       self.num_padded_total, self.num_tokens,
                                       self.num_jobs, self._l1_seq)
        tk.moe_final_reduce_push(self.combine_staging, self.final_contrib,
                                 self.recv_from, self.combine_out, self.barrier_l1,
                                 self.num_tokens, self._l1_seq)
        return self.combine_out
