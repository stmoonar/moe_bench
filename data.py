# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input, weight and routing generation for the MoE benchmark.

Everything a MoE implementation needs to run one problem instance is bundled in
:class:`MoEProblem`. Generation is deterministic given ``config.seed`` so the
naive baseline and any efficient implementation see identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

from .config import Distribution, MoEBenchConfig, ParallelMode, Precision

FP8_DTYPE = current_platform.fp8_dtype()


@dataclass
class MoEProblem:
    """A fully-materialized MoE problem instance (device tensors)."""

    config: MoEBenchConfig
    num_tokens: int

    # Activations: (num_tokens, hidden_size) in the base dtype.
    hidden_states: torch.Tensor

    # Expert weights (already quantized when precision == FP8):
    #   w1: (E_local, 2 * intermediate_shard, hidden_size)
    #   w2: (E_local, hidden_size, intermediate_shard)
    w1: torch.Tensor
    w2: torch.Tensor

    # Routing (over the *global* expert set):
    #   topk_ids:     (num_tokens, topk) int32 global expert ids
    #   topk_weights: (num_tokens, topk) float32 combine weights
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    # EP only: maps a global expert id -> local slot (or -1 if not on rank).
    expert_map: torch.Tensor | None

    # Quantization metadata for fused_experts (None for bf16/fp16).
    quant_config: FusedMoEQuantConfig | None

    @property
    def global_num_experts(self) -> int:
        return self.config.num_experts

    @property
    def num_local_assignments(self) -> int:
        """Token-expert assignments computed on this rank's weights."""
        if self.expert_map is None:  # TP: all assignments are local.
            return self.topk_ids.numel()
        return int((self.expert_map[self.topk_ids.long()] != -1).sum().item())


@dataclass
class MoEWeights:
    """Per-rank expert weights, independent of the token count.

    Build once per (config, rank) and reuse across the token sweep — weight
    generation (and fp8 re-quantization) is the expensive part of problem
    setup.
    """

    w1: torch.Tensor
    w2: torch.Tensor
    quant_config: FusedMoEQuantConfig | None


def _mix_seed(*parts: int) -> int:
    """Deterministically mix seed components into one 31-bit seed.

    Polynomial hash with a large odd multiplier; distinct component tuples
    give (practically) uncorrelated generator streams.
    """
    h = 0
    for p in parts:
        h = (h * 1_000_003 + p) & 0x7FFF_FFFF
    return h


# Stream tags keep input and weight generators disjoint even when the
# remaining seed components collide.
_STREAM_INPUT = 0
_STREAM_WEIGHT = 1


def _gen_expert_chunk(
    config: MoEBenchConfig,
    global_e: int,
    shard: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One expert's bf16 weight chunk, seeded by ``(seed, global_e, shard)``.

    The single source of truth for weight values: :func:`make_weights` calls it
    per local expert, and :func:`make_logical_weights` reassembles the full
    (unsharded) weights from the same chunks for the golden reference — so the
    sharded run and its reference are byte-consistent by construction.

    Shapes are the *shard* shapes (``n = intermediate_shard``): TP splits the
    intermediate dim across shards, EP keeps it whole (a single shard 0).
    """
    device = config.device
    dtype = config.torch_dtype
    k = config.hidden_size
    n = config.intermediate_shard
    generator.manual_seed(_mix_seed(config.seed, _STREAM_WEIGHT, global_e, shard))
    # Small magnitude keeps fp8/bf16 GEMMs numerically sane.
    w1_e = torch.randn(2 * n, k, device=device, dtype=dtype, generator=generator) / (
        k**0.5
    )
    w2_e = torch.randn(k, n, device=device, dtype=dtype, generator=generator) / (
        n**0.5
    )
    return w1_e, w2_e


def make_weights(config: MoEBenchConfig, rank: int = 0) -> MoEWeights:
    """Create ``rank``'s expert weights (and fp8 block-quant scales).

    Each weight chunk is seeded by ``(seed, global_expert_id, shard)`` — not by
    rank — so the sharded weights are consistent slices of one well-defined
    logical model:

    - EP: expert ``e`` has identical weights whoever owns it (shard 0), so an
      EP run is comparable against a single-GPU reference over all experts.
    - TP: rank ``r`` holds chunk ``r`` of every expert; the logical expert
      weight is the concatenation of the per-shard chunks along the
      intermediate dim, so the TP all-reduce output is a reproducible forward
      of that logical model.
    """
    device = config.device
    e_local = config.num_local_experts
    is_tp = config.parallel_mode == ParallelMode.TP
    quantize = config.precision == Precision.FP8

    if quantize:
        from vllm.utils.deep_gemm import per_block_cast_to_fp8

    generator = torch.Generator(device=device)
    w1_list, w2_list = [], []
    w1_scales, w2_scales = [], []
    for i in range(e_local):
        global_e = i if is_tp else rank * e_local + i
        shard = rank if is_tp else 0
        w1_e, w2_e = _gen_expert_chunk(config, global_e, shard, generator)
        if quantize:
            w1_e, s1 = per_block_cast_to_fp8(w1_e, config.block_shape)
            w2_e, s2 = per_block_cast_to_fp8(w2_e, config.block_shape)
            w1_scales.append(s1)
            w2_scales.append(s2)
        w1_list.append(w1_e)
        w2_list.append(w2_e)

    w1 = torch.stack(w1_list)
    w2 = torch.stack(w2_list)
    if not quantize:
        return MoEWeights(w1=w1, w2=w2, quant_config=None)

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=FP8_DTYPE,
        w1_scale=torch.stack(w1_scales),
        w2_scale=torch.stack(w2_scales),
        block_shape=config.block_shape,
    )
    return MoEWeights(w1=w1, w2=w2, quant_config=quant_config)


def _make_routing(
    config: MoEBenchConfig,
    num_tokens: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``topk_ids`` / ``topk_weights`` with the configured skew.

    ``topk_ids`` are global expert ids; each token gets ``topk`` *distinct*
    experts. Returns int32 ids and float32 weights.
    """
    device = config.device
    e = config.num_experts
    topk = config.topk
    routing = config.routing
    dist = routing.distribution

    # Bounds are validated in MoEBenchConfig.__post_init__ (topk <= active).
    active = routing.num_active_experts
    active = e if active is None else min(active, e)

    if dist == Distribution.SINGLE:
        # Every token routes to experts [0, topk): maximal load imbalance.
        # num_active_experts is irrelevant here by design.
        topk_ids = (
            torch.arange(topk, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(num_tokens, topk)
            .contiguous()
        )
    elif dist == Distribution.BALANCED:
        # Round-robin assignment so every (active) expert gets equal load.
        # For row t the ids are (t*topk + [0..topk)) % active shifted by a
        # per-row offset. Since topk <= active these are distinct within a row,
        # and adding a constant (mod active) preserves distinctness.
        base = torch.arange(num_tokens * topk, device=device, dtype=torch.int64)
        ids = (base % active).view(num_tokens, topk)
        offs = (torch.arange(num_tokens, device=device) % active).view(-1, 1)
        topk_ids = ((ids + offs) % active).to(torch.int32)
    else:
        # UNIFORM or SKEWED: sample distinct experts per token from a prob vec.
        if dist == Distribution.SKEWED:
            ranks = torch.arange(1, active + 1, device=device, dtype=torch.float32)
            probs = ranks ** (-routing.skew_alpha)
        else:  # UNIFORM
            probs = torch.ones(active, device=device, dtype=torch.float32)
        probs = probs / probs.sum()
        probs = probs.unsqueeze(0).expand(num_tokens, active)
        # Sample topk distinct experts per token without replacement.
        topk_ids = torch.multinomial(
            probs, topk, replacement=False, generator=generator
        ).to(torch.int32)

    # Combine weights: random logits over the chosen experts, softmax-normed.
    logits = torch.rand(
        num_tokens, topk, device=device, dtype=torch.float32, generator=generator
    )
    topk_weights = torch.softmax(logits, dim=-1)
    return topk_ids, topk_weights


def _make_expert_map(config: MoEBenchConfig, rank: int) -> torch.Tensor | None:
    """EP: map global expert id -> local slot for ``rank``'s expert block.

    Rank ``r`` owns experts ``[r*local, (r+1)*local)``, which map to local
    slots ``[0, local)``; every other global id maps to ``-1`` (not on rank).
    """
    if config.parallel_mode != ParallelMode.EP:
        return None
    e = config.num_experts
    local = config.num_local_experts
    start = rank * local
    expert_map = torch.full((e,), -1, device=config.device, dtype=torch.int32)
    expert_map[start : start + local] = torch.arange(
        local, device=config.device, dtype=torch.int32
    )
    return expert_map


def _input_rank(config: MoEBenchConfig, rank: int) -> int:
    """Seed component for a rank's token shard.

    In distributed mode (and always in EP) each rank owns a *distinct* token
    batch that is gathered into the full batch, so the seed includes ``rank``.
    In single-process TP the input is rank-independent.
    """
    if config.distributed or config.parallel_mode == ParallelMode.EP:
        return rank
    return 0


def _gen_input(
    config: MoEBenchConfig, num_tokens: int, input_rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic ``(hidden_states, topk_ids, topk_weights)`` for one shard.

    Shared by :func:`make_problem` (the benchmarked shard) and
    :func:`make_golden_problem` (the reference), so a rank's tokens are identical
    whether generated for timing or for verification. Draw order (hidden, then
    routing) must stay fixed for that reproducibility.
    """
    generator = torch.Generator(device=config.device)
    generator.manual_seed(_mix_seed(config.seed, _STREAM_INPUT, num_tokens, input_rank))
    hidden_states = torch.randn(
        num_tokens,
        config.hidden_size,
        device=config.device,
        dtype=config.torch_dtype,
        generator=generator,
    )
    topk_ids, topk_weights = _make_routing(config, num_tokens, generator)
    return hidden_states, topk_ids, topk_weights


def make_problem(
    config: MoEBenchConfig,
    num_tokens: int,
    rank: int = 0,
    weights: MoEWeights | None = None,
) -> MoEProblem:
    """Build one :class:`MoEProblem` of ``num_tokens`` tokens on ``rank``.

    Pass a prebuilt ``weights`` (from :func:`make_weights`) when sweeping token
    counts — weights don't depend on ``num_tokens`` and rebuilding them (plus
    fp8 re-quantization) per sweep point is the dominant setup cost.
    """
    hidden_states, topk_ids, topk_weights = _gen_input(
        config, num_tokens, _input_rank(config, rank)
    )

    if weights is None:
        weights = make_weights(config, rank)
    expert_map = _make_expert_map(config, rank)

    return MoEProblem(
        config=config,
        num_tokens=num_tokens,
        hidden_states=hidden_states,
        w1=weights.w1,
        w2=weights.w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        expert_map=expert_map,
        quant_config=weights.quant_config,
    )


def make_logical_weights(config: MoEBenchConfig) -> MoEWeights:
    """Full (unsharded) weights for *every* global expert.

    Reassembles the logical model the sharded run reconstructs, from the same
    per-``(expert, shard)`` chunks :func:`make_weights` uses — so it is the
    ground-truth weight set for the distributed golden reference:

    - **EP**: each expert is full-intermediate already (shard 0).
    - **TP**: an expert's full weight is the concatenation of its per-shard
      chunks along the intermediate dim (gate/up halves concatenated
      separately, matching the fused ``w1`` layout).

    fp8 weights are block-quantized here the same way; when ``intermediate_shard``
    is a multiple of ``block_shape`` the per-shard and concatenated quantizations
    agree, so the reference stays consistent with the sharded run.
    """
    is_tp = config.parallel_mode == ParallelMode.TP
    shards = range(config.world_size) if is_tp else (0,)
    quantize = config.precision == Precision.FP8
    if quantize:
        from vllm.utils.deep_gemm import per_block_cast_to_fp8

    generator = torch.Generator(device=config.device)
    w1_list, w2_list, w1_scales, w2_scales = [], [], [], []
    for e in range(config.num_experts):
        gates, ups, w2_chunks = [], [], []
        for shard in shards:
            w1_c, w2_c = _gen_expert_chunk(config, e, shard, generator)
            gate, up = w1_c.chunk(2, dim=0)  # (n, K) each
            gates.append(gate)
            ups.append(up)
            w2_chunks.append(w2_c)  # (K, n)
        w1_e = torch.cat(gates + ups, dim=0)  # (2 * intermediate, K)
        w2_e = torch.cat(w2_chunks, dim=1)  # (K, intermediate)
        if quantize:
            w1_e, s1 = per_block_cast_to_fp8(w1_e, config.block_shape)
            w2_e, s2 = per_block_cast_to_fp8(w2_e, config.block_shape)
            w1_scales.append(s1)
            w2_scales.append(s2)
        w1_list.append(w1_e)
        w2_list.append(w2_e)

    w1 = torch.stack(w1_list)
    w2 = torch.stack(w2_list)
    if not quantize:
        return MoEWeights(w1=w1, w2=w2, quant_config=None)
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=FP8_DTYPE,
        w1_scale=torch.stack(w1_scales),
        w2_scale=torch.stack(w2_scales),
        block_shape=config.block_shape,
    )
    return MoEWeights(w1=w1, w2=w2, quant_config=quant_config)


def make_golden_problem(
    config: MoEBenchConfig,
    num_tokens: int,
    rank: int,
    weights: MoEWeights | None = None,
) -> MoEProblem:
    """Rank ``rank``'s own tokens through the full logical model (all experts).

    The distributed ground truth for that rank: EP's reduce-scatter returns this
    rank's tokens combined over *all* experts, and TP's reduce-scatter scatters
    this rank's tokens after summing the intermediate shards — both equal a full
    MoE over the rank's own token shard. So :func:`reference_moe` on this problem
    is exactly the expected output (all experts local, no ``expert_map``), and it
    costs only ``num_tokens`` rows rather than the whole gathered batch.

    Pass a prebuilt ``weights`` (from :func:`make_logical_weights`) to reuse the
    full weight set across the token sweep.
    """
    if weights is None:
        weights = make_logical_weights(config)
    hidden_states, topk_ids, topk_weights = _gen_input(config, num_tokens, rank)
    return MoEProblem(
        config=config,
        num_tokens=num_tokens,
        hidden_states=hidden_states,
        w1=weights.w1,
        w2=weights.w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        expert_map=None,
        quant_config=weights.quant_config,
    )
