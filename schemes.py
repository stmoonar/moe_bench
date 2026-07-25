# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-layer MoE schemes (compute **and** communication) for distributed mode.

A *scheme* is one rank's implementation of the whole MoE layer — dispatch,
expert compute, and combine — so different comm/compute strategies (serial,
dual-stream overlap, a fused layer op) are directly comparable under one timing
harness. This is a different abstraction from the compute-only
:class:`~moe_bench.baseline.MoEImplementation` (which times just the local expert
GEMMs); a scheme owns the collectives too and returns *this rank's* final output.

The I/O contract is uniform across precisions, taken straight off the shared
:class:`~moe_bench.data.MoEProblem`:

- **input**  — ``problem.hidden_states``: this rank's ``(num_tokens, hidden)``
  token shard, bf16. fp8 schemes quantize activations internally.
- **weights** — ``problem.w1`` / ``problem.w2`` (bf16, or fp8 when
  ``problem.quant_config`` is set) plus ``problem.expert_map`` (EP only).
- **output** — ``(num_tokens, hidden)`` bf16, this rank's tokens after combine.

To add your own (e.g. the overlap or fused-layer variant), subclass
:class:`DistributedScheme`, implement ``setup``/``run``, and register it in
:data:`SCHEMES`. See ``ADDING_IMPLEMENTATIONS.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from .config import ParallelMode
from .context import DistContext
from .data import MoEProblem


class DistributedScheme(ABC):
    """One rank's full MoE layer including cross-rank communication.

    ``setup`` runs once (allocate comm buffers, prep weights/streams) and is not
    timed. ``run`` executes the full layer and returns this rank's output; it is
    called repeatedly in the timing loop, so — like the compute-only baseline —
    it must be side-effect free and allocation-stable (reuse the buffers stashed
    in ``setup``).
    """

    name: str

    def setup(self, problem: MoEProblem, ctx: DistContext) -> None:
        """Optional one-time setup. Not timed."""

    @abstractmethod
    def run(self) -> torch.Tensor:
        """Full layer on this rank; returns ``(num_tokens, hidden)``."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional teardown (free streams/buffers)."""


class SerialNaive(DistributedScheme):
    """Serial baseline: communication and compute do **not** overlap.

    The reference dataflow every overlap/fused scheme is measured against:

    - **EP**: ``AllGather`` the token shard *and* its routing from every rank,
      run ``fused_experts`` with ``expert_map`` (non-local experts masked), then
      ``ReduceScatter`` the per-token results back to each rank.
    - **TP**: ``AllGather`` the token shard, run ``fused_experts`` with the
      intermediate-sharded weights (each rank produces a partial down-projection
      sum), then ``ReduceScatter`` — which sums the partials across ranks and
      scatters each rank its tokens.

    Uses plain ``torch.distributed`` collectives and vLLM's ``fused_experts`` —
    no modular-kernel machinery — so the serial structure is explicit and an
    overlap variant is a direct diff against it.
    """

    name = "serial"

    def setup(self, problem: MoEProblem, ctx: DistContext) -> None:
        cfg = problem.config
        self.ctx = ctx
        self.problem = problem
        self.is_ep = cfg.parallel_mode == ParallelMode.EP
        self.global_num_experts = cfg.num_experts
        self.quant_config = problem.quant_config
        self.w1 = problem.w1
        self.w2 = problem.w2
        self.expert_map = problem.expert_map if self.is_ep else None

        m_local = problem.num_tokens
        m_full = m_local * ctx.world_size
        k = cfg.hidden_size
        topk = cfg.topk
        dtype = cfg.torch_dtype
        device = problem.hidden_states.device

        # Preallocated, allocation-stable buffers reused every timed call.
        self.hidden_local = problem.hidden_states.contiguous()
        self.hidden_full = torch.empty(m_full, k, device=device, dtype=dtype)
        self.output = torch.empty(m_local, k, device=device, dtype=dtype)
        # Routing is gathered for both modes so the per-token order matches
        # hidden_full (rank-major concatenation).
        self.topk_ids_local = problem.topk_ids.contiguous()
        self.topk_weights_local = problem.topk_weights.contiguous()
        self.topk_ids_full = torch.empty(
            m_full, topk, device=device, dtype=self.topk_ids_local.dtype
        )
        self.topk_weights_full = torch.empty(
            m_full, topk, device=device, dtype=self.topk_weights_local.dtype
        )

    def run(self) -> torch.Tensor:
        group = self.ctx.group

        # --- dispatch: AllGather token shard (+ routing) into the full batch ---
        dist.all_gather_into_tensor(self.hidden_full, self.hidden_local, group=group)
        dist.all_gather_into_tensor(
            self.topk_ids_full, self.topk_ids_local, group=group
        )
        dist.all_gather_into_tensor(
            self.topk_weights_full, self.topk_weights_local, group=group
        )

        # --- expert compute: gate/up GEMM -> SiLU -> down GEMM ---
        result = fused_experts(
            hidden_states=self.hidden_full,
            w1=self.w1,
            w2=self.w2,
            topk_weights=self.topk_weights_full,
            topk_ids=self.topk_ids_full,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            quant_config=self.quant_config,
        )

        # --- combine: ReduceScatter per-token results back to each rank ---
        dist.reduce_scatter_tensor(self.output, result, group=group)
        return self.output


# Registry of distributed schemes, keyed by ``--scheme`` name.
SCHEMES: dict[str, type[DistributedScheme]] = {
    SerialNaive.name: SerialNaive,
}


def _register_optional_schemes() -> None:
    """Register schemes with heavy/optional deps (TK extension) lazily, so a
    missing build doesn't break the serial baseline import."""
    import sys as _sys
    import traceback as _tb
    try:
        from .tk_tp_scheme import TKFusedTP
        SCHEMES[TKFusedTP.name] = TKFusedTP
    except Exception:
        print("[schemes] tktp unavailable:", file=_sys.stderr)
        _tb.print_exc()


_register_optional_schemes()


def get_scheme(name: str) -> DistributedScheme:
    if name not in SCHEMES:
        raise ValueError(
            f"Unknown scheme '{name}'. Available: {sorted(SCHEMES)}"
        )
    return SCHEMES[name]()
