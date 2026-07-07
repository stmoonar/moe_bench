# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE implementations under benchmark.

The :class:`MoEImplementation` interface is the contract every implementation
(baseline or efficient) must satisfy so they can be swapped in the timing
harness while sharing identical inputs. To add an efficient kernel, subclass
:class:`MoEImplementation`, implement :meth:`run`, and register it in
:data:`IMPLEMENTATIONS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from .config import ParallelMode
from .data import MoEProblem


class MoEImplementation(ABC):
    """Uniform interface for a benchmarked MoE forward.

    ``run`` receives a fully-built :class:`MoEProblem` and must return the
    layer output of shape ``(num_tokens, hidden_size)``. It is called repeatedly
    for timing and may be captured in a CUDA graph, so it must be side-effect
    free and read only from ``problem`` (no dependence on Python state that
    changes between calls). Allocating outputs/workspaces internally is fine —
    the graph's private memory pool owns them during capture — as long as the
    allocation pattern is identical across calls (the ``fused_experts`` baseline
    satisfies this).
    """

    name: str

    def setup(self, problem: MoEProblem) -> None:
        """Optional one-time setup (weight reshaping, kernel selection)."""

    @abstractmethod
    def run(self, problem: MoEProblem) -> torch.Tensor:
        raise NotImplementedError


class NaiveFusedExperts(MoEImplementation):
    """vLLM's default (Triton) fused-MoE, used as the baseline.

    This is the same ``fused_experts`` path the unquantized/native MoE method
    dispatches to. It handles both parallel modes uniformly:

    - **TP**: all ``E`` experts are local; ``expert_map`` is ``None`` and every
      token-expert assignment is computed.
    - **EP**: only ``E/world_size`` experts are local; ``expert_map`` masks out
      global experts that don't live on this rank so only the local share of
      the routed tokens is computed (mirrors post-dispatch compute).
    """

    name = "naive"

    def run(self, problem: MoEProblem) -> torch.Tensor:
        cfg = problem.config
        expert_map = (
            problem.expert_map if cfg.parallel_mode == ParallelMode.EP else None
        )
        return fused_experts(
            hidden_states=problem.hidden_states,
            w1=problem.w1,
            w2=problem.w2,
            topk_weights=problem.topk_weights,
            topk_ids=problem.topk_ids,
            global_num_experts=problem.global_num_experts,
            expert_map=expert_map,
            quant_config=problem.quant_config,
        )


# Registry of available implementations, keyed by ``--impl`` name.
IMPLEMENTATIONS: dict[str, type[MoEImplementation]] = {
    NaiveFusedExperts.name: NaiveFusedExperts,
}


def get_implementation(name: str) -> MoEImplementation:
    if name not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation '{name}'. "
            f"Available: {sorted(IMPLEMENTATIONS)}"
        )
    return IMPLEMENTATIONS[name]()
