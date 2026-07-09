# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration schema for the MoE layer benchmark.

Every knob that affects the benchmarked problem lives here so that a single
config file fully determines the workload. The defaults match DeepSeek V3.2.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ParallelMode(str, Enum):
    """How the experts/weights are sharded across a hypothetical device group.

    The benchmark runs on a single device; the mode only changes the *local*
    weight shapes and routing so the measured kernel matches what one rank of a
    real deployment would execute.

    - ``TP`` (tensor parallel): every expert lives on the rank, but each
      expert's intermediate dimension is sharded by ``world_size``.
    - ``EP`` (expert parallel): each expert is full-size, but only
      ``num_experts / world_size`` experts live on the rank. Tokens are routed
      over the *global* expert set and non-local experts are masked out via an
      ``expert_map`` (mirrors the local compute after dispatch).
    """

    TP = "tp"
    EP = "ep"


class Distribution(str, Enum):
    """Token-to-expert routing distribution (controls load balance)."""

    # Perfectly balanced: every expert receives (near) equal token load.
    BALANCED = "balanced"
    # Each token picks ``topk`` distinct experts uniformly at random.
    UNIFORM = "uniform"
    # Zipf-like skew controlled by ``skew_alpha`` (a few hot experts).
    SKEWED = "skewed"
    # Extreme imbalance: every token routes to experts [0, topk).
    SINGLE = "single"


class Precision(str, Enum):
    """Activation/weight precision for the benchmarked GEMMs."""

    BF16 = "bf16"
    FP16 = "fp16"
    # Per-block fp8 (w8a8) with a ``block_shape`` scale grid (DeepSeek style).
    FP8 = "fp8"


@dataclass
class RoutingConfig:
    """Routing / load-balance configuration."""

    distribution: Distribution = Distribution.BALANCED
    # Zipf exponent for ``SKEWED`` (larger -> more concentrated).
    skew_alpha: float = 1.0
    # Restrict routing to the first ``num_active_experts`` experts. ``None``
    # means all experts are eligible. Combine with ``BALANCED`` to spread load
    # over an arbitrarily small subset without going fully ``SINGLE``.
    num_active_experts: int | None = None


@dataclass
class MoEBenchConfig:
    """Top-level benchmark configuration.

    Shapes follow vLLM's fused-MoE convention:
      - ``w1``/``w13``: (E_local, 2 * intermediate_shard, hidden_size)
      - ``w2``:         (E_local, hidden_size, intermediate_shard)
    where ``E_local`` and ``intermediate_shard`` are derived from the parallel
    mode and ``world_size`` (see :mod:`moe_bench.data`).
    """

    # ---- Model / weight shape (defaults: E=64, hidden=4096, gate_up=6144) ----
    hidden_size: int = 4096
    intermediate_size: int = 3072  # gate_up = 2 * intermediate = 6144
    num_experts: int = 64
    topk: int = 8

    # ---- Parallelism ----
    parallel_mode: ParallelMode = ParallelMode.TP
    world_size: int = 8

    # ---- Precision ----
    precision: Precision = Precision.BF16
    # fp8 block-quant scale grid (only used when precision == FP8).
    block_shape: list[int] = field(default_factory=lambda: [128, 128])

    # ---- Workload sweep ----
    # Number of tokens (rows) fed to the layer for each measured point.
    num_tokens: list[int] = field(
        default_factory=lambda: [1, 4, 16, 64, 256, 1024, 4096]
    )

    # ---- Routing ----
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    # ---- Distributed (real cross-rank communication) ----
    # When True, the benchmark spawns ``world_size`` processes (one GPU each)
    # and measures the *full* MoE layer including communication:
    #   - TP: expert compute + output all-reduce across the TP group.
    #   - EP: naive all-gather dispatch + expert compute + reduce-scatter combine
    #         across the EP group (vLLM's naive_dp_ep prepare/finalize).
    # When False, a single process measures only the local expert compute.
    distributed: bool = False
    # EP dispatch/combine backend. "allgather_reducescatter" is vLLM's naive
    # baseline (AllGather dispatch + ReduceScatter combine). Other backends
    # (deepep_high_throughput, ...) require the corresponding libs/hardware.
    all2all_backend: str = "allgather_reducescatter"

    # ---- Timing ----
    warmup_iters: int = 10
    bench_iters: int = 50
    # Capture the op in a CUDA graph before timing (removes launch overhead,
    # matching how MoE runs under vLLM's graph capture). Disabled automatically
    # in distributed mode (graph capture across NCCL collectives is fragile).
    use_cuda_graph: bool = True
    seed: int = 0

    # ---- Correctness ----
    # Cross-check each implementation's output against a pure-torch reference
    # (computed before timing) and report a pass/fail per token count. The
    # reference is derived only from the shared problem inputs, so it validates
    # the baseline and any efficient implementation alike.
    verify: bool = True
    # Elementwise tolerances for the torch.testing.assert_close check. None uses
    # a per-precision default (see moe_bench.reference); set a float to override.
    verify_atol: float | None = None
    verify_rtol: float | None = None

    # ---- Output ----
    device: str = "cuda"
    output_json: str | None = None

    # --- derived / validation ---
    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "topk",
            "world_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        # vLLM removed the "naive" backend name; its remap to the canonical
        # name runs only at ParallelConfig construction, which the distributed
        # harness bypasses by assigning the field afterward. Normalize here.
        if self.all2all_backend == "naive":
            self.all2all_backend = "allgather_reducescatter"
        if self.topk > self.num_experts:
            raise ValueError(
                f"topk={self.topk} > num_experts={self.num_experts}"
            )

        # Routing must leave enough distinct experts to pick topk from.
        active = self.routing.num_active_experts
        if active is not None and not (1 <= active <= self.num_experts):
            raise ValueError(
                f"num_active_experts={active} out of range "
                f"[1, num_experts={self.num_experts}]"
            )
        effective_active = active if active is not None else self.num_experts
        if self.topk > effective_active:
            raise ValueError(
                f"topk={self.topk} > num_active_experts={effective_active}; "
                "cannot route to that many distinct experts"
            )

        if self.parallel_mode == ParallelMode.TP:
            if self.intermediate_size % self.world_size != 0:
                raise ValueError(
                    f"intermediate_size={self.intermediate_size} not divisible "
                    f"by world_size={self.world_size} for TP mode"
                )
        else:  # EP
            if self.num_experts % self.world_size != 0:
                raise ValueError(
                    f"num_experts={self.num_experts} not divisible by "
                    f"world_size={self.world_size} for EP mode"
                )
        if self.precision == Precision.FP8 and len(self.block_shape) != 2:
            raise ValueError("block_shape must be [block_n, block_k] for fp8")

    @property
    def num_local_experts(self) -> int:
        if self.parallel_mode == ParallelMode.EP:
            return self.num_experts // self.world_size
        return self.num_experts

    @property
    def intermediate_shard(self) -> int:
        if self.parallel_mode == ParallelMode.TP:
            return self.intermediate_size // self.world_size
        return self.intermediate_size

    @property
    def torch_dtype(self):
        import torch

        return {
            Precision.BF16: torch.bfloat16,
            Precision.FP16: torch.float16,
            # fp8 uses a bf16 "base" dtype for activations/dequant.
            Precision.FP8: torch.bfloat16,
        }[self.precision]

    # --- (de)serialization ---
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoEBenchConfig:
        data = dict(data)
        if "parallel_mode" in data:
            data["parallel_mode"] = ParallelMode(data["parallel_mode"])
        if "precision" in data:
            data["precision"] = Precision(data["precision"])
        if "routing" in data and isinstance(data["routing"], dict):
            r = dict(data["routing"])
            if "distribution" in r:
                r["distribution"] = Distribution(r["distribution"])
            data["routing"] = RoutingConfig(**r)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> MoEBenchConfig:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["parallel_mode"] = self.parallel_mode.value
        d["precision"] = self.precision.value
        d["routing"]["distribution"] = self.routing.distribution.value
        return d
