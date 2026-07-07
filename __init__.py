# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark harness for vLLM's MoE layer.

Provides a config-driven benchmark of the MoE layer with vLLM's naive
fused-MoE as the baseline, supporting both TP and EP sharding, fp8 and
bf16/fp16 precision, a token sweep, and adjustable routing skew.
"""

from .baseline import IMPLEMENTATIONS, MoEImplementation, get_implementation
from .config import (
    Distribution,
    MoEBenchConfig,
    ParallelMode,
    Precision,
    RoutingConfig,
)
from .context import DistContext
from .data import (
    MoEProblem,
    MoEWeights,
    make_golden_problem,
    make_logical_weights,
    make_problem,
    make_weights,
)
from .reference import VerificationResult, reference_moe, verify_output
from .schemes import SCHEMES, DistributedScheme, get_scheme

__all__ = [
    "IMPLEMENTATIONS",
    "MoEImplementation",
    "get_implementation",
    "Distribution",
    "MoEBenchConfig",
    "ParallelMode",
    "Precision",
    "RoutingConfig",
    "MoEProblem",
    "MoEWeights",
    "make_problem",
    "make_weights",
    "make_logical_weights",
    "make_golden_problem",
    "reference_moe",
    "verify_output",
    "VerificationResult",
    "DistContext",
    "DistributedScheme",
    "get_scheme",
    "SCHEMES",
]
