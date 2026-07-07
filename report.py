# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared result reporting for compute-only and distributed benchmark modes."""

from __future__ import annotations

import json
from typing import Any

from .config import MoEBenchConfig, Precision

# Columns rendered when present in a result row, in this order.
_COLUMNS = [
    ("num_tokens", "tokens", "{:>8}"),
    ("latency_us", "latency(us)", "{:>14.2f}"),
    ("latency_min_us", "min(us)", "{:>12.2f}"),
    ("latency_med_us", "med(us)", "{:>12.2f}"),
    ("tflops", "TFLOP/s", "{:>10.2f}"),
    ("local_assignments", "local_assign", "{:>14}"),
    ("timing", "timing", "{:>8}"),
    ("rel_err", "rel_err", "{:>10.2e}"),
    ("verify", "verify", "{:>7}"),
]


def print_report(
    config: MoEBenchConfig,
    impl_name: str,
    results: list[dict[str, Any]],
    comm: str | None = None,
) -> None:
    print()
    print(f"Implementation : {impl_name}")
    print(
        f"Shape          : hidden={config.hidden_size} "
        f"inter={config.intermediate_size} experts={config.num_experts} "
        f"topk={config.topk}"
    )
    parallel = (
        f"{config.parallel_mode.value.upper()} world_size={config.world_size} "
        f"(E_local={config.num_local_experts} "
        f"inter_shard={config.intermediate_shard})"
    )
    if comm:
        parallel += f" comm={comm}"
    print(f"Parallel       : {parallel}")
    print(
        f"Precision      : {config.precision.value}"
        + (f" block={config.block_shape}" if config.precision == Precision.FP8 else "")
    )
    print(f"Routing        : {config.routing.distribution.value}")
    print()

    columns = [c for c in _COLUMNS if c[0] in results[0]] if results else []
    # Column titles can be wider than their data format; align on the max.
    titles = [title for _, title, _ in columns]
    widths = [max(len(t), len(fmt.format(0))) for (_, t, fmt) in columns]
    print(" ".join(t.rjust(w) for t, w in zip(titles, widths)))
    print("-" * (sum(widths) + len(widths) - 1))
    for row in results:
        cells = []
        for (key, _, fmt), w in zip(columns, widths):
            cells.append(fmt.format(row[key]).rjust(w))
        print(" ".join(cells))
    print()


def write_json(
    path: str,
    config: MoEBenchConfig,
    impl_name: str,
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "config": config.to_dict(),
        "implementation": impl_name,
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote results to {path}")
