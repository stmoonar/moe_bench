# SPDX-License-Identifier: Apache-2.0
"""Run the standard TP performance matrix in one four-worker process group.

The base workload always comes from the explicit ``--config`` YAML. The run
manifest contributes only per-case precision/token/iteration/environment A/B
overrides. Correctness and stage-attribution tools intentionally remain outside
this suite so a persistent-kernel deadlock cannot hide their diagnostics.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import yaml

from moe_bench.config import MoEBenchConfig, Precision
from moe_bench.distributed import DistributedRunSpec, run_distributed_suite


DEFAULT_TK_ENV = {
    "TK_L0": "v2",
    "TK_L0_GLU": "1",
    "TK_L1": "v1",
    "TK_TP_DISPATCH": "pull",
    "TK_L1_FP8": "1",
    "TK_L0_CE": "0",
    "TK_L1_CE": "0",
    "TK_COMM_SMS": "24",
    "TK_COMM_SMS_L1": "24",
    "TK_GPU_SCHED": "1",
}


def _string_env(overrides: dict[str, Any] | None) -> dict[str, str]:
    env = dict(DEFAULT_TK_ENV)
    env.update({key: str(value) for key, value in (overrides or {}).items()})
    # Historical TK_COMM_SMS sweeps leave L1 unspecified, so L1 follows L0.
    if overrides and "TK_COMM_SMS" in overrides and "TK_COMM_SMS_L1" not in overrides:
        env["TK_COMM_SMS_L1"] = str(overrides["TK_COMM_SMS"])
    return env


def _make_spec(
    base: MoEBenchConfig,
    output_dir: Path,
    case: dict[str, Any],
    *,
    default_precision: str | None = None,
    default_scheme: str | None = None,
    default_tokens: int | None = None,
    default_iters: int | None = None,
    default_verify: bool = False,
) -> DistributedRunSpec:
    name = str(case["name"])
    precision = Precision(case.get("precision", default_precision or base.precision.value))
    scheme = str(case.get("scheme", default_scheme or "tktp"))
    tokens = int(case.get("tokens_per_rank", default_tokens or base.num_tokens[0]))
    bench_iters = int(case.get("bench_iters", default_iters or base.bench_iters))
    verify = bool(case.get("verify", default_verify))
    if "output" not in case:
        raise ValueError(f"Benchmark case {name!r} is missing its output filename")
    output_name = str(case["output"])
    output_path = (output_dir / output_name).resolve()
    if output_path != output_dir and output_dir not in output_path.parents:
        raise ValueError(f"Benchmark case {name!r} output escapes output-dir: {output_name}")
    config = dataclasses.replace(
        base,
        precision=precision,
        num_tokens=[tokens],
        bench_iters=bench_iters,
        verify=verify,
        output_json=str(output_path),
    )
    return DistributedRunSpec(
        name=name,
        config=config,
        scheme_name=scheme,
        env=_string_env(case.get("env")),
    )


def build_specs(
    base: MoEBenchConfig, manifest: dict[str, Any], output_dir: Path
) -> list[DistributedRunSpec]:
    if manifest.get("format_version") != 2:
        raise ValueError("Unsupported suite manifest; expected format_version: 2")
    suite = manifest.get("suite") or {}
    benchmark = suite.get("benchmarks") or {}
    specs = [
        _make_spec(
            base,
            output_dir,
            case,
            default_verify=bool(benchmark.get("verify", False)),
        )
        for case in benchmark.get("cases", [])
    ]

    sweep = suite.get("comm_sms_sweeps") or {}
    for sweep_case in sweep.get("cases", []):
        for value in sweep_case.get("values", []):
            name = str(sweep_case["case_name_template"]).format(value=value)
            env = dict(sweep_case.get("fixed_env") or {})
            env[str(sweep_case["env_key"])] = value
            output = str(sweep_case["output_template"]).format(value=value)
            specs.append(
                _make_spec(
                    base,
                    output_dir,
                    {"name": name, "env": env, "output": output},
                    default_precision=str(sweep.get("precision", "fp8")),
                    default_scheme=str(sweep.get("scheme", "tktp")),
                    default_tokens=int(sweep.get("tokens_per_rank", base.num_tokens[0])),
                    default_iters=int(sweep.get("bench_iters", base.bench_iters)),
                    default_verify=bool(sweep.get("verify", False)),
                )
            )
    if not specs:
        raise ValueError("Run manifest contains no benchmark cases")
    names = [spec.name for spec in specs]
    outputs = [spec.config.output_json for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Run manifest contains duplicate benchmark case names")
    if len(outputs) != len(set(outputs)):
        raise ValueError("Run manifest contains duplicate benchmark output paths")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="MoEBenchConfig base YAML")
    parser.add_argument("--manifest", required=True, help="suite run manifest YAML")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="print resolved cases only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = MoEBenchConfig.from_file(args.config)
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(base, manifest, output_dir)
    print(
        f"[suite] config={Path(args.config).resolve()} cases={len(specs)} "
        f"world_size={base.world_size} warmup={base.warmup_iters}",
        flush=True,
    )
    for index, spec in enumerate(specs, start=1):
        print(
            f"  {index:02d}. {spec.name}: {spec.scheme_name} "
            f"{spec.config.precision.value} T={spec.config.num_tokens[0]} "
            f"iters={spec.config.bench_iters} -> {spec.config.output_json}",
            flush=True,
        )
    if not args.dry_run:
        run_distributed_suite(specs)


if __name__ == "__main__":
    main()
