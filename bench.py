# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE benchmark entry point.

Times a MoE implementation across a token sweep and reports per-point latency
and achieved compute throughput. Baseline is vLLM's naive fused-MoE; efficient
implementations register in :mod:`moe_bench.baseline` and are selected with
``--impl``.

Usage:
    python -m moe_bench.bench --config moe_bench/configs/deepseek_v32.yaml
    python -m moe_bench.bench --config <cfg> --mode ep --precision fp8
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any

import torch

from .baseline import MoEImplementation, get_implementation
from .config import (
    Distribution,
    MoEBenchConfig,
    ParallelMode,
    Precision,
)
from .data import MoEProblem, make_problem, make_weights
from .reference import reference_moe, verify_output
from .report import print_report, write_json
from .routing_stats import expert_token_counts, print_routing_stats

# Invocations captured per CUDA graph. Replaying a bundle and dividing
# amortizes the fixed graph-launch overhead out of the per-op number
# (mirrors benchmarks/kernels/benchmark_moe.py, which captures 10).
GRAPH_CAPTURE_BATCH = 10


def _tflops(problem: MoEProblem, latency_s: float) -> float:
    """Achieved TFLOP/s for the two per-expert GEMMs (gate/up + down).

    Per computed assignment: gemm1 is (2N x K), gemm2 is (K x N), each a
    multiply-add (2 flops), giving 6*N*K flops.
    """
    n = problem.config.intermediate_shard
    k = problem.config.hidden_size
    flops = 6.0 * n * k * problem.num_local_assignments
    return flops / latency_s / 1e12


def _time_run(
    impl: MoEImplementation,
    problem: MoEProblem,
    config: MoEBenchConfig,
) -> tuple[float, str]:
    """Time ``impl.run(problem)``; returns (seconds per call, timing method)."""

    def run():
        return impl.run(problem)

    # Warmup + JIT compilation.
    for _ in range(config.warmup_iters):
        run()
    torch.cuda.synchronize()

    graph = None
    calls_per_iter = 1
    if config.use_cuda_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(GRAPH_CAPTURE_BATCH):
                    run()
            calls_per_iter = GRAPH_CAPTURE_BATCH
            # Warm the instantiated graph: the first replays after capture
            # are slower and must not land in the timed loop.
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] CUDA graph capture failed ({e}); using eager timing")
            graph = None
            calls_per_iter = 1

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(config.bench_iters):
        if graph is not None:
            graph.replay()
        else:
            run()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    if graph is not None:
        graph.reset()
    latency_s = (total_ms / (config.bench_iters * calls_per_iter)) / 1e3
    return latency_s, "graph" if graph is not None else "eager"


def run_sweep(
    config: MoEBenchConfig, impl: MoEImplementation
) -> list[dict[str, Any]]:
    # Weights don't depend on the token count; build them once for the sweep.
    weights = make_weights(config, rank=0)
    results: list[dict[str, Any]] = []
    any_verify_failed = False
    for num_tokens in config.num_tokens:
        problem = make_problem(config, num_tokens, rank=0, weights=weights)
        impl.setup(problem)

        # 路由分布 + BLOCK 布局(计时区外)。单卡 compute-only 口径, counts 就是
        # 这份 token shard 自己的。
        print_routing_stats(
            expert_token_counts(
                problem.topk_ids, config.num_experts, global_counts=False
            ),
            impl.block_config(problem),
            f"{impl.name}, E={config.num_experts}, topk={config.topk}, "
            f"T={num_tokens}, dist={config.routing.distribution.value}",
        )

        # Compute the ground-truth output once, up front, before timing — so the
        # reference reflects the freshly generated inputs and the timed loop is
        # untouched by verification work.
        reference = reference_moe(problem) if config.verify else None

        latency_s, timing = _time_run(impl, problem, config)

        row: dict[str, Any] = {
            "num_tokens": num_tokens,
            "latency_us": latency_s * 1e6,
            "tflops": _tflops(problem, latency_s),
            "local_assignments": problem.num_local_assignments,
            "timing": timing,
        }
        if reference is not None:
            # One extra untimed forward to capture the output actually produced.
            check = verify_output(
                problem,
                impl.run(problem),
                reference,
                config.verify_atol,
                config.verify_rtol,
            )
            row["rel_err"] = check.rel_err
            row["verify"] = "ok" if check.passed else "FAIL"
            if not check.passed:
                print(f"  [verify] num_tokens={num_tokens}: {check.summary()}")
            any_verify_failed |= not check.passed

        results.append(row)
        del problem, reference
        torch.cuda.empty_cache()

    if any_verify_failed:
        print(
            "  [warn] correctness check FAILED for one or more token counts; "
            "the timed implementation does not match the torch reference."
        )
    return results


def _apply_overrides(
    config: MoEBenchConfig, args: argparse.Namespace
) -> MoEBenchConfig:
    changed = {}
    if args.mode is not None:
        changed["parallel_mode"] = ParallelMode(args.mode)
    if args.precision is not None:
        changed["precision"] = Precision(args.precision)
    if args.world_size is not None:
        changed["world_size"] = args.world_size
    if args.num_tokens is not None:
        changed["num_tokens"] = args.num_tokens
    if args.distribution is not None:
        routing = dataclasses.replace(
            config.routing, distribution=Distribution(args.distribution)
        )
        changed["routing"] = routing
    if args.seed is not None:
        changed["seed"] = args.seed
    if args.no_cuda_graph:
        changed["use_cuda_graph"] = False
    if args.no_verify:
        changed["verify"] = False
    if args.verify_atol is not None:
        changed["verify_atol"] = args.verify_atol
    if args.verify_rtol is not None:
        changed["verify_rtol"] = args.verify_rtol
    if args.distributed:
        changed["distributed"] = True
    if args.all2all_backend is not None:
        changed["all2all_backend"] = args.all2all_backend
    if args.output_json is not None:
        changed["output_json"] = args.output_json
    if not changed:
        return config
    return dataclasses.replace(config, **changed)


def main() -> None:
    # NOTE: deliberately NOT using vLLM's FlexibleArgumentParser. Its
    # _pull_args_from_config() reads the --config YAML and injects every key
    # as a CLI arg (verify: true -> --verify, distributed: true -> --distributed,
    # …), which collides with our own MoEBenchConfig.from_file() loading and
    # makes --verify ambiguous with --verify-atol / --verify-rtol. We load the
    # YAML ourselves below, so a plain ArgumentParser is sufficient.
    parser = argparse.ArgumentParser(description="vLLM MoE layer benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (defaults to DeepSeek V3.2 built-in).",
    )
    parser.add_argument("--impl", type=str, default="naive")
    parser.add_argument(
        "--scheme",
        type=str,
        default="serial",
        help="Distributed full-layer scheme (with comm); used only with "
        "--distributed. Default: serial baseline.",
    )
    parser.add_argument("--mode", type=str, choices=["tp", "ep"], default=None)
    parser.add_argument(
        "--precision", type=str, choices=["bf16", "fp16", "fp8"], default=None
    )
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=None)
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["balanced", "uniform", "skewed", "single"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the torch-reference correctness cross-check.",
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=None,
        help="Absolute tolerance for the correctness check (default: per-precision).",
    )
    parser.add_argument(
        "--verify-rtol",
        type=float,
        default=None,
        help="Relative tolerance for the correctness check (default: per-precision).",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Spawn world_size GPU procs and measure the full layer with comm.",
    )
    parser.add_argument("--all2all-backend", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if args.config is not None:
        config = MoEBenchConfig.from_file(args.config)
    else:
        config = MoEBenchConfig()
    config = _apply_overrides(config, args)

    if config.distributed:
        # Full MoE layer including cross-rank communication (multi-GPU). The
        # unit under benchmark is a DistributedScheme (compute + comm), selected
        # with --scheme; --impl (compute-only) does not apply here.
        from .distributed import run_distributed

        run_distributed(config, args.scheme)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device.")
    torch.set_default_device(config.device)

    impl = get_implementation(args.impl)
    results = run_sweep(config, impl)
    print_report(config, impl.name, results)

    if config.output_json:
        write_json(config.output_json, config, impl.name, results)


if __name__ == "__main__":
    main()
