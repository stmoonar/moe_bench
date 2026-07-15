# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed MoE-layer benchmark (real cross-rank communication).

Spawns ``world_size`` single-GPU processes and times a
:class:`~moe_bench.schemes.DistributedScheme` — one rank's *full* MoE layer
including dispatch and combine. The baseline scheme (``serial``) is the
non-overlapped ``AllGather -> fused_experts -> ReduceScatter`` path; register
your own overlap/fused scheme in :mod:`moe_bench.schemes` and select it with
``--scheme`` (see ``ADDING_IMPLEMENTATIONS.md``).

Uses plain ``torch.distributed`` collectives and vLLM's ``fused_experts`` — no
modular-kernel / forward-context setup — so the harness stays small and a scheme
only has to implement ``setup``/``run``.
"""

from __future__ import annotations

import dataclasses
import gc
import os
import statistics
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .config import MoEBenchConfig, ParallelMode
from .context import DistContext
from .data import make_golden_problem, make_logical_weights, make_problem, make_weights
from .reference import reference_moe, verify_output
from .report import print_report, write_json
from .schemes import DistributedScheme, get_scheme


@dataclass(frozen=True)
class DistributedRunSpec:
    """One case in a shared-worker distributed benchmark suite."""

    name: str
    config: MoEBenchConfig
    scheme_name: str
    env: dict[str, str] = field(default_factory=dict)


@contextmanager
def _temporary_environment(overrides: dict[str, str]):
    """Apply per-case environment switches without leaking into the next case."""

    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _weights_cache_key(config: MoEBenchConfig) -> tuple:
    """Fields that determine the generated local weight tensors."""

    return (
        config.hidden_size,
        config.intermediate_size,
        config.num_experts,
        config.parallel_mode.value,
        config.world_size,
        config.precision.value,
        tuple(config.block_shape),
        config.seed,
        config.device,
    )


def _time_scheme(scheme: DistributedScheme, config: MoEBenchConfig) -> dict[str, float]:
    """Per-call latency stats (ms) of ``scheme.run`` on this rank.

    Collectives keep the ranks in lockstep, so each rank times its own calls and
    the harness reports the max across ranks (the slowest gates the step). CUDA
    graphs are not used here (capture across NCCL collectives is fragile).
    """
    for _ in range(config.warmup_iters):
        scheme.run()
    torch.cuda.synchronize()
    dist.barrier()

    samples_ms: list[float] = []
    for _ in range(config.bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scheme.run()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
    return {
        "avg": statistics.fmean(samples_ms),
        "min": min(samples_ms),
        "med": statistics.median(samples_ms),
    }


def _verify_rank(
    config: MoEBenchConfig,
    scheme: DistributedScheme,
    num_tokens: int,
    rank: int,
    golden_weights,
) -> tuple[float, bool]:
    """Check this rank's output against a full-model torch reference.

    The rank's combined output equals a full MoE over its own token shard, so the
    reference is :func:`reference_moe` on that shard through the logical (all
    expert) weights — see :func:`make_golden_problem`. Returns ``(rel_err,
    passed)``.
    """
    golden_problem = make_golden_problem(
        config, num_tokens, rank, weights=golden_weights
    )
    reference = reference_moe(golden_problem)
    output = scheme.run()  # untimed forward for the actual output
    check = verify_output(golden_problem, output, reference)
    if not check.passed:
        # fp8 远程排障(docs/38): 表格只有 rel_err, 失败时把 max_abs/atol/rtol
        # 全打出来, 一次 zip 回流就能定位是量化口径差异还是真 bug。
        print(f"  [verify rank {rank}] {check}", flush=True)
    return check.rel_err, check.passed


def _run_rank(
    config: MoEBenchConfig,
    ctx: DistContext,
    scheme_name: str,
    *,
    weights=None,
    golden_weights=None,
) -> list[dict]:
    """Run the token sweep on this rank; rank 0 reports max latency over ranks."""
    # Weights don't depend on the token count; build once for the sweep.
    if weights is None:
        weights = make_weights(config, rank=ctx.rank)
    if config.verify and golden_weights is None:
        golden_weights = make_logical_weights(config)
    results: list[dict] = []
    any_verify_failed = False

    for num_tokens in config.num_tokens:
        problem = make_problem(config, num_tokens, rank=ctx.rank, weights=weights)
        scheme = get_scheme(scheme_name)
        scheme.setup(problem, ctx)

        stats = _time_scheme(scheme, config)

        rel_err, passed = (0.0, True)
        if config.verify:
            rel_err, passed = _verify_rank(
                config, scheme, num_tokens, ctx.rank, golden_weights
            )

        # Latency is gated by the slowest rank; verify fails if any rank fails.
        agg = torch.tensor(
            [
                stats["avg"],
                stats["min"],
                stats["med"],
                float(problem.num_local_assignments),
                rel_err,
                0.0 if passed else 1.0,
            ],
            device=ctx.device,
        )
        dist.all_reduce(agg, op=dist.ReduceOp.MAX)
        any_verify_failed |= agg[5].item() > 0.0

        row = {
            "num_tokens": num_tokens,
            "latency_us": float(agg[0].item()) * 1e3,
            "latency_min_us": float(agg[1].item()) * 1e3,
            "latency_med_us": float(agg[2].item()) * 1e3,
            "local_assignments": int(agg[3].item()),
        }
        if config.verify:
            row["rel_err"] = float(agg[4].item())
            row["verify"] = "ok" if agg[5].item() == 0.0 else "FAIL"
        results.append(row)

        scheme.close()
        del problem, scheme
        gc.collect()
        torch.cuda.empty_cache()

    if ctx.is_rank0:
        comm = (
            "all-reduce"
            if config.parallel_mode == ParallelMode.TP
            else f"all-gather+reduce-scatter ({config.all2all_backend})"
        )
        impl_name = f"{scheme_name} (distributed, with comm)"
        print_report(config, impl_name, results, comm=comm)
        if any_verify_failed:
            print(
                "  [warn] correctness check FAILED on one or more token counts; "
                "the scheme does not match the torch reference."
            )
        if config.output_json:
            write_json(
                config.output_json, config, f"{scheme_name}-distributed", results
            )
    return results


def _worker(
    local_rank: int,
    world_size: int,
    init_method: str,
    config: MoEBenchConfig,
    scheme_name: str,
) -> None:
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=local_rank,
        world_size=world_size,
        device_id=device,
    )
    # Warm the world group.
    dist.all_reduce(torch.tensor([local_rank], device=device))

    ctx = DistContext(
        rank=local_rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        group=None,  # default (world) group
    )
    try:
        _run_rank(config, ctx, scheme_name)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_suite(
    local_rank: int,
    world_size: int,
    init_method: str,
    specs: list[DistributedRunSpec],
) -> None:
    """Run many cases inside one worker/NCCL lifetime."""

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=local_rank,
        world_size=world_size,
        device_id=device,
    )
    dist.all_reduce(torch.tensor([local_rank], device=device))
    ctx = DistContext(
        rank=local_rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        group=None,
    )
    weights_cache: dict[tuple, object] = {}
    golden_cache: dict[tuple, object] = {}
    try:
        for index, spec in enumerate(specs, start=1):
            dist.barrier()
            if ctx.is_rank0:
                print(
                    f"\n[suite {index}/{len(specs)}] {spec.name}: "
                    f"scheme={spec.scheme_name} env={spec.env or '{}'}",
                    flush=True,
                )
            key = _weights_cache_key(spec.config)
            weights = weights_cache.get(key)
            if weights is None:
                weights = make_weights(spec.config, rank=ctx.rank)
                weights_cache[key] = weights
            golden_weights = None
            if spec.config.verify:
                golden_weights = golden_cache.get(key)
                if golden_weights is None:
                    golden_weights = make_logical_weights(spec.config)
                    golden_cache[key] = golden_weights
            with _temporary_environment(spec.env):
                _run_rank(
                    spec.config,
                    ctx,
                    spec.scheme_name,
                    weights=weights,
                    golden_weights=golden_weights,
                )
            dist.barrier()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed(config: MoEBenchConfig, scheme_name: str = "serial") -> None:
    """Spawn ``world_size`` GPU workers and run the distributed benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed benchmark requires CUDA GPUs.")
    n_gpus = torch.cuda.device_count()
    if n_gpus < config.world_size:
        raise RuntimeError(
            f"world_size={config.world_size} but only {n_gpus} GPU(s) visible."
        )

    from vllm.utils.network_utils import get_open_port

    if config.use_cuda_graph:
        print("[info] CUDA graphs are disabled in distributed mode; timing eager.")
        config = dataclasses.replace(config, use_cuda_graph=False)

    host = os.getenv("LOCALHOST", "localhost")
    init_method = f"tcp://{host}:{get_open_port()}"
    mp.spawn(
        _worker,
        args=(config.world_size, init_method, config, scheme_name),
        nprocs=config.world_size,
        join=True,
    )


def run_distributed_suite(specs: list[DistributedRunSpec]) -> None:
    """Run multiple distributed cases with one process-group initialization.

    Cases remain isolated at the scheme level (fresh ``setup``/``close``), while
    worker processes, NCCL initialization, and compatible weight tensors are
    reused. A suite failure terminates the whole worker group; rerun with the
    legacy per-case runner when deadlock isolation is needed.
    """

    if not specs:
        raise ValueError("Distributed suite must contain at least one case")
    world_size = specs[0].config.world_size
    if any(spec.config.world_size != world_size for spec in specs):
        raise ValueError("All suite cases must use the same world_size")
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed benchmark requires CUDA GPUs.")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"world_size={world_size} but only {torch.cuda.device_count()} GPU(s) visible."
        )

    normalized: list[DistributedRunSpec] = []
    for spec in specs:
        config = spec.config
        if config.use_cuda_graph:
            config = dataclasses.replace(config, use_cuda_graph=False)
        normalized.append(dataclasses.replace(spec, config=config))

    from vllm.utils.network_utils import get_open_port

    host = os.getenv("LOCALHOST", "localhost")
    init_method = f"tcp://{host}:{get_open_port()}"
    mp.spawn(
        _worker_suite,
        args=(world_size, init_method, normalized),
        nprocs=world_size,
        join=True,
    )
