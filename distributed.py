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
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .config import MoEBenchConfig, ParallelMode
from .context import DistContext
from .data import make_golden_problem, make_logical_weights, make_problem, make_weights
from .reference import reference_moe, verify_output
from .report import print_report, write_json
from .schemes import DistributedScheme, get_scheme


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
    config: MoEBenchConfig, ctx: DistContext, scheme_name: str
) -> list[dict]:
    """Run the token sweep on this rank; rank 0 reports max latency over ranks."""
    # Weights don't depend on the token count; build once for the sweep.
    weights = make_weights(config, rank=ctx.rank)
    golden_weights = make_logical_weights(config) if config.verify else None
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


def _fail_fast_exit() -> None:
    """异常(含 kernel trap 后的 CUDA error)后的硬退出。

    死锁红线: 出错后不再执行 torch.cuda.synchronize / destroy_process_group /
    解释器正常退出 —— 这些路径要做 NCCL 跨 rank 握手和逐个 IPC unmap/context
    销毁, 会排队在被卡住的 RM/uvm 锁后面(2026-07-16 wedge 取证: 受害 worker
    卡在 uvm_map_external_allocation)。os._exit 让内核态驱动统一回收本进程
    GPU 资源, 其余 rank 由 mp.spawn 按非零退出码收割。
    """
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


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
    # 需要 NVSHMEM 的 scheme(如 tdtp): 用 triton_dist 的初始化(内部完成 PG
    # init + NVSHMEM init, 不能再重复 init PG)。env RANK/WORLD_SIZE/
    # LOCAL_RANK 由 mp.spawn 提供; LOCAL_WORLD_SIZE 默认 8, 本机 4 卡要覆盖。
    from .schemes import SCHEMES as _SCHEMES
    _nvshmem = (scheme_name in _SCHEMES
                and getattr(_SCHEMES[scheme_name], "requires_nvshmem", False))
    if _nvshmem:
        # mp.spawn 只把 local_rank 作为参数传入, 不设 RANK/WORLD_SIZE/
        # LOCAL_RANK(那是 elastic launch 的行为); initialize_distributed 全
        # 靠 env, 缺省会退化成"每进程都当 rank 0"(抢 TCPStore server 端口
        # 冲突) + world_size=1(Gloo 0 peers)。从参数补齐。
        os.environ.setdefault("RANK", str(local_rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))
        from triton_dist.utils import initialize_distributed
        initialize_distributed()
    else:
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
        torch.cuda.synchronize()
    except Exception:
        _fail_fast_exit()
    if dist.is_initialized():
        if _nvshmem:
            from triton_dist.utils import finalize_distributed
            finalize_distributed()
        else:
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
    port = get_open_port()
    init_method = f"tcp://{host}:{port}"
    # NVSHMEM scheme(tdtp)走 triton_dist.initialize_distributed, 其内部
    # init_process_group 用 env:// rendezvous, 需要 MASTER_ADDR/MASTER_PORT;
    # mp.spawn 不设置, 与 init_method 同源补上(子进程继承父进程环境)。
    os.environ.setdefault("MASTER_ADDR", host)
    os.environ.setdefault("MASTER_PORT", str(port))
    mp.spawn(
        _worker,
        args=(config.world_size, init_method, config, scheme_name),
        nprocs=config.world_size,
        join=True,
    )


