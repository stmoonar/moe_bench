# SPDX-License-Identifier: Apache-2.0
"""TP L0 三段独立 microbench：push、pull-order scatter、grouped GEMM。

默认严格读取 ``configs/tp_rtx_pro5000_4gpu_fp8.yaml``，遍历：

* token push: 1, 2, 4, 8, 16, 20, 24, 32 SM；
* 本地 token scatter（真实 ``pull_order``）: 4, 8, 16, 20, 24, 32, 40, 48, 64 SM；
* grouped GEMM: 对每个 ``(push_sms, scatter_sms)`` 组合使用
  ``device_sms - push_sms - scatter_sms`` 个 persistent blocks。

从 ``moe_bench`` 上一级目录运行：

    python -m moe_bench.tools.microbench_l0_sms

可选覆盖会写入结果 JSON：

    --experts E --tokens T --warmup N --iters N
    --push-sms 1,2,4,8,16,20,24,32 --scatter-sms 4,8,16,20,24,32,40,48,64
    --dist balanced|uniform|skewed|single --output path.json

三段没有前后数据依赖：scatter 的完整 staging/scales/arrival flags 在 sweep 前
直接预填，所有 token 均已到达；GEMM 的完整 gathered/scales 也从全量输入直接预填，
不消费 scatter kernel 的结果。计时只覆盖目标 kernel；counter 清零、scatter 行块
计数 seed、fixture 构造、同步和正确性检查都在计时区外。每次迭代先在各 rank
本地计时，再逐样本取 rank max。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moe_bench.config import Distribution, MoEBenchConfig, RoutingConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tp_rtx_pro5000_4gpu_fp8.yaml"


def _csv_ints(value: str) -> list[int]:
    values = [int(x) for x in value.split(",") if x.strip()]
    if not values or any(x <= 0 for x in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("需要逗号分隔的正整数，且不能重复")
    return values


def _stats(samples_us: list[float]) -> dict[str, float]:
    return {
        "avg_us": statistics.fmean(samples_us),
        "min_us": min(samples_us),
        "median_us": statistics.median(samples_us),
    }


def _bench(
    prepare: Callable[[], None],
    launch: Callable[[], None],
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        prepare()
        launch()
    torch.cuda.synchronize(device)
    dist.barrier()

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(iters):
        prepare()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        launch()
        end.record()
        events.append((begin, end))
    torch.cuda.synchronize(device)

    samples = torch.tensor(
        [begin.elapsed_time(end) * 1000.0 for begin, end in events],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(samples, op=dist.ReduceOp.MAX)
    return _stats(samples.cpu().tolist())


def _gather_full_inputs(s, world: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build immutable golden inputs once; never timed and not produced by a stage."""
    tokens_u8 = s.pre_tokens.data_.view(torch.uint8)
    full_u8 = torch.empty(
        world * s.num_tokens, s.H, dtype=torch.uint8, device=tokens_u8.device
    )
    full_scales = torch.empty(
        world * s.num_tokens,
        s.H // 128,
        dtype=torch.float32,
        device=tokens_u8.device,
    )
    dist.all_gather_into_tensor(full_u8, tokens_u8)
    dist.all_gather_into_tensor(full_scales, s.pre_scales.data_)
    torch.cuda.synchronize()
    dist.barrier()
    return full_u8, full_scales


def _check_push(
    s,
    seq: int,
    world: int,
    rank: int,
    full_u8: torch.Tensor,
    full_scales: torch.Tensor,
) -> None:
    remote = torch.arange(world, device=full_u8.device) != rank
    rows = remote.repeat_interleave(s.num_tokens)
    if not torch.equal(s.ag_staging_fp8.data_.view(torch.uint8)[rows], full_u8[rows]):
        raise AssertionError("push payload mismatch")
    if not torch.equal(s.ag_sscales.data_[rows], full_scales[rows]):
        raise AssertionError("push scale mismatch")
    if not torch.all(s.ag_flags.data_.view(-1)[rows] == seq):
        raise AssertionError("push arrival flag mismatch")


def _prime_scatter_fixture(
    s, seq: int, full_u8: torch.Tensor, full_scales: torch.Tensor
) -> None:
    """Make every source token locally available before scatter starts."""
    s.ag_staging_fp8.data_.view(torch.uint8).copy_(full_u8)
    s.ag_sscales.data_.copy_(full_scales)
    s.ag_flags.data_.fill_(seq)


def _prime_gemm_fixture(
    s, full_u8: torch.Tensor, full_scales: torch.Tensor
) -> None:
    """Build gathered rows directly from golden inputs, without running scatter."""
    slots = s.tp_slots.reshape(-1).long()
    if slots.numel() != torch.unique(slots).numel() or int(slots.min()) < 0:
        raise AssertionError("tp_slots must be unique non-negative GEMM rows")
    jobs = torch.arange(
        s.ctx.world_size * s.num_tokens, device=slots.device
    ).repeat_interleave(s.top_k)
    s.gathered.zero_()
    s.gathered_scales.zero_()
    s.gathered.view(torch.uint8).index_copy_(0, slots, full_u8.index_select(0, jobs))
    s.gathered_scales.index_copy_(0, slots, full_scales.index_select(0, jobs))


def _check_scatter(s, full_u8: torch.Tensor, full_scales: torch.Tensor) -> None:
    slots = s.tp_slots.reshape(-1).long()
    jobs = torch.arange(
        s.ctx.world_size * s.num_tokens, device=slots.device
    ).repeat_interleave(s.top_k)
    got_u8 = s.gathered.view(torch.uint8)[slots]
    if not torch.equal(got_u8, full_u8[jobs]):
        raise AssertionError("pull-order scatter token mismatch")
    if not torch.equal(s.gathered_scales[slots], full_scales[jobs]):
        raise AssertionError("pull-order scatter scale mismatch")
    nblk = s.num_padded_total // 128
    counters = s.barrier_l0.data_[0, :nblk]
    if not torch.all(counters == 128):
        raise AssertionError("scatter row-block counter did not reach ROW_BLOCK")


def _worker_impl(
    rank: int,
    world: int,
    init_method: str,
    cfg: MoEBenchConfig,
    push_sms: list[int],
    scatter_sms: list[int],
    output: str,
) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=rank,
        world_size=world,
        device_id=device,
        timeout=timedelta(minutes=3),
    )
    dist.all_reduce(torch.tensor([rank], device=device))

    from moe_bench.context import DistContext
    from moe_bench.data import make_problem, make_weights
    from moe_bench.tk_tp_scheme import TKFusedTP

    props = torch.cuda.get_device_properties(device)
    total_sms = props.multi_processor_count
    requested_push_sms = list(push_sms)
    requested_scatter_sms = list(scatter_sms)
    device_desc = [None] * world
    dist.all_gather_object(device_desc, (props.name, total_sms))
    if len(set(device_desc)) != 1:
        raise RuntimeError(f"各 rank GPU/SM 数不一致: {device_desc}")
    skipped_push = [sms for sms in push_sms if sms > total_sms]
    skipped_scatter = [sms for sms in scatter_sms if sms > total_sms]
    push_sms = [sms for sms in push_sms if sms <= total_sms]
    scatter_sms = [sms for sms in scatter_sms if sms <= total_sms]
    if not push_sms or not scatter_sms:
        raise ValueError(f"没有不超过设备 SM 数 {total_sms} 的 sweep 点")
    combinations = [
        (p, q, total_sms - p - q)
        for p in push_sms
        for q in scatter_sms
        if p + q < total_sms
    ]
    if not combinations:
        raise ValueError(f"没有满足 push+scatter < {total_sms} 的 GEMM 组合")

    free_b, total_b = torch.cuda.mem_get_info(device)
    if rank == 0:
        print(
            f"[microbench] device={props.name}, SM={total_sms}, "
            f"启动前显存空闲={free_b / 2**30:.1f}/{total_b / 2**30:.1f} GiB",
            flush=True,
        )
        if skipped_push or skipped_scatter:
            print(
                f"[microbench] 跳过超过设备上限的点: "
                f"push={skipped_push}, scatter={skipped_scatter}",
                flush=True,
            )

    ctx = DistContext(
        rank=rank, world_size=world, local_rank=rank, device=device, group=None
    )
    tokens = cfg.num_tokens[0]
    problem = make_problem(cfg, tokens, rank=rank, weights=make_weights(cfg, rank))
    s = TKFusedTP()
    s.setup(problem, ctx)
    tk = s.tk
    seq = 1

    tk.rowgroup_quant_fp8(
        s.problem.hidden_states, s.pre_tokens.data_, s.pre_scales.data_
    )
    torch.cuda.synchronize(device)
    full_u8, full_scales = _gather_full_inputs(s, world)

    push_results = []
    for sms in push_sms:
        def prepare_push() -> None:
            s.push_next.zero_()

        def launch_push() -> None:
            tk.moe_tp_push_microbench(
                s.pre_tokens,
                s.pre_scales,
                s.ag_staging_fp8,
                s.ag_sscales,
                s.ag_flags,
                s.push_order,
                s.push_next,
                sms,
                s.num_tokens,
                seq,
            )

        prepare_push()
        launch_push()
        torch.cuda.synchronize(device)
        dist.barrier()
        _check_push(s, seq, world, rank, full_u8, full_scales)
        row = {"sms": sms, **_bench(
            prepare_push, launch_push, cfg.warmup_iters, cfg.bench_iters, device
        )}
        push_results.append(row)
        if rank == 0:
            print(f"  push    SM={sms:2d}: {row['avg_us']:9.2f} us", flush=True)

    # Scatter fixture 与 push sweep 无关：直接预填完整 staging/scales 和所有
    # arrival flags，因此 kernel 启动前每个本地/远端源 token 都已到达。
    _prime_scatter_fixture(s, seq, full_u8, full_scales)
    torch.cuda.synchronize(device)
    dist.barrier()

    nblk = s.num_padded_total // 128
    scatter_results = []
    for sms in scatter_sms:
        def prepare_scatter() -> None:
            s.pull_next.zero_()
            s.barrier_l0.data_[0, :nblk].copy_(s.slack)

        def launch_scatter() -> None:
            tk.moe_tp_scatter_microbench(
                s.pre_tokens,
                s.pre_scales,
                s.ag_staging_fp8,
                s.ag_sscales,
                s.ag_flags,
                s.gathered,
                s.gathered_scales,
                s.tp_slots,
                s.pull_order,
                s.pull_next,
                s.barrier_l0,
                sms,
                s.num_tokens,
                seq,
            )

        # 正确性门从空输出开始，避免前一个 SM 配置的旧数据掩盖漏写。
        s.gathered.zero_()
        s.gathered_scales.zero_()
        prepare_scatter()
        launch_scatter()
        torch.cuda.synchronize(device)
        _check_scatter(s, full_u8, full_scales)
        row = {"sms": sms, **_bench(
            prepare_scatter, launch_scatter,
            cfg.warmup_iters, cfg.bench_iters, device
        )}
        scatter_results.append(row)
        if rank == 0:
            print(f"  scatter SM={sms:2d}: {row['avg_us']:9.2f} us", flush=True)

    # GEMM fixture 与 scatter sweep 无关：直接按 routing slots 从 golden 输入构造
    # 全量 gathered rows。kernel 启动前所有 row block 均已就绪且没有 readiness gate。
    _prime_gemm_fixture(s, full_u8, full_scales)
    torch.cuda.synchronize(device)

    # 参考和 sweep 都走生产 L0 同款 grouped GEMM + fp32-acc SwiGLU store
    # policy，只改变 persistent grid.x。
    gemm_ref = torch.zeros_like(s.act)
    gemm_out = torch.zeros_like(s.act)
    gemm_counter = torch.zeros(1, dtype=torch.int32, device=device)
    gemm_counter.zero_()
    tk.grouped_gemm_fp8_glu_sms(
        s.gathered,
        s.gathered_scales,
        s.w_gateup_fp8,
        s.w1_il_scales,
        gemm_ref,
        s.padded,
        s.blk_expert,
        gemm_counter,
        s.slack if s.two_level else s.slack.new_empty(0),
        total_sms,
    )
    torch.cuda.synchronize(device)

    gemm_by_sms: dict[int, dict[str, float]] = {}
    remaining_sms = sorted({remain for _, _, remain in combinations}, reverse=True)
    for sms in remaining_sms:
        gemm_out.zero_()
        gemm_counter.zero_()
        tk.grouped_gemm_fp8_glu_sms(
            s.gathered,
            s.gathered_scales,
            s.w_gateup_fp8,
            s.w1_il_scales,
            gemm_out,
            s.padded,
            s.blk_expert,
            gemm_counter,
            s.slack if s.two_level else s.slack.new_empty(0),
            sms,
        )
        torch.cuda.synchronize(device)
        if not torch.equal(gemm_out, gemm_ref):
            raise AssertionError(f"grouped GEMM+SwiGLU mismatch at {sms} SM")

        def prepare_gemm() -> None:
            gemm_counter.zero_()

        def launch_gemm() -> None:
            tk.grouped_gemm_fp8_glu_sms(
                s.gathered,
                s.gathered_scales,
                s.w_gateup_fp8,
                s.w1_il_scales,
                gemm_out,
                s.padded,
                s.blk_expert,
                gemm_counter,
                s.slack if s.two_level else s.slack.new_empty(0),
                sms,
            )

        gemm_by_sms[sms] = _bench(
            prepare_gemm, launch_gemm,
            cfg.warmup_iters, cfg.bench_iters, device
        )
        if rank == 0:
            print(
                f"  gemm     SM={sms:2d}: {gemm_by_sms[sms]['avg_us']:9.2f} us",
                flush=True,
            )

    gemm_results = [
        {
            "push_sms": p,
            "scatter_sms": q,
            "gemm_sms": remain,
            **gemm_by_sms[remain],
        }
        for p, q, remain in combinations
    ]
    if rank == 0:
        print("\n  push scatter remain_gemm | gemm avg (us)")
        for row in gemm_results:
            print(
                f"  {row['push_sms']:4d} {row['scatter_sms']:7d} "
                f"{row['gemm_sms']:11d} | {row['avg_us']:12.2f}"
            )
        payload = {
            "config": str(CONFIG),
            "overrides": {
                "num_experts": cfg.num_experts,
                "num_tokens": tokens,
                "warmup_iters": cfg.warmup_iters,
                "bench_iters": cfg.bench_iters,
                "routing": cfg.routing.distribution.value,
                "requested_push_sms": requested_push_sms,
                "requested_scatter_sms": requested_scatter_sms,
                "push_sms": push_sms,
                "scatter_sms": scatter_sms,
                "skipped_push_sms": skipped_push,
                "skipped_scatter_sms": skipped_scatter,
            },
            "device": props.name,
            "device_sms": total_sms,
            "world_size": world,
            "num_padded_total": s.num_padded_total,
            "timing": "CUDA events; per-sample max over ranks; preparation excluded",
            "fixture": (
                "dependency-free: scatter staging/scales/flags are fully prefilled; "
                "GEMM gathered/scales are independently materialized from golden inputs"
            ),
            "interpretation": (
                "three standalone all-inputs-ready throughput curves; grouped GEMM uses "
                "the initial remaining blocks (device-push-scatter). Production fused "
                "comm blocks join GEMM after finishing, so these rows are not fused "
                "latency predictions"
            ),
            "push": push_results,
            "scatter": scatter_results,
            "grouped_gemm_remaining": gemm_results,
        }
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"\n[microbench] 结果已写入 {out}", flush=True)

    dist.destroy_process_group()


def _worker(*args) -> None:
    try:
        _worker_impl(*args)
    except BaseException:
        # kernel trap 后禁止走 CUDA synchronize / NCCL destroy / IPC 析构。
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument(
        "--push-sms",
        type=_csv_ints,
        default=_csv_ints("1,2,4,8,16,20,24,32"),
    )
    parser.add_argument(
        "--scatter-sms",
        type=_csv_ints,
        default=_csv_ints("4,8,16,20,24,32,40,48,64"),
    )
    parser.add_argument("--dist", choices=[x.value for x in Distribution])
    parser.add_argument("--skew-alpha", type=float)
    parser.add_argument("--active", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = MoEBenchConfig.from_file(str(CONFIG))
    overrides = {}
    if args.experts is not None:
        overrides["num_experts"] = args.experts
    if args.tokens is not None:
        overrides["num_tokens"] = [args.tokens]
    if args.warmup is not None:
        overrides["warmup_iters"] = args.warmup
    if args.iters is not None:
        overrides["bench_iters"] = args.iters
    if args.dist is not None or args.skew_alpha is not None or args.active is not None:
        overrides["routing"] = RoutingConfig(
            distribution=(
                Distribution(args.dist) if args.dist else cfg.routing.distribution
            ),
            skew_alpha=(
                args.skew_alpha
                if args.skew_alpha is not None
                else cfg.routing.skew_alpha
            ),
            num_active_experts=(
                args.active
                if args.active is not None
                else cfg.routing.num_active_experts
            ),
        )
    cfg = dataclasses.replace(cfg, **overrides) if overrides else cfg
    if cfg.world_size != 4 or not cfg.distributed:
        raise ValueError("该 microbench 要求主配置的 distributed=true, world_size=4")
    if cfg.warmup_iters < 1 or cfg.bench_iters < 1:
        raise ValueError("warmup/iters 必须为正数")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or str(
        ROOT / "tp_test_results" / f"tp_run_{stamp}" / "microbench_l0_sms.json"
    )
    shown = {
        **overrides,
        "push_sms": args.push_sms,
        "scatter_sms": args.scatter_sms,
        "output": output,
    }
    print(f"[microbench] config={CONFIG.name} overrides={shown}", flush=True)

    from vllm.utils.network_utils import get_open_port

    init_method = f"tcp://localhost:{get_open_port()}"
    mp.spawn(
        _worker,
        args=(
            cfg.world_size,
            init_method,
            cfg,
            args.push_sms,
            args.scatter_sms,
            output,
        ),
        nprocs=cfg.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
