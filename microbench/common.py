# SPDX-License-Identifier: Apache-2.0
"""microbench 公共设施。

- token sweep:MB_TOKENS(全 rank 总 token 数,逗号分隔)覆盖默认
  [128, 512, 1024, 2048, 5120, 6648, 8192];每卡 tokens = 总数 / 4。
- 4 卡锁步计时:每次迭代前 dist.barrier,CUDA events 计时,取 median,
  再 max-over-ranks(最慢 rank gate 整层)——与 tools/analyze_overlap 同口径。
- TK 环境开关在 worker 内显式固定(apply_tk_env),不受调用方环境影响,保证可复现。
- 结果 JSON 写入 $MB_OUT(run_all.sh 设置)或 microbench/results/adhoc/。

所有 microbench 必须以 `python -m moe_bench.microbench.<name>` 从上级目录运行。
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time

WORLD = 4

# 全 rank token 总数 sweep;每卡 tokens = 总数 / WORLD。
GLOBAL_TOKEN_SWEEP = [128, 512, 1024, 2048, 5120, 6648, 8192]

# TK 默认路径开关(与 HANDOFF §5 一致)。microbench worker 里显式设置,
# 保证每次运行的路径确定(公平性/可复现性)。
TK_DEFAULT_ENV = {
    "TK_DISPATCH": "pull",
    "TK_COMBINE": "prered_push",
    "TK_FUSE_GATEUP": "1",
    "TK_GPU_SCHED": "1",
    "TK_DEDUP": "0",
    "TK_ROW_BLOCK": "128",
}


def apply_tk_env(**overrides) -> dict:
    """固定 TK_* 开关(默认路径 + 指定覆盖),返回生效的取值。"""
    env = dict(TK_DEFAULT_ENV)
    env.update({k: str(v) for k, v in overrides.items()})
    for k, v in env.items():
        os.environ[k] = v
    return env


def global_token_sweep() -> list[int]:
    env = os.environ.get("MB_TOKENS")
    vals = [int(x) for x in env.split(",")] if env else list(GLOBAL_TOKEN_SWEEP)
    for v in vals:
        assert v % WORLD == 0, f"total tokens {v} must be divisible by world={WORLD}"
    return vals


def warmup_iters() -> int:
    return int(os.environ.get("MB_WARMUP", "10"))


def bench_iters() -> int:
    return int(os.environ.get("MB_ITERS", "30"))


def open_port() -> int:
    try:
        from vllm.utils.network_utils import get_open_port
        return get_open_port()
    except Exception:
        import socket
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port


# ---------------- worker 侧(4 卡分布式) ----------------

def init_dist_worker(rank: int, world: int, init_method: str):
    """初始化本 rank 的 device + 进程组;返回 (device, DistContext)。"""
    import torch
    import torch.distributed as dist
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)
    dist.all_reduce(torch.tensor([rank], device=device))
    torch.manual_seed(0)
    from moe_bench.context import DistContext
    ctx = DistContext(rank=rank, world_size=world, local_rank=rank,
                      device=device, group=None)
    return device, ctx


def precision_str() -> str:
    """microbench 的精度口径:MB_PRECISION ∈ {bf16(默认), fp16, fp8}。

    fp8 依赖 TK scheme 支持 quant_config(当前 TKFusedEP 是 bf16-only,
    见 README「切 FP8」一节);vLLM/NCCL 侧(mb1 的 vllm_compute、mb2、
    mb4 的 serial)对 fp8 天然可用。
    """
    return os.environ.get("MB_PRECISION", "bf16")


def make_cfg(num_tokens_per_rank: int, num_experts: int = 64,
             distribution: str = "balanced", seed: int = 0,
             precision: str | None = None):
    """默认 shape(E=64/topk8/hidden4096/inter3072, EP world=4)的配置。

    precision 缺省取 MB_PRECISION(默认 bf16)。精度改这里即全套 microbench
    生效:make_weights 会按精度生成(fp8 时做 per-block 量化并构造
    quant_config),vllm fused_experts / SerialNaive / TK scheme 都从
    problem.quant_config 取量化信息。
    """
    from moe_bench.config import (Distribution, MoEBenchConfig, ParallelMode,
                                  Precision, RoutingConfig)
    if precision is None:
        precision = precision_str()
    return MoEBenchConfig(
        hidden_size=4096, intermediate_size=3072, num_experts=num_experts, topk=8,
        parallel_mode=ParallelMode.EP, world_size=WORLD,
        precision=Precision(precision),
        num_tokens=[num_tokens_per_rank],
        routing=RoutingConfig(distribution=Distribution(distribution)),
        distributed=True, use_cuda_graph=False, seed=seed, verify=False,
        device="cuda")


def build_tk_fixture(cfg, num_tokens_per_rank: int, rank: int, ctx):
    """构建 problem + 完成 setup 的 TKFusedEP(含全部 schedule/buffer)。"""
    from moe_bench.data import make_problem, make_weights
    from moe_bench.tk_scheme import TKFusedEP
    problem = make_problem(cfg, num_tokens_per_rank, rank=rank,
                           weights=make_weights(cfg, rank=rank))
    sch = TKFusedEP()
    sch.setup(problem, ctx)
    return sch, problem


def lockstep_median_ms(fn, warmup: int, iters: int) -> float:
    """全 rank 锁步计时(barrier + events),返回本 rank 的 median ms。"""
    import torch
    import torch.distributed as dist
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(iters):
        dist.barrier()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples)


def maxrank_timings(timings: dict[str, float], device) -> dict[str, float]:
    """各 rank 的 {name: ms} 按 MAX 归并(慢者 gate 整层);全 rank 结果一致。"""
    import torch
    import torch.distributed as dist
    keys = sorted(timings)
    t = torch.tensor([timings[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return {k: float(v) for k, v in zip(keys, t.tolist())}


# ---------------- 单进程多卡(mb6/mb7 探针) ----------------

def single_dev_median_ms(fn, dev: int, warmup: int, iters: int) -> float:
    """单进程下对一个 device 上的操作计时(events on that device)。"""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    samples = []
    for _ in range(iters):
        with torch.cuda.device(dev):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples)


# ---------------- parent 侧 ----------------

def spawn_workers(worker_fn, *extra_args) -> list:
    """spawn 4 个 rank 跑 worker_fn(rank, world, init_method, out_list, *extra)。"""
    import torch.multiprocessing as mp
    init_method = f"tcp://localhost:{open_port()}"
    mgr = mp.Manager()
    out_list = mgr.list()
    mp.spawn(worker_fn, args=(WORLD, init_method, out_list, *extra_args),
             nprocs=WORLD, join=True)
    return list(out_list)


def _git_commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def result_meta(extra: dict | None = None) -> dict:
    import torch
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "n/a"),
        "world": WORLD,
        "warmup": warmup_iters(),
        "iters": bench_iters(),
        "tk_env": {k: os.environ.get(k, v) for k, v in TK_DEFAULT_ENV.items()},
        "token_sweep_total": global_token_sweep(),
        "shape": {"hidden": 4096, "intermediate": 3072, "num_experts": 64,
                  "topk": 8, "precision": precision_str(), "parallel": "ep"},
    }
    try:
        import vllm
        meta["vllm"] = vllm.__version__
    except Exception:
        pass
    if extra:
        meta.update(extra)
    return meta


def out_dir() -> str:
    d = os.environ.get("MB_OUT")
    if not d:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "adhoc")
    os.makedirs(d, exist_ok=True)
    return d


def write_json(name: str, meta: dict, rows: list[dict]) -> str:
    path = os.path.join(out_dir(), f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")
    return path
