# SPDX-License-Identifier: Apache-2.0
"""baseline 侧调优:给 vLLM triton fused_moe 扫出本机的 tile 配置。

背景:RTX PRO 5000(sm120)在 vLLM 的 `fused_moe/configs/` 里没有任何条目,
`try_get_optimal_moe_config` 查表落空后走 `get_default_config` 的 blockwise
兜底分支——那里 BLOCK_N/BLOCK_K 直接抄 `block_shape`,与硬件无关。docs/09 从
NCU grid 反解出的 BM=64/BN=128 就是这个兜底值,也就是说 serial baseline 一直
在跑未调优默认值。本脚本按主配置口径扫 tile 空间,产出 vLLM 能直接查表命中
的 JSON。

口径从主配置读(configs/tp_rtx_pro5000_4gpu_fp8.yaml),命令行只覆盖最小字段
并打印。被测对象就是 baseline 的 `NaiveFusedExperts`,输入按 `SerialNaive` 的
AllGather 语义复刻(rank-major 拼接 world_size 份 token shard),所以扫出来的
config 与 serial e2e 里那两颗 `fused_moe_kernel` 的形状逐项一致。

  单卡运行(先确认卡空闲):
    CUDA_VISIBLE_DEVICES=0 python -m moe_bench.tools.tune_triton_moe

  常用覆盖:
    --dist uniform    路由分布(默认取主配置)。建议 balanced/uniform 各调一份:
                      按 docs/09,BLOCK_M 抬大在 balanced 下是免费的(256 行/expert
                      整除),uniform 下要付 padding 税,两者最优解可能不同。
    --tokens 512      每 rank token 数,M = world_size × tokens
    --iters 50        阶段2 精测迭代数(默认取主配置 bench_iters)
    --keep 15         阶段1 之后进入精测的候选数
    --start N         从第 N 个候选续跑(某个 config 打崩 CUDA 上下文后用)
    --out DIR         结果目录(默认 tp_test_results/triton_tune_<时间戳>)
    --dry-run         只打印形状/候选数/目标文件名,不碰 GPU

产出 `DIR/E=..,N=..,device_name=..,dtype=fp8_w8a8,block_shape=[128,128].json`,
接回 benchmark 的方式(不改 site-packages):
    VLLM_TUNED_CONFIG_FOLDER=DIR python -m moe_bench.tools.run_tktp \\
        --scheme serial --no-verify --iters 50
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from itertools import product

from moe_bench.config import Distribution, MoEBenchConfig, ParallelMode, Precision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "configs", "tp_rtx_pro5000_4gpu_fp8.yaml")

# 与 vLLM benchmarks/kernels/benchmark_moe.py 的 get_configs_compute_bound 同源。
BLOCK_M_RANGE = [16, 32, 64, 128, 256]
BLOCK_N_RANGE = [32, 64, 128, 256]
BLOCK_K_RANGE = [64, 128, 256]
GROUP_M_RANGE = [1, 16, 32, 64]
NUM_WARPS_RANGE = [4, 8]
NUM_STAGES_RANGE = [2, 3, 4, 5]

# host 端就被拒的候选(smem 不够 / launch 参数非法 / 编译失败)—— CUDA 上下文
# 没被弄脏,跳过即可。要显式列出来是因为其中几条也带 "CUDA error" 前缀。
_SKIP_MARKERS = (
    "out of resource",
    "outofresources",
    "shared memory",
    "out of memory",
    "outofmemory",
    "invalid configuration argument",
    "invalid argument",
)
# 异步 CUDA 错误 —— 上下文已损坏,后续所有测量都不可信,必须硬退出。
_FATAL_MARKERS = (
    "illegal memory access",
    "misaligned address",
    "unspecified launch failure",
    "device-side assert",
)


def _arg(flag, default, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def _candidates(cfg: MoEBenchConfig) -> list[dict[str, int]]:
    """tile 候选集合,已按精度剪掉非法/等价项。

    fp8 blockwise 的两条约束:
      1. BLOCK_SIZE_N 必须是 block_shape[0] 的整数倍 —— kernel 用
         ``offs_bsn = offs_bn // group_n`` 向量化取 scale,跨多个 scale 块合法,
         但不能切碎一个 scale 块。
      2. BLOCK_SIZE_K 只保留 min(block_shape):invoke_fused_moe_kernel 里有
         ``BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape))``(fused_moe.py
         的 SPLIT_K 段),更大的值会被静默压回去,扫描纯属重复计时。
    """
    is_fp8 = cfg.precision == Precision.FP8
    block_k_range = BLOCK_K_RANGE
    block_n_range = BLOCK_N_RANGE
    if is_fp8:
        block_n, block_k = cfg.block_shape
        block_n_range = [n for n in BLOCK_N_RANGE if n % block_n == 0]
        block_k_range = [min(block_n, block_k)]

    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
            "GROUP_SIZE_M", "num_warps", "num_stages")
    ranges = (BLOCK_M_RANGE, block_n_range, block_k_range,
              GROUP_M_RANGE, NUM_WARPS_RANGE, NUM_STAGES_RANGE)
    return [dict(zip(keys, v)) | {"SPLIT_K": 1} for v in product(*ranges)]


def _build_problem(cfg: MoEBenchConfig, tokens: int):
    """复刻 SerialNaive AllGather 之后喂给 fused_experts 的那一份输入。

    hidden/routing 按 rank-major 拼接 world_size 份 shard(与
    ``dist.all_gather_into_tensor`` 的落盘顺序一致),权重取 rank 0 的分片。
    用 data._gen_input 而不是 make_problem(T*world_size) 是为了逐 bit 对齐:
    per-rank 生成的 routing 与整批一次生成的不同。
    """
    import torch

    from moe_bench.data import MoEProblem, _gen_input, make_weights

    weights = make_weights(cfg, rank=0)
    shards = [_gen_input(cfg, tokens, r) for r in range(cfg.world_size)]
    return MoEProblem(
        config=cfg,
        num_tokens=tokens * cfg.world_size,
        hidden_states=torch.cat([s[0] for s in shards]).contiguous(),
        w1=weights.w1,
        w2=weights.w2,
        topk_ids=torch.cat([s[1] for s in shards]).contiguous(),
        topk_weights=torch.cat([s[2] for s in shards]).contiguous(),
        expert_map=None,  # TP:全部 expert 本地
        quant_config=weights.quant_config,
    )


def _bench(fn, iters: int, warmup: int) -> float:
    """平均耗时(µs)。warmup 同时承担 triton JIT 编译。"""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def _classify(exc: Exception) -> str:
    """skip(换下一个候选)还是 fatal(上下文已脏,必须退出)。

    判据是"CUDA 上下文有没有被弄脏",不是"错误严不严重":host 端抛的异常
    (triton 编译失败、smem 超限、launch 参数校验)都不落到设备上,跳过即可;
    只有异步设备错误会毒化后续每一次测量。计时循环里紧跟 synchronize,
    异步错误一定会在当次被捕获,不会漏到下一个候选。
    """
    msg = f"{type(exc).__name__}: {exc}".lower()
    if any(m in msg for m in _SKIP_MARKERS):
        return "skip"
    if "cuda error" in msg or any(m in msg for m in _FATAL_MARKERS):
        return "fatal"
    return "skip"


def main() -> int:
    cfg = MoEBenchConfig.from_file(CONFIG)
    over: dict = {}
    tokens = _arg("--tokens", None, int)
    if tokens is not None:
        over["num_tokens"] = [tokens]
    dist = _arg("--dist", None)
    if dist is not None:
        over["routing"] = dataclasses.replace(
            cfg.routing, distribution=Distribution(dist))
    iters = _arg("--iters", None, int)
    if iters is not None:
        over["bench_iters"] = iters
    if over:
        print(f"[tune] config={os.path.basename(CONFIG)} overrides={over}")
    cfg = dataclasses.replace(cfg, **over) if over else cfg

    if cfg.parallel_mode != ParallelMode.TP:
        print("[tune] 只支持 TP(EP 的 AllGather 语义与 expert_map 不同)", file=sys.stderr)
        return 2
    if not cfg.distributed:
        print("[tune] 主配置需 distributed=true(per-rank 输入种子分流)", file=sys.stderr)
        return 2

    tokens = cfg.num_tokens[0]
    keep = _arg("--keep", 15, int)
    start_at = _arg("--start", 0, int)
    tol = _arg("--tol", 2e-2, float)
    stage1_iters, stage1_warmup = 5, 2
    m_full = tokens * cfg.world_size

    if "--dry-run" in sys.argv:
        cands = _candidates(cfg)
        n = cfg.intermediate_shard
        print(f"[tune] M={m_full} (= {cfg.world_size} rank × {tokens} tok), "
              f"E={cfg.num_local_experts}, N(key)={n}")
        print(f"[tune] GEMM1 N={2 * n} K={cfg.hidden_size} | "
              f"GEMM2 N={cfg.hidden_size} K={n} (两颗共用同一 config)")
        print(f"[tune] 候选数={len(cands)}  precision={cfg.precision.value} "
              f"block_shape={cfg.block_shape}")
        print("[tune] 目标文件名需在 GPU 上取 device_name 后确定")
        return 0

    import torch

    from vllm.model_executor.layers.fused_moe import override_config
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_config_file_name,
        get_default_config,
    )

    from moe_bench.baseline import NaiveFusedExperts

    out_dir = _arg("--out", os.path.join(
        ROOT, "tp_test_results", "triton_tune_" + time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(out_dir, exist_ok=True)

    # 空闲量要在自己分配权重之前读,否则读到的是扣掉自己那份的数。
    free_gb, total_gb = (x / 2**30 for x in torch.cuda.mem_get_info())
    print(f"[tune] device={torch.cuda.get_device_name(0)}  "
          f"显存 {free_gb:.1f}/{total_gb:.1f} GiB 空闲 —— 占用异常说明卡上有别的任务")

    problem = _build_problem(cfg, tokens)
    impl = NaiveFusedExperts()
    run = lambda: impl.run(problem)  # noqa: E731

    e_local, _, n_key = problem.w2.shape
    # dtype 串与 vLLM 的 _get_config_dtype_str 一致:fp8 w8a8 -> "fp8_w8a8",
    # bf16/fp16 -> None(文件名不带 dtype 段)。
    dtype_str = "fp8_w8a8" if cfg.precision == Precision.FP8 else None
    block_shape = cfg.block_shape if cfg.precision == Precision.FP8 else None
    json_name = get_config_file_name(e_local, n_key, dtype_str, block_shape)

    default_cfg = get_default_config(
        m_full, e_local, n_key, cfg.hidden_size, cfg.topk, dtype_str, block_shape)
    cands = _candidates(cfg)

    print(f"[tune] M={m_full} (= {cfg.world_size} rank × {tokens} tok) "
          f"E={e_local} N(key)={n_key} dist={cfg.routing.distribution.value}")
    print(f"[tune] GEMM1 N={2 * n_key} K={cfg.hidden_size} | "
          f"GEMM2 N={cfg.hidden_size} K={n_key} (两颗共用同一 config)")
    print(f"[tune] 默认(兜底)config = {default_cfg}")
    print(f"[tune] 候选 {len(cands)} 个, 起点 {start_at}, 输出 {out_dir}")

    # 基准 + 正确性参考:默认 config 的输出。tile 换法只改累加顺序,rel_err
    # 应在 1e-3 量级;超 --tol 说明该 config 真算错了(而不是数值噪声)。
    with override_config(default_cfg):
        ref = run().float()  # .float() 必复制(输出是 bf16), 可安全跨迭代持有
        base_us = _bench(run, cfg.bench_iters, cfg.warmup_iters)
    ref_norm = ref.norm()
    print(f"[tune] 默认 config 基准 {base_us:.1f}µs")

    rows, results = [], []
    n_skip = 0
    for i, cand in enumerate(cands):
        if i < start_at:
            continue
        status, us, rel = "ok", float("inf"), float("nan")
        try:
            with override_config(cand):
                out = run()
                torch.cuda.synchronize()
                rel = float((out.float() - ref).norm() / ref_norm)
                if rel > tol:
                    status = "wrong"
                else:
                    us = _bench(run, stage1_iters, stage1_warmup)
        except Exception as exc:  # noqa: BLE001
            kind = _classify(exc)
            if kind == "fatal":
                print(f"\n[tune] 致命错误 @ 候选 {i} {cand}\n  {type(exc).__name__}: {exc}")
                print(f"[tune] CUDA 上下文已不可信, 结果作废。排除该项后续跑:"
                      f" --start {i + 1} --out {out_dir}")
                _dump_scan(out_dir, rows)
                # 硬退出(不 synchronize / 不析构 context), 对齐 AGENTS.md 的
                # fail-fast 红线。os._exit 不 flush, 先手动 flush。
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(3)
            status, n_skip = "skip", n_skip + 1
        rows.append({"index": i, **cand, "stage1_us": us,
                     "rel_err": rel, "status": status})
        if status == "ok":
            results.append((us, cand))
        if (i + 1) % 25 == 0 or i == len(cands) - 1:
            best = min((us for us, _ in results), default=float("nan"))
            print(f"[tune] 阶段1 {i + 1}/{len(cands)}  可用 {len(results)}  "
                  f"跳过 {n_skip}  当前最优 {best:.1f}µs")

    _dump_scan(out_dir, rows)
    if not results:
        print("[tune] 没有任何候选跑通", file=sys.stderr)
        return 1

    # 阶段2:top-keep 用主配置的 warmup/iters 精测,消掉粗扫噪声。
    # 按耗时排序时必须给 key —— 元素是 (us, dict), 耗时打平会去比 dict 而报错。
    results.sort(key=lambda x: x[0])
    finals = []
    print(f"\n[tune] 阶段2 精测 top-{min(keep, len(results))} "
          f"(warmup={cfg.warmup_iters} iters={cfg.bench_iters})")
    for us1, cand in results[:keep]:
        try:
            with override_config(cand):
                us = _bench(run, cfg.bench_iters, cfg.warmup_iters)
        except Exception as exc:  # noqa: BLE001  阶段1 已跑通, 这里失败只会是偶发
            print(f"  [skip] {type(exc).__name__} {cand}")
            continue
        finals.append((us, cand))
        print(f"  {us:8.1f}µs (粗扫 {us1:7.1f})  {cand}")
    if not finals:
        print("[tune] 阶段2 全部失败", file=sys.stderr)
        return 1
    finals.sort(key=lambda x: x[0])
    best_us, best = finals[0]

    payload = {str(m_full): best, "triton_version": _triton_version()}
    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out_dir, "tune_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_config": os.path.relpath(CONFIG, ROOT),
            "overrides": {k: str(v) for k, v in over.items()},
            "device": torch.cuda.get_device_name(0),
            "M": m_full, "E": e_local, "N_key": n_key,
            "distribution": cfg.routing.distribution.value,
            "default_config": default_cfg, "default_us": base_us,
            "best_config": best, "best_us": best_us,
            "speedup": base_us / best_us,
            "num_candidates": len(cands), "num_ok": len(results),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[tune] 默认 {base_us:8.1f}µs  {default_cfg}")
    print(f"[tune] 最优 {best_us:8.1f}µs  {best}")
    print(f"[tune] 加速 {base_us / best_us:.3f}x ({base_us - best_us:+.1f}µs)")
    print(f"[tune] 写出 {json_path}")
    print(f"[tune] 接回 e2e:\n"
          f"  VLLM_TUNED_CONFIG_FOLDER={out_dir} \\\n"
          f"  python -m moe_bench.tools.run_tktp --scheme serial --no-verify --iters 50")
    print("[tune] 注意 e2e 收益 < GEMM 收益(serial 还有 AG/RS 和 act 量化 kernel);"
          "另一路由分布要另调一份, 见 docs/09")
    return 0


def _dump_scan(out_dir: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = os.path.join(out_dir, "scan.csv")
    cols = list(rows[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def _triton_version() -> str:
    try:
        import triton

        return triton.__version__
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
