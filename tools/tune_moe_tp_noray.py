#!/usr/bin/env python
"""TP-T3 v2:vLLM triton fused_moe 的无 ray 本机调优器(docs/28)。

第八轮取证:benchmark_moe.py --tune 的 ray 在本机(256 核共享机)卡死在
CoreWorker RegisterClient 2.5h,被 run_step 的 timeout SIGTERM 掉,一个
trial 都没跑。本脚本不用 ray:

  - 编排进程按 GPU 数把 config 空间切片,每片一个子进程
    (CUDA_VISIBLE_DEVICES 各钉一张卡),纯 subprocess,零外部依赖;
  - worker 直接按 serial 基线的口径调 vllm 的 fused_experts
    (bf16、E 个全量 expert、N=768 分片 intermediate、topk=8 全命中),
    monkeypatch try_get_optimal_moe_config 注入候选 config;
  - patch 有效性在跑之前先自证(计数器 + 两个极端 config 的耗时差),
    失效立刻退出并打印诊断(vllm 版本/模块路径/dir),避免整轮白跑;
  - 产物与 vllm 官方 tune 同格式:E=<E>,N=768,device_name=<dev>.json,
    装入 vllm configs 目录后 serial 基线自动变快(验证点:serial 日志
    不再出现 "Using default MoE config")。

用法(编排模式,默认 4 卡并行,单个 E 约 30~40 分钟):
  python tune_moe_tp_noray.py --gpus 9,11,13,15 --num-experts 64
  python tune_moe_tp_noray.py --gpus 9 --smoke          # <2min 自检
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

# TP 形状(docs/23):fused_experts 见到全部 E 个 expert,intermediate 分片 768,
# batch = world*T,M∈{1024,2048,4096} 是真实档,512/8192 是余量。
HIDDEN = 4096
INTER_SHARD = 768
TOPK = 8
BATCHES = [512, 1024, 2048, 4096, 8192]

# sm120 单 block smem 上限(RTX Pro 5000 Blackwell 实测 99KB optin)。
SMEM_LIMIT = 99 * 1024


def config_space() -> list[dict]:
    """与 vllm benchmark_moe.py get_configs_compute_bound 相同的搜索空间,
    外加 smem 预过滤(A/B tile 双缓冲估算超限的直接跳过,省掉必败的编译)。"""
    space = []
    for bm, bn, bk, gs, nw, ns in itertools.product(
        [16, 32, 64, 128, 256],
        [32, 64, 128, 256],
        [64, 128, 256],
        [1, 16, 32, 64],
        [4, 8],
        [2, 3, 4, 5],
    ):
        smem_est = (bm * bk + bk * bn) * 2 * ns  # bf16 A tile + B tile × stages
        if smem_est > SMEM_LIMIT:
            continue
        space.append(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": gs,
                "num_warps": nw,
                "num_stages": ns,
            }
        )
    return space


# --------------------------------------------------------------------------
# worker:单 GPU 上跑自己那片 config
# --------------------------------------------------------------------------


def _diag_dump(fm) -> None:
    import vllm

    print("[diag] vllm =", getattr(vllm, "__version__", "?"), file=sys.stderr)
    print("[diag] fused_moe =", fm.__file__, file=sys.stderr)
    names = [n for n in dir(fm) if "config" in n.lower() or "fused" in n.lower()]
    print("[diag] candidates =", names, file=sys.stderr)


class ConfigPatcher:
    """把候选 config 注进 fused_experts 的查表路径。

    首选 monkeypatch try_get_optimal_moe_config(该名字在 vllm 里多年稳定,
    fused_experts 通过模块全局引用它,patch 模块属性即生效);没有该符号时
    退化为"写 config 文件 + get_moe_configs.cache_clear()"。两条路都带
    调用计数,由 verify() 用行为差异做最终裁决。
    """

    def __init__(self, fm):
        self.fm = fm
        self.current: dict | None = None
        self.calls = 0
        if hasattr(fm, "try_get_optimal_moe_config"):
            self.mode = "monkeypatch"
            self._orig = fm.try_get_optimal_moe_config

            def patched(*args, **kwargs):
                self.calls += 1
                if self.current is not None:
                    return dict(self.current)
                return self._orig(*args, **kwargs)

            fm.try_get_optimal_moe_config = patched
        elif hasattr(fm, "get_moe_configs") and hasattr(
            fm.get_moe_configs, "cache_clear"
        ):
            self.mode = "config-file"
            self._cfg_path = None  # set 时写文件
        else:
            _diag_dump(fm)
            raise RuntimeError(
                "vllm fused_moe 里既没有 try_get_optimal_moe_config 也没有可清缓存的 "
                "get_moe_configs,无法注入 config,见上方 [diag]"
            )

    def set(self, cfg: dict | None, num_experts: int) -> None:
        self.current = cfg
        if self.mode == "config-file":
            path = os.path.join(
                os.path.dirname(self.fm.__file__), "configs", _config_filename(self.fm, num_experts)
            )
            if cfg is None:
                if os.path.exists(path):
                    os.remove(path)
            else:
                with open(path, "w") as f:
                    json.dump({"1": cfg}, f)  # 单键,任意 M 就近命中
            self.fm.get_moe_configs.cache_clear()
            self._cfg_path = path
            self.calls += 1


def _config_filename(fm, num_experts: int) -> str:
    try:
        return fm.get_config_file_name(num_experts, INTER_SHARD, None)
    except Exception:
        pass
    try:
        from vllm.platforms import current_platform

        dev = current_platform.get_device_name()
    except Exception:
        import torch

        dev = torch.cuda.get_device_name(0)
    return f"E={num_experts},N={INTER_SHARD},device_name={dev.replace(' ', '_')}.json"


def _build_inputs(torch, num_experts: int):
    """按 serial(schemes.py SerialNaive)的口径造输入:bf16 全量 E、
    分片 N、topk=8 balanced 随机路由。每个 M 一套,预分配复用。"""
    dev = "cuda"
    dt = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(0)
    w1 = torch.randn(num_experts, 2 * INTER_SHARD, HIDDEN, device=dev, dtype=dt, generator=g) / 32
    w2 = torch.randn(num_experts, HIDDEN, INTER_SHARD, device=dev, dtype=dt, generator=g) / 32
    per_m = {}
    for m in BATCHES:
        hidden = torch.randn(m, HIDDEN, device=dev, dtype=dt, generator=g)
        scores = torch.rand(m, num_experts, device=dev, generator=g)
        topk_ids = torch.topk(scores, TOPK, dim=-1).indices.to(torch.int32)
        logits = torch.randn(m, TOPK, device=dev, generator=g)
        topk_weights = torch.softmax(logits, dim=-1).float()
        per_m[m] = (hidden, topk_ids, topk_weights)
    return w1, w2, per_m


def _make_call(fm, w1, w2, num_experts):
    """按当前 vllm 版本的 fused_experts 签名组 kwargs(能带的都带上,
    与 serial 调用一致;quant_config bf16 为 None,缺省即可)。"""
    import inspect

    sig = inspect.signature(fm.fused_experts)
    base = {"w1": w1, "w2": w2}
    if "global_num_experts" in sig.parameters:
        base["global_num_experts"] = num_experts
    if "expert_map" in sig.parameters:
        base["expert_map"] = None
    if "quant_config" in sig.parameters:
        base["quant_config"] = None  # bf16 与 serial 一致(data.py: bf16 为 None)

    def call(hidden, topk_ids, topk_weights):
        return fm.fused_experts(
            hidden_states=hidden,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            **base,
        )

    return call


def _bench(torch, call, inputs, iters=10, warmup=3) -> float:
    hidden, topk_ids, topk_weights = inputs
    for _ in range(warmup):
        call(hidden, topk_ids, topk_weights)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        call(hidden, topk_ids, topk_weights)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


def worker(args) -> None:
    import torch
    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    torch.manual_seed(0)
    patcher = ConfigPatcher(fm)
    if patcher.mode == "config-file" and args.num_shards > 1:
        raise RuntimeError(
            "回退到了 config-file 注入模式,多 shard 会互踩同一个 config 文件;"
            "请用单卡重跑:--gpus <一张卡>"
        )
    w1, w2, per_m = _build_inputs(torch, args.num_experts)
    call = _make_call(fm, w1, w2, args.num_experts)
    print(
        f"[worker {args.shard}] dev={torch.cuda.get_device_name(0)} "
        f"patch={patcher.mode} file={_config_filename(fm, args.num_experts)}",
        flush=True,
    )

    # -- patch 有效性自证:极端 config 行为差 + 调用计数 --
    m_probe = 2048
    slow_cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    fast_cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3}
    patcher.set(None, args.num_experts)
    t_default = _bench(torch, call, per_m[m_probe])
    patcher.set(slow_cfg, args.num_experts)
    t_slow = _bench(torch, call, per_m[m_probe])
    patcher.set(fast_cfg, args.num_experts)
    t_fast = _bench(torch, call, per_m[m_probe])
    out = call(*per_m[m_probe])
    if patcher.calls == 0 or torch.isnan(out).any():
        _diag_dump(fm)
        raise RuntimeError(
            f"patch 无效或输出 NaN:calls={patcher.calls} nan={torch.isnan(out).any()}"
        )
    spread = abs(t_slow - t_fast) / max(min(t_slow, t_fast), 1e-6)
    print(
        f"[worker {args.shard}] self-check M={m_probe}: default={t_default:.1f}us "
        f"slow={t_slow:.1f}us fast={t_fast:.1f}us spread={spread:.1%} "
        f"calls={patcher.calls}",
        flush=True,
    )
    if spread < 0.02:
        _diag_dump(fm)
        raise RuntimeError(
            f"极端 config 耗时差仅 {spread:.1%},注入疑似未生效(阈值 2%)"
        )
    if args.smoke:
        print(f"[worker {args.shard}] smoke OK", flush=True)
        json.dump({"smoke": "ok"}, open(args.out, "w"))
        return

    # -- 正式扫描:本 shard 的 config 片,config 为外层(摊薄编译),M 为内层 --
    space = config_space()[args.shard :: args.num_shards]
    default_us = {}
    patcher.set(None, args.num_experts)
    for m in BATCHES:
        default_us[str(m)] = _bench(torch, call, per_m[m])
    best: dict[str, dict] = {}
    n_fail = 0
    for i, cfg in enumerate(space):
        patcher.set(cfg, args.num_experts)
        try:
            for m in BATCHES:
                us = _bench(torch, call, per_m[m], iters=8, warmup=2)
                k = str(m)
                if k not in best or us < best[k]["us"]:
                    best[k] = {"us": us, "config": cfg}
        except Exception:
            n_fail += 1  # OutOfResources / 编译失败等,正常淘汰
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        if (i + 1) % 25 == 0:
            print(
                f"[worker {args.shard}] {i + 1}/{len(space)} fail={n_fail} "
                f"best@2048={best.get('2048', {}).get('us', float('nan')):.1f}us",
                flush=True,
            )
    patcher.set(None, args.num_experts)  # config-file 模式下清掉试探文件
    json.dump(
        {"best": best, "default_us": default_us, "n_fail": n_fail,
         "n_tried": len(space), "shard": args.shard},
        open(args.out, "w"), indent=1,
    )
    print(f"[worker {args.shard}] done: {len(space)} tried, {n_fail} failed", flush=True)


# --------------------------------------------------------------------------
# orchestrator:切片 -> 子进程 -> 合并 -> 装配 config
# --------------------------------------------------------------------------


def orchestrate(args) -> int:
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if args.smoke:
        gpus = gpus[:1]
    workdir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "build_tune", f"tune_tp_noray_E{args.num_experts}",
    )
    os.makedirs(workdir, exist_ok=True)
    procs = []
    for i, g in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = g
        out = os.path.join(workdir, f"shard_{i}.json")
        if os.path.exists(out):
            os.remove(out)
        log = open(os.path.join(workdir, f"shard_{i}.log"), "w")
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker", "--shard", str(i), "--num-shards", str(len(gpus)),
            "--num-experts", str(args.num_experts), "--out", out,
        ]
        if args.smoke:
            cmd.append("--smoke")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), out, log))
        print(f"[tune-tp] shard {i} -> GPU {g} (log: {log.name})", flush=True)

    failed = False
    for i, (p, out, log) in enumerate(procs):
        rc = p.wait()
        log.close()
        if rc != 0 or not os.path.exists(out):
            failed = True
            print(f"[tune-tp] shard {i} FAILED rc={rc},log 尾部:", file=sys.stderr)
            with open(log.name) as f:
                sys.stderr.write("".join(f.readlines()[-30:]))
    if failed:
        return 1
    if args.smoke:
        print("[tune-tp] smoke OK(patch 生效,口径可跑)")
        return 0

    # 合并各 shard,按 M 取最优
    merged: dict[str, dict] = {}
    default_us: dict[str, float] = {}
    for _, out, _ in procs:
        d = json.load(open(out))
        default_us = default_us or d["default_us"]
        for k, v in d["best"].items():
            if k not in merged or v["us"] < merged[k]["us"]:
                merged[k] = v
    final = {k: merged[k]["config"] for k in sorted(merged, key=int)}

    # 定位安装目录与文件名(用与 worker 相同的 vllm)
    probe = subprocess.run(
        [sys.executable, "-c",
         "import os, vllm.model_executor.layers.fused_moe.fused_moe as fm;"
         "print(os.path.join(os.path.dirname(fm.__file__), 'configs'))"],
        capture_output=True, text=True,
    )
    cfg_dir = probe.stdout.strip().splitlines()[-1]
    fname_probe = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--print-filename",
         "--num-experts", str(args.num_experts)],
        capture_output=True, text=True,
    )
    fname = fname_probe.stdout.strip().splitlines()[-1]
    for dst_dir in (cfg_dir, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_tune")):
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, fname)
        with open(dst, "w") as f:
            json.dump(final, f, indent=1)
        print(f"[tune-tp] 写入 {dst}")

    print(f"\n[tune-tp] E={args.num_experts} 结果(us):")
    print(f"  {'M':>6} {'default':>9} {'tuned':>9} {'gain':>7}")
    for k in sorted(merged, key=int):
        d0, d1 = default_us.get(k, float("nan")), merged[k]["us"]
        gain = (d0 - d1) / d0 if d0 == d0 else float("nan")
        print(f"  {k:>6} {d0:>9.1f} {d1:>9.1f} {gain:>6.1%}  {merged[k]['config']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=os.environ.get("MB_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "9,11,13,15")))
    ap.add_argument("--num-experts", type=int, default=64)
    ap.add_argument("--smoke", action="store_true", help="单卡 2 config 自检,<2min")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--print-filename", action="store_true")
    args = ap.parse_args()
    if args.print_filename:
        import vllm.model_executor.layers.fused_moe.fused_moe as fm

        print(_config_filename(fm, args.num_experts))
        return 0
    if args.worker:
        worker(args)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
