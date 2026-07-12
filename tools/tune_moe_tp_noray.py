#!/usr/bin/env python
"""TP-T3 v2.1:vLLM triton fused_moe 的无 ray 本机调优器(docs/28/29)。

第八轮教训(ray 卡死)见 docs/28;第九轮教训(docs/29):v2 直接按
"key=token 数 M"写 config 文件,e2e 只有 T=1024 兑现收益,T=256/512 反而
变差——查表键与真实 lookup 的 M 可能不一致(如按 M×topk),且 648 选 1
的小 iters 初扫有赢家诅咒。v2.1 加 finalize 终审阶段,不再盲写:

  1. 初扫(4 卡分片,iters=8):每 M 留 top-4 入围,不定终名次;
  2. finalize(单卡):
     a. 删掉旧 config 文件,记录真实 lookup:用 M=1000 探针跑一次
        fused_experts,从 try_get_optimal_moe_config 收到的实参里找
        1000(键=M)还是 8000(键=M×topk),自校准写键映射;
     b. 记录各 M 的 vllm 默认 config(文件缺失时 orig 的返回值);
     c. 入围 config + 默认 config 复审(iters=30),消赢家诅咒;
        无净增益的 M 档显式写入默认 config(钉死行为,防近邻键污染);
     d. 按校准后的键写文件,装入 vllm configs 目录;
     e. 端到端自证:走真实查表路径(不打 patch)确认每个 M 选中的
        config 与预期一致、耗时与复审值吻合,打印 PASS/FAIL。

用法(编排模式,默认 4 卡并行,单个 E 约 15~20 分钟):
  python tune_moe_tp_noray.py --gpus 8,10,12,14 --num-experts 64
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
PROBE_M = 1000  # lookup 键校准探针:1000/8000 与形状常数(768/1536/4096/8)无碰撞

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


def _cfg_key(cfg: dict) -> tuple:
    return tuple(sorted(cfg.items()))


# --------------------------------------------------------------------------
# 注入与记录
# --------------------------------------------------------------------------


def _diag_dump(fm) -> None:
    import vllm

    print("[diag] vllm =", getattr(vllm, "__version__", "?"), file=sys.stderr)
    print("[diag] fused_moe =", fm.__file__, file=sys.stderr)
    names = [n for n in dir(fm) if "config" in n.lower() or "fused" in n.lower()]
    print("[diag] candidates =", names, file=sys.stderr)


class ConfigPatcher:
    """把候选 config 注进 fused_experts 的查表路径,并记录真实 lookup。

    monkeypatch try_get_optimal_moe_config(该名字在 vllm 里多年稳定,
    fused_experts 通过模块全局引用它,patch 模块属性即生效):
    - current 非 None 时返回候选 config;
    - current 为 None 时透传 orig,并记录实参里的 int(校准键映射)与
      orig 的返回值(即 vllm 实际选中的 config)。
    """

    def __init__(self, fm):
        self.fm = fm
        self.current: dict | None = None
        self.calls = 0
        self.last_ints: list[int] = []
        self.last_result: dict | None = None
        if not hasattr(fm, "try_get_optimal_moe_config"):
            _diag_dump(fm)
            raise RuntimeError(
                "vllm fused_moe 里没有 try_get_optimal_moe_config,无法注入/记录,"
                "见上方 [diag]"
            )
        self._orig = fm.try_get_optimal_moe_config

        def patched(*args, **kwargs):
            self.calls += 1
            if self.current is not None:
                return dict(self.current)
            ints = []
            for a in list(args) + list(kwargs.values()):
                if isinstance(a, bool):
                    continue
                if isinstance(a, int):
                    ints.append(a)
                elif hasattr(a, "__iter__") and not isinstance(a, (str, dict)):
                    ints.extend(x for x in a if isinstance(x, int))
            self.last_ints = ints
            res = self._orig(*args, **kwargs)
            if isinstance(res, dict):
                self.last_result = dict(res)
            return res

        fm.try_get_optimal_moe_config = patched


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


def _configs_dir(fm) -> str:
    return os.path.join(os.path.dirname(fm.__file__), "configs")


def _clear_moe_config_cache(fm) -> None:
    if hasattr(fm, "get_moe_configs") and hasattr(fm.get_moe_configs, "cache_clear"):
        fm.get_moe_configs.cache_clear()


# --------------------------------------------------------------------------
# 输入与计时(口径对齐 schemes.py SerialNaive)
# --------------------------------------------------------------------------


def _build_weights(torch, num_experts: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randn(num_experts, 2 * INTER_SHARD, HIDDEN, device="cuda",
                     dtype=torch.bfloat16, generator=g) / 32
    w2 = torch.randn(num_experts, HIDDEN, INTER_SHARD, device="cuda",
                     dtype=torch.bfloat16, generator=g) / 32
    return w1, w2, g


def _build_batch(torch, g, num_experts: int, m: int):
    hidden = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    scores = torch.rand(m, num_experts, device="cuda", generator=g)
    topk_ids = torch.topk(scores, TOPK, dim=-1).indices.to(torch.int32)
    logits = torch.randn(m, TOPK, device="cuda", generator=g)
    topk_weights = torch.softmax(logits, dim=-1).float()
    return hidden, topk_ids, topk_weights


def _make_call(fm, w1, w2, num_experts):
    """按当前 vllm 版本的 fused_experts 签名组 kwargs(与 serial 调用一致;
    quant_config bf16 为 None)。"""
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


def _self_check(torch, patcher, call, per_m, num_experts: int, shard: int) -> None:
    """开跑前自证注入有效:调用计数 + 极端 config 耗时差(阈值 2%)。"""
    m_probe = 2048
    slow_cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    fast_cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3}
    patcher.current = None
    t_default = _bench(torch, call, per_m[m_probe])
    patcher.current = slow_cfg
    t_slow = _bench(torch, call, per_m[m_probe])
    patcher.current = fast_cfg
    t_fast = _bench(torch, call, per_m[m_probe])
    patcher.current = None
    out = call(*per_m[m_probe])
    if patcher.calls == 0 or torch.isnan(out).any():
        _diag_dump(patcher.fm)
        raise RuntimeError(
            f"patch 无效或输出 NaN:calls={patcher.calls} nan={torch.isnan(out).any()}"
        )
    spread = abs(t_slow - t_fast) / max(min(t_slow, t_fast), 1e-6)
    print(
        f"[worker {shard}] self-check M={m_probe}: default={t_default:.1f}us "
        f"slow={t_slow:.1f}us fast={t_fast:.1f}us spread={spread:.1%} "
        f"calls={patcher.calls}",
        flush=True,
    )
    if spread < 0.02:
        _diag_dump(patcher.fm)
        raise RuntimeError(
            f"极端 config 耗时差仅 {spread:.1%},注入疑似未生效(阈值 2%)"
        )


# --------------------------------------------------------------------------
# worker:单 GPU 初扫自己那片 config,每 M 留 top-4
# --------------------------------------------------------------------------


def worker(args) -> None:
    import torch

    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    torch.manual_seed(0)
    patcher = ConfigPatcher(fm)
    w1, w2, g = _build_weights(torch, args.num_experts)
    per_m = {m: _build_batch(torch, g, args.num_experts, m) for m in BATCHES}
    call = _make_call(fm, w1, w2, args.num_experts)
    print(
        f"[worker {args.shard}] dev={torch.cuda.get_device_name(0)} "
        f"file={_config_filename(fm, args.num_experts)}",
        flush=True,
    )
    _self_check(torch, patcher, call, per_m, args.num_experts, args.shard)
    if args.smoke:
        print(f"[worker {args.shard}] smoke OK", flush=True)
        json.dump({"smoke": "ok"}, open(args.out, "w"))
        return

    # 初扫:本 shard 的 config 片,config 为外层(摊薄编译),M 为内层;
    # 每 M 维护 top-4 入围名单,终名次交给 finalize 复审。
    space = config_space()[args.shard :: args.num_shards]
    tops: dict[str, list] = {str(m): [] for m in BATCHES}
    n_fail = 0
    for i, cfg in enumerate(space):
        patcher.current = cfg
        try:
            for m in BATCHES:
                us = _bench(torch, call, per_m[m], iters=8, warmup=2)
                lst = tops[str(m)]
                lst.append({"us": us, "config": cfg})
                lst.sort(key=lambda r: r["us"])
                del lst[4:]
        except Exception:
            n_fail += 1  # OutOfResources / 编译失败等,正常淘汰
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        if (i + 1) % 25 == 0:
            b = tops["2048"][0]["us"] if tops["2048"] else float("nan")
            print(
                f"[worker {args.shard}] {i + 1}/{len(space)} fail={n_fail} "
                f"best@2048={b:.1f}us",
                flush=True,
            )
    patcher.current = None
    json.dump(
        {"top": tops, "n_fail": n_fail, "n_tried": len(space), "shard": args.shard},
        open(args.out, "w"), indent=1,
    )
    print(f"[worker {args.shard}] done: {len(space)} tried, {n_fail} failed", flush=True)


# --------------------------------------------------------------------------
# finalize:单 GPU 终审 + 键校准 + 写文件 + 端到端自证
# --------------------------------------------------------------------------


def finalize(args) -> None:
    import torch

    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    torch.manual_seed(0)
    patcher = ConfigPatcher(fm)
    fname = _config_filename(fm, args.num_experts)
    cfg_path = os.path.join(_configs_dir(fm), fname)
    # 旧文件先删掉:a) 默认 config 记录需要走"文件缺失"路径;b) 本轮要重写
    if os.path.exists(cfg_path):
        os.remove(cfg_path)
        print(f"[finalize] 移除旧 config: {cfg_path}", flush=True)
    _clear_moe_config_cache(fm)

    w1, w2, g = _build_weights(torch, args.num_experts)
    per_m = {m: _build_batch(torch, g, args.num_experts, m) for m in BATCHES}
    call = _make_call(fm, w1, w2, args.num_experts)

    # a) 键映射校准:M=1000 探针,看 lookup 实参里是 1000 还是 1000*topk
    probe = _build_batch(torch, g, args.num_experts, PROBE_M)
    patcher.current = None
    call(*probe)
    torch.cuda.synchronize()
    if PROBE_M in patcher.last_ints:
        key_of = lambda m: m  # noqa: E731
        mapping = "M(token 数)"
    elif PROBE_M * TOPK in patcher.last_ints:
        key_of = lambda m: m * TOPK  # noqa: E731
        mapping = f"M*topk(×{TOPK})"
    else:
        key_of = lambda m: m  # noqa: E731
        mapping = f"未识别(ints={patcher.last_ints}),按 M 兜底"
    print(f"[finalize] lookup 键映射: {mapping}", flush=True)

    # b) 各 M 的默认 config + 默认耗时(文件已删,orig 走 default 路径)
    default_cfg: dict[str, dict] = {}
    default_us: dict[str, float] = {}
    for m in BATCHES:
        patcher.current = None
        patcher.last_result = None
        us = _bench(torch, call, per_m[m], iters=args.iters, warmup=5)
        default_us[str(m)] = us
        if patcher.last_result is None:
            raise RuntimeError("未捕获到默认 config(orig 返回非 dict?)")
        default_cfg[str(m)] = patcher.last_result

    # c) 入围复审(大 iters 消赢家诅咒;无净增益档写默认 config 钉死行为)
    finalists = json.load(open(args.finalists))
    final: dict[str, dict] = {}
    report = []
    for m in BATCHES:
        k = str(m)
        seen = set()
        cands = []
        for r in finalists.get(k, []):
            ck = _cfg_key(r["config"])
            if ck not in seen:
                seen.add(ck)
                cands.append(r["config"])
        best_us, best_cfg = default_us[k], None  # None = 默认胜出
        for cfg in cands:
            patcher.current = cfg
            try:
                us = _bench(torch, call, per_m[m], iters=args.iters, warmup=5)
            except Exception:
                continue
            if us < best_us:
                best_us, best_cfg = us, cfg
        patcher.current = None
        chosen = best_cfg if best_cfg is not None else default_cfg[k]
        final[k] = chosen
        report.append((m, default_us[k], best_us, best_cfg is not None, chosen))

    # d) 按校准后的键写文件(键碰撞防御)
    keyed = {str(key_of(m)): final[str(m)] for m in BATCHES}
    if len(keyed) != len(BATCHES):
        raise RuntimeError(f"键映射产生碰撞: {sorted(keyed)}")
    backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_tune")
    os.makedirs(backup_dir, exist_ok=True)
    for dst in (cfg_path, os.path.join(backup_dir, fname)):
        with open(dst, "w") as f:
            json.dump(keyed, f, indent=1)
        print(f"[finalize] 写入 {dst}", flush=True)

    # e) 端到端自证:真实查表路径(current=None 透传 orig)选中的 config
    #    必须与预期一致,耗时与复审值吻合(±3% 为 PASS,超出打 WARN 供人裁)
    _clear_moe_config_cache(fm)
    print(f"\n[finalize] E={args.num_experts} 终表(us, iters={args.iters}):")
    print(f"  {'M':>6} {'default':>9} {'final':>9} {'gain':>7} {'e2e':>9}  verdict")
    n_bad = 0
    for m, d_us, b_us, tuned_won, chosen in report:
        k = str(m)
        patcher.current = None
        patcher.last_result = None
        e2e_us = _bench(torch, call, per_m[m], iters=args.iters, warmup=5)
        picked = patcher.last_result
        ok_cfg = picked is not None and _cfg_key(picked) == _cfg_key(chosen)
        ok_us = e2e_us <= b_us * 1.03
        verdict = "PASS" if (ok_cfg and ok_us) else "WARN"
        if not (ok_cfg and ok_us):
            n_bad += 1
        gain = (d_us - b_us) / d_us
        src = "tuned" if tuned_won else "default(钉死)"
        print(
            f"  {m:>6} {d_us:>9.1f} {b_us:>9.1f} {gain:>6.1%} {e2e_us:>9.1f}  "
            f"{verdict} [{src}] {chosen if ok_cfg else f'选中={picked} 预期={chosen}'}",
            flush=True,
        )
    if n_bad:
        raise RuntimeError(f"{n_bad} 个 M 档端到端自证未过(见上方 WARN)")
    print("[finalize] 端到端自证全部 PASS", flush=True)


# --------------------------------------------------------------------------
# orchestrator:切片 -> 初扫子进程 -> 合并入围 -> finalize 子进程
# --------------------------------------------------------------------------


def _spawn(cmd_extra: list[str], gpu: str, log_path: str) -> tuple:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log = open(log_path, "w")
    cmd = [sys.executable, os.path.abspath(__file__)] + cmd_extra
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log


def _tail(path: str, n: int = 30) -> str:
    with open(path) as f:
        return "".join(f.readlines()[-n:])


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
        out = os.path.join(workdir, f"shard_{i}.json")
        if os.path.exists(out):
            os.remove(out)
        extra = ["--worker", "--shard", str(i), "--num-shards", str(len(gpus)),
                 "--num-experts", str(args.num_experts), "--out", out]
        if args.smoke:
            extra.append("--smoke")
        p, log = _spawn(extra, g, os.path.join(workdir, f"shard_{i}.log"))
        procs.append((p, out, log))
        print(f"[tune-tp] shard {i} -> GPU {g} (log: {log.name})", flush=True)

    failed = False
    for i, (p, out, log) in enumerate(procs):
        rc = p.wait()
        log.close()
        if rc != 0 or not os.path.exists(out):
            failed = True
            print(f"[tune-tp] shard {i} FAILED rc={rc},log 尾部:", file=sys.stderr)
            sys.stderr.write(_tail(log.name))
    if failed:
        return 1
    if args.smoke:
        print("[tune-tp] smoke OK(patch 生效,口径可跑)")
        return 0

    # 合并入围名单(每 M 取全局 top-6,去重)
    finalists: dict[str, list] = {str(m): [] for m in BATCHES}
    for _, out, _ in procs:
        d = json.load(open(out))
        for k, lst in d["top"].items():
            finalists[k].extend(lst)
    for k in finalists:
        finalists[k].sort(key=lambda r: r["us"])
        del finalists[k][6:]
    fin_path = os.path.join(workdir, "finalists.json")
    json.dump(finalists, open(fin_path, "w"), indent=1)

    # finalize:单卡终审 + 键校准 + 写文件 + 自证
    p, log = _spawn(
        ["--finalize", "--finalists", fin_path,
         "--num-experts", str(args.num_experts), "--iters", str(args.iters)],
        gpus[0], os.path.join(workdir, "finalize.log"),
    )
    print(f"[tune-tp] finalize -> GPU {gpus[0]} (log: {log.name})", flush=True)
    rc = p.wait()
    log.close()
    print(_tail(log.name, 60))
    if rc != 0:
        print("[tune-tp] finalize FAILED", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=os.environ.get("MB_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "8,10,12,14")))
    ap.add_argument("--num-experts", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30, help="finalize 复审迭代数")
    ap.add_argument("--smoke", action="store_true", help="单卡自检,<2min")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--finalists", default="")
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
    if args.finalize:
        finalize(args)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
