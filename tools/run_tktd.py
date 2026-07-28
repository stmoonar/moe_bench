# SPDX-License-Identifier: Apache-2.0
"""tktd(TD 风格 TP 复刻, docs/17)的驱动脚本。

基准口径从主配置读(configs/tp_rtx_pro5000_4gpu_fp8.yaml), 命令行只覆盖最小
字段并打印覆盖项。**死锁红线第 3 条**: 新协议 kernel 首测单步隔离 —— 默认
(无 flag)只跑正确性门这一步(verify 开、短迭代), 门不过不进任何性能矩阵;
--perf/--sweep 也先跑门, 门过了才继续, 每一步各自起独立的四进程/NCCL 生命
周期(子进程), 单步失败不影响其余步骤。

  # 步 1: 正确性门(默认, 单步隔离)
  python -m moe_bench.tools.run_tktd
  # 步 2: 门 + 计时(tktd, 可加 --with tktp,serial 同口径对照)
  python -m moe_bench.tools.run_tktd --perf [--with tktp,serial]
  # 步 3: 门 + TKTD_NCHUNKS sweep
  python -m moe_bench.tools.run_tktd --sweep [--nchunks-list 2,4,8,16]

  通用覆盖: [--tokens T] [--iters N] [--dist balanced|uniform|skewed]
            [--nchunks N] [--comm-sms N] [--push-sms N] [--rs-sms N]
            [--no-verify](仅作用于 perf/sweep 计时步, 门永远 verify)
  结果落 tp_test_results/tp_run_<时间戳>/ (每步 log + json + 汇总 summary)。

必须在 /workspace 下运行(python -m moe_bench.tools.run_tktd)。
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import signal
import subprocess
import sys
import threading

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "configs", "tp_rtx_pro5000_4gpu_fp8.yaml")
# 单步兜底超时。kernel 侧协议挂死 ~40s 内 spin guard 自爆; 这里兜的是
# host 侧挂死(如某 rank 早退后其余卡死在 NCCL 集合通信里)。首跑含 nvcc
# 重编时可用 RUN_TKTD_TIMEOUT 放宽。
STEP_TIMEOUT_S = int(os.environ.get("RUN_TKTD_TIMEOUT", "1200"))


def _arg(flag, default, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def _one() -> None:
    """子进程单步入口: 读 YAML + 最小覆盖, 跑一个 scheme。env 旋钮
    (TKTD_*)由父进程设置并继承。"""
    from moe_bench.config import Distribution, MoEBenchConfig, RoutingConfig

    cfg = MoEBenchConfig.from_file(CONFIG)
    scheme = _arg("--_one", "tktd")
    over = {}
    tokens = _arg("--tokens", None, int)
    if tokens is not None:
        over["num_tokens"] = [tokens]
    iters = _arg("--iters", None, int)
    if iters is not None:
        over["bench_iters"] = iters
    warmup = _arg("--warmup", None, int)
    if warmup is not None:
        over["warmup_iters"] = warmup
    if "--no-verify" in sys.argv:
        over["verify"] = False
    json_out = _arg("--json", None)
    if json_out is not None:
        over["output_json"] = json_out
    dist = _arg("--dist", None)
    if dist is not None:
        over["routing"] = RoutingConfig(
            distribution=Distribution(dist),
            skew_alpha=cfg.routing.skew_alpha,
            num_active_experts=cfg.routing.num_active_experts)
    if over:
        print(f"[run_tktd] config={os.path.basename(CONFIG)} scheme={scheme} "
              f"overrides={over}")
    knobs = {k: v for k, v in sorted(os.environ.items()) if k.startswith("TKTD_")}
    print(f"[run_tktd] TKTD env: {knobs}")
    cfg = dataclasses.replace(cfg, **over) if over else cfg
    from moe_bench.distributed import run_distributed
    run_distributed(cfg, scheme)


def _run_step(name: str, outdir: str, scheme: str, extra_args: list[str],
              env_over: dict[str, str], require_verify: bool) -> tuple[bool, dict | None]:
    """独立子进程跑一步(独立 NCCL 生命周期), log/json 落盘, 返回
    (是否通过, json payload)。require_verify 时任一 token 档 verify!=ok 判败。"""
    log_path = os.path.join(outdir, f"{name}.log")
    json_path = os.path.join(outdir, f"{name}.json")
    cmd = [sys.executable, "-m", "moe_bench.tools.run_tktd",
           "--_one", scheme, "--json", json_path, *extra_args]
    env = dict(os.environ)
    env.update(env_over)
    print(f"[run_tktd] step '{name}': scheme={scheme} args={extra_args} "
          f"env_over={env_over}")
    with open(log_path, "w", encoding="utf-8") as lf:
        # start_new_session: 子进程自成进程组 —— 超时/收割时 killpg 连
        # mp.spawn 的 4 个 worker 一起带走(否则残留 worker 的 NCCL/自旋
        # kernel 会把 GPU 顶在 100%, 2026-07-28 实测踩过)。
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env,
                                start_new_session=True)

        def _pump():
            for line in proc.stdout:            # 透传 + 落盘
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
                lf.flush()

        # 读管道放后台线程: 主线程的 wait(timeout) 才能真正生效(直接在主
        # 线程 for line in stdout 会在子进程挂住时永远阻塞, 超时形同虚设)。
        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        def _kill_group():
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            pump.join(timeout=5)

        try:
            ret = proc.wait(timeout=STEP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_group()
            lf.write(f"\n[run_tktd] step '{name}' TIMEOUT {STEP_TIMEOUT_S}s, "
                     f"进程组已 SIGKILL; 用 nvidia-smi 确认 util 归零"
                     f"(spin guard ~40s 内应自行 trap)再重跑\n")
            print(f"[run_tktd] step '{name}' TIMEOUT {STEP_TIMEOUT_S}s, "
                  f"进程组已 SIGKILL — 等 ~1 分钟看 nvidia-smi 是否归零")
            return False, None
        except KeyboardInterrupt:
            # Ctrl-C 只打到父进程 —— 不带走 worker 会把自旋 kernel 留在卡上
            # (GPU 100% 残留)。收割整组后再抛。
            _kill_group()
            lf.write(f"\n[run_tktd] step '{name}' 被中断, 进程组已 SIGKILL\n")
            print(f"[run_tktd] 中断: 进程组已 SIGKILL — 等 ~1 分钟看 "
                  f"nvidia-smi 是否归零后再重跑")
            raise
        pump.join(timeout=5)
    if ret != 0:
        print(f"[run_tktd] step '{name}' FAILED (exit {ret}), log: {log_path}")
        return False, None
    payload = None
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    if require_verify:
        rows = (payload or {}).get("results", [])
        ok = bool(rows) and all(r.get("verify") == "ok" for r in rows)
        if not ok:
            print(f"[run_tktd] step '{name}' verify FAIL: "
                  f"{[(r.get('num_tokens'), r.get('verify'), r.get('rel_err')) for r in rows]}")
            return False, payload
    return True, payload


def main() -> None:
    if "--_one" in sys.argv:
        _one()
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("tp_test_results", f"tp_run_{ts}")
    os.makedirs(outdir, exist_ok=True)
    print(f"[run_tktd] results dir: {outdir}")

    # 通用覆盖(透传给每个子步)
    common: list[str] = []
    for flag in ("--tokens", "--dist"):
        v = _arg(flag, None)
        if v is not None:
            common += [flag, v]
    iters = _arg("--iters", None)
    # env 旋钮 -> 子进程环境
    env_over: dict[str, str] = {}
    for flag, env_key in (("--nchunks", "TKTD_NCHUNKS"),
                          ("--comm-sms", "TKTD_COMM_SMS"),
                          ("--push-sms", "TKTD_PUSH_SMS"),
                          ("--rs-sms", "TKTD_RS_SMS")):
        v = _arg(flag, None)
        if v is not None:
            env_over[env_key] = v

    summary: list[str] = []

    def note(line: str) -> None:
        print(f"[run_tktd] {line}")
        summary.append(line)

    # ---- 步 0: 正确性门(单步隔离, 红线 3)。verify 强制开, 短迭代。 ----
    gate_args = common + ["--iters", "5", "--warmup", "5"]
    ok, payload = _run_step("gate_tktd", outdir, "tktd", gate_args, env_over,
                            require_verify=True)
    if payload:
        for r in payload.get("results", []):
            note(f"gate tktd T={r['num_tokens']}: verify={r.get('verify')} "
                 f"rel_err={r.get('rel_err'):.3e} lat={r['latency_us']:.1f}us")
    if not ok:
        note("正确性门未通过 —— 按红线不进性能矩阵。排查后重跑。")
        _write_summary(outdir, summary)
        sys.exit(1)
    note("正确性门通过。")

    if "--perf" not in sys.argv and "--sweep" not in sys.argv:
        note("(只跑门。计时用 --perf, nchunks 扫描用 --sweep)")
        _write_summary(outdir, summary)
        return

    timed_args = common + (["--iters", iters] if iters else [])
    if "--no-verify" in sys.argv:
        timed_args += ["--no-verify"]

    # ---- 步 1+: 计时(tktd 与可选对照, 每步独立进程) ----
    if "--perf" in sys.argv:
        schemes = ["tktd"] + [s for s in _arg("--with", "", str).split(",") if s]
        for s in schemes:
            ok, payload = _run_step(f"perf_{s}", outdir, s, timed_args,
                                    env_over if s == "tktd" else {},
                                    require_verify=False)
            if ok and payload:
                for r in payload.get("results", []):
                    note(f"perf {s} T={r['num_tokens']}: "
                         f"avg={r['latency_us']:.1f}us med={r['latency_med_us']:.1f}us "
                         f"verify={r.get('verify', '-')}")

    # ---- 步 2+: TKTD_NCHUNKS sweep ----
    if "--sweep" in sys.argv:
        chunks = _arg("--nchunks-list", "2,4,8,16")
        for nc in chunks.split(","):
            eo = dict(env_over)
            eo["TKTD_NCHUNKS"] = nc.strip()
            ok, payload = _run_step(f"sweep_nc{nc.strip()}", outdir, "tktd",
                                    timed_args, eo, require_verify=False)
            if ok and payload:
                for r in payload.get("results", []):
                    note(f"sweep nchunks={nc.strip()} T={r['num_tokens']}: "
                         f"avg={r['latency_us']:.1f}us med={r['latency_med_us']:.1f}us "
                         f"verify={r.get('verify', '-')}")

    _write_summary(outdir, summary)


def _write_summary(outdir: str, lines: list[str]) -> None:
    path = os.path.join(outdir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[run_tktd] summary: {path}")


if __name__ == "__main__":
    main()
