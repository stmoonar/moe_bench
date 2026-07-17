# SPDX-License-Identifier: Apache-2.0
"""故障注入: 运行中杀死一个 rank, 验证死锁兜底整链有界退出。

复现 2026-07-16 wedge 事故的触发形态(一个 rank 先死, 其余 rank 的持久
kernel 等待其信号/经 IPC 访问其显存), 验证兜底后的预期行为:

  1. 幸存 rank 的 kernel 在有界时间内 trap(自旋 guard / guarded_wait,
     ~32-40s) -> CUDA error -> worker 经 _fail_fast_exit 硬退出;
  2. 整个进程树在 --timeout 内消失, 无 D 态残留;
  3. nvidia-smi 在限时内正常返回(RM 锁未被抱死);
  4. 每张测试卡能新建 CUDA context 并跑一个小 kernel(卡可复用)。

用法(GPU 机器, /workspace 下; 红线第 3 条: 注入目标选单步/单 case 运行,
不要挂进长矩阵):

  python moe_bench/tools/fault_inject_kill_rank.py \\
      --kill-rank 2 --delay 10 --timeout 120 --gpus 0,1,2,3 -- \\
      python -m moe_bench.bench --config moe_bench/configs/tp_rtx_pro5000_4gpu_fp8.yaml \\
      --distributed --scheme tk_tp

注意: 该测试本身就是在人为制造一次故障, 只允许在确认所有测试卡空闲、且
接受"失败时可能需要按 docs/04 流程恢复"的前提下运行。
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def _run(cmd: list[str], timeout: float) -> tuple[int | None, str]:
    """Run a command with a hard timeout; return (returncode|None, output)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT after {timeout}s: {' '.join(cmd)}"


def _descendants(root_pid: int) -> list[int]:
    """All descendant pids of root_pid (via /proc, ascending == spawn order)."""
    children: dict[int, list[int]] = {}
    for pid_dir in os.listdir("/proc"):
        if not pid_dir.isdigit():
            continue
        try:
            with open(f"/proc/{pid_dir}/stat") as f:
                ppid = int(f.read().split(")")[-1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(pid_dir))
    out: list[int] = []
    stack = [root_pid]
    while stack:
        for child in sorted(children.get(stack.pop(), [])):
            out.append(child)
            stack.append(child)
    return sorted(out)


def _pid_on_gpu(pids: list[int], gpu_index: int) -> int | None:
    """Map the launcher's descendant that holds a context on gpu_index."""
    rc, out = _run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        timeout=15,
    )
    if rc != 0:
        return None
    rc, uuid_out = _run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], timeout=15
    )
    if rc != 0:
        return None
    index_by_uuid = {}
    for line in uuid_out.strip().splitlines():
        idx, uuid = (x.strip() for x in line.split(","))
        index_by_uuid[uuid] = int(idx)
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid_s, uuid = (x.strip() for x in line.split(","))
        if int(pid_s) in pids and index_by_uuid.get(uuid) == gpu_index:
            return int(pid_s)
    return None


def _ps_snapshot(pids: list[int]) -> str:
    if not pids:
        return "(none)"
    rc, out = _run(
        ["ps", "-o", "pid,stat,wchan:32,etime,cmd", "-p", ",".join(map(str, pids))],
        timeout=10,
    )
    return out if rc == 0 else out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kill-rank", type=int, default=0, help="要杀死的 rank(=可见卡序)")
    parser.add_argument("--delay", type=float, default=10.0, help="启动后多少秒注入")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="注入后等待整树退出的上限(须 > kernel guard ~40s)")
    parser.add_argument("--gpus", default="0,1,2,3", help="物理卡号列表, 用于事后体检")
    parser.add_argument("--python", default=sys.executable, help="事后体检用的 python")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- 之后是被测命令(单步/单 case 运行)")
    args = parser.parse_args()
    cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not cmd:
        parser.error("缺少被测命令: ... -- python -m moe_bench.bench ...")
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]

    print(f"[inject] 启动: {' '.join(cmd)}", flush=True)
    victim_target = None
    launcher = subprocess.Popen(cmd, start_new_session=True)
    try:
        time.sleep(args.delay)
        if launcher.poll() is not None:
            print(f"[inject] FAIL: 被测命令在注入前已退出(code={launcher.returncode}), "
                  "增大 --delay 或检查命令", flush=True)
            return 2
        pids = _descendants(launcher.pid)
        # 优先按 "第 kill-rank 张可见卡上的进程" 定位受害者; 失败则退回
        # spawn 顺序(mp.spawn 按 rank 升序 fork, pid 通常升序)。
        if args.kill_rank < len(gpus):
            victim_target = _pid_on_gpu([launcher.pid] + pids, gpus[args.kill_rank])
        if victim_target is None:
            workers = [p for p in pids if p != launcher.pid]
            if len(workers) <= args.kill_rank:
                print(f"[inject] FAIL: 找不到 rank {args.kill_rank} 的 worker "
                      f"(descendants={pids})", flush=True)
                return 2
            victim_target = workers[args.kill_rank]
        print(f"[inject] SIGKILL rank {args.kill_rank} (pid={victim_target})", flush=True)
        os.kill(victim_target, signal.SIGKILL)

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            survivors = [p for p in _descendants(launcher.pid) if p != victim_target]
            if launcher.poll() is not None and not survivors:
                break
            time.sleep(2)
        else:
            survivors = [p for p in _descendants(launcher.pid) if p != victim_target]
            print(f"[inject] FAIL: {args.timeout}s 后进程树未退净 —— 存在无界等待/D 态。",
                  flush=True)
            print(_ps_snapshot([launcher.pid] + survivors), flush=True)
            return 1
        elapsed = args.timeout - max(0.0, deadline - time.time())
        print(f"[inject] 进程树已在 {elapsed:.0f}s 内退净 "
              f"(launcher code={launcher.returncode})", flush=True)
    finally:
        if launcher.poll() is None:
            try:
                os.killpg(launcher.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    ok = True
    rc, out = _run(["nvidia-smi"], timeout=20)
    if rc == 0:
        print("[check] nvidia-smi 正常返回", flush=True)
    else:
        ok = False
        print(f"[check] FAIL: nvidia-smi 异常: {out[:500]}", flush=True)

    for g in gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        code = "import torch; print(torch.ones(4, device='cuda').sum().item())"
        try:
            proc = subprocess.run([args.python, "-c", code], env=env,
                                  capture_output=True, text=True, timeout=60)
            good = proc.returncode == 0
        except subprocess.TimeoutExpired:
            good, proc = False, None
        if good:
            print(f"[check] GPU {g}: 新 CUDA context + kernel 正常", flush=True)
        else:
            ok = False
            detail = proc.stderr[:300] if proc else "timeout 60s (context 创建被卡 = RM 锁征兆)"
            print(f"[check] FAIL: GPU {g}: {detail}", flush=True)

    print(f"[inject] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
