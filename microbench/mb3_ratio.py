# SPDX-License-Identifier: Apache-2.0
"""MB3 汇总分析:通信/计算占比、融合理论上限、收益来源分解。

纯后处理(不占 GPU):读取同一结果目录下 mb1_compute / mb2_comm / mb4_fusion 的
JSON,输出 mb3_ratio.json + mb3_report.md。核心问题——**tkfused 的收益里,多少
来自"TK 计算 kernel 本身比 vLLM triton 快",多少来自"通信方案 + 通算重叠"**:

  总收益        = serial_e2e - tk_e2e_fair            (tk_e2e_fair 含 schedule 成本)
  计算收益      = vllm_compute - tk_compute_fair      (tk_compute = 纯 GEMM 链 + silu
                                                       + schedule;均不含通信)
  通信+重叠收益 = 总收益 - 计算收益

理论上限:
  bound_overlap_only = max(serial_comm, vllm_compute) + 小项
      —— 不换计算算子、只把 NCCL 通信与 triton 计算完全重叠的下限
  bound_tk_full      = prep + max(L0_gemm, disp_comm) + silu
                       + max(L1_gemm, comb_comm) + sched
      —— TK 算子 + 每层通信完全被该层 GEMM 掩盖的下限
      (disp_comm 用 tk_push_data 近似 dispatch 数据面、comb_comm 用
       tk_final_reduce 近似 combine 数据面;均为实测通信 kernel)

纯 stdlib、不 import moe_bench 包(也就不 import vllm/torch),因此既可以
`python -m moe_bench.microbench.mb3_ratio`,也可以在任何机器上直接
`python mb3_ratio.py --results <dir>` 离线重跑分析。

  python moe_bench/microbench/mb3_ratio.py --results <dir>
"""
from __future__ import annotations

import argparse
import json
import os


def _default_results_dir() -> str:
    d = os.environ.get("MB_OUT")
    if not d:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "adhoc")
    return d


def _load(results_dir: str, name: str) -> dict[int, dict]:
    path = os.path.join(results_dir, f"{name}.json")
    if not os.path.exists(path):
        print(f"[mb3][warn] {path} 不存在,相关列将为空")
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {row["tokens_total"]: row for row in data["rows"]
            if "tokens_total" in row}


def analyze(results_dir: str, sched_ms: float | None = None) -> tuple[list[dict], str]:
    mb1 = _load(results_dir, "mb1_compute")
    mb2 = _load(results_dir, "mb2_comm")
    mb4 = _load(results_dir, "mb4_fusion")
    tokens = sorted(set(mb1) & set(mb2) & set(mb4))
    if not tokens:
        raise SystemExit("[mb3] mb1/mb2/mb4 结果没有共同的 token 档,无法分析")

    rows = []
    for tt in tokens:
        r1, r2, r4 = mb1[tt], mb2[tt], mb4[tt]
        serial_comm = r2["serial_comm_total_ms"]
        vllm_compute = r1["vllm_compute_ms"]
        serial_e2e = r4["serial_e2e_ms"]
        tk_e2e = r4["tk_e2e_ms"]
        # mb4 测的 sched 是 eager 版(~0.8ms);真实 run() 走 CUDA graph 后
        # ~0.2ms(docs/14)。--sched-ms 可用 graph 口径覆盖。
        sched = sched_ms if sched_ms is not None else r4["sched_ms"]
        tk_e2e_fair = tk_e2e + sched
        # tk_chain = l0 + silu + l1(mb1 里连续执行,纯计算);加 schedule 成本
        tk_compute_fair = r1["tk_chain_ms"] + sched
        total_gain = serial_e2e - tk_e2e_fair
        compute_gain = vllm_compute - tk_compute_fair
        comm_overlap_gain = total_gain - compute_gain

        bound1 = max(serial_comm, vllm_compute)
        disp_comm = r2["tk_push_data_ms"]
        comb_comm = r2["tk_final_reduce_ms"]
        bound_tk = (r4["prep_ms"] + max(r4["L0_gemm_ms"], disp_comm)
                    + r4["silu_ms"] + max(r4["L1_gemm_ms"], comb_comm) + sched)

        row = {
            "tokens_total": tt,
            # 串行分解
            "serial_e2e_ms": serial_e2e,
            "serial_comm_ms": serial_comm,
            "vllm_compute_ms": vllm_compute,
            "serial_pred_ms": serial_comm + vllm_compute,
            "serial_unaccounted_ms": serial_e2e - serial_comm - vllm_compute,
            "comm_share_of_serial": serial_comm / serial_e2e,
            # TK 分解
            "tk_e2e_ms": tk_e2e,
            "tk_sched_ms": sched,
            "tk_e2e_fair_ms": tk_e2e_fair,
            "tk_compute_fair_ms": tk_compute_fair,
            "tk_exposed_comm_ms": tk_e2e_fair - tk_compute_fair,
            # 收益分解(核心)
            "total_gain_ms": total_gain,
            "compute_gain_ms": compute_gain,
            "comm_overlap_gain_ms": comm_overlap_gain,
            "compute_gain_share": (compute_gain / total_gain
                                   if abs(total_gain) > 1e-9 else float("nan")),
            # 上限
            "bound_overlap_only_ms": bound1,
            "speedup_bound_overlap_only": serial_e2e / bound1,
            "bound_tk_full_ms": bound_tk,
            "speedup_bound_tk_full": serial_e2e / bound_tk,
            "speedup_actual": serial_e2e / tk_e2e,
            "speedup_actual_fair": serial_e2e / tk_e2e_fair,
            "headroom_vs_bound_ms": tk_e2e_fair - bound_tk,
        }
        rows.append(row)

    # ---- markdown 报告 ----
    L = []
    L.append("# MB3 通信/计算占比与收益来源分解\n")
    L.append("- `fair` 口径 = tk_e2e + schedule 重建成本(当前默认路径 "
             "prered_push 下 run() 未计入 schedule,见 README 注记)。\n")
    L.append("## 1. 串行侧:通信 vs 计算占比\n")
    L.append("| tokens | serial e2e | 通信 | 计算(vllm) | 通信占比 | 未归账 |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['tokens_total']} | {r['serial_e2e_ms']*1e3:.0f}us "
                 f"| {r['serial_comm_ms']*1e3:.0f}us "
                 f"| {r['vllm_compute_ms']*1e3:.0f}us "
                 f"| {r['comm_share_of_serial']*100:.0f}% "
                 f"| {r['serial_unaccounted_ms']*1e3:+.0f}us |")
    L.append("\n## 2. 收益来源分解(回答:收益是不是主要来自 TK 计算算子更快)\n")
    L.append("| tokens | 总收益 | 计算收益 | 通信+重叠收益 | **计算收益占比** |")
    L.append("|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['tokens_total']} | {r['total_gain_ms']*1e3:+.0f}us "
                 f"| {r['compute_gain_ms']*1e3:+.0f}us "
                 f"| {r['comm_overlap_gain_ms']*1e3:+.0f}us "
                 f"| **{r['compute_gain_share']*100:.0f}%** |")
    L.append("\n## 3. 理论上限 vs 实际\n")
    L.append("| tokens | serial | 只重叠上限(x) | TK+全重叠上限(x) "
             "| 实际 tkfused(x, fair) | 距上限 |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['tokens_total']} | {r['serial_e2e_ms']*1e3:.0f}us "
                 f"| {r['bound_overlap_only_ms']*1e3:.0f}us "
                 f"(x{r['speedup_bound_overlap_only']:.2f}) "
                 f"| {r['bound_tk_full_ms']*1e3:.0f}us "
                 f"(x{r['speedup_bound_tk_full']:.2f}) "
                 f"| {r['tk_e2e_fair_ms']*1e3:.0f}us "
                 f"(x{r['speedup_actual_fair']:.2f}) "
                 f"| {r['headroom_vs_bound_ms']*1e3:+.0f}us |")
    L.append("\n> 口径:计算收益 = vllm_compute - (tk GEMM链 + schedule);"
             "通信+重叠收益 = 总收益 - 计算收益。disp/comb 通信下限分别用 "
             "tk_push_data / tk_final_reduce 实测值近似。\n")
    report = "\n".join(L)
    return rows, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                        help="结果目录(默认 $MB_OUT)")
    parser.add_argument("--sched-ms", type=float, default=None,
                        help="覆盖 schedule 成本(ms)。mb4 测的是 eager 版"
                             "(~0.8ms);真实 run() 走 CUDA graph 后 ~0.205ms"
                             "(docs/14),建议 --sched-ms 0.205 复算公平口径。")
    args = parser.parse_args()
    results_dir = args.results or _default_results_dir()

    rows, report = analyze(results_dir, sched_ms=args.sched_ms)
    suffix = "" if args.sched_ms is None else f"_sched{int(args.sched_ms*1e3)}us"
    jpath = os.path.join(results_dir, f"mb3_ratio{suffix}.json")
    mpath = os.path.join(results_dir, f"mb3_report{suffix}.md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "sched_ms_override": args.sched_ms}, f,
                  indent=2, ensure_ascii=False)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"[saved] {jpath}")
    print(f"[saved] {mpath}")


if __name__ == "__main__":
    main()
