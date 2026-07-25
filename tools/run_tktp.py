# SPDX-License-Identifier: Apache-2.0
"""TP 驱动脚本:直接跑分布式 tktp / serial, 绕开 bench.py 的 argparse
(vllm 的 FlexibleArgumentParser 在这里会吃掉参数)。

基准口径**从主配置读**(configs/tp_rtx_pro5000_4gpu_fp8.yaml), 命令行只覆盖
本次实验需要的最小字段, 覆盖项会打印出来。

  python -m moe_bench.tools.run_tktp [num_experts] [--scheme tktp|serial]
      [--tokens T] [--iters N] [--no-verify] [--json out.json]
      [--dist balanced|uniform|skewed|single] [--skew-alpha A] [--active N]

不带 --no-verify 时, harness 会用 reference_moe 对拍整层输出并打印 rel_err。
"""
from __future__ import annotations

import dataclasses
import os
import sys

from moe_bench.config import Distribution, MoEBenchConfig, RoutingConfig

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "configs", "tp_rtx_pro5000_4gpu_fp8.yaml")


def _arg(flag, default, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def main():
    cfg = MoEBenchConfig.from_file(CONFIG)
    over = {}
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        over["num_experts"] = int(sys.argv[1])
    tokens = _arg("--tokens", None, int)
    if tokens is not None:
        over["num_tokens"] = [tokens]
    iters = _arg("--iters", None, int)
    if iters is not None:
        over["bench_iters"] = iters
    if "--no-verify" in sys.argv:
        over["verify"] = False
    json_out = _arg("--json", None)
    if json_out is not None:
        over["output_json"] = json_out
    dist = _arg("--dist", None)
    skew_alpha = _arg("--skew-alpha", None, float)
    active = _arg("--active", None, int)
    if dist is not None or skew_alpha is not None or active is not None:
        over["routing"] = RoutingConfig(
            distribution=Distribution(dist) if dist else cfg.routing.distribution,
            skew_alpha=skew_alpha if skew_alpha is not None else cfg.routing.skew_alpha,
            num_active_experts=active if active is not None
            else cfg.routing.num_active_experts)
    scheme = _arg("--scheme", "tktp")

    if over:
        print(f"[run_tktp] config={os.path.basename(CONFIG)} overrides={over}")
    cfg = dataclasses.replace(cfg, **over) if over else cfg
    from moe_bench.distributed import run_distributed
    run_distributed(cfg, scheme)


if __name__ == "__main__":
    main()
