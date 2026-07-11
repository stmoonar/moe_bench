# SPDX-License-Identifier: Apache-2.0
"""TP driver: run the distributed tktp layer with verify, bypassing bench.py's
argparse (vllm FlexibleArgumentParser mangles argv here). Mirror of run_tkfused.

  python -m moe_bench.tools.run_tktp  <num_experts>  [--no-verify] [--iters N]
      [--dist balanced|uniform|skewed] [--tokens T] [--scheme tktp|serial]

Prints per-token latency + rel_err (harness verifies vs reference_moe through
the full logical weight set automatically unless --no-verify).
"""
from __future__ import annotations

import sys

from moe_bench.config import Distribution, MoEBenchConfig, ParallelMode, Precision, RoutingConfig


def _arg(flag, default, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def main():
    ne = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 64
    verify = "--no-verify" not in sys.argv
    biter = _arg("--iters", 20, int)
    dist = _arg("--dist", "balanced")
    tokens = _arg("--tokens", 512, int)
    scheme = _arg("--scheme", "tktp")
    json_out = _arg("--json", None)
    cfg = MoEBenchConfig(
        hidden_size=4096, intermediate_size=3072, num_experts=ne, topk=8,
        parallel_mode=ParallelMode.TP, world_size=4, precision=Precision.BF16,
        num_tokens=[tokens], routing=RoutingConfig(distribution=Distribution(dist)),
        distributed=True, warmup_iters=5, bench_iters=biter, use_cuda_graph=False,
        seed=0, verify=verify, device="cuda", output_json=json_out)
    from moe_bench.distributed import run_distributed
    run_distributed(cfg, scheme)


if __name__ == "__main__":
    main()
