# SPDX-License-Identifier: Apache-2.0
"""Step-5 driver (docs/09 §5.5): run the distributed tkfused layer with verify,
bypassing bench.py's argparse (vllm FlexibleArgumentParser mangles argv here).

  python -m moe_bench.tools.run_tkfused  <num_experts>  [--no-verify] [--iters N]

Honors TK_DISPATCH (pull|push2|push3). Prints per-token latency + rel_err.
"""
from __future__ import annotations

import sys

from moe_bench.config import Distribution, MoEBenchConfig, ParallelMode, Precision, RoutingConfig
from moe_bench.distributed import run_distributed


def main():
    ne = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 256
    verify = "--no-verify" not in sys.argv
    biter = 20
    if "--iters" in sys.argv:
        biter = int(sys.argv[sys.argv.index("--iters") + 1])
    cfg = MoEBenchConfig(
        hidden_size=7168, intermediate_size=2048, num_experts=ne, topk=8,
        parallel_mode=ParallelMode.EP, world_size=4, precision=Precision.BF16,
        num_tokens=[512], routing=RoutingConfig(distribution=Distribution.BALANCED),
        distributed=True, warmup_iters=5, bench_iters=biter, use_cuda_graph=False,
        seed=0, verify=verify, device="cuda")
    run_distributed(cfg, "tkfused")


if __name__ == "__main__":
    main()
