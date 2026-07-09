# SPDX-License-Identifier: Apache-2.0
"""Ad-hoc shape bench: E=64, TOP_K=8, hidden=4096, gate_up=6144 (=2*inter=3072).

Runs tkfused (default: pull+prered+fused gate/up+GPU schedule) and serial for the
same shape, 512 tokens/rank, bf16 EP world=4, reports latency + rel_err.

  python -m moe_bench.tools.bench_shape_4096
"""
from __future__ import annotations

from moe_bench.config import Distribution, MoEBenchConfig, ParallelMode, Precision, RoutingConfig
from moe_bench.distributed import run_distributed


def main():
    for scheme in ("tkfused", "serial"):
        print(f"\n===== {scheme} =====", flush=True)
        cfg = MoEBenchConfig(
            hidden_size=4096, intermediate_size=3072, num_experts=64, topk=8,
            parallel_mode=ParallelMode.EP, world_size=4, precision=Precision.BF16,
            num_tokens=[512], routing=RoutingConfig(distribution=Distribution.BALANCED),
            distributed=True, warmup_iters=5, bench_iters=20, use_cuda_graph=False,
            seed=0, verify=(scheme == "tkfused"), device="cuda")
        run_distributed(cfg, scheme)


if __name__ == "__main__":
    main()
