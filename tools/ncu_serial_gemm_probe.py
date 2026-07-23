# SPDX-License-Identifier: Apache-2.0
"""单卡 serial baseline GEMM 探针: 复刻 SerialNaive.run() 的 fused_experts 一步,
剥掉前后的 NCCL 集合通信 -> 单进程单卡, kernel replay 安全, 用于 ncu 采
vllm fused_moe GEMM 的计算效率(与 tools/verify_fp8_gemm.py 的 gg8 探针同口径对比)。

口径 = 主配置 TP4 的 rank 0 视角: 权重 TP 分片 w1 (64, 1536, 4096) /
w2 (64, 4096, 768), fp8 128x128 block 量化(data.make_weights 同一路径);
输入 = 4 个 rank 各 512 token 按 rank-major 拼接的 2048 token 全批,
routing 与分布式 serial 逐 bit 相同(同种子链)。

  单卡运行(选一张空闲卡), 形状与 run_tktp 同风格, 默认 = 主配置口径:
    CUDA_VISIBLE_DEVICES=<n> python -m moe_bench.tools.ncu_serial_gemm_probe \
        [num_experts] [--tokens T每rank] [--topk K] [--hidden H] [--inter I] \
        [--world W] [--precision bf16|fp8] [--iters N]

  注意: fp8 时 I/W(intermediate_shard)必须是 128 的倍数(block 量化对齐)。
  ncu 包法见 HANDOFF/对话记录: --replay-mode kernel + -k 'regex:fused_moe_kernel'。
"""
from __future__ import annotations

import sys

import torch


def _arg(flag, default, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def main():
    ne = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 64
    tokens = _arg("--tokens", 512, int)
    topk = _arg("--topk", 8, int)
    hidden = _arg("--hidden", 4096, int)
    inter = _arg("--inter", 3072, int)
    world = _arg("--world", 4, int)
    iters = _arg("--iters", 8, int)
    from moe_bench.config import MoEBenchConfig, ParallelMode, Precision
    from moe_bench.data import make_problem, make_weights

    cfg = MoEBenchConfig(
        hidden_size=hidden, intermediate_size=inter, num_experts=ne, topk=topk,
        parallel_mode=ParallelMode.TP, world_size=world,
        precision=Precision(_arg("--precision", "fp8")),
        num_tokens=[tokens], distributed=False, seed=0, verify=False, device="cuda")

    weights = make_weights(cfg, rank=0)
    probs = [make_problem(cfg, tokens, rank=r, weights=weights) for r in range(world)]
    hidden_full = torch.cat([p.hidden_states for p in probs])
    topk_ids_full = torch.cat([p.topk_ids for p in probs])
    topk_weights_full = torch.cat([p.topk_weights for p in probs])

    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    def run():
        return fused_experts(
            hidden_states=hidden_full, w1=weights.w1, w2=weights.w2,
            topk_weights=topk_weights_full, topk_ids=topk_ids_full,
            global_num_experts=cfg.num_experts, expert_map=None,
            quant_config=weights.quant_config)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000.0 / iters
    print(f"serial fused_experts (single-GPU, M={tokens * world}, E={ne}, topk={topk}, "
          f"H={hidden}, I_shard={inter // world}, {cfg.precision.value}): {us:.1f} us/iter")


if __name__ == "__main__":
    main()
