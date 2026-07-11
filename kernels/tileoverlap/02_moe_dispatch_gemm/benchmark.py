"""MoE EP dispatch + grouped GEMM fused kernel (SM120 + PCIe) benchmark.

Run via `make run NUM_GPUS=<n>` (torchrun, one process per GPU).
The compiled extension is built with -DTK_NUM_DEVICES=<n>; world size must match.

Adapted from kernels/parallel/moe_dispatch_gemm/benchmark.py (H100 version).
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ThunderKittens", "kernels", "parallel")))

import torch

from common import (
    init_distributed_environment,
    destroy_distributed_environment,
    check_diff,
    benchmark_no_l2_clear,
    profile,
    clean_print,
)

from _C import TKParallelTensor, moe_dispatch_gemm, pcie_device_barrier  # type: ignore

ROW_BLOCK = 128
_barrier_seq = 0


def device_barrier(barrier: TKParallelTensor) -> None:
    """Kernel-level all-device barrier (slot+seq protocol, PCIe-safe)."""
    global _barrier_seq
    _barrier_seq += 1
    pcie_device_barrier(barrier, _barrier_seq)


@torch.no_grad()
def torch_reference(*, inputs_local, inputs_gathered, weights, outputs,
                    padded_tokens_per_expert, pull_dispatch_indices,
                    local_rank, local_world_size):
    inputs_full = torch.empty(local_world_size, inputs_local.shape[0], inputs_local.shape[1],
                              device=inputs_local.device, dtype=inputs_local.dtype)
    torch.distributed.all_gather_into_tensor(inputs_full, inputs_local)
    src_dev = pull_dispatch_indices[:, 0].long()
    src_tok = pull_dispatch_indices[:, 1].long()
    valid = (src_dev >= 0) & (src_tok >= 0)
    inputs_gathered[valid] = inputs_full[src_dev[valid], src_tok[valid]]

    num_experts = padded_tokens_per_expert.shape[0]
    num_experts_per_dev = num_experts // local_world_size
    expert_offset = num_experts_per_dev * local_rank
    start = 0
    for e in range(num_experts_per_dev):
        end = start + int(padded_tokens_per_expert[expert_offset + e])
        torch.matmul(inputs_gathered[start:end], weights[e], out=outputs[start:end])
        start = end


def make_dispatch_schedule(chosen_experts, padded_tokens_per_expert, num_init_tokens_per_dev,
                           local_rank, local_world_size, num_experts_per_dev, device):
    """Per-token pull schedule: (src_dev, src_token) per padded local slot,
    sorted by local expert, ring-ordered by source device."""
    expert_start = num_experts_per_dev * local_rank
    expert_end = num_experts_per_dev * (local_rank + 1)
    num_padded_local_tokens = int(padded_tokens_per_expert[expert_start:expert_end].sum())

    write_pos = torch.cat([
        torch.zeros(1, dtype=torch.int64),
        torch.cumsum(padded_tokens_per_expert[expert_start:expert_end - 1].cpu().long(), dim=0)
    ]).tolist()
    schedule = torch.full((num_padded_local_tokens, 2), -1, dtype=torch.int32)
    chosen_cpu = chosen_experts.cpu()
    for i in range(local_world_size):
        src_dev_idx = (i + local_rank) % local_world_size  # ring order: spread PCIe load
        base = src_dev_idx * num_init_tokens_per_dev
        for src_token_idx in range(num_init_tokens_per_dev):
            for expert_idx in chosen_cpu[base + src_token_idx].tolist():
                if expert_start <= expert_idx < expert_end:
                    e = expert_idx - expert_start
                    schedule[write_pos[e], 0] = src_dev_idx
                    schedule[write_pos[e], 1] = src_token_idx
                    write_pos[e] += 1
    return schedule.to(device), num_padded_local_tokens


def run(B, S, H, I, num_experts, top_k, num_comm_sms, local_rank, local_world_size,
        num_warmup_iters=2, num_iters=10, check_correctness=False, do_profile=False):
    device = f"cuda:{local_rank}"
    num_init_tokens_per_dev = B * S // local_world_size
    num_experts_per_dev = num_experts // local_world_size

    # ---- inputs ----
    torch.random.manual_seed(42 + local_rank)
    inputs_local = torch.randn(num_init_tokens_per_dev, H, device=device, dtype=torch.bfloat16) / H ** 0.5
    inputs_local_tk = TKParallelTensor((num_init_tokens_per_dev, H), dtype=torch.bfloat16,
                                       local_rank=local_rank, local_world_size=local_world_size,
                                       multicast=False)
    inputs_local_tk.data_.copy_(inputs_local)
    weights = torch.randn(num_experts_per_dev, H, I, device=device, dtype=torch.bfloat16) / H ** 0.5

    # ---- routing (identical on all ranks) ----
    if local_rank == 0:
        routing_weights = torch.rand(num_experts, device=device, dtype=torch.float32)
        chosen_experts = torch.multinomial(routing_weights.repeat(B * S, 1), top_k, replacement=False).to(torch.int32)
        tokens_per_expert = torch.bincount(chosen_experts.view(-1), minlength=num_experts).to(torch.int32)
    else:
        chosen_experts = torch.empty(B * S, top_k, device=device, dtype=torch.int32)
        tokens_per_expert = torch.empty(num_experts, device=device, dtype=torch.int32)
    torch.distributed.broadcast(chosen_experts, 0)
    torch.distributed.broadcast(tokens_per_expert, 0)

    padded_tokens_per_expert = (tokens_per_expert + ROW_BLOCK - 1) // ROW_BLOCK * ROW_BLOCK
    num_padded_max_tokens = int(padded_tokens_per_expert.reshape(local_world_size, num_experts_per_dev).sum(dim=1).amax())

    clean_print("Generating dispatch schedule...", print_once=True)
    pull_dispatch_indices, num_padded_local_tokens = make_dispatch_schedule(
        chosen_experts, padded_tokens_per_expert, num_init_tokens_per_dev,
        local_rank, local_world_size, num_experts_per_dev, device)

    # ---- buffers ----
    inputs_gathered = torch.zeros(num_padded_local_tokens, H, dtype=torch.bfloat16, device=device)
    inputs_gathered_torch = torch.zeros_like(inputs_gathered)
    outputs = torch.zeros(num_padded_local_tokens, I, dtype=torch.bfloat16, device=device)
    outputs_torch = torch.zeros_like(outputs)
    # row 0: per-row-block dispatch counters (local writes only)
    # row 1: pcie_barrier_all arrival slots (peer release-stores)
    barrier_cols = max(num_padded_max_tokens // ROW_BLOCK + 1, local_world_size, 32)
    barrier = TKParallelTensor((2, barrier_cols), dtype=torch.int,
                               local_rank=local_rank, local_world_size=local_world_size,
                               multicast=False)
    barrier.data_.zero_()

    torch.distributed.barrier()
    torch.cuda.synchronize()

    tk_run = lambda: moe_dispatch_gemm(
        inputs_local_tk, inputs_gathered, weights, outputs,
        padded_tokens_per_expert, pull_dispatch_indices, barrier,
        num_comm_sms, num_padded_local_tokens)
    torch_run = lambda: torch_reference(
        inputs_local=inputs_local, inputs_gathered=inputs_gathered_torch,
        weights=weights, outputs=outputs_torch,
        padded_tokens_per_expert=padded_tokens_per_expert,
        pull_dispatch_indices=pull_dispatch_indices,
        local_rank=local_rank, local_world_size=local_world_size)

    if check_correctness:
        clean_print("Checking correctness...", print_once=True)
        torch_run()
        tk_run()
        device_barrier(barrier)  # usage demo: peers done pulling before buffers change
        torch.distributed.barrier()
        torch.cuda.synchronize()
        check_diff("MoE Dispatch GEMM (SM120/PCIe) TK vs Torch", outputs, outputs_torch)
        check_diff("Gathered tokens TK vs Torch", inputs_gathered, inputs_gathered_torch)

    if do_profile:
        clean_print("Profiling...", print_once=True)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        profile(tk_run, num_iters=1)
        torch.distributed.barrier()
        torch.cuda.synchronize()

    tk_avg_ms = benchmark_no_l2_clear(tk_run, num_warmup_iters, num_iters)

    # decomposition baselines for overlap-efficiency math (09篇三件套):
    # T_comm alone ~ torch all_gather; T_comp alone measured in 01_grouped_gemm.
    flops = 2.0 * (B * S * top_k) * H * I / local_world_size
    clean_print("===============================================================================", print_once=True)
    clean_print(f"<MoE Dispatch GEMM SM120 | world={local_world_size} | E={num_experts} top{top_k} | "
                f"{B}x{S}x{H}x{I} | comm_sms={num_comm_sms}>", print_once=True)
    clean_print(f"TK fused: {tk_avg_ms:.3f} ms | {flops * 1e-9 / tk_avg_ms:.2f} TFLOP/s "
                f"(padded_local_tokens={num_padded_local_tokens})")


if __name__ == "__main__":
    local_rank, local_world_size = init_distributed_environment()

    # DeepSeek-V3-ish config; shrink seq_len first if VRAM is tight.
    TOP_K = 8
    NUM_EXPERTS = 256
    BATCH_SIZE = 1
    HIDDEN_SIZE = 7168       # must match compiled TK_HIDDEN
    EXPERT_HIDDEN_SIZE = 2048

    first = True
    for seq_len in [4096, 8192]:
        for num_comm_sms in [1, 2, 4, 8, 16]:
            run(B=BATCH_SIZE, S=seq_len, H=HIDDEN_SIZE, I=EXPERT_HIDDEN_SIZE,
                num_experts=NUM_EXPERTS, top_k=TOP_K, num_comm_sms=num_comm_sms,
                local_rank=local_rank, local_world_size=local_world_size,
                check_correctness=first, do_profile=False)
            first = False

    destroy_distributed_environment()
