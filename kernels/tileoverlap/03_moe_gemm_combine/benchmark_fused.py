"""Phase 3b: fused W2 GEMM + combine, verified against a torch reference.

Each rank is simultaneously an expert card (runs W2 grouped GEMM over its local
experts) and a source card (combines its own tokens' top-k expert outputs pulled
from peer cards). One kernel launch does both; the ready-signal gates the pull.

Torch reference: run the same grouped W2 GEMM on each rank, all_gather the expert
outputs, then combine on the host side. bf16 throughout, FP32 combine accumulation.

Run via `make -f Makefile.fused run NUM_GPUS=<n>` or the CMD in that Makefile.
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
    clean_print,
)

from _Cf import TKParallelTensor, moe_gemm_combine_fused  # type: ignore

ROW_BLOCK = 128
TOP_K = 8


def build_combine_schedule(chosen_experts, padded_tokens_per_expert,
                           num_init_tokens_per_dev, local_rank, local_world_size,
                           num_experts_per_dev, device):
    """See 03a benchmark.py — inverse of the dispatch schedule, for this rank's
    source tokens: (num_src*TOP_K, 2) combine indices (e_rank, remote_slot)."""
    chosen_cpu = chosen_experts.cpu()
    padded_cpu = padded_tokens_per_expert.cpu().long()
    num_src = num_init_tokens_per_dev
    combine_idx = torch.full((num_src * TOP_K, 2), -1, dtype=torch.int32)
    for e_rank in range(local_world_size):
        expert_start = num_experts_per_dev * e_rank
        expert_end = num_experts_per_dev * (e_rank + 1)
        write_pos = torch.cat([
            torch.zeros(1, dtype=torch.int64),
            torch.cumsum(padded_cpu[expert_start:expert_end - 1], dim=0)
        ]).tolist()
        for i in range(local_world_size):
            src_dev_idx = (i + e_rank) % local_world_size
            base = src_dev_idx * num_init_tokens_per_dev
            for src_token_idx in range(num_init_tokens_per_dev):
                for kpos, expert_idx in enumerate(chosen_cpu[base + src_token_idx].tolist()):
                    if expert_start <= expert_idx < expert_end:
                        e = expert_idx - expert_start
                        slot = write_pos[e]
                        write_pos[e] += 1
                        if src_dev_idx == local_rank:
                            row = src_token_idx * TOP_K + kpos
                            combine_idx[row, 0] = e_rank
                            combine_idx[row, 1] = slot
    return combine_idx.to(device), None

_seq = 0


def run(S, H, I, num_experts, top_k, num_comm_sms, local_rank, local_world_size,
        num_warmup_iters=2, num_iters=10, check_correctness=False):
    global _seq
    assert top_k == TOP_K
    device = f"cuda:{local_rank}"
    num_init_tokens_per_dev = S // local_world_size
    num_experts_per_dev = num_experts // local_world_size

    # ---- routing (identical on all ranks) ----
    torch.random.manual_seed(1234)
    routing_weights = torch.rand(num_experts, device=device, dtype=torch.float32)
    chosen_experts = torch.multinomial(routing_weights.repeat(S, 1), top_k, replacement=False).to(torch.int32)
    topk_w = torch.softmax(torch.rand(S, top_k, device=device), dim=-1)
    tokens_per_expert = torch.bincount(chosen_experts.view(-1), minlength=num_experts).to(torch.int32)
    padded_tokens_per_expert = (tokens_per_expert + ROW_BLOCK - 1) // ROW_BLOCK * ROW_BLOCK

    expert_start = num_experts_per_dev * local_rank
    expert_end = num_experts_per_dev * (local_rank + 1)
    num_padded_local = int(padded_tokens_per_expert[expert_start:expert_end].sum())
    num_padded_max = int(padded_tokens_per_expert.reshape(local_world_size, num_experts_per_dev).sum(dim=1).amax())

    # ---- W2 GEMM inputs: random activations h (padded_local, I) and weights (E_local, I, H) ----
    torch.random.manual_seed(42 + local_rank)
    h = torch.randn(num_padded_local, I, device=device, dtype=torch.bfloat16) / I ** 0.5
    weights = torch.randn(num_experts_per_dev, I, H, device=device, dtype=torch.bfloat16) / I ** 0.5

    # expert_outputs: peer-readable, sized to max so every card's pgl view matches
    expert_out_tk = TKParallelTensor((num_padded_max, H), dtype=torch.bfloat16,
                                     local_rank=local_rank, local_world_size=local_world_size,
                                     multicast=False)
    expert_out_tk.data_.zero_()

    barrier_cols = max(num_padded_max // ROW_BLOCK + 1, 32)
    barrier = TKParallelTensor((1 + local_world_size, barrier_cols), dtype=torch.int,
                               local_rank=local_rank, local_world_size=local_world_size,
                               multicast=False)
    barrier.data_.zero_()

    # ---- combine schedule for this rank's source tokens ----
    num_src = num_init_tokens_per_dev
    combine_idx, _ = build_combine_schedule(
        chosen_experts, padded_tokens_per_expert, num_init_tokens_per_dev,
        local_rank, local_world_size, num_experts_per_dev, device)
    my_topk_w = topk_w[local_rank * num_src:(local_rank + 1) * num_src]
    combine_w = my_topk_w.reshape(-1, 1).contiguous().to(device)

    combine_out = torch.zeros(num_src, H, device=device, dtype=torch.bfloat16)
    combine_out_ref = torch.zeros_like(combine_out)

    torch.distributed.barrier()
    torch.cuda.synchronize()

    def tk_run():
        global _seq
        _seq += 1
        moe_gemm_combine_fused(
            h, weights, expert_out_tk, padded_tokens_per_expert,
            combine_out, combine_idx, combine_w, barrier,
            num_comm_sms, num_padded_local, num_src, _seq)

    if check_correctness:
        # torch reference: grouped W2 GEMM per rank -> expert outputs, gather, combine.
        eo_ref = torch.zeros(num_padded_local, H, device=device, dtype=torch.bfloat16)
        start = 0
        for e in range(num_experts_per_dev):
            end = start + int(padded_tokens_per_expert[expert_start + e])
            if end > start:
                torch.matmul(h[start:end], weights[e], out=eo_ref[start:end])
            start = end
        eo_full = torch.empty(local_world_size, num_padded_max, H, device=device, dtype=torch.bfloat16)
        eo_padded = torch.zeros(num_padded_max, H, device=device, dtype=torch.bfloat16)
        eo_padded[:num_padded_local] = eo_ref
        torch.distributed.all_gather_into_tensor(eo_full, eo_padded)
        idx_cpu = combine_idx.cpu()
        for t in range(num_src):
            acc = torch.zeros(H, device=device, dtype=torch.float32)
            for k in range(top_k):
                e_rank = int(idx_cpu[t * top_k + k, 0]); slot = int(idx_cpu[t * top_k + k, 1])
                if e_rank < 0 or slot < 0:
                    continue
                acc += combine_w[t * top_k + k, 0] * eo_full[e_rank, slot].float()
            combine_out_ref[t] = acc.to(torch.bfloat16)

        tk_run()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        check_diff("Fused W2 GEMM+combine TK vs Torch", combine_out, combine_out_ref)

    tk_ms = benchmark_no_l2_clear(tk_run, num_warmup_iters, num_iters)
    flops = 2.0 * num_padded_local * I * H
    clean_print("===============================================================================", print_once=True)
    clean_print(f"<Fused W2+combine SM120 | world={local_world_size} | E={num_experts} top{top_k} | "
                f"S={S} H={H} I={I} | comm_sms={num_comm_sms}>", print_once=True)
    clean_print(f"TK fused: {tk_ms:.3f} ms | {flops * 1e-9 / tk_ms:.1f} TFLOP/s GEMM (num_src={num_src})")


if __name__ == "__main__":
    local_rank, local_world_size = init_distributed_environment()
    NUM_EXPERTS = 256
    HIDDEN_SIZE = 7168
    INTER = 2048
    first = True
    for seq_len in [2048, 4096]:
        for num_comm_sms in [8, 16, 32]:
            run(S=seq_len, H=HIDDEN_SIZE, I=INTER, num_experts=NUM_EXPERTS, top_k=TOP_K,
                num_comm_sms=num_comm_sms, local_rank=local_rank, local_world_size=local_world_size,
                check_correctness=first)
            first = False
    destroy_distributed_environment()
