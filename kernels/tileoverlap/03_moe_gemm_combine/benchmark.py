"""Phase 3a: standalone MoE combine (layer1 reduce) correctness anchor.

Each rank owns a slab of expert outputs (num_padded_local_tokens_this_rank, H).
For each of this rank's SOURCE tokens, the combine kernel pulls that token's
top-k expert-output rows from the expert cards (P2P) and FP32 weighted-sums them.

We verify against a torch reference built from the same routing. No real W1/W2:
expert_outputs are random, so this isolates the pull+reduce primitive.

Run via `make run NUM_GPUS=<n>`.
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

from _C import TKParallelTensor, moe_combine  # type: ignore

ROW_BLOCK = 128
TOP_K = 8


def build_combine_schedule(chosen_experts, padded_tokens_per_expert,
                           num_init_tokens_per_dev, local_rank, local_world_size,
                           num_experts_per_dev, device):
    """For THIS rank's source tokens, produce (num_src*TOP_K, 2) combine indices
    (e_rank, remote_slot) and (num_src*TOP_K,) weights.

    Mirrors 02's dispatch schedule but inverted: we replay every expert card's
    write positions, and for our own source tokens record where each of their
    top-k experts landed (which card, which padded slot).
    """
    chosen_cpu = chosen_experts.cpu()
    padded_cpu = padded_tokens_per_expert.cpu().long()
    num_src = num_init_tokens_per_dev

    # write_pos[e_rank][local_expert] = running slot cursor on that expert card,
    # exactly reproducing 02's make_dispatch_schedule ring/order per card.
    combine_idx = torch.full((num_src * TOP_K, 2), -1, dtype=torch.int32)
    combine_w = torch.zeros((num_src * TOP_K, 1), dtype=torch.float32)

    for e_rank in range(local_world_size):
        expert_start = num_experts_per_dev * e_rank
        expert_end = num_experts_per_dev * (e_rank + 1)
        # slot cursor per local expert on card e_rank (prefix sum of padded counts)
        write_pos = torch.cat([
            torch.zeros(1, dtype=torch.int64),
            torch.cumsum(padded_cpu[expert_start:expert_end - 1], dim=0)
        ]).tolist()
        # card e_rank pulls source tokens in ring order (i + e_rank) % world
        for i in range(local_world_size):
            src_dev_idx = (i + e_rank) % local_world_size
            base = src_dev_idx * num_init_tokens_per_dev
            for src_token_idx in range(num_init_tokens_per_dev):
                for kpos, expert_idx in enumerate(chosen_cpu[base + src_token_idx].tolist()):
                    if expert_start <= expert_idx < expert_end:
                        e = expert_idx - expert_start
                        slot = write_pos[e]
                        write_pos[e] += 1
                        # Only record for tokens that belong to THIS rank as source.
                        if src_dev_idx == local_rank:
                            row = src_token_idx * TOP_K + kpos
                            combine_idx[row, 0] = e_rank
                            combine_idx[row, 1] = slot
    return combine_idx.to(device), combine_w  # weights filled by caller from topk_weights


def run(S, H, num_experts, top_k, local_rank, local_world_size,
        num_warmup_iters=2, num_iters=10, check_correctness=False):
    assert top_k == TOP_K
    device = f"cuda:{local_rank}"
    num_init_tokens_per_dev = S // local_world_size
    num_experts_per_dev = num_experts // local_world_size

    # ---- routing (identical on all ranks, like 02) ----
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

    # ---- this rank's expert-output slab (random; IPC-shared for peers to read) ----
    torch.random.manual_seed(42 + local_rank)
    expert_out_local = torch.randn(num_padded_local, H, device=device, dtype=torch.bfloat16)
    # allocate to the max so every card's pgl view has a valid, uniform size
    expert_out_tk = TKParallelTensor((num_padded_max, H), dtype=torch.bfloat16,
                                     local_rank=local_rank, local_world_size=local_world_size,
                                     multicast=False)
    expert_out_tk.data_.zero_()
    expert_out_tk.data_[:num_padded_local].copy_(expert_out_local)

    barrier = TKParallelTensor((2, max(local_world_size, 32)), dtype=torch.int,
                               local_rank=local_rank, local_world_size=local_world_size,
                               multicast=False)
    barrier.data_.zero_()

    # ---- combine schedule for this rank's source tokens ----
    num_src = num_init_tokens_per_dev
    combine_idx, combine_w = build_combine_schedule(
        chosen_experts, padded_tokens_per_expert, num_init_tokens_per_dev,
        local_rank, local_world_size, num_experts_per_dev, device)
    # fill weights: this rank's source tokens are the global rows [local_rank*num_src, ...)
    my_topk_w = topk_w[local_rank * num_src:(local_rank + 1) * num_src]  # (num_src, top_k)
    combine_w = my_topk_w.reshape(-1, 1).contiguous().to(device)

    outputs = torch.zeros(num_src, H, device=device, dtype=torch.bfloat16)
    outputs_ref = torch.zeros_like(outputs)

    torch.distributed.barrier()
    torch.cuda.synchronize()

    def tk_run():
        moe_combine(expert_out_tk, outputs, combine_idx, combine_w, barrier, num_src)

    # ---- torch reference: gather all cards' expert-output slabs, then combine ----
    if check_correctness:
        all_expert_out = torch.empty(local_world_size, num_padded_max, H, device=device, dtype=torch.bfloat16)
        torch.distributed.all_gather_into_tensor(all_expert_out, expert_out_tk.data_)
        idx_cpu = combine_idx.cpu()
        for t in range(num_src):
            acc = torch.zeros(H, device=device, dtype=torch.float32)
            for k in range(top_k):
                e_rank = int(idx_cpu[t * top_k + k, 0])
                slot = int(idx_cpu[t * top_k + k, 1])
                if e_rank < 0 or slot < 0:
                    continue
                acc += combine_w[t * top_k + k, 0] * all_expert_out[e_rank, slot].float()
            outputs_ref[t] = acc.to(torch.bfloat16)

        tk_run()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        check_diff("MoE combine (SM120/PCIe) TK vs Torch", outputs, outputs_ref)

    tk_ms = benchmark_no_l2_clear(tk_run, num_warmup_iters, num_iters)
    bytes_pulled = num_src * top_k * H * 2  # bf16 rows pulled per rank
    gbps = bytes_pulled * 1e-6 / tk_ms      # bytes/ms * 1e-6 = GB/s
    clean_print("===============================================================================", print_once=True)
    clean_print(f"<MoE combine SM120 | world={local_world_size} | E={num_experts} top{top_k} | S={S} H={H}>", print_once=True)
    clean_print(f"TK combine: {tk_ms:.3f} ms | {gbps:.1f} GB/s effective pull (num_src={num_src})")


if __name__ == "__main__":
    local_rank, local_world_size = init_distributed_environment()
    NUM_EXPERTS = 256
    HIDDEN_SIZE = 7168
    first = True
    for seq_len in [2048, 4096]:
        run(S=seq_len, H=HIDDEN_SIZE, num_experts=NUM_EXPERTS, top_k=TOP_K,
            local_rank=local_rank, local_world_size=local_world_size,
            check_correctness=first)
        first = False
    destroy_distributed_environment()
