"""Single-GPU grouped GEMM: correctness vs torch + TFLOPS. No distribution."""
import torch

from _C import grouped_gemm  # type: ignore


def torch_reference(inputs, weights, outputs, padded_tokens_per_expert):
    start = 0
    for e in range(weights.shape[0]):
        end = start + int(padded_tokens_per_expert[e])
        torch.matmul(inputs[start:end], weights[e], out=outputs[start:end])
        start = end


def run(num_experts=32, H=7168, I=2048, mean_tokens_per_expert=1024, seed=42,
        num_warmup_iters=2, num_iters=10):
    device = "cuda:0"
    torch.random.manual_seed(seed)

    # random per-expert token counts, padded to 128 (like real MoE routing)
    counts = torch.randint(0, mean_tokens_per_expert * 2, (num_experts,), dtype=torch.int32)
    padded = (counts + 127) // 128 * 128
    padded = padded.to(device)
    num_tokens = int(padded.sum())
    print(f"experts={num_experts}, padded tokens={num_tokens}, H={H}, I={I}")

    inputs = torch.randn(num_tokens, H, device=device, dtype=torch.bfloat16) / H ** 0.5
    weights = torch.randn(num_experts, H, I, device=device, dtype=torch.bfloat16) / H ** 0.5
    outputs = torch.zeros(num_tokens, I, device=device, dtype=torch.bfloat16)
    outputs_ref = torch.zeros_like(outputs)

    # correctness
    grouped_gemm(inputs, weights, outputs, padded)
    torch_reference(inputs, weights, outputs_ref, padded)
    torch.cuda.synchronize()
    diff = (outputs.float() - outputs_ref.float()).abs()
    print(f"max diff: {diff.max().item():.6f} | mean diff: {diff.mean().item():.6f} "
          f"| ref mean abs: {outputs_ref.float().abs().mean().item():.6f}")
    assert diff.max().item() < 0.1, "CORRECTNESS FAILURE"

    # performance
    def bench(fn):
        for _ in range(num_warmup_iters):
            fn()
        torch.cuda.synchronize()
        start_ev, end_ev = torch.cuda.Event(True), torch.cuda.Event(True)
        start_ev.record()
        for _ in range(num_iters):
            fn()
        end_ev.record()
        torch.cuda.synchronize()
        return start_ev.elapsed_time(end_ev) / num_iters

    flops = 2.0 * num_tokens * H * I
    tk_ms = bench(lambda: grouped_gemm(inputs, weights, outputs, padded))
    torch_ms = bench(lambda: torch_reference(inputs, weights, outputs_ref, padded))
    print(f"TK:    {tk_ms:8.3f} ms | {flops / tk_ms * 1e-9:8.2f} TFLOP/s")
    print(f"torch: {torch_ms:8.3f} ms | {flops / torch_ms * 1e-9:8.2f} TFLOP/s")
    print(f"TK / torch: {torch_ms / tk_ms * 100:.1f}%")


if __name__ == "__main__":
    for mean_tokens in [256, 1024, 4096]:
        run(mean_tokens_per_expert=mean_tokens)
        print()
