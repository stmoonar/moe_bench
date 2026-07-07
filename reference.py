# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-torch reference MoE forward for correctness verification.

Computes the same MoE-layer output as the benchmarked implementations but with a
straightforward, unfused torch implementation, so any implementation's output
can be checked against a trusted ground truth. The math mirrors the reference
used by vLLM's own fused-MoE tests
(``tests/kernels/utils.py::torch_experts``) for the two precisions the
benchmark supports:

- **bf16 / fp16**: dense per-expert GEMMs accumulated in fp32.
- **fp8 (per-block w8a8)**: activations are block-quantized to fp8 the same way
  the kernel quantizes them, then a native block matmul dequantizes weights and
  activations tile-by-tile — so the reference carries the *same* quantization
  error as the kernel and the two stay comparable within a tight tolerance.

The reference is unified over TP and EP through ``expert_map``: EP masks out
global experts that don't live on this rank (their contribution is zero),
exactly as the benchmarked ``fused_experts`` path does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import Precision
from .data import MoEProblem

# Per-precision (atol, rtol) for the elementwise ``torch.testing.assert_close``
# pass/fail check, matching the tolerances vLLM's own fused-MoE tests use
# (tests/kernels/moe): bf16/fp16 are limited by the kernel's low-precision
# accumulation (absolute error, rtol=0); fp8 additionally carries per-block
# activation-quantization error that scales with magnitude, so it also gets
# a relative term.
_DEFAULT_TOL = {
    Precision.BF16: (2e-2, 0.0),
    Precision.FP16: (1e-2, 0.0),
    Precision.FP8: (3.5e-2, 3.5e-2),
}


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: ``silu(x[:, :d]) * x[:, d:]`` (vLLM's gate/up order)."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _local_slots(problem: MoEProblem) -> torch.Tensor:
    """Map each ``(token, k)`` routing choice to a local expert slot.

    Returns a ``(num_tokens * topk,)`` int64 tensor of local slot ids, with
    ``-1`` for global experts that don't live on this rank (EP). In TP every
    expert is local so the ids pass through unchanged.
    """
    topk_ids = problem.topk_ids.long()
    if problem.expert_map is not None:  # EP: global id -> local slot or -1.
        topk_ids = problem.expert_map[topk_ids].long()
    return topk_ids.view(-1)


def _block_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    block_shape: list[int],
) -> torch.Tensor:
    """Block-quantized ``a @ b.T`` with fp32 accumulation.

    Mirrors ``tests/kernels/quant_utils.py::native_w8a8_block_matmul``:
    ``a`` is ``(M, K)`` fp8 with per-``block_k`` column scales ``a_s`` and ``b``
    is ``(N, K)`` fp8 with per-(``block_n``, ``block_k``) tile scales ``b_s``.
    """
    block_n, block_k = block_shape
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    m, k = a.shape
    n = b.shape[0]
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k

    out = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    for i in range(k_tiles):
        ks, ke = i * block_k, min((i + 1) * block_k, k)
        a_tile = a[:, ks:ke]
        a_scale = a_s[:, i : i + 1]
        for j in range(n_tiles):
            ns, ne = j * block_n, min((j + 1) * block_n, n)
            b_tile = b[ns:ne, ks:ke]
            out[:, ns:ne] += (a_tile @ b_tile.t()) * (a_scale * b_s[j, i])
    return out


def _reference_dense(problem: MoEProblem) -> torch.Tensor:
    """bf16/fp16 reference: per-expert GEMMs accumulated in fp32."""
    cfg = problem.config
    m, k = problem.hidden_states.shape
    topk = cfg.topk
    a = problem.hidden_states.float().view(m, 1, k).expand(m, topk, k).reshape(-1, k)
    w1 = problem.w1.float()
    w2 = problem.w2.float()

    slots = _local_slots(problem)
    out = torch.zeros(m * topk, k, dtype=torch.float32, device=a.device)
    for slot in range(w1.shape[0]):
        mask = slots == slot
        if not mask.any():
            continue
        tmp1 = a[mask] @ w1[slot].t()
        tmp2 = _silu_and_mul(tmp1)
        out[mask] = tmp2 @ w2[slot].t()
    return out.view(m, topk, k)


def _reference_fp8(problem: MoEProblem) -> torch.Tensor:
    """fp8 reference: block-quantize activations, then native block matmul.

    Carries the same activation-quantization error as the fused kernel so the
    two are comparable within a tight tolerance.
    """
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    from .data import FP8_DTYPE

    cfg = problem.config
    qc = problem.quant_config
    assert qc is not None and qc.w1_scale is not None and qc.w2_scale is not None
    block_shape = cfg.block_shape
    m, k = problem.hidden_states.shape
    topk = cfg.topk

    a = (
        problem.hidden_states.view(m, 1, k)
        .expand(m, topk, k)
        .reshape(-1, k)
        .contiguous()
    )
    # Quantize activations exactly as the kernel does (per-block, no per-token).
    a_q, a_s = moe_kernel_quantize_input(
        a, None, FP8_DTYPE, per_act_token_quant=False, block_shape=block_shape
    )

    slots = _local_slots(problem)
    out = torch.zeros(m * topk, k, dtype=torch.float32, device=a.device)
    for slot in range(problem.w1.shape[0]):
        mask = slots == slot
        if not mask.any():
            continue
        tmp1 = _block_matmul(
            a_q[mask], problem.w1[slot], a_s[mask], qc.w1_scale[slot], block_shape
        )
        tmp2 = _silu_and_mul(tmp1)
        tmp2_q, tmp2_s = moe_kernel_quantize_input(
            tmp2.to(cfg.torch_dtype),
            None,
            FP8_DTYPE,
            per_act_token_quant=False,
            block_shape=block_shape,
        )
        out[mask] = _block_matmul(
            tmp2_q, problem.w2[slot], tmp2_s, qc.w2_scale[slot], block_shape
        )
    return out.view(m, topk, k)


def reference_moe(problem: MoEProblem) -> torch.Tensor:
    """Ground-truth MoE-layer output ``(num_tokens, hidden_size)`` in fp32.

    Depends only on ``problem`` (the same inputs every implementation receives),
    so a benchmarked output can be checked against it directly.
    """
    if problem.config.precision == Precision.FP8:
        per_assignment = _reference_fp8(problem)
    else:
        per_assignment = _reference_dense(problem)
    # Weighted combine over the topk experts of each token.
    weights = problem.topk_weights.float().unsqueeze(-1)
    return (per_assignment * weights).sum(dim=1)


@dataclass
class VerificationResult:
    """Outcome of comparing an implementation output against the reference."""

    max_abs_err: float
    rel_err: float
    atol: float
    rtol: float
    passed: bool
    # First line of assert_close's diagnostic when the check fails ("" on pass).
    message: str = ""

    def summary(self) -> str:
        status = "ok" if self.passed else "FAIL"
        base = (
            f"{status} (rel_err={self.rel_err:.2e}, max_abs={self.max_abs_err:.2e}, "
            f"atol={self.atol:.1e}, rtol={self.rtol:.1e})"
        )
        return base if self.passed else f"{base}\n    {self.message}"


def verify_output(
    problem: MoEProblem,
    output: torch.Tensor,
    reference: torch.Tensor,
    atol: float | None = None,
    rtol: float | None = None,
) -> VerificationResult:
    """Elementwise-compare ``output`` against ``reference``.

    Uses ``torch.testing.assert_close`` (the same check vLLM's fused-MoE tests
    use) with a precision-aware ``(atol, rtol)``, so a single wildly-off element
    fails the check rather than being averaged away. Wrapped in try/except so a
    failing token count is recorded and the sweep still finishes. ``rel_err``
    (relative L2) and ``max_abs_err`` are also reported as at-a-glance metrics.
    """
    default_atol, default_rtol = _DEFAULT_TOL[problem.config.precision]
    atol = default_atol if atol is None else atol
    rtol = default_rtol if rtol is None else rtol

    out = output.float()
    ref = reference.float()
    diff = out - ref
    max_abs_err = diff.abs().max().item()
    rel_err = (diff.norm() / ref.norm().clamp_min(1e-12)).item()

    passed = True
    message = ""
    try:
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    except AssertionError as e:
        passed = False
        # Surface assert_close's "Mismatched elements: N / M (P%)" line if
        # present; fall back to its first line otherwise.
        lines = [ln.strip() for ln in str(e).strip().splitlines() if ln.strip()]
        mismatch = next((ln for ln in lines if ln.startswith("Mismatched")), None)
        message = mismatch or (lines[0] if lines else "assert_close failed")

    return VerificationResult(
        max_abs_err=max_abs_err,
        rel_err=rel_err,
        atol=atol,
        rtol=rtol,
        passed=passed,
        message=message,
    )
