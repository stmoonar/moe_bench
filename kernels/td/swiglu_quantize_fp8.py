################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

"""
Fused SwiGLU + FP8 block-wise quantization kernel.

Replaces the two-step process of:
  1. swiglu_forward(fc1_output, scale=routing_weight) -> BF16 intermediate
  2. _quantize_fp8_blockwise(intermediate, fp8_dtype)  -> FP8 + scale

With a single Triton kernel that avoids writing/reading the BF16 intermediate
to/from global memory.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_quantize_fp8_kernel(
    # Input: fc1_output, shape [M, 2*dim], contains gate (first half) and up (second half)
    AB_ptr,
    AB_row_stride,
    # Routing weight per token, shape [M]
    routing_weight_ptr,
    # Output: quantized FP8 tensor, shape [M, dim]
    out_ptr,
    out_row_stride,
    # Output: per-block scale, shape [M, dim // BLOCK_K]
    scale_ptr,
    scale_row_stride,
    # Dimensions
    n_cols,          # dim (half of the last dimension of AB)
    n_rows,
    # FP8 constants
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
    # Block size for quantization (typically 128)
    BLOCK_K: tl.constexpr,
):
    """
    Each program instance handles one row and one BLOCK_K chunk.
    Grid: (n_rows, n_cols // BLOCK_K)
    """
    row_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)

    # Offsets within the block
    col_offsets = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = col_offsets < n_cols

    # Load gate (A) and up (B) from the same row
    # A is AB[:, :n_cols], B is AB[:, n_cols:]
    a_ptr = AB_ptr + row_idx * AB_row_stride + col_offsets
    b_ptr = AB_ptr + row_idx * AB_row_stride + n_cols + col_offsets

    a = tl.load(a_ptr, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr, mask=mask, other=0.0).to(tl.float32)

    # SiLU(a) * b
    silu_a = a * tl.sigmoid(a)
    y = silu_a * b

    # Apply routing weight
    if routing_weight_ptr is not None:
        rw = tl.load(routing_weight_ptr + row_idx)
        y = y * rw

    # Block-wise FP8 quantization
    # Compute amax for this block
    amax = tl.max(tl.abs(y), axis=0)

    # Compute scale: amax / FP8_MAX (avoid division by zero)
    eps: tl.constexpr = 1e-12
    scale = tl.where(amax > eps, amax / FP8_MAX, 1.0)

    # Quantize
    y_q = tl.clamp(y / scale, FP8_MIN, FP8_MAX)

    # Store quantized output
    out_offset = row_idx * out_row_stride + col_offsets
    tl.store(out_ptr + out_offset, y_q.to(out_ptr.dtype.element_ty), mask=mask)

    # Store scale (one per block per row)
    scale_offset = row_idx * scale_row_stride + block_idx
    tl.store(scale_ptr + scale_offset, scale)


def swiglu_quantize_fp8(
    AB: torch.Tensor,
    routing_weight: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    block_k: int = 128,
) -> tuple:
    """
    Fused SwiGLU activation + FP8 block-wise quantization.

    Args:
        AB: Input tensor of shape [M, 2*dim] (fc1 output), dtype BF16/FP16/FP32
        routing_weight: Per-token routing weight, shape [M] or [M, 1]
        fp8_dtype: Target FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2)
        block_k: Block size for quantization (default 128)

    Returns:
        out_fp8: Quantized output tensor, shape [M, dim], dtype fp8_dtype
        scale: Per-block scale tensor, shape [M, dim // block_k], dtype float32
    """
    assert AB.dim() == 2, f"Expected 2D input, got {AB.dim()}D"
    M, two_dim = AB.shape
    assert two_dim % 2 == 0, "Last dimension must be even"
    dim = two_dim // 2
    assert dim % block_k == 0, f"dim={dim} must be divisible by block_k={block_k}"

    # Prepare routing weight
    if routing_weight is not None:
        routing_weight = routing_weight.view(-1)
        assert routing_weight.shape[0] == M

    # FP8 info
    finfo = torch.finfo(fp8_dtype)
    fp8_max = finfo.max
    fp8_min = -finfo.max  # For e4m3: -448.0

    # Allocate output
    out_fp8 = torch.empty((M, dim), dtype=fp8_dtype, device=AB.device)
    scale = torch.empty((M, dim // block_k), dtype=torch.float32, device=AB.device)

    # Launch kernel
    num_blocks_per_row = dim // block_k
    grid = (M, num_blocks_per_row)

    # Choose num_warps based on BLOCK_K
    if block_k >= 512:
        num_warps = 8
    elif block_k >= 256:
        num_warps = 4
    else:
        num_warps = 4

    _swiglu_quantize_fp8_kernel[grid](
        AB,
        AB.stride(0),
        routing_weight,
        out_fp8,
        out_fp8.stride(0),
        scale,
        scale.stride(0),
        dim,
        M,
        FP8_MAX=fp8_max,
        FP8_MIN=fp8_min,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )

    return out_fp8, scale
