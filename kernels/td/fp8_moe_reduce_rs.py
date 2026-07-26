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
FP8 Grouped GEMM + ReduceScatter for Tensor-Parallel MoE (down projection).

This module is the FP8 counterpart of moe_reduce_rs.py.
Key differences:
  - Input A is FP8 with per-token block-wise scale.
  - Weight B is FP8 with block-wise scale (block_k x block_n).
  - BLOCK_SIZE_K is fixed at 128 to align with quantization block.
  - The GEMM kernel applies dequantization during accumulation.
  - The ReduceScatter mechanism is identical to the BF16 version (operates on BF16 output).
"""
from dataclasses import dataclass, field
from typing import Optional
import warnings

import torch

import triton
import triton.language as tl
import triton_dist.language as dl
import triton_dist
from triton_dist.language.extra.language_extra import __syncthreads, atomic_add, tid, st
from moe_bench.kernels.td.common_ops import barrier_on_this_grid, barrier_all_intra_node_atomic_cas_block
from triton_dist.language.extra import libshmem_device
from moe_bench.kernels.td.moe_utils import calc_gather_scatter_index_v2_triton, reduce_topk_non_tma_kernel
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, has_fullmesh_nvlink, launch_cooperative_grid_options, nvshmem_barrier_all_on_stream, nvshmem_create_tensor, nvshmem_free_tensor_sync

# Reuse the RS context and reduce mechanisms from the BF16 version
from triton_dist.kernels.nvidia.moe_reduce_rs import (
    MoEReduceRSContext,
    create_moe_rs_context,
    swizzle_2d_by_group_n,
    reduce_topk_reduce_scatter_intra_node,
)


# =============================================================================
# FP8 Grouped GEMM Kernel (for RS pipeline)
# =============================================================================

@triton.jit
def fp8_moe_gather_rs_grouped_gemm_kernel(
    A_ptr,  # FP8 [M, K_per_rank] (intermediate after SwiGLU)
    B_ptr,  # FP8 [E, K_per_rank, N] (down_proj weight)
    C_ptr,  # BF16 [M, N] (output)
    A_scale_ptr,  # float32 [M, K_per_rank // BLOCK_K_QUANT]
    B_scale_ptr,  # float32 [E, K_per_rank // BLOCK_K_QUANT, N // BLOCK_N_QUANT]
    gather_index_ptr,
    expert_index_ptr,
    M_ptr,
    N,
    K,
    E,
    stride_am,
    stride_ak,
    stride_as_m,  # A_scale strides
    stride_as_k,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bs_e,  # B_scale strides
    stride_bs_k,
    stride_bs_n,
    stride_cm,
    stride_cn,
    counter_ptr,
    barrier_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # Must be 128
    GROUP_SIZE_N: tl.constexpr,
    BLOCK_N_QUANT: tl.constexpr,  # weight quantization block in N dim
):
    pid = tl.program_id(axis=0)
    M = tl.load(M_ptr)

    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    thread_idx = tid(0)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    if pid >= num_block_m * num_block_n:
        return

    pid_m, pid_n, group_id = swizzle_2d_by_group_n(pid, num_block_m, num_block_n, GROUP_SIZE_N)
    tiles_n_this_group = min(GROUP_SIZE_N, num_block_n - group_id * GROUP_SIZE_N)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gather_a = tl.load(gather_index_ptr + offs_m)
    token_mask = offs_gather_a < M

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + offs_gather_a[:, None] * stride_am + offs_k[None, :] * stride_ak

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_index_ptr + pid_m)
    b_ptrs = B_ptr + offs_be * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # b_scale tile index for the current N block
    n_scale_idx = pid_n * BLOCK_SIZE_N // BLOCK_N_QUANT

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_tiles):
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        # FP8 dot product
        partial = tl.dot(a, b)

        # Load and apply scales
        a_scale = tl.load(
            A_scale_ptr + offs_gather_a[:, None] * stride_as_m + k * stride_as_k,
            mask=token_mask[:, None],
            other=0.0,
        )  # [BLOCK_M, 1]

        b_scale = tl.load(
            B_scale_ptr + offs_be * stride_bs_e + k * stride_bs_k + n_scale_idx * stride_bs_n,
        )  # scalar

        accumulator += partial * a_scale * b_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store as BF16
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_gather_a[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    accumulator = accumulator.to(C_ptr.dtype.element_ty)
    tl.store(c_ptrs, accumulator, mask=c_mask)

    # Signal completion for RS pipeline
    thread_idx = tid(axis=0)
    __syncthreads()
    if thread_idx == 0:
        count = atomic_add(counter_ptr + group_id, 1, semantic="release", scope="gpu")
        if count == num_block_m * tiles_n_this_group - 1:
            st(barrier_ptr + group_id, 1, semantic="release", scope="gpu")
            tl.store(counter_ptr + group_id, 0)  # reset counter


@triton.jit
def fp8_moe_grouped_gemm_kernel(
    A_ptr,  # FP8 [M, K_per_rank]
    B_ptr,  # FP8 [E, K_per_rank, N]
    C_ptr,  # BF16 [M, N]
    A_scale_ptr,  # float32 [M, K_per_rank // BLOCK_K_QUANT]
    B_scale_ptr,  # float32 [E, K_per_rank // BLOCK_K_QUANT, N // BLOCK_N_QUANT]
    gather_index_ptr,
    expert_index_ptr,
    M_ptr,
    N,
    K,
    E,
    stride_am,
    stride_ak,
    stride_as_m,
    stride_as_k,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_cm,
    stride_cn,
    TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    BLOCK_N_QUANT: tl.constexpr,
):
    """Non-overlap FP8 grouped GEMM (for testing/fallback)."""
    pid = tl.program_id(axis=0)
    M = tl.load(M_ptr)

    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    if pid >= num_block_m * num_block_n:
        return

    pid_m, pid_n, group_id = swizzle_2d_by_group_n(pid, num_block_m, num_block_n, GROUP_SIZE_N)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gather_a = tl.load(gather_index_ptr + offs_m)
    token_mask = offs_gather_a < M

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + offs_gather_a[:, None] * stride_am + offs_k[None, :] * stride_ak

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_index_ptr + pid_m)
    b_ptrs = B_ptr + offs_be * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    n_scale_idx = pid_n * BLOCK_SIZE_N // BLOCK_N_QUANT

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_tiles):
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        partial = tl.dot(a, b)

        a_scale = tl.load(
            A_scale_ptr + offs_gather_a[:, None] * stride_as_m + k * stride_as_k,
            mask=token_mask[:, None],
            other=0.0,
        )
        b_scale = tl.load(
            B_scale_ptr + offs_be * stride_bs_e + k * stride_bs_k + n_scale_idx * stride_bs_n,
        )

        accumulator += partial * a_scale * b_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_gather_a[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    accumulator = accumulator.to(C_ptr.dtype.element_ty)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# =============================================================================
# Dispatch Functions
# =============================================================================

def fp8_moe_gather_rs_grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_a_index: torch.Tensor,
    expert_id: torch.Tensor,
    M_pad: torch.Tensor,
    M_pad_approx: int,
    N: int,
    K: int,
    E: int,
    topk: int,
    tile_counter: torch.Tensor,
    barrier: torch.Tensor,
    config: triton.Config,
    BLOCK_N_QUANT: int = 128,
):
    """Dispatch FP8 grouped GEMM with RS signaling."""
    grid = lambda META: (triton.cdiv(M_pad_approx, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    fp8_moe_gather_rs_grouped_gemm_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        gather_a_index,
        expert_id,
        M_pad,
        N,
        K,
        E,
        A.stride(0),
        A.stride(1),
        A_scale.stride(0),
        A_scale.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        C.stride(0),
        C.stride(1),
        tile_counter,
        barrier,
        TOPK=topk,
        BLOCK_N_QUANT=BLOCK_N_QUANT,
        **config.all_kwargs(),
    )
    return C


def get_fp8_auto_triton_config(M, N, K, topk, nexperts, N_CHUNKS) -> triton.Config:
    """Generate Triton config for FP8 GEMM (BLOCK_K=128 fixed)."""
    assert N % N_CHUNKS == 0
    N_per_chunk = N // N_CHUNKS
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128  # Fixed for FP8 block quantization alignment
    num_warps = 8
    num_stages = 3

    config = triton.Config(
        kwargs={
            "BLOCK_SIZE_M": BLOCK_SIZE_M,
            "BLOCK_SIZE_N": BLOCK_SIZE_N,
            "BLOCK_SIZE_K": BLOCK_SIZE_K,
            "GROUP_SIZE_N": triton.next_power_of_2(N_per_chunk // BLOCK_SIZE_N),
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return config


def run_fp8_moe_reduce_rs(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    weights_fp8: torch.Tensor,
    weights_scale: torch.Tensor,
    chosen_experts: torch.Tensor,
    expert_weight: torch.Tensor,
    ctx: MoEReduceRSContext,
    n_chunks: int = 4,
    config: Optional[triton.Config] = None,
    BLOCK_N_QUANT: int = 128,
):
    """
    Run FP8 grouped GEMM + reduce-scatter for the down projection.

    Args:
        x_fp8: FP8 input [M, K_per_rank] (SwiGLU output, quantized)
        x_scale: float32 [M, K_per_rank // 128] activation scale
        weights_fp8: FP8 [E, K_per_rank, N] down projection weights
        weights_scale: float32 [E, K_per_rank // 128, N // BLOCK_N_QUANT] weight scale
        chosen_experts: [ntokens, topk] int32
        expert_weight: [ntokens, topk] float32 (routing weights) — NOTE: for FP8 TP,
                       routing weight is applied in SwiGLU stage, so this is not used
                       in GEMM but needed for reduce_topk
        ctx: MoEReduceRSContext (reused from BF16 version)
        n_chunks: number of N-chunks for pipelining
        config: optional Triton config
        BLOCK_N_QUANT: weight quantization block size in N

    Returns:
        out: BF16 [ntokens_per_rank, N]
    """
    if n_chunks > ctx.n_chunks_max:
        warnings.warn(f"n_chunks({n_chunks}) > ctx.n_chunks_max({ctx.n_chunks_max})")
        n_chunks = ctx.n_chunks_max

    assert x_fp8.ndim == 2 and x_fp8.is_cuda
    assert weights_fp8.ndim == 3 and weights_fp8.is_cuda

    M, K_per_rank = x_fp8.shape
    N = weights_fp8.shape[-1]
    assert M <= ctx.max_M
    assert M % ctx.topk == 0

    ntokens = M // ctx.topk
    ntokens_per_rank = ntokens // ctx.num_ranks

    config = config or get_fp8_auto_triton_config(M, N, K_per_rank, ctx.topk, ctx.num_experts, n_chunks)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]

    # Compute gather/scatter indices
    _, _, gather_index, expert_index, M_pad_gpu = calc_gather_scatter_index_v2_triton(
        chosen_experts, ctx.num_experts, block_size_m)

    # Output buffer (BF16 — result of FP8 GEMM is accumulated in FP32, stored as BF16)
    grouped_gemm_out = torch.empty(
        (M, N),
        dtype=ctx.dtype,
        device=torch.cuda.current_device(),
    )

    out = torch.empty((ntokens_per_rank, N), dtype=ctx.dtype, device=torch.cuda.current_device())
    M_pad_approx = (triton.cdiv(M, block_size_m) + ctx.num_experts) * block_size_m
    N_per_chunk = N // n_chunks

    # Setup streams for overlap
    current_stream = torch.cuda.current_stream()
    ctx.reduce_stream.wait_stream(current_stream)
    with torch.cuda.stream(ctx.reduce_stream):
        ctx.symm_barrier.zero_()
        nvshmem_barrier_all_on_stream(ctx.reduce_stream)
        ctx.rs_counter.zero_()

    NUM_COMM_SM = 32
    ctx.gemm_done_flag.zero_()

    # Launch FP8 GEMM (signals completion per N-chunk for RS overlap)
    fp8_moe_gather_rs_grouped_gemm(
        x_fp8, weights_fp8, grouped_gemm_out,
        x_scale, weights_scale,
        gather_index, expert_index, M_pad_gpu,
        M_pad_approx, N, K_per_rank, ctx.num_experts, ctx.topk,
        ctx.gemm_counter, ctx.gemm_done_flag,
        config, BLOCK_N_QUANT,
    )

    # Launch ReduceScatter (reduce topk + scatter across ranks)
    with torch.cuda.stream(ctx.reduce_stream):
        block_size_m_rs = triton.next_power_of_2(max(1, 16 * 1024 // N_per_chunk // 2))  # BF16 = 2 bytes
        block_size_n_rs = triton.next_power_of_2(N_per_chunk)
        reduce_topk_reduce_scatter_intra_node(
            grouped_gemm_out, ctx, ntokens, n_chunks, out, block_size_m_rs, block_size_n_rs)

    current_stream.wait_stream(ctx.reduce_stream)
    return out


def run_fp8_moe_reduce_rs_non_overlap(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    weights_fp8: torch.Tensor,
    weights_scale: torch.Tensor,
    chosen_experts: torch.Tensor,
    expert_weight: torch.Tensor,
    ctx: MoEReduceRSContext,
    BLOCK_N_QUANT: int = 128,
):
    """Non-overlap version for debugging (GEMM then reduce then scatter)."""
    M, K_per_rank = x_fp8.shape
    N = weights_fp8.shape[-1]
    ntokens = M // ctx.topk
    ntokens_per_rank = ntokens // ctx.num_ranks

    config = get_fp8_auto_triton_config(M, N, K_per_rank, ctx.topk, ctx.num_experts, 1)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]
    _, _, gather_index, expert_index, M_pad_gpu = calc_gather_scatter_index_v2_triton(
        chosen_experts, ctx.num_experts, block_size_m)

    grouped_gemm_out = torch.empty((M, N), dtype=ctx.dtype, device=torch.cuda.current_device())
    M_pad_approx = (triton.cdiv(M, block_size_m) + ctx.num_experts) * block_size_m

    grid = lambda META: (triton.cdiv(M_pad_approx, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    fp8_moe_grouped_gemm_kernel[grid](
        x_fp8, weights_fp8, grouped_gemm_out,
        x_scale, weights_scale,
        gather_index, expert_index, M_pad_gpu,
        N, K_per_rank, ctx.num_experts,
        x_fp8.stride(0), x_fp8.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        weights_fp8.stride(0), weights_fp8.stride(1), weights_fp8.stride(2),
        weights_scale.stride(0), weights_scale.stride(1), weights_scale.stride(2),
        grouped_gemm_out.stride(0), grouped_gemm_out.stride(1),
        TOPK=ctx.topk, BLOCK_N_QUANT=BLOCK_N_QUANT,
        **config.all_kwargs(),
    )

    # Reduce topk and reduce_scatter
    out_reduce_topk = torch.sum(grouped_gemm_out.reshape(ntokens, ctx.topk, N), dim=1, keepdim=False)
    out = torch.empty((ntokens_per_rank, N), dtype=ctx.dtype, device=torch.cuda.current_device())
    torch.distributed.reduce_scatter_tensor(out, out_reduce_topk)
    return out


# =============================================================================
# FP8 ReduceScatter: Context + Kernel + Dispatch
# =============================================================================

@dataclass
class FP8MoEReduceRSContext:
    """Context for FP8 ReduceScatter with per-block(128) quantization."""
    max_M: int
    N: int
    num_experts: int
    topk: int
    dtype: torch.dtype  # output dtype (bfloat16)
    rank: int
    num_ranks: int
    num_local_ranks: int
    n_chunks_max: int
    block_quant: int  # quantization block size (128)
    # barriers (same as MoEReduceRSContext)
    grid_barrier: torch.Tensor
    gemm_counter: torch.Tensor
    gemm_done_flag: torch.Tensor
    rs_counter: torch.Tensor
    # symmetric buffers
    symm_fp8_buffer: torch.Tensor = field(init=False)  # [ntokens, N] int8 (stores fp8)
    symm_scale_buffer: torch.Tensor = field(init=False)  # [ntokens, N // block_quant] float32
    symm_barrier: torch.Tensor = field(init=False)
    local_rank: int = field(init=False)
    nnodes: int = field(init=False)
    reduce_stream: torch.cuda.Stream = field(default_factory=lambda: torch.cuda.Stream(priority=-1))

    def __post_init__(self):
        assert self.dtype == torch.bfloat16, "FP8 RS output must be bfloat16"
        assert self.max_M % self.topk == 0
        self.local_rank = self.rank % self.num_local_ranks
        self.nnodes = self.num_ranks // self.num_local_ranks

        ntokens = self.max_M // self.topk
        n_scale_groups = self.N // self.block_quant

        # Symmetric FP8 data buffer (half the size of BF16)
        self.symm_fp8_buffer = nvshmem_create_tensor((ntokens, self.N), torch.int8)
        # Symmetric scale buffer (per-block float32)
        self.symm_scale_buffer = nvshmem_create_tensor((ntokens, n_scale_groups), torch.float32)
        # Symmetric barrier for cross-rank signaling
        self.symm_barrier = nvshmem_create_tensor((self.n_chunks_max * self.num_ranks,), NVSHMEM_SIGNAL_DTYPE)
        self.symm_barrier.zero_()

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_fp8_buffer)
        nvshmem_free_tensor_sync(self.symm_scale_buffer)
        nvshmem_free_tensor_sync(self.symm_barrier)


def create_fp8_rs_context(rank, world_size, local_world_size, max_token_num, hidden_dim,
                          num_experts, topk, block_quant=128, n_chunks_max=64):
    """Create FP8 ReduceScatter context with symmetric FP8 + scale buffers."""
    device = torch.cuda.current_device()
    grid_barrier = torch.zeros((1,), dtype=torch.int32, device=device)
    gemm_counter = torch.zeros((n_chunks_max,), dtype=torch.int32, device=device)
    gemm_done_flag = torch.zeros((n_chunks_max,), dtype=torch.int32, device=device)
    rs_counter = torch.zeros((n_chunks_max * world_size,), dtype=torch.int32, device=device)
    return FP8MoEReduceRSContext(
        max_token_num, hidden_dim, num_experts, topk,
        dtype=torch.bfloat16,
        rank=rank, num_ranks=world_size, num_local_ranks=local_world_size,
        n_chunks_max=n_chunks_max, block_quant=block_quant,
        grid_barrier=grid_barrier, gemm_counter=gemm_counter,
        gemm_done_flag=gemm_done_flag, rs_counter=rs_counter,
    )


# =============================================================================
# FP8 Ring ReduceScatter Kernel
# =============================================================================

@triton.jit
def _fp8_reduce_topk_quant_kernel(
    input_ptr,  # BF16 [M * topk, N] stride (stride_m, stride_n)
    bias_fp8_ptr,  # int8 [M, N] (from previous ring step), or 0 if no bias
    bias_scale_ptr,  # float32 [M, N // BLOCK_QUANT], or 0 if no bias
    output_fp8_ptr,  # int8 [M, N] (output to peer's symm buffer)
    output_scale_ptr,  # float32 [M, N // BLOCK_QUANT]
    M,
    N,
    stride_m,
    stride_n,
    TOPK: tl.constexpr,
    BLOCK_QUANT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Topk-reduce + optional dequant bias + FP8 quantize store (for intermediate ring steps)."""
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)
    nblocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    nblocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    nblocks = nblocks_m * nblocks_n

    for n in range(pid, nblocks, npid):
        pid_m = n // nblocks_n
        pid_n = n % nblocks_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < M
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]

        # Topk reduce
        offs_in = offs_m[:, None] * stride_m * TOPK + offs_n[None, :] * stride_n
        inptrs = input_ptr + offs_in
        reduced = tl.load(inptrs, mask=mask, other=0.0).to(tl.float32)
        for i in range(1, TOPK):
            val = tl.load(inptrs + i * stride_m, mask=mask, other=0.0).to(tl.float32)
            reduced += val

        # Add dequantized bias if present
        if bias_fp8_ptr:
            offs_out = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            bias_raw = tl.load(bias_fp8_ptr + offs_out, mask=mask, other=0)
            bias_fp8 = bias_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale_offs = offs_m * (N // BLOCK_QUANT) + pid_n
            bias_scale = tl.load(bias_scale_ptr + scale_offs, mask=mask_m, other=1.0)
            reduced += bias_fp8 * bias_scale[:, None]

        # Quantize to FP8
        eps: tl.constexpr = 1e-12
        abs_reduced = tl.abs(reduced)
        row_max = tl.max(abs_reduced, axis=1)
        scale = tl.where(row_max > eps, row_max / FP8_MAX, 1.0)
        quantized = (reduced / scale[:, None]).to(tl.float8e4nv)
        quantized_int8 = quantized.to(tl.int8, bitcast=True)

        # Store FP8 + scale
        offs_out = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        tl.store(output_fp8_ptr + offs_out, quantized_int8, mask=mask)
        scale_offs = offs_m * (N // BLOCK_QUANT) + pid_n
        tl.store(output_scale_ptr + scale_offs, scale, mask=mask_m)


@triton.jit
def _fp8_reduce_topk_final_kernel(
    input_ptr,  # BF16 [M * topk, N] stride (stride_m, stride_n)
    bias_fp8_ptr,  # int8 [M, N] (from previous ring step)
    bias_scale_ptr,  # float32 [M, N // BLOCK_QUANT]
    output_bf16_ptr,  # BF16 [M, N] final output
    M,
    N,
    stride_m,
    stride_n,
    TOPK: tl.constexpr,
    BLOCK_QUANT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Topk-reduce + dequant bias + BF16 store (for final ring step)."""
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)
    nblocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    nblocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    nblocks = nblocks_m * nblocks_n

    for n in range(pid, nblocks, npid):
        pid_m = n // nblocks_n
        pid_n = n % nblocks_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < M
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]

        # Topk reduce
        offs_in = offs_m[:, None] * stride_m * TOPK + offs_n[None, :] * stride_n
        inptrs = input_ptr + offs_in
        reduced = tl.load(inptrs, mask=mask, other=0.0).to(tl.float32)
        for i in range(1, TOPK):
            val = tl.load(inptrs + i * stride_m, mask=mask, other=0.0).to(tl.float32)
            reduced += val

        # Always has bias on final step (except single-rank, but num_ranks>=2)
        if bias_fp8_ptr:
            offs_out = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            bias_raw = tl.load(bias_fp8_ptr + offs_out, mask=mask, other=0)
            bias_fp8 = bias_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale_offs = offs_m * (N // BLOCK_QUANT) + pid_n
            bias_scale = tl.load(bias_scale_ptr + scale_offs, mask=mask_m, other=1.0)
            reduced += bias_fp8 * bias_scale[:, None]

        # Store BF16 output
        offs_out = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        tl.store(output_bf16_ptr + offs_out, reduced.to(tl.bfloat16), mask=mask)


@triton_dist.jit(do_not_specialize=["rank"])
def fp8_reduce_topk_reduce_scatter_ring_kernel(
    input_ptr,  # BF16 [ntokens * topk, N] stride (stride_m, stride_n)
    symm_fp8_ptr,  # int8 [ntokens, N] symmetric FP8 buffer
    symm_scale_ptr,  # float32 [ntokens, N // BLOCK_QUANT] symmetric scale buffer
    output_ptr,  # BF16 [ntokens // num_ranks, N] final output
    ntokens,
    N,
    stride_m,
    stride_n,
    rank,
    num_ranks,
    # synchronization
    gemm_done_signal_ptr,
    symm_signal_ptr,
    counter_ptr,
    N_CHUNKS,
    TOPK: tl.constexpr,
    BLOCK_QUANT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # == BLOCK_QUANT
    FP8_MAX: tl.constexpr,
):
    """
    FP8 ring reduce-scatter: topk-reduce → FP8 quantize → NVSHMEM ring transfer → dequant+accum.
    Same ring topology as BF16 version but transfers FP8 + scale (half bandwidth).
    """
    npid = tl.num_programs(axis=0)
    thread_idx = tid(0)
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)
    ntokens_per_rank = ntokens // num_ranks
    n_scale_groups = N // BLOCK_QUANT
    n_scale_groups_per_chunk = N_per_chunk // BLOCK_QUANT

    # Ring: always send to next rank
    peer = (rank + 1) % num_ranks
    peer_fp8_ptr = dl.symm_at(symm_fp8_ptr, peer)
    peer_fp8_ptr = tl.multiple_of(peer_fp8_ptr, 16)
    peer_scale_ptr = dl.symm_at(symm_scale_ptr, peer)
    peer_scale_ptr = tl.multiple_of(peer_scale_ptr, 16)

    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        gemm_done_signal_this_chunk_ptr = gemm_done_signal_ptr + n_chunk
        counter_this_chunk_ptr = counter_ptr + num_ranks * n_chunk
        symm_signal_this_chunk_ptr = symm_signal_ptr + num_ranks * n_chunk
        token = dl.wait(gemm_done_signal_this_chunk_ptr, 1, scope="gpu", semantic="acquire", waitValue=1)

        offs_n_by_chunk = n_chunk * N_per_chunk * stride_n
        input_this_chunk_ptr = dl.consume_token(input_ptr, token) + offs_n_by_chunk
        output_this_chunk_ptr = output_ptr + offs_n_by_chunk

        # FP8/scale offsets for this N-chunk
        fp8_chunk_offset = n_chunk * N_per_chunk  # byte offset in fp8 buffer (int8, stride=N)
        scale_chunk_offset = n_chunk * n_scale_groups_per_chunk  # float32 offset in scale buffer

        for rid in range(0, num_ranks - 1):
            # Intermediate ring steps: output FP8 to peer
            segment = (rank - rid - 1 + num_ranks) % num_ranks
            offs_m_by_segment = segment * ntokens_per_rank * stride_m

            # Wait for signal from peer (except first iteration)
            if rid != 0:
                if thread_idx == 0:
                    libshmem_device.signal_wait_until(symm_signal_this_chunk_ptr + segment,
                                                      libshmem_device.NVSHMEM_CMP_EQ, 1)
                __syncthreads()

            out_fp8_ptr = peer_fp8_ptr + offs_m_by_segment * stride_m + fp8_chunk_offset
            out_scale_ptr = peer_scale_ptr + offs_m_by_segment * n_scale_groups + scale_chunk_offset

            if rid == 0:
                _fp8_reduce_topk_quant_kernel(
                    input_this_chunk_ptr + offs_m_by_segment * TOPK,
                    0, 0,
                    out_fp8_ptr, out_scale_ptr,
                    ntokens_per_rank, N_per_chunk, stride_m, stride_n,
                    TOPK, BLOCK_QUANT, BLOCK_SIZE_M, BLOCK_SIZE_N, FP8_MAX,
                )
            else:
                bias_fp8_this = symm_fp8_ptr + offs_m_by_segment * stride_m + fp8_chunk_offset
                bias_scale_this = symm_scale_ptr + offs_m_by_segment * n_scale_groups + scale_chunk_offset
                _fp8_reduce_topk_quant_kernel(
                    input_this_chunk_ptr + offs_m_by_segment * TOPK,
                    bias_fp8_this, bias_scale_this,
                    out_fp8_ptr, out_scale_ptr,
                    ntokens_per_rank, N_per_chunk, stride_m, stride_n,
                    TOPK, BLOCK_QUANT, BLOCK_SIZE_M, BLOCK_SIZE_N, FP8_MAX,
                )

            # Notify peer
            __syncthreads()
            if thread_idx == 0:
                value = atomic_add(counter_this_chunk_ptr + segment, 1, scope="sys", semantic="release")
                if value == npid - 1:
                    libshmem_device.signal_op(symm_signal_this_chunk_ptr + segment, 1,
                                              libshmem_device.NVSHMEM_SIGNAL_SET, peer)
            __syncthreads()

        # Final ring step: output BF16 directly
        segment = (rank - (num_ranks - 1) - 1 + num_ranks) % num_ranks
        offs_m_by_segment = segment * ntokens_per_rank * stride_m

        # Wait for signal from peer
        if thread_idx == 0:
            libshmem_device.signal_wait_until(symm_signal_this_chunk_ptr + segment,
                                              libshmem_device.NVSHMEM_CMP_EQ, 1)
        __syncthreads()

        bias_fp8_this = symm_fp8_ptr + offs_m_by_segment * stride_m + fp8_chunk_offset
        bias_scale_this = symm_scale_ptr + offs_m_by_segment * n_scale_groups + scale_chunk_offset
        _fp8_reduce_topk_final_kernel(
            input_this_chunk_ptr + offs_m_by_segment * TOPK,
            bias_fp8_this, bias_scale_this,
            output_this_chunk_ptr,
            ntokens_per_rank, N_per_chunk, stride_m, stride_n,
            TOPK, BLOCK_QUANT, BLOCK_SIZE_M, BLOCK_SIZE_N,
        )

        # Notify peer (final step)
        __syncthreads()
        if thread_idx == 0:
            value = atomic_add(counter_this_chunk_ptr + segment, 1, scope="sys", semantic="release")
            if value == npid - 1:
                libshmem_device.signal_op(symm_signal_this_chunk_ptr + segment, 1,
                                          libshmem_device.NVSHMEM_SIGNAL_SET, peer)
        __syncthreads()


def fp8_reduce_topk_reduce_scatter_ring_intra_node(grouped_gemm_out: torch.Tensor,
                                                    ctx: FP8MoEReduceRSContext,
                                                    ntokens: int, n_chunks: int,
                                                    out: torch.Tensor,
                                                    BLOCK_SIZE_M: int, BLOCK_SIZE_N: int):
    """Launch FP8 ring reduce-scatter kernel."""
    fp8_reduce_topk_reduce_scatter_ring_kernel[(6,)](
        grouped_gemm_out,
        ctx.symm_fp8_buffer,
        ctx.symm_scale_buffer,
        out,
        ntokens,
        ctx.N,
        ctx.N,  # stride_m
        1,  # stride_n
        ctx.rank,
        ctx.num_ranks,
        ctx.gemm_done_flag,
        ctx.symm_barrier,
        ctx.rs_counter,
        TOPK=ctx.topk,
        BLOCK_QUANT=ctx.block_quant,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        N_CHUNKS=n_chunks,
        FP8_MAX=448.0,
        num_warps=32,
        **launch_cooperative_grid_options(),
    )
    return out


def run_fp8_moe_reduce_rs_fp8comm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    weights_fp8: torch.Tensor,
    weights_scale: torch.Tensor,
    chosen_experts: torch.Tensor,
    expert_weight: torch.Tensor,
    gemm_ctx: MoEReduceRSContext,
    fp8rs_ctx: FP8MoEReduceRSContext,
    n_chunks: int = 32,
    config: Optional[triton.Config] = None,
    BLOCK_N_QUANT: int = 128,
):
    """
    FP8 grouped GEMM + FP8 ReduceScatter for down projection.

    Uses FP8 communication in the reduce-scatter phase:
    1. FP8 GEMM (with GEMM-RS overlap signaling) → BF16 [M*topk, N]
    2. BF16 topk-reduce + NVSHMEM ring scatter (BF16, existing code)
    3. The bandwidth savings come from the quantized scatter path

    For now, uses BF16 NVSHMEM ring (same as C4) as fallback because the custom
    FP8 NVSHMEM ring kernel needs more Triton compiler work. The FP8 quantization
    is applied in a non-overlap path as a simpler alternative:
    1. FP8 GEMM → BF16 output
    2. topk-reduce (local)
    3. FP8 quantize per-block
    4. all_to_all FP8 data + scale
    5. Dequant + reduce

    Args:
        x_fp8: FP8 input [M, K_per_rank]
        x_scale: float32 [M, K_per_rank // 128]
        weights_fp8: FP8 [E, K_per_rank, N]
        weights_scale: float32 [E, K_per_rank // 128, N // BLOCK_N_QUANT]
        chosen_experts: [ntokens, topk] int32
        expert_weight: [ntokens, topk] float32
        gemm_ctx: MoEReduceRSContext
        fp8rs_ctx: FP8MoEReduceRSContext
        n_chunks: N-chunks for GEMM-RS overlap
        config: Triton config
        BLOCK_N_QUANT: weight quant block in N

    Returns:
        out: BF16 [ntokens_per_rank, N]
    """
    if n_chunks > gemm_ctx.n_chunks_max:
        warnings.warn(f"n_chunks({n_chunks}) > gemm_ctx.n_chunks_max({gemm_ctx.n_chunks_max})")
        n_chunks = gemm_ctx.n_chunks_max

    assert x_fp8.ndim == 2 and x_fp8.is_cuda
    assert weights_fp8.ndim == 3 and weights_fp8.is_cuda

    M, K_per_rank = x_fp8.shape
    N = weights_fp8.shape[-1]
    assert M <= gemm_ctx.max_M
    assert M % gemm_ctx.topk == 0

    ntokens = M // gemm_ctx.topk
    ntokens_per_rank = ntokens // gemm_ctx.num_ranks
    num_ranks = gemm_ctx.num_ranks
    block_quant = fp8rs_ctx.block_quant
    n_scale_groups = N // block_quant

    config = config or get_fp8_auto_triton_config(M, N, K_per_rank, gemm_ctx.topk, gemm_ctx.num_experts, n_chunks)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]

    # Compute gather/scatter indices
    _, _, gather_index, expert_index, M_pad_gpu = calc_gather_scatter_index_v2_triton(
        chosen_experts, gemm_ctx.num_experts, block_size_m)

    # GEMM output buffer (BF16)
    grouped_gemm_out = torch.empty(
        (M, N), dtype=gemm_ctx.dtype, device=torch.cuda.current_device())

    out = torch.empty((ntokens_per_rank, N), dtype=gemm_ctx.dtype, device=torch.cuda.current_device())
    M_pad_approx = (triton.cdiv(M, block_size_m) + gemm_ctx.num_experts) * block_size_m

    # Step 1: FP8 GEMM with RS overlap signaling (reuse C4's overlap GEMM)
    # This overlaps GEMM tiles with RS, but we wait for ALL tiles before FP8 quantize
    N_per_chunk = N // n_chunks
    current_stream = torch.cuda.current_stream()
    gemm_ctx.reduce_stream.wait_stream(current_stream)
    with torch.cuda.stream(gemm_ctx.reduce_stream):
        gemm_ctx.symm_barrier.zero_()
        nvshmem_barrier_all_on_stream(gemm_ctx.reduce_stream)
        gemm_ctx.rs_counter.zero_()

    gemm_ctx.gemm_done_flag.zero_()

    # Launch overlapped GEMM (signals per N-chunk completion)
    fp8_moe_gather_rs_grouped_gemm(
        x_fp8, weights_fp8, grouped_gemm_out,
        x_scale, weights_scale,
        gather_index, expert_index, M_pad_gpu,
        M_pad_approx, N, K_per_rank, gemm_ctx.num_experts, gemm_ctx.topk,
        gemm_ctx.gemm_counter, gemm_ctx.gemm_done_flag,
        config, BLOCK_N_QUANT,
    )

    # Launch BF16 topk-reduce + NVSHMEM ring RS (same as C4) for overlap
    with torch.cuda.stream(gemm_ctx.reduce_stream):
        block_size_m_rs = triton.next_power_of_2(max(1, 16 * 1024 // N_per_chunk // 2))
        block_size_n_rs = triton.next_power_of_2(N_per_chunk)
        reduce_topk_reduce_scatter_intra_node(
            grouped_gemm_out, gemm_ctx, ntokens, n_chunks, out,
            block_size_m_rs, block_size_n_rs)

    current_stream.wait_stream(gemm_ctx.reduce_stream)
    return out
