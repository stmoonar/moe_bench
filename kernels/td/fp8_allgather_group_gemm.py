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
FP8 AllGather + Grouped GEMM for Tensor-Parallel MoE (tile-level overlap).

This module mirrors allgather_group_gemm.py but operates in FP8:
  - NVSHMEM symmetric workspace stores FP8 data (1 byte/elem, halving comm).
  - Activation scale is AllGathered separately via NCCL (small, ~3% volume).
  - Consumer GEMM kernel waits for AG tiles then runs FP8 dequant-GEMM.
  - BLOCK_SIZE_K=128 to align with quantization block_k.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch

import triton
import triton.language as tl
import triton_dist
from moe_bench.kernels.td.common_ops import _set_signal_cuda
import triton_dist.language as dl
from triton_dist.language.extra.language_extra import __syncthreads
from triton_dist.kernels.nvidia.allgather import (AllGatherMethod, cp_engine_producer_all_gather_inter_node,
                                                  cp_engine_producer_all_gather_intra_node, get_auto_all_gather_method)
from moe_bench.kernels.td.common_ops import (barrier_on_this_grid, next_power_of_2)
from triton_dist.kernels.nvidia.threadblock_swizzle_ag_moe_triton import \
    threadblock_swizzle_ag_moe_kernel
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, launch_cooperative_grid_options, nvshmem_barrier_all_on_stream, nvshmem_create_tensors, nvshmem_free_tensor_sync

# Reuse the sorting utilities from the BF16 version
from triton_dist.kernels.nvidia.allgather_group_gemm import (
    sort_topk_ids_align_block_size,
    swizzle_2d,
    _copy_and_reset_and_barrier_all_kernel,
)
from moe_bench.kernels.td.moe_utils import calc_gather_scatter_index_v2_triton


# =============================================================================
# FP8 AG Context
# =============================================================================

@dataclass
class FP8MoEAllGatherGroupGEMMContext:
    """Context for FP8 AllGather + Grouped GEMM with tile-level overlap."""
    # problem size
    max_ntokens: int
    N_per_rank: int
    K: int
    num_experts: int
    topk: int
    fp8_dtype: torch.dtype  # torch.float8_e4m3fn
    output_dtype: torch.dtype  # typically torch.bfloat16
    BLOCK_K_QUANT: int  # quantization block size, typically 128
    # parallelism info
    rank: int
    num_ranks: int
    num_local_ranks: int = 8
    is_multinode: bool = field(init=False)
    n_nodes: int = field(init=False)
    node_rank: int = field(init=False)
    local_rank: int = field(init=False)
    # distributed mem (FP8 data via NVSHMEM)
    symm_workspaces: List[torch.Tensor] = field(init=False)
    symm_barriers: List[torch.Tensor] = field(init=False)
    symm_workspace: torch.Tensor = field(init=False)
    symm_barrier: torch.Tensor = field(init=False)
    barrier_target = 1
    grid_barrier: torch.Tensor = field(init=False)
    # async streams
    ag_intranode_stream: Optional[torch.cuda.streams.Stream] = None
    ag_internode_stream: Optional[torch.cuda.streams.Stream] = None
    # triton compute kernel config
    BLOCK_M: int = 128
    BLOCK_N: int = 128
    BLOCK_K: int = 128  # Must be 128 to align with FP8 block quantization
    GROUP_SIZE_M: int = 8
    stages: int = 4
    warps: int = 8
    all_gather_method: AllGatherMethod = AllGatherMethod.Auto

    def __post_init__(self):
        assert self.num_ranks % self.num_local_ranks == 0
        assert self.BLOCK_K == self.BLOCK_K_QUANT
        assert self.K % self.BLOCK_K_QUANT == 0

        self.is_multinode = self.num_ranks > self.num_local_ranks
        self.n_nodes = self.num_ranks // self.num_local_ranks
        self.node_rank = self.rank // self.num_local_ranks
        self.local_rank = self.rank % self.num_local_ranks
        self.grid_barrier = torch.zeros((1, ), dtype=torch.int32, device="cuda")

        # FP8 symmetric workspace: [max_ntokens, K] in fp8_dtype
        self.symm_workspaces = nvshmem_create_tensors(
            (self.max_ntokens, self.K), self.fp8_dtype, self.rank, self.num_local_ranks)
        self.symm_workspace = self.symm_workspaces[self.local_rank]

        # Barriers (same as BF16 version)
        self.symm_barriers = nvshmem_create_tensors(
            (self.num_ranks, ), NVSHMEM_SIGNAL_DTYPE, self.rank, self.num_local_ranks)
        self.symm_barrier = self.symm_barriers[self.local_rank]
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    def finalize(self):
        nvshmem_free_tensor_sync(self.symm_barrier)
        nvshmem_free_tensor_sync(self.symm_workspace)

    def copy_and_reset_and_barrier_all_triton(self, local_data_fp8: torch.Tensor):
        """Copy local FP8 data to symmetric workspace, reset barriers.

        Uses the same kernel as the BF16 version (_copy_and_reset_and_barrier_all_kernel)
        since it operates on raw bytes.
        """
        elems_per_rank = local_data_fp8.numel()
        assert local_data_fp8.is_contiguous() and local_data_fp8.is_cuda
        _copy_and_reset_and_barrier_all_kernel[(16, )](
            local_data_fp8.view(torch.int8),
            self.symm_workspace.view(torch.int8).flatten()[
                elems_per_rank * self.rank:elems_per_rank * (self.rank + 1)],
            elems_per_rank,
            self.symm_barrier,
            self.rank,
            self.symm_barrier.numel(),
            self.grid_barrier,
            BLOCK_SIZE=1024 * 16,
            num_warps=32,
            use_cooperative=True,
            **launch_cooperative_grid_options(),
        )


def create_fp8_ag_group_gemm_context(
    max_ntokens,
    N_per_rank,
    K,
    num_experts,
    topk,
    fp8_dtype: torch.dtype,
    output_dtype: torch.dtype,
    rank,
    num_ranks,
    num_local_ranks,
    BLOCK_K_QUANT: int = 128,
    ag_intranode_stream: Optional[torch.cuda.Stream] = None,
    ag_internode_stream: Optional[torch.cuda.Stream] = None,
    BLOCK_SIZE_M=128,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=128,
    GROUP_SIZE_M=8,
    stages=4,
    num_warps=8,
) -> FP8MoEAllGatherGroupGEMMContext:
    """Create context for FP8 AllGather + Grouped GEMM (tile-level overlap)."""
    ctx = FP8MoEAllGatherGroupGEMMContext(
        max_ntokens=max_ntokens,
        N_per_rank=N_per_rank,
        K=K,
        num_experts=num_experts,
        topk=topk,
        fp8_dtype=fp8_dtype,
        output_dtype=output_dtype,
        BLOCK_K_QUANT=BLOCK_K_QUANT,
        rank=rank,
        num_ranks=num_ranks,
        num_local_ranks=num_local_ranks,
        ag_intranode_stream=ag_intranode_stream or torch.cuda.Stream(),
        ag_internode_stream=ag_internode_stream or torch.cuda.Stream(),
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_N=BLOCK_SIZE_N,
        BLOCK_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        stages=stages,
        warps=num_warps,
        all_gather_method=get_auto_all_gather_method(num_ranks, num_local_ranks),
    )
    return ctx


# =============================================================================
# FP8 Consumer GEMM Kernel (tile-level overlap with AG)
# =============================================================================

@triton.jit(do_not_specialize=["rank"])
def fp8_kernel_consumer_ag_group_gemm(
    a_ptr,  # FP8 [ntokens, K] (symmetric workspace, being allgathered)
    a_scale_ptr,  # float32 [ntokens, K // BLOCK_K_QUANT] (already fully allgathered)
    b_ptr,  # FP8 [E, K, N_per_rank]
    b_scale_ptr,  # float32 [E, K // BLOCK_K_QUANT, N_per_rank // BLOCK_N_QUANT]
    c_ptr,  # output BF16 [M, N_per_rank] where M = ntokens * topk
    block_barrier_ptr,  # NVSHMEM barrier signals
    sorted_token_ids_ptr,  # sorted gather index
    expert_ids_ptr,  # expert id per tile
    tiled_m_ptr,  # tile M index
    segment_start_ptr,  # AG segment start for this tile
    segment_end_ptr,  # AG segment end for this tile
    ntiles_pad_ptr,  # total number of padded tiles
    M,  # ntokens * topk (total expanded tokens)
    N,
    K,
    stride_am,
    stride_ak,
    stride_as_m,  # a_scale strides
    stride_as_k,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bs_e,  # b_scale strides
    stride_bs_k,
    stride_bs_n,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # Must be 128
    GROUP_SIZE_M: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_N_QUANT: tl.constexpr,
):
    """FP8 consumer GEMM kernel with tile-level AG overlap.

    Identical structure to BF16 kernel_consumer_m_parallel_scatter_group_gemm
    but with FP8 dequantization (a_scale * b_scale applied per K-tile).
    """
    pid = tl.program_id(axis=0)
    ntiles_pad = tl.load(ntiles_pad_ptr)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    npid = tl.num_programs(axis=0)
    num_block_m = npid // num_block_n

    pid_m, pid_n = swizzle_2d(pid, num_block_m, num_block_n, GROUP_SIZE_M)

    if pid_m >= ntiles_pad:
        return

    tiled_m = tl.load(tiled_m_ptr + pid_m)
    offs_token_id = tiled_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = (offs_token < M) & (offs_token >= 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # Index into A using token // TOP_K (map expanded token to actual hidden state)
    a_ptrs = (a_ptr + offs_token[:, None] // TOP_K * stride_am + offs_k[None, :] * stride_ak)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_ids_ptr + pid_m)
    segment_start = tl.load(segment_start_ptr + pid_m)
    segment_end = tl.load(segment_end_ptr + pid_m)
    __syncthreads()

    b_ptrs = (b_ptr + offs_be * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ─── Wait for AG data to be ready (tile-level overlap) ───
    token = dl.wait(block_barrier_ptr + segment_start, segment_end - segment_start + 1, "gpu", "acquire")
    __syncthreads()
    a_ptrs = dl.consume_token(a_ptrs, token)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # b_scale tile index for the current N block
    n_scale_idx = pid_n * BLOCK_SIZE_N // BLOCK_N_QUANT

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_tiles):
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        # FP8 dot product → FP32
        partial = tl.dot(a, b)

        # Load activation scale: [BLOCK_M, 1]
        a_scale = tl.load(
            a_scale_ptr + offs_token[:, None] // TOP_K * stride_as_m + k * stride_as_k,
            mask=token_mask[:, None],
            other=0.0,
        )

        # Load weight scale: scalar (one per k-tile × n-tile × expert)
        b_scale = tl.load(
            b_scale_ptr + offs_be * stride_bs_e + k * stride_bs_k + n_scale_idx * stride_bs_n,
        )

        # Dequantize and accumulate
        accumulator += partial * a_scale * b_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store as BF16
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# =============================================================================
# Non-overlap FP8 Grouped GEMM Kernel (fallback / testing)
# =============================================================================

@triton.jit
def fp8_grouped_gemm_kernel(
    a_ptr,  # FP8 [ntokens, K] (allgathered)
    a_scale_ptr,  # float32 [ntokens, K // BLOCK_K_QUANT]
    b_ptr,  # FP8 [E, K, N_per_rank]
    b_scale_ptr,  # float32 [E, K // BLOCK_K_QUANT, N_per_rank // BLOCK_N_QUANT]
    c_ptr,  # output BF16 [M, N_per_rank]
    gather_index_ptr,
    expert_index_ptr,
    M_ptr,
    N,
    K,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_N_QUANT: tl.constexpr,
):
    """Non-overlap FP8 Grouped GEMM (uses calc_gather_scatter_index_v2_triton indexing)."""
    pid = tl.program_id(axis=0)
    M = tl.load(M_ptr)

    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    if pid >= num_block_m * num_block_n:
        return

    # Swizzle
    num_pid_in_group = GROUP_SIZE_N * num_block_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_N
    group_size_m = min(num_block_m - first_pid_m, GROUP_SIZE_N)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gather = tl.load(gather_index_ptr + offs_m, mask=offs_m < M, other=0x7FFFFFFF)
    token_mask = offs_gather < M
    a_token_idx = offs_gather // TOP_K

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + a_token_idx[:, None] * stride_am + offs_k[None, :] * stride_ak

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_index_ptr + pid_m)
    b_ptrs = b_ptr + offs_be * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    n_scale_idx = pid_n * BLOCK_SIZE_N // BLOCK_N_QUANT

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
        partial = tl.dot(a, b)
        a_scale = tl.load(
            a_scale_ptr + a_token_idx[:, None] * stride_as_m + k * stride_as_k,
            mask=token_mask[:, None], other=0.0)
        b_scale = tl.load(
            b_scale_ptr + offs_be * stride_bs_e + k * stride_bs_k + n_scale_idx * stride_bs_n)
        accumulator += partial * a_scale * b_scale
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_gather[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# =============================================================================
# Entry Point: Tile-level overlap version
# =============================================================================

def fp8_ag_group_gemm(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
    ctx: FP8MoEAllGatherGroupGEMMContext,
    full_topk_ids: torch.Tensor,
):
    """Run FP8 AllGather + Grouped GEMM with tile-level AG-GEMM overlap.

    Flow (optimized for overlap):
      1. Async AllGather scale via NCCL on ag_intranode_stream (overlaps with step 2).
      2. Copy local FP8 data to NVSHMEM symmetric workspace + barrier (on compute stream).
      3. Launch NVSHMEM FP8 AG producer on ag_intranode_stream (after scale AG completes on same stream).
      4. Launch consumer FP8 GEMM kernel on compute stream (waits per-tile for NVSHMEM barrier).
         Scale is guaranteed ready since ag_intranode_stream completes scale AG before FP8 AG,
         and consumer GEMM only reads scale after dl.wait confirms FP8 data arrival.

    Args:
        a_fp8: Local FP8 input [M_per_rank, K]
        a_scale: Local activation scale [M_per_rank, K // BLOCK_K_QUANT]
        b_fp8: FP8 weight [E, K, N_per_rank]
        b_scale: Weight scale [E, K // BLOCK_K_QUANT, N_per_rank // BLOCK_N_QUANT]
        ctx: FP8MoEAllGatherGroupGEMMContext
        full_topk_ids: All-gathered topk ids [ntokens, topk]

    Returns:
        c: Output [ntokens * topk, N_per_rank] in output_dtype
    """
    ntokens_per_rank, K = a_fp8.shape
    ntokens = ntokens_per_rank * ctx.num_ranks
    M = ntokens * ctx.topk

    c = torch.empty([M, ctx.N_per_rank], dtype=ctx.output_dtype, device=a_fp8.device)

    # Allocate scale AG output buffer (small: ~160KB for typical configs)
    ag_scale = torch.empty(ntokens, K // ctx.BLOCK_K_QUANT, dtype=torch.float32, device=a_fp8.device)

    # Step 1: Launch async NCCL allgather for scale (non-blocking, overlaps with steps 2-3)
    scale_ag_handle = torch.distributed.all_gather_into_tensor(
        ag_scale, a_scale.contiguous(), async_op=True)

    # Step 2: Copy local FP8 data to NVSHMEM workspace + barrier (overlaps with scale AG)
    ctx.copy_and_reset_and_barrier_all_triton(a_fp8)

    # Step 3: Sort topk ids for grouped GEMM tile scheduling (overlaps with scale AG)
    sorted_gather_index, expert_idx, tiled_m, segment_start, segment_end, ntiles_gpu = \
        sort_topk_ids_align_block_size(
            full_topk_ids,
            ctx.num_experts,
            ctx.rank,
            ctx.num_ranks,
            ctx.num_local_ranks,
            ctx.BLOCK_M,
        )

    # Step 4: Wait for scale AG to complete before launching GEMM consumer
    scale_ag_handle.wait()

    # Step 5: Launch NVSHMEM FP8 AG producer on separate stream
    current_stream = torch.cuda.current_stream()
    if ctx.is_multinode:
        ctx.ag_internode_stream.wait_stream(current_stream)
    ctx.ag_intranode_stream.wait_stream(current_stream)

    if not ctx.is_multinode:
        cp_engine_producer_all_gather_intra_node(
            ctx.rank,
            ctx.num_ranks,
            a_fp8,
            ctx.symm_workspaces,
            ctx.symm_barriers,
            ctx.ag_intranode_stream,
            all_gather_method=ctx.all_gather_method,
        )
    else:
        cp_engine_producer_all_gather_inter_node(
            a_fp8, ctx.symm_workspaces, ctx.symm_barriers, ctx.barrier_target,
            ctx.rank, ctx.num_local_ranks, ctx.num_ranks,
            ctx.ag_intranode_stream, ctx.ag_internode_stream,
            all_gather_method=ctx.all_gather_method)

    # AG buffer contains all tokens' FP8 data
    local_ag_buffer = ctx.symm_workspace[:ntokens]

    # Step 6: Launch consumer FP8 GEMM kernel (overlaps with AG)
    # The kernel uses dl.wait to synchronize per-tile with AG completion.
    # Scale is guaranteed ready because:
    #   - Scale AG completes on ag_intranode_stream BEFORE FP8 AG producer starts
    #   - Consumer only reads scale AFTER dl.wait confirms FP8 data arrived
    #   - FP8 data can only arrive after FP8 AG producer runs (which is after scale AG)
    BLOCK_N_QUANT = ctx.N_per_rank // b_scale.shape[2] if b_scale.dim() == 3 else 128

    grid = lambda META: ((triton.cdiv(M, META["BLOCK_SIZE_M"]) + ctx.num_experts - 1) * triton.cdiv(
        ctx.N_per_rank, META["BLOCK_SIZE_N"]), )

    fp8_kernel_consumer_ag_group_gemm[grid](
        local_ag_buffer,
        ag_scale,
        b_fp8,
        b_scale,
        c,
        ctx.symm_barrier,
        sorted_gather_index,
        expert_idx,
        tiled_m,
        segment_start,
        segment_end,
        ntiles_gpu,
        M,
        ctx.N_per_rank,
        ctx.K,
        local_ag_buffer.stride(0),
        local_ag_buffer.stride(1),
        ag_scale.stride(0),
        ag_scale.stride(1),
        b_fp8.stride(0),
        b_fp8.stride(1),
        b_fp8.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        c.stride(0),
        c.stride(1),
        ctx.BLOCK_M,
        ctx.BLOCK_N,
        ctx.BLOCK_K,
        ctx.GROUP_SIZE_M,
        ctx.topk,
        BLOCK_N_QUANT,
        num_stages=ctx.stages,
        num_warps=ctx.warps,
    )

    # Wait for AG streams to complete
    if ctx.is_multinode:
        current_stream.wait_stream(ctx.ag_internode_stream)
    current_stream.wait_stream(ctx.ag_intranode_stream)

    return c
