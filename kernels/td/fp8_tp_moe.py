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
FP8 Tensor Parallel MoE Layer.

This is the FP8 version of tp_moe.py. It uses:
  1. FP8 AllGather + FP8 Grouped GEMM (gate_up projection)
  2. Fused SwiGLU + FP8 quantization
  3. FP8 Grouped GEMM + ReduceScatter (down projection)

Token quantization: row-wise (block_k=128)
Weight quantization: block-wise (block_k=128, block_n=128)
"""

import torch
from torch import nn
import torch.distributed

import triton
import triton.language as tl

from moe_bench.kernels.td.fp8_allgather_group_gemm import (
    create_fp8_ag_group_gemm_context,
    fp8_ag_group_gemm,
    FP8MoEAllGatherGroupGEMMContext,
)
from moe_bench.kernels.td.fp8_moe_reduce_rs import (
    run_fp8_moe_reduce_rs,
    create_fp8_rs_context,
    run_fp8_moe_reduce_rs_fp8comm,
    FP8MoEReduceRSContext,
)
from triton_dist.kernels.nvidia.moe_reduce_rs import create_moe_rs_context
from moe_bench.kernels.td.swiglu_quantize_fp8 import swiglu_quantize_fp8


_FP8_AUTOTUNE_KERNELS_INSTALLED = False


def _make_fp8_autotuned_ops():
    """按 Triton-distributed TP_MoE 的方式包装两个分布式 GEMM。"""
    global _FP8_AUTOTUNE_KERNELS_INSTALLED

    from triton_dist.autotuner import contextual_autotune
    from moe_bench.kernels.td import fp8_allgather_group_gemm as ag_module
    from moe_bench.kernels.td import fp8_moe_reduce_rs as rs_module

    if not _FP8_AUTOTUNE_KERNELS_INSTALLED:
        # FP8 的 K tile 必须与 1x128/128x128 量化块对齐；M/N tile 又参与
        # MoE 排序表和 RS chunk 完成协议，不能像 BF16 TP_MoE 那样直接扫描。
        # 因此只调不改变协议几何的 pipeline stages / warps。
        configs = [
            triton.Config({}, num_stages=stages, num_warps=warps)
            for stages in (3, 4)
            for warps in (4, 8)
        ]
        ag_module.fp8_kernel_consumer_ag_group_gemm = triton.autotune(
            configs=configs,
            key=["M", "N", "K"],
        )(ag_module.fp8_kernel_consumer_ag_group_gemm)
        rs_module.fp8_moe_gather_rs_grouped_gemm_kernel = triton.autotune(
            configs=configs,
            key=["M", "N", "K"],
        )(rs_module.fp8_moe_gather_rs_grouped_gemm_kernel)
        _FP8_AUTOTUNE_KERNELS_INSTALLED = True

    return (
        contextual_autotune(is_dist=True)(fp8_ag_group_gemm),
        contextual_autotune(is_dist=True)(run_fp8_moe_reduce_rs),
    )


# =============================================================================
# FP8 Quantization Utilities
# =============================================================================

@triton.jit
def _quantize_fp8_blockwise_kernel(
    input_ptr,
    input_row_stride,
    output_ptr,
    output_row_stride,
    scale_ptr,
    scale_row_stride,
    n_cols,
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused FP8 block-wise quantization kernel.
    Each program handles one row and one BLOCK_K chunk.
    Grid: (n_rows, n_cols // BLOCK_K)
    """
    row_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)

    col_offsets = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = col_offsets < n_cols

    # Load input block
    in_offset = row_idx * input_row_stride + col_offsets
    x = tl.load(input_ptr + in_offset, mask=mask, other=0.0).to(tl.float32)

    # Compute amax for this block
    amax = tl.max(tl.abs(x), axis=0)

    # Compute scale: amax / FP8_MAX (avoid division by zero)
    eps: tl.constexpr = 1e-12
    scale = tl.where(amax > eps, amax / FP8_MAX, 1.0)

    # Quantize
    x_q = tl.clamp(x / scale, FP8_MIN, FP8_MAX)

    # Store quantized output
    out_offset = row_idx * output_row_stride + col_offsets
    tl.store(output_ptr + out_offset, x_q.to(output_ptr.dtype.element_ty), mask=mask)

    # Store scale (one per block per row)
    scale_offset = row_idx * scale_row_stride + block_idx
    tl.store(scale_ptr + scale_offset, scale)


def quantize_fp8_blockwise(tensor: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                           block_k: int = 128):
    """Quantize a 2D tensor to FP8 with block-wise scaling along the last dimension.

    Uses a fused Triton kernel for efficiency.

    Args:
        tensor: [M, K] in BF16/FP16/FP32
        fp8_dtype: target FP8 dtype
        block_k: quantization block size

    Returns:
        q: [M, K] in fp8_dtype
        scale: [M, K // block_k] in float32
    """
    assert tensor.dim() == 2
    m, k = tensor.shape
    assert k % block_k == 0, f"K={k} must be divisible by block_k={block_k}"

    finfo = torch.finfo(fp8_dtype)
    fp8_max = finfo.max
    fp8_min = -finfo.max

    out_fp8 = torch.empty((m, k), dtype=fp8_dtype, device=tensor.device)
    scale = torch.empty((m, k // block_k), dtype=torch.float32, device=tensor.device)

    num_blocks_per_row = k // block_k
    grid = (m, num_blocks_per_row)

    # Choose num_warps based on BLOCK_K
    if block_k >= 512:
        num_warps = 8
    elif block_k >= 256:
        num_warps = 4
    else:
        num_warps = 4

    _quantize_fp8_blockwise_kernel[grid](
        tensor,
        tensor.stride(0),
        out_fp8,
        out_fp8.stride(0),
        scale,
        scale.stride(0),
        k,
        FP8_MAX=fp8_max,
        FP8_MIN=fp8_min,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )

    return out_fp8, scale


def quantize_fp8_weight_blockwise(tensor: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                                  block_k: int = 128, block_n: int = 128):
    """Quantize a 3D weight tensor [E, K, N] to FP8 with 2D block-wise scaling.

    For TP MoE, weights are stored as [E, K, N_per_rank]. We quantize in blocks of
    (block_k, block_n) in the (K, N) dimensions.

    Args:
        tensor: [E, K, N] in BF16/FP16/FP32
        fp8_dtype: target FP8 dtype
        block_k: block size along K dimension
        block_n: block size along N dimension

    Returns:
        q: [E, K, N] in fp8_dtype
        scale: [E, K // block_k, N // block_n] in float32
    """
    assert tensor.dim() == 3
    e, k, n = tensor.shape
    assert k % block_k == 0, f"K={k} must be divisible by block_k={block_k}"
    assert n % block_n == 0, f"N={n} must be divisible by block_n={block_n}"
    finfo = torch.finfo(fp8_dtype)

    tensor_fp32 = tensor.float().reshape(e, k // block_k, block_k, n // block_n, block_n)
    amax = tensor_fp32.abs().amax(dim=(2, 4))  # [E, K//block_k, N//block_n]
    scale = torch.where(amax > 0, amax / finfo.max, torch.ones_like(amax)).to(torch.float32)
    q = torch.clamp(
        tensor_fp32 / scale[:, :, None, :, None],
        min=-finfo.max, max=finfo.max
    ).to(fp8_dtype)
    return q.reshape(e, k, n).contiguous(), scale.contiguous()


# =============================================================================
# FP8_TP_MoE Layer
# =============================================================================

class FP8_TP_MoE:
    """
    FP8 Tensor Parallel MoE.

    Uses the same TP strategy as TP_MoE (column-parallel gate_up, row-parallel down),
    but with FP8 computation and communication:
    - AllGather in FP8 (halves communication bandwidth)
    - GEMM in FP8 with block-wise dequantization
    - Fused SwiGLU + FP8 quantization
    - ReduceScatter on BF16 GEMM output
    """

    def __init__(self, rank=0, world_size=8, group=None,
                 fp8_dtype=torch.float8_e4m3fn, block_k_quant=128, block_n_quant=128,
                 autotune=False):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.fp8_dtype = fp8_dtype
        self.block_k_quant = block_k_quant
        self.block_n_quant = block_n_quant
        self.autotune = autotune
        if autotune:
            self._ag_group_gemm, self._run_moe_reduce_rs = _make_fp8_autotuned_ops()
        else:
            self._ag_group_gemm = fp8_ag_group_gemm
            self._run_moe_reduce_rs = run_fp8_moe_reduce_rs

        # Weights (FP8 quantized)
        self.gate_up_proj_fp8 = None  # [E, K, N_gateup_per_tp] in FP8
        self.gate_up_proj_scale = None  # [E, K // block_k, N_gateup_per_tp // block_n] in float32
        self.down_proj_fp8 = None  # [E, intermediate_per_tp, hidden_size] in FP8
        self.down_proj_scale = None  # [E, intermediate_per_tp // block_k, hidden_size // block_n] in float32

        # Model params
        self.top_k = None
        self.num_experts = None
        self.hidden_size = None
        self.intermediate_per_tp = None

        # Contexts
        self.ag_ctx = None
        self.rs_ctx = None

    def _init_parameters_from_bf16(self, gate_up_proj_bf16: torch.Tensor, down_proj_bf16: torch.Tensor,
                                   num_experts: int, top_k: int, hidden_size: int, verbose=False):
        """Initialize FP8 weights from BF16 tensors (already TP-sharded and transposed).

        Args:
            gate_up_proj_bf16: [E, K, N_gateup_per_tp] in BF16 (transposed for GEMM)
            down_proj_bf16: [E, intermediate_per_tp, hidden_size] in BF16 (transposed for GEMM)
            num_experts: total number of experts
            top_k: number of experts per token
            hidden_size: model hidden dimension
        """
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        # gate_up: [E, K, N_gateup_per_tp]
        self.gate_up_proj_fp8, self.gate_up_proj_scale = quantize_fp8_weight_blockwise(
            gate_up_proj_bf16, self.fp8_dtype, self.block_k_quant, self.block_n_quant)

        # down: [E, intermediate_per_tp, hidden_size]
        self.intermediate_per_tp = down_proj_bf16.shape[1]
        self.down_proj_fp8, self.down_proj_scale = quantize_fp8_weight_blockwise(
            down_proj_bf16, self.fp8_dtype, self.block_k_quant, self.block_n_quant)

        if verbose:
            print(
                f"[RANK {self.rank}] FP8_TP_MoE initialized: "
                f"gate_up_proj_fp8={self.gate_up_proj_fp8.shape}, "
                f"down_proj_fp8={self.down_proj_fp8.shape}, "
                f"fp8_dtype={self.fp8_dtype}"
            )

    def _init_ctx(self, M: int, n_chunks_max: int = 8):
        """Initialize AG and RS contexts.

        Args:
            M: total number of tokens across all ranks (global M)
            n_chunks_max: synchronization-buffer capacity for RS N-chunks
        """
        N_gateup_per_tp = self.gate_up_proj_fp8.shape[2]

        self.ag_ctx = create_fp8_ag_group_gemm_context(
            max_ntokens=M,
            N_per_rank=N_gateup_per_tp,
            K=self.hidden_size,
            num_experts=self.num_experts,
            topk=self.top_k,
            fp8_dtype=self.fp8_dtype,
            output_dtype=torch.bfloat16,
            rank=self.rank,
            num_ranks=self.world_size,
            num_local_ranks=self.world_size,
            BLOCK_K_QUANT=self.block_k_quant,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=8,
            stages=4,
            num_warps=8,
        )

        self.rs_ctx = create_moe_rs_context(
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.world_size,
            max_token_num=M * self.top_k,
            hidden_dim=self.hidden_size,
            num_experts=self.num_experts,
            topk=self.top_k,
            input_dtype=torch.bfloat16,  # RS operates on BF16 GEMM output
            n_chunks_max=n_chunks_max,
        )

        torch.cuda.synchronize()
        if self.world_size > 1:
            torch.distributed.barrier(group=self.group)

    def autotune_report(self) -> str:
        """Return the rank-local choices after contextual autotune finishes."""
        from moe_bench.kernels.td import fp8_allgather_group_gemm as ag_module
        from moe_bench.kernels.td import fp8_moe_reduce_rs as rs_module

        def describe(name, tuner):
            config = getattr(tuner, "best_config", None)
            best_ms = getattr(tuner, "best_time", None)
            if config is None:
                return f"{name}=not-run"
            latency = f", slowest-rank={best_ms:.4f}ms" if best_ms is not None else ""
            return (f"{name}=(stages={config.num_stages}, warps={config.num_warps}"
                    f"{latency})")

        return " ".join((
            describe("AG", ag_module.fp8_kernel_consumer_ag_group_gemm),
            describe("RS", rs_module.fp8_moe_gather_rs_grouped_gemm_kernel),
        ))

    def finalize(self):
        if self.rs_ctx:
            self.rs_ctx.finalize()
        if self.ag_ctx:
            self.ag_ctx.finalize()

    @torch.inference_mode()
    def dist_triton_fwd(self, x: torch.Tensor, full_topk_ids: torch.Tensor,
                        full_topk_weight: torch.Tensor, n_chunks_rs: int = 4):
        """
        FP8 TP MoE forward pass with tile-level AG-GEMM overlap and GEMM-RS overlap.

        Args:
            x: local hidden states [M_local, hidden_size] in BF16
            full_topk_ids: all-gathered routing IDs [M, top_k] in int32
            full_topk_weight: all-gathered routing weights [M, top_k] in float32
            n_chunks_rs: number of chunks for RS overlap

        Returns:
            output: [M_local, hidden_size] in BF16
        """
        M_local = x.shape[0]
        ntokens = M_local * self.world_size
        M = ntokens * self.top_k

        # Step 1: Quantize local input to FP8
        x_fp8, x_scale = quantize_fp8_blockwise(x, self.fp8_dtype, self.block_k_quant)

        # Step 2: FP8 AG + GroupGEMM (gate_up) with tile-level overlap
        out_fused = self._ag_group_gemm(
            x_fp8, x_scale,
            self.gate_up_proj_fp8, self.gate_up_proj_scale,
            self.ag_ctx,
            full_topk_ids,
        )

        # Step 3: Fused SwiGLU + FP8 quantization with routing weight
        routing_weight_expanded = full_topk_weight.flatten()  # [M]

        swiglu_fp8, swiglu_scale = swiglu_quantize_fp8(
            out_fused,
            routing_weight=routing_weight_expanded,
            fp8_dtype=self.fp8_dtype,
            block_k=self.block_k_quant,
        )

        # Step 4: FP8 Grouped GEMM (down) + ReduceScatter
        output = self._run_moe_reduce_rs(
            swiglu_fp8,
            swiglu_scale,
            self.down_proj_fp8,
            self.down_proj_scale,
            full_topk_ids,
            full_topk_weight,
            self.rs_ctx,
            n_chunks=n_chunks_rs,
            BLOCK_N_QUANT=self.block_n_quant,
        )

        return output.view(M_local, self.hidden_size)

    @torch.inference_mode()
    def torch_fwd(self, x: torch.Tensor, full_topk_ids: torch.Tensor,
                  full_topk_weight: torch.Tensor):
        """
        Reference PyTorch forward pass for correctness verification.
        Uses FP8 quantized weights but standard PyTorch ops (no triton_dist).

        Args:
            x: local hidden states [M_local, hidden_size] in BF16
            full_topk_ids: [M, top_k] int32
            full_topk_weight: [M, top_k] float32

        Returns:
            output: [M_local, hidden_size] BF16
        """
        M_local, hidden_size = x.shape
        M = M_local * self.world_size

        # AllGather hidden states (in BF16 for reference)
        hidden_full = torch.empty(M, hidden_size, dtype=x.dtype, device=x.device)
        torch.distributed.all_gather_into_tensor(hidden_full, x.contiguous(), group=self.group)

        # Dequantize weights for reference computation
        gate_up_bf16 = _dequant_weight(self.gate_up_proj_fp8, self.gate_up_proj_scale,
                                       self.block_k_quant, self.block_n_quant)
        down_bf16 = _dequant_weight(self.down_proj_fp8, self.down_proj_scale,
                                    self.block_k_quant, self.block_n_quant)

        # Expand tokens by topk
        ntokens = M
        topk = self.top_k
        expert_mask = torch.nn.functional.one_hot(
            full_topk_ids.long(), num_classes=self.num_experts).permute(2, 1, 0)

        out = torch.zeros((ntokens, hidden_size), dtype=x.dtype, device=x.device)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hitted:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_full[top_x]
            # gate_up GEMM: [tokens, K] @ [K, N_gateup]
            cur_out_fused = current_state @ gate_up_bf16[expert_idx].squeeze(0)
            wg, w1 = torch.chunk(cur_out_fused, 2, dim=-1)
            cur_out = torch.nn.functional.silu(wg) * w1
            # down GEMM: [tokens, intermediate] @ [intermediate, hidden]
            cur_out = cur_out @ down_bf16[expert_idx].squeeze(0)
            cur_out = cur_out * full_topk_weight[top_x, idx, None]
            out.index_add_(0, top_x, cur_out.to(x.dtype))

        # ReduceScatter
        out_local = torch.empty(M_local, hidden_size, dtype=x.dtype, device=x.device)
        torch.distributed.reduce_scatter_tensor(out_local, out, group=self.group)
        return out_local


def _dequant_weight(w_fp8: torch.Tensor, w_scale: torch.Tensor,
                    block_k: int, block_n: int) -> torch.Tensor:
    """Dequantize FP8 weight [E, K, N] back to BF16 using block-wise scale."""
    e, k, n = w_fp8.shape
    w_fp32 = w_fp8.float().reshape(e, k // block_k, block_k, n // block_n, block_n)
    # scale: [E, K // block_k, N // block_n]
    result = (w_fp32 * w_scale[:, :, None, :, None]).reshape(e, k, n)
    return result.to(torch.bfloat16)


# =============================================================================
# FP8_TP_MoE_FP8RS: FP8 TP MoE with FP8 ReduceScatter
# =============================================================================

class FP8_TP_MoE_FP8RS(FP8_TP_MoE):
    """
    FP8 Tensor Parallel MoE with FP8 ReduceScatter.

    Inherits from FP8_TP_MoE but replaces BF16 reduce-scatter with FP8 communication:
    - After down GEMM produces BF16 output, topk-reduce is done in FP32
    - Result is quantized to FP8 (per-block=128) for NVSHMEM ring transfer
    - Receiver dequantizes and accumulates in FP32
    - Final output is BF16

    This halves the NVSHMEM transfer bandwidth compared to BF16 RS.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp8_rs_ctx = None

    def _init_ctx(self, M: int):
        """Initialize AG context (same as C4) + FP8 RS context (new)."""
        N_gateup_per_tp = self.gate_up_proj_fp8.shape[2]

        self.ag_ctx = create_fp8_ag_group_gemm_context(
            max_ntokens=M,
            N_per_rank=N_gateup_per_tp,
            K=self.hidden_size,
            num_experts=self.num_experts,
            topk=self.top_k,
            fp8_dtype=self.fp8_dtype,
            output_dtype=torch.bfloat16,
            rank=self.rank,
            num_ranks=self.world_size,
            num_local_ranks=self.world_size,
            BLOCK_K_QUANT=self.block_k_quant,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=8,
            stages=4,
            num_warps=8,
        )

        # BF16 RS context (for GEMM signaling infrastructure)
        self.rs_ctx = create_moe_rs_context(
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.world_size,
            max_token_num=M * self.top_k,
            hidden_dim=self.hidden_size,
            num_experts=self.num_experts,
            topk=self.top_k,
            input_dtype=torch.bfloat16,
        )

        # FP8 RS context (new: FP8 symm buffers for communication)
        self.fp8_rs_ctx = create_fp8_rs_context(
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.world_size,
            max_token_num=M * self.top_k,
            hidden_dim=self.hidden_size,
            num_experts=self.num_experts,
            topk=self.top_k,
            block_quant=self.block_k_quant,  # 128
            n_chunks_max=64,
        )

        torch.cuda.synchronize()
        if self.world_size > 1:
            torch.distributed.barrier(group=self.group)

    def finalize(self):
        if self.fp8_rs_ctx:
            self.fp8_rs_ctx.finalize()
        if self.rs_ctx:
            self.rs_ctx.finalize()
        if self.ag_ctx:
            self.ag_ctx.finalize()

    @torch.inference_mode()
    def dist_triton_fwd(self, x: torch.Tensor, full_topk_ids: torch.Tensor,
                        full_topk_weight: torch.Tensor, n_chunks_rs: int = 32):
        """
        FP8 TP MoE forward with FP8 ReduceScatter.

        Same as parent but step 4 uses FP8 communication.
        """
        M_local = x.shape[0]
        ntokens = M_local * self.world_size
        M = ntokens * self.top_k

        # Step 1: Quantize local input to FP8
        x_fp8, x_scale = quantize_fp8_blockwise(x, self.fp8_dtype, self.block_k_quant)

        # Step 2: FP8 AG + GroupGEMM (gate_up) with tile-level overlap
        out_fused = self._ag_group_gemm(
            x_fp8, x_scale,
            self.gate_up_proj_fp8, self.gate_up_proj_scale,
            self.ag_ctx,
            full_topk_ids,
        )

        # Step 3: Fused SwiGLU + FP8 quantization with routing weight
        routing_weight_expanded = full_topk_weight.flatten()

        swiglu_fp8, swiglu_scale = swiglu_quantize_fp8(
            out_fused,
            routing_weight=routing_weight_expanded,
            fp8_dtype=self.fp8_dtype,
            block_k=self.block_k_quant,
        )

        # Step 4: FP8 Grouped GEMM (down) + FP8 ReduceScatter
        output = run_fp8_moe_reduce_rs_fp8comm(
            swiglu_fp8,
            swiglu_scale,
            self.down_proj_fp8,
            self.down_proj_scale,
            full_topk_ids,
            full_topk_weight,
            self.rs_ctx,
            self.fp8_rs_ctx,
            n_chunks=n_chunks_rs,
            BLOCK_N_QUANT=self.block_n_quant,
        )

        return output.view(M_local, self.hidden_size)
