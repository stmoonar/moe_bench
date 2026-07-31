"""tdtp: Triton-distributed 原语实现的 FP8 TP MoE（FP8 AG + FP8 GroupGEMM + BF16 RS）。

移植自 td_benchmark 的 c4（`tdx/layers/fp8_tp_moe.py`，kernel 在
`moe_bench/kernels/td/` 下）。数据流：

```text
quantize_fp8_blockwise(x)                      # token 1x128 group, 每次 run 计入
  -> fp8_ag_group_gemm                         # NVSHMEM FP8 AG + tile 级 overlap 的 group GEMM
  -> swiglu_quantize_fp8(routing_weight)       # SwiGLU + 路由权重 + FP8 量化融合
  -> run_fp8_moe_reduce_rs(n_chunks)           # FP8 down GEMM + BF16 RS 分块 overlap
```

与 tktp 的差别：GEMM 引擎是 triton tl.dot（非手工 TK/mma.sync），通算融合靠
多 stream + tile 级 wait（非单 kernel persistent dispatcher）。量化方案与主
配置口径一致（token 1x128 / weight 128x128 blockwise fp8 e4m3），权重直接用
problem 的 fp8 数据（与 tktp/serial 同一份，公平）。

依赖 triton_dist（服务器环境）与 NVSHMEM；worker 的初始化见 distributed.py
（`requires_nvshmem = True` 时走 triton_dist.utils.initialize_distributed）。设置
``TD_AUTOTUNE=1`` 可启用 Triton-distributed TP_MoE 同款 contextual autotune；
首次运行会扫描 FP8 GEMM 的 stages/warps 并在各 rank 的最慢耗时上选优。
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from .config import Precision
from .context import DistContext
from .data import MoEProblem
from .schemes import DistributedScheme


class TDTFusedTP(DistributedScheme):
    """Triton-distributed FP8 TP MoE (c4 dataflow)。"""

    name = "tdtp"
    requires_nvshmem = True  # worker 据此走 NVSHMEM 初始化(不重复 init PG)

    def setup(self, problem: MoEProblem, ctx: DistContext) -> None:
        from .kernels.td.fp8_tp_moe import FP8_TP_MoE
        import triton

        if problem.config.precision != Precision.FP8 or problem.quant_config is None:
            raise ValueError("tdtp 是 FP8-only scheme, 需要 fp8 精度配置")

        # td 的 triton kernel 内部的 device 分配钩子
        def alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device=ctx.device, dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        pcfg = problem.config
        world = ctx.world_size
        T = problem.num_tokens

        self.ctx = ctx
        self.hidden_local = problem.hidden_states.contiguous()
        self.ids_local = problem.topk_ids.contiguous()
        self.w_local = problem.topk_weights.contiguous()
        self.full_ids = torch.empty(world * T, pcfg.topk, device=ctx.device,
                                    dtype=torch.int32)
        self.full_w = torch.empty(world * T, pcfg.topk, device=ctx.device,
                                  dtype=torch.float32)
        # RS 分块数(GEMM-RS overlap 粒度), 与 td_benchmark 的 c4 实际值一致(32)
        self.n_chunks_rs = int(os.environ.get("TD_N_CHUNKS_RS", "32"))
        if (self.n_chunks_rs <= 0
                or self.n_chunks_rs & (self.n_chunks_rs - 1)
                or pcfg.hidden_size % self.n_chunks_rs):
            raise ValueError("TD_N_CHUNKS_RS 必须是能整除 hidden_size 的正 2 次幂")
        self.autotune = os.environ.get("TD_AUTOTUNE", "0") == "1"
        self._autotune_reported = False
        if self.autotune and ctx.is_rank0:
            print("[tdtp] TD_AUTOTUNE=1: tuning FP8 AG/RS GEMM stages × warps", flush=True)

        layer = FP8_TP_MoE(rank=ctx.rank, world_size=world, group=ctx.group,
                           block_k_quant=128, block_n_quant=128,
                           autotune=self.autotune)
        # 直接接入 problem 的 fp8 权重(与 tktp/serial 同一份量化数据):
        # problem.w1 (E, 2I, K) fp8 -> gate_up_proj [E, K, 2I]
        # problem.w2 (E, K, I)  fp8 -> down_proj    [E, I, K]
        # scale 同为 (…/128) 网格, 转置仅换索引序, block 对齐不变。
        layer.num_experts = pcfg.num_experts
        layer.top_k = pcfg.topk
        layer.hidden_size = pcfg.hidden_size
        layer.intermediate_per_tp = problem.w2.shape[2]
        layer.gate_up_proj_fp8 = problem.w1.transpose(1, 2).contiguous()
        layer.gate_up_proj_scale = (
            problem.quant_config.w1_scale.transpose(1, 2).contiguous())
        layer.down_proj_fp8 = problem.w2.transpose(1, 2).contiguous()
        layer.down_proj_scale = (
            problem.quant_config.w2_scale.transpose(1, 2).contiguous())
        layer._init_ctx(M=world * T, n_chunks_max=max(8, self.n_chunks_rs))
        self.layer = layer

    def run(self) -> torch.Tensor:
        # 与 tktp 同口径: 只 all_gather 路由(hidden 的 AG 在 fp8 kernel 内部走
        # NVSHMEM); serial 按它自己的数据流多 all_gather 一次 bf16 hidden。
        dist.all_gather_into_tensor(self.full_ids, self.ids_local,
                                    group=self.ctx.group)
        dist.all_gather_into_tensor(self.full_w, self.w_local, group=self.ctx.group)
        output = self.layer.dist_triton_fwd(
            self.hidden_local, self.full_ids, self.full_w, self.n_chunks_rs)
        if self.autotune and not self._autotune_reported:
            if self.ctx.is_rank0:
                print(f"[tdtp] autotune best: {self.layer.autotune_report()}", flush=True)
            self._autotune_reported = True
        return output

    def close(self) -> None:
        layer = getattr(self, "layer", None)
        if layer is not None:
            layer.finalize()
            self.layer = None
