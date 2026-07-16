# SPDX-License-Identifier: Apache-2.0
"""MB1 计算 kernel 单测:vLLM(triton fused_experts)vs TK(grouped GEMM 链)。

同一路由、同一 gathered batch 下,只测本地专家计算,不含任何跨卡通信:

  vllm_compute : fused_experts(hidden_full, w1, w2, topk_full, expert_map)
                 —— serial baseline run() 里的计算段原样(含其内部
                 moe_align_block_size 路由元数据 + scatter/gather + 加权求和)
  tk_l0        : TK grouped_gemm gate+up(N=2*inter=6144)
  tk_silu      : silu(gate)*up(torch elementwise)
  tk_l1        : TK grouped_gemm W2
  tk_chain     : l0 + silu + l1 连续执行 = TK 计算全链
  tk_sched     : TK GPU schedule 重建(路由 all_gather + builder,eager)
                 —— TK 侧的"路由元数据"成本,公平对账时应计入 TK
  cublas_l0/l1 : 同形状稠密 GEMM(cuBLAS)= 机器 GEMM 效率上限参考

口径说明:
- vllm 计算的是真实 assignment 数;TK 按 128-padding 的行数算(padding 是 TK
  方案的固有成本)。两者的 tflops_eff 用同一 flop 基数(真实 assignment),
  直接可比;tk_chain_tflops_padded 是 TK 含 padding 的名义吞吐。
- vllm_compute 里含 topk 加权求和(combine 的一部分),TK 的加权在 combine 通信
  kernel 里;层间边界略有不同,分析时以 mb3 的整层分解为准。

  python -m moe_bench.microbench.mb1_compute
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import common


def _worker(rank, world, init_method, out_list):
    tk_env = common.apply_tk_env()
    device, ctx = common.init_dist_worker(rank, world, init_method)
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from moe_bench.tk_scheme import _build_schedules_gpu

    warmup, iters = common.warmup_iters(), common.bench_iters()
    rows = []
    try:
        for total in common.global_token_sweep():
            T = total // world
            cfg = common.make_cfg(T)
            sch, problem = common.build_tk_fixture(cfg, T, rank, ctx)
            tk = sch.tk
            H, inter = sch.H, sch.inter
            e_local = cfg.num_local_experts
            eoff = rank * e_local
            npl = sch.num_padded_local
            m_full = T * world
            dt = cfg.torch_dtype  # 激活基精度(bf16/fp16;fp8 的基精度也是 bf16)

            # ---- vllm 侧输入:gathered full batch + full 路由(等价 serial 的
            #      AllGather 结果,这里在 setup 里做,不计时) ----
            hidden_full = torch.empty(m_full, H, device=device, dtype=dt)
            ids_full = torch.empty(m_full, cfg.topk, device=device,
                                   dtype=problem.topk_ids.dtype)
            w_full = torch.empty(m_full, cfg.topk, device=device,
                                 dtype=problem.topk_weights.dtype)
            dist.all_gather_into_tensor(hidden_full, problem.hidden_states.contiguous())
            dist.all_gather_into_tensor(ids_full, problem.topk_ids.contiguous())
            dist.all_gather_into_tensor(w_full, problem.topk_weights.contiguous())

            # ---- TK 侧输入:按 disp_idx 从 hidden_full 填 gathered
            #      (逐字节等价 dispatch 的结果,padding 行清零) ----
            sd = sch.disp_idx[:, 0].long()
            st = sch.disp_idx[:, 1].long()
            valid = sd >= 0
            rows_idx = torch.where(valid, sd * T + st, torch.zeros_like(sd))
            g = hidden_full[rows_idx]
            g[~valid] = 0
            sch.gathered.data_[:npl].copy_(g)
            torch.nn.init.normal_(sch.act, std=0.02)

            # ---- 稠密 GEMM 上限参考(同形状 cuBLAS) ----
            x0 = torch.randn(npl, H, device=device, dtype=dt)
            wd0 = torch.randn(H, 2 * inter, device=device, dtype=dt)
            od0 = torch.empty(npl, 2 * inter, device=device, dtype=dt)
            x1 = torch.randn(npl, inter, device=device, dtype=dt)
            wd1 = torch.randn(inter, H, device=device, dtype=dt)
            od1 = torch.empty(npl, H, device=device, dtype=dt)

            def vllm_compute():
                fused_experts(hidden_states=hidden_full, w1=problem.w1, w2=problem.w2,
                              topk_weights=w_full, topk_ids=ids_full,
                              global_num_experts=cfg.num_experts,
                              expert_map=problem.expert_map,
                              quant_config=problem.quant_config)

            def tk_l0():
                tk.grouped_gemm(sch.gathered.data_, sch.w_gateup, sch.gateup_out,
                                sch.padded, eoff)

            def tk_silu():
                torch.mul(F.silu(sch.gateup_out[:, :inter]),
                          sch.gateup_out[:, inter:], out=sch.act)

            def tk_l1():
                tk.grouped_gemm(sch.act, sch.w2, sch.expert_out.data_,
                                sch.padded, eoff)

            def tk_chain():
                tk_l0()
                tk_silu()
                tk_l1()

            def tk_sched():
                dist.all_gather_into_tensor(sch._all_topk, sch._topk_ids_local)
                dist.all_gather_into_tensor(sch._all_w, sch._topk_w_local)
                _build_schedules_gpu(sch._all_topk, sch._all_w, world,
                                     sch._num_experts, sch._e_local, rank,
                                     sch._sched_out)

            def cublas_l0():
                torch.matmul(x0, wd0, out=od0)

            def cublas_l1():
                torch.matmul(x1, wd1, out=od1)

            t = {}
            for name, fn in [("vllm_compute", vllm_compute), ("tk_l0", tk_l0),
                             ("tk_silu", tk_silu), ("tk_l1", tk_l1),
                             ("tk_chain", tk_chain), ("tk_sched", tk_sched),
                             ("cublas_l0", cublas_l0), ("cublas_l1", cublas_l1)]:
                t[name] = common.lockstep_median_ms(fn, warmup, iters)
            t = common.maxrank_timings(t, device)

            assign = int((ids_full.long() // e_local == rank).sum().item())
            agg = torch.tensor([float(assign), float(npl)], device=device)
            dist.all_reduce(agg, op=dist.ReduceOp.MAX)
            assign_max, npl_max = int(agg[0].item()), int(agg[1].item())

            if rank == 0:
                flops_eff = 6.0 * inter * H * assign_max
                flops_pad = 6.0 * inter * H * npl_max
                row = {
                    "tokens_total": total, "tokens_per_rank": T,
                    "local_assignments": assign_max,
                    "num_padded_local": npl_max,
                    "padding_ratio": npl_max / max(assign_max, 1),
                    **{f"{k}_ms": v for k, v in t.items()},
                    "vllm_tflops_eff":
                        flops_eff / (t["vllm_compute"] * 1e-3) / 1e12,
                    "tk_chain_tflops_eff":
                        flops_eff / (t["tk_chain"] * 1e-3) / 1e12,
                    "tk_chain_tflops_padded":
                        flops_pad / (t["tk_chain"] * 1e-3) / 1e12,
                    "tk_time_reduction_vs_vllm_pct":
                        (t["vllm_compute"] - t["tk_chain"])
                        / t["vllm_compute"] * 100.0,
                }
                rows.append(row)
                print(f"[mb1] tokens={total:5d}  "
                      f"vllm={t['vllm_compute']*1e3:8.1f}us "
                      f"({row['vllm_tflops_eff']:6.1f} TF)  "
                      f"tk_chain={t['tk_chain']*1e3:8.1f}us "
                      f"({row['tk_chain_tflops_eff']:6.1f} TF)  "
                      f"tk_sched={t['tk_sched']*1e3:7.1f}us  "
                      f"time_reduction="
                      f"{row['tk_time_reduction_vs_vllm_pct']:+.2f}%", flush=True)

            del sch, problem, hidden_full, ids_full, w_full, g
            del x0, wd0, od0, x1, wd1, od1
            torch.cuda.empty_cache()

        if rank == 0:
            out_list.append({"rows": rows, "tk_env": tk_env})
    finally:
        torch.cuda.synchronize()
        dist.destroy_process_group()


def main():
    out = common.spawn_workers(_worker)
    assert out, "no results collected"
    meta = common.result_meta({"test": "mb1_compute", "tk_env": out[0]["tk_env"]})
    common.write_json("mb1_compute", meta, out[0]["rows"])


if __name__ == "__main__":
    main()
