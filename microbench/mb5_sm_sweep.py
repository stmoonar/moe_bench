# SPDX-License-Identifier: Apache-2.0
"""MB5 inter-SM 方案下 comm SM 数量扫参(真实 kernel)。

当前 TK 融合 kernel 是 inter-SM 编排:grid 里前 num_comp_sms 个 block 跑 GEMM,
其余 block 做通信;num_comm_sms 决定让渡给通信的 SM 数。本测试对每个 token 档、
每个 k=num_comm_sms:

  L0_fused_k : prep + moe_dispatch_gemm(..., k)         —— dispatch⊕gate+up @ k
  L1_fused_k : prered_push GEMM⊕combine(..., k)         —— layer1 @ k
  e2e_k      : TKFusedEP.run()(sch.num_comm_sms = k)
  gemm0_nb   : grouped_gemm_nb(gate+up, blocks = SM总数 - k)—— 纯"SM 让渡"代价,
  gemm1_nb   : grouped_gemm_nb(W2,      blocks = SM总数 - k)   无通信无干扰
  另给 gemm0_full / gemm1_full(全 SM 纯 GEMM)作 k=0 参考。

结论口径:
  让渡损失   = gemm_nb(SM-k) - gemm_full
  干扰+等待  = L_fused(k) - gemm_nb(SM-k) - (prep)
通信侧带宽随 SM 数的变化由 mb7(sm_copy 块数扫参)给出,两者对照即可回答
"多少 comm SM 是合理的"。

  MB_COMM_SMS=1,2,4,8,12,16,24,32(默认)可覆盖。
  python -m moe_bench.microbench.mb5_sm_sweep
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import common

DEFAULT_K_SWEEP = [1, 2, 4, 8, 12, 16, 24, 32]


def _k_sweep() -> list[int]:
    env = os.environ.get("MB_COMM_SMS")
    return [int(x) for x in env.split(",")] if env else list(DEFAULT_K_SWEEP)


def _worker(rank, world, init_method, out_list):
    tk_env = common.apply_tk_env()
    device, ctx = common.init_dist_worker(rank, world, init_method)

    warmup, iters = common.warmup_iters(), common.bench_iters()
    sm_total = torch.cuda.get_device_properties(device).multi_processor_count
    rows = []
    try:
        for total in common.global_token_sweep():
            T = total // world
            cfg = common.make_cfg(T)
            sch, problem = common.build_tk_fixture(cfg, T, rank, ctx)
            tk = sch.tk
            inter = sch.inter
            eoff = rank * cfg.num_local_experts
            gw, go = sch.w_gateup, sch.gateup_out
            torch.nn.init.normal_(sch.act, std=0.02)

            def _prep():
                sch._l0_seq += 1
                tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
                sch.pre_tokens.data_.copy_(problem.hidden_states)
                sch._l0_seq += 1
                tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)

            def gemm0_full():
                tk.grouped_gemm(sch.gathered.data_, gw, go, sch.padded, eoff)

            def gemm1_full():
                tk.grouped_gemm(sch.act, sch.w2, sch.expert_out.data_,
                                sch.padded, eoff)

            # 填充 gathered(一次 fused dispatch,不计时)
            _prep()
            tk.moe_dispatch_gemm(sch.pre_tokens, sch.gathered.data_, gw, go,
                                 sch.padded, sch.disp_idx, sch.barrier_l0,
                                 sch.num_comm_sms, sch.num_padded_local)
            torch.cuda.synchronize()

            base = {}
            base["prep"] = common.lockstep_median_ms(_prep, warmup, iters)
            base["gemm0_full"] = common.lockstep_median_ms(gemm0_full, warmup, iters)
            base["gemm1_full"] = common.lockstep_median_ms(gemm1_full, warmup, iters)
            base = common.maxrank_timings(base, device)

            for k in _k_sweep():
                if k >= sm_total:
                    continue
                nb = sm_total - k

                def L0_fused_k():
                    _prep()
                    tk.moe_dispatch_gemm(sch.pre_tokens, sch.gathered.data_, gw,
                                         go, sch.padded, sch.disp_idx,
                                         sch.barrier_l0, k,
                                         sch.num_padded_local)

                def L1_fused_k():
                    sch._l1_seq += 1
                    sch.combine_local_cnt.zero_()
                    tk.moe_gemm_prered_push_fused(
                        sch.act, sch.w2, sch.expert_out, sch.padded,
                        sch.combine_staging, sch.prered_dst, sch.prered_slots,
                        sch.prered_w, sch.combine_local_cnt,
                        sch.push_expected_l1, sch.barrier_l1, k,
                        sch.num_padded_local, sch.num_tokens, sch.num_jobs,
                        sch._l1_seq)
                    tk.moe_final_reduce_push(
                        sch.combine_staging, sch.final_contrib, sch.recv_from,
                        sch.combine_out, sch.barrier_l1, sch.num_tokens,
                        sch._l1_seq)

                def gemm0_nb():
                    tk.grouped_gemm_nb(sch.gathered.data_, gw, go, sch.padded,
                                       eoff, nb)

                def gemm1_nb():
                    tk.grouped_gemm_nb(sch.act, sch.w2, sch.expert_out.data_,
                                       sch.padded, eoff, nb)

                def e2e_k():
                    sch.run()

                t = {}
                t["L0_fused"] = common.lockstep_median_ms(L0_fused_k, warmup, iters)
                t["L1_fused"] = common.lockstep_median_ms(L1_fused_k, warmup, iters)
                t["gemm0_nb"] = common.lockstep_median_ms(gemm0_nb, warmup, iters)
                t["gemm1_nb"] = common.lockstep_median_ms(gemm1_nb, warmup, iters)
                # 重新 silu 保持 act 内容合理(不影响速度,保持确定性)
                torch.mul(F.silu(go[:, :inter]), go[:, inter:], out=sch.act)
                saved = sch.num_comm_sms
                sch.num_comm_sms = k
                t["e2e"] = common.lockstep_median_ms(e2e_k, warmup, iters)
                sch.num_comm_sms = saved
                t = common.maxrank_timings(t, device)

                if rank == 0:
                    row = {
                        "tokens_total": total, "tokens_per_rank": T,
                        "num_comm_sms": k, "sm_total": sm_total,
                        "prep_ms": base["prep"],
                        "gemm0_full_ms": base["gemm0_full"],
                        "gemm1_full_ms": base["gemm1_full"],
                        **{f"{n}_ms": v for n, v in t.items()},
                        "l0_yield_cost_ms": t["gemm0_nb"] - base["gemm0_full"],
                        "l1_yield_cost_ms": t["gemm1_nb"] - base["gemm1_full"],
                        "l0_interference_ms":
                            t["L0_fused"] - base["prep"] - t["gemm0_nb"],
                        "l1_interference_ms": t["L1_fused"] - t["gemm1_nb"],
                    }
                    rows.append(row)
                    print(f"[mb5] tokens={total:5d} k={k:2d}  "
                          f"L0={t['L0_fused']*1e3:8.1f}us "
                          f"L1={t['L1_fused']*1e3:8.1f}us "
                          f"e2e={t['e2e']*1e3:8.1f}us | "
                          f"gemm0@{sm_total-k}={t['gemm0_nb']*1e3:7.1f}us "
                          f"gemm1@{sm_total-k}={t['gemm1_nb']*1e3:7.1f}us",
                          flush=True)

            del sch, problem
            torch.cuda.empty_cache()

        if rank == 0:
            out_list.append({"rows": rows, "tk_env": tk_env})
    finally:
        torch.cuda.synchronize()
        dist.destroy_process_group()


def main():
    out = common.spawn_workers(_worker)
    assert out, "no results collected"
    meta = common.result_meta({"test": "mb5_sm_sweep", "k_sweep": _k_sweep(),
                               "tk_env": out[0]["tk_env"]})
    common.write_json("mb5_sm_sweep", meta, out[0]["rows"])


if __name__ == "__main__":
    main()
