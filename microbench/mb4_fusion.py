# SPDX-License-Identifier: Apache-2.0
"""MB4 融合收益/损失:e2e tkfused vs serial + 分层「纯 GEMM vs 融合」。

每个 token 档测(默认路径 pull dispatch⊕gate+up + prered_push combine):

  serial_e2e : SerialNaive.run(AG×3 + fused_experts + RS),vLLM 串行基线
  tk_e2e     : TKFusedEP.run(按当前默认路径,原样)
  prep       : layer0 的 barrier+copy+barrier 前导(计入 L0_fused,单独给出便于扣除)
  L0_gemm    : 纯 gate+up grouped GEMM(compute-only 参考)
  L0_fused   : prep + moe_dispatch_gemm(跨卡 pull ⊕ gate+up GEMM)
  silu       : silu(gate)*up
  L1_gemm    : 纯 W2 grouped GEMM
  L1_fused   : moe_gemm_prered_push_fused + moe_final_reduce_push(T6-v1 默认 combine)
  sched      : TK GPU schedule 重建(eager;当前 run() 在 prered_push 默认路径下
               并未把它计入计时 —— 公平口径应把它加到 tk_e2e 上,见 mb3)

融合损失 = fused - gemm(与 tools/analyze_overlap、docs/17 同口径,并推广到
token sweep)。同时给出 tkfused 与 serial 输出的相对误差(同一 problem,双方都应
逼近参考,交叉误差应在 1e-2 量级)。

  python -m moe_bench.microbench.mb4_fusion
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import common


def _worker(rank, world, init_method, out_list):
    tk_env = common.apply_tk_env()  # 默认路径
    device, ctx = common.init_dist_worker(rank, world, init_method)
    from moe_bench.schemes import SerialNaive
    from moe_bench.tk_scheme import _build_schedules_gpu

    warmup, iters = common.warmup_iters(), common.bench_iters()
    rows = []
    try:
        for total in common.global_token_sweep():
            T = total // world
            cfg = common.make_cfg(T)
            sch, problem = common.build_tk_fixture(cfg, T, rank, ctx)
            serial = SerialNaive()
            serial.setup(problem, ctx)
            tk = sch.tk
            H, inter = sch.H, sch.inter
            eoff = rank * cfg.num_local_experts
            npl = sch.num_padded_local
            assert sch.fuse_gateup and sch.combine_mode == "prered_push", \
                "mb4 假定默认路径(fused gate+up + prered_push)"

            def _prep():
                sch._l0_seq += 1
                tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)
                sch.pre_tokens.data_.copy_(problem.hidden_states)
                sch._l0_seq += 1
                tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)

            gw, go = sch.w_gateup, sch.gateup_out

            def L0_gemm():
                tk.grouped_gemm(sch.gathered.data_, gw, go, sch.padded, eoff)

            def L0_fused():
                _prep()
                tk.moe_dispatch_gemm(sch.pre_tokens, sch.gathered.data_, gw, go,
                                     sch.padded, sch.disp_idx, sch.barrier_l0,
                                     sch.num_comm_sms, sch.num_padded_local)

            def silu():
                torch.mul(F.silu(go[:, :inter]), go[:, inter:], out=sch.act)

            def L1_gemm():
                tk.grouped_gemm(sch.act, sch.w2, sch.expert_out.data_,
                                sch.padded, eoff)

            def L1_fused():
                sch._l1_seq += 1
                sch.combine_local_cnt.zero_()
                tk.moe_gemm_prered_push_fused(
                    sch.act, sch.w2, sch.expert_out, sch.padded,
                    sch.combine_staging, sch.prered_dst, sch.prered_slots,
                    sch.prered_w, sch.combine_local_cnt, sch.push_expected_l1,
                    sch.barrier_l1, sch.num_comm_sms, sch.num_padded_local,
                    sch.num_tokens, sch.num_jobs, sch._l1_seq)
                tk.moe_final_reduce_push(
                    sch.combine_staging, sch.final_contrib, sch.recv_from,
                    sch.combine_out, sch.barrier_l1, sch.num_tokens, sch._l1_seq)

            def sched():
                dist.all_gather_into_tensor(sch._all_topk, sch._topk_ids_local)
                dist.all_gather_into_tensor(sch._all_w, sch._topk_w_local)
                _build_schedules_gpu(sch._all_topk, sch._all_w, world,
                                     sch._num_experts, sch._e_local, rank,
                                     sch._sched_out)

            t = {}
            # e2e 先测(run() 自带前导,状态自洽)
            t["serial_e2e"] = common.lockstep_median_ms(serial.run, warmup, iters)
            t["tk_e2e"] = common.lockstep_median_ms(sch.run, warmup, iters)
            # 分层:L0(fused 会填充 gathered,先 fused 后 gemm 顺序无碍——
            # GEMM 速度与内容无关)
            t["prep"] = common.lockstep_median_ms(_prep, warmup, iters)
            t["L0_fused"] = common.lockstep_median_ms(L0_fused, warmup, iters)
            t["L0_gemm"] = common.lockstep_median_ms(L0_gemm, warmup, iters)
            t["silu"] = common.lockstep_median_ms(silu, warmup, iters)
            torch.nn.init.normal_(sch.act, std=0.02)
            t["L1_gemm"] = common.lockstep_median_ms(L1_gemm, warmup, iters)
            t["L1_fused"] = common.lockstep_median_ms(L1_fused, warmup, iters)
            t["sched"] = common.lockstep_median_ms(sched, warmup, iters)
            t = common.maxrank_timings(t, device)

            # 交叉校验:同一 problem 下 tkfused 与 serial 的输出应一致(~1e-2)
            out_tk = sch.run().float()
            out_serial = serial.run().float()
            rel = ((out_tk - out_serial).norm() / out_serial.norm().clamp_min(1e-9))
            rel_t = rel.detach().reshape(1).clone()
            dist.all_reduce(rel_t, op=dist.ReduceOp.MAX)
            rel_err = float(rel_t.item())

            agg = torch.tensor([float(npl)], device=device)
            dist.all_reduce(agg, op=dist.ReduceOp.MAX)
            npl_max = int(agg[0].item())

            if rank == 0:
                row = {
                    "tokens_total": total, "tokens_per_rank": T,
                    "num_padded_local": npl_max,
                    **{f"{k}_ms": v for k, v in t.items()},
                    "l0_fusion_penalty_ms": t["L0_fused"] - t["L0_gemm"],
                    "l1_fusion_penalty_ms": t["L1_fused"] - t["L1_gemm"],
                    "tk_e2e_plus_sched_ms": t["tk_e2e"] + t["sched"],
                    "speedup_vs_serial": t["serial_e2e"] / t["tk_e2e"],
                    "speedup_vs_serial_with_sched":
                        t["serial_e2e"] / (t["tk_e2e"] + t["sched"]),
                    "tk_vs_serial_rel_err": rel_err,
                }
                rows.append(row)
                print(f"[mb4] tokens={total:5d}  serial={t['serial_e2e']*1e3:8.1f}us  "
                      f"tk={t['tk_e2e']*1e3:8.1f}us (+sched {t['sched']*1e3:5.0f}us)  "
                      f"x{row['speedup_vs_serial']:.2f}/"
                      f"x{row['speedup_vs_serial_with_sched']:.2f}  "
                      f"L0pen={row['l0_fusion_penalty_ms']*1e3:6.0f}us "
                      f"L1pen={row['l1_fusion_penalty_ms']*1e3:6.0f}us  "
                      f"relerr={rel_err:.2e}", flush=True)

            serial.close()
            del sch, problem, serial
            torch.cuda.empty_cache()

        if rank == 0:
            out_list.append({"rows": rows, "tk_env": tk_env})
    finally:
        torch.cuda.synchronize()
        dist.destroy_process_group()


def main():
    out = common.spawn_workers(_worker)
    assert out, "no results collected"
    meta = common.result_meta({"test": "mb4_fusion", "tk_env": out[0]["tk_env"]})
    common.write_json("mb4_fusion", meta, out[0]["rows"])


if __name__ == "__main__":
    main()
