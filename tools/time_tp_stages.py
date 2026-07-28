# SPDX-License-Identifier: Apache-2.0
"""TP 分阶段归因(方法论见 docs/03):在各 rank 上同步地给 tktp run() 的每个
阶段计时, 并给出"同 GEMM 纯算"对照, 于是每个融合阶段暴露出来的通信/排空
成本可以直接读出来:

  L0 exposure = t(L0 fused) - t(L0 GEMM alone)     (AG + gate 等待)
  L1 exposure = t(L1 fused) - t(L1 GEMM alone)     (预归约/push 排空)

  python -m moe_bench.tools.time_tp_stages [ne] [iters] [tokens_per_rank]
      [--dist balanced|uniform|skewed|single] [--skew-alpha A] [--active N]

形状/精度取自主配置 configs/tp_rtx_pro5000_4gpu_fp8.yaml, 只覆盖 CLI 给的
num_experts / num_tokens / 迭代数 / token 路由分布(覆盖项会打印出来)。每阶段取**各 rank 的最大值**(最慢者定门),
再对迭代取均值。阶段间的 cuda.synchronize 会轻微扰动重叠, 但融合 kernel
本身是原样跑的。
"""
from __future__ import annotations

import dataclasses
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "configs", "tp_rtx_pro5000_4gpu_fp8.yaml")


def _worker(rank, world, init_method, ne, iters, tokens, routing):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)
    dist.all_reduce(torch.tensor([rank], device=device))

    from moe_bench.config import Distribution, MoEBenchConfig
    from moe_bench.context import DistContext
    from moe_bench.data import make_problem, make_weights
    from moe_bench.routing_stats import (
        enabled as routing_stats_enabled,
        expert_token_counts,
        print_routing_stats,
    )
    from moe_bench.tk_tp_scheme import TKFusedTP

    cfg = MoEBenchConfig.from_file(CONFIG)
    over = dict(num_experts=ne, num_tokens=[tokens], world_size=world,
                verify=False)
    if routing is not None:
        over["routing"] = routing
    cfg = dataclasses.replace(cfg, **over)
    ctx = DistContext(rank=rank, world_size=world, local_rank=rank,
                      device=device, group=None)
    weights = make_weights(cfg, rank)
    problem = make_problem(cfg, tokens, rank=rank, weights=weights)
    s = TKFusedTP()
    s.setup(problem, ctx)

    # 路由分布 + BLOCK 布局。counts 是全局口径, all-reduce 要所有 rank 参与,
    # 打印只在 rank 0。
    if routing_stats_enabled():
        counts = expert_token_counts(problem.topk_ids, cfg.num_experts)
        if rank == 0:
            print_routing_stats(
                counts, s.block_config(),
                f"tktp stages, E={ne}, topk={cfg.topk}, T={tokens}/rank × "
                f"{world} = {tokens * world} tokens, "
                f"dist={cfg.routing.distribution.value}")

    # scratch for GEMM-alone references (same shapes as the fused calls)
    ref_gateup = torch.empty(s.num_padded_total, 2 * s.inter, device=device,
                             dtype=torch.bfloat16)
    ref_expout = torch.empty_like(s.expert_out)
    ref_task_next = torch.zeros(1, dtype=torch.int32, device=device)

    stages = ["sched", "tok_copy", "L0_fused", "L1_fused", "final_red",
              "L0_gemm_alone", "L1_gemm_alone", "full_run"]
    acc = {k: 0.0 for k in stages}

    def timed(key, fn):
        dist.barrier()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        acc[key] += e0.elapsed_time(e1) * 1000  # us

    for _ in range(3):  # warmup (captures the sched graph too)
        s.run()
    torch.cuda.synchronize()

    for _ in range(iters):
        def st_sched():
            s._packed_local[..., 0].copy_(s._topk_ids_local)
            s._packed_local[..., 1].copy_(s._topk_w_bits)
            dist.all_gather_into_tensor(s._packed_all.view(-1), s._packed_local.view(-1))
            s._sched_graph.replay()
        timed("sched", st_sched)

        def st_copy():
            s._l0_seq += 1
            s.tk.pcie_device_barrier(s.barrier_l0, s._l0_seq)
            s.tk.rowgroup_quant_fp8(s.problem.hidden_states,
                                    s.pre_tokens.data_, s.pre_scales.data_)
            s._l0_seq += 1
            s.tk.pcie_device_barrier(s.barrier_l0, s._l0_seq)
        timed("tok_copy", st_copy)

        def st_l0():
            s.gemm_next.zero_()
            s.push_next.zero_()
            s.pull_next.zero_()
            s.tk.moe_tp_dispatch_gemm_fp8_push(
                s.pre_tokens, s.pre_scales, s.ag_staging_fp8,
                s.ag_sscales, s.ag_flags, s.gathered, s.gathered_scales,
                s.w_gateup_fp8, s.w1_il_scales, s.act, s.padded,
                s.tp_slots, s.slack, s.pull_order, s.push_order,
                s.blk_expert, s.gemm_next, s.push_next, s.pull_next,
                s.barrier_l0, s.num_comm_sms, s.l0_push_sms,
                s.num_padded_total, s.num_tokens, s._l0_seq, s.two_level,
                s.l0_no_gate)
        timed("L0_fused", st_l0)

        def st_l1():
            s._l1_seq += 1
            s.combine_local_cnt.zero_()
            s.job_next.zero_()
            s.l1_gemm_next.zero_()
            s.tk.rowgroup_quant_fp8(s.act, s.act_fp8, s.act_scales)
            s.tk.moe_tp_gemm_prered_push_fp8(
                s.act_fp8, s.act_scales, s.w2_fp8, s.w2_scales,
                s.expert_out, s.padded, s.combine_partial,
                s.slot_job, s.slot_w, s.combine_staging,
                s.prered_dst, s.tp_slots, s.prered_w, s.combine_local_cnt,
                s.push_expected_l1, s.blk_expert, s.l1_gemm_next,
                s.job_order, s.job_next, s.barrier_l1, s.num_comm_sms_l1,
                s.num_padded_total, s.num_tokens, s.num_jobs, s._l1_seq,
                s.l1_epired, s.slack, 0 if s.l1_epired else s.two_level,
                s.l1_no_gate)
        timed("L1_fused", st_l1)

        timed("final_red", lambda: s.tk.moe_final_reduce_push(
            s.combine_staging, s.final_contrib, s.recv_from, s.combine_out,
            s.barrier_l1, s.num_tokens, s._l1_seq))

        # 参考: 同样的 GEMM, 没有通信/预归约(gathered / act 已经是填好的)。
        # gg8 是 plain store(输出 (P, 2I)), 少了 GLU epilogue —— 实测 GLU 在
        # GEMM 级零开销, 参考仍然苹果对苹果。
        # 两级 tile 必须跟着 fused 一起开, 否则参考走全满块而 fused 走尾块,
        # exposure 会被系统性低估(balanced 无尾块 → slack 全 0, 行为与旧口径
        # 逐指令相同, 历史数字不受影响; uniform / local_first 下才有差别)。
        def _gemm_alone(a, asc, w, wsc, out):
            ref_task_next.zero_()
            if s.two_level:
                s.tk.grouped_gemm_fp8(a, asc, w, wsc, out, s.padded,
                                      s.blk_expert, ref_task_next, 0, False,
                                      s.slack)
            else:
                s.tk.grouped_gemm_fp8(a, asc, w, wsc, out, s.padded,
                                      s.blk_expert, ref_task_next, 0, False)

        timed("L0_gemm_alone", lambda: _gemm_alone(
            s.gathered, s.gathered_scales, s.w_gateup_fp8, s.w1_il_scales,
            ref_gateup))
        timed("L1_gemm_alone", lambda: _gemm_alone(
            s.act_fp8, s.act_scales, s.w2_fp8, s.w2_scales, ref_expout))

        timed("full_run", s.run)

    t = torch.tensor([acc[k] / iters for k in stages], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        rt = cfg.routing
        rdesc = rt.distribution.value
        if rt.distribution == Distribution.SKEWED:
            rdesc += f"(a={rt.skew_alpha})"
        if rt.num_active_experts is not None:
            rdesc += f",act={rt.num_active_experts}"
        print(f"\n== tktp stage attribution (fp8, NE={ne}, T={tokens}, "
              f"dist={rdesc}, iters={iters}, comm_sms={s.num_comm_sms}, "
              f"comm_sms_l1={s.num_comm_sms_l1}, push_sms={s.l0_push_sms}, "
              f"two_level={s.two_level}, local_first={s.local_first}, "
              f"no_gate={s.l0_no_gate}/{s.l1_no_gate}, epired={s.l1_epired}, "
              f"P={s.num_padded_total}, "
              f"max over ranks, us) ==")
        for k, v in zip(stages, t.tolist()):
            print(f"  {k:14} {v:10.1f}")
        r = dict(zip(stages, t.tolist()))
        print(f"  -> L0 exposure {r['L0_fused'] - r['L0_gemm_alone']:10.1f}")
        print(f"  -> L1 exposure {r['L1_fused'] - r['L1_gemm_alone']:10.1f}")
        print(f"  -> stage sum   {sum(r[k] for k in stages[:5]):10.1f} "
              f"(vs full_run {r['full_run']:.1f})")
    dist.destroy_process_group()


def main():
    flag_cast = {"--dist": str, "--skew-alpha": float, "--active": int}
    argv, flags, pos = sys.argv[1:], {}, []
    i = 0
    while i < len(argv):
        if argv[i] in flag_cast:
            flags[argv[i]] = flag_cast[argv[i]](argv[i + 1])
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    ne = int(pos[0]) if len(pos) > 0 else 64
    iters = int(pos[1]) if len(pos) > 1 else 20
    tokens = int(pos[2]) if len(pos) > 2 else 512

    from moe_bench.config import Distribution, MoEBenchConfig, RoutingConfig
    cfg = MoEBenchConfig.from_file(CONFIG)
    world = cfg.world_size
    routing = None
    if flags:
        d, alpha, active = (flags.get("--dist"), flags.get("--skew-alpha"),
                            flags.get("--active"))
        routing = RoutingConfig(
            distribution=Distribution(d) if d else cfg.routing.distribution,
            skew_alpha=alpha if alpha is not None else cfg.routing.skew_alpha,
            num_active_experts=active if active is not None
            else cfg.routing.num_active_experts)
        # 在 spawn 前触发 __post_init__ 校验(如 topk > active), 避免 4 个
        # worker 各自炸一遍。
        dataclasses.replace(cfg, num_experts=ne, routing=routing)
        print(f"[time_tp_stages] config={os.path.basename(CONFIG)} "
              f"routing overrides={flags}")
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mp.spawn(_worker, args=(world, init_method, ne, iters, tokens, routing),
             nprocs=world, join=True)


if __name__ == "__main__":
    main()
