# SPDX-License-Identifier: Apache-2.0
"""方案A(通信 warp 化)定位工具：把"融合后的 GEMM 比纯 GEMM 慢多少"拆成
可辨成分，并逐迭代打印（tp_run_20260716_063155 的 L0 双稳直接可见）。

L0 探针阶梯（kernel_warp_probe 的两个旋钮：gate_off / pull_next 预填）：

  gg8_alone     纯 fp8 dispenser GEMM（全 SM, plain store）——跨口径参考
  l0_warp_gemm  gate 关 + pull 跳过（pull_next=s_max, comm lane 立即退出)
                = 满 SM 纯 GEMM 硬上限（warp 几何, 含 GLU store）
  l0_warp_ng    gate 关 + pull 正常 = 上限 + 共存税（发射槽/TMA 队列争用,
                GEMM 不等数据）—— 机制②(TMA HoL)的直接读数
  l0_warp_s{n}  gate 开 + pull 正常, n 个 comm lane = 完整融合;
                l0_warp_s4 - l0_warp_ng = 数据等待/straggler 车队税(机制①)
  l0_default    默认路径（comm 块, TK_COMM_SMS）—— 现役对照

L1 阶梯（无需新 kernel：job_next 预填 = 排空团队立即退出）：

  l1_gemm_alone 纯 fp8 W2 GEMM（gg8, 全 SM）
  l1_warp_gemm  warp kernel + job_next 预填 = 满 SM GEMM + signal epilogue
  l1_warp       完整 warp 融合
  l1_default    默认路径（comm 块）

⚠️ gate 关掉的档位 GEMM 会读 gathered 旧值，输出无意义 —— 只用于计时。

  python -m moe_bench.tools.diag_warp [ne] [iters] [tokens_per_rank]

每档报 max-over-ranks 的逐迭代 µs + med/min 汇总 + 派生税目。
"""
from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

L0_SLOTS = [4, 2, 1]   # gated 完整融合的 comm lane 数 sweep


def _worker(rank, world, init_method, ne, iters, tokens):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method=init_method,
                            rank=rank, world_size=world, device_id=device)
    dist.all_reduce(torch.tensor([rank], device=device))

    from moe_bench.config import Distribution, MoEBenchConfig, ParallelMode, Precision, RoutingConfig
    from moe_bench.context import DistContext
    from moe_bench.data import make_problem, make_weights
    from moe_bench.tk_tp_scheme import TKFusedTP

    cfg = MoEBenchConfig(
        hidden_size=4096, intermediate_size=3072, num_experts=ne, topk=8,
        parallel_mode=ParallelMode.TP, world_size=world, precision=Precision.FP8,
        num_tokens=[tokens], routing=RoutingConfig(distribution=Distribution.BALANCED),
        distributed=True, verify=False, device="cuda")
    ctx = DistContext(rank=rank, world_size=world, local_rank=rank,
                      device=device, group=None)
    weights = make_weights(cfg, rank)
    problem = make_problem(cfg, tokens, rank=rank, weights=weights)
    s = TKFusedTP()
    s.setup(problem, ctx)
    assert s.fp8 and s.l1_fp8, "diag_warp 只支持 fp8 + TK_L1_FP8=1 口径"

    ref_gateup = torch.empty_like(s.gateup_out)
    ref_expout = torch.empty_like(s.expert_out)
    ref_task_next = torch.zeros(1, dtype=torch.int32, device=device)
    s_max = world * tokens

    # ---- 阶段定义（每个都是完整 kernel 调用, 独立可计时） ----
    def proto():
        """iter 前置协议: 与 run() 相同的 barrier+quant, 保证 peers 的
        pre_tokens 有效且 seq 口径一致(不计入任何阶段)。"""
        s._l0_seq += 1
        s.tk.pcie_device_barrier(s.barrier_l0, s._l0_seq)
        s.tk.rowgroup_quant_fp8(s.problem.hidden_states,
                                s.pre_tokens.data_, s.pre_scales.data_)
        s._l0_seq += 1
        s.tk.pcie_device_barrier(s.barrier_l0, s._l0_seq)

    def gg8_alone():
        ref_task_next.zero_()
        s.tk.grouped_gemm_fp8(s.gathered, s.gathered_scales, s.w_gateup_fp8,
                              s.w1_il_scales, ref_gateup, s.padded,
                              s.blk_expert, ref_task_next, 0, False)

    def _l0_probe(gate_off, skip_pull, num_slots):
        s.gemm_next.zero_()
        if skip_pull:
            s.pull_next.fill_(s_max)
        else:
            s.pull_next.zero_()
        s.tk.moe_tp_dispatch_gemm_fp8_warp_probe(
            s.pre_tokens, s.pre_scales, s.ag_tokens, s.ag_scales,
            s.gathered, s.gathered_scales, s.w_gateup_fp8, s.w1_il_scales,
            s.act, s.padded, s.tp_slots, s.slack, s.pull_order, s.blk_expert,
            s.gemm_next, s.pull_next, s.barrier_l0,
            s.num_padded_total, s.num_tokens, gate_off, num_slots)

    def l0_default():
        s.gemm_next.zero_()
        s.tk.moe_tp_dispatch_gemm_fp8(
            s.pre_tokens, s.pre_scales, s.ag_tokens, s.ag_scales, s.ce_flags,
            s.gathered, s.gathered_scales, s.w_gateup_fp8, s.w1_il_scales,
            s.act, s.padded, s.tp_slots, s.slack, s.pull_order, s.blk_expert,
            s.gemm_next, s.barrier_l0, s.num_comm_sms,
            s.num_padded_total, s.num_tokens, False)

    def l1_quant():   # L1 各档共用的 act 量化(不计时, act 来自 L0 各档)
        s.tk.rowgroup_quant_fp8(s.act, s.act_fp8, s.act_scales)

    def l1_gemm_alone():
        ref_task_next.zero_()
        s.tk.grouped_gemm_fp8(s.act_fp8, s.act_scales, s.w2_fp8, s.w2_scales,
                              ref_expout, s.padded, s.blk_expert,
                              ref_task_next, 0, False)

    def _l1_warp(skip_jobs):
        s._l1_seq += 1
        s.combine_local_cnt.zero_()
        s.l1_gemm_next.zero_()
        if skip_jobs:
            s.job_next.fill_(s.num_jobs)
        else:
            s.job_next.zero_()
        s.tk.moe_tp_gemm_prered_push_fp8_warp(
            s.act_fp8, s.act_scales, s.w2_fp8, s.w2_scales, s.expert_out,
            s.out_planes, s.padded, s.combine_staging, s.prered_dst,
            s.tp_slots, s.prered_w, s.combine_local_cnt, s.push_expected_l1,
            s.blk_expert, s.l1_gemm_next, s.job_order, s.job_next,
            s.barrier_l1, s.num_padded_total, s.num_tokens, s.num_jobs,
            s._l1_seq)

    def l1_default():
        s._l1_seq += 1
        s.combine_local_cnt.zero_()
        s.job_next.zero_()
        s.l1_gemm_next.zero_()
        s.tk.moe_tp_gemm_prered_push_fp8(
            s.act_fp8, s.act_scales, s.w2_fp8, s.w2_scales, s.expert_out,
            s.out_planes, s.padded, s.combine_staging, s.prered_dst,
            s.tp_slots, s.prered_w, s.combine_local_cnt, s.push_expected_l1,
            s.blk_expert, s.l1_gemm_next, s.job_order, s.job_next,
            s.barrier_l1, s.num_comm_sms_l1, s.num_padded_total,
            s.num_tokens, s.num_jobs, s._l1_seq, False)

    stages = [
        ("gg8_alone", gg8_alone),
        ("l0_warp_gemm", lambda: _l0_probe(True, True, 4)),
        ("l0_warp_ng", lambda: _l0_probe(True, False, 4)),
    ]
    stages += [(f"l0_warp_s{n}", (lambda n=n: _l0_probe(False, False, n)))
               for n in L0_SLOTS]
    stages += [
        ("l0_default", l0_default),
        ("l1_gemm_alone", l1_gemm_alone),
        ("l1_warp_gemm", lambda: _l1_warp(True)),
        ("l1_warp", lambda: _l1_warp(False)),
        ("l1_default", l1_default),
    ]
    names = [n for n, _ in stages]
    rec = torch.zeros(len(stages), iters, device=device)

    # warmup: 默认 L0 融合一次填充 gathered/act(gg8/纯 GEMM 档要读),
    # 每个阶段各跑一遍(编译/缓存热身)
    proto()
    l0_default()
    l1_quant()
    for _, fn in stages:
        fn()
    torch.cuda.synchronize()

    for it in range(iters):
        proto()
        l1_quant()
        for si, (_, fn) in enumerate(stages):
            dist.barrier()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            rec[si, it] = e0.elapsed_time(e1) * 1000  # us

    dist.all_reduce(rec, op=dist.ReduceOp.MAX)
    if rank == 0:
        r = rec.cpu()
        print(f"\n== diag_warp (NE={ne}, T={tokens}, iters={iters}, fp8, "
              f"comm_sms={s.num_comm_sms}/{s.num_comm_sms_l1}, "
              f"max over ranks, us) ==")
        hdr = "iter " + " ".join(f"{n:>13}" for n in names)
        print(hdr)
        for it in range(iters):
            print(f"{it:4d} " + " ".join(f"{r[si, it]:13.1f}"
                                         for si in range(len(names))))
        med = r.median(dim=1).values
        mn = r.min(dim=1).values
        d = {n: (med[i].item(), mn[i].item()) for i, n in enumerate(names)}
        print("stage".ljust(14), "med".rjust(9), "min".rjust(9))
        for n in names:
            print(f"{n:14} {d[n][0]:9.1f} {d[n][1]:9.1f}")
        print("-- L0 税目分解(med / min) --")
        for label, a, b in [
            ("融合税(warp)      = l0_warp_s4 - l0_warp_gemm", "l0_warp_s4", "l0_warp_gemm"),
            ("  共存税(发射/TMA) = l0_warp_ng - l0_warp_gemm", "l0_warp_ng", "l0_warp_gemm"),
            ("  gate/straggler  = l0_warp_s4 - l0_warp_ng", "l0_warp_s4", "l0_warp_ng"),
            ("warp几何差         = l0_warp_gemm - gg8_alone", "l0_warp_gemm", "gg8_alone"),
            ("融合税(默认@86SM)  = l0_default - gg8_alone", "l0_default", "gg8_alone"),
        ]:
            print(f"  {label}: {d[a][0]-d[b][0]:+8.1f} / {d[a][1]-d[b][1]:+8.1f}")
        print("-- L1 税目分解(med / min) --")
        for label, a, b in [
            ("融合税(warp)      = l1_warp - l1_warp_gemm", "l1_warp", "l1_warp_gemm"),
            ("warp几何差         = l1_warp_gemm - l1_gemm_alone", "l1_warp_gemm", "l1_gemm_alone"),
            ("融合税(默认@86SM)  = l1_default - l1_gemm_alone", "l1_default", "l1_gemm_alone"),
        ]:
            print(f"  {label}: {d[a][0]-d[b][0]:+8.1f} / {d[a][1]-d[b][1]:+8.1f}")
    dist.destroy_process_group()


def main():
    args = sys.argv[1:]
    ne = int(args[0]) if len(args) > 0 else 64
    iters = int(args[1]) if len(args) > 1 else 30
    tokens = int(args[2]) if len(args) > 2 else 512
    world = 4
    from vllm.utils.network_utils import get_open_port
    init_method = f"tcp://localhost:{get_open_port()}"
    mp.spawn(_worker, args=(world, init_method, ne, iters, tokens),
             nprocs=world, join=True)


if __name__ == "__main__":
    main()
