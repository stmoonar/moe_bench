# SPDX-License-Identifier: Apache-2.0
"""MB2 通信 kernel 单测:NCCL 集合通信(vLLM serial 用)vs TK 通信 kernel。

同一 token 数下,只测通信,不含 GEMM:

  vLLM serial 侧(NCCL):
    nccl_ag_hidden : AllGather token shard (T,H) bf16   —— serial dispatch 主体
    nccl_ag_ids    : AllGather topk_ids (T,8) int32     —— serial dispatch 路由
    nccl_ag_w      : AllGather topk_weights (T,8) f32
    nccl_rs        : ReduceScatter (4T,H)->(T,H) bf16   —— serial combine
    nccl_a2a       : AllToAll (4T,H) 均分                —— NCCL 侧 dispatch 替代方案参考
    nccl_ar_small  : AllReduce 4B                        —— NCCL 信号延迟参考

  TK 侧:
    tk_barrier      : pcie_device_barrier(跨卡水位/发布信号的基础延迟)
    tk_push_data    : push2 数据面(moe_push_data,源侧 TMA push 每条 assignment
                      行到 expert 卡 gathered;强路径,无原子)—— TK dispatch 通信量
    tk_final_reduce : T6-v0 final_reduce(源卡跨卡 pull 各贡献卡 partial 行并求和)
                      —— TK combine 通信(弱路径 pull)

口径说明:
- NCCL AG 与 TK push 的语义字节数不同:AG 搬 shard 复制(每卡收 (world-1)·T 行),
  TK dispatch 按 assignment 搬(每卡发 T·topk 行,其中 ~3/4 跨卡)。结果里同时给出
  时间、字节数与有效带宽,分析时以字节归一。
- busbw 用 NCCL 惯例:S·(n-1)/n / t,S 为全量张量字节数。

  python -m moe_bench.microbench.mb2_comm
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from . import common


def _worker(rank, world, init_method, out_list):
    # push_data 需要 w_gate(FUSE_GATEUP=0);final_reduce 需要 partials(prered)
    tk_env = common.apply_tk_env(TK_FUSE_GATEUP="0", TK_COMBINE="prered")
    device, ctx = common.init_dist_worker(rank, world, init_method)

    warmup, iters = common.warmup_iters(), common.bench_iters()
    rows = []
    try:
        for total in common.global_token_sweep():
            T = total // world
            cfg = common.make_cfg(T)
            sch, problem = common.build_tk_fixture(cfg, T, rank, ctx)
            tk = sch.tk
            H = sch.H
            m_full = T * world
            dt = cfg.torch_dtype  # 激活基精度(bf16/fp16;fp8 的基精度也是 bf16)

            # NCCL 缓冲
            hidden_local = problem.hidden_states.contiguous()
            hidden_full = torch.empty(m_full, H, device=device, dtype=dt)
            ids_local = problem.topk_ids.contiguous()
            ids_full = torch.empty(m_full, cfg.topk, device=device,
                                   dtype=ids_local.dtype)
            w_local = problem.topk_weights.contiguous()
            w_full = torch.empty(m_full, cfg.topk, device=device, dtype=w_local.dtype)
            result_full = torch.randn(m_full, H, device=device, dtype=dt)
            out_local = torch.empty(T, H, device=device, dtype=dt)
            a2a_in = torch.randn(m_full, H, device=device, dtype=dt)
            a2a_out = torch.empty(m_full, H, device=device, dtype=dt)
            small = torch.zeros(1, device=device)  # 零值:反复 all_reduce 不会溢出

            # TK 缓冲:pre_tokens 填一次;partials 填随机并发布(final_reduce 的输入)
            sch.pre_tokens.data_.copy_(problem.hidden_states)
            torch.nn.init.normal_(sch.partials.data_, std=0.02)
            sch._l0_seq += 1
            tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)

            def nccl_ag_hidden():
                dist.all_gather_into_tensor(hidden_full, hidden_local)

            def nccl_ag_ids():
                dist.all_gather_into_tensor(ids_full, ids_local)

            def nccl_ag_w():
                dist.all_gather_into_tensor(w_full, w_local)

            def nccl_rs():
                dist.reduce_scatter_tensor(out_local, result_full)

            def nccl_a2a():
                dist.all_to_all_single(a2a_out, a2a_in)

            def nccl_ar_small():
                dist.all_reduce(small)

            def tk_barrier():
                sch._l0_seq += 1
                tk.pcie_device_barrier(sch.barrier_l0, sch._l0_seq)

            def tk_push_data():
                tk.moe_push_data(sch.pre_tokens, sch.gathered, sch.w_gate,
                                 sch.gate_out, sch.padded, sch.push_idx,
                                 sch.push_src, sch.barrier_l0, sch.num_push)

            def tk_final_reduce():
                tk.moe_final_reduce(sch.partials, sch.final_contrib,
                                    sch.combine_out, sch.barrier_l0,
                                    sch.num_tokens)

            t = {}
            for name, fn in [("nccl_ag_hidden", nccl_ag_hidden),
                             ("nccl_ag_ids", nccl_ag_ids),
                             ("nccl_ag_w", nccl_ag_w),
                             ("nccl_rs", nccl_rs),
                             ("nccl_a2a", nccl_a2a),
                             ("nccl_ar_small", nccl_ar_small),
                             ("tk_barrier", tk_barrier),
                             ("tk_push_data", tk_push_data),
                             ("tk_final_reduce", tk_final_reduce)]:
                t[name] = common.lockstep_median_ms(fn, warmup, iters)
            t = common.maxrank_timings(t, device)

            # 字节账本(本 rank,取 max-over-ranks)
            remote_push = int((sch.push_idx[:, 0] != rank).sum().item())
            fc = sch.final_contrib
            remote_cols = [d for d in range(world) if d != rank]
            remote_pull_rows = int(fc[:, remote_cols].sum().item())
            agg = torch.tensor([float(remote_push), float(remote_pull_rows),
                                float(sch.num_push)], device=device)
            dist.all_reduce(agg, op=dist.ReduceOp.MAX)
            remote_push, remote_pull_rows, num_push = (
                int(agg[0].item()), int(agg[1].item()), int(agg[2].item()))

            if rank == 0:
                esz = hidden_full.element_size()  # 激活元素字节数(bf16/fp16=2)
                full_bytes = m_full * H * esz
                def busbw(bytes_full, ms):
                    return bytes_full * (world - 1) / world / (ms * 1e-3) / 1e9

                row = {
                    "tokens_total": total, "tokens_per_rank": T,
                    **{f"{k}_ms": v for k, v in t.items()},
                    "shard_bytes": T * H * esz,
                    "nccl_ag_hidden_busbw_gbps": busbw(full_bytes, t["nccl_ag_hidden"]),
                    "nccl_rs_busbw_gbps": busbw(full_bytes, t["nccl_rs"]),
                    "nccl_a2a_busbw_gbps": busbw(full_bytes, t["nccl_a2a"]),
                    "tk_push_rows": num_push,
                    "tk_push_remote_rows": remote_push,
                    "tk_push_remote_bytes": remote_push * H * esz,
                    "tk_push_gbps":
                        remote_push * H * esz / (t["tk_push_data"] * 1e-3) / 1e9,
                    "tk_final_reduce_remote_rows": remote_pull_rows,
                    "tk_final_reduce_remote_bytes": remote_pull_rows * H * esz,
                    "tk_final_reduce_gbps":
                        remote_pull_rows * H * esz / (t["tk_final_reduce"] * 1e-3) / 1e9,
                    # serial 的全部通信(dispatch AG×3 + combine RS)
                    "serial_comm_total_ms": (t["nccl_ag_hidden"] + t["nccl_ag_ids"]
                                             + t["nccl_ag_w"] + t["nccl_rs"]),
                }
                rows.append(row)
                print(f"[mb2] tokens={total:5d}  "
                      f"AGh={t['nccl_ag_hidden']*1e3:7.1f}us "
                      f"RS={t['nccl_rs']*1e3:7.1f}us "
                      f"A2A={t['nccl_a2a']*1e3:7.1f}us | "
                      f"tk_push={t['tk_push_data']*1e3:7.1f}us "
                      f"({row['tk_push_gbps']:5.1f}GB/s) "
                      f"tk_finred={t['tk_final_reduce']*1e3:7.1f}us "
                      f"({row['tk_final_reduce_gbps']:5.1f}GB/s) "
                      f"barrier={t['tk_barrier']*1e3:6.1f}us", flush=True)

            del sch, problem, hidden_full, ids_full, w_full
            del result_full, out_local, a2a_in, a2a_out
            torch.cuda.empty_cache()

        if rank == 0:
            out_list.append({"rows": rows, "tk_env": tk_env})
    finally:
        torch.cuda.synchronize()
        dist.destroy_process_group()


def main():
    out = common.spawn_workers(_worker)
    assert out, "no results collected"
    meta = common.result_meta({"test": "mb2_comm", "tk_env": out[0]["tk_env"]})
    common.write_json("mb2_comm", meta, out[0]["rows"])


if __name__ == "__main__":
    main()
