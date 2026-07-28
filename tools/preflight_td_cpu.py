# SPDX-License-Identifier: Apache-2.0
"""tktd(TD 风格 TP 复刻, docs/17)的 CPU preflight — 本机可跑(无 GPU/vllm/
torch.distributed), 复用 preflight_tp_cpu 的加载器/DEVICE GUARD/伪 all_gather:

  1. TD 专用表: host golden(纯 python 循环) vs GPU 向量化 builder 逐元素
     一致, 外加不变量(0<=lo<=hi<world; row_perm/pull_order 双射; stage 沿
     row_perm 非降; stage 0 的块只含本 rank 行; pull_order 前 T 项恰为本
     rank 的 token 升序)。
  2. dispenser CHUNKED 任务映射(sm120_common.cuh 的公式复刻): 对 (row, col)
     双射, 且与"chunk 外层、行主序内层"的嵌套循环枚举逐项相同。
  3. L1 reduce-RS 数据流仿真: 按 kernel 的索引算术(chunk 列窗 -> 加权归约
     -> staging[dst] 行 my_dev*T+tok -> 最终对 world 个 plane 求和)对拍
     直接参考(sum_dev sum_k w * expert_out_dev[slot])。

  python moe_bench/tools/preflight_td_cpu.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_mods():
    helper = _load("preflight_tp_cpu", os.path.join(HERE, "preflight_tp_cpu.py"))
    mtp = helper._load_tk_tp_scheme()
    mtd = _load("moe_bench.tk_td_scheme", os.path.join(BASE, "tk_td_scheme.py"))
    return helper, mtp, mtd


def check_td_tables(helper, mtp, mtd):
    fails = []
    world, T, topk = 4, 512, 8
    for dist_kind in ("balanced", "skewed"):
        for ne in (64, 256):
            for rank in (0, 3):
                tag = f"[td_tables {dist_kind} ne={ne} rank={rank}]"
                ids, w = helper._make_topk(world, T, ne, topk, dist_kind, 0)
                helper._STATE["ids"], helper._STATE["w"] = ids, w
                with helper.RequireExplicitDevice():
                    (padded, slots, w_t, slack, _pc, _jo, _po, blk_e, sjob,
                     _sw, _rp, P) = mtp._build_tp_schedules(
                        ids[rank], w[rank], T, world, ne, rank, "cpu")
                    g_lo, g_hi, g_perm, g_pull = mtd._td_tables_golden(
                        sjob, P, T, world, rank)
                    nblk = P // mtd.ROW_BLOCK
                    out = {
                        "blk_lo": torch.empty(nblk, dtype=torch.int32, device="cpu"),
                        "blk_hi": torch.empty(nblk, dtype=torch.int32, device="cpu"),
                        "row_perm": torch.empty(nblk, dtype=torch.int32, device="cpu"),
                        "pull_order": torch.empty(world * T, dtype=torch.int32,
                                                  device="cpu"),
                    }
                    mtd._build_td_tables_gpu(sjob, T, world, rank, out)
                for name, gold in (("blk_lo", g_lo), ("blk_hi", g_hi),
                                   ("row_perm", g_perm), ("pull_order", g_pull)):
                    if not torch.equal(out[name], gold):
                        fails.append(f"{tag} {name}: GPU builder != golden")
                # ---- 不变量 ----
                lo, hi = g_lo.long(), g_hi.long()
                if not bool(((lo >= 0) & (lo <= hi) & (hi < world)).all()):
                    fails.append(f"{tag} 区间越界")
                for name, t_ in (("row_perm", g_perm), ("pull_order", g_pull)):
                    if not torch.equal(torch.sort(t_.long()).values,
                                       torch.arange(t_.numel(), device="cpu")):
                        fails.append(f"{tag} {name} 不是双射")
                dist_of = ((torch.arange(world, device="cpu") - rank) % world)
                # stage 独立重算(区间内 ring 距离最大值), 沿 row_perm 非降
                dr = torch.arange(world, device="cpu").unsqueeze(0)
                in_range = (dr >= lo.unsqueeze(1)) & (dr <= hi.unsqueeze(1))
                stage = torch.where(in_range, dist_of.unsqueeze(0),
                                    dist_of.new_full((), -1)).amax(1)
                sp = stage[g_perm.long()]
                if not bool((sp[1:] >= sp[:-1]).all()):
                    fails.append(f"{tag} stage 沿 row_perm 非单调")
                # stage 0 的块只含本 rank 行(区间 == [rank, rank])
                z = g_perm.long()[sp == 0]
                if not bool(((lo[z] == rank) & (hi[z] == rank)).all()):
                    fails.append(f"{tag} stage0 块含非本 rank 行")
                # 块 src 区间与 slot_job 直查一致(独立于 golden 的抽查)
                sj = sjob[:nblk * mtd.ROW_BLOCK].view(nblk, mtd.ROW_BLOCK).long()
                src = torch.div(sj, T, rounding_mode="floor")
                real = sj >= 0
                lo2 = torch.where(real, src, src.new_full((), world)).amin(1)
                hi2 = torch.where(real, src, src.new_full((), -1)).amax(1)
                if not (torch.equal(lo, lo2) and torch.equal(hi, hi2)):
                    fails.append(f"{tag} 区间与 slot_job 直查不一致")
                # pull_order: src 的 ring 距离沿序非降; 前 T 项 = 本 rank token 升序
                psrc = dist_of[(g_pull.long() // T)]
                if not bool((psrc[1:] >= psrc[:-1]).all()):
                    fails.append(f"{tag} pull_order ring 距离非单调")
                exp_local = torch.arange(rank * T, (rank + 1) * T, device="cpu")
                if not torch.equal(g_pull[:T].long(), exp_local):
                    fails.append(f"{tag} pull_order 前 T 项非本 rank token")
    print(f"check_td_tables: {'FAIL' if fails else 'ok'}")
    return fails


def check_chunked_mapping():
    """复刻 sm120_common.cuh CHUNKED 分支的任务映射公式并裁决。"""
    fails = []
    for nblk, col_blocks, n_chunks in ((11, 64, 4), (7, 64, 16), (5, 64, 64),
                                       (13, 64, 1), (9, 24, 4)):
        chunk_cols = col_blocks // n_chunks
        num_tasks = nblk * col_blocks
        got = []
        for t in range(num_tasks):
            tpc = nblk * chunk_cols
            chunk = t // tpc
            rem = t - chunk * tpc
            rb = rem // chunk_cols
            col = chunk * chunk_cols + (rem - rb * chunk_cols)
            got.append((rb, col))
        want = [(rb, c * chunk_cols + k)
                for c in range(n_chunks)
                for rb in range(nblk)
                for k in range(chunk_cols)]
        tag = f"[chunked nblk={nblk} cb={col_blocks} nc={n_chunks}]"
        if got != want:
            fails.append(f"{tag} 枚举序与嵌套循环不一致")
        if len(set(got)) != num_tasks:
            fails.append(f"{tag} 非双射")
    print(f"check_chunked_mapping: {'FAIL' if fails else 'ok'}")
    return fails


def check_rs_dataflow(helper, mtp, mtd):
    """L1 reduce-RS 的索引算术仿真(小形状), 对拍直接参考。"""
    fails = []
    world, T, topk, ne, H, n_chunks = 4, 64, 8, 16, 32, 4
    ce = H // n_chunks
    ids, w = helper._make_topk(world, T, ne, topk, "skewed", 3)
    helper._STATE["ids"], helper._STATE["w"] = ids, w
    (padded, slots, w_t, _sl, _pc, _jo, _po, _be, sjob, _sw, _rp,
     P) = mtp._build_tp_schedules(ids[0], w[0], T, world, ne, 0, "cpu")
    S = world * T
    g = torch.Generator().manual_seed(9)
    exp = [torch.randn(P, H, generator=g, device="cpu") for _ in range(world)]
    staging = [torch.zeros(S, H, device="cpu") for _ in range(world)]
    for dev in range(world):            # 每张卡的 RS kernel
        for c in range(n_chunks):
            c0 = c * ce
            for j in range(S):
                dst, tok = j // T, j % T
                acc = torch.zeros(ce, device="cpu")
                for k in range(topk):
                    s = int(slots[j, k])
                    if s >= 0:
                        acc += float(w_t[j, k]) * exp[dev][s, c0:c0 + ce]
                staging[dst][dev * T + tok, c0:c0 + ce] = acc
    for r in range(world):              # 源卡最终归约
        for t in range(0, T, 17):       # 抽查
            out = sum(staging[r][d * T + t] for d in range(world))
            j = r * T + t
            ref = sum(float(w_t[j, k]) * exp[d][int(slots[j, k])]
                      for d in range(world) for k in range(topk)
                      if int(slots[j, k]) >= 0)
            if not torch.allclose(out, ref, atol=1e-4, rtol=1e-4):
                fails.append(f"[rs_dataflow] rank={r} tok={t} 不匹配")
    print(f"check_rs_dataflow: {'FAIL' if fails else 'ok'}")
    return fails


def main():
    helper, mtp, mtd = _load_mods()
    torch.distributed.all_gather_into_tensor = helper._fake_all_gather_into_tensor
    fails = check_td_tables(helper, mtp, mtd)
    fails += check_chunked_mapping()
    fails += check_rs_dataflow(helper, mtp, mtd)
    for f in fails:
        print("  " + f)
    print("PREFLIGHT-TD", "FAIL" if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
