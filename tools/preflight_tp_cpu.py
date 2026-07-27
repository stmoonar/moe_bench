# SPDX-License-Identifier: Apache-2.0
"""TP CPU preflight — runs ANYWHERE (no GPU, no vllm, no torch.distributed):

  1. DEVICE GUARD (docs/04): every tensor creation inside the TP schedule
     builders must carry an explicit device= — anything else is the
     set_default_device(cuda) bug class that killed rounds 1 and 3 on the
     machine. Enforced with a TorchFunctionMode, not grep discipline.
  2. host golden vs GPU-vectorized builder: element-for-element equality plus
     the slot/slack/order invariants — the schedule tables are adjudicated here,
     nothing else re-checks them.
  3. end-to-end CPU dataflow simulation driven by the real tables vs a naive
     TP MoE reference (validates the semantics the CUDA kernels implement).

Run BEFORE shipping to the GPU box (and the one-click script runs it before
burning any GPU step):

  python moe_bench/tools/preflight_tp_cpu.py        # standalone, stubs deps
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # moe_bench/


def _load_tk_tp_scheme():
    """Load tk_tp_scheme.py with its moe_bench deps stubbed (no vllm/GPU)."""
    if "moe_bench" not in sys.modules:
        pkg = types.ModuleType("moe_bench")
        pkg.__path__ = [BASE]
        sys.modules["moe_bench"] = pkg
    for name, attrs in [
        ("config", {"ParallelMode": type("ParallelMode", (), {"TP": "tp", "EP": "ep"})}),
        ("context", {"DistContext": object}),
        ("data", {"MoEProblem": object}),
        ("schemes", {"DistributedScheme": object}),
    ]:
        m = types.ModuleType(f"moe_bench.{name}")
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[f"moe_bench.{name}"] = m
    spec = importlib.util.spec_from_file_location(
        "moe_bench.tk_tp_scheme", os.path.join(BASE, "tk_tp_scheme.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_bench.tk_tp_scheme"] = mod
    spec.loader.exec_module(mod)
    return mod


_STATE = {}


def _fake_all_gather_into_tensor(out, local):
    key = "ids" if out.dtype in (torch.int32, torch.int64) else "w"
    out.copy_(_STATE[key])


from torch.overrides import TorchFunctionMode  # noqa: E402

_CREATORS = {torch.zeros, torch.ones, torch.empty, torch.full, torch.arange,
             torch.rand, torch.randn, torch.tensor}


class RequireExplicitDevice(TorchFunctionMode):
    """Fail any tensor creation without device= (docs/04 的坑索引)。"""

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _CREATORS and "device" not in kwargs:
            raise RuntimeError(
                f"DEVICE GUARD: {getattr(func, '__name__', func)} without explicit "
                "device= (would land on GPU under set_default_device(cuda))")
        return func(*args, **kwargs)


def _make_topk(world, T, ne, topk, dist_kind, seed):
    g = torch.Generator().manual_seed(seed)
    if dist_kind == "skewed":
        hot = max(topk, ne // 8)
        logits = torch.zeros(world, T, ne, device="cpu")
        logits[..., :hot] += 4.0
        logits += torch.rand(world, T, ne, generator=g, device="cpu")
    else:
        logits = torch.rand(world, T, ne, generator=g, device="cpu")
    ids = torch.topk(logits, topk, dim=-1).indices.to(torch.int32)
    gw = torch.Generator().manual_seed(seed + 777)
    w = torch.rand(world, T, topk, generator=gw, device="cpu").float()
    return ids, w


def check_builders(mod, ROW_BLOCK=128):
    fails = []
    world, T, topk = 4, 512, 8
    # local_first (docs/14) 的表和 canonical 表一起裁决: 同一份路由、同一批
    # 不变量, 只是布局分成本地/远端两段。thr5 = 分段的边际成本阈值(0 = 只分
    # 免费的那批, 99 = 全分段上界档)。
    canon_cost = {}
    for dist_kind in ("balanced", "skewed"):
        for ne in (64, 256):
            for rank, local_first, thr5 in ((0, False, 0), (3, False, 0),
                                            (0, True, 0), (3, True, 0),
                                            (0, True, 2), (3, True, 99)):
                all_ids, all_w = _make_topk(world, T, ne, topk, dist_kind, 0)
                _STATE["ids"], _STATE["w"] = all_ids, all_w
                with RequireExplicitDevice():
                    (padded_g, slots_g, w_g, slack_g, pull_g, job_g, pusho_g,
                     blk_g, sjob_g, sw_g, P) = mod._build_tp_schedules(
                         all_ids[rank], all_w[rank], T, world, ne, rank, "cpu",
                         local_first=local_first, seg_thr5=thr5)
                N = world * T * topk
                packed_all = torch.stack(
                    [all_ids, all_w.contiguous().view(torch.int32)], dim=-1).contiguous()
                out = {
                    "padded": torch.zeros(ne, dtype=torch.int32, device="cpu"),
                    "tp_slots": torch.full((world * T, topk), -1, dtype=torch.int32, device="cpu"),
                    "prered_w": torch.zeros(world * T, topk, dtype=torch.float32, device="cpu"),
                    "slack": torch.zeros(P // ROW_BLOCK, dtype=torch.int32, device="cpu"),
                    "pull_order": torch.zeros(world * T, dtype=torch.int32, device="cpu"),
                    "job_order": torch.zeros(world * T, dtype=torch.int32, device="cpu"),
                    "push_order": torch.zeros(world, T, dtype=torch.int32, device="cpu"),
                    "blk_expert": torch.zeros(P // ROW_BLOCK, dtype=torch.int32, device="cpu"),
                    "slot_job": torch.zeros(P, dtype=torch.int32, device="cpu"),
                    "slot_w": torch.zeros(P, dtype=torch.float32, device="cpu"),
                }
                with RequireExplicitDevice():
                    mod._build_tp_schedules_gpu(packed_all, world, ne, rank, out,
                                                local_first=local_first,
                                                seg_thr5=thr5)
                tag = (f"NE={ne} {dist_kind} rank={rank} "
                       f"lf={int(local_first)}/thr{thr5}")
                for name, got, ref in [("padded", out["padded"], padded_g),
                                       ("tp_slots", out["tp_slots"], slots_g),
                                       ("prered_w", out["prered_w"], w_g),
                                       ("slack", out["slack"], slack_g),
                                       ("pull_order", out["pull_order"], pull_g),
                                       ("job_order", out["job_order"], job_g),
                                       ("push_order", out["push_order"], pusho_g),
                                       ("blk_expert", out["blk_expert"], blk_g),
                                       ("slot_job", out["slot_job"], sjob_g),
                                       ("slot_w", out["slot_w"], sw_g)]:
                    if got.shape != ref.shape or not torch.equal(got, ref):
                        fails.append(f"[{tag}] {name} host/GPU MISMATCH")
                # blk_expert invariants (dispenser GEMM 的 B tile 索引)。
                # 语义判据(对两种布局都完备): 每个行块里的每一条真实
                # assignment 的 expert 必须等于 blk_expert[该行块] —— 行块与
                # expert 的对应才是 dispenser 取 B tile 的依据, 单调性只是
                # canonical 布局的副产品。
                blkl = blk_g.long()
                flat = slots_g.reshape(-1).long()
                eid_flat = all_ids.reshape(-1).long()
                if not torch.equal(blkl[flat // ROW_BLOCK], eid_flat):
                    fails.append(f"[{tag}] blk_expert != assignment's expert")
                nseg = int((blkl[1:] < blkl[:-1]).sum())
                if nseg > (1 if local_first else 0):
                    fails.append(f"[{tag}] blk_expert has {nseg} descents "
                                 f"(expected <= {1 if local_first else 0})")
                if not torch.equal(torch.bincount(blkl, minlength=ne),
                                   padded_g.long() // ROW_BLOCK):
                    fails.append(f"[{tag}] blk_expert counts != padded/ROW_BLOCK")
                # 两级 tile 口径的总块成本(满块 1 / 真实行<=RB/2 的尾块 0.6)
                tail_m = slack_g >= ROW_BLOCK // 2
                cost = float((~tail_m).sum()) + 0.6 * float(tail_m.sum())
                ckey = (ne, dist_kind, rank)
                if not local_first:
                    canon_cost[ckey] = cost
                elif thr5 == 0 and cost > canon_cost[ckey] + 1e-9:
                    # thr5=0 的承诺是**严格帕累托**: 只分边际成本 <= 0 的
                    # expert, 总成本不得高于 canonical(docs/14 §4)。
                    fails.append(f"[{tag}] thr5=0 not free: cost {cost:.1f} > "
                                 f"canonical {canon_cost[ckey]:.1f}")
                if local_first:
                    # 方案的核心正确性(docs/14 §2): 段 1 的行块**一条远端行都
                    # 不能有** —— 这是"零到达等待"的充要条件(它们只由本卡
                    # scatter 的 src == dev_idx 直读分支喂), 而 dispenser 按行块
                    # id 升序发任务, 所以段 1 必须坐在 [0, nb_local)。
                    # 段 1 的块数由分段决策重算(不能用"纯本地块"反推 —— 段 2
                    # 里也可能偶然出现只有本地行的 expert 块)。
                    own = torch.zeros(world * T, dtype=torch.bool, device="cpu")
                    own[rank * T:(rank + 1) * T] = True
                    own_f = own.repeat_interleave(topk)
                    nb = P // ROW_BLOCK
                    has_rem = torch.zeros(nb, dtype=torch.bool, device="cpu")
                    has_rem[flat[~own_f] // ROW_BLOCK] = True
                    c_all = torch.bincount(all_ids.reshape(-1).long(), minlength=ne)
                    c_loc = torch.bincount(all_ids[rank].reshape(-1).long(),
                                           minlength=ne)
                    seg = mod._local_seg_mask(c_loc, c_all - c_loc, c_all, thr5)
                    nb_local = int((((c_loc + ROW_BLOCK - 1) // ROW_BLOCK)
                                    * seg).sum())
                    if nb_local and bool(has_rem[:nb_local].any()):
                        fails.append(f"[{tag}] segment-1 block carries a remote row")
                    if thr5 == 0 and nb_local == 0 and dist_kind != "balanced":
                        fails.append(f"[{tag}] thr5=0 yielded no local block")
                if flat.unique().numel() != N or int(flat.min()) < 0 or int(flat.max()) >= P:
                    fails.append(f"[{tag}] tp_slots not a bijection onto [0,P)")
                # slot_job/slot_w = tp_slots/prered_w 的逆映射 (EPIRED 查表)
                sj = sjob_g.long()
                if int(sj.max()) >= world * T:
                    fails.append(f"[{tag}] slot_job out of range")
                elif not torch.equal(sj[flat], torch.arange(N) // topk):
                    fails.append(f"[{tag}] slot_job not the inverse of tp_slots")
                if not torch.equal(sw_g[flat], w_g.reshape(-1)):
                    fails.append(f"[{tag}] slot_w not the inverse of prered_w")
                real_per_blk = torch.bincount(flat // ROW_BLOCK, minlength=P // ROW_BLOCK)
                if not torch.equal(real_per_blk + slack_g.long(),
                                   torch.full_like(real_per_blk, ROW_BLOCK)):
                    fails.append(f"[{tag}] slack + real != ROW_BLOCK")
                S = world * T
                for name, ordv, key in [("pull_order", pull_g, slots_g.min(dim=1).values),
                                        ("job_order", job_g, slots_g.max(dim=1).values)]:
                    o = ordv.long()
                    if o.unique().numel() != S:
                        fails.append(f"[{tag}] {name} not a permutation")
                    elif not bool((key[o][1:] > key[o][:-1]).all()):
                        fails.append(f"[{tag}] {name} not sorted by its key")
                # push_order: per-source permutation of [0,T), sorted by
                # (min expert, src_tok), and CANONICAL — identical whichever
                # rank builds it and whichever segmentation is in effect (TP-T1;
                # 正确性其实不依赖它 —— push 落点/flag 都由 src_dev*T+src_tok
                # 定死 —— 但"源按目的卡消费序推"是流水启发式, 丢了会掉性能)。
                mins2 = slots_g.min(dim=1).values.view(world, T)
                key2 = all_ids.min(dim=2).values.long()          # (world, T)
                for srow in range(world):
                    o = pusho_g[srow].long()
                    if o.unique().numel() != T:
                        fails.append(f"[{tag}] push_order[{srow}] not a permutation")
                    elif not bool((  # 全序键 = (min expert, src_tok)
                            (key2[srow] * T + torch.arange(T))[o][1:] >
                            (key2[srow] * T + torch.arange(T))[o][:-1]).all()):
                        fails.append(f"[{tag}] push_order[{srow}] not sorted by "
                                     f"(min expert, src_tok)")
                    elif not local_first and not bool(
                            (mins2[srow][o][1:] > mins2[srow][o][:-1]).all()):
                        # canonical 布局下新键必须与旧的 min-slot 序逐位等价
                        fails.append(f"[{tag}] push_order[{srow}] != min-slot order")
                if rank == 0:
                    canon = pusho_g.clone()
                elif not torch.equal(pusho_g, canon):
                    fails.append(f"[{tag}] push_order differs from rank 0 (not canonical)")
    return fails


def check_dataflow(mod, local_first=False, seg_thr5=0):
    """Small-shape fused-dataflow simulation vs naive TP reference (fp32)."""
    torch.manual_seed(0)
    world, T, topk = 4, 96, 8
    H, inter, E = 128, 32, 16
    X = torch.randn(world * T, H, device="cpu")
    ids = torch.stack([torch.randperm(E, device="cpu")[:topk]
                       for _ in range(world * T)]).view(world, T, topk).to(torch.int32)
    w = torch.rand(world, T, topk, device="cpu").float()
    _STATE["ids"], _STATE["w"] = ids, w
    W1 = [torch.randn(E, 2 * inter, H, device="cpu") * 0.05 for _ in range(world)]
    W2 = [torch.randn(E, H, inter, device="cpu") * 0.05 for _ in range(world)]

    idsf, wf = ids.view(-1, topk).long(), w.view(-1, topk)
    ref = torch.zeros(world * T, H, device="cpu")
    for r in range(world):
        for kpos in range(topk):
            e = idsf[:, kpos]
            gate = torch.einsum("nh,nih->ni", X, W1[r][e][:, :inter, :])
            up = torch.einsum("nh,nih->ni", X, W1[r][e][:, inter:, :])
            act = torch.nn.functional.silu(gate) * up
            ref += wf[:, kpos:kpos + 1] * torch.einsum("ni,nhi->nh", act, W2[r][e])

    RB = mod.ROW_BLOCK
    staging = [torch.zeros(world, T, H, device="cpu") for _ in range(world)]
    for r in range(world):
        (_padded, tp_slots, tp_w, _slack, _po, _jo, _pso, blk_expert, _sj, _sw,
         P) = mod._build_tp_schedules(ids[r], w[r], T, world, E, r, "cpu",
                                      local_first=local_first,
                                      seg_thr5=seg_thr5)
        gathered = torch.zeros(P, H, device="cpu")
        gathered[tp_slots.view(-1).long()] = X.repeat_interleave(topk, dim=0)
        expert_out = torch.zeros(P, H, device="cpu")
        # 按行块走(dispenser 的真实语义), 不假设同一 expert 的行块连续 ——
        # local_first 布局下它们分处两段。
        for blk in range(P // RB):
            e = int(blk_expert[blk])
            rows = slice(blk * RB, (blk + 1) * RB)
            gu = gathered[rows] @ W1[r][e].T.float()
            a = torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:]
            expert_out[rows] = a @ W2[r][e].T          # (RB,inter)@(inter,H)
        part = (tp_w.unsqueeze(-1) * expert_out[tp_slots.long()]).sum(dim=1)  # (world*T, H)
        for s in range(world):
            staging[s][r] = part[s * T:(s + 1) * T]
    out = torch.cat([staging[r].sum(dim=0) for r in range(world)])
    rel = (out - ref).norm() / ref.norm()
    tag = f" (local_first thr5={seg_thr5})" if local_first else ""
    return [] if rel < 1e-5 else [f"dataflow{tag} rel_err {float(rel):.2e} >= 1e-5"]


def main():
    torch.distributed.all_gather_into_tensor = _fake_all_gather_into_tensor
    mod = _load_tk_tp_scheme()
    fails = check_builders(mod)
    fails += check_dataflow(mod, local_first=False)
    fails += check_dataflow(mod, local_first=True)      # thr5=0 (选择性分段)
    fails += check_dataflow(mod, local_first=True, seg_thr5=99)   # 全分段
    for f in fails:
        print("  " + f)
    print(f"[preflight_tp_cpu] {'OK' if not fails else f'FAIL ({len(fails)})'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
