# SPDX-License-Identifier: Apache-2.0
"""TP CPU preflight — runs ANYWHERE (no GPU, no vllm, no torch.distributed):

  1. DEVICE GUARD (docs/21 §1): every tensor creation inside the TP schedule
     builders must carry an explicit device= — anything else is the
     set_default_device(cuda) bug class that killed rounds 1 and 3 on the
     machine. Enforced with a TorchFunctionMode, not grep discipline.
  2. host golden vs GPU-vectorized builder: element-for-element equality plus
     the slot/slack/order invariants (subset of tools/verify_tp_schedule.py).
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
        ("tk_scheme", {"ROW_BLOCK": 128}),
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
    """Fail any tensor creation without device= (docs/21 §1 bug class)."""

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
    for dist_kind in ("balanced", "skewed"):
        for ne in (64, 256):
            for rank in (0, 3):
                all_ids, all_w = _make_topk(world, T, ne, topk, dist_kind, 0)
                _STATE["ids"], _STATE["w"] = all_ids, all_w
                with RequireExplicitDevice():
                    (padded_g, slots_g, w_g, slack_g, pull_g, job_g, pusho_g,
                     P) = mod._build_tp_schedules(all_ids[rank], all_w[rank],
                                                  T, world, ne, rank, "cpu")
                N = world * T * topk
                ar = torch.arange(N, device="cpu")
                out = {
                    "src_dev_grid": ar // (T * topk),
                    "src_tok_grid": (ar // topk) % T,
                    "kpos_grid": ar % topk,
                    "padded": torch.zeros(ne, dtype=torch.int32, device="cpu"),
                    "tp_slots": torch.full((world * T, topk), -1, dtype=torch.int32, device="cpu"),
                    "prered_w": torch.zeros(world * T, topk, dtype=torch.float32, device="cpu"),
                    "slack": torch.zeros(P // ROW_BLOCK, dtype=torch.int32, device="cpu"),
                    "pull_order": torch.zeros(world * T, dtype=torch.int32, device="cpu"),
                    "job_order": torch.zeros(world * T, dtype=torch.int32, device="cpu"),
                    "push_order": torch.zeros(world, T, dtype=torch.int32, device="cpu"),
                }
                with RequireExplicitDevice():
                    mod._build_tp_schedules_gpu(all_ids, all_w, world, ne, rank, out)
                tag = f"NE={ne} {dist_kind} rank={rank}"
                for name, got, ref in [("padded", out["padded"], padded_g),
                                       ("tp_slots", out["tp_slots"], slots_g),
                                       ("prered_w", out["prered_w"], w_g),
                                       ("slack", out["slack"], slack_g),
                                       ("pull_order", out["pull_order"], pull_g),
                                       ("job_order", out["job_order"], job_g),
                                       ("push_order", out["push_order"], pusho_g)]:
                    if got.shape != ref.shape or not torch.equal(got, ref):
                        fails.append(f"[{tag}] {name} host/GPU MISMATCH")
                flat = slots_g.reshape(-1).long()
                if flat.unique().numel() != N or int(flat.min()) < 0 or int(flat.max()) >= P:
                    fails.append(f"[{tag}] tp_slots not a bijection onto [0,P)")
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
                # push_order: per-source permutation of [0,T), sorted by min slot,
                # and CANONICAL — identical whichever rank builds it (TP-T1).
                mins2 = slots_g.min(dim=1).values.view(world, T)
                for srow in range(world):
                    o = pusho_g[srow].long()
                    if o.unique().numel() != T:
                        fails.append(f"[{tag}] push_order[{srow}] not a permutation")
                    elif not bool((mins2[srow][o][1:] > mins2[srow][o][:-1]).all()):
                        fails.append(f"[{tag}] push_order[{srow}] not sorted")
                if rank == 0:
                    canon = pusho_g.clone()
                elif not torch.equal(pusho_g, canon):
                    fails.append(f"[{tag}] push_order differs from rank 0 (not canonical)")
    return fails


def check_dataflow(mod):
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

    staging = [torch.zeros(world, T, H, device="cpu") for _ in range(world)]
    for r in range(world):
        (padded, tp_slots, tp_w, _slack, _po, _jo, _pso, P) = mod._build_tp_schedules(
            ids[r], w[r], T, world, E, r, "cpu")
        gathered = torch.zeros(P, H, device="cpu")
        gathered[tp_slots.view(-1).long()] = X.repeat_interleave(topk, dim=0)
        expert_out = torch.zeros(P, H, device="cpu")
        base = 0
        for e in range(E):
            pe = int(padded[e])
            gu = gathered[base:base + pe] @ W1[r][e].T.float()
            a = torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:]
            expert_out[base:base + pe] = a @ W2[r][e].T  # (pe,inter)@(inter,H)
            base += pe
        part = (tp_w.unsqueeze(-1) * expert_out[tp_slots.long()]).sum(dim=1)  # (world*T, H)
        for s in range(world):
            staging[s][r] = part[s * T:(s + 1) * T]
    out = torch.cat([staging[r].sum(dim=0) for r in range(world)])
    rel = (out - ref).norm() / ref.norm()
    return [] if rel < 1e-5 else [f"dataflow rel_err {float(rel):.2e} >= 1e-5"]


def main():
    torch.distributed.all_gather_into_tensor = _fake_all_gather_into_tensor
    mod = _load_tk_tp_scheme()
    fails = check_builders(mod)
    fails += check_dataflow(mod)
    for f in fails:
        print("  " + f)
    print(f"[preflight_tp_cpu] {'OK' if not fails else f'FAIL ({len(fails)})'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
