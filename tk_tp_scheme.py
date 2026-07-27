# SPDX-License-Identifier: Apache-2.0
"""TK 通算融合 MoE scheme(FP8, TP)—— `--scheme tktp`。

TP 把 intermediate 维切片(每张卡有全部 E 个 expert, 但每个 expert 更"瘦"),
所以跨卡流量是稠密的、与路由无关的, 一层 MoE 只剩两次跨卡搬运, 各自被融进
一个 persistent kernel:

  1. layer0  AllGather ⊕ gate+up GEMM ⊕ SwiGLU   (moe_tp_dispatch_gemm_fp8_push)
       源卡按**消费序**把自己的 fp8 token 行 push 进 3 个 peer 的 staging,
       收卡等 per-token 到达 flag 后从本地 staging 散到该 token 的 TOP_K 个
       gathered slot; 行块计数满就放行对应 GEMM tile。TP 下每个 token 的全部
       top-k expert 都在本卡, 所以"每行只跨卡搬一次"是天然形态。
  2. layer1  W2 GEMM ⊕ 本地 top-k 预归约 ⊕ 稠密 ReduceScatter push
                                                 (moe_tp_gemm_prered_push_fp8)
       TP 下 top-k 加权合并**完全是本地的**, 跨卡步骤退化成 (T, H) 部分和的
       稠密 ReduceScatter, 用边算边推 + 水位信号完成。
  3. moe_final_reduce_push                        源卡把 world 个 partial plane 求和。

所有通信都是 PCIe 安全的(单播 push、本地 red.release.gpu 计数、st.release.sys
定值信号): 本平台既没有远端原子 RMW 也没有 multimem, 见 docs/04。

约束: FP8(w8a8, block [128,128])、TOP_K 必须等于 kernel 编译期的 8。
其它已试过的数据面(peer pull / per-lane pull / 通信 warp 化 / copy engine /
layer1 按 N 维分解)都已判负, 结论留在 docs/04, 代码不再保留。
"""
from __future__ import annotations

import os

import torch

from .config import ParallelMode
from .context import DistContext
from .data import MoEProblem
from .schemes import DistributedScheme

# 每个 GEMM tile 的 token 行数, 同时是 expert 的 padding 单位。必须与
# sm120_common.cuh 的 gemm_config_fp8::ROW_BLOCK 一致(build.py 用同一个值
# 传 -DTK_ROW_BLOCK 并进 .so 文件名)。
ROW_BLOCK = 128


def _build_tp_schedules(topk_ids, topk_weights, num_tokens, world_size,
                        num_experts, rank, device, local_first=False):
    """Host golden TP schedule (setup only, not timed). Produces:
      - padded (num_experts,) int32: per-expert ROW_BLOCK-padded token counts
        over the FULL world*T batch (identical on every rank; under
        local_first = the SUM of the expert's two segments' padded counts);
      - tp_slots (world*T, TOP_K) int32: gathered slot of every assignment,
        row j = src_dev*T + src_tok, column = kpos. This ONE table drives both
        layer0 (dispatch scatters the pulled token row to slots) and layer1
        (prered gathers the same slots' W2 rows) — in TP they are the same map.
      - tp_w (world*T, TOP_K) float32: routing weight per assignment;
      - slack (nblk,) int32: ROW_BLOCK - real tokens per row block (counter seed);
      - pull_order (world*T,) int32: unique tokens sorted by MIN gathered slot
        (expert-major; docs/03 — a ring-by-source order made every row block
        wait for the LAST ring stage, stalling the GEMM behind the whole AG);
      - job_order (world*T,) int32: jobs sorted by MAX slot = readiness order
        for the layer1 dispenser;
      - push_order (world, T) int32: source s's tokens sorted by their MIN slot
        — the order card s PUSHES its shard in layer0, which must be
        byte-identical on every rank (producer and consumers replay it for the
        arrival-flag position mapping);
      - blk_expert (nblk,) int32: row block -> expert id (the dispenser GEMM's
        B-tile index);
      - slot_job (num_padded_total,) int32: gathered row -> dense job id
        (src_dev*T + src_tok), -1 on padding rows — the INVERSE map of
        tp_slots, driving the layer1 EPIRED epilogue's weighted red.add;
      - slot_w (num_padded_total,) float32: routing weight per gathered row,
        0 on padding rows;
      - num_padded_total: gathered rows (= sum(padded)).

    Slot order within an expert is CANONICAL (src_dev, src_tok, kpos) —
    identical on every rank.

    local_first (TK_LOCAL_FIRST, docs/14) splits the layout into TWO segments
    per rank: row blocks [0, E) hold expert e's OWN-rank assignments only, the
    rest hold every other rank's. Segment 1 needs no arrival flag at all
    (scatter_lane's src == dev_idx branch reads pre_tokens directly), and the
    dispenser hands out row blocks in id order, so the GEMM's first
    E*col_blocks tasks are completely independent of the AllGather. The layout
    becomes rank-dependent, which the P2 push data plane allows: push targets
    staging row src_dev*T+src_tok (never a gathered slot) and scatter reads the
    LOCAL tp_slots; push_order stays canonical (docs/14 §3 proof). The generic
    code path below indexes a group table of size G = 2E (local, remote) or
    G = E (canonical), so both layouts share one implementation.
    """
    top_k = topk_ids.shape[1]
    all_topk = torch.empty(world_size, num_tokens, top_k, device=device,
                           dtype=topk_ids.dtype)
    torch.distributed.all_gather_into_tensor(all_topk, topk_ids.contiguous())
    all_w = torch.empty(world_size, num_tokens, top_k, device=device,
                        dtype=torch.float32)
    torch.distributed.all_gather_into_tensor(all_w, topk_weights.float().contiguous())
    all_topk_cpu = all_topk.cpu()
    all_w_cpu = all_w.cpu()

    counts = torch.bincount(all_topk.view(-1), minlength=num_experts).cpu().long()
    # group g = segment*E + expert. canonical: one segment (G=E); local_first:
    # segment 0 = this rank's own assignments, segment 1 = every other rank's.
    if local_first:
        counts_loc = torch.bincount(all_topk[rank].reshape(-1),
                                    minlength=num_experts).cpu().long()
        counts_g = torch.cat([counts_loc, counts - counts_loc])
    else:
        counts_g = counts
    padded_g = (counts_g + ROW_BLOCK - 1) // ROW_BLOCK * ROW_BLOCK
    # per-expert padded rows (the two segments' blocks are NOT adjacent, but no
    # kernel reads this table's values — only its length, = E)
    padded = (padded_g[:num_experts] + padded_g[num_experts:] if local_first
              else padded_g)
    num_padded_total = int(padded_g.sum())

    # NOTE: explicit device="cpu" everywhere — the distributed worker runs under
    # torch.set_default_device(cuda), which would silently move these host-side
    # tables (and turn the fill loop into 16k+ single-element GPU writes).
    write_pos = torch.cat([
        torch.zeros(1, dtype=torch.int64, device="cpu"),
        torch.cumsum(padded_g[:-1], dim=0)
    ]).tolist()
    S = world_size * num_tokens
    tp_slots = torch.full((S, top_k), -1, dtype=torch.int32, device="cpu")
    tp_w = torch.zeros((S, top_k), dtype=torch.float32, device="cpu")
    slot_job = torch.full((num_padded_total,), -1, dtype=torch.int32, device="cpu")
    slot_w = torch.zeros(num_padded_total, dtype=torch.float32, device="cpu")
    for src_dev in range(world_size):  # canonical: src_dev ascending on EVERY rank
        seg = num_experts if (local_first and src_dev != rank) else 0
        for src_tok in range(num_tokens):
            j = src_dev * num_tokens + src_tok
            for kpos, eid in enumerate(all_topk_cpu[src_dev, src_tok].tolist()):
                slot = write_pos[seg + eid]
                write_pos[seg + eid] += 1
                tp_slots[j, kpos] = slot
                wjk = float(all_w_cpu[src_dev, src_tok, kpos])
                tp_w[j, kpos] = wjk
                slot_job[slot] = j
                slot_w[slot] = wjk

    nblk = num_padded_total // ROW_BLOCK
    slack = torch.zeros(nblk, dtype=torch.int32, device="cpu")
    blk = 0
    for g in range(counts_g.shape[0]):
        real_g = int(counts_g[g])
        for b in range(int(padded_g[g]) // ROW_BLOCK):
            real_in = max(0, min(ROW_BLOCK, real_g - b * ROW_BLOCK))
            slack[blk] = ROW_BLOCK - real_in
            blk += 1

    # consumption orders. min/max slots are already unique across tokens (slots
    # are a bijection and each slot belongs to one token), so argsort needs no
    # stability; the +index tie-break just keeps host/GPU byte-identical under
    # any future table change.
    mins = tp_slots.min(dim=1).values.long()
    pull_order = torch.argsort(mins * S + torch.arange(S, device="cpu")).to(torch.int32)
    maxs = tp_slots.max(dim=1).values.long()
    job_order = torch.argsort(maxs * S + torch.arange(S, device="cpu")).to(torch.int32)
    # per-source push order = that source's tokens by min slot, identical on
    # every rank (min slots are unique within a row, argsort stability
    # irrelevant). Canonical: trivially, the layout itself is shared. Under
    # local_first: both segments lay experts out in ascending order, so the key
    # reduces to (min expert, src_tok) for a fixed source either way —
    # rank-independent (docs/14 §3).
    push_order = torch.argsort(mins.view(world_size, num_tokens), dim=1).to(torch.int32)

    # row block -> group -> expert id: first group whose cumulative row-block
    # end exceeds the block index (local_first: group g covers expert g % E).
    rb_end = torch.cumsum(padded_g // ROW_BLOCK, dim=0)
    blk_expert = (torch.searchsorted(
        rb_end, torch.arange(nblk, device="cpu"), right=True)
        % num_experts).to(torch.int32)

    return (padded.to(torch.int32).to(device), tp_slots.to(device),
            tp_w.to(device), slack.to(device), pull_order.to(device),
            job_order.to(device), push_order.to(device), blk_expert.to(device),
            slot_job.to(device), slot_w.to(device), num_padded_total)


def _build_tp_schedules_gpu(packed_all, world_size, num_experts, rank, out,
                            local_first=False):
    """GPU-vectorized rebuild of the TP schedule tables, element-for-element
    identical to the host `_build_tp_schedules` golden (tools/preflight_tp_cpu
    adjudicates). Called each run() and counted in timing — the fair analogue of
    serial's per-run routing all_gathers + moe_align_block_size (docs/05 公平
    口径). Single-argsort trick: slot = rank of the assignment in the total
    order (eid, src_dev, src_tok, kpos), every key unique (bijection over
    flattened all_topk) so stability is not required.

    The input is ONE gathered tensor `packed_all (world, T, TOP_K, 2) int32` —
    [..., 0] = topk ids, [..., 1] = the float32 routing weights' BIT PATTERN
    (view(int32)); one NCCL all_gather carries both. Weights are bit-copied into
    prered_w through its int32 view (reinterpret, not a dtype cast).

    In TP every assignment is local, so there is no trash-row redirect; and the
    layer1 tables (push_expected_l1 / recv_from / final_contrib / prered_dst)
    are routing-INDEPENDENT constants (every card holds all experts), set once
    in setup and not rebuilt here.

    Fixed shapes, no host sync, no bincount — CUDA-graph capturable.
    """
    device = packed_all.device
    T = packed_all.shape[1]
    top_k = packed_all.shape[2]
    N = world_size * T * top_k

    n_idx = torch.arange(N, device=device)
    eid = packed_all[..., 0].reshape(N).long()
    # group g = segment*E + expert. canonical: G = E (one segment). local_first
    # (docs/14): G = 2E, segment 0 = this rank's own assignments (row blocks
    # [0, E) — no arrival flag needed), segment 1 = every other rank's.
    # Order within a group = (src_dev, src_tok, kpos) = flat index n.
    if local_first:
        grp = (n_idx // (T * top_k) != rank).long() * num_experts + eid
        G = 2 * num_experts
    else:
        grp = eid
        G = num_experts
    # key 用 int32(值域 G*N ≤ 512*32768 < 2^31, 排序结果与 int64 逐元素
    # 相同, host/GPU 对拍不变), argsort 提速 ~30-40%。
    key = (grp * N + n_idx).to(torch.int32)
    order = torch.argsort(key)
    grp_s = grp[order]

    # per-group counts + padding (scatter_add: capture-safe, unlike bincount)
    counts = torch.zeros(G, dtype=torch.long, device=device)
    counts.scatter_add_(0, grp, torch.ones_like(grp))
    padded = (counts + ROW_BLOCK - 1) // ROW_BLOCK * ROW_BLOCK
    out["padded"].copy_((padded[:num_experts] + padded[num_experts:]
                         if local_first else padded).to(torch.int32))
    grp_start = torch.zeros(G, dtype=torch.long, device=device)
    grp_start[1:] = torch.cumsum(counts, dim=0)[:-1]
    pos_in_expert = n_idx - grp_start[grp_s]
    padded_base = torch.zeros(G, dtype=torch.long, device=device)
    padded_base[1:] = torch.cumsum(padded, dim=0)[:-1]
    slot_sorted = padded_base[grp_s] + pos_in_expert

    # tp_slots: scatter each sorted assignment's slot back to its original flat
    # index n = (src_dev*T + src_tok)*top_k + kpos == row/col of the table.
    slot_by_n = torch.empty(N, dtype=torch.int32, device=device)
    slot_by_n.scatter_(0, order, slot_sorted.to(torch.int32))
    out["tp_slots"].copy_(slot_by_n.view(world_size * T, top_k))
    # weights: flat n already IS (j, kpos); bit-copy the float32 pattern
    # through prered_w's int32 view.
    out["prered_w"].view(torch.int32).copy_(
        packed_all[..., 1].reshape(world_size * T, top_k))
    # inverse map slot -> (job, w) for the layer1 EPIRED epilogue (weighted
    # red.add needs job id + weight per gathered row). slot_by_n is a
    # bijection onto the valid slots, so padding rows keep the fill value
    # (job=-1 -> epilogue skips; w=0). Rebuilt every run: clear then scatter.
    job_of_n = (torch.arange(N, device=device) // top_k).to(torch.int32)
    out["slot_job"].fill_(-1)
    out["slot_job"].scatter_(0, slot_by_n.long(), job_of_n)
    out["slot_w"].zero_()
    out["slot_w"].scatter_(0, slot_by_n.long(), out["prered_w"].reshape(N))

    # slack: only the tail block of each group carries (padded - real); scatter
    # via a trash slot for empty groups (capture-safe, no boolean compaction).
    nb_total = out["slack"].shape[0]
    tail_blk = (padded_base + padded) // ROW_BLOCK - 1          # (G,)
    slackv = (padded - counts).to(torch.int32)
    idx = torch.where(padded > 0, tail_blk, torch.full_like(tail_blk, nb_total))
    tmp = torch.zeros(nb_total + 1, dtype=torch.int32, device=device)
    tmp.scatter_(0, idx, slackv)
    out["slack"].copy_(tmp[:nb_total])

    # consumption orders (see the host golden for the rationale)
    S = world_size * T
    tpl = out["tp_slots"].long()
    mins = tpl.min(dim=1).values
    out["pull_order"].copy_(
        torch.argsort((mins * S + torch.arange(S, device=device))
                      .to(torch.int32)).to(torch.int32))
    out["job_order"].copy_(
        torch.argsort((tpl.max(dim=1).values * S + torch.arange(S, device=device))
                      .to(torch.int32)).to(torch.int32))
    out["push_order"].copy_(
        torch.argsort(mins.view(world_size, T), dim=1).to(torch.int32))

    # row block -> group -> expert (dispenser GEMM B-tile index). searchsorted
    # keeps shapes fixed (capture-safe), mirrors the host golden formula.
    nblk = out["blk_expert"].shape[0]
    rb_end = torch.cumsum(padded // ROW_BLOCK, dim=0)
    out["blk_expert"].copy_((torch.searchsorted(
        rb_end, torch.arange(nblk, device=device), right=True)
        % num_experts).to(torch.int32))


class TKFusedTP(DistributedScheme):
    """FP8 TP 通算融合层。环境旋钮(默认值即当前最好配置, 见 docs/05):

    - ``TK_COMM_SMS``      layer0 通信块数, 默认 24(拐点实测值)
    - ``TK_L0_PUSH_SMS``   其中做 push 的块数, 其余做本地 scatter, 默认 4
    - ``TK_COMM_SMS_L1``   layer1 通信块数, 默认跟随 ``TK_COMM_SMS``
    - ``TK_GPU_SCHED``     1(默认)= 调度表在 run() 内用 GPU 重建并计入耗时
                           (与 serial 的每次路由 all_gather + align 同口径);
                           0 = 只用 setup 的 host 表, 归因用, **不是公平口径**
    """

    name = "tktp"

    def setup(self, problem: MoEProblem, ctx: DistContext) -> None:
        cfg = problem.config
        assert cfg.parallel_mode == ParallelMode.TP, "TKFusedTP is TP-only"
        assert cfg.topk == 8, "kernels compile TOP_K=8"
        assert problem.quant_config is not None, (
            "TKFusedTP 只有 FP8 路径(主配置 configs/tp_rtx_pro5000_4gpu_fp8.yaml);"
            " BF16 的融合 kernel 已在 slim 分支移除, 需要时检出历史提交")
        assert cfg.block_shape == [128, 128], "fp8 kernels assume [128,128] blocks"
        self.ctx = ctx
        self.problem = problem
        H = cfg.hidden_size
        inter = cfg.intermediate_shard          # I / world (TP shard)
        world = ctx.world_size
        num_tokens = problem.num_tokens         # per-rank T
        num_experts = cfg.num_experts
        device = problem.hidden_states.device
        self.H, self.inter, self.num_tokens, self.top_k = H, inter, num_tokens, cfg.topk
        # GEMM template constraints (sm120_common.cuh)
        assert H % 128 == 0, "hidden must be tile-aligned"
        assert (2 * inter) % 64 == 0, "gate_up shard % COL_BLOCK(64)"
        assert inter % 128 == 0, "intermediate shard % RED_BLOCK(128)"

        from importlib.util import spec_from_file_location, module_from_spec
        _build_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "kernels", "tk", "build.py")
        _spec = spec_from_file_location("_tk_build", _build_py)
        _bmod = module_from_spec(_spec)
        _spec.loader.exec_module(_bmod)
        self.tk = _bmod.build_and_load(world, hidden=H, row_block=ROW_BLOCK)

        # 本地优先分段(TK_LOCAL_FIRST, docs/14): gathered 行块 [0, E) 只装本
        # rank 自己的 assignment —— 这些行 scatter 走 src==dev_idx 直读
        # pre_tokens, 不等任何到达 flag, 而 dispenser 按行块 id 升序发任务,
        # 于是 GEMM 的头 E*col_blocks 个任务完全不依赖 AllGather。代价是两段
        # 各自补齐带来的 padding(balanced T=512 是最坏点 +10% GEMM/+50% 行数;
        # T=1024 每 expert 每 rank 恰好 128 行 → 零代价), 账见 docs/14 §4。
        self.local_first = int(os.environ.get("TK_LOCAL_FIRST", "0"))

        # ---- host golden schedule (not timed) ----
        (padded, tp_slots, tp_w, slack, pull_order, job_order, push_order,
         blk_expert, slot_job, slot_w, num_padded_total) = _build_tp_schedules(
            problem.topk_ids, problem.topk_weights, num_tokens, world,
            num_experts, ctx.rank, device, local_first=bool(self.local_first))
        self.padded = padded
        self.tp_slots = tp_slots.contiguous()
        self.prered_w = tp_w.contiguous()
        self.slack = slack.contiguous()
        self.pull_order = pull_order.contiguous()
        self.job_order = job_order.contiguous()
        self.push_order = push_order.contiguous()
        self.blk_expert = blk_expert.contiguous()
        self.slot_job = slot_job.contiguous()
        self.slot_w = slot_w.contiguous()
        self.num_padded_total = num_padded_total
        self.num_jobs = world * num_tokens
        # 每迭代清零的计数器(同 stream, 无需额外同步)
        self.job_next = torch.zeros(1, dtype=torch.int32, device=device)     # L1 job dispenser
        self.gemm_next = torch.zeros(1, dtype=torch.int32, device=device)    # L0 GEMM task dispenser
        self.l1_gemm_next = torch.zeros(1, dtype=torch.int32, device=device)  # L1 GEMM task dispenser
        self.push_next = torch.zeros(1, dtype=torch.int32, device=device)    # L0 push 领取
        self.pull_next = torch.zeros(1, dtype=torch.int32, device=device)    # L0 scatter 领取

        # ---- layer1 constants (routing-independent in TP) ----
        # prered_dst: job j -> (src_dev, src_tok) of the dense job space.
        nj = self.num_jobs
        self.prered_dst = torch.empty(nj, 2, dtype=torch.int32, device=device)
        self.prered_dst[:, 0] = (torch.arange(nj, device=device) // num_tokens).to(torch.int32)
        self.prered_dst[:, 1] = (torch.arange(nj, device=device) % num_tokens).to(torch.int32)
        # every card holds all experts -> every job non-empty, every card
        # contributes to every token: dense expected counts / contrib masks.
        self.push_expected_l1 = torch.full((world,), num_tokens,
                                           dtype=torch.int32, device=device)
        self.recv_from = torch.ones(world, dtype=torch.int32, device=device)
        self.final_contrib = torch.ones(num_tokens, world,
                                        dtype=torch.int32, device=device)
        self.combine_local_cnt = torch.zeros(world, dtype=torch.int32, device=device)

        # ---- GPU schedule rebuild (fairness: counted in run()) ----
        self.gpu_schedule = os.environ.get("TK_GPU_SCHED", "1") == "1"
        if self.gpu_schedule:
            # packed[..., 0] = topk ids, packed[..., 1] = float32 bit pattern —
            # ids + weights ride ONE all_gather.
            self._packed_local = torch.empty(num_tokens, self.top_k, 2,
                                             device=device, dtype=torch.int32)
            self._packed_all = torch.empty(world, num_tokens, self.top_k, 2,
                                           device=device, dtype=torch.int32)
            self._topk_ids_local = problem.topk_ids.contiguous()
            self._topk_w_local = problem.topk_weights.float().contiguous()
            self._topk_w_bits = self._topk_w_local.view(torch.int32)
            self._num_experts = num_experts
            self._sched_out = {
                "padded": self.padded, "tp_slots": self.tp_slots,
                "prered_w": self.prered_w, "slack": self.slack,
                "pull_order": self.pull_order,
                "job_order": self.job_order,
                "push_order": self.push_order,
                "blk_expert": self.blk_expert,
                "slot_job": self.slot_job, "slot_w": self.slot_w,
            }
            # 融合 kernel(tk.tp_sched_build, 单 block 1024 线程)。
            # smem 需求随 P 变(scan_a 4P + misc/seg4 + 小表), 超 99KB
            # 自动回退 torch 版; TK_SCHED_FUSED=0 强制回退(A/B)。
            _smem_need = (4 * num_padded_total
                          + 4 * max(world * num_tokens, 1024 * world)
                          + 264 * 4 * 3 + 1024 * 4 + 512)
            # 融合 sched kernel 只实现了 canonical 单段布局(tpsched 的
            # per-expert compaction), local_first 下回退 torch 向量化版
            # —— sched 阶段会慢 ~120µs, A/B 必须用 time_tp_stages 的分阶段
            # 数字裁决 L0, e2e 只在 local_first 定案后再补融合版(docs/14)。
            self._sched_fused = (os.environ.get("TK_SCHED_FUSED", "1") == "1"
                                 and _smem_need <= 101376
                                 and not self.local_first)
            self._sched_graph = None  # captured lazily on first run()

        # ---- weights ----
        # w1: fp8 权重先反量化(setup 一次), 按 GLU 列交织后重量化 —— scale 块
        # 与 GEMM 的 B tile 天然对齐; w2 直接用原布局(见下)。
        qc = problem.quant_config
        FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
        # w2 (E, H, inter) 就是 B^T 布局, qc.w2_scale (E, H/128, inter/128)
        # 原样可用 —— 零转置、零重量化、零二次量化误差。
        self.w2_fp8 = problem.w2.contiguous()
        self.w2_scales = qc.w2_scale.float().contiguous()
        s1 = qc.w1_scale.float()                         # (E, 2I/128, H/128)
        w1_bf = (problem.w1.float() * s1.repeat_interleave(128, 1)
                                        .repeat_interleave(128, 2))
        # GLU 列交织(按 N 行, 单位 32): [gate32 | up32] per 64-N block
        # (fp8 GEMM COL_BLOCK=64, docs/10; 同一 128 列 scale 块内置换)
        E = w1_bf.shape[0]
        gate = w1_bf[:, :inter].view(E, inter // 32, 32, H)
        up = w1_bf[:, inter:].view(E, inter // 32, 32, H)
        w1_il = torch.stack([gate, up], dim=2).view(E, 2 * inter, H)
        # 交织后 128(N)×128(K) 重量化 -> scale 块与 B tile 天然对齐
        v = w1_il.view(E, 2 * inter // 128, 128, H // 128, 128)
        amax = v.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-8)
        self.w1_il_scales = (amax / FP8_MAX).view(
            E, 2 * inter // 128, H // 128).contiguous()
        self.w_gateup_fp8 = (v / (amax / FP8_MAX)).to(torch.float8_e4m3fn) \
                                                  .view(E, 2 * inter, H).contiguous()
        del w1_bf, w1_il, v

        # ---- buffers ----
        TK = self.tk.TKParallelTensor
        lr, lws = ctx.local_rank, world
        P = num_padded_total
        S_ag = world * num_tokens
        # 源端量化后的 token 分片: 4KB/token + 128B scales(fp8 AG 字节减半)
        self.pre_tokens = TK((num_tokens, H), dtype=torch.float8_e4m3fn,
                             local_rank=lr, local_world_size=lws, multicast=False)
        self.pre_scales = TK((num_tokens, H // 128), dtype=torch.float32,
                             local_rank=lr, local_world_size=lws, multicast=False)
        # push 落点: plane s 只由源卡 s 写(单写者, 无需原子); flags 值 = 到达
        # seq(单调, 免清零)。staging 的覆写安全由 run() 里的双 barrier 保证。
        self.ag_staging_fp8 = TK((S_ag, H), dtype=torch.float8_e4m3fn,
                                 local_rank=lr, local_world_size=lws,
                                 multicast=False)
        self.ag_sscales = TK((S_ag, H // 128), dtype=torch.float32,
                             local_rank=lr, local_world_size=lws,
                             multicast=False)
        self.ag_flags = TK((1, S_ag), dtype=torch.int,
                           local_rank=lr, local_world_size=lws, multicast=False)
        self.ag_flags.data_.zero_()

        bar_cols = max(P // ROW_BLOCK + 1, 32)
        # barrier_l0: row 0 = dispatch row-block counters (slack-seeded), row 1 =
        # pcie_barrier_all slots. barrier_l1: row 0 = W2 col-block counters,
        # row 1 = local W2 completion signal, rows 2+d = card d's combine watermark.
        self.barrier_l0 = TK((2 + world, bar_cols), dtype=torch.int, local_rank=lr,
                             local_world_size=lws, multicast=False)
        self.barrier_l1 = TK((2 + world, bar_cols), dtype=torch.int, local_rank=lr,
                             local_world_size=lws, multicast=False)
        self.barrier_l0.data_.zero_()
        self.barrier_l1.data_.zero_()
        # seed dispatch counters with padding slack (restored by the kernel's
        # reset-to-slack after each iteration; padding slots are never scattered).
        nblk = P // ROW_BLOCK
        self.barrier_l0.data_[0, :nblk].copy_(self.slack)

        # LOCAL workspaces (TP: peers never touch gathered / expert_out).
        # padding 行 scale=0 -> 反量化恒 0, GEMM 对 padding 行的结果是干净的 0。
        self.gathered = torch.zeros(P, H, device=device, dtype=torch.float8_e4m3fn)
        self.gathered_scales = torch.zeros(P, H // 128, device=device,
                                           dtype=torch.float32)
        self.act = torch.zeros(P, inter, device=device, dtype=torch.bfloat16)
        self.act_fp8 = torch.zeros(P, inter, device=device,
                                   dtype=torch.float8_e4m3fn)
        self.act_scales = torch.zeros(P, inter // 128, device=device,
                                      dtype=torch.float32)
        self.expert_out = torch.zeros(P, H, device=device, dtype=torch.bfloat16)
        # EPIRED(TK_L1_EPIRED, 默认关 —— 2026-07-26 A/B 判负, docs/04):
        # W2 GEMM 的 epilogue 把 C tile 乘 w 直接 red.add 进这张 fp32 部分和,
        # expert_out 不落地、push_job 不重读 8 行。数值正确(rel_err 同基线),
        # 但标量 red +193us / v2 向量 red +108us: 本机 L2 fp32 原子吞吐打不进
        # per-task epilogue 关键路径。代码保留供复现, 默认走原路径。
        self.combine_partial = torch.zeros(self.num_jobs, H, device=device,
                                           dtype=torch.float32)
        self.l1_epired = int(os.environ.get("TK_L1_EPIRED", "0"))
        # 两级 tile(TK_TWO_LEVEL, 默认开, docs/09 §4): slack>=64 的尾块只算
        # 前 64 行。slack 表与调度表同源(host golden/tpsched 逐比特对拍),
        # balanced 下无尾块 → 满块路径逐指令不变, 零开销。EPIRED 的 wred
        # tail 是 trap 桩, 互斥(L1 侧自动降级为关)。
        self.two_level = int(os.environ.get("TK_TWO_LEVEL", "1"))
        # 归因探针(TK_L0_NOGATE, docs/14 §5): L0 GEMM 不等行块到达计数 ——
        # **输出数值是错的**, 只用于把 gate 等待从 L0 exposure 里拆出来。
        self.l0_no_gate = int(os.environ.get("TK_L0_NOGATE", "0"))
        # peer-writable combine staging: plane d (rows [d*T, d*T+T)) is written
        # only by card d (single writer, no atomics).
        self.combine_staging = TK((world * num_tokens, H), dtype=torch.bfloat16,
                                  local_rank=lr, local_world_size=lws, multicast=False)
        self.combine_staging.data_.zero_()
        self.combine_out = torch.zeros(num_tokens, H, device=device, dtype=torch.bfloat16)

        # comm SM 预算(docs/03 的 sweep 结论): L0 拐点 24 —— push 本身 2~4 个
        # 块就饱和, 但收侧 scatter 还需要并发; comm 块推完会转岗领 GEMM task,
        # 所以给多了也不是纯浪费。L1 单独给预算(它的通信块不会在 GEMM 结束前
        # 帮上忙, 可以更小)。
        self.num_comm_sms = int(os.environ.get("TK_COMM_SMS", "24"))
        self.l0_push_sms = int(os.environ.get("TK_L0_PUSH_SMS", "4"))
        self.num_comm_sms_l1 = int(os.environ.get("TK_COMM_SMS_L1",
                                                  str(self.num_comm_sms)))
        self._l0_seq = 0
        self._l1_seq = 0

    def _sched_torch_call(self):
        """torch 向量化调度表构建(preflight 已裁决其与 host golden 一致)。"""
        _build_tp_schedules_gpu(self._packed_all, self.ctx.world_size,
                                self._num_experts, self.ctx.rank,
                                self._sched_out,
                                local_first=bool(self.local_first))

    def _sched_fused_call(self):
        """单 kernel 调度表构建(与 _build_tp_schedules_gpu 逐元素一致,
        首跑对拍)。canonical 布局专用 —— local_first 下 setup 已关掉它。"""
        self.tk.tp_sched_build(
            self._packed_all, self.padded, self.tp_slots, self.prered_w,
            self.slack, self.pull_order, self.job_order, self.push_order,
            self.blk_expert, self.slot_job, self.slot_w,
            self.ctx.world_size, self.num_tokens, self._num_experts,
            self.num_padded_total)

    def run(self) -> torch.Tensor:
        tk = self.tk
        # 公平口径: 调度表在计时区内用 GPU 重建(serial 每次也要付路由
        # all_gather + moe_align_block_size)。表的形状/地址是固定的(路由在一个
        # problem 内不变), 纯计算部分首次用后被 CUDA graph 捕获。
        if self.gpu_schedule:
            # ids + weight bits 打包进一次 all_gather(省一次 NCCL 启动 ~40µs;
            # 打包 copy 是 ~5µs 的设备 kernel, 按公平口径留在计时区内)。
            self._packed_local[..., 0].copy_(self._topk_ids_local)
            self._packed_local[..., 1].copy_(self._topk_w_bits)
            torch.distributed.all_gather_into_tensor(
                self._packed_all.view(-1), self._packed_local.view(-1))
            if self._sched_graph is None:
                if self._sched_fused:
                    # 首跑对拍(此时 packed_all 已有真实路由): torch 向量化版
                    # (基准, preflight 已裁决其与 golden 一致) vs 融合 kernel,
                    # 逐表 torch.equal, 不过直接 raise, 不静默退化。
                    self._sched_torch_call()
                    snap = {k: v.clone() for k, v in self._sched_out.items()}
                    self._sched_fused_call()
                    torch.cuda.synchronize()
                    bad = [k for k in snap
                           if not torch.equal(snap[k], self._sched_out[k])]
                    assert not bad, f"tp_sched_build vs torch builder mismatch: {bad}"
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        self._sched_fused_call()
                else:
                    self._sched_torch_call()                       # warmup
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        self._sched_torch_call()
                self._sched_graph = g
            else:
                self._sched_graph.replay()

        # barrier BEFORE overwriting pre_tokens (no peer still reading last
        # iter's tokens) and AFTER (my tokens visible before peers are pushed).
        self._l0_seq += 1
        tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)
        # 源端 1×128 group 量化(单 kernel, torch 链 ~80µs → ~15µs)
        tk.rowgroup_quant_fp8(self.problem.hidden_states,
                              self.pre_tokens.data_, self.pre_scales.data_)
        self._l0_seq += 1
        tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)

        # layer0: fp8 AllGather(push)⊕ dispenser GEMM ⊕ SwiGLU epilogue。
        # seq 用 _l0_seq 当前值(每迭代单调 +2), 到达 flag 因此免清零。
        self.gemm_next.zero_()
        self.push_next.zero_()
        self.pull_next.zero_()
        tk.moe_tp_dispatch_gemm_fp8_push(
            self.pre_tokens, self.pre_scales, self.ag_staging_fp8,
            self.ag_sscales, self.ag_flags, self.gathered,
            self.gathered_scales, self.w_gateup_fp8, self.w1_il_scales,
            self.act, self.padded, self.tp_slots, self.slack,
            self.pull_order, self.push_order, self.blk_expert,
            self.gemm_next, self.push_next, self.pull_next,
            self.barrier_l0, self.num_comm_sms, self.l0_push_sms,
            self.num_padded_total, self.num_tokens, self._l0_seq,
            self.two_level, self.l0_no_gate)

        # layer1: act 量化 -> W2 GEMM ⊕ 本地预归约 ⊕ push(稠密 ReduceScatter),
        # 然后源卡侧对 world 个 partial plane 做最终归约。
        self._l1_seq += 1
        self.combine_local_cnt.zero_()
        self.job_next.zero_()
        self.l1_gemm_next.zero_()
        tk.rowgroup_quant_fp8(self.act, self.act_fp8, self.act_scales)
        tk.moe_tp_gemm_prered_push_fp8(
            self.act_fp8, self.act_scales, self.w2_fp8, self.w2_scales,
            self.expert_out, self.padded,
            self.combine_partial, self.slot_job, self.slot_w,
            self.combine_staging, self.prered_dst, self.tp_slots,
            self.prered_w, self.combine_local_cnt, self.push_expected_l1,
            self.blk_expert, self.l1_gemm_next, self.job_order,
            self.job_next, self.barrier_l1, self.num_comm_sms_l1,
            self.num_padded_total, self.num_tokens, self.num_jobs,
            self._l1_seq, self.l1_epired, self.slack,
            0 if self.l1_epired else self.two_level)
        tk.moe_final_reduce_push(self.combine_staging, self.final_contrib,
                                 self.recv_from, self.combine_out, self.barrier_l1,
                                 self.num_tokens, self._l1_seq)
        return self.combine_out
