# SPDX-License-Identifier: Apache-2.0
"""TD 风格 TP 复刻 scheme(FP8)—— `--scheme tktd`(docs/17)。

用 TK/PK 原语按 Triton-distributed tp_moe 的通算融合逻辑重组数据流, 与 tktp
只差"调度策略层"(数据面原语同源), 用来把 tdtp vs tktp 的差距拆成"引擎差异"
和"调度差异"两部分:

  1. layer0  独立 producer kernel(独立 stream)推 fp8 token 分片(朴素 token
     序, per-src-rank 段到达 flag = TD 的 per-segment barrier), 散完一个 src
     的整个分片才发本地 ready flag; GEMM kernel 只算不通信, 行块 gate 在其
     真实行覆盖的 src 段区间上(TD 的 dl.wait(segment_start..end), 换有界
     自旋), 发放序按"最晚就绪段"排(TD 的 threadblock swizzle, 本 rank tile
     先算)。                          (moe_td_ag_producer + moe_td_ag_gemm)
  2. layer1  W2 GEMM 按 N-chunk-major 任务序推进, 每 chunk 发本地完成信号
     (TD 的 gemm_done_flag); 独立 stream 的 reduce-RS kernel 逐 chunk 追赶:
     本地 top-k 加权归约该列窗 -> 向量 st 直推源卡 staging(TD 的
     reduce_topk + RS on reduce_stream)。
                                     (moe_td_gemm_nchunk + moe_td_reduce_rs)
  3. moe_final_reduce_push           复用 tktp 的源卡最终归约(等 watermark)。

已知代价(docs/04 已判负形态的刻意复刻, A/B 归因用): L0 粗粒度段 gate 让
大部分 tile 等整个 AllGather(docs/03 ring-by-source 教训); L1 N 维分解 =
Comet-N; comm SM 不转岗。见 docs/17 的账。

环境旋钮(默认值见 docs/17):
  - ``TKTD_COMM_SMS``   layer0 producer kernel 的块数, 默认 24
  - ``TKTD_PUSH_SMS``   其中 push 块数(其余 scatter), 默认 4
  - ``TKTD_RS_SMS``     layer1 reduce-RS kernel 的块数, 默认 24
  - ``TKTD_NCHUNKS``    layer1 N-chunk 数(须整除 H/64), 默认 4(TD 同款)
  - ``TKTD_L0_NOGATE``  归因探针: L0 GEMM 跳过段等待, **数值是错的**
  - ``TKTD_TWO_LEVEL``  两级尾块(docs/09 §4), 默认 1
  - ``TKTD_GPU_SCHED``  1(默认)= 调度表在 run() 内 GPU 重建并计入耗时
                        (公平口径, 同 tktp); 0 = 只用 setup 的 host 表
"""
from __future__ import annotations

import os

import torch

from .config import ParallelMode
from .context import DistContext
from .data import MoEProblem
from .schemes import DistributedScheme
from .tk_tp_scheme import ROW_BLOCK, _build_tp_schedules, _build_tp_schedules_gpu


def _td_tables_golden(slot_job_cpu, num_padded_total, num_tokens, world, rank):
    """Host golden(纯 python 循环)的 TD 专用表, 与 GPU 向量化版逐元素对拍:
      - blk_lo/blk_hi (nblk,): 行块真实行覆盖的 src rank 区间(canonical 布局
        下 expert 内按 (src_dev, src_tok) 升序 → 区间连续);
      - row_perm (nblk,): L0 发放序 = 按 (最晚就绪段, 块 id) 排, 段 = 区间内
        ring 距离 (d - rank) % world 的最大值(TD threadblock swizzle 语义:
        本 rank 数据的 tile 最先, 需要全 AG 的最后);
      - pull_order (S,): scatter 领取序 = (ring_dist(src), token id) 升序
        (本 rank 分片免等待, 先散)。
    """
    # NOTE: 显式 device="cpu" —— worker 在 set_default_device(cuda) 下运行,
    # 无显式 device 的创建会落 GPU(docs/04 的坑类; preflight 的 DEVICE GUARD
    # 同样裁决本函数)。
    RB = ROW_BLOCK
    nblk = num_padded_total // RB
    T = num_tokens
    sj = slot_job_cpu.tolist()
    blk_lo = torch.empty(nblk, dtype=torch.int32, device="cpu")
    blk_hi = torch.empty(nblk, dtype=torch.int32, device="cpu")
    stage = [0] * nblk
    dist = [(d - rank) % world for d in range(world)]
    for b in range(nblk):
        srcs = [sj[s] // T for s in range(b * RB, (b + 1) * RB) if sj[s] >= 0]
        lo, hi = min(srcs), max(srcs)
        blk_lo[b], blk_hi[b] = lo, hi
        stage[b] = max(dist[d] for d in range(lo, hi + 1))
    row_perm = torch.tensor(
        sorted(range(nblk), key=lambda b: (stage[b], b)),
        dtype=torch.int32, device="cpu")
    S = world * T
    pull_order = torch.tensor(
        sorted(range(S), key=lambda j: (dist[j // T], j)),
        dtype=torch.int32, device="cpu")
    return blk_lo, blk_hi, row_perm, pull_order


def _build_td_tables_gpu(slot_job, num_tokens, world, rank, out):
    """GPU 向量化版(固定形状、无 host sync, CUDA-graph 可捕获), 与 golden
    逐元素一致(setup 首建即对拍, 不过直接 raise)。输入 slot_job 是共享调度
    builder 的产物(每 run 重建), 因此本函数必须排在它之后同图执行。"""
    device = slot_job.device
    RB = ROW_BLOCK
    nblk = out["blk_lo"].shape[0]
    sj = slot_job[:nblk * RB].view(nblk, RB).long()
    src = torch.div(sj, num_tokens, rounding_mode="floor")   # padding(-1) -> -1
    real = sj >= 0
    lo = torch.where(real, src, torch.full_like(src, world)).amin(1)
    hi = torch.where(real, src, torch.full_like(src, -1)).amax(1)
    out["blk_lo"].copy_(lo.to(torch.int32))
    out["blk_hi"].copy_(hi.to(torch.int32))
    dists = (torch.arange(world, device=device) - rank) % world
    dr = torch.arange(world, device=device).unsqueeze(0)
    in_range = (dr >= lo.unsqueeze(1)) & (dr <= hi.unsqueeze(1))
    stage = torch.where(in_range, dists.unsqueeze(0),
                        dists.new_full((), -1)).amax(1)
    # 键值域 4*nblk / 4*S, int32 充裕; 键唯一(+arange tie-break) → argsort
    # 无需稳定性。
    out["row_perm"].copy_(torch.argsort(
        (stage * nblk + torch.arange(nblk, device=device)).to(torch.int32)
    ).to(torch.int32))
    S = world * num_tokens
    jj = torch.arange(S, device=device)
    out["pull_order"].copy_(torch.argsort(
        (dists[jj // num_tokens] * S + jj).to(torch.int32)).to(torch.int32))


class TKTDFusedTP(DistributedScheme):
    """TD 风格 TP 复刻层(FP8, TP)。约束与 tktp 相同: TOP_K=8, FP8 w8a8
    block [128,128], TP-only。"""

    name = "tktd"

    def setup(self, problem: MoEProblem, ctx: DistContext) -> None:
        cfg = problem.config
        assert cfg.parallel_mode == ParallelMode.TP, "TKTDFusedTP is TP-only"
        assert cfg.topk == 8, "kernels compile TOP_K=8"
        assert problem.quant_config is not None, "TKTDFusedTP 只有 FP8 路径"
        assert cfg.block_shape == [128, 128], "fp8 kernels assume [128,128] blocks"
        self.ctx = ctx
        self.problem = problem
        H = cfg.hidden_size
        inter = cfg.intermediate_shard          # I / world (TP shard)
        world = ctx.world_size
        num_tokens = problem.num_tokens         # per-rank T
        num_experts = cfg.num_experts
        device = problem.hidden_states.device
        self.H, self.inter, self.num_tokens = H, inter, num_tokens
        self.top_k = cfg.topk
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

        # ---- 旋钮 ----
        self.num_comm_sms = int(os.environ.get("TKTD_COMM_SMS", "24"))
        self.push_sms = int(os.environ.get("TKTD_PUSH_SMS", "4"))
        self.rs_sms = int(os.environ.get("TKTD_RS_SMS", "24"))
        self.n_chunks = int(os.environ.get("TKTD_NCHUNKS", "4"))
        self.l0_no_gate = int(os.environ.get("TKTD_L0_NOGATE", "0"))
        self.two_level = int(os.environ.get("TKTD_TWO_LEVEL", "1"))
        # 排障相位定位(TKTD_SYNC_DEBUG=1): 每个相位后全设备 synchronize 并
        # 打印, 挂死/трap 会在**出事相位**的 sync 处抛出 —— 看各 rank 最后
        # 一个 "ok" 标签即可定位。相位边界都是全 rank 同构点(所有 rank 都
        # 先发完本相位的全部 kernel 再 sync), 不会引入跨相位依赖倒挂。
        # 只用于排障, 会破坏 overlap, 不是计时口径。
        self.sync_debug = int(os.environ.get("TKTD_SYNC_DEBUG", "0"))
        # L1 kernel 级二分(TKTD_L1_SERIAL):
        #   1 = 三 kernel 全串行逐个 sync(2026-07-28 实测**整门通过**且
        #       rel_err 与 tktp 逐位一致 → 三 kernel 单独均正确, bug 在
        #       并发共驻交互);
        #   2 = RS∥nchunk 真实 overlap, sync, 再 final 串行 —— 裁决
        #       "RS 追赶 GEMM"这一半;
        #   3 = nchunk 先行 sync, 再 RS∥final 并发 —— 裁决"final 与 RS
        #       共驻"这一半。只用于排障。
        self.l1_serial = int(os.environ.get("TKTD_L1_SERIAL", "0"))
        col_blocks = H // 64
        assert col_blocks % self.n_chunks == 0, \
            f"TKTD_NCHUNKS 须整除 H/64={col_blocks}"

        # ---- host golden schedule(setup, 不计时): 复用 tktp 的共享 builder
        # (canonical 单段布局, local_first/l1_seg 均关) ----
        (padded, tp_slots, tp_w, slack, _pull_canon, job_order, _push_order,
         blk_expert, slot_job, slot_w, _row_perm_l1,
         num_padded_total) = _build_tp_schedules(
            problem.topk_ids, problem.topk_weights, num_tokens, world,
            num_experts, ctx.rank, device, local_first=False, l1_seg=0)
        self.padded = padded
        self.tp_slots = tp_slots.contiguous()
        self.prered_w = tp_w.contiguous()
        self.slack = slack.contiguous()
        self.blk_expert = blk_expert.contiguous()
        self.slot_job = slot_job.contiguous()
        self.slot_w = slot_w.contiguous()
        self.job_order = job_order.contiguous()   # 共享 builder 输出, tktd 不消费
        self.num_padded_total = num_padded_total
        self.num_jobs = world * num_tokens
        nblk = num_padded_total // ROW_BLOCK

        # ---- TD 专用表: host golden + GPU 向量化版逐元素对拍 ----
        g_lo, g_hi, g_perm, g_pull = _td_tables_golden(
            slot_job.cpu(), num_padded_total, num_tokens, world, ctx.rank)
        self.blk_lo = torch.empty(nblk, dtype=torch.int32, device=device)
        self.blk_hi = torch.empty(nblk, dtype=torch.int32, device=device)
        self.l0_row_perm = torch.empty(nblk, dtype=torch.int32, device=device)
        self.pull_order = torch.empty(self.num_jobs, dtype=torch.int32,
                                      device=device)
        self._td_out = {"blk_lo": self.blk_lo, "blk_hi": self.blk_hi,
                        "row_perm": self.l0_row_perm,
                        "pull_order": self.pull_order}
        _build_td_tables_gpu(self.slot_job, num_tokens, world, ctx.rank,
                             self._td_out)
        for name, gold in (("blk_lo", g_lo), ("blk_hi", g_hi),
                           ("row_perm", g_perm), ("pull_order", g_pull)):
            assert torch.equal(self._td_out[name].cpu(), gold), \
                f"td table '{name}' GPU builder vs host golden mismatch"

        # ---- 计数器(每迭代 run 内清零, 同 stream 无需额外同步) ----
        zeros = lambda n: torch.zeros(n, dtype=torch.int32, device=device)
        self.gemm_next_l0 = zeros(1)
        self.gemm_next_l1 = zeros(1)
        self.push_next = zeros(1)
        self.pull_next = zeros(1)
        self.push_done = zeros(1)
        self.scattered_cnt = zeros(world)
        self.chunk_cnt = zeros(self.n_chunks)
        self.chunk_done = zeros(self.n_chunks)   # 值 = seq, 单调免清零
        self.done_blocks = zeros(1)

        # ---- GPU schedule rebuild(公平口径: 计入 run(), 同 tktp) ----
        self.gpu_schedule = os.environ.get("TKTD_GPU_SCHED", "1") == "1"
        if self.gpu_schedule:
            self._packed_local = torch.empty(num_tokens, self.top_k, 2,
                                             device=device, dtype=torch.int32)
            self._packed_all = torch.empty(world, num_tokens, self.top_k, 2,
                                           device=device, dtype=torch.int32)
            self._topk_ids_local = problem.topk_ids.contiguous()
            self._topk_w_local = problem.topk_weights.float().contiguous()
            self._topk_w_bits = self._topk_w_local.view(torch.int32)
            self._num_experts = num_experts
            # 共享 builder 需要的全部输出槽(pull/push/l1 相关表 tktd 不消费,
            # 但 builder 固定写全套; row_perm 这里是 L1 语义的恒等表)。
            self._sched_out = {
                "padded": self.padded, "tp_slots": self.tp_slots,
                "prered_w": self.prered_w, "slack": self.slack,
                "pull_order": torch.empty_like(self.pull_order),
                "job_order": self.job_order,
                "push_order": torch.empty(world, num_tokens,
                                          dtype=torch.int32, device=device),
                "blk_expert": self.blk_expert,
                "slot_job": self.slot_job, "slot_w": self.slot_w,
                "row_perm": torch.arange(nblk, dtype=torch.int32,
                                         device=device),
            }
            self._sched_graph = None  # captured lazily on first run()

        # ---- weights(与 tktp 同款处理, docs/10) ----
        # w1: 反量化 -> GLU 列交织([gate32|up32] per 64-N block) -> 重量化,
        # scale 块与 B tile 对齐; w2 原布局直接可用(零转置零重量化)。
        qc = problem.quant_config
        FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
        self.w2_fp8 = problem.w2.contiguous()
        self.w2_scales = qc.w2_scale.float().contiguous()
        s1 = qc.w1_scale.float()                         # (E, 2I/128, H/128)
        w1_bf = (problem.w1.float() * s1.repeat_interleave(128, 1)
                                        .repeat_interleave(128, 2))
        E = w1_bf.shape[0]
        gate = w1_bf[:, :inter].view(E, inter // 32, 32, H)
        up = w1_bf[:, inter:].view(E, inter // 32, 32, H)
        w1_il = torch.stack([gate, up], dim=2).view(E, 2 * inter, H)
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
        self.pre_tokens = TK((num_tokens, H), dtype=torch.float8_e4m3fn,
                             local_rank=lr, local_world_size=lws, multicast=False)
        self.pre_scales = TK((num_tokens, H // 128), dtype=torch.float32,
                             local_rank=lr, local_world_size=lws, multicast=False)
        self.ag_staging_fp8 = TK((S_ag, H), dtype=torch.float8_e4m3fn,
                                 local_rank=lr, local_world_size=lws,
                                 multicast=False)
        self.ag_sscales = TK((S_ag, H // 128), dtype=torch.float32,
                             local_rank=lr, local_world_size=lws,
                             multicast=False)
        # barrier_l0: row 0 = per-src 段到达 flag(远端写, col=src), row 1 =
        # pcie_barrier_all slots(约定行), row 2 = per-src 段 ready(本地写)。
        # flag 值 = seq(单调, 免清零)。
        self.barrier_l0 = TK((3, 32), dtype=torch.int, local_rank=lr,
                             local_world_size=lws, multicast=False)
        # barrier_l1: rows 2+d = 卡 d 的 combine watermark(moe_final_reduce_push
        # 的既有协议)。
        self.barrier_l1 = TK((2 + world, 32), dtype=torch.int, local_rank=lr,
                             local_world_size=lws, multicast=False)
        self.barrier_l0.data_.zero_()
        self.barrier_l1.data_.zero_()

        # LOCAL workspaces。padding 行 scale=0 -> GEMM 对 padding 行结果为 0。
        self.gathered = torch.zeros(P, H, device=device, dtype=torch.float8_e4m3fn)
        self.gathered_scales = torch.zeros(P, H // 128, device=device,
                                           dtype=torch.float32)
        self.act = torch.zeros(P, inter, device=device, dtype=torch.bfloat16)
        self.act_fp8 = torch.zeros(P, inter, device=device,
                                   dtype=torch.float8_e4m3fn)
        self.act_scales = torch.zeros(P, inter // 128, device=device,
                                      dtype=torch.float32)
        self.expert_out = torch.zeros(P, H, device=device, dtype=torch.bfloat16)
        self.combine_staging = TK((world * num_tokens, H), dtype=torch.bfloat16,
                                  local_rank=lr, local_world_size=lws, multicast=False)
        self.combine_staging.data_.zero_()
        self.combine_out = torch.zeros(num_tokens, H, device=device,
                                       dtype=torch.bfloat16)
        # 最终归约的稠密贡献表(TP 下每张卡给每个 token 都有部分和)
        self.final_contrib = torch.ones(num_tokens, world,
                                        dtype=torch.int32, device=device)
        self.recv_from = torch.ones(world, dtype=torch.int32, device=device)

        # ---- streams / events(TD 的 producer / reduce stream) ----
        self.comm_stream = torch.cuda.Stream()
        self._ev0 = torch.cuda.Event()
        self._ev_end = torch.cuda.Event()
        self._l0_seq = 0
        self._l1_seq = 0

    def _sched_call(self):
        """torch 向量化调度表构建(共享 builder + TD 专用表, 同图执行)。"""
        _build_tp_schedules_gpu(self._packed_all, self.ctx.world_size,
                                self._num_experts, self.ctx.rank,
                                self._sched_out)
        _build_td_tables_gpu(self.slot_job, self.num_tokens,
                             self.ctx.world_size, self.ctx.rank, self._td_out)

    def run(self) -> torch.Tensor:
        tk = self.tk
        main = torch.cuda.current_stream()

        def _dbg(tag):
            if self.sync_debug:
                torch.cuda.synchronize()
                print(f"[tktd r{self.ctx.rank}] {tag} ok", flush=True)
        # 公平口径: 调度表在计时区内 GPU 重建(同 tktp 的 TK_GPU_SCHED=1 语义;
        # tktd 不接 tpsched 融合 kernel, sched 阶段比 tktp 慢 ~120µs, A/B 用
        # 分阶段数字或对 tktp 设 TK_SCHED_FUSED=0 对齐口径)。
        if self.gpu_schedule:
            self._packed_local[..., 0].copy_(self._topk_ids_local)
            self._packed_local[..., 1].copy_(self._topk_w_bits)
            torch.distributed.all_gather_into_tensor(
                self._packed_all.view(-1), self._packed_local.view(-1))
            if self._sched_graph is None:
                self._sched_call()                       # warmup
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._sched_call()
                self._sched_graph = g
            else:
                self._sched_graph.replay()
        _dbg("sched")

        # 双 barrier 包住 pre_tokens 覆写窗口 + 保证所有 rank 已消费完上一迭代
        # 的 staging(与 tktp 相同的协议; barrier 用 barrier_l0 row 1)。
        self._l0_seq += 1
        tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)
        tk.rowgroup_quant_fp8(self.problem.hidden_states,
                              self.pre_tokens.data_, self.pre_scales.data_)
        self._l0_seq += 1
        tk.pcie_device_barrier(self.barrier_l0, self._l0_seq)
        _dbg("barrier+tok_quant")

        # 计数器清零(main stream; comm stream 经 ev0 排在其后)
        self.gemm_next_l0.zero_()
        self.gemm_next_l1.zero_()
        self.push_next.zero_()
        self.pull_next.zero_()
        self.push_done.zero_()
        self.scattered_cnt.zero_()
        self.chunk_cnt.zero_()
        self.done_blocks.zero_()
        self._l1_seq += 1

        # ---- layer0: producer(comm stream)∥ 段 gate GEMM(main) ----
        self._ev0.record(main)
        self.comm_stream.wait_event(self._ev0)
        with torch.cuda.stream(self.comm_stream):
            tk.moe_td_ag_producer(
                self.pre_tokens, self.pre_scales, self.ag_staging_fp8,
                self.ag_sscales, self.gathered, self.gathered_scales,
                self.tp_slots, self.pull_order, self.barrier_l0,
                self.push_next, self.pull_next, self.push_done,
                self.scattered_cnt, self.num_comm_sms, self.push_sms,
                self.num_tokens, self._l0_seq)
        tk.moe_td_ag_gemm(
            self.gathered, self.gathered_scales, self.w_gateup_fp8,
            self.w1_il_scales, self.act, self.padded, self.blk_expert,
            self.blk_lo, self.blk_hi, self.l0_row_perm, self.gemm_next_l0,
            self.barrier_l0, self.num_comm_sms, self.num_padded_total,
            self._l0_seq, self.slack, self.two_level, self.l0_no_gate)
        _dbg("L0(producer+gemm)")

        # ---- layer1: act 量化 -> N-chunk GEMM(main)∥ reduce-RS(comm) ----
        tk.rowgroup_quant_fp8(self.act, self.act_fp8, self.act_scales)
        _dbg("act_quant")
        if self.l1_serial:
            # kernel 级二分(排障), 三档见 setup 注释。
            def _dbg2(tag):
                torch.cuda.synchronize()
                print(f"[tktd r{self.ctx.rank}] {tag} ok", flush=True)

            def _nchunk():
                tk.moe_td_gemm_nchunk(
                    self.act_fp8, self.act_scales, self.w2_fp8, self.w2_scales,
                    self.expert_out, self.padded, self.blk_expert,
                    self.gemm_next_l1, self.chunk_cnt, self.chunk_done,
                    self.n_chunks, self.rs_sms, self.num_padded_total,
                    self._l1_seq, self.slack, self.two_level)

            def _rs():
                tk.moe_td_reduce_rs(
                    self.expert_out, self.tp_slots, self.prered_w,
                    self.combine_staging, self.chunk_done, self.done_blocks,
                    self.barrier_l1, self.n_chunks, self.rs_sms,
                    self.num_tokens, self._l1_seq)

            def _final():
                tk.moe_final_reduce_push(self.combine_staging,
                                         self.final_contrib, self.recv_from,
                                         self.combine_out, self.barrier_l1,
                                         self.num_tokens, self._l1_seq)

            if self.l1_serial == 2:
                # RS∥nchunk 真实 overlap, final 隔离在 sync 之后
                with torch.cuda.stream(self.comm_stream):
                    _rs()
                    self._ev_end.record(self.comm_stream)
                _nchunk()
                main.wait_event(self._ev_end)
                _dbg2("L1ab nchunk||rs")
                _final()
                _dbg2("L1c final")
            elif self.l1_serial == 3:
                # nchunk 隔离在前, RS∥final 并发在后
                _nchunk()
                _dbg2("L1a nchunk_gemm")
                with torch.cuda.stream(self.comm_stream):
                    _rs()
                    self._ev_end.record(self.comm_stream)
                _final()
                main.wait_event(self._ev_end)
                _dbg2("L1bc rs||final")
            else:
                # 全串行(模式 1)
                _nchunk()
                _dbg2("L1a nchunk_gemm")
                _rs()
                _dbg2("L1b reduce_rs")
                _final()
                _dbg2("L1c final")
            return self.combine_out
        # 发射顺序 = TD 原版的序(fp8_moe_reduce_rs.run: GEMM 先于 RS):
        # **等待者(RS, 自旋 chunk_done)必须晚于被等者(GEMM)发射**。反过来
        # (RS 先发射常驻自旋、GEMM 后发射)在 2026-07-28 首测实测死锁 →
        # 32s guard trap: 只要执行栈任何一层把"后发射"排在"先发射的完成"
        # 之后, 就构成 RS↔GEMM 等待环(TKTD_L1_SERIAL=2/3 二分坐实, 档 3
        # = 安全序通过, 档 2 = 反序死锁)。L0 的 producer/GEMM 天然是
        # 安全序(等待者 GEMM 晚发射), 故无此问题。overlap 不受损: GEMM
        # 网格 = sm - rs_sms, 给 RS 留着 SM。
        tk.moe_td_gemm_nchunk(
            self.act_fp8, self.act_scales, self.w2_fp8, self.w2_scales,
            self.expert_out, self.padded, self.blk_expert, self.gemm_next_l1,
            self.chunk_cnt, self.chunk_done, self.n_chunks, self.rs_sms,
            self.num_padded_total, self._l1_seq, self.slack, self.two_level)
        with torch.cuda.stream(self.comm_stream):
            # comm stream 上排在 producer 之后; 对 GEMM 的依赖走 chunk_done
            # flag(seq 门), 无需 stream 级同步。
            tk.moe_td_reduce_rs(
                self.expert_out, self.tp_slots, self.prered_w,
                self.combine_staging, self.chunk_done, self.done_blocks,
                self.barrier_l1, self.n_chunks, self.rs_sms,
                self.num_tokens, self._l1_seq)
            self._ev_end.record(self.comm_stream)
        tk.moe_final_reduce_push(self.combine_staging, self.final_contrib,
                                 self.recv_from, self.combine_out,
                                 self.barrier_l1, self.num_tokens, self._l1_seq)
        # 形式上把 comm stream 汇回 main(下一迭代的清零/覆写以此为序)
        main.wait_event(self._ev_end)
        _dbg("L1(nchunk+rs)+final")
        return self.combine_out
