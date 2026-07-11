# SPDX-License-Identifier: Apache-2.0
"""MB7 PCIe 下不同通信方案对比(单进程多卡探针)。

对每个卡对、每个消息尺寸(= token sweep 对应的每卡 shard 字节数 + 64MB 饱和点):

  memcpy_peer   : copy engine(cudaMemcpyPeerAsync)
  sm_pull       : 目的卡 SM 读远端(ld,弱路径)—— TK pull dispatch 的数据面
  sm_push       : 源卡 SM 写远端(st,强路径)—— TK push 的数据面
  row_pull_seq  : 按 8KB token 行顺序 pull(行粒度开销)
  row_pull_scat : 按 8KB token 行乱序散布 pull —— dispatch/combine 的真实访问模式
  row_push_scat : 散布行 push

另测:
  块数扫参      : sm_pull/sm_push 带宽 vs block(SM)数,回答"几个 SM 打满 PCIe"
                  (供 mb5 选 num_comm_sms 对照)
  pingpong      : st.release.sys/ld.acquire.sys 信号 RTT(信号协议成本)
  concurrent    : 4 卡同时 ring pull/push(共享根节点/交换机的并发争用,
                  MoE dispatch/combine 的实际形态)

  python -m moe_bench.microbench.mb7_pcie_schemes
"""
from __future__ import annotations

import os

import torch

from . import common
from .build_probes import load_probes

ROW_BYTES = 8192  # 一个 token 行:4096 × bf16
BLOCK_SWEEP = [1, 2, 4, 8, 16, 32]
DEFAULT_BLOCKS = 16
THREADS = 256


def _sizes() -> list[int]:
    sizes = sorted({(t // common.WORLD) * ROW_BYTES
                    for t in common.global_token_sweep()})
    sizes.append(64 << 20)
    return sizes


def main():
    ext = load_probes()
    ext.enable_p2p()
    ndev = torch.cuda.device_count()
    assert ndev >= 2, "mb7 needs >= 2 visible GPUs"
    warmup, iters = common.warmup_iters(), common.bench_iters()
    sizes = _sizes()
    max_bytes = max(sizes)
    torch.manual_seed(0)

    # 每卡两块缓冲(a=写目标, b=读源),覆盖最大尺寸
    buf_a, buf_b = [], []
    for d in range(ndev):
        buf_a.append(torch.empty(max_bytes // 4, dtype=torch.float32, device=d))
        buf_b.append(torch.randn(max_bytes // 4, dtype=torch.float32, device=d))

    rows = []

    def bw(bytes_, ms):
        return bytes_ / (ms * 1e-3) / 1e9

    pairs = [(0, j) for j in range(1, ndev)]

    # ---- 方案 × 尺寸(固定 DEFAULT_BLOCKS) ----
    for a, b in pairs:
        for size in sizes:
            n = size // 4
            da, sb = buf_a[a][:n], buf_b[b][:n]
            db, sa = buf_a[b][:n], buf_b[a][:n]
            nrows = size // ROW_BYTES
            rblocks = max(min(nrows, 64), 1)
            # 行索引放在执行卡(a)上,kernel 本地读索引、远端读/写数据行
            idx = torch.randperm(nrows, device=a).to(torch.int32)

            cases = {
                "memcpy_peer": lambda: ext.memcpy_peer(da, sb, a),
                "sm_pull": lambda: ext.sm_copy(da, sb, a, DEFAULT_BLOCKS, THREADS),
                "sm_push": lambda: ext.sm_copy(db, sa, a, DEFAULT_BLOCKS, THREADS),
                "row_pull_seq": lambda: ext.row_copy(
                    da, sb, None, None, nrows, ROW_BYTES, a, rblocks, THREADS),
                "row_pull_scat": lambda: ext.row_copy(
                    da, sb, None, idx, nrows, ROW_BYTES, a, rblocks, THREADS),
                "row_push_scat": lambda: ext.row_copy(
                    db, sa, idx, None, nrows, ROW_BYTES, a, rblocks, THREADS),
            }
            for name, fn in cases.items():
                ms = common.single_dev_median_ms(fn, a, warmup, iters)
                rows.append({"kind": "scheme", "pair": [a, b], "scheme": name,
                             "bytes": size, "ms": ms, "gbps": bw(size, ms)})
            r = {c["scheme"]: c["gbps"] for c in rows[-len(cases):]}
            print(f"[mb7] pair({a},{b}) {size>>20:3d}MB  "
                  f"memcpy={r['memcpy_peer']:5.1f}  pull={r['sm_pull']:5.1f}  "
                  f"push={r['sm_push']:5.1f}  rowpull_scat={r['row_pull_scat']:5.1f}  "
                  f"rowpush_scat={r['row_push_scat']:5.1f}  GB/s", flush=True)

    # ---- 块数扫参(pair(0,1),16MB) ----
    size = min(16 << 20, max_bytes)
    n = size // 4
    for k in BLOCK_SWEEP:
        for scheme, fn in [
                ("sm_pull", lambda k=k: ext.sm_copy(buf_a[0][:n], buf_b[1][:n],
                                                    0, k, THREADS)),
                ("sm_push", lambda k=k: ext.sm_copy(buf_a[1][:n], buf_b[0][:n],
                                                    0, k, THREADS))]:
            ms = common.single_dev_median_ms(fn, 0, warmup, iters)
            rows.append({"kind": "block_sweep", "pair": [0, 1], "scheme": scheme,
                         "blocks": k, "bytes": size, "ms": ms,
                         "gbps": bw(size, ms)})
        print(f"[mb7] blocks={k:2d}  "
              f"pull={rows[-2]['gbps']:5.1f}GB/s  push={rows[-1]['gbps']:5.1f}GB/s",
              flush=True)

    # ---- 信号 RTT ----
    pp_iters = int(os.environ.get("MB_PINGPONG_ITERS", "20000"))
    for a, b in pairs:
        flag_a = torch.zeros(1, dtype=torch.int32, device=a)
        flag_b = torch.zeros(1, dtype=torch.int32, device=b)
        err_a = torch.zeros(1, dtype=torch.int32, device=a)
        err_b = torch.zeros(1, dtype=torch.int32, device=b)
        sec = ext.pingpong(flag_a, flag_b, err_a, err_b, pp_iters)
        assert int(err_a.item()) == 0 and int(err_b.item()) == 0, \
            f"pingpong timeout on pair ({a},{b})"
        rtt_us = sec / pp_iters * 1e6
        rows.append({"kind": "pingpong", "pair": [a, b], "iters": pp_iters,
                     "rtt_us": rtt_us})
        print(f"[mb7] pingpong pair({a},{b})  RTT={rtt_us:.2f}us", flush=True)

    # ---- 4 卡并发 ring(争用形态) ----
    if ndev >= 4:
        size = min(32 << 20, max_bytes)
        n = size // 4
        for direction in ("pull", "push"):
            # warmup
            for _ in range(3):
                for d in range(ndev):
                    peer = (d + 1) % ndev
                    if direction == "pull":
                        ext.sm_copy(buf_a[d][:n], buf_b[peer][:n], d,
                                    DEFAULT_BLOCKS, THREADS)
                    else:
                        ext.sm_copy(buf_a[peer][:n], buf_b[d][:n], d,
                                    DEFAULT_BLOCKS, THREADS)
            for d in range(ndev):
                torch.cuda.synchronize(d)
            evs = []
            for d in range(ndev):
                peer = (d + 1) % ndev
                with torch.cuda.device(d):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    if direction == "pull":
                        ext.sm_copy(buf_a[d][:n], buf_b[peer][:n], d,
                                    DEFAULT_BLOCKS, THREADS)
                    else:
                        ext.sm_copy(buf_a[peer][:n], buf_b[d][:n], d,
                                    DEFAULT_BLOCKS, THREADS)
                    e1.record()
                evs.append((d, e0, e1))
            for d in range(ndev):
                torch.cuda.synchronize(d)
            per_dev = {d: e0.elapsed_time(e1) for d, e0, e1 in evs}
            worst = max(per_dev.values())
            agg = bw(size * ndev, worst)
            rows.append({"kind": "concurrent_ring", "direction": direction,
                         "bytes_per_dev": size, "per_dev_ms": per_dev,
                         "worst_ms": worst, "aggregate_gbps": agg})
            print(f"[mb7] concurrent ring {direction}: worst={worst*1e3:.0f}us  "
                  f"aggregate={agg:.1f}GB/s "
                  f"(单卡 {agg/ndev:.1f}GB/s)", flush=True)

    meta = common.result_meta({
        "test": "mb7_pcie_schemes", "row_bytes": ROW_BYTES,
        "default_blocks": DEFAULT_BLOCKS, "sizes": sizes,
        "pairs_note": "pair 下标是 CUDA_VISIBLE_DEVICES 重映射后的逻辑序号",
    })
    common.write_json("mb7_pcie_schemes", meta, rows)


if __name__ == "__main__":
    main()
