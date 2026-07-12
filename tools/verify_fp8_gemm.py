# SPDX-License-Identifier: Apache-2.0
"""FP8 grouped GEMM 单卡裁决(docs/37 P1):对拍 + 计时,一步定生死。

  1. 正确性:grouped_gemm_fp8(A 1×128 group 量化 + W 128×128 block 量化,
     mma e4m3,fp32 重标定)vs fp32 反量化参考,rel_err 必须 < 5e-3
     (两者输入 bit 相同,差异只有累加顺序;垮了 = scale 行映射/mma 布局错);
  2. 性能:同 shape 的 bf16 grouped_gemm 对照,报 TFLOP/s 比(目标 ≥1.8×)。

  单卡运行(设 CUDA_VISIBLE_DEVICES 选一张空闲卡):
    python -m moe_bench.tools.verify_fp8_gemm [E] [rows_per_e] [K] [N] [iters]

  默认 L0 形状:E=64, 256 行/expert(P=16384), K=4096, N=1536。
  L1 形状用:  python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096
"""
from __future__ import annotations

import os
import sys

import torch

BLOCK = 128


def quant_a_group(x: torch.Tensor):
    """1×128 group 量化(e4m3 + fp32 scale),x (rows, K) bf16。"""
    rows, K = x.shape
    g = x.float().view(rows, K // BLOCK, BLOCK)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / torch.finfo(torch.float8_e4m3fn).max
    q = (g / scale).to(torch.float8_e4m3fn)
    return q.view(rows, K), scale.view(rows, K // BLOCK).contiguous()


def quant_w_block(w: torch.Tensor):
    """128×128 block 量化,w = B^T (N, K) bf16 → (fp8, scales (N/128, K/128))。
    docs/38: kernel 用 mma_ABt,权重保持 (E, N, K) 原始布局。"""
    N, K = w.shape
    b = w.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
    amax = b.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-8)
    scale = amax / torch.finfo(torch.float8_e4m3fn).max
    q = (b / scale).to(torch.float8_e4m3fn)
    return q.view(N, K), scale.view(N // BLOCK, K // BLOCK).contiguous()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    E = int(args[0]) if len(args) > 0 else 64
    rows_e = int(args[1]) if len(args) > 1 else 256
    K = int(args[2]) if len(args) > 2 else 4096
    N = int(args[3]) if len(args) > 3 else 1536
    iters = int(args[4]) if len(args) > 4 else 20
    device = "cuda"
    torch.manual_seed(0)

    from importlib.util import module_from_spec, spec_from_file_location
    build_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "kernels", "tk", "build.py")
    spec = spec_from_file_location("_tk_build", build_py)
    bmod = module_from_spec(spec)
    spec.loader.exec_module(bmod)
    tk = bmod.build_and_load(4, hidden=4096, row_block=128)

    P = E * rows_e
    A_bf = torch.randn(P, K, device=device, dtype=torch.bfloat16) / 8
    W_bf = torch.randn(E, N, K, device=device, dtype=torch.bfloat16) / 8  # B^T (E,N,K)

    A_q, A_s = quant_a_group(A_bf)
    Wq_list, Ws_list = zip(*[quant_w_block(W_bf[e]) for e in range(E)])
    W_q = torch.stack(list(Wq_list)).contiguous()
    W_s = torch.stack(list(Ws_list)).contiguous()

    padded = torch.full((E,), rows_e, dtype=torch.int32, device=device)
    nblk = P // 128
    blk_expert = torch.arange(E, device=device, dtype=torch.int32) \
        .repeat_interleave(rows_e // 128).contiguous()
    task_next = torch.zeros(1, dtype=torch.int32, device=device)
    out = torch.zeros(P, N, device=device, dtype=torch.bfloat16)

    # ---- 正确性: fp32 反量化参考(与 kernel 完全相同的 fp8 输入 bit) ----
    a_deq = A_q.float() * A_s.repeat_interleave(BLOCK, dim=1)
    ref = torch.empty(P, N, device=device, dtype=torch.float32)
    for e in range(E):
        w_deq = W_q[e].float() * W_s[e].repeat_interleave(BLOCK, dim=0) \
                                      .repeat_interleave(BLOCK, dim=1)  # (N, K)
        ref[e * rows_e:(e + 1) * rows_e] = a_deq[e * rows_e:(e + 1) * rows_e] @ w_deq.T

    task_next.zero_()
    tk.grouped_gemm_fp8(A_q, A_s, W_q, W_s, out, padded, blk_expert, task_next, 0, False)
    torch.cuda.synchronize()
    rel = (out.float() - ref).norm() / ref.norm()
    ok = rel < 5e-3
    print(f"[fp8 gemm] E={E} P={P} K={K} N={N}  rel_err={rel:.2e}  "
          f"{'OK' if ok else 'FAIL'}")

    # ---- 性能: vs bf16 grouped_gemm(同 shape) ----
    out_bf = torch.zeros(P, N, device=device, dtype=torch.bfloat16)

    def bench(fn, n=iters, warm=5):
        for _ in range(warm):
            fn()
        s, e_ = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(n):
            fn()
        e_.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e_) * 1000 / n  # us

    def run_fp8():
        task_next.zero_()
        tk.grouped_gemm_fp8(A_q, A_s, W_q, W_s, out, padded, blk_expert, task_next, 0, False)

    def run_raw():  # 裸 mma 上限探针(docs/38: 裁决 fp32 累加税假说)
        task_next.zero_()
        tk.grouped_gemm_fp8(A_q, A_s, W_q, W_s, out, padded, blk_expert, task_next, 0, True)

    W_bf_kn = W_bf.transpose(1, 2).contiguous()  # bf16 参考要 (E, K, N)
    t8 = bench(run_fp8)
    traw = bench(run_raw)
    t16 = bench(lambda: tk.grouped_gemm(A_bf, W_bf_kn, out_bf, padded, 0))
    fl = 2.0 * P * K * N
    print(f"[fp8 gemm] fp8 {t8:8.1f}us ({fl / t8 / 1e6:6.1f} TFLOP/s)   "
          f"raw {traw:8.1f}us ({fl / traw / 1e6:6.1f} TFLOP/s)   "
          f"bf16 {t16:8.1f}us ({fl / t16 / 1e6:6.1f} TFLOP/s)   "
          f"speedup {t16 / t8:.2f}x  raw-cap {t16 / traw:.2f}x")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
