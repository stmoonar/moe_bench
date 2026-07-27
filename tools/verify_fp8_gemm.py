# SPDX-License-Identifier: Apache-2.0
"""FP8 grouped GEMM 单卡裁决(docs/03):对拍 + 计时,一步定生死。

  1. 正确性:grouped_gemm_fp8(A 1×128 group 量化 + W 128×128 block 量化,
     mma e4m3,fp32 重标定)vs fp32 反量化参考,rel_err 必须 < 5e-3
     (两者输入 bit 相同,差异只有累加顺序;垮了 = scale 行映射/mma 布局错);
  2. 效率:与"裸 mma"探针(跳过重标定, 结果不正确)对比, 得到重标定的
     指令代价占比 —— 这是判断 GEMM 引擎还有没有肉的第一根探针。

  单卡运行(设 CUDA_VISIBLE_DEVICES 选一张空闲卡):
    python -m moe_bench.tools.verify_fp8_gemm [E] [rows_per_e] [K] [N] [iters] [row_block] [tail_rows]

  默认 L0 形状:E=64, 256 行/expert(P=16384), K=4096, N=1536。
  L1 形状用:  python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096
  row_block(默认 128): TK_ROW_BLOCK 编译宏, 传 64 测"64 行任务的单位
  经济学"(docs/09 §4 两级 tile 立项判据; .so 按 rb 分开缓存, 互不污染)。
  tail_rows(默认 0, 1-64): 两级 tile Phase 1 裁决模式 —— 每 expert 真实
  行数 = rows_per_e + tail_rows, 128 补齐后多出一个近空尾块(uniform 税
  的形状), A/B 对比两级 on(blk_rows 表)/off(全满块), 正确性只对真实行。
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
    kernel 用 mma_ABt,权重保持 (E, N, K) 原始布局(docs/02)。"""
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
    rb = int(args[5]) if len(args) > 5 else 128
    tail = int(args[6]) if len(args) > 6 else 0
    assert rows_e % rb == 0, f"rows_per_e={rows_e} 必须整除 row_block={rb}"
    assert 0 <= tail <= 64, "tail_rows 取值 [0, 64]"
    if rb != 128:
        print(f"[fp8 gemm] override row_block={rb} (默认 128)")
        assert tail == 0, "tail 模式按主配置 RB128 口径测"
    device = "cuda"
    torch.manual_seed(0)

    from importlib.util import module_from_spec, spec_from_file_location
    build_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "kernels", "tk", "build.py")
    spec = spec_from_file_location("_tk_build", build_py)
    bmod = module_from_spec(spec)
    spec.loader.exec_module(bmod)
    tk = bmod.build_and_load(4, hidden=4096, row_block=rb)

    # 几何: tail 模式下每 expert 真实行 rows_e+tail, 128 补齐到 rows_e+128
    # (最后一个块只有 tail 行是真的 —— uniform 税的形状, docs/09 §2)。
    per_e_pad = rows_e + (BLOCK if tail else 0)
    per_e_real = rows_e + tail
    P = E * per_e_pad
    A_bf = torch.randn(P, K, device=device, dtype=torch.bfloat16) / 8
    W_bf = torch.randn(E, N, K, device=device, dtype=torch.bfloat16) / 8  # B^T (E,N,K)

    A_q, A_s = quant_a_group(A_bf)
    Wq_list, Ws_list = zip(*[quant_w_block(W_bf[e]) for e in range(E)])
    W_q = torch.stack(list(Wq_list)).contiguous()
    W_s = torch.stack(list(Ws_list)).contiguous()

    padded = torch.full((E,), per_e_pad, dtype=torch.int32, device=device)
    blk_expert = torch.arange(E, device=device, dtype=torch.int32) \
        .repeat_interleave(per_e_pad // rb).contiguous()
    blk_rows = None
    if tail:
        blk_rows = torch.tensor(([BLOCK] * (rows_e // BLOCK) + [tail]) * E,
                                dtype=torch.int32, device=device)
    task_next = torch.zeros(1, dtype=torch.int32, device=device)
    out = torch.zeros(P, N, device=device, dtype=torch.bfloat16)

    # ---- 正确性: fp32 反量化参考(与 kernel 完全相同的 fp8 输入 bit),
    #      只算/只比每 expert 的真实行 ----
    a_deq = A_q.float() * A_s.repeat_interleave(BLOCK, dim=1)
    ref = torch.zeros(P, N, device=device, dtype=torch.float32)
    real_idx = torch.cat([
        torch.arange(e * per_e_pad, e * per_e_pad + per_e_real) for e in range(E)
    ]).to(device)
    for e in range(E):
        w_deq = W_q[e].float() * W_s[e].repeat_interleave(BLOCK, dim=0) \
                                      .repeat_interleave(BLOCK, dim=1)  # (N, K)
        rs = e * per_e_pad
        ref[rs:rs + per_e_real] = a_deq[rs:rs + per_e_real] @ w_deq.T

    def run(fn_raw: bool, rows_tbl):
        task_next.zero_()
        if rows_tbl is None:
            tk.grouped_gemm_fp8(A_q, A_s, W_q, W_s, out, padded, blk_expert,
                                task_next, 0, fn_raw)
        else:
            tk.grouped_gemm_fp8(A_q, A_s, W_q, W_s, out, padded, blk_expert,
                                task_next, 0, fn_raw, rows_tbl)

    def check(tag, rows_tbl):
        out.zero_()
        run(False, rows_tbl)
        torch.cuda.synchronize()
        rel = (out[real_idx].float() - ref[real_idx]).norm() / ref[real_idx].norm()
        ok = rel < 5e-3
        print(f"[fp8 gemm] E={E} P={P} K={K} N={N} real/e={per_e_real}  "
              f"{tag} rel_err={rel:.2e}  {'OK' if ok else 'FAIL'}")
        return ok

    ok = check("full", None)
    if tail:
        ok = check("2level", blk_rows) and ok

    # ---- 效率: fp8 vs 裸 mma 上限(tail 模式下再加两级 on/off A/B) ----
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

    fl = 2.0 * P * K * N  # padded 口径 FLOPs(与历史口径一致)
    t8 = bench(lambda: run(False, None))
    traw = bench(lambda: run(True, None))
    print(f"[fp8 gemm] fp8 {t8:8.1f}us ({fl / t8 / 1e6:6.1f} TFLOP/s)   "
          f"raw {traw:8.1f}us ({fl / traw / 1e6:6.1f} TFLOP/s)   "
          f"重标定代价 {(t8 - traw) / traw * 100:+.1f}%")
    if tail:
        t8_2 = bench(lambda: run(False, blk_rows))
        traw_2 = bench(lambda: run(True, blk_rows))
        print(f"[fp8 gemm] 2level fp8 {t8_2:8.1f}us ({(t8_2 - t8) / t8 * 100:+.1f}% vs full)   "
              f"raw {traw_2:8.1f}us ({(traw_2 - traw) / traw * 100:+.1f}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
