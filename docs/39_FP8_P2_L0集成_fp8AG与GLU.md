# 39 FP8 P2:税坐实、L0 集成(fp8 AG ⊕ fp8 GEMM ⊕ GLU)

> 承接 docs/38。raw 探针裁决 + P2 落码。

## 1. fp32 累加税:坐实(sm120 平台事实,归档)

raw(去掉全部重标定/scale 读)= **304/311 TFLOP/s**,仅比正式版高 7-12%
——纯 mma+流水的硬上限就在 ~305,**fp8+fp32acc 峰值 ≈ bf16 峰值**确认
(boost 频率略高于估算,~2.7GHz)。正式版 283 = raw 的 93%(重标定税
~7%,blockwise fp8 固有)。结论:

- **消费级硅上 blockwise fp8 的 GEMM 收益 ≈ 1.28-1.34×,全部来自字节
  减半,不是 mma 加速**;f16 累加可免税但 128-K 块内有真实溢出风险
  (DeepSeek 在 Hopper 用 fp32 promotion 的同一个原因),不做;
- 修订账:L0 GEMM 969→~750、L1 508→~380(P3);AG 字节减半是我方独有
  结构优势(serial 的 AG 传 bf16)。

## 2. P2 落码:L0 全链 fp8(tpdisp8,本轮)

- **源端 1×128 group 量化**(run() 计时区内,torch 实现 ~25µs):
  hidden → pre_tokens(fp8, 4KB/token)+ pre_scales(float, 128B/token),
  两个 TKParallelTensor,复用现有 pcie 双 barrier;
- **fp8 AG-dispatch**(dispatch_persistent 同构):每 token 两个 TMA
  (行 + scale 行)共用一个 mbarrier(expect 4224B),各自 scatter 到
  gathered / gathered_scales 的 TOP_K slot;**AG 线上字节 8KB → 4.125KB**;
  TOKENS_PER_BLOCK 12→20(fp8 行更小);
- **L0 GEMM = grouped_gemm_sm120_fp8_dispenser + glu_store_policy**:
  a_scales 直读 gathered_scales;权重在 **GLU 列交织后的布局上重量化**
  (setup 一次:dequant problem.w1 → 交织 [gate64|up64] → 128×128 重量化
  → scale 块与 B tile 天然对齐,B^T (E,N,K) 免转置);fp32 epilogue 上
  silu*up 直存 bf16 act;
- **L1/push/combine 全部保持 bf16**(w2 setup 时反量化),阶段边界干净;
- padding 行 gathered_scales=0 → 反量化恒 0,与 bf16 padding 语义一致。

## 3. 预期与已知风险

- e2e 预期:L0_fused 1170 →(GEMM ~750 + 让渡 ~150)~900,+ 量化 ~25
  → **~1900-1950 vs serial fp8 2074(~1.07-1.09×)**;P3(L1 fp8)后
  ~1750-1800(~1.15-1.18×)。fp32 累加税把 docs/37 的 1.25-1.4× 预期
  砍到了这个水平——报数时如实注明平台上限;
- **verify 预期会 FAIL 在 fp8 容差上**(serial fp8 也 FAIL:rel_err
  1.67e-2 vs 元素级 3.5e-2 容差;我们的权重还是"反量化-交织-重量化",
  多一层量化差)。诊断打印已加,拿到 max_abs 后统一校准 TP fp8 容差
  ——rel_err 与 serial 同量级(~2e-2)即算过。

## 4. 下一步

```bash
STEPS='^00_|^01_|^03f8_|^04f8_|^04f_' bash tools/run_tp_all.sh   # ~3 分钟
```

看点:03f8 的 rel_err 量级(vs serial fp8 的 1.67e-2)+ 诊断行;
04f8 的 e2e vs serial fp8 2074。过了进 P3(act 量化融 GLU epilogue +
L1 fp8 GEMM)。
