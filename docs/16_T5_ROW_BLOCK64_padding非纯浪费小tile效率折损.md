# 16 T5:ROW_BLOCK=64 —— 揭示 padding 不是纯浪费,小 tile 效率折损抵消收益

**日期**:2026-07-09
**任务**:docs/11 §T5 —— NE=256 每专家 64 真 token padding 到 128,两层 GEMM 算 2× 行;
预期 ROW_BLOCK=64 减半 padding → GEMM 减半 → e2e −1.5ms。
**状态**:✅ 基础设施落地(编译期 `TK_ROW_BLOCK` 开关,全档位正确),但**预期收益被推翻**:
padding 行本以满效率(143 TFLOP/s)计算,去掉后 npl=4096 落进小 tile 低效区(75 TFLOP/s),
W2 GEMM 时间几乎不变(1683→1609µs)。**默认保持 128**;`TK_ROW_BLOCK=64` 仅 NE=256 净赢
~260µs,NE≤128 反而变慢。

---

## 1. 实现(编译期开关,正确)

- `ThunderKittens/.../sm120_common.cuh` `gemm_config`:`ROW_BLOCK = TK_ROW_BLOCK`
  (默认 128),`CONSUMER_WARPS = ROW_BLOCK/16`,`A_tile = st_bf<ROW_BLOCK,RED_BLOCK>`。
  ROW_BLOCK=64 → 4 consumer warp,每 warp 一个 16 行条带(4×16=64)。
- `build.py`:`-DTK_ROW_BLOCK`,`.so` 缓存名加 `_rb{N}`,`build_and_load(row_block=)`。
- `tk_scheme.py`:`ROW_BLOCK = int(os.environ.get("TK_ROW_BLOCK","128"))`,传给 build。
  下游 padding/schedule/GPU builder 全部由此常量派生,自动换挡。

**docs/05 行映射坑规避**:WG=4 时 `group::store` 交织映射 `w/4 + (w%4)*(WG/4) == w`
**退化为恒等**(不像 WG=8 的 0,2,4,6,1,3,5,7);consumer 加载用同一 `store_strip` 公式,
两档都对。Phase-1 单卡 grouped_gemm 对拍:4 个 16 行条带 mean_diff 全相等(0.00051),
ratio 0.0014 ≪ 1(docs/05 判据:看 mean_diff/ref_mean 比值),**无系统性 bug**。

## 2. 正确性(全档位)

- Phase-1 单卡 grouped_gemm vs torch:ratio 0.0014(K∈{2048,7168,64});
- verify_schedule_gpu(RB=64):NE∈{64,128,256}×{balanced,skewed} 逐元素全等;
- validate_prered(RB=64):NE∈{64,128,256} 30 迭代 total_failures=0,max_rel~7e-3;
- run_tkfused 全链路:NE=64/128 rel_err 4.42/4.34e-3 ok。

## 3. 性能:预期被推翻(核心发现)

W2 GEMM 单测(act(npl,2048)@w2(2048,7168),NE=256,64 专家,含 padding 零行):

| 配置 | npl | 时间 | 有效 TFLOP/s |
|---|---|---|---|
| RB=128 | 8192(64→128 padding) | 1683µs | 143 |
| RB=64  | 4096(64→64,无 padding) | 1609µs | **75** |
| RB=64  | 8192(128→128,2 块/专家) | 1699µs | 142 |

**关键**:RB=64 在 npl=8192(每专家 128 行=2 块)时是满效率 142 TFLOP/s——64 行 tile 本身
不慢。慢的是 **npl=4096(每专家恰好 64 行=1 块)** 这个点:75 TFLOP/s。半的行数 × 半的效率
= 几乎相同时间。**padding 行原本是以满效率计算的,不是纯浪费**;去掉它反而把 GEMM 推进
小规模低效区(wave/tail 量化,K-reduction 流水填充摊销不足)。docs/11 §T5 的"−1.5ms"预期
基于"padding 是零成本浪费"的错误前提。

e2e(4 卡,512 tok/rank):

| NE | RB=128 | RB=64 | Δ |
|---|---|---|---|
| 64  | 3462µs | 3587µs | **+125(变慢)** |
| 128 | 3557µs | 3814µs | **+257(变慢)** |
| 256 | 5832µs | 5573µs | **−259(赢)** |

NE≤128 每专家 token 多(NE=64:256/专家),RB=128 本就少 padding,RB=64 的小 tile 低效
占上风 → 变慢。只有 NE=256(每专家 ~64,padding 2×)RB=64 净赢,且赢的不是 W2 GEMM
(那是 wash)而是 combine 尾/scatter 等非 GEMM 段因行块数/npl 减半而变快。

## 4. 结论

- **默认 RB=128**(全档位安全,NE≤128 更快);`TK_ROW_BLOCK=64` 是 NE=256 专用开关(−260µs)。
- 真正的大收益需 **8 warp × 8 行**变体(每 warp 8 行 mma,`rt_fl<8,...>`,改 mma 形状 +
  store 映射)——让 64 行 tile 保持满效率同时减半行数。工作量大、docs/05 级风险,单独立项。
- 或者 fp8 **计算**(SM120 原生 fp8 mma)直接砍 GEMM 时间,比动 tile 形状更干净(精度切换线)。
- **账本再修正**:layer0/layer1 的 GEMM 在 NE=256 是满效率 compute-bound;padding 的
  2× 行确实在算,但以满速算,所以"减 padding"不等于"减时间"——除非同时保住 tile 效率。

## 5. 复现

```bash
CUDA_VISIBLE_DEVICES=9,11,13,15 TK_ROW_BLOCK=64 python -m moe_bench.tools.run_tkfused 64
CUDA_VISIBLE_DEVICES=9,11,13,15 TK_ROW_BLOCK=64 python -m moe_bench.bench --distributed \
    --scheme tkfused --mode ep --precision bf16 --world-size 4 --no-verify --num-tokens 512
```
