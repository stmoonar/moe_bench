# 12 T1 归因实测:layer1 慢的是 combine gather,不是 fusion 拖慢 GEMM

**日期**:2026-07-09
**任务**:docs/11 §T1 —— ncu/隔离归因 layer1 W2⊕combine 的 71 vs ~160 TFLOP/s。
**结论**:**docs/11 §0.1(a) 的假设被推翻**。W2 GEMM 本身没有被 fusion 拖慢
(带 fence epilogue 仍跑 ~140 TFLOP/s);"71 TFLOP/s" 是把 GEMM 的 FLOP 除以
(GEMM 时间 + 一条几乎串行的 combine gather 尾巴)得到的**伪指标**。真正的成本在
**combine 跨卡 gather+reduce**(每 token 拉 8 条散布的 ~14KB peer 行,PCIe 读带宽顶),
占全部拖慢的 **94~99%**。→ **T2(epilogue 选举信号)按 docs/11 止损规则降级/冻结,
T6(combine 预归约+push)提为 layer1 的主攻方向。**

---

## 1. 方法:5 点隔离 + 闭合校验

全部在真实 W2 GEMM 形状(`act(npl,2048) @ w2(2048,7168) -> (npl,7168)`)、4 卡
(9,11,13,15)锁步下测,`tools/time_layer1.py`,50 迭代取 max-over-ranks 中位数。
FLOP 恒为 `2·npl·2048·7168`。

| 点 | 内容 | 隔离对象 |
|---|---|---|
| A | 纯 `grouped_gemm` @ 110 blocks | ~160 TFLOP/s 参考基线 |
| B | 纯 `grouped_gemm` @ 94 blocks | SM 让渡成本 = t_B − t_A |
| D | 融合 kernel,`num_source_tokens=0`(94 GEMM block + `combine_signal_epilogue`,无 combine block,不会挂) | fence/信号成本 = t_D − t_B |
| E | 完整融合 kernel(94 GEMM + 512 combine) | combine 成本 = t_E − t_D;**这就是 71 TFLOP/s 基线** |
| F | 仅 combine block(barrier 预置到已满 seq,wait 全部立即通过),无 GEMM | 纯 gather+reduce 带宽(与 HOL-wait 解耦) |

新增 debug 入口:`gg::entry_nb`(可配 block 数的纯 GEMM)、`comb::combine_only_entry`
(仅 combine block,复用已满足的 barrier seq)。均在 `tk_moe.cu`,不影响生产路径。

**闭合校验**:三段成本之和 == (t_E − t_A),三个 NE 全部**完美闭合到 0.1µs**,
证明分解无遗漏项。

## 2. 实测数据(max-tE rank)

| NE | npl | A@110 | E full | 慢多少(t_E−t_A) | SM让渡 | **fence/信号** | combine | F(纯gather) |
|---|---|---|---|---|---|---|---|---|
| 64  | 4096 | 565µs (213 TF) | 2440µs (49 TF) | 1875µs | 3.5% | **1.9%** | 94.6% | 1824µs |
| 128 | 4096 | 848µs (142 TF) | 2593µs (46 TF) | 1745µs | 0.9% | **1.2%** | 97.9% | 1831µs |
| 256 | 8192 | 1687µs (143 TF) | 3368µs (71 TF) | 1681µs | 0.7% | **2.8%** | 96.4% | 1838µs |

(NE=64 的 A 达 213 TFLOP/s 因每专家仅 64 真行 padding 到 128,但 npl 相同下
per-expert 段更短、wave 更满;NE=256 npl 翻倍 GEMM 更长,占比口径不变。)

## 3. 三个嫌疑人的裁决

docs/11 §0.1(a) 列的三个嫌疑人,实测比例:

1. **per-tile 全员 `fence.sys`(原假设的头号嫌疑)**:实测 **1.2~2.8%**。
   关键证据:D(GEMM + fence epilogue,94 block)= **~140 TFLOP/s**,与纯 GEMM
   (A 143 TFLOP/s / B 同 block 数)几乎相同。**fence 几乎免费**——每 row block 只有
   consumer 全员 fence 一次,但 fence.sys 的成本被 GEMM 的算力完全吸收,不在关键路径。
   ncu 单卡佐证:纯 W2 GEMM `gpu__compute_memory_throughput` = **91%**,是健康的
   near-compute-bound kernel,无 membar stall。
2. **16 SM 让渡给 comm(理论 ~0.85×)**:实测 **0.7~3.5%**。94 vs 110 block 的纯 GEMM
   差 12~66µs,远小于理论——因为 GEMM 在 wave≈1 下 94 block 已够覆盖,尾波影响小。
3. **combine 的跨卡 gather(原被归为 "L2 污染" 的 Z%)**:实测 **94~99%**,是**唯一大项**。
   F(纯 gather+reduce,无 GEMM 无 wait)= 1824~1838µs ≈ 整条 combine 尾巴。
   流量:512 token × 8 experts × 14KB = **58.7MB**(其中 ~3/4=44MB 跨卡),
   1838µs → **~32 GB/s 总 / ~24 GB/s 跨卡**——正是 PCIe SM-pull 弱路径带宽顶
   (probe [C2] 的 ~20GB/s 量级)。且 F 用 110 SM 跑 combine ≈ E 里 16 comm SM 跑 combine,
   **加 SM 不提速 → 带宽 bound 坐实,不是算力/占用问题**。

## 4. 为什么"71 TFLOP/s"是伪指标

融合 kernel 里 GEMM(94 comp block)和 combine(512 comm block)是**同一 grid 同时启动**
的两类 block,但 combine 的每个 token block 要**等它的 8 个 expert 行块 GEMM 完成信号**
才能开始拉,而拉本身是 PCIe 弱路径、~32GB/s。结果 combine 尾巴几乎**串行拖在 GEMM 之后**
(t_E ≈ t_GEMM + t_combine,而非 max)。把 GEMM 的 FLOP 除以这个总时间,自然得到
71 TFLOP/s——但 GEMM 本身从头到尾都是 ~140 TFLOP/s。docs/11 §0.1(a) 把这个比值误读成
"fusion 把 GEMM 拖慢 2.2×",据此推 T2 能省 1ms,方向错了。

## 5. 对路线图的影响(止损裁决)

docs/11 §T1 止损规则:**"若 fence 占比 <30% 则 T2 降级、T5/T6 提级"**。
实测 fence = 1.2~2.8% ≪ 30%,**触发降级**:

- **T2(combine epilogue 换选举信号)**:预期收益从 "layer1 −1ms" 修正为 **≤0.05ms**
  (fence 全部成本才 20~48µs)。**冻结 T2**,不值得那条 cumulativity 内存序验证的风险。
- **T6(combine 预归约 + push 化)提为 layer1 唯一主攻**:它直击 94~99% 的大项——
  - **预归约**:同卡多 expert 命中同 token 先本地 FP32 加和,发送行数 4096→~1800(§0.3);
  - **push 化**:数据面从 SM-pull 弱路径(~24GB/s)换成 TMA-push 强路径(~51GB/s);
  - **消 HOL**:源卡终归约只等 ≤3 个卡的水位信号,不再 per-token 等 8 个 max。
    账:44MB→25MB、24→51GB/s,combine 尾 1838µs → **~0.5ms**(4× 量级),且可藏进
    W2 GEMM(T5 后 ~1.7ms)之下。这是 layer1 从 3.4ms 压到 ~1.8ms 的关键。
- **T5(ROW_BLOCK=64)**:仍利好——GEMM 段本身 NE=256 有 2× padding 空转
  (A@256 1687µs vs A@64/128 更短已侧证),减半后 GEMM 段更短、combine 更好藏。
- **T4(gate+up 合并)**:不受影响,继续。

## 6. 复现

```bash
source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
cd /data/cinnzhang_vllm_td_test/xxy
# 全套 5 点隔离 + 闭合校验(三个 NE)
for ne in 64 128 256; do
  CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.time_layer1 $ne 10 50
done
# 单卡 ncu 佐证纯 GEMM 是 compute-bound
CUDA_VISIBLE_DEVICES=9 ncu --launch-count 1 --kernel-name-base demangled -k "regex:gg::kernel" \
  --metrics gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  python -m moe_bench.tools.ncu_gemm_probe 256 A
```

工具:`tools/time_layer1.py`(5 点隔离计时)、`tools/ncu_gemm_probe.py`(单卡 ncu)。
代码新增:`gg::entry_nb`、`comb::combine_only_entry`(均 debug-only,已 bind)。
