# 41 FP8:A 刀证伪(pull 延迟受限)、tok_copy 量化 kernel 化

> tp_run_20260712_121959(05f sweep + 08f 首份 fp8 归因)。承接 docs/40。

## 1. A 刀证伪:fp8 拐点不左移,24 仍单调最优

05f:8:2868 / 12:2261 / 16:2131 / **24:1922**。docs/40 的假设(AG 字节
减半 → 拉取 SM 减半)错在把 pull 当成带宽受限——**pull 是延迟/并发受限**
(mb 平台事实:pull 弱路径;PCIe 往返 ~2µs):fp8 把每 token 行从 8KB 减到
4KB,减字节不减 RTT,在飞并发深度照样需要 24 个 SM。归档负结果:
**fp8 不改变 comm_sms 拐点**。

## 2. 首份 fp8 归因(08f, T=512)与新头号便宜肉

```
sched 275 | tok_copy 112(!) | L0_fused 935 | silu 6 | L1(bf16) 695
| final_red 20 | full 1931
L1 nb 分解: GEMM@86=590(SM 让渡 +63), 真实尾 105
```

- **tok_copy 30→112:torch 版 token 量化(五连发小 kernel)~80µs**,
  远超预估(25),成为第三大单项 → 本轮 kernel 化:
  `rowgroup_quant_fp8`(单 kernel,block=行,8 warp 按 group 跨步,
  shfl 归约 amax;预期 ~15µs,**−75µs**);token(groups=32)与 P3 的
  act 量化(groups=6)共用;
- L0_fused 935(比 bf16 1170 省 235 ✓);
- **异常记档**:L0_gemm_alone(gg8 参考)= 1032,比隔离工具的 729 慢
  ~300µs,产生 −97 的假负暴露——stages 环境下 gg8 参考失真(嫌疑:上下
  文 L2/功耗状态或参考写 (P,1536) 的 store 差异),不影响 fused 侧结论,
  待查;
- L1 尾巴首次分解:105µs 真实尾 + 63 让渡 → P3(L1 fp8)预期修正。

## 3. 更新账(T=512)

1931 → 量化 kernel(−75)→ P3 L1 fp8(−135)→ C sched(−80)≈
**~1640 → 1.26× vs serial fp8 2074**;T=1024 预计 ~1.35×+。

P3 集成阻力低:w2 的 B^T = problem.w2 原布局 (E,H,inter) 免转置,
**qc.w2_scale (E,H/128,inter/128) 原样可用(无 GLU 交织,无重量化)**;
act 量化复用本轮 kernel;L1 kernel = v1 结构 + fp8 dispenser
(signal_epilogue 不变,push/combine 读 bf16 expert_out 零改动)。

## 4. 下一步

```bash
STEPS='^00_|^01_|^03f8_|^04f8_|^08f_' bash tools/run_tp_all.sh   # 验证量化 kernel
```

看点:tok_copy 112→~35、e2e ~1845、正确性 rel_err 不变(数值路径等价)。
过了进 P3。
