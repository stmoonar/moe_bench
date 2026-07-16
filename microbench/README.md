# microbench — 通算融合收益来源归因微基准

回答的核心问题:**tkfused 相对 vLLM serial 的收益,到底来自哪里?**
是 (a) TK 计算 kernel 本身比 vLLM 的 triton 算子快(纯计算收益),
还是 (b) 通信方案更好 + 通算重叠(通信收益)。以及:融合的理论上限在哪里、
当前离上限还有多远、inter/intra-SM 编排与 comm SM 数怎么选。

## 一键运行

```bash
bash moe_bench/microbench/run_all.sh
# 结果整目录拷走即可离线分析:
#   moe_bench/microbench/results/<时间戳>/{mb*.json, mb3_report.md, *.log, 环境快照}
```

选卡优先级:`MB_GPUS` > 已有的 `CUDA_VISIBLE_DEVICES` > 自动选第一组空闲的
优先卡组(9,11,13,15 → 8,10,12,14 → 1,3,5,7 → 0,2,4,6)。脚本会先做 preflight
(torch 看到 ≥4 卡 + vllm import 链走通)再开跑。

单个测试也可独立跑,**必须带 `CUDA_VISIBLE_DEVICES` 前缀**、在 moe_bench 的
上级目录:

```bash
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.microbench.mb1_compute
```

> ⚠ 任何 `python -m moe_bench.*` 都会在 **import 阶段** 加载 vllm →
> torch._inductor → 定制版 triton,后者 import 时就初始化 GPU driver。
> 进程此时看不到可用 CUDA 设备(没带 CUDA_VISIBLE_DEVICES 前缀、卡号无效等)
> 会直接报 `RuntimeError: 0 active drivers ([])`——这不是 microbench 的代码
> 问题。若带了有效前缀、`python -c "import vllm"` 仍报同样错误,则是环境层
> 问题(venv/torch/triton),此时 `python -m moe_bench.bench` 也会同样失败。
> mb3_ratio 是唯一例外:纯 stdlib,可在任何机器上
> `python moe_bench/microbench/mb3_ratio.py --results <dir>` 离线重跑分析。

Token sweep(**全 rank 总 token 数**,每卡 = 总数/4):
`[128, 512, 1024, 2048, 5120, 6648, 8192]`,`MB_TOKENS=...` 可覆盖。
形状固定为默认 shape:E=64, topk=8, hidden=4096, inter=3072, EP world=4;
精度默认 bf16,`MB_PRECISION=fp8` 可切(见下节)。

## 切 FP8(FP8 版 TK scheme 落地后)

精度集中在 `common.make_cfg()` 一处,由 `MB_PRECISION` 环境变量控制
(bf16 默认 / fp16 / fp8):

```bash
MB_PRECISION=fp8 bash moe_bench/microbench/run_all.sh
```

**自动生效的部分**(走 `MoEProblem`/`quant_config` 的一切):
- `data.make_weights` 按精度生成权重,fp8 时做 per-block 量化并构造
  `quant_config`;激活基精度仍是 bf16(`cfg.torch_dtype`),所以 mb1/mb2 的
  gathered/AG/RS 缓冲与字节账本(取 `element_size()`)自动正确;
- vLLM 侧:mb1 的 `vllm_compute`、mb4 的 `SerialNaive` 都把
  `problem.quant_config` 传给 `fused_experts`,自动走 fp8 w8a8 triton 路径;
- 结果 meta 的 `shape.precision` 跟随实际取值。

**不会自动生效、需要跟着 FP8 实现同步的部分**:
1. `TKFusedEP.setup` 目前断言 bf16-only(`quant_config is None`)——FP8 scheme
   落地前,`MB_PRECISION=fp8` 会在 `build_tk_fixture` 处报错(蓄意 fail-fast);
2. mb1 的 `tk_l0/tk_l1`、mb4 的 `L0_fused/L1_fused`、mb5 的全部 TK 调用都是
   **照抄 tk_scheme.run() 内部的 kernel 调用**(entry 名、buffer 属性名、参数
   顺序)。若 FP8 版保持相同的属性名与 entry 签名(仅内部 dtype/scale 变化),
   这些调用零改动;若新增了 scale 参数或换了 entry 名,需要同步改这几处
   (搜 `tk.grouped_gemm` / `tk.moe_dispatch_gemm` / `tk.moe_gemm_prered_push_fused`);
3. mb1 里 gathered 的旁路填充(`sch.gathered.data_[:npl].copy_(g)`)假定 gathered
   与激活同 dtype;若 FP8 dispatch 改为传输量化后的 fp8 字节,这里要改成
   先量化再填(或直接跑一次 fused dispatch 填充);
4. mb1 的 cuBLAS 上限参考仍是基精度 GEMM;fp8 想要对应 roof 可加
   `torch._scaled_mm` 参考(可选);
5. mb4 的交叉校验 rel_err 只记录不断言,fp8 下数值会更大(~1e-2),属预期。

## 测试清单

| 文件 | 对应需求 | 内容 |
|---|---|---|
| mb1_compute | ① 计算 kernel 单测 | 同一 gathered batch:vLLM `fused_experts`(triton) vs TK grouped-GEMM 链(gate+up → silu → W2);另给 cuBLAS 稠密上限、TK schedule 重建成本 |
| mb2_comm | ② 通信 kernel 单测 | NCCL AG(hidden/ids/w)/RS/A2A/小包 vs TK `pcie_device_barrier` / `moe_push_data`(push 数据面)/ `moe_final_reduce`(combine pull);带字节账本与有效带宽 |
| mb3_ratio | ③ 占比与理论上限 | 纯后处理:通信占比、收益分解(计算收益 vs 通信+重叠收益)、两条理论上限(只重叠 / TK+全重叠),输出 `mb3_report.md` |
| mb4_fusion | ④ 融合 vs 串行 vs 纯计算 | e2e serial / e2e tkfused / 每层「纯 GEMM vs 融合」(融合损失),含 tkfused-serial 输出交叉校验 |
| mb5_sm_sweep | ⑤ inter-SM comm SM 数 | 真实 kernel `num_comm_sms` 扫参(L0/L1/e2e)+ `grouped_gemm_nb` 纯 SM 让渡损失,可分离"让渡"与"干扰" |
| mb6_inter_intra | ⑥ inter vs intra SM | 合成探针(纯 CUDA):同量通信+同量计算在「整 block 分工」vs「block 内 warp 分工」下的 makespan / 干扰 |
| mb7_pcie_schemes | ⑦ PCIe 通信方案 | copy engine / SM pull / SM push / 顺序行 / 散布行(8KB token 行)/ 信号 RTT / 4 卡并发 ring 争用;块数扫参回答"几个 SM 打满 PCIe" |

## 公平性与可复现性设计

- **同输入**:全部基于 `moe_bench.data` 的确定性生成(seed=0, balanced 路由);
  mb1/mb4 里 vLLM 与 TK 用同一个 problem、同一路由,mb4 还做输出交叉校验。
- **同口径计时**:4 卡锁步(每次迭代前 `dist.barrier`)、CUDA events、median、
  max-over-ranks(慢卡 gate 整层)—— 与 `tools/analyze_overlap` / bench 一致;
  eager 计时(分布式 bench 同款,无 CUDA graph)。
- **路径固定**:每个 worker 显式设置 TK_* 开关(默认路径 pull + prered_push +
  fused gate/up + GPU sched),不受外部环境影响;实际取值写进结果 meta。
- **环境快照**:run_all 保存 nvidia-smi、SM/显存时钟、git commit(含 dirty 状态)、
  torch/vllm 版本、MB_*/TK_*/NCCL_* 环境变量。
- 抖动参考 docs/17:~5%;必要时增大 `MB_ITERS`。

## 口径注意(分析结果前必读)

1. **flop 基数**:vLLM 按真实 assignment 计算,TK 按 128-padding 行数计算
   (padding 是 TK 方案固有成本)。mb1 的 `*_tflops_eff` 用同一真实 assignment
   基数,直接可比;`tk_chain_tflops_padded` 是 TK 名义吞吐。
2. **通信语义字节不同**:NCCL AG 搬 shard 复制(每卡收 (world-1)·T 行),
   TK dispatch 按 assignment 搬(每卡发 T·topk 行,~3/4 跨卡,dedup 前)。
   mb2 同时给出时间、字节、带宽,对比时按字节归一。
3. **层边界不同**:vLLM `fused_experts` 内含 topk 加权求和与 scatter/gather;
   TK 的加权在 combine 里。整层对账以 mb3 的分解为准。
4. **⚠ schedule 计时公平性**:当前 `tk_scheme.py` 的 GPU schedule 重建只在
   `combine_mode == "prered"` 时计入 `run()`(tk_scheme.py:681),而默认
   combine 是 `prered_push` —— 即 **当前默认路径的 e2e 数字不含 schedule 成本**
   (~200µs 量级)。mb4 单独测了 `sched`,mb3/mb4 同时给出
   `time_reduction_vs_serial_pct`(原样)与
   `time_reduction_vs_serial_with_sched_pct`(公平口径)。两者均按
   `(serial_time - tk_time) / serial_time * 100%` 计算，负数表示比 serial 更慢。
5. mb2 的 TK 侧在 `TK_FUSE_GATEUP=0, TK_COMBINE=prered` 下构建(push_data 需要
   w_gate、final_reduce 需要 partials),只影响 buffer 准备,不影响被测通信 kernel。
6. mb6 的计算代理是 fp32 FMA(对 SM 数/warp 数敏感),不是 tensor-core GEMM;
   真实 GEMM 的 SM 让渡损失以 mb5 的 `grouped_gemm_nb` 为准。mb6 回答的是
   编排(inter/intra)与干扰的相对结论。
7. mb6/mb7 的探针扩展(`probes.cu`)走 `cudaDeviceEnablePeerAccess` 的运行时
   P2P;TK 走 VMM+IPC。两者同一 fabric 能力(同 00_probe 的论证),带宽可比,
   但不逐字节等价。
8. 首次跑 mb6/mb7 会现场编译 `probes.cu`(~1 分钟,缓存在 microbench/build/)。
   mb1/2/4/5 复用 kernels/tk 已编译的 .so(按 world/hidden/row_block 缓存)。

## 结果文件 schema

每个 `mb*.json`:`{"meta": {...环境/配置...}, "rows": [...]}`,
rows 按 token 档(mb5 再乘 comm SM 档;mb6/mb7 按配置)。时间字段一律
`*_ms`(毫秒),带宽 `*_gbps`(GB/s),字节 `*_bytes`。
`mb3_report.md` 是人读的汇总结论表。
