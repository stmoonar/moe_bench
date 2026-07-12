# 28 TP 第八轮:环境干扰假说确认、TP-T3 ray 卡死尸检与无 ray 重写

> 第八轮(tp_run_20260711_154440,`TUNE=1` 首跑):39/40 过,唯一 FAIL 是
> step 09 调优本体。承接 docs/27。

## 1. 结果更新(fair 口径,comm24 默认,med µs)

| 配置 | serial | tktp | 加速 | 备注 |
|---|---:|---:|---:|---|
| NE=64, T=512 | 2626 | **2263** | **1.16×** | 与第七轮 2264 完全复现 |
| T=1024 | 5022 | **4178**(rep 4176) | **1.20×** | 两次重复差 <0.05% |
| T=256 | 1669 | 1650 | ~1.01× | 上轮 serial 离群本轮消失 |
| NE=128 | 3277 | 2971 | 1.10× | |
| NE=256 (RB128) | 5286 | 5399 | 0.98× | |
| NE=256 (RB64) | 5286 | **4921**(rep 4922) | **1.07×** | 上轮双峰本轮完全恢复 |

- comm sweep 三连庄:8/16/24/32/40 = 2720/2333/**2258**/2412/2610,拐点 24 稳定。
- push 冻结回归点 2349(仍慢于 pull ✓)。
- 分阶段归因(T=512)与上轮一致:sched 247、L0 暴露 251、L1 暴露 189、
  GEMM-alone 1488;理论地板 ~1900µs(1.38×)。

## 2. 环境干扰假说:裁决 = 确认(外因)

docs/27 布下的两条取证线本轮全部收口:

1. **同 session 重复跑**:两个历史波动档(t1024、ne256_rb64)本轮各重复两次,
   全部一致到 0.1% 以内 → 内因(协议 bug)应当稳定复现双峰,排除;
2. **clocks_per_step.csv**:GPU0 全程被其它租户占 59.9GB(印证共享机);我们
   4 张卡的步间时钟快照在 487~2370MHz 间波动,但本轮所有结果稳定 → 步间
   快照的低时钟是空闲降频,不是 throttle 证据;
3. 三轮汇总:双峰在不同配置间游走、serial 也中招、min 恒等于稳定轮值——
   全部与"机器级间歇干扰"自洽,与任何内因模型矛盾。

**报数纪律落地**:今后按 med 报数;若某档出现双峰,当轮重复跑一次,以
"重复一致的 med"为准,不再单独取 min。残留噪音示例:本轮 06b_stages_cs16
的 L0_fused 3932(同配置 bench 是稳定的 4291),即一次被抓到的干扰窗口。

## 3. TP-T3 v1 尸检:ray 卡死 2.5h,零 trial

step 09(`tune_vllm_moe_tp.sh` v1 → vllm `benchmark_moe.py --tune`)rc=134:

- 15:57 ray 起了本地实例后,driver 卡死在
  `CoreWorkerProcess::Initialize → RegisterClient → recv`(与 raylet 的
  socket 握手),**挂满 2.5 小时**被 run_step 的 timeout SIGTERM → abort;
- 一个 benchmark trial 都没跑:`serial_tuned` 与未调优 serial 完全同速
  (2628 vs 2627µs),且日志仍报 `Using default MoE config`;
- 日志开头的 `Failed to import Triton kernels (triton.language.target_info)`
  是**良性噪音**——正常 serial bench 里同样出现,不是失败原因;
- 有价值的副产物:warning 给出了精确的查表文件名
  `E=64,N=768,device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell.json`,且证明
  venv 的 vllm 就是源码树 `vllm_td-main`(可编辑安装),装 config 的目录就是
  `vllm_td-main/vllm/model_executor/layers/fused_moe/configs/`。

教训:**在共享大机器上不要引入 ray 这类重编排层**,失败模式是"无限挂起",
比崩溃更贵(烧光 timeout 才暴露)。

## 4. TP-T3 v2:`tools/tune_moe_tp_noray.py`(无 ray 重写)

- **编排**:父进程把 config 空间按 GPU 数切片,每片一个 `subprocess`
  (各钉一张 `CUDA_VISIBLE_DEVICES`),无 ray/无多进程 pickling;
- **口径对齐 serial**:直接调 vllm `fused_experts`,bf16、全量 E、N=768、
  topk=8 balanced、`quant_config=None`,kwargs 按签名内省组装(容版本漂移);
- **注入**:monkeypatch `try_get_optimal_moe_config`(模块全局引用,patch 属性
  即生效);无该符号时退化为"写 config 文件 + `get_moe_configs.cache_clear()`"
  (该模式强制单卡,避免多 shard 互踩同一文件);
- **先自证再开跑**:调用计数 + 两个极端 config 的耗时差(阈值 2%),注入无效
  立刻退出并打印诊断(vllm 版本/模块路径/候选符号),不再白烧;一键脚本里
  smoke(<2min)在全量之前;
- **搜索空间**:与 `benchmark_moe.py` 相同的 1920 组,加 smem 预过滤
  ((BM·BK+BK·BN)·2·stages > 99KB 直接跳过)后剩 **648 组**,省掉必败编译;
  config 外层循环、M 内层循环摊薄 triton 编译;
- **三档 E 全调**:NE sweep 的 serial 查表键是 `E=<NE>,N=768`,E∈{64,128,256}
  各产一份 config(单档 4 卡 ~15min,三档 <1h);
- **产物落两处**:vllm configs 目录(生效)+ `tools/build_tune/`(留档)。

一键脚本 step 09 同步升级:调优后复测 serial **全网格**(t256/t512/t1024/
ne128/ne256),并加裁决步 `09v_tuned_applied`(任一 tuned 日志仍报
`Using default MoE config` 即 FAIL,不再靠肉眼)。

## 5. 下一步

1. **重跑 `TUNE=1 bash tools/run_tp_all.sh`**(预计比上轮多 ~1h,不再有
   3h 的 ray 黑洞)。看点:09_tune 的增益表 + 09v 裁决 + tuned serial 全网格;
2. 拿到 tuned serial 后**报终数**(当前 1.16×/1.20× 是对未调优 triton 的,
   预期比率会回落,幅度取决于默认 config 离最优多远);
3. 若比率回落过多需要找补:sched 第二刀(合并 all_gather/TK 侧路由 gather,
   预估 −40~60µs,docs/27 §3)是下一个可压项。
