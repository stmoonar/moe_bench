# 11 评审:性能账本重算、layer1 归因假设与下一步优化路线图

**日期**:2026-07-09
**评审对象**:push3 落地后的全量代码(`tk_moe.cu` / `sm120_common.cuh` / `tk_scheme.py`)
+ docs 04/07/08/09/10 的全部实测数据 + experience/01 相关工作对照(COMET / Triton-Distributed / Flux / DeepEP)。
**前置**:docs/07(上一轮评审)、docs/10(push3 实测)。
**性质**:本文是**任务清单文档**——每个任务给出现状证据、具体改法、验证锚点、预期收益、
风险与止损。后续每完成一项,在对应小节标注结果并开新文档记录细节。

---

## 0. 核心结论(为什么路线图长这样)

### 0.1 性能账本重算(NE=256, 512 token/rank, 4 卡组 9,11,13,15)

e2e pull 7124µs 的构成:

| 段 | 时间 | 性质 | 判断 |
|---|---|---|---|
| dispatch⊕gate GEMM | ~2005µs | **comm-bound**:跨卡 ~43MB pull @ ~20GB/s ≈ 2.15ms | 已贴带宽顶,协议微调无肉 |
| up GEMM(独立 kernel) | ~1500µs | compute,~160 TFLOP/s | **完全没被通信掩盖,裸露串行** |
| silu·mul | ~100µs | 杂项 | — |
| W2⊕combine | ~3500µs | **GEMM-bound**:240 GFLOP / 3.4ms ≈ **71 TFLOP/s** | GEMM 效率只有独立 GEMM 的一半不到 |

两个此前未被点名的关键事实:

**(a) layer1 的 combine 通信其实已经藏住了。** combine pull 43MB @ 32~49GB/s ≈
0.9~1.4ms < W2 GEMM 的 3.4ms。layer1 慢不是因为通信裸露,而是**融合把 GEMM 本身拖慢了
~2.2×**(71 vs ~160 TFLOP/s)。嫌疑人按大小排:
1. `combine_signal_epilogue` **每个 tile 全体 consumer 线程 `__threadfence_system()`**
   ——每 row block 有 col_blocks=56 个列 tile,每 tile 256 线程都 fence 一次
   (docs/07 P5 当时说"可测占比,暂不动",现在它是全局最大单项嫌疑);
2. 16 个 SM 让给 comm(理论 ~0.85×);
3. peer 跨卡读 expert_out 污染 L2(DeepSeek-V3 经验:通信数据不应驻留 L2)。

**(b) 两层 GEMM 都在为 padding 付 ~2× 代价。** NE=256 时每专家均值 64 真 token
padding 到 128,8192 padded 行里一半是零,同时拖长 up、W2,并推高 push3 的行块数
(docs/10 §5 交叉点的成因之一)。

### 0.2 push3 在 NE=256 退化的补充归因

docs/10 §5 归因为"信号/gate 协议开销 ∝ 行块数"。本轮评审补一个同样重要的因素:
**push 顺序按源 token 序(`src_tok*topk+k`),目的行块的就绪时间全部后置**。
pull 模式 dispatch 按 slot 序拉,块 0 最先满,GEMM 流水起步;push3 下任何一个目的块要等
"源卡推给它的最后一个 token",而它均匀散布在整个 push 流里——行块越多,GEMM 起步越晚、
尾部越裸露。这能解释退化幅度(dispatch-only +774µs)远超多出的 ~192 个远端小 store 本身。
→ 对应 T8 的两个解法(重排 + 水位信号)。

### 0.3 与 COMET / Triton-Distributed 的差距清单(layer1 视角)

| 维度 | COMET / TD 的做法 | 我们现状 | 对应任务 |
|---|---|---|---|
| GEMM 完整度 | TBS 下 GEMM CTA 全速,通信 CTA 专职;尾部 split 全员帮忙收尾 | epilogue 每 tile 全员 fence.sys,GEMM 掉到 71 TFLOP/s | T1/T2 |
| 通信流量 | expert 卡**先本地归约再发**(同卡多 expert 命中同 token 只发一行) | 源卡按 (token,expert) 逐行 pull,4096 行全传 | T6 |
| 数据面方向 | 推式写远端(强路径)/原子加 | 源卡 SM pull(本机弱~中路径) | T6 |
| 消费粒度 | layer1 沿 N 切 SPLITS,列块就绪即启动归约 | per-token block 等 8 个信号的 max,HOL 阻塞 | T6 |
| 信号聚合 | TD"计数聚合通知":细粒度完成廉价聚合成粗信号 | push3 信号数 ∝ 行块数 | T8 |
| 资源配比 | 通信 CTA 数随 shape 自适应 | `num_comm_sms=16` 写死 | T9 |
| 调度顺序 | Flux swizzle / CommFuse"执行序跟随到达序" | push 按源 token 序,块就绪后置 | T8 |

我们已经对齐的:两阶段选举信号协议(= TD 同款,push3/combine 已用)、warp
specialization 的 gate 融合、无远端原子纪律。差距集中在 **epilogue 成本、流量去重、
数据面方向、信号聚合、调度顺序**五项——全部有对应任务。

### 0.4 正确性 review 结论

无新 bug。推演过成立的:push3 选举信号链与 host/device 对账、pull reset 的 stream 序、
P2 显式 barrier、padding 闭环、barrier 生命周期。遗留风险(非 bug,记录在案):
- P1 公平性(schedule 不计时)仍未解 → T3,**做任何提速宣称前必须**;
- push3 / combine epilogue 的跨卡内存序依赖 PTX cumulativity 在 PCIe 上的实际表现,
  **仅本机验证过**;换卡/换拓扑必须重跑 `validate_push3` 类协议裁决(写进 probe 清单);
- `disp::dispatch` 每线程 load→wait→store→wait 全串行、每 block 仅 6 token 槽
  (96KB/14KB),282/288 线程闲置——当前 16 comm SM × 6 = 96 路并发够打满 20GB/s,
  但减 comm SM 或换快链路时会先瓶颈(暂不动,记录);
- combine per-token block FIFO 等 8 信号的 HOL(docs/07 P5)仍在;skewed 分布未测 → T9;
- `AGENTS.md` 写"当前实现和 baseline 都使用 FP8",实际 `TKFusedEP` assert bf16-only,
  文档与代码不一致 → T9 修正口径(fp8 是 T7 的目标,不是现状)。

---

## 1. 任务清单(按优先级)

依赖关系总览:

```
T1 (ncu归因) ──决定──> T2 (epilogue选举信号)
T3 (schedule GPU化) —— 独立,信用前提
T4 (gate+up合并) ──┬── 与 T5 叠加后 up 完全隐藏
T5 (ROW_BLOCK=64) ─┘    T5 同时利好 T2/T6/T8
T6 (combine 预归约+push) —— 依赖 T2 的协议模板,复用 push3 骨架
T7 (dispatch 去重+fp8) —— 独立,与 push/pull 正交
T8 (push3 重排+水位) —— 依赖 T7 后收益更明显;成功则 push 设默认
T9 (工程杂项) —— 穿插
```

---

### T1:ncu 归因 layer1 的 71 vs ~160 TFLOP/s(半天,决定 T2 预期)

**现状证据**:§0.1(a)。三个嫌疑人(per-tile 全员 fence.sys / 16 SM 让渡 / L2 污染)
比例未知,docs/07 P5 留的坑。

**做法**:
1. 构造三个对照 kernel 单测(复用 tools/ 模式,4 卡跑真实 shape):
   - (i) `moe_gemm_combine_fused` 原样(71 TFLOP/s 基线);
   - (ii) 同 kernel 但 epilogue 换 no-op(隔离 fence+信号成本;combine block 不等信号
     会挂,故此档只跑 GEMM block:`num_source_tokens=0` 或加 debug 入口);
   - (iii) 独立 `grouped_gemm` 同 shape 同 grid(num_comp_sms 个 block,隔离 SM 让渡)。
2. ncu 抓 (i):`smsp__inst_executed_pipe_lsu`、`membar` 相关计数、
   `lts__t_sectors_srcunit_tex_op_read`(L2 侧看 peer 读流量)、SM 占用与 stall 分布。
3. 出一页归因表:fence X%、SM 让渡 Y%、L2 干扰 Z%。

**验证锚点**:三档 kernel 时间差闭合(|i−ii−fence成本| 等自洽)。
**预期**:确定 T2 的收益上限;若 fence 占比 <30% 则 T2 降级、T5/T6 提级。
**止损**:纯测量任务,无止损问题;半天内出结论。

> **【已完成 2026-07-09 — 见 docs/12】结论:假设推翻。** 5 点隔离(A纯GEMM@110 /
> B纯GEMM@94 / D GEMM+fence / E完整融合 / F纯combine)+ 完美闭合校验实测:
> **fence 仅 1.2~2.8%、SM让渡 0.7~3.5%、combine gather 94~99%**(三个 NE 一致)。
> D(GEMM+fence)= ~140 TFLOP/s == 纯 GEMM,**fence 几乎免费**;"71 TFLOP/s" 是
> GEMM-FLOP 除以(GEMM + 串行 combine 尾)的伪指标,GEMM 本身从未被拖慢。combine 尾
> F=1838µs 拉 58.7MB @ ~32GB/s(跨卡 44MB @ ~24GB/s)= PCIe 弱路径带宽顶,加 SM 不提速。
> ncu 佐证纯 GEMM `compute_memory_throughput`=91%(健康 compute-bound)。
> **→ 触发止损:T2 冻结,T6 提为 layer1 唯一主攻。**

---

### T2:combine epilogue 换 push3 式选举信号(预期 layer1 −1ms 级)

> **【已冻结 2026-07-09 — T1(docs/12)止损裁决】** T1 实测 fence 仅 1.2~2.8%
> (全部成本 20~48µs),远低于 30% 止损线。原"预期 layer1 −1ms"基于 §0.1(a) 的
> 错误归因(fusion 拖慢 GEMM),已被推翻:GEMM 带 fence 仍跑 ~140 TFLOP/s。本任务
> 收益上限 ≤0.05ms,不值得 cumulativity 内存序验证风险。**冻结,收益转投 T6。**
> 以下原文保留备查。

**现状证据**:`comb::combine_signal_epilogue` 每 tile:全体 256 consumer 线程
`__threadfence_system()` → group sync → 单线程 `atom.acq_rel.gpu` 计数 → 满额者向
4 卡广播 signal_slot。fence 是 per-tile × 全员。

**做法**(协议 = push3 §2 已裁决的同款,方向反过来):
1. epilogue 改为:每 tile 完成后**单线程** `atom.release.gpu` 计数该 row block 的列块数
   (release 保证本 CTA 先前的输出写先于计数可见,gpu scope 本地合法);
2. 满额者(唯一)`__threadfence_system()` 一次 → `signal_slot` 广播。
   即:fence.sys 从 `每tile×256线程` 降到 `每row block×1线程`(56×256 → 1,每行块)。
3. **内存序差异必须裁决**:push3 的数据写是 TMA `store_async_wait`(bulk 提交),
   这里是 consumer 的普通 `st.global`(register path)。"其他 CTA 的普通 store →
   其 release-add → 选举者 acquire → 选举者 fence.sys → 远端信号"这条 cumulativity 链
   与 push3 R1 同类但**不同写方式**,需要同样的隔离验证,不能只跑对拍。

**验证锚点**(照搬 docs/10 §2 方法论):
1. 新增 debug 入口 `moe_gemm_combine_signal_only`(GEMM+新epilogue,combine block 不跑),
   目的卡对账:信号到达 ⇔ 该行块 expert_out 逐字节 == 非融合锚点(Phase 3a 产物);
2. NE=64/256 各 30 迭代 total_failures=0 才算过;
3. 融合全链路对拍 reference_moe(NE=64),rel_err 与现版一致(~4.3e-3);
4. 性能:W2⊕combine 段计时 + TFLOP/s 复算。

**预期收益**:layer1 3.5ms → 2.2~2.5ms(若 T1 证实 fence 主导)。
**风险与止损**:若 30 迭代出现"信号先于数据"失败,退一档——保留 per-tile fence 但只让
**每 warp lane 0** fence(成本降 32×,内存序等价于现状,因为 fence 只需覆盖本线程可见的
先前写?否——不等价,需全员或逐 strip 论证;此退档方案需单独推演)。再不行冻结 T2,
转 T5/T6,现协议保底正确。

---

### T3:schedule GPU 化并计入 run()(公平性 P1,信用前提)

> **【已完成 2026-07-09 — 见 docs/14】** 默认路径(pull+prered+fused)的 6 张 schedule
> 表 GPU 向量化(单次全局 argsort 复现 ring-order slot + 稠密 job 空间),CUDA graph 捕获
> (~205µs),每迭代计入 run()。对 host golden element-wise 全等(NE∈{64,128,256}×
> {balanced,skewed})。e2e NE=256 计入后 5832µs(+229µs),仍 < serial 7063(1.21×),
> **星号已去掉**。`TK_GPU_SCHED=1` 默认。

**现状证据**:docs/07 P1。`_build_schedules` host 四重循环在 setup() 不计时;serial 的
`moe_align_block_size` 每次 run 在 GPU 上做。"打平 serial"带星号。

**做法**:
1. 把 replay 循环向量化上 GPU:对 all_topk (world, T, topk) 做
   `argsort(expert_id)` + `cumsum(padded)` 反推 slot;ring 序用 (src_dev − e_rank) mod W
   作为次级排序键。产出 disp_idx / comb_idx / push_* / gate_expected 全套,
   与现 host 版**逐元素对账**(现版即 golden);
2. 目标 <100µs(docs/07 的账),计入 `run()` 每迭代;all_gather topk_ids 也计入
   (serial 侧 routing 广播同样在计时内,口径对齐);
3. bench 报告口径备注更新:去掉"routing 外置"星号。

**验证锚点**:GPU schedule vs host schedule 逐元素相等(NE=64/256 × balanced/skewed);
计时前后对拍 reference_moe 不变。
**预期收益**:负收益(e2e +~0.1ms),买的是结论可信度。**必须在下一轮性能宣称前完成**。
**止损**:若向量化后 >300µs,允许保留"CPU 版预计算 + 报告注明"双轨,但报告必须给
两个口径的数字。

---

### T4:gate+up 合并为一次 GEMM + silu·mul epilogue(省 ~0.5ms,叠 T5 后省整段 up)

> **【已完成 2026-07-09 — 见 docs/13 §6】** pull 路径把 gate+up 合并为一次
> dispatch⊕GEMM(w_gateup = w1.T (E,H,2*inter),N 翻倍,GEMM 模板零改动),up 计算
> 藏进 comm-bound dispatch,省掉独立 up GEMM。实测 NE=256 e2e 5775→5571µs(−204)、
> NE=64 3712→3234µs(−478);正确(run_tkfused NE=64/128 rel_err ~4.3e-3 ok)。
> **默认开启**(`TK_FUSE_GATEUP=1`);push* dispatch 自动回退双 GEMM(它们 push 进 w_gate)。
> NE=256 收益小于 ~0.5ms 预估(N 翻倍后 dispatch 更偏 compute,up 未完全隐藏),
> **T5(ROW_BLOCK=64)落地后 up 才完全沉入通信之下**(docs/11 §T4 原预期)。

**现状证据**:§0.1 账本——up GEMM ~1.5ms 裸露串行;`w1` 本来就是 `[gate; up]` 连排
(E, 2*inter, H),`self.w_gate/self.w_up` 是人为拆开的。

**做法**:
1. `tk_scheme.py`:不拆 w1,直接 `w_gateup = w1.transpose(1,2).contiguous()`
   (E_local, H, 2*inter);`gate_out/up_out` 合并为 `gateup_out (padded, 2*inter)`;
2. `moe_dispatch_gemm`(pull)与 `moe_dispatch_push3` 的 N 维从 inter 变 2*inter,
   **gate functor 与 GEMM 模板零改动**(col_blocks 翻倍自动生效);
3. silu·mul 作为该 kernel 的 output epilogue:tile store 后,持有 gate 半区 tile 的
   consumer 无法看到 up 半区(不同 col tile)——所以 v1 先不做 tile 内融合,改为
   **独立小 kernel** `silu_mul(gateup_out) -> act`(读一次写一次,~0.2ms,替代现在
   torch 的 silu+mul 两次读写);v2 若要 epilogue 化,需按 col 配对调度
   (col c 与 col c+inter/128 同 SM),另开任务;
4. 删除独立 up `grouped_gemm` 调用。

**验证锚点**:对拍 reference_moe(NE=64),rel_err 不变;dispatch-only 计时看 N 翻倍后
comm-bound 是否仍成立(NE=256 会变 compute-bound ~3.0ms,NE≤128 仍 comm-bound)。
**预期收益**:up 的 1.5ms 计算改为藏在 dispatch 通信下:单独做省 ~0.5ms(NE=256,
compute 反超 comm);**与 T5 叠加后 gate+up 计算减半(~1.5ms)完全沉到 2.0ms 通信之下,
省整段**。
**风险与止损**:低风险(纯 shape 变化)。若 NE=256 下 fused kernel 变 compute-bound
且总时长 > 现状(2005+1500),按档位选择:NE=256 回退双 kernel,NE≤128 用合并版
——但 T5 落地后应无此分支。

---

### T5:ROW_BLOCK=64(两层 GEMM 各省 ~35~50% 行,NE=256 最大单项)

**现状证据**:§0.1(b)。NE=256 每专家均值 64 真 token padding 到 128,GEMM 算 2× 的行;
docs/07 #4 已论证 tile 64×128 warp mma 无障碍。

**做法**:
1. `gemm_config` 出 64 行变体:`A_tile = st_bf<64, RED_BLOCK>`,CONSUMER_WARPS=4
   (每 warp 16 行)或保持 8 warp 每 warp 8 行(需查 st 子 tile 与 mma 形状约束,
   取编译通过且无 spill 的那档);smem 预算重算(A tile 减半,可考虑 PIPELINE_STAGES=4);
2. `group::store` 的 warpgroup 交织映射(docs/05 的坑)按新 WARPS 数重推
   `store_strip` 公式,**先单卡 grouped_gemm 对拍 torch**(Phase 1 锚点重跑);
3. padding 单位、slack seed、行块计数、barrier 列数全部由 ROW_BLOCK 常量派生,
   编译期换挡(-DTK_ROW_BLOCK=64),host 侧 `tk_scheme.py` 的 ROW_BLOCK 同步;
4. 对 dispatch/push3/combine 的影响:行块数翻倍 → pull gate 自旋数、push3 信号数、
   combine 等待槽数都 ×2——**T8 的水位信号是对冲**;NE=256 档收益(GEMM −50% 行)
   远大于协议开销增量,NE=64 档(padding 本就少)可能持平,保持双档编译可切。

**验证锚点**:Phase 1 单卡对拍 → Phase 2/3 融合对拍 → reference_moe 全链路,
NE∈{64,128,256} 全扫(docs/10 教训:必须全档位)。
**预期收益**:NE=256:up/gate+up 与 W2 的 GEMM 时间近似减半(W2 3.4→~1.7ms 上限),
e2e 预期 −1.5ms 级。
**风险与止损**:group::store 行映射是本仓库踩过的最深的坑(docs/05),必须锚点递进,
禁止直接上全链路。若 4-warp 形态 mma 效率掉(wave 不满),试 8 warp × 8 行;
两档都差则冻结,仅在 NE=256 档用"每专家变长尾块"(只对尾块用 64)的窄化版。

---

### T6:combine 预归约 + push 化(A' 镜像,layer1 通信 4×,消 HOL)

> **【进行中 2026-07-09 — v0 地基已落地,见 docs/13】** 第一步(v0/v1 共用的 host
> schedule 预归约表 + 对账工具)完成:`_build_prereduce_schedule` 把 combine 求和按
> expert 卡重新分组(数学等价,FP32),`tools/reconcile_prereduce.py` 对 golden
> `comb_idx`/`combine_w` **双向逐项对账**(覆盖+无多余 bijection + contrib mask),
> **NE∈{64,128,256} 全过 total_failures=0**。纯 host、未接入 run()。
> 下一步:v0 kernel(expert 侧预归约 + 源卡终归约,数据面仍 pull)。

**现状证据**:§0.3 差距表前四行。现状源卡逐 (token,expert) pull 4096 行 = 43MB 跨卡
@ 弱~中路径;per-token block 等 8 个跨卡信号的 max。

**做法**(= push3 骨架反向复用,协议全部已验证):
1. **expert 卡侧**,`moe_gemm_combine_fused` 改造:
   - GEMM epilogue 只做本地事(T2 的选举信号,发**本地** slot,不跨卡);
   - 同 grid 的 comm block(原 combine block 位置)改职能:等本地行块信号 →
     对同一 (源 token, 本卡) 命中的多条 expert 输出行做 **FP32 加权预归约**
     (权重 topk_weight 在 expert 卡侧乘,host schedule 新增本卡视角的
     (src_dev, src_tok) → 本卡 slot 列表 + 权重表)→ 归约结果一行 14KB
     smem → **TMA push** 到源卡 staging(与 dpush3::push3 的数据面同构)→
     per (本卡→源卡) 水位信号(T8 同款,或 v1 先 per-块 slot 信号);
2. **源卡 staging**:TKParallelTensor (world, num_tokens, H) bf16 = 4×512×14KB=28MB,
   每写者(expert 卡)独占 [d] 平面,无原子;
3. **源卡终归约**:`out[t] = Σ_d staging[d][t]`(只加实际有贡献的卡,host 给
   contrib mask),纯本地 elementwise 小 kernel(28MB 读,~0.1ms);v2 可融进
   下一算子或做成同 kernel 第三种 block;
4. 流量账:发送行数从 4096 → unique(token, expert卡) ≈ 1800(与 dispatch 去重同因子),
   25MB push @51GB/s ≈ **0.5ms**,且全藏在 W2 GEMM(T5 后 ~1.7ms)之下;
5. 精度:预归约 FP32、传输 bf16(比现版多一次 bf16 舍入,容差需重standing:
   现版 rel_err 4.3e-3,预归约版预期同量级,验收线不放松);
6. **combine 保持 bf16 传输,不上 fp8**(求和端不冒险,业界通例 DeepEP 同)。

**验证锚点**:
1. host schedule:预归约表 vs comb_idx 逆映射逐元素对账;
2. 预归约-push-only debug 入口(不跑终归约):staging 内容 vs
   "现版 pull 结果按 (token,卡) 分组 FP32 和"逐字节;30 迭代 ×{64,256};
3. 全链路对拍 reference_moe;
4. 计时:layer1 段 + e2e,NE∈{64,128,256}。

**预期收益**:layer1 通信 43MB pull → 25MB push(~4× 时间);HOL 消失(源卡只等 ≤3 个
水位信号);与 T2/T5 叠加后 layer1 3.5ms → ~1.8ms 有账可算。
**风险与止损**:工程量最大的一项(host schedule 新表 + kernel 双侧改)。风险点:
(a) TMA push 的源是 HBM 中的预归约行,需先落 smem——comm block 的 smem 预算与 GEMM
tile 共存(comm block 不跑 GEMM,可整块 reinterpret,COMET 同款);(b) 内存序链 =
push3 R1 同款(TMA bulk 写 + 选举信号),验证方法直接搬。若正确性裁决反复失败,
退半步:预归约仍做、push 改回源卡 pull(pull 预归约行,流量同样 4096→1800 行,
拿一半收益,零新协议)。

---

### T7:dispatch (token,dst) 去重 + FP8 传输(dispatch 2.0 → ~0.5ms)

**现状证据**:docs/07 P3/#2。每卡 assignments≈4096 而 unique 源 token≈1800,
去重省 ~2.2×;fp8 再省 2×,两者正交,且 serial 的 NCCL 通信恒为 bf16,
这是对 serial-fp8 的单方面优势。

**做法**:
1. **去重(先做,pull/push 双路都吃)**:
   - host schedule:pull_dispatch_indices 改两级——unique 拉取表
     (每 (token,src_dev) 一条,拉到 staging)+ 本地 scatter 表
     (staging 行 → gathered 的多个 slot,本地带宽近乎免费);
   - pull 版:dispatch block 拉 unique 行到 staging,scatter 到各 slot 后对
     **每个** slot 的行块计数 red.release.gpu(gate 语义不变);
   - push 版(T8 后):源卡每 unique token 推一份到目的卡 staging,目的卡本地
     scatter——push 天然适合去重(docs/10 §6);
2. **fp8 传输(第二步)**:pre_tokens 增出 fp8e4m3 副本 + per-token scale
   (128 分块 scale 或 per-token,对齐 DeepSeek 口径);传输 fp8,GEMM 入口反量化
   bf16(v1),SM120 原生 fp8 mma 换挡为 v2(与 harness `--precision fp8` 对比口径
   同步切换,对手从 serial-bf16 换成 serial-fp8,docs/07 §0 的"最终目标线");
   token_vec 从 sv_bf<7168>(14KB)变 sv_fl8<7168>(7KB),TOKENS_PER_BLOCK 翻倍。

**验证锚点**:去重版 gathered vs 现版逐字节(去重只改搬运不改结果);fp8 版对拍
reference_moe 用 fp8 容差(bench 现成);dispatch-only 计时 NE∈{64,128,256}。
**预期收益**:57MB → 26MB(去重)→ 13MB(fp8);pull @20GB/s:2.0 → 0.9 → ~0.6ms;
push @51GB/s(T8 后):→ ~0.25ms。
**风险与止损**:去重零精度风险;fp8 反量化的 GEMM 入口改动局部在 producer load 后
(或预先独立反量化 kernel,v0 最稳)。fp8 若 rel_err 超容差,检查 scale 分块粒度。

---

### T8:push3 按目的块重排 + 水位信号聚合(让 push 在 NE=256 翻盘,设为默认)

**现状证据**:§0.2 补充归因 + docs/10 §5。两个正交开销:块就绪后置(重排解)、
信号数 ∝ 行块数(水位解)。

**做法**:
1. **重排(host 侧,零 kernel 改动,先single做)**:`_build_schedules` 产出后,对
   `push_idx/push_src/push_cnt_idx` 三表按键 `(ring_offset(dst_dev, rank), dst_slot)`
   稳定排序——每张源卡先推满目的块 0 再推块 1,且各源卡从不同目的卡起步
   (Flux swizzle:`dst = (rank + 1 + i) mod W` 起,错开 incast);
   块就绪由"均匀散布"变"按序前压",GEMM 流水起步。**单独实测这一条**,
   可能已把 NE=256 拉回 pull 之下;
2. **水位信号(kernel 小改)**:重排后每 (src,dst) 对的块完成天然趋序,信号收敛为
   per-(src,dst) **单 slot 水位值**:选举者写 `seq*nblk_dst + (已完成块数)`,
   gate 等 `barrier[2+s][0] >= seq*nblk + rb+1`;
   乱序兜底:块 b 的选举者先本地自旋等水位 == seq*nblk+b 再写 b+1(本地链,
   无远端原子);每卡远端 store 从 ~256 → 3 个,信号成本与 NE 解耦
   (TD"计数聚合通知");barrier 列 0 一个槽即可,行布局不变;
3. 与 T7 去重叠加:push 每 unique token 一份 + 目的卡 scatter;
4. 全档位扫 NE∈{64,128,256}(docs/10 教训),NE=256 若 push < pull,
   `TK_DISPATCH` 默认切 push3。

**验证锚点**:重排版走现有 `verify_push3_schedule` + `validate_push3`(对账逻辑
与顺序无关,直接复用);水位版新增水位一致性对账(最终水位 == seq*nblk+nblk_contrib);
融合对拍 + 30 迭代 stress。
**预期收益**:dispatch-only NE=256 从 2779µs 压回 <2000(重排)→ 数据面账
~1.1ms(去重后 26MB@51)+ 协议 ~0.2ms(水位后)。
**风险与止损**:重排零风险。水位的本地自旋链在**严重乱序**时退化为串行等待
(最坏 = 现状信号成本),不会更差;若实现复杂度超预期,重排+per-块信号(现协议)
已可能达标,水位降为增强项。

---

### T9:工程杂项与口径修正(穿插做)

1. **AGENTS.md 口径修正**:改"当前实现使用 FP8"为"目标 FP8(T7),当前 bf16
   对 bf16";避免下个 session 误判状态。
2. **skewed / single 分布测试**:自旋协议的尾延迟工况(docs/07 P4),T2/T6/T8 每项
   落地后都要在 skewed 下跑一遍 stress(不挂死 + 时间不爆炸);bench 加
   `--distribution skewed` 的 tkfused 通道。
3. **num_comm_sms 扫参**:{8,12,16,24} × NE∈{64,256} × {pull,push3},打表进 yaml
   (COMET 的自适应我们用查表版);T4/T5 改变 compute/comm 比后要重扫。
4. **probe 增强**(docs/08 遗留):高并发散射远端红加压力档 + cumulativity 协议档
   (push3 式两阶段),换机器先跑。
5. **HANDOFF.md 建立并随任务更新**(AGENTS.md 要求,此前缺失)。
6. **远期(全部落地后)**:整层单 persistent kernel——W2 行块等 act 行块本地计数即可
   跨层流水(layer0 尾部与 layer1 头部重叠),FlashDMoE 方向但用 TK 静态任务划分;
   以及 silu·mul 的 epilogue 化(T4 v2)。到时候再立项,不在本轮。

---

## 2. 总账与验收

> **【2026-07-09 T1 后修订】** 原表含 T2 行,已随 T2 冻结作废。T1(docs/12)实测
> combine 尾 1838µs **完全裸露**(t_E ≈ t_GEMM + t_combine,零重叠),layer1 的收益
> 全部从 T6 出,且比原账大;终点量级不变。

预期逐项叠加(NE=256, 4 卡, 512 token/rank,现状 7124µs = L0 2005 + up 1500 +
silu 100 + L1 3500):

| 里程碑 | e2e 预期 | 备注 |
|---|---|---|
| T6-v0 预归约(pull 预归约行) | ~6.2ms | combine 尾 1838→~840µs(行数 4096→~1800) |
| + T6-v1 push 化 | ~5.6ms | 强路径 51GB/s + 边就绪边推,尾藏进 GEMM,L1 ≈ 2.0ms |
| + T4 gate+up 合并 | ~5.1ms | up 部分藏进 dispatch 通信 |
| + T5 ROW_BLOCK=64 | ~3.9ms | 两层 GEMM 减半,up 完全隐藏,L1 ≈ 1.1ms |
| + T7 去重+fp8 | ~2.6ms | dispatch 2.0→~0.6(pull) |
| + T8 push 默认 | **~2.3ms** | dispatch 数据面 ~0.3ms |

对照 docs/07 §3 的账(~2.5–3.5ms vs serial 7ms)一致;且 serial-fp8 的通信仍是
bf16 AG/RS,fp8 口径下差距只会更大。**所有性能宣称在 T3 落地(schedule 计入 run)
之后才对外报数。**

每完成一项:对拍锚点全过 → 提交(规范 commit message)→ 结果写回本文档对应小节
+ 细节开新编号文档 → 更新 HANDOFF.md。

---

## 3. T1 之后的执行顺序修订(2026-07-09)

T1(docs/12)裁决后,§1 的依赖图更新为:

```
T6 (combine 预归约+push) —— layer1 唯一主攻,拆 v0 → v1 两个台阶
T4 (gate+up 合并) —— 独立、低风险、~1天,作为 T6 期间的并行小活
T3 (schedule GPU化) —— 排在 T6 的 schedule 新表定型之后、报数之前
T5 (ROW_BLOCK=64) —— 后移到 T6 之后
T7 → T8 —— 顺序不变
T2 —— 冻结(T1 止损)
```

理由:

1. **T6 拆 v0/v1**(两者共用同一套 host schedule 预归约表 + expert 侧预归约逻辑,
   v0 是 v1 的中间检查点,不是绕路):
   - **v0:预归约 + 源卡 pull 预归约行**——零新协议,信号/pull 数据面沿用现状,
     只换索引表;风险集中在 host schedule 新表,用"预归约表 vs comb_idx 逆映射对账"
     裁决;顺带验证预归约的精度账(多一次 bf16 舍入)。预期 combine 尾减半。
   - **v1:数据面反转为 expert 侧 TMA push + 源卡终归约**——dpush3 骨架镜像,
     内存序链 = push3 R1 同款,`validate_push3` 方法论直接搬。v1 同时治好 T1 发现的
     "零重叠"病:预归约行就绪即推,推流贯穿 GEMM,per-token 等 8 信号的 max 消失。
   - **第一件事是 v0 的 host schedule 新表 + 对账工具**——v0/v1 共用的地基,
     风险最集中处,先裁决干净。
2. **T5 后移**:T1 证明 GEMM 本身健康(~140 TFLOP/s),combine 尾裸露时省 GEMM
   只是让尾更裸露,T5 的收益要等 T6 兑现后才能落袋;且行块数翻倍加重信号协议,
   先有 T6/T8 的聚合信号再上 T5 更顺。
3. **T3 的插入时点**:T6 会新增预归约表,T3 的 GPU 向量化范围顺势把新表一起覆盖,
   避免向量化两遍;但必须在下一次对外报数之前完成。
