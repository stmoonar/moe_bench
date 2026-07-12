# HANDOFF — TK 通算融合 MoE 进度交接

> 新 session 从这里接手。读完本文 + 最新一篇 docs/ 即可继续。

**最后更新**:2026-07-11(TP 分支 tp_test 第二轮:**首轮实测正确性全绿但性能 0.73× serial
(3613 vs 2624µs),归因为两处调度串行**(docs/20):L0 ring 拉取序使 GEMM 停在整个 AG 后 +
L1 每 job 一块在 16 comm SM 上排 128 波。已修复:pull_order(min-slot 序,链路并发+边到边算)
+ 全员 dispenser(gemm_push_kernel_tp,job_order 就绪序,comp 块跑完 GEMM 加入排空)+
分阶段归因工具 time_tp_stages(一键脚本 step 08)。**等第二轮实测**。注意:TP serial 通信占比
仅 ~23%,重叠天花板 ≈2.2ms,收益结构性低于 EP;大 NE 是相对机会(serial 随 NE 恶化)。
首轮落地记录见 docs/19。
**【FP8 重叠上限·A 证伪+量化 kernel 化 2026-07-12·分支 fp8_tp·当前状态】**
05f 裁决:**fp8 拐点不左移(24 仍单调最优)**——pull 是延迟/并发受限,减字节
不减 RTT,A 刀归档负结果。08f 首份 fp8 归因:full 1931 = sched 275 +
**tok_copy 112(torch 量化链 ~80µs,新头号便宜肉)** + L0_fused 935 + L1 695
+ final_red 20;L1 nb 分解 = 让渡 63 + 真实尾 105;异常记档:gg8 alone 参考在
stages 环境失真(1032 vs 隔离 729,假负暴露),不影响 fused 结论。**本轮:
rowgroup_quant_fp8 单 kernel(预期 112→~35,token 与 P3 act 共用)已落码**。
账:1931 → −75(量化)→ −135(P3 L1 fp8:w2 原布局+qc.w2_scale 原样可用,
act 量化复用,signal_epilogue/push 零改动)→ −80(C sched)≈ **~1640 =
1.26×**。**下一步:`STEPS='^00_|^01_|^03f8_|^04f8_|^08f_' bash
tools/run_tp_all.sh` 验证量化 kernel,过了进 P3**。docs/41。
**【FP8 P2 兑现(历史)】**P2 一次
通:**tktp fp8 T=512 ~1920(1.08× vs serial fp8 2074)、T=1024 3239(1.24×)**,
AG 减半结构优势兑现;数值 rel_err 3.76e-2(serial 1.67e-2 同 FAIL 元素级容差,
我们多二次量化+混合块保守 scale,容差校准待定口径)。用户裁定:fp8 计算收益
大家封顶,**转向重叠上限**。账(docs/40):当前 1920,上限 ~1510(1.37×);
抓手 A fp8 comm_sms 重扫(AG 减半→拐点应 24→~12-16,−40~80)已落 05f;
B L1 fp8(P3,−100~130);C sched 双流+瘦身(−60~100);D final_red 融合
(−20)。stages 工具已 fp8 化(08f,tok_copy 含量化/L0 走 tpdisp8/gg8 参考)。
**下一步:`STEPS='^00_|^01_|^05f_|^08f_' bash tools/run_tp_all.sh` 看拐点+
fp8 归因,然后 P3(L1 fp8)**。docs/40。
**【FP8 P2 落码(历史)】****税坐实**:raw
探针 304/311 TFLOP/s(仅高 7-12%)→ fp8+fp32acc 峰值≈bf16 峰值,消费级硅
blockwise fp8 收益≈1.3×(纯字节红利),f16 累加有 128-K 溢出风险不做——
sm120 一手平台事实归档(docs/39 §1)。**P2 已落码**:tpdisp8 = 源端 1×128
量化(torch, 计时区)+ fp8 AG(行+scale 行两 TMA 一 mbarrier,线上 8KB→
4.125KB,TOKENS_PER_BLOCK 20)+ fp8 dispenser GEMM(a_scales 直读
gathered_scales;权重 dequant→GLU 交织→128×128 重量化,scale 块对齐 tile,
B^T 免转置)+ GLU epilogue 直存 bf16 act;L1/push/combine 全 bf16。scheme
fp8 分支 + 脚本 03f8/04f8。预期 e2e ~1900-1950 vs serial fp8 2074(P3 后
~1.15-1.18×)。⚠️ verify 预期 FAIL 在 fp8 容差(serial fp8 同病,rel_err
1.67e-2;我们多一层重量化差),诊断行已加,同量级即过,容差校准待数据。
**下一步:`STEPS='^00_|^01_|^03f8_|^04f8_|^04f_' bash tools/run_tp_all.sh`**。
docs/39。
**【FP8 P1(历史·税坐实)】**P1 两轮:**正确性一次
全对**(两档形状 rel_err 1.68e-3,scale 行映射/B^T+mma_ABt/重标定全对);TK
平台坑沉淀(col-layout fp8 加载没写完 → B 转置 (E,N,K)+row ldmatrix+mma_ABt,
副产品 w1 免转置)。性能 283 TFLOP/s = 1.29×(预取假说证伪,仅 +1.5%)。
**fp32 累加税假说**(docs/38):GeForce 系 mma.f32.e4m3 指令率减半 → fp8 峰值
≈ bf16 峰值 289,我们已在 98% 税后天花板;f16 累加有 128-K 块溢出风险(RMS
92k>65504)无免费出路。**raw 探针已落码**(grouped_gemm_fp8(...,raw=True) 跳
过重标定测硬上限):raw≈290 → 税坐实,按修订账进 P2(e2e ~1770 ≈ 1.17× vs
serial fp8 2074);raw>350 → 查 FFMA/流水深度。**下一步:
`STEPS='^00_|^01_|^02f_' bash tools/run_tp_all.sh` 看 raw-cap**。docs/38。
**【FP8 P0 定标+P1 落码(历史)】**P0 定标完成:
**serial fp8 = 2074(T=512)/4030(T=1024)**,vllm triton fp8 仅 1.27× 自家
bf16;**serial fp8 已快过我们 bf16 融合(2115)——fp8 是保住领先的必需品**。
我方 fp8 靶:~1500(比率 ~1.38×)。⚠️ serial fp8 verify FAIL(rel_err
1.67e-2 > tol,步骤未失败=会埋 bug):已加失败时打印完整 check(max_abs/
atol/rtol),下轮定位是量化口径差还是真 bug。**P1 已落码待编译验证**:
`gemm_config_fp8`(RED=64 不变,量化块 K=128=2 step,子累加器每 2 step
fp32 重标定;scale 直读 global 不进 smem/不动 TMA expect;行映射 data偶→
lane/4、奇→+8 已从 TK 源码确认)+ `grouped_gemm_sm120_fp8_dispenser` +
`gg8::entry`(binding grouped_gemm_fp8)+ **tools/verify_fp8_gemm.py**
(单卡对拍 fp32 反量化参考 rel<5e-3 + vs bf16 计时,目标 ≥1.8×)。脚本
02f 两档(L0/L1 形状)。**下一步:
`STEPS='^00_|^01_|^02f_' bash tools/run_tp_all.sh` 裁决 P1**;过了进 P2
(源端 token 量化+fp8 AG+L0 集成)。docs/37。
**【FP8 阶段启动(历史)】**bf16 阶段收官
(1.21×/1.33×),15 轮经验整理进 **docs/36**(平台事实/调度结构/协议/
方法论/负结果五类)。fp8 方案:token 1×128 group + weight 128×128 block
(DeepSeek 式);harness 现成度高(Precision.FP8/per_block_cast/vllm w8a8
serial/fp8-aware reference 全在位)。**收益分析(docs/37)**:我方 GEMM 2×
+ AG 字节减半(serial 的 AG 不减半)→ e2e 2116→~1500,预期比率 1.25-1.4×。
关键设计:K-stage 从 64 提到 128(fp8 tile 16KB×2×3 stage=96KB 贴预算,
每 stage 恰好一个 scale 块,per-stage fp32 重标定);GLU 列交织下权重量化
在交织后布局上做(scale 块对齐 tile);push/combine 保持 bf16(协议零
改动)。**P0 已落**:run_tktp --precision fp8 + 脚本 03f/04f serial fp8
基线步。**下一步:FOCUS 跑 P0 定标 serial fp8**
(`STEPS='^00_|^01_|^03f_|^04f_|^04_bench' bash tools/run_tp_all.sh`),
然后 P1(grouped_gemm_sm120_fp8 单卡对拍)。docs/36/37。
**【TP 第十五轮(历史·bf16 收官)】**测量轮裁决:**小预算假设证伪**
(cs_l1 2/4/8/16/24 = 2223/2224/2216/2200/2171,单调反向;08c 显示尾部
wire 全暴露)。**L1 N 维分解定案为负结果,TK_L1 默认回 v1**(e2e 2116)。
根因是平台差异(docs/35):本机 L1 GEMM SM-bound,重叠是 SM 零和,v1 的
"全速 GEMM+全员后排空"已把 wire 藏进高并行尾巴;Comet 的 N 维分解成立的
前提是通信不占 SM(NVLink/copy engine),PCIe+SM 推送平台不成立——对
sm120 移植是一手平台事实。v2 代码/门保留(TK_L1=v2,03l/04l/08l 已翻转为
v2 对照)。stages 新增 L1_gemm_nb(86 块纯算)下轮裁决 L1 是否关账。
**主攻切换:sched ~268(v1 含 job_order)、L0 暴露 ~200、tok_copy 并入
L0**。下一步:FOCUS=1 跑默认回归 + nb 分解。docs/35。
**【TP 第十四轮(历史·测量轮)】**GRP=16 兑现大头(L1_fused 1294→786,
−508µs),cm 探针判换序无罪(+12µs);但 **v2 2174 仍比 v1 2116 差 +58**。
关键洞察(docs/34):L1 GEMM 是 SM-bound → **通算重叠在 SM 维度是零和,真能
藏的只有 PCIe 线上时间;v1 的"GEMM 全速+全员后排空"已接近该结构最优**。v2
的独有翻盘自由度 = 列扫聚合不需要常驻守望者,可把 TK_COMM_SMS_L1 压到 2~8
(v1 砍不得,十一轮已证伪)。本轮零 kernel 改动:05b 扩扫 {2,4,8,16} + 08c
(cs_l1=4 归因)。**下一步(~4 分钟)**:
`STEPS='^00_|^01_|^03_correct_ne64$|^04_bench_tktp_512$|^05b_|^06_tktp_t1024$|^08_time_stages$|^08c_' bash tools/run_tp_all.sh`
裁决树:小预算赢 → v2 定档;wire 喂不满 → 延迟选举/GRP=32;都追不平 → v1
回默认,负结果沉淀,主攻转 sched(233)/L0 暴露(202)。docs/34。
**【TP 第十三轮(历史·GRP 已兑现)】**L1 v2 首测(083157)48/48 正确但**性能
回退 +556µs(2675 vs v1 2119)**:协议对、粒度错——job=1 token×1KB,每 job
一次 TMA+wait 串行化,PCIe 延迟暴露 16384 次,L1_fused 691→1294;sched 删
job_order 兑现 −35(233)。**已修复:GRP=16 组批推送**(j 编号天然目的卡
优先,连续 16 个 j 同卡且行连续 → 一个 job 归约 16×512 列段,背靠背 16 个
TMA 一次 wait,延迟摊薄 16 倍,job 数 16384→1024);新增 `grouped_gemm_cm`
探针(列外层纯算参考),stages 报 cm-rm delta 分离"换序代价 vs 协议代价"。
预期 L1_fused→~600-700,e2e ~2000-2100。教训入 docs/33:**工作粒度 = 能触发
一次高效通信的最小单位(docs/02 §3),1KB/次的 wait 串行是反模式**。
下一步:bash tools/run_tp_all.sh。docs/33。
**【TP 第十二轮(历史·粒度病已修)】**用户点破 + experience/02 §2 印证:L1
combine 的可分解维度是 **N(输出列)**,v1 按 M 分解(job=token,等 max slot)
在 topk=8 下九成 job 拖到 GEMM 尾 ~11% 才解锁——这才是 L1 暴露 186µs 的真身。
已落地 **L1 v2(tppr2,TK_L1=v2 默认/v1 回滚)**:①W2 GEMM 列外层 dispenser
(模板加 COL_MAJOR);②信号按列扫聚合(cb 计数满 nblk → 单信号放行整列全部
token 的 combine,per-job wait 与 job_order 全删,job_order 移出 sched 计时);
③push job=(token,chunk=512列/1KB),chunk-major 从 GEMM ~1/8 进度起流推;
watermark expected×NCHUNKS,final_red 零改动。脚本加 03l/04l/08l(v1 门/A B/
归因对照)。风险:1KB push 的 PCIe 效率(不够就 TK_L1_CHUNK_CB=8)、列外层的
L2 复用变化。**预期 L1 暴露 186→~50-80,e2e 2125→~2000-2050;下一步:
bash tools/run_tp_all.sh**。docs/32。
**【TP 第十一轮(历史)】**45/45 全过,docs/30 三刀兑现:
**主形状 NE=64/T=512 = 2125µs(1.24×),T=1024 = 3781(1.32×),NE=128 1.15×,
NE=256(RB64) 1.10×,T=256 1.04×**。A/B 阶梯:v1 2276 → v2(dispenser+转岗)
2208(−68)→ +GLU 2125(−83);归因:silu 109→6、L0 暴露 255→201(T=1024
477→306,batch 越大回收越多)、GLU store GEMM 级零开销(969 vs 978)。
comm 拐点仍 24 但右翼大幅变平(32: 2412→2186,转岗旁证);**TK_COMM_SMS_L1
减 SM 假设证伪**(8/16 反慢 ~90µs,L1 comm 块是有效推流工,保持 24);
**sched 合并 all_gather 未兑现反 +20µs**(strided copy 倒贴,packed 布局
(2,T,K) 待微修)。剩余账:sched 267 重回最大单项、L1 暴露 186(comm 块
spin 期帮 GEMM 的反向转岗待设计)、L0 暴露 201 刚性。docs/31。
**【TP 第十轮(历史·已兑现)】**用户裁定弃 vLLM 调优线,回归自研
算子。**通信拖慢计算的定量账(docs/30)**:L0 +254µs≈100% 是 SM 让渡(110/86
模型吻合,数据等待≈0)、L1 +190µs 中 142 让渡+48 排空尾;comm_sms U 型 = 静态
折衷,结构解法是转岗。**本轮四项落地(全带回滚开关,42 步脚本已更新)**:
① L0 v2(TK_L0=v2):dispenser GEMM(全局原子发放 + smem 描述环,流水跨 task
连续)+ comm 块常驻分波拉取后 bar.sync 2 转岗加入 GEMM(镜像 L1 的 comp 转岗);
新表 blk_expert host/GPU 双实现+裁决扩展;② SwiGLU 融进 L0 epilogue
(TK_L0_GLU=1):权重列交织 [gate64|up64],fp32 累加器上算 silu*up 直存 act,
省 109µs silu + 75MB 读写,精度更好;风险=group::store 行置换(docs/05),
03g/03o 分档正确性门可二分定位;③ sched 合并 all_gather(ids+权重位打包单
gather,builder 改 packed 输入,位拷贝过 int32 视图);④ TK_COMM_SMS_L1 独立
预算(05b 首扫 8/16)。本地 CPU 预检全绿。**e2e 预期 2260→~1950-2050
(1.28-1.35×);下一步:跑 bash tools/run_tp_all.sh(不用 TUNE)看 03 三档门
+ 04 A/B 阶梯 + 08 归因(silu→0、L0_fused→~1050、sched→~205)**。docs/30。
**【TP 第九轮(历史)】**42/42 全过,TP-T3 v2 无 ray 调优 3 分钟
跑完(vs v1 ray 黑洞 2.5h),09v PASS。**kernel 级增益 3~7%,但 e2e 只有
T=1024 兑现(5025→4883,−142µs),T=256/512 反而 +13/+44µs**——主嫌:vLLM
查表键可能按 M×topk 而非 token 数,三档全部就近命中 8192 键拿到 M=4096 的
赢家 config;次嫌:iters=8 初扫赢家诅咒。已升级 v2.1:**finalize 终审**
(键映射 M=1000 探针自校准、入围 top-6 复审 iters=30、无增益档钉死默认
config、写完走真实查表路径端到端自证,WARN 即 FAIL);run_tp_all 新增
00_reset_tuned_cfg(开跑删旧调优 config,保 04/06/07 未调优口径可比)。
其余档位第三轮连续稳定:t512 2261(1.16×)、t1024 4176(1.17× vs tuned)、
comm 拐点 24 四连庄。**下一步:重跑 TUNE=1 看键映射结论+finalize 终表,
按 tuned serial 报终数;比率压薄再上 sched 第二刀**。docs/29。
**【TP 第八轮(历史)】**39/40(唯一 FAIL=09 调优本体)。
**环境干扰假说裁决=确认(外因)**:两个历史波动档同 session 重复全部一致到 0.1%,
双峰三轮游走+serial 中招+clocks 取证(GPU0 被别人占 59.9GB)收口;报数纪律改为
"med 为准,出双峰当轮重跑取重复一致值"。全档复现第七轮:1.16×/1.20×/1.10×/
1.07×(RB64)/1.01×,comm 拐点 24 三连庄。**TP-T3 v1 尸检:ray 在共享机上卡死
RegisterClient 2.5h 被 SIGTERM,零 trial,serial_tuned≡serial;triton import
报错是良性噪音**。已重写 v2:`tools/tune_moe_tp_noray.py`(无 ray、subprocess
4 卡分片、monkeypatch 注入+自证、smem 预过滤 1920→648),**默认只调主报数形状
E=64/topk8/hidden4096/gateup6144(键 E=64,N=768,~15min;NE sweep 档用
TUNE_E="64 128 256")**,step 09 升级(smoke 先行 + tuned serial t256/512/1024
复测 + 09v 自动裁决)。
**下一步:重跑 `TUNE=1 bash tools/run_tp_all.sh` 拿 tuned serial 报终数**。docs/28。
**【TP 第七轮(历史)】**35/35 全过。**T=1024 异常解除且创最佳比率
(4199 vs 5021 = 1.20×)**;NE=64 2264(1.16×)、NE=128 1.11×。双峰漂移(上轮 t1024→本轮
ne256_rb64,且 serial_t256 也离群)→ **环境干扰假说**(min 恰等稳定轮值);已加取证:每步
clocks 快照 + 波动档同 session 重复跑,下轮裁决。**TP-T3 调优脚本就绪**(tools/
tune_vllm_moe_tp.sh,TP 形状 E=64/N=768;一键脚本 TUNE=1 开 step 09)——报终数前最后
一块。docs/27。
**【TP 第六轮(历史)】**32/32 全过。comm 拐点确认=24(32/40 反降);
**RB64 翻盘 NE=256(4950 vs serial 5287 = 1.07×,padding 归零净赚 477µs)**;首份分阶段
归因落地:GEMM-alone 1486µs(~210TFLOP/s,远快于预估)、L0 暴露 254、L1 暴露 186、
**sched 277µs 是最大可压项(12%)**(已做第一刀:pull 不再重建 push_order);理论地板
~1900µs(1.38×)。当前:NE=64 **2301/1.14×**、NE=128 1.09×、NE=256(RB64) 1.07×。
**新异常:T=1024 双峰(med 10020/min 4210,comm24;comm16 上轮稳定 4325)**→ 脚本已加
06b 对照+两档归因,下轮裁决。终数前必须 TP-T3(triton 调优)。docs/26。
**【TP 第五轮(历史)】**首次全绿并**超 serial**:pull 路径 NE=64/T=512
**2290µs vs serial 2630 = 1.15×**(comm_sms=24,fair 口径;T=1024 1.16×,NE=128 1.05×)。
docs/20 修复兑现(3613→2363,-35%)。push 路径(TP-T1)正确但慢于 pull → **冻结**
(TP 是 GEMM-bound,mb 带宽差不在关键路径;docs/25 §2)。遗留:NE=256 0.97×(50% padding
→ 下轮 RB=64)、T=256 0.98×、comm_sms 拐点未到(默认已提 24,扫到 40)、time_tp_stages
签名 bug 已修(下轮拿 L0/L1 暴露归因)、TP-T3 triton 调优基线未做(报终数前必须)。docs/25。
**【TP 第四轮 2026-07-11】**第四轮死于 gemm_push_kernel_tp 的 barrier 混用 UB
(__syncthreads=bar0@288 与 GEMM consumer group 的 bar0@256 并发混计数 → illegal
instruction),已改专用命名 barrier(bar.sync 2)汇合;同轮落地 **TP-T1 dispatch push 化**
(canonical 布局 + push_order + tppdisp 三角色常驻 kernel + chunk 水位,TK_TP_DISPATCH=push)
与 **EP P1 口径修复**。docs/24。
**【microbench 导入 2026-07-11】**并行工作区(SYNC07101059_2)的 EP microbench 套件
(`microbench/`,mb1~mb7)+ 实测结果(`microbench/results/20260710_071848/`)+ 任务清单
(docs/22,原编号19)已导入。平台事实(pull 并发 23.5GB/s 弱路径 vs push 50.9GB/s 强路径
4SM 打满、comm SM 让渡 8~16%、干扰≈0、vLLM triton 未调优、EP fair 口径 1.34×)直接改写
TP 路线:**dispatch push 化为必选项**,见 docs/23(TP 第三轮计划)。)

> **【EP 上一轮 2026-07-09】** T6-v0 预归约 + T4 gate+up + T3 schedule GPU化(公平口径)
+ T7 dispatch 去重 + T5 ROW_BLOCK=64,五项落地;NE=256 e2e 7131→5830µs 公平口径超 serial
7063 的 1.21×,见 docs/13~16。

## 1. 项目一句话

在 moe_bench 里用 ThunderKittens 实现通算融合的 EP MoE 层(dispatch⊕GEMM、GEMM⊕combine
单 kernel 融合,不做 stream overlap 降级方案),对比 vLLM serial baseline。
平台:16×RTX Pro 5000(sm120)PCIe,用 4 卡组(优先 9,11,13,15),无 NVLink、
**远端原子高并发不可靠**(docs/08)、无 multimem。

## 2. 当前状态(全部已提交,分支 tk_dev)

> **【仓库结构重构 2026-07-11】** ThunderKittens 已改为**正式 git submodule**
> (`.gitmodules`,钉在上游 `02e9acbd`,TK 核心零改动)。原先压在 TK 本地提交里的
> 自研代码 `ThunderKittens/tileoverlap/` 已整体迁至 **`kernels/tileoverlap/`**,
> 由 moe_bench 自己追踪(TK 内留有 `tileoverlap-archive` 分支存档旧提交,仅本地)。
> 路径已同步修正:各 `Makefile` include `../../../ThunderKittens/kernels/common.mk`、
> benchmark 脚本 sys.path 指向 `../../../ThunderKittens/kernels/parallel`、
> `kernels/tk/build.py` 的 `_COMMON` 指向 `kernels/tileoverlap/common/sm120_common.cuh`。
> 新 clone 后需 `git submodule update --init`。**服务器侧
> (`/data/cinnzhang_vllm_td_test/xxy/moe_bench`)尚未做同样的目录调整**,下次上
> 服务器时需同步(迁移目录 + git pull 本仓库),否则两边结构不一致。

> **【默认形状已改 2026-07-09】** 默认 shape 现为 **E=64, TOP_K=8, hidden=4096,
> gate_up=6144(intermediate=3072)**,512 token/rank, bf16 EP world=4(E_local=16,
> 每专家 ~256 token,padding 少)。config.py / configs/tk_ep_bf16.yaml / 各 tool 默认
> 已同步。此 shape 下 fair 口径 **tkfused ~2170µs vs serial 2905µs ≈ 1.34×**
> (P1 修复后 schedule ~205µs 计入默认路径;旧记录 1963µs/1.48× **漏计 sched**,
> 见 docs/22 P1 与 microbench mb3_report_sched205us;对拍 rel_err 4.43e-3 ok)。
> 下面 §2 里 NE=256/hidden=7168 的历史数字是旧 shape 的记录,保留备查。

- **正确性**:tkfused 全链路对拍 reference_moe 通过(bf16, EP, 4 卡, rel_err ~4.4e-3)。
- **性能(新默认 shape E=64/hidden=4096,fair 口径)**:tkfused **~2170µs vs serial
  2905µs(1.34×)**;旧数字 1963µs/1.48× 漏计 sched(docs/22 P1,已修)。v0 prered 为 2331+sched。
- **combine 三条路径**:`TK_COMBINE=prered_push`(默认,T6-v1,docs/18,边算边推)|
  `prered`(v0,barrier+pull,docs/13)| `pull`(旧 moe_gemm_combine_fused)。
- **性能(旧 shape NE=256/hidden=7168,历史记录)**:tkfused 5832µs vs serial 7063µs(1.21×)。
- **combine 两条路径**:`TK_COMBINE=prered`(默认,T6-v0,docs/13,全档位优于 pull)|
  `pull`(旧 moe_gemm_combine_fused,保留)。
- **T4 gate+up 合并**:`TK_FUSE_GATEUP=1`(默认,pull dispatch)。
- **T3 schedule GPU 化**:`TK_GPU_SCHED=1`(默认,pull+prered 路径,CUDA graph,docs/14)。
- **dispatch 三条路径**:`TK_DISPATCH=pull`(默认,已对拍)| `push3`(正确,NE≤128 更快、
  NE=256 更慢,docs/10)| `push`/`push2`(冻结,docs/08)。
- **代码**:`tk_scheme.py`(scheme + host schedule)、`kernels/tk/tk_moe.cu`(gg/disp/
  dpush/dpush3/comb 五个 namespace)、`kernels/tk/sm120_common.cuh`(pcie_sync +
  grouped_gemm_sm120 模板)。
- **工具**:`tools/run_tkfused.py`(对拍)、`time_dispatch.py`(dispatch-only 隔离计时)、
  `time_layer1.py`(T1 layer1 5 点隔离归因)、`ncu_gemm_probe.py`(单卡 ncu)、
  `validate_push3.py` / `verify_push3_schedule.py`(协议裁决)、
  `reconcile_prereduce.py`(T6-v0 预归约表对账,docs/13)、
  `validate_prered.py`(T6-v0 全链路正确性:prered vs pull combine,30 迭代 × 3 NE)、
  `verify_schedule_gpu.py`(T3 GPU schedule vs host golden element-wise,docs/14)、
  `time_schedule.py`(T3 schedule 重算成本:all_gather/eager/graph 分解)、
  `validate_prered_push.py`(T6-v1 push combine 正确性,30 迭代 × 3 NE,docs/18)、
  `analyze_overlap.py`(通算重叠 + 融合损失分析,docs/17/18)、
  `bench_shape_4096.py`(默认 shape tkfused vs serial)。
  ⚠️ `time_layer1.py` / `ncu_gemm_probe.py` 及 `tk_moe.cu` 的 T1 debug 入口
  (`gg::entry_nb`、`comb::combine_only_entry`)在服务器侧,**尚未同步入本仓库提交**。

## 3. 下一步:按 docs/11 路线图执行

**docs/11_评审_layer1归因与下一步优化路线图.md 是当前的任务清单**,按优先级:

| # | 任务 | 状态 |
|---|---|---|
| T1 | ncu 归因 layer1 GEMM 为何只有 71 TFLOP/s(fence/SM让渡/L2) | **✅ 完成(docs/12):假设推翻,慢在 combine gather 94~99%,非 fusion** |
| T2 | combine epilogue 换 push3 式选举信号(预期 layer1 −1ms) | **❄️ 冻结(T1 止损:fence 仅 1.2~2.8%,收益 ≤0.05ms)** |
| T3 | schedule GPU 化并计入 run()(公平性,报数前必须) | **✅ 完成(docs/14):默认路径 6 表 GPU 化 + CUDA graph(~205µs),计入 run();对拍 host golden 全等(NE×{balanced,skewed});e2e NE=256 计入后 5832µs 仍 < serial(1.21×),星号已去** |
| T4 | gate+up 合并一次 GEMM(up 的 1.5ms 藏进 dispatch) | **✅ 完成(docs/13 §6):NE=256 −204µs、NE=64 −478µs,默认开;T5 后 up 才完全隐藏** |
| T5 | ROW_BLOCK=64(NE=256 两层 GEMM 各省一半行) | **✅ 完成(docs/16):编译期 TK_ROW_BLOCK 开关,全档正确;但预期被推翻——padding 本以满效率算,减 padding 后 npl=4096 落小 tile 低效区(143→75 TFLOP/s),W2 wash。RB=64 仅 NE=256 净赢 ~260µs,NE≤128 变慢。默认保持 128。大收益需 8warp×8行/fp8 计算** |
| T6 | combine 预归约 + push 化(A' 镜像,layer1 通信 4×) | **✅ v0+v1 完成:v0 预归约(docs/13)+ v1 push 化(docs/18,边算边推消零重叠)。默认 prered_push;layer1 融合损失 283→192µs,e2e(默认shape)2331→1963µs(1.48× serial)** |
| T7 | dispatch (token,dst) 去重 + fp8 传输 | **去重✅(docs/15):v0 落地默认关(TK_DEDUP=1);NE≤128 大赢(e2e NE=64 −0.47ms),NE=256 无收益因 dispatch 实为 GEMM-bound(账本修正);与 T5 协同后 NE=256 生效。fp8 未做** |
| T8 | push3 目的块重排 + 水位信号(NE=256 翻盘后设默认) | 未开始 |
| T9 | 杂项:AGENTS.md 口径、skewed 测试、comm_sms 扫参、probe 增强 | 未开始 |

核心评审结论(细节在 docs/11 §0,**T1 实测修正见 docs/12**):
1. **layer1 是最大的矿,但慢的不是 GEMM 而是 combine gather**:T1 实测推翻 §0.1(a)——
   W2 GEMM 带 fence epilogue 仍跑 ~140 TFLOP/s,"71 TFLOP/s" 是伪指标;真正的 94~99%
   在 combine 每 token 拉 8 条散布 ~14KB peer 行(58.7MB @ ~32GB/s,PCIe 弱路径顶)。
   → T2 冻结,**T6(combine 预归约+push)是 layer1 唯一主攻**;
2. dispatch pull 已贴 20GB/s 带宽顶,剩余杠杆全在减字节(去重 2.2×、fp8 2×)和换强路径;
3. push3 在 NE=256 退化的补充归因:push 按源 token 序,目的块就绪全部后置 → 重排解决;
   信号数 ∝ 行块数 → 水位聚合解决;
4. 总账目标:全部落地后 e2e ~2.4ms vs serial 7ms。

## 4. 踩坑索引(改代码前必读)

- **`kernels/tk/sm120_common.cuh` 是构建产物,不是源文件**(build.py 每次编译
  前从正本 `kernels/tileoverlap/common/sm120_common.cuh` 覆盖拷贝,且被
  .gitignore)。改 GEMM 模板必须改正本;改了副本 = 不进 git + 下次编译被覆盖
  (第十轮首跑 01_build 就死在这,符号未定义)。

- **远端原子会丢增量**(高并发散射 red.add,probe 弱压力测不出)→ docs/08。
  跨卡完成检测只用"本地 atom.acq_rel 选举 + 单写者 st.release.sys"两阶段协议。
- **worker 里 set_default_device(cuda) 劫持无显式 device 的张量创建**(host 调度表
  必须每处写 device="cpu",本地 CPU 测试测不出)→ docs/21 §1。
- **persistent kernel 角色混跑先盘点 named barrier**(__syncthreads 就是 bar0;GEMM
  consumer group 按 256 计数,全块 288 计数并发混用同一 barrier = UB/illegal
  instruction)→ docs/24 §1。
- **block-per-job 是反模式**(大 smem 下按块调度粒度串行排空)→ 常驻块 + 原子
  dispenser + 就绪序;拉取/到达序要对齐**消费序**而不是源的远近 → docs/21 §4、docs/20。
- **group::store 行映射置换**(warpgroup 交织,改 CONSUMER_WARPS 数必重推)→ docs/05。
- **新方案必须全档位扫 NE∈{64,128,256}**,单点结论会误导 → docs/10 §7。
- **协议正确性靠隔离裁决**(xx-only debug 入口 + 30 迭代 + 双重对账),
  不靠"跑跑看对不对" → docs/08、docs/10。
- 内存序 cumulativity 链仅本机验证过,换卡先跑 validate_push3 类裁决。

## 5. 环境与运行速查

```bash
source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
cd /data/cinnzhang_vllm_td_test/xxy          # 必须在上级目录跑 -m moe_bench.*
nvidia-smi                                    # 跑前确认卡空闲
# 对拍(全链路 vs reference_moe)
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.run_tkfused 64
# benchmark(bf16 EP;--distributed 必带,--scheme 单值,分别跑 tkfused / serial)
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.bench --distributed --scheme tkfused \
    --mode ep --precision bf16 --world-size 4 --no-verify --num-tokens 512
# dispatch-only 隔离计时
CUDA_VISIBLE_DEVICES=9,11,13,15 TK_DISPATCH=push3 python -m moe_bench.tools.time_dispatch 256 10 50
```

**默认路径 = pull dispatch + prered combine + fused gate+up + GPU schedule(计入 run)。**
环境开关速查:
- `TK_COMBINE` = `prered_push`(默认,T6-v1)| `prered`(v0)| `pull`(旧 combine)
- `TK_FUSE_GATEUP` = `1`(默认,T4)| `0`
- `TK_GPU_SCHED` = `1`(默认,T3,CUDA graph)| `0`(schedule 回 setup 不计时)
- `TK_DEDUP` = `0`(默认)| `1`(T7,NE≤128 赢、NE=256 无收益)
- `TK_ROW_BLOCK` = `128`(默认)| `64`(T5,仅 NE=256 净赢 ~260µs,NE≤128 变慢)
- `TK_DISPATCH` = `pull`(默认)| `push3`/`push2`/`push`(实验/冻结)

## 6. 文档索引

| 文档 | 内容 |
|---|---|
| docs/01~06 | Phase 0~4:probe、grouped GEMM、dispatch⊕GEMM、combine、scheme 接入 |
| docs/05 | group::store 行映射 bug(最深的坑) |
| docs/07 | 第一轮评审:公平性 P1~P5 + 优化方向 #1~#6 |
| docs/08 | push 冻结:远端原子丢增量归因 + push2 |
| docs/09/10 | push3 规划 / 实现验证实测(NE 交叉点) |
| **docs/11** | **第二轮评审 + 当前任务清单(T1~T9,做事看这篇)** |
| **docs/12** | **T1 归因实测:layer1 慢在 combine gather(94~99%),非 fusion;T2 冻结→T6** |
| **docs/13** | **T6-v0:combine 预归约 host 表设计(按 expert 卡重分组,等价)+ 双向对账工具 + kernel 落地 + T4 gate+up 合并** |
| **docs/14** | **T3:schedule GPU 化(单 argsort ring-order + 稠密 job 空间)+ CUDA graph 捕获,计入 run() 公平口径** |
| **docs/15** | **T7:dispatch 去重(稠密 staging,gathered 逐字节等价);揭示 NE=256 dispatch 是 GEMM-bound(账本修正)** |
| **docs/16** | **T5:ROW_BLOCK=64 编译期开关;揭示 padding 非纯浪费(满效率算),小 tile 效率折损抵消,默认保持 128** |
| **docs/17** | **通算重叠与融合损失分析:layer0 融合 +30%、layer1 +56% 且通信零重叠(T6-v1 目标)** |
| **docs/18** | **T6-v1:combine 预归约 push 化(边算边推 + 水位选举,消 barrier+零重叠);layer1 融合损失 283→192µs,默认** |
| **docs/19** | **TP 版 tile overlap(tktp,分支 tp_test):AG⊕gate+up GEMM + 本地prered⊕稠密RS push;复用矩阵/调度表/账/上机风险清单/一键脚本** |
| **docs/20** | **TP 首轮实测归因(0.73× serial:L0 ring 序拉取 + L1 job 块排队)与修复(pull_order/全员 dispenser/time_tp_stages);修复后预期账与天花板提醒** |
| **docs/21** | **经验:set_default_device 坑(host 表显式 device 纪律)、远程 zip 回流工程实践(timeout/可归因失败/干净编译/sweep 顺手带)、无 profiler 归因三条账、可泛化调度教训** |
| **docs/22** | **(导入)EP 第三轮计划:microbench 归因(收益分解/平台事实表/P0 口径修复/T10~T15);§0 平台事实两线共享** |
| **docs/23** | **TP 第三轮计划:mb 事实映射到 TP(pull 弱路径→push 化必选 TP-T1、comm SM 自适应、triton 调优、copy engine 远期);修正预期与报数规范** |
| **docs/24** | **TP 第四轮:barrier 混用 UB(bar0 混计数→illegal instruction)修复(专用命名 barrier);TP-T1 push 化落地(canonical 布局/push_order/tppdisp 三角色/chunk 水位);EP P1 口径修复** |
| **docs/25** | **TP 第五轮:首次超 serial(1.11~1.15×,docs/20 修复兑现-35%);push 冻结归因(GEMM-bound+scatter 粒度);NE=256 padding→RB64、comm_sms 拐点、预期账对数与微基准外推教训** |
| **docs/26** | **TP 第六轮:comm 拐点=24 确认、RB64 翻盘 NE=256(1.07×)、首份分阶段归因(GEMM 1486µs/sched 277 最大可压项/理论地板 1900µs);T=1024 双峰异常待裁决** |
| **docs/27** | **TP 第七轮:T=1024 解除(1.20× 最佳)、双峰漂移+serial 离群→环境干扰假说与取证(clocks_per_step/重复跑);TP-T3 调优脚本就绪(TUNE=1);报数纪律(波动档看 min+重复一致性)** |
| **docs/28** | **TP 第八轮:环境假说裁决=确认(外因,重复跑全一致+clocks 取证);TP-T3 v1 尸检(ray 卡死 RegisterClient 2.5h 零 trial)与 v2 无 ray 重写(subprocess 分片/注入自证/smem 预过滤/E 三档);step 09 全网格+自动裁决** |
| **docs/29** | **TP 第九轮:TP-T3 v2 跑通(kernel 级 3~7%)但 e2e 收益错位(仅 T=1024 兑现,主嫌查表键 M×topk 错位/次嫌赢家诅咒);v2.1 finalize 终审(键探针自校准+复审+钉死默认+端到端自证);00_reset_tuned_cfg 保口径可比** |
| **docs/30** | **TP 第十轮:通信拖慢计算定量账(L0 +254µs=纯 SM 让渡/L1 +190 中 48 是协议尾);L0 v2 dispenser+comm 转岗、SwiGLU 融合(列交织+fp32 epilogue)、sched 单 all_gather、L1 独立 comm 预算;回滚开关 TK_L0/TK_L0_GLU/TK_COMM_SMS_L1** |
| **docs/31** | **TP 第十一轮:三刀兑现 1.24×/1.32×;A/B 阶梯定价(转岗 −68/GLU −83);comm 曲线右翼变平=转岗旁证;L1 减 SM 证伪;sched 合并反 +20µs(strided copy 账);剩余:sched 第二刀、L1 反向转岗** |
| **docs/32** | **TP 第十二轮计划:L1 combine 按 N 维分解(Comet layer1-N;M 维九成 job 拖到 GEMM 尾的结构病);tppr2 = 列外层 dispenser + 列扫聚合信号 + (token,chunk) push;协议净简化(删 per-job wait/job_order);风险=1KB push 效率与 L2 复用** |
| **docs/33** | **TP 第十三轮:L1 v2 首测回退归因(job=1token×1KB 的 wait 串行,延迟暴露 16384 次,L1_fused 691→1294;sched −35 兑现)与 GRP=16 组批修复(同目的卡连续行,16 TMA 一次 wait);grouped_gemm_cm 探针分离换序/协议代价** |
| **docs/34** | **TP 第十四轮:GRP 兑现(786)、换序无罪(+12);L1 零和洞察(GEMM SM-bound → 重叠只藏得住 wire,v1 全员后排空近最优);v2 翻盘自由度=压小 TK_COMM_SMS_L1(列扫聚合无需守望者);测量轮 05b{2,4,8,16}+08c 与裁决树** |
| **docs/35** | **TP 第十五轮:小预算证伪(单调反向)→ L1 N 维分解定案负结果,v1 回默认;平台差异沉淀(Comet-N 成立前提=通信不占 SM,PCIe+SM 推送平台零和);L1_gemm_nb 探针;主攻切 sched/L0/tok_copy** |
| **docs/36** | **bf16 阶段经验总结(15 轮):平台事实/调度结构(消费序对齐、转岗、GLU 融合、组批粒度)/PCIe 协议三件套/方法论(公平口径、归因探针、A/B 阶梯、FOCUS)/负结果(Comet-N、push、sched 合并)** |
| **docs/37** | **FP8 路径收益分析与计划(分支 fp8_tp):token 1×128 + weight 128×128;e2e 预估 ~1500(1.25-1.4×);K-stage=128 对齐 scale 块、交织后量化、push/combine 保 bf16;P0 定标→P1 GEMM→P2 L0→P3 L1→P4 调优** |
| experience/ | 12 篇相关工作与平台经验(01 总览、12 SM120/PCIe 适配最常用) |
| blogs/ | 教学博客系列(6 篇, Astro 格式):TK 融合算子教程 + 本仓库实现细节 + 优化经验, 面向入门读者 |
