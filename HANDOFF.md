# HANDOFF — TK 通算融合 MoE 进度交接

> 这份文档只记"接手要知道的当前状态"。原理与账在 [`docs/`](docs/README.md)，
> 历史过程在 git（分支 `fp8_tp` / `tk_dev` 及其提交信息）。

## 2026-07-28：每次运行打印每 expert 的 token 数（工具改动，不改任何结论）

新增 `moe_bench/routing_stats.py`（约 70 行）。三个入口（`bench.py` 单卡 /
`distributed.py` 分布式 / `tools/time_tp_stages.py`）在 `setup()` 之后、计时之前
各打一行：

```
[routing] 每 expert token 数 (tktp, E=64, T=512/rank × 4):
  [264, 249, 260, 254, 286, ... 共 E 个 ...]
```

- 数的是 **assignment**（不是 token），合计恒为 `world × T × TopK`；分布式下取
  **全局**口径（一次 all-reduce），与 `tk_tp_scheme` 建表的 `bincount(all_topk)` 同源。
- 关闭：`MOE_BENCH_ROUTING_STATS=0`。打印和那次 all-reduce 都在计时区外，
  不进任何测量。开关必须对所有 rank 一致（worker 继承父进程环境，天然满足）。
- 详见 [docs/05 §7](docs/05_测试与调试指南.md)。

只碰这三个入口 + 新文件，`schemes.py` / `baseline.py` / `tk_tp_scheme.py` /
kernel 代码零改动。**GPU 上没跑过**（开发机无 CUDA/vLLM），但改动面只有一行打印，
上机跑任意一条既有命令即可确认。

> 本条的第一版还打了 BLOCK 配置（每 expert 块数 / padding / 尾块，以及一个
> `block_config()` 接口 + vLLM triton config 探测），因为探测失败时会多打一行
> 告警、看着像报错，已按用户要求整段撤掉。要重新捞的话在 `9aa7dac`。

## 最新（2026-07-27 深夜 11）：uniform 归因**已实测坐实** + EPIRED 二次判负（docs/15 §4）

四档分阶段实测（卡组 0-3，iters=20）把 docs/15 的假说钉死了，且比模型预测更彻底：

| T=4096 | balanced | uniform | Δ |
|---|---:|---:|---:|
| L0_gemm_alone | 5318.0 | 5061.1 | **−256.9** |
| L1_gemm_alone | 2947.2 | 2807.6 | **−139.6** |
| L0 exposure | 405.8 | 644.2 | +238.4 |
| **L1 exposure** | **1686.6** | **4095.4** | **+2408.8** |
| full_run | 10920.6 | 13157.3 | +2236.7 |

- **padding 税在 T=4096 是负的（−397µs）**：P 涨 3.3% 但两级 tile 把尾块压到
  0.6× 后倒赚。T=512 时它还值 +115µs（30%），T=4096 已经反号——**docs/09 的
  "padding 粒度税"归因不能外推到大 token**。
- **内存局部性假说被排除**：若访存变差 `gemm_alone` 首先会变慢，而它反而快了。
  随机性只作用在 scatter 的写与预归约的读上，即 exposure 里。
- **劣化 100% 落在 L1**（+2409 vs full_run +2237），T=512 也有 46% 在这里。

**EPIRED 二次判负（更彻底）**：uniform T=4096 下 `TK_L1_EPIRED=1` 使 L1_fused
6857→9223（**+2366µs**），比 balanced 口径的 +108~193µs 严重一个数量级。机制是
同一根因的第二个受害者：一个 token 的 8 份 `red.add` 落在同一 cache line 上，
balanced 下 8 个行块被相近时间处理（时间聚集 → L2 命中），uniform 下散布在整个
GEMM（每次都重新拉线）→ 268MB × 8 ≈ 2.1GB 读改写。**docs/04 的判负在不均匀路由
下依然成立且更强，重开路线关闭。**

**模型更正**：release-time 模型只算 wire（100MB/1978µs）预测 +921µs，实测 +2409µs。
缺口在访存侧——L1 尾部堆的大头是**预归约的 1.07GB `expert_out` 读**（`S×TopK×H×2B`），
不是 wire。压进 GEMM 最后 8% 的窗口需要 4.8TB/s，物理不可能，只能拖长 L1。
**因此"combine 改 FP8"要降级**：它减的是 wire 那 100MB，瓶颈却在 1.07GB 的读上。

**下一步 = 先拆 4053µs，再动手（docs/15 §5，两个探针都已就绪）**：
① 新增 `TK_L1_NOGATE`（push_job 跳过 8 个 slot 的行块等待，**数值错**，只读时间）
→ 差值 = 就绪塌缩的代价。接近 2400µs 说明是依赖结构问题；只有几百 µs 说明 1.07GB
读量本身才是瓶颈，杠杆要转向"怎么少读"。② `TK_COMM_SMS_L1` 扫 12/24/48/64 分离
让渡与 comm 吞吐。**拆分结果出来之前不改 kernel。**

**结构性观察（新的 cmep 立项理由）**：TP 下每卡持全部 E 个 expert，combine 必须等
**全部 8 个** → 塌缩是 `f⁸`；EP 下每卡只有 16 个 expert，一个 token 平均只有 **2 个**
落在本卡 → `f²`（f=0.8 时 0.64 vs 0.17），预归约读量也只有 2/8。**随机路由越打散，
EP 相对 TP 的这项结构优势越大** —— docs/13 的 cmep 因此不只是"复刻 COMET 原生
场景"，而是 TP 有一个 EP 没有的结构性瓶颈。

**附带发现（未立项）**：T=512 uniform 的 sched 227→300µs，是融合 sched kernel 的
smem 需求 `4·P+…` 在 P=19712 时越过 99KB 阈值 → 自动回退 torch 版。**一进不均匀
路由就掉出快路径**，值 ~73µs。

## 2026-07-27 深夜 10：uniform 劣化的归因被推翻 —— 不是 padding 税（docs/15）

用户实测：uniform 下 T=512 劣化 +374µs（24.4%）、**T=4096 劣化 +2134µs（19.5%）**，
而 serial 只有 +3.9% / +0.7%。反常点：token 放大 8 倍后我们的劣化几乎没摊薄。

**padding 税已出局**（实算，两级 tile 口径 vs 无 padding 理想）：uniform T=512 是
14.5%，**T=4096 只有 1.6%**（μ=2048、σ≈45，相对抖动 2.2%）≈ 140µs，解释不了 2134µs。
docs/09 的"padding 粒度税"归因只对 T=512 成立，**不能外推**。

**新根因：TopK 依赖让 combine 的就绪时刻塌缩到尾部。** L1 的 job 必须等它全部 8 个
slot 的 W2 输出都算完，而 GEMM 按行块 id 升序推进 → job 就绪进度 ≈ max(8 个 expert
id)/E。要命的是 **balanced 的 round-robin 生成器给每个 token 的是连续的 8 个
expert**（`(t·9+[0..8))%64`，max=min+7），job 就绪均匀线性铺开、push 全程铺满；
uniform 下 8 个 expert 随机散布，max 的期望 56.5/64，**87.9% 的 token 的 max
expert ≥ 50** → 几乎所有 job 都在 GEMM 90% 之后才就绪。

| 路由 | T | 50% job 就绪于 GEMM 进度 | push 尾部暴露（release-time 下界） |
|---|---:|---:|---:|
| balanced | 4096 | 60.9% | 204µs |
| **uniform** | 4096 | **92.1%** | **1125µs** |

即 uniform 多出 ~921µs 纯尾部暴露 vs padding 的 140µs。**serial 不受影响**是因为
它本就串行、没有流水可破坏——**我们相对 serial 的全部优势来自流水重叠，而随机路由
恰好摧毁了流水的前提**。这也是结构性的：处理了 f 比例的 expert 后一个 token 全部
8 个 expert 都就绪的概率是 f⁸，**改 job_order / 行块顺序都无解**。

**模型只解释了一半**（140+921+55 ≈ 1116 vs 实测 2134），剩余 ~1000µs 最大嫌疑是
**内存局部性**（uniform 下 8 个 slot 随机散布在 537MB，scatter 的 TMA store 和预
归约的 8 行读全退化成随机访问）——**未验证，不要当结论**。

**下一步（先验证再动手，docs/15 §6）**：跑
`time_tp_stages 64 20 {512,4096} --dist {balanced,uniform}` 读分阶段数字：
① 若 `L1 exposure` 在 uniform 下暴涨（T=4096 预期 +900µs 量级）→ 根因坐实，按
docs/15 §5 的杠杆序动手；② 若是 `L*_gemm_alone` 自己涨了 → 主因是访存局部性，
转 NCU 做访存归因，杠杆完全不同。**分不清之前不要改 kernel。**

杠杆排序（估算，待校准）：① **combine partial 改 FP8**（尾部是 wire-bound，流量
减半 → T=4096 估算省 ~850µs）；② **重开 EPIRED**（docs/04 的判负口径是 balanced，
那里 expert_out 重读是顺序的、job 就绪本就均匀，uniform 下收益结构完全不同）；
③ 分批 push **已算不可行**（wire 余量只有 1.41×，流量翻倍反而更差）；
④ L0 的 `TK_LOCAL_FIRST` 只值 13~99µs，**优先级降到 L1 之后**。

## 2026-07-27 深夜 9：分支 `ag_local_overlap` —— L0 本地优先分段已实现（待上机）

目标：让**本地已有、不用通信就能拿到的 token** 的 W1 GEMM 不等 AllGather。
现状里没有任何行块是纯本地的——canonical 布局下 expert e 的 256 行是
`[r0 64][r1 64] | [r2 64][r3 64]`，每个 128 行块都混了两个 src_dev，于是本卡
那 64 行虽然 scatter 时走 `src==dev_idx` 直读 `pre_tokens`（连 staging 都不落），
仍要陪着另外 64 行等 peer 的到达 flag。

**做法（docs/14）**：gathered 布局改 rank-dependent 两段式——领头的行块只装本
rank 自己的 assignment，其后是其余一切。dispenser 按行块 id 升序发任务，于是
**头 `nb_local×col_blocks` 个 GEMM 任务完全不依赖通信**。开关 `TK_LOCAL_FIRST`
（默认 0，零风险接入）。

**⚠️ 口径变更（用户 2026-07-27 指示）**：此后一律以**路由不均匀**为准
（uniform/skewed），balanced 只作完美对齐的上界参照。这直接改变了本方案的设计：
均匀路由下每个 expert 的分段代价相同（要么全免费要么全收费），不均匀路由下代价
沿 expert 散开，于是**"只分免费的那批 expert"成为可能**。分段决策因此改成
**逐 expert 按边际块成本**（`_local_seg_mask`，阈值 `TK_LOCAL_SEG_THR5`，默认 0
= 严格帕累托）。实测（真实路由生成器 + 真实调度表）：

| 路由 | T | thr5=0 代价 | 零等待工作量 | thr5=99(全分段)代价 |
|---|---:|---:|---:|---:|
| uniform | 512 | **+0.0%** | 7.5% | +13.2% |
| uniform | 1024 | **+0.0%** | 14.2% | +6.3% |
| skewed | 512 | **+0.0%** | 9.0% | +15.7% |
| skewed | 1024 | **+0.0%** | 16.6% | +5.8% |

即 **thr5=0 在四个不均匀档全部零代价，同时白拿 7.5%~16.6% 的零等待工作量**——
严格的帕累托改进，即使 gate 等待很小也不可能亏。thr5=1/2 是可扫的收益-代价曲线
（skewed T=1024：+0.8% 买到 19.2%）。全分段（原朴素方案）不应作默认。

关键前提均已核实：① dispenser 的行块→expert 完全走 `blk_expert` 查表，没有
"同 expert 行块必须连续"的约束；② `padded` 的**值**从未被任何 kernel 读（只用
`.size(0)` 取 E）；③ P2 push 数据面允许 rank-dependent 布局——push 落点是 staging
的 `src_dev*T+src_tok` 行、与 gathered 布局无关，scatter 读**本地** `tp_slots`；
④ `pull_order` 按 min slot 排 → 本地行自动排最前，公式不用改。

**`push_order` 换了排序键**（这是选择性分段暴露出来的）：原键"本地表的 min slot"
在选择性分段下失效——本卡 token 可能落进段 1（行块排最前），本地 min slot 序就
不再等于目的卡看到的消费序。改用 **(min expert, src_tok)**，与建表的卡无关，
canonical 下与旧键**逐位等价**（preflight 有专门的等价断言 + 步 1 回归门实测）。
正确性本不依赖它（push 落点/flag 都由 `src_dev*T+src_tok` 定死），但"源按目的卡
消费序推"是流水启发式。

**新增探针 `TK_L0_NOGATE=1`**：L0 GEMM 跳过行块到达等待（**输出数值是错的**，
只读时间），`L0_fused − L0_fused@NOGATE` = gate 等待的真实成本。由于 thr5=0 档
代价实测为零，探针不再是"要不要做"的关卡，而是"能拿回多少"的标尺；若它本身
接近 0，说明 min-slot pull 序已把等待消化掉，应转去扫 `TK_COMM_SMS`。

- **死锁审计（红线）**：**无新增、无修改等待点**。段 1 的行块计数由本卡 scatter
  直接喂（`src==dev_idx` 分支根本不进 flag 自旋），段 2 与现状逐字同源；
  `dispatch_gate_p` 的 `PCIE_SPIN_GUARD`、scatter 的 5e8 自旋上界、`guarded_wait`、
  `pcie_barrier_all` 全部原样保留。跨 rank 依赖链仍是 `GEMM ← scatter ← flag ←
  peer push`，无环。NOGATE 探针只**移除**一个等待点。逐条表见 docs/14 §9。
- **正确性强判据**：布局重排不改变任何一行的运算序（每行 GEMM 独立、K 维累加序
  不变；L1 预归约按 kpos 顺序；final reduce 按 rank 顺序）→ **`TK_LOCAL_FIRST`
  on/off 的 e2e rel_err 必须逐位一致**（0.042817506939172745）。不一致即 bug。
- **已在本地验证**：`preflight_tp_cpu` 全绿，6 个布局档（canonical / lf×thr5 ∈
  {0,2,99}）× balanced/skewed × NE=64/256 × rank 0/3 —— host golden ↔ torch GPU
  版逐元素一致，外加四条新不变量：① "blk_expert == 该行块每条 assignment 的
  expert"（语义判据，取代 canonical 专用的单调性断言）；② **thr5=0 的成本不得
  高于 canonical**（帕累托承诺的机器化）；③ 段 1 的行块一条远端行都不能有
  （零等待的充要条件，按行块表述——段 2 里可能偶然出现纯本地块）；④ canonical
  下新 `push_order` 键与旧 min-slot 序逐位等价。CPU 数据流仿真改为按行块走，
  跑 canonical / thr5=0 / thr5=99 三档。
- **口径修正**：`time_tp_stages` 的 `L0/L1_gemm_alone` 参考现在跟随
  `TK_TWO_LEVEL` 传 slack——否则参考走全满块而 fused 走尾块，exposure 被系统性
  低估。balanced 下 slack 全 0，行为与旧口径逐指令相同，**历史数字不受影响**。
- **未做（留 Phase 2）**：① 融合 sched kernel（tpsched）只实现了 canonical 单段
  compaction，`TK_LOCAL_FIRST=1` 时自动回退 torch 版（sched +~120µs）→ **裁决
  看 time_tp_stages 分阶段数字，不看 e2e**；② docs/14 §4.1 的 L1 风险：本地 job
  集中在段 1，跨卡 RS 的可用窗口按段 1 占比缩水（thr5=0 下余量 1.35~1.5×，比
  全分段的 1.13× 安全得多，但仍要盯 `L1_fused`），解法是给 dispenser 加可选
  `row_perm` 行块访问序表让 L1 先算段 2。

上机 runbook 见 [docs/14 §7](docs/14_L0本地优先分段与AG重叠.md)（先确认卡空闲；
步 3 是 local_first 布局首测，单步隔离）。

## 2026-07-27 深夜 8：COMET 复刻可行性裁决 + cmep（EP 形态）实现计划已立（docs/13）

对照 `docs/paper_row/` 两篇论文裁决：**可以用本仓库 PK 原语层
（pcie_sync + TKParallelTensor + tma_cta + gg8 dispenser）实现 COMET
风格 MoE 层**，计划全文见 [`docs/13`](docs/13_COMET_EP复刻实现计划.md)。
要点：① tktp 已是 COMET 两大机制（线程块特化 + shared tensor 依赖解析）
的 TP 形态落地，增量价值 = **EP 形态新 scheme `cmep`**（dispatch/combine
由路由驱动，COMET 原生场景）；② 三处平台强制偏离有既有判负背书——
NVSHMEM/UVA→P2 源端 push、L1 N 维分解→M 维行块预归约（Comet-N 判负）、
预编译 n_c 库→运行时 CM_COMM_SMS；③ 五阶段 M0-M5（EP 基线→host 调度
golden/preflight→L0 单步隔离首测→L1+故障注入→定标→论文三组复刻实验），
死锁审计表已预填。**未动任何代码**，下一步从 M0（EP serial 基线 +
gg8 EP 形状探路）开始。

## 2026-07-27 深夜 7：Phase 1b 验收通过 —— uniform e2e −45µs，正确性逐位等价

四步全过（卡组 0-3，docs/09 §6）：① balanced 回归 rel_err 逐位一致 +
e2e 1530.9 零回退；② uniform 下 two_level on/off 四 rank rel_err/
max_abs_err **全部逐位相同**（尾块逐比特等价在真实路由坐实）；③
uniform e2e **1903.1 vs 1946.0 = −43~48µs**（劣化 415→372µs）；④ ptxas
全零 spill。低于预估 100-180 的账：真实 uniform 尾块占比仅 ~20%（探针
33%）+ L0 省时部分被 AG 重叠窗口 comm-bound 吃掉。**两级 tile 全线收官，
TK_TWO_LEVEL 默认开**。残余方向（未立项）：uniform 口径重扫
TK_COMM_SMS；64 粒度 act 布局。正式报数建议 balanced+uniform 双口径。

## 2026-07-27 深夜 6：两级 tile Phase 1b 已接线（验收见上节）

v3 定案后（L0 −12.2%/L1 −9.9%，尾块成本 0.63/0.70）Phase 1b 接入 fused
两层。**关键简化：blk_rows ≡ ROW_BLOCK − slack**，调度表的 slack 已有
golden/tpsched 逐比特对拍且每迭代重建 → 调度链零改动。dispenser 参数
语义改为 blk_slack（tail ⟺ slack ≥ 64）；L0 入口本就接收 slack（barrier
重置用），只多传给 dispenser；L1 入口新增 slack + two_level 参数；两层
globals 的 A-源 gl 挂 A_tail_tile 描述符（A64 装载自动启用）。开关
`TK_TWO_LEVEL`（默认 1）；EPIRED 与 two_level 互斥（C++ TORCH_CHECK +
scheme 自动降级）。探针/time_tp_stages 调用同步。

- **死锁审计（红线）**：无新增等待点。gate/barrier 语义不变（barrier 初
  值=slack、放行阈=ROW_BLOCK，与尾块无关）；epilogue/行块信号计数不变；
  expect_bytes 尾块 16KB 与 A64+B 装载严格相等，producer/consumer tail
  判定同源只读 slack。尾块上半 act/expert_out 行保持 stale——与既有
  padding 行同语义（slot_job=-1 不被消费，fp8 NaN 类风险与现状同类）。
- **正确性强判据**：真实行的计算序列与满块逐指令相同 → balanced 与
  uniform 的输出都应与 TK_TWO_LEVEL=0 **逐位一致**。

上机 runbook（先确认卡空闲；步 2 是尾块首次实战，单步隔离）：

```bash
cd /workspace
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4 2>&1 | grep -E 'registers|spill'
# 1) balanced 回归门(无尾块): rel_err 须与基线逐位一致(0.042817...)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10
# 2) uniform 正确性门(首次真尾块): 与 two_level=0 的 rel_err 逐位对照
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10 --dist uniform
TK_TWO_LEVEL=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10 --dist uniform
# 3) uniform e2e A/B: on 预期比 off 快 ~100-180µs(off ≈ 历史 1988µs 水位)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --no-verify --dist uniform --iters 50
TK_TWO_LEVEL=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --no-verify --dist uniform --iters 50
# 4) balanced e2e 零回退确认
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --no-verify
```

## 2026-07-27 深夜 5：v2 A64 实测 −8.9%/−6.7%，v3 修 SMSP 聚集（已验，v3 定案见 docs/09）

v2（A64 装载）实测（GPU0）：L0 −8.9% / L1 −6.7%（v1 −6.7/−5.0），
正确性全 OK、默认路径无回退。尾块成本 ~0.73-0.80 仍距 0.57，残差主因
定位：active warp 沿 store 交织映射选出 {0,4,1,5} → warp_id%4 聚在
SMSP {0,0,1,1}，**半数 tensor 单元闲置**（RB64 的 4 warp 恰铺满 4 个
SMSP 故 0.57）。**v3 已提交**：尾块改恒等 strip 映射（warp 0-3 →
条带 0-3，SMSP 全铺满；尾块 store 单 warp 版不受 group 交织约束，
满块路径逐指令不变）。预期尾块成本 → ~0.6，tail=32 探针 → −11~13%。
复测同款三条命令；之后进 Phase 1b（tpsched 发尾块 + L0 fused 挂
A_tail 描述符 → uniform e2e 报数）。

## 2026-07-27 深夜 4：Phase 1a 首测通过（−6.7%/−5.0%）+ A64 装载 v2（已验，见上节）

Phase 1a 首测（GPU0）：**正确性全过**（默认路径无回退 660.5/359.5 +
rel_err 1.68e-03；tail=32 探针 full/2level 双双 OK；四卡回归 rel_err
逐位一致）。性能 v1 = L0 −6.7% / L1 −5.0%（预期 −14/−13）→ 尾块成本
~0.80× 而非 0.57×，归因 = v1 producer 仍装满 128 行 A tile，尾块 stage
被 TMA 喂料托底。**v2 已提交**：A64 装载——TK st 布局等价性（128 列
fp8 → swizzle_bytes=128 单 panel → st<128,128> 前 64 行与 st<64,128>
逐字节同）使 consumer 零改动；producer 对尾块 expect/装载 16KB（A64 8K
+ B 8K），编译期 requires 检测 gl 的 A_tail_tile 描述符（gg8 已挂，
fused 未接入自动回退）。死锁审计：expect 与实际字节严格相等、两侧 tail
判定同源 blk_rows；无新增等待点。复测：runbook 步 2-3 同款（默认路径
仍须 1.68e-03 + ~649/351；tail=32 预期向 −11~14% 靠拢）。

## 2026-07-27 深夜 3：两级 tile Phase 1a 已实现（首测见上节）

dispenser 支持"尾块任务"（docs/09 §4）：`blk_rows[i]` = 行块真实行数，
≤64 的尾块只由前半条带 warp（store_strip<4，即 warp {0,4,1,5}）走完整
计算路径，其余 warp 走独立"信号伴走"分支（同节奏 wait arrived / arrive
finished，计数与全满块一致）。**布局零改动**：A/act 仍 128 补齐，尾块
上半行保持 stale（与既有 padding 行语义相同）；producer 逐指令不变。
全满块路径指令流与改动前逐条相同（active 分支即原代码），balanced 零
开销。store 尾块走单 warp 版（plain/glu 已实现；wred 非 EPIRED 走
warp store、EPIRED 桩 trap——L1 Phase 1c 前不发尾块）。gg8 入口加可选
`blk_rows` tensor（pybind 默认空 → nullptr → 既有行为）；融合 kernel
调用点零改动（默认参数）。

- 死锁审计（红线）：**无新增等待点**。伴走分支只同节奏消费/到达既有
  arrived/finished，每信号计数与全满块完全一致（生产者视角不可区分）；
  task_ready/task_done/sync 在分支外全 warp 照常；跨 rank 零改动。
- 探针：`verify_fp8_gemm` 第 7 参 `tail_rows`（1-64）= uniform 税形状
  （每 expert 真实行 rows_e+tail → 多一个近空尾块），一次跑出
  full/2level 正确性 + A/B 计时。

上机 runbook（先确认卡空闲）：

```bash
cd /workspace
# 1) 重编 ptxas: 预期与基线一致量级(gg8 ~156/tppr8 ~158), 零 spill
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4 2>&1 | grep -E 'registers|spill'
# 2) 回归门: 默认路径 rel_err 1.68e-03, 时间 ~649/351 (不得回退)
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 4096 1536 20
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 20
# 3) 两级裁决(tail=32): full/2level 两个 rel_err 均 OK; 2level 预期快 ~12-14%
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 4096 1536 20 128 32
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 20 128 32
# 4) 四卡回归(fused 传 nullptr, 行为应零变化)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10
```

判据：步 3 的 2level fp8 ≈ full × (2+0.57)/3 ≈ −14%（L0）/−13%（L1）则
机制兑现 → Phase 1b（调度表发尾块 + L0 fused 接入）；正确性任一 FAIL
或步 2/4 回退 → 停下找我。

## 2026-07-27 深夜 2：P3 判负 revert；两级 tile Phase 0 定案、进入 Phase 1

**P3 交织判负定案（docs/04 §2）**：v2（load_kk(1) 后置）ptxas 读数与
性能均与 v1 纹丝不动（仍 168+8B spill；654.4/359.7 vs 基线 649.1/351.0）
→ 源码级指令交织不优于 ptxas 自身调度（串行源码 ≠ 串行 SASS），只把
活跃度顶穿 168 帽。**kernel 已 revert 回串行重标定版**（两版实现留在
git 历史 49b4956/0a9d83d）。数值门两版全过。引擎侧在册杠杆只剩任务
边界 store 后置（~20µs）。

**两级 tile Phase 0 定案（docs/09 §4）**：RB64 vs RB128——L0 +14.5% /
L1 +21.3%（任务 ×2）→ **64 行任务成本 ≈ 0.57/0.61 × 128 任务**（含
2× B 读悲观成分，尾块不付）。立项成立但收益打折：uniform 税 22.9% →
预计 ~13-15%（e2e −150~180µs）。附带发现：RB64 的 160 线程口径无 168
寄存器帽（gg8 178/tppr8 184 零 spill）。**下一步 = Phase 1（L0-only
先行，docs/09 §4 裁决点 1-4 + 点 5 前者）**：任务描述加 M 型别、
producer 64 行 A tile TMA、尾任务 warps 4-7 跳算保信号、半高 store。

⚠️ revert 后需重编重验：`rm -rf moe_bench/kernels/tk/build && python
moe_bench/kernels/tk/build.py 4`，verify 两形状应回 649/351 水位、
rel_err 1.68e-03。

**P3v1 首测（GPU0）**：数值全过（verify 1.68e-03 两形状；e2e rel_err
0.042817506939172745 与基线**逐位一致**，FAIL 是已知容差口径）。性能小幅
判负：L0 fp8 654.1（基线 649.1，+5µs）、L1 361.6（351.0，+10.6µs）。
ptxas 定位根因：gg8::kernel 156→**168 顶帽 + 8B spill**、tppr8 也见
32B/20B spill——v1 里 `load_kk(1)` 在交织块之前发射，a/b_reg[1]（~20
regs）在整个交织期被迫存活，顶穿预算、ptxas 失去调度自由。

**P3v2（已提交，待上机 A/B）**：单变量修正——交织块移到 `load_kk(1)`
之前（kk1 LDSM 延迟由 kk0 的 8 条 QMMA 在 tensor 管线的积压掩护），
raw 路径逐指令不变。判据：ptxas 回落 ~156-158 零 spill；L0 < 649.1 /
L1 < 351.0 才算赢；**仍输则 git revert P3 两个提交、负结果入 docs/04**
（EPIRED 同款流程）。

**uniform 劣化（+370µs, docs/09）**：两级 tile 立项推进，Phase 0 概念
验证已就绪——`verify_fp8_gemm` 第 6 参传 row_block（TK_ROW_BLOCK 宏，
.so 分开缓存）。RB64 vs RB128 判 64 行任务单位成本，判据与 Phase 1
设计裁决点在 docs/09 §4。

上机 runbook（先确认卡空闲）：

```bash
cd /workspace
# A) P3v2 A/B
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4 2>&1 | grep -E 'registers|spill'
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 4096 1536 20
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 20
# B) 两级 tile Phase 0 (RB64 单位经济学; 首次会现编 rb64 的 .so)
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 4096 1536 20 64
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 20 64
```

## 2026-07-27 晚：P3 重标定切片交织已实现（已被上一节首测结果取代）

改动只有 `sm120_common.cuh` 消费者主循环：重标定 FFMA 从 stage 边界移进
**下一 stage 的 kk0**，按 16 列 base-tile 切片与 QMMA 交错
（`FFMA(片j)→清零(片j)→QMMA(片j)`，用 `warp::mma_ABt_base` 单原子调用，
TK 的 `d.tiles[0][m]↔b.tiles[m][0]` 配对与 fp8 base tile 16×32 已对源码
核实）；首块用 s0=s1=0 走同一路径（fma(+0,+0,+0)=+0 逐比特无操作，免
分支），尾块在循环外收尾。依据：E1/E5 定案——重标定暴露 107µs（锁频
L0）/68.4µs（boost）是剩余差距唯一大头，stall_wait 第一大 stall + 发射
槽 60% 空闲（docs/11 §3、docs/12）。

- **数值**：每元素运算次序与串行版完全一致（旧块 QMMA K 升序→FFMA→清零
  →新块 QMMA）→ 逐比特等价。验收：verify rel_err 必须仍 1.68e-03，
  e2e rel_err 与基线逐位一致（0.042817…）。
- **死锁审计（红线）**：无新增/修改等待点。全部 wait/arrive（task_ready/
  inputs_arrived/inputs_finished/task_done）位置与次序不变，移动的只有
  寄存器本地 FFMA/清零指令；生产者与跨 rank 协议零改动。
- **寄存器**：+2 浮点（s0/s1 跨 stage 存活），预期 ~158，零 spill；
  若 ptxas 超 168 或 spill 即回退。
- **预期**：L0 fp8 795.4→~720（锁频，tensor 68.7%→75%+）/ 649→~610
  （boost），向 raw（688.4/580.7）靠拢；L1 同理（438.4→~405）。

上机 runbook（先确认卡空闲）：

```bash
cd /workspace
# 1) 重编看 ptxas: 预期零 spill, gg8::kernel REG ~156-160
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4 2>&1 | grep -E 'registers|spill|STACK'
# 2) 单卡正确性门(两形状, rel_err 必须仍 1.68e-03)+ boost 计时
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 4096 1536 20
CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 20
# 3) NCU 锁频对照 E1(可选先跳过): probe_engine.sh <空闲卡>
# 4) 四卡全链路(融合 kernel 同引擎): 正确性门 + e2e
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --no-verify
```

## 最新（2026-07-27）：gg8 最优性论证（docs/12）+ CUTLASS 同卡复测推翻"L1 反超"

新增 [`docs/12`](docs/12_纯GroupGEMM引擎最优性论证.md) 并当日按同卡 boost
复测数据更新：**CUTLASS 87c 两层都快于我们——L0 506.1µs vs 659.4（我们
76.8%）、L1 315.7µs vs 357.8（88.2%）**。docs/11 §2 的"L1 已反超 CUTLASS"
证伪：旧 453µs 是跨卡（GPU4）锁频锚点，按时钟比折算本身偏慢 ~9%；"L1
引擎关闭"降级为暂缓（已在 docs/11 §2 加更正）。gg8 boost 随频缩放仅
1.17×（CUTLASS 1.31× 线性）——固定延迟 stall 不随频率缩放，重标定切片
交织的优先级进一步上调。当前成立的主张：**同类（路由感知 + blockwise FP8
+ 可通算融合）中最优**（对照 triton，调优水位待 E3）。
**E3 已完成（07-27）**：`tune_triton_moe.py` 调优扫描未找到优于兜底
config 的配置——triton 现有数字即其最优水位，"同类最优"主张坐实，e2e
报数无需重报（docs/08 §6 已加定案注）。docs/12 已重写为仅含最新口径的
精简版。**E4 已定案（07-27）**：COL=128 压力编译，CALL.ABS=0 前提下
四个大 kernel 仍被 ptxas 压 168 reg 并 spill（gg8::kernel 504B stack）
→ **168 帽与 ABI call 无因果，4×2+COL=128 判死，COL=64 平台强制最优**
（docs/10 §6、docs/11 §6 已加定案注）。**E5 已定案（07-27，GPU0）**：L1
`dram__bytes.sum` tk 313.4MB vs CUTLASS 317.9MB——**流量相同，11.8%
差距在延迟/调度**，L1 引擎重开、杠杆与 L0 同类（交织/任务边界）；NCU
签名顺带实锤 CUTLASS 87c 用 4×2 几何 + SM90_TMA_LOAD 2D + 384 线程。
**E1 已跑（07-27，GPU0，probe_20260727_040652）**：boost ×3 极稳——
L0 fp8 649.1/raw 580.7，L1 fp8 351.0/raw 330.7，rel_err 1.68e-03；vs
cta 前 raw −1.7%、fp8 持平；litmus 补验 ld5d_cta 原生。docs/12 §2.1 已
按 ×3 刷新（L0 78.0% / L1 89.9% of CUTLASS），NCU 锁频读数已回填 §2.2
（L0 fp8 795.4µs/68.7%、raw 688.4/80.9%；L1 438.4/61.7%、390.7/69.7%）。
**关键新信号**：cta 后 raw 改善但 fp8 略升——load 瓶颈拆除使重标定链
成为更裸的关键路径，fp8−raw 从 76.9 涨到 **107µs**，交织账面收益变大。
**实验矩阵 E1/E3/E4/E5 全部闭环**（E2 可选：CUTLASS 当前卡锁频 tensor%，
可坐实结构性 ~24µs 残差）。引擎侧下一步进入实施：**重标定切片交织**
（L0/L1 通用头号杠杆，验收 = SASS 交错 + rel_err 1.68e-03 逐比特不变）。另：`time_tp_stages.py`
已支持 `--dist/--skew-alpha/--active` 覆盖 token 路由分布（与 run_tktp
同口径）。

## 2026-07-26 晚间：修复 tdtp `setup` 接口参数错位（待复跑正确性门）

首次上机已完成 Gloo/NCCL/NVSHMEM 初始化，但在 scheme setup 阶段报
`TDTFusedTP.setup() missing 1 required positional argument: 'ctx'`。根因是实现误写为
`setup(problem, cfg, ctx)`，与 `DistributedScheme.setup(problem, ctx)` 契约不一致；且
多出的 `cfg` 从未使用，配置本来就由 `problem.config` 读取。现已改回标准二参数接口。
本次没有修改 kernel、通信协议或等待点；应继续按单步隔离口径复跑：

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --scheme tdtp --iters 10
```

## 最新（2026-07-26 午后 2）：tdtp 新 scheme（Triton-distributed FP8 TP）已接入（待上机验证）

应需求新增对照实现 **tdtp**：用 Triton-distributed 原语实现的 FP8 TP MoE
（移植自 `tmp/td_benchmark` 的 c4，bf16 参考 = triton_dist 上游 `tp_moe.py`）。
数据流：token 量化(1×128) → `fp8_ag_group_gemm`（NVSHMEM FP8 AG + tile 级
overlap GEMM）→ swiglu+路由权重+量化融合 → `run_fp8_moe_reduce_rs(n_chunks=32)`
（FP8 down GEMM + BF16 RS 分块 overlap）。与 tktp 的差别：GEMM 引擎是
triton tl.dot（非手工 TK/mma.sync），融合靠多 stream + tile 级 wait（非单
kernel persistent）。定位：**对照实验**（预期落在 serial 与 tktp 之间，
GEMM 引擎差距决定上限），也是两级 tile 的快速原型土壤。

- 移植：`kernels/td/` 下 6 个文件（fp8_tp_moe/fp8_allgather_group_gemm/
  fp8_moe_reduce_rs/swiglu_quantize_fp8/common_ops/moe_utils，import 路径
  已改为 `moe_bench.kernels.td.*`，依赖 triton_dist 上游包）；`td_tp_scheme.py`
  （权重直接用 problem 的 fp8 数据 + 转置视图，与 tktp/serial 同一份量化、
  公平口径）；`schemes.py` lazy 注册；`distributed.py` worker 在
  `requires_nvshmem` 时改走 `triton_dist.utils.initialize_distributed()`
  （内部 PG+NVSHMEM init，**不能再重复 init PG**；LOCAL_WORLD_SIZE=4 已覆盖）。
- 死锁审计（红线）：triton_dist 的 wait 是**无界自旋**、NVSHMEM barrier_all
  是对称集合——rank 崩溃时无 trap guard（与 07-16 同类风险，第三方库原语
  不可改）。处理：① 首测严格单步隔离；② worker fail-fast 硬退出已有；
  ③ tdtp 定位实验对照、不进生产路径；④ 若要转正，需用 compat 机制给
  wait 打超时 guard patch。**本次未引入自有等待点。**
- ⚠️ 未上机（开发机无 CUDA/triton_dist/vllm，本地仅 lint/结构验证）。
  runbook（先确认卡空闲；triton_dist 环境与 NVSHMEM_SYMMEMRIC_SIZE≥1G 前提）：

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --scheme tdtp --iters 10   # 正确性门(单步隔离)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --scheme tdtp --no-verify \
  --json tp_test_results/tdtp_t512/tdtp_t512.json                                          # 性能(vs serial/tktp)
```

## 2026-07-26 午后：正式里程碑 —— T=512 耗时降低 28.2% / T=1024 31.5%

**主配置正式回归（`tp_test_results/tp_run_20260726_114848`，PASS=12 FAIL=0，
卡组 0-3，warmup=20/bench=50）**：

| 口径 | tktp | serial（triton 兜底未调优） | 耗时降低 |
|---|---:|---:|---:|
| T=512 | **1508.1µs**（med 1503.2） | 2099.6µs | **28.2%** |
| T=1024 | **2768.9µs**（med 2765.0） | 4044.7µs | **31.5%** |

今日三条战线（详见下文与 docs/03）：L1 通信（EPIRED 判负入 docs/04、流水化
保留、sweep 定档 24）、L1 exposure 223→~185（机器噪声区间）、**sched 融合
kernel 259→204.6µs**。正确性：单卡 GEMM 裁决 rel_err 1.68e-03 OK；全链路
tktp 4.28e-2（W1 二次量化）/serial 1.67e-2 均超旧容差（口径待裁决，非回归）。
未闭环：① serial 调优（`tune_triton_moe.py` 未上机，调优后领先需重报）；
② FP8 容差/W1 布局裁决；③ 剩余优化（NCCL 进 graph ~20-40µs、act 量化
下沉 ~35-50µs）。

## 2026-07-26 深夜 2：sched 融合 kernel（tpsched）已实现（待上机验证）

L1 三条快速杠杆收口后（见下节），转向 e2e 第二大项 **sched 259µs**：
`_build_tp_schedules_gpu` 的 ~20 个串行 torch op（CUDA graph 内 ~215µs）
融合为**单 block kernel**（`tpsched::sched_build_kernel`，`tk.tp_sched_build`）。

- 算法（与 golden/torch 版逐元素一致）：warp compaction 按 eid 保序分桶
  （`__vcmpeq4` + shfl 前缀）一趟写 tp_slots/slot_job/slot_w；pull/job_order
  利用 slot 双射 ⇒ mins/maxs 唯一，P 项出现标记 + 段式 scan 求秩**免
  argsort**；push_order = pull_order 按 source 过滤派生，免第三次排序。
- 正确性三层链：host golden ↔ torch 向量化版（preflight 裁决，未动）↔
  融合 kernel（**首次 run 时逐表 `torch.equal` 对拍，不过直接 raise**）。
  开关 `TK_SCHED_FUSED`（默认 1），smem 需求超 99KB（P 过大时）自动回退
  torch 版。
- 死锁审计：单 block kernel，无跨块/跨卡等待、无自旋、无 mbarrier，仅
  `__syncthreads`/warp shfl，无等待点；graph 捕获安全。
- 预期 sched 259 → ~85µs（all_gather 40 + pack 5 + kernel ~30-40），
  e2e 1571 → ~1400。本地 preflight 回归通过（torch 版未动）。

**上机验证通过（2026-07-26 11:27，卡组 0-3）**：对拍通过、正确性门
rel_err 4.28e-2 同基线。**sched 259 → 227.2 →（微优化）204.6µs**；
e2e full_run 在 1510-1550 区间（verify run min 1508.67 历史最低，med
有双峰——共享机器老问题，终数不能挑孤立 min）。四轮修复/优化：
① `push_order` 值域（全局 j → 局部 j−s*T）；② `slot_w` 随机读→写
依赖链摘出（576→374）；③ 1024 线程（8→32 warp 延迟隐藏，374→227）；
④ int4 分量 scan + phase 2 挪位 + 向量化读（227→204.6，省 22.6µs）。
**关键教训：单 block kernel 是延迟受限，8 warp 时全局读/shfl 串行链
藏不住，指令账无效；随机读→写依赖链必须拆成顺序读+fire-and-forget 写。**

sched 204.6 的构成 ≈ pack ~5-10 + NCCL all_gather ~40-70 + kernel
~120-150。剩余空间（未做）：**NCCL all_gather 捕获进 graph**（消启动
开销 ~20-40µs，torch NCCL 支持 capture 但有坑，风险中）；kernel 继续
抠（barrier/clock 主导，ROI 低）。

## 2026-07-26 深夜：L1 EPIRED（epilogue 直推加权归约）已实现（待上机验证）

主配置 stage 归因（`time_tp_stages 64 20 512`，卡组 0-3）实测
**L1 exposure 222.8µs**（L1_fused 572.2 vs L1_gemm_alone 349.4），账：
SM 让渡 ~98（L1 comm 块全程专职、不能像 L0 推完转岗）+ act 量化 ~35 +
预归约重读争带宽 ~40 + push 尾部 ~25 + 杂项 ~10。对比 L0 exposure 仅 88.3。
其中**预归约每 job 读 8 行 expert_out，全体合计恰好把 expert_out
（P×H bf16 ≈ 134MB）完整重读一遍**，加上它的写 134MB，是纯增量流量。

本轮实现**方案 4（EPIRED）**：W2 GEMM 的 C tile 在寄存器里乘 w 后直接
`red.global.add.f32` 累加进 `combine_partial (num_jobs, H) fp32` 部分和，
expert_out 不再落地；push_job 从"读 8 行 expert_out 加权"变成"读 1 行
fp32 部分和转 bf16 并顺手清零（迭代间自维持，无 reset kernel）"。省
134MB 写 + 134MB 重读，代价是 67M 次 fp32 red（L2 原子，fire-and-forget）；
少一次 bf16 中间舍入，数值只会更准。

- 开关 `TK_L1_EPIRED`，默认 1，=0 回退原路径（A/B 用，无需重编译，
  store policy 内运行时分支）。
- 新表 `slot_job/slot_w (P,)` = tp_slots/prered_w 的逆映射（padding 行
  -1/0），host golden 与 GPU builder 同步构建，preflight 逐元素对拍 +
  逆映射不变式已过（本机 CPU，2026-07-26）。
- 改动文件：`tk_tp_scheme.py`（调度表 + combine_partial + 开关）、
  `kernels/tk/tk_moe.cu`（tppr8：globals/wred_store_policy/push_job
  双路径/entry 签名）、`tools/time_tp_stages.py`（st_l1 传参）、
  `tools/preflight_tp_cpu.py`（对拍范围）。
- 死锁审计（红线）：**无新增等待点**。red.add 无返回无等待；行块信号链
  不变且均有界（~32s trap）；partial 生产者（GEMM block）与消费者
  （push_job）同 kernel 同生命周期，trap 连带；跨 rank 依赖链不变无环。

**⚠️ 尚未上机编译/验证**（开发机无 nvcc）。上机 runbook（先确认卡空闲）：

```bash
cd /workspace
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4
#   ↑ 看 ptxas -v: epilogue 新增 ~10 寄存器, 主循环不应 spill
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10   # 正确性门(单步隔离)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.time_tp_stages 64 20 512  # EPIRED=1
TK_L1_EPIRED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.time_tp_stages 64 20 512  # A/B 回退路径
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --no-verify  # e2e 性能
```

预期：L1_fused 572 → ~420-480（省 268MB DRAM 流量，付 L2 原子与 epilogue
指令）。**风险**：fp32 red.add 的 L2 原子吞吐未在本机定量——若 A/B 不
及预期，回退 `TK_L1_EPIRED=0` 并补一个 red 吞吐探针再裁决。

**首轮 A/B（2026-07-26 10:08，卡组 0-3，`time_tp_stages 64 20 512`）**：
EPIRED=1 exposure **385.6**（L1_fused 735.6）vs EPIRED=0 **192.7**
（542.9，与旧基线 222.8/572.2 相当，差异在波动范围；注意第一组
tok_copy 48.5 有干扰痕迹）。EPIRED=1 慢 ~193µs：**标量 red 判负（暂定）**。
根因假设（待 SASS 裁决）：① epilogue 反压——8192 次标量 red/tile，
LSU/L2 原子延迟把 per-task epilogue 打进 GEMM 关键路径（95 tasks/block
× ~1.5µs ≈ 143µs，量级吻合）；② 寄存器/spill；③ 噪声放大（只解释零头）。

**修正（v2，已提交）**：`red.add.v2.f32` 向量原子——float2 本就覆盖相邻
2 列且 8B 对齐，一条顶两条，原子数 67M→33.5M，吞吐/反压同时减半。
验证序列（先确认卡空闲；**若第 0 步没打印 V2_SUPPORTED 就不要重编 moe**）：

```bash
cd /workspace
# 0) v2 试金石: 确认 sm120 支持 red.add.v2.f32
printf '__global__ void k(float*p){asm volatile("red.global.add.v2.f32 [%%0],{%%1,%%2};"::"l"(p),"f"(1.f),"f"(2.f):"memory");}\n' > /tmp/redv2.cu
nvcc -arch=sm_120a -c /tmp/redv2.cu -o /tmp/redv2.o && echo V2_SUPPORTED
# 1) SASS 诊断(旧 .so, 可选但决定性): CALL.ABS 计数(判 syscall)、RED 形态
cuobjdump -sass moe_bench/kernels/tk/build/tk_moe_w4_h4096_rb128.so | grep -c CALL.ABS
cuobjdump -sass moe_bench/kernels/tk/build/tk_moe_w4_h4096_rb128.so | grep -m5 RED
# 2) 重编 + 正确性门(v2 版, 单步隔离)
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10
# 3) A/B: v2 vs 回退路径 (正确性过了再跑)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.time_tp_stages 64 20 512
TK_L1_EPIRED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.time_tp_stages 64 20 512
```

裁决口径：EPIRED=1(v2) 必须打赢 EPIRED=0 的 192.7 才算方案成立；仍输则
把默认翻回 0、负结果记入 docs/04（附 SASS 证据），转向 push_job 流水化
（不依赖原子吞吐）和 `TK_COMM_SMS_L1` sweep。

**v2 A/B 定案（2026-07-26 10:18，卡组 0-3）**：EPIRED=1(v2) exposure
**300.4**（L1_fused 651.5）vs EPIRED=0 **192.2**（543.8，full_run 1572.9
为目前最好）。标量→v2 改善 85µs（与原子数减半近似线性）仍输 108µs →
**EPIRED 判负定案，默认已翻回 `TK_L1_EPIRED=0`**。SASS 证据：RED 原生指令
（CALL.ABS=0）、158 寄存器无 spill → 瓶颈就是本机 L2 fp32 原子吞吐
（~10^11 ops/s 量级）。**数值正确性已验证**：rel_err 4.278e-2 与历史基线
4.28e-2 一致（W1 二次量化的既有口径问题，非本次引入；列/行映射裁决通过）。
负结果已记入 docs/04 §2。

**下一步（按 ROI）**：让渡税 ~98µs 是 L1 exposure 最大头（24 个 comm SM
全程不做 GEMM），两条线：
1. `TK_COMM_SMS_L1` sweep（0 改动）：12/16/20/32 × `time_tp_stages`，
   找让渡税 vs push 吞吐的新拐点（当前 24 沿用 L0，L1 未必同点）；
2. push_job 流水化（小改动）：TMA store 的 wait 推迟、双 buffer 允许 2 个
   push 在飞，per-job 关键路径去掉 PCIe RTT（~1.5-2µs/job × 85 串行
   job/块），给小 comm_sms 腾出空间，与 1 协同。

**流水化 + sweep 收官（2026-07-26 10:32，卡组 0-3）**：正确性门 rel_err
4.28e-2 同基线（流水协议正确）。**流水化零收益**（exposure 192.6 vs
192.2）——归因：24 块时 push **供给受限**（job 就绪速率 ≈ 2048/GEMM 窗口
≈ 4.6 jobs/µs < 24 块消费能力），per-job RTT 本就不在瓶颈，消费侧再提速
无处发挥。代码保留（协议等价、零成本，供给侧变化时受益）。
**`TK_COMM_SMS_L1` sweep 见底**：12→281.5 / 16→260.5 / 20→196.0 /
**24→192.6（甜点）** / 32→223.2，U 型底确认 20-24；减块尾部暴涨、增块
让渡税线性涨。exposure 分解（让渡模型 GEMM×110/(110−comm)）：让渡税 97
+ act 量化 ~35 + 归约争带宽 ~40 + 供给受限尾 ~11。**L1 通信侧已无快速
杠杆**，剩余可动的见下节"下一步"。

**下一步候选（按 ROI）**：
1. **act 量化下沉 L0 epilogue（代号 L0Q，~35-50µs）**：glu epilogue 的
   tile（128 行 × 64 列）恰好覆盖 1×64 量化组——组内 amax 可在 tile 内
   shfl 归约，直接产 act_fp8 + 半组 scale，**独立量化 kernel 和 act bf16
   落地（50MB 写读）全消失**；代价 = L1 主循环重标定改双半组（每 stage
   2 次，复用 sub 寄存器组不加预算，L1 只 6 stage 约 +10µs）+ 数值口径
   1×128→1×64（更细更准）。三处改动：glu epilogue、主循环重标定、缓冲。
2. **sched 融合 kernel（259µs → 预期 ~150，另一条战线）**：当前 GPU
   builder 是 ~15 个串行小 kernel + 4 次 argsort（CUDA graph 内），手写
   融合 kernel（计数+prefix+slot+三序）可省一半；需 preflight 逐比特对拍
   扩展。收益比 L1 内部继续挖更大，但偏离 L1 战线。

## 2026-07-26 晚：引擎归因探针出结论 + TMA cta 形态修复（待上机验证）

单卡探针一键跑完（`tools/probe_engine.sh`，产物
`tp_test_results/probe_20260726_080115_engine/`），结论与账全在
[`docs/11`](docs/11_引擎归因探针与TMA_cta形态.md)：

- **L0 差距 110.6µs = 重标定 76.9µs（70%）+ 结构性 33.7µs**（NCU 锁频，
  fp8 774.6µs/71.2% tensor vs raw 697.7/79.4% vs CUTLASS 锚点 664/84.9%）；
  四份 NCU 第一大 stall 全是固定延迟依赖（31-36%），发射槽 60% 空闲
  → 重标定切片交织是头号杠杆。
- **L1 已反超 CUTLASS**（425 vs 453µs）且 raw DRAM 67.7% 近带宽墙，
  引擎优化只做 L0。
- **TASK_Q 2→4 判负**（+0.2~0.6% 变慢）；K 扫描定价每任务边界 ~0.5-0.7µs
  （L0 合计 ~20µs，小头）。
- **试金石实锤：`.shared::cta` 载入是原生 UTMALDG，TK 用的 `.shared::cluster`
  才走驱动 syscall**（.so 里 109 个 CALL.ABS 全来自 load）。已在
  `sm120_common.cuh` 加 `tma_cta::load_async`（tile 5d/4d + vec 4d）替换全部
  8 个 load 调用点（dispenser 2 + push 2 + scatter 4）。**⚠️ 未上机编译/验证**，
  runbook 在 docs/11 §8——先编 litmus 的 `ld5d_cta`（5d cta 形态当时没测，
  已补），再重编看 ptxas REG（168 帽是否随 CALL 消失而解除，决定
  4×2+COL=128 路线生死），census 应 109→0，verify rel_err 应仍 1.68e-03。

## 2026-07-26 早：baseline 的 triton 一直跑未调优兜底 config

查 vLLM 源码确认：本机（RTX PRO 5000 / sm120）在 `fused_moe/configs/` 里没有任何
条目，`try_get_optimal_moe_config` 查 `E=64,N=768,...` 落空后走 `get_default_config`
的 blockwise 兜底分支，得到 `BM=64/BN=128/BK=128/GROUP_M=32/warps=4/stages=3`
——与 docs/09 §2 从 NCU grid 反解出的值完全吻合。**即历史所有 serial 数字
（含 e2e 2074µs、tensor 66.7%）都不是 triton 的调优水位。**

- 新增 `tools/tune_triton_moe.py`：读主配置，单卡扫 320 个候选（blockwise 下
  `BLOCK_SIZE_K` 会被 kernel 静默 clamp 回 128，名义 640 里一半是重复），
  两阶段计时 + 逐候选正确性门，产出 vLLM 能直接查表命中的 JSON。
  接回方式 `VLLM_TUNED_CONFIG_FOLDER=<结果目录>`，不动 site-packages。
- 机制、结构性上限（两颗 GEMM 共用一个 config、sm120 smem ~100KB）和对报数的
  影响写在 [`docs/08 §6`](docs/08_NCU单kernel计算效率_gg8_vs_triton.md)。
- **⚠️ 脚本尚未在 GPU 机器上跑过**（开发机无 CUDA），先 `--dry-run` 再实跑。
- 预期：调优后 baseline 会净变快（balanced 下抬大 `BLOCK_SIZE_M` 的 padding 税
  为零，见 docs/09），当前 21.5% 领先需重新报数；balanced / uniform 要各调一份。

## 0. 分支状态：`slim`（2026-07-25）

`slim` 从 `fp8_tp` 清理而来，**只保留当前性能最好的那一条实现路径**和长期维护的
文档，历史上试过的其它路径（bf16 融合 kernel、EP 融合实现、peer pull / per-lane
pull 数据面、通信 warp 化、copy engine、L1 按 N 维分解、P2.5 warp scatter、
各类一次性探针与调优脚本）全部从工作树移除。结论保留在 docs，代码在历史提交里。

留下来的实现是一条：**FP8 + TP + 源端 push 的 L0 融合 + 本地预归约 push 的 L1 融合**。

本次清理的代码侧改动：

| 文件 | 变化 |
|---|---|
| `kernels/tileoverlap/common/sm120_common.cuh` | 746→441 行。删掉 bf16 的 `gemm_config` / 两个 `grouped_gemm_sm120` 重载 / `grouped_gemm_sm120_dispenser`（已无调用者）；`plain_store_policy` / `glu_store_policy` 改挂 `gemm_config_fp8`（`CONSUMER_WARPS` 两边同值、`COL` 由实参推导，语义不变）；fp8 dispenser 去掉恒为 false 的 `COL_MAJOR` 模板参数 |
| `kernels/tk/tk_moe.cu` | 只剩 5 个命名空间 / 6 个导出入口。本轮再删 P2.5 的 `scatter_warp` + `SCAT_WARP` 模板、L1 的 `CE` 模板与 `out_planes` 参数、warp 版遗留的 `push_job_team` |
| `tk_tp_scheme.py` | 755→494 行。只剩 fp8 push 路径；`ROW_BLOCK` 本地定义（原来从已删的 `tk_scheme.py` import）；bf16 / CE / warp / lane / v2 分支与对应缓冲全删 |
| `distributed.py` | 删掉多 case 复用一次 NCCL 生命周期的 suite 机制（唯一驱动脚本已删） |
| `schemes.py` | 只注册 `serial` + `tktp` |
| `tools/` | 只剩 5 个脚本 + 一键回归；`run_tktp.py` / `time_tp_stages.py` 改成**直接读主配置 YAML**，命令行只做最小覆盖并打印 |

**⚠️ 这些改动没有在 GPU 机器上编译/运行过**（清理是在没有 nvcc 的开发机上做的）。
接手后第一件事是按下面第 3 节的顺序上机验证，不要直接信任性能数字。

## 1. 实现是什么（一句话数据流）

```text
rowgroup_quant_fp8(token)  →  [ L0 融合 kernel ]  →  act(bf16)
                                push 到 3 个 peer 的 staging → per-token 到达 flag
                                → 本地 scatter 到 TOP_K 个 gathered slot
                                → 行块计数满 → 对应 GEMM tile 开算 → SwiGLU epilogue
                                → comm 块推完转岗领 GEMM task

rowgroup_quant_fp8(act)    →  [ L1 融合 kernel ]  →  combine_staging(peer 可写)
                                W2 GEMM → 行块就绪信号 → 按就绪序领 job
                                → 本地 top-k 加权归约 → TMA 推源卡 → 水位信号

                              [ final reduce ]     →  本 rank 的 (T, H) 输出
```

TP 把 intermediate 维切片，所以跨卡流量稠密、与路由无关，一层只剩这两次搬运。
细节见 [`docs/02_实现架构.md`](docs/02_实现架构.md)。

## 2. 默认口径

唯一默认配置 [`configs/tp_rtx_pro5000_4gpu_fp8.yaml`](configs/tp_rtx_pro5000_4gpu_fp8.yaml)：
hidden 4096 / intermediate 3072 / E=64 / topk=8 / TP / world=4 / FP8 [128,128] /
512 token per rank / balanced / warmup 20 / bench 50 / verify on。

运行时开关只剩四个（默认值就是当前最好配置）：

| 开关 | 默认 | 作用 |
|---|---|---|
| `TK_COMM_SMS` | 24 | L0 通信块数（换机器必须重扫拐点） |
| `TK_L0_PUSH_SMS` | 4 | 其中做 push 的块数，其余做本地 scatter |
| `TK_COMM_SMS_L1` | 跟随 `TK_COMM_SMS` | L1 通信块数 |
| `TK_GPU_SCHED` | 1 | 调度表在 `run()` 内重建并计时（公平口径）；0 只用于归因 |

`TK_ROW_BLOCK` 是编译期宏（`build.py` 的 `row_block`，默认 128），不是运行时开关。

## 3. 上机验证顺序（接手第一步）

必须在**确认空闲**的四卡组上按顺序做，前一步不过不要往下走：

```bash
source /root/miniconda3/envs/vllm-td/bin/activate   # 或你机器上的 vllm-td 环境
cd /workspace                                       # 必须在 moe_bench 的上一级
nvidia-smi                                          # 先确认卡空闲

# 1) CPU 预检: 无 GPU, 验设备守卫 + host/GPU 调度表逐元素一致 + 数据流语义
python moe_bench/tools/preflight_tp_cpu.py

# 2) 干净编译(本轮改了 .cuh/.cu, 必须清缓存; 关注 ptxas 的 spill 行)
rm -rf moe_bench/kernels/tk/build
python moe_bench/kernels/tk/build.py 4

# 3) 单卡 GEMM 裁决(对拍 + 重标定代价)
CUDA_VISIBLE_DEVICES=0 python -m moe_bench.tools.verify_fp8_gemm
CUDA_VISIBLE_DEVICES=0 python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096

# 4) 全链路正确性(四卡, 10 迭代)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10

# 5) 性能 + 归因
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --no-verify
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --scheme serial --no-verify
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.time_tp_stages 64 20 512

# 或者一键(会自己挑空闲卡组并打包结果 zip)
QUICK=1 bash moe_bench/tools/run_tp_all.sh
```

第 2 步如果编译失败，大概率就是本轮 slim 手术碰到的三处签名/模板改动之一
（store policy 挂的 config、fp8 dispenser 少了一个模板参数、L1 入口少了 `out_planes`）——
`git diff fp8_tp -- kernels/` 一眼能对出来。

## 4. 性能现状（历史数字，**待用当前主配置复测**）

| 口径 | T=512 | T=1024 |
|---|---:|---:|
| tktp FP8 | 1708µs | 2970µs |
| vLLM serial FP8 | 2074µs | 4025µs |
| 相对 serial 的耗时降低 | 17.6% | 26.2% |

另有一轮 P2（COL_BLOCK 64 解寄存器墙）之后的实测 e2e T=512 **1548µs**、
T=1024 3115µs，NCU 下 L0 789µs / tensor 71.3%（CUTLASS 参照 664µs / 84.9%）。

这些数字的四个限制没有变：主要来自早期 16 卡服务器的卡组、历史轮 warmup 多为 5、
共享机器有可复现的间歇干扰、并且**不是**在当前主配置（warmup=20）下重新生成的。
最终对外数字必须按第 3 节重跑。演进与账见
[`docs/03_性能优化与方法论.md`](docs/03_性能优化与方法论.md)。

## 5. 尚未闭环

1. **slim 后的首次上机回归**（第 3 节全流程），拿到当前机器 + 当前主配置下的终数。
2. **FP8 正确性口径待裁决**：当前 rel_err 约 `4.28e-2`，高于旧的元素级 `3.5e-2` 容差。
   根因是 W1 为了 GLU tile 交织走了"反量化 → 交织 → 重量化"的二次量化。两条出路：
   把 FP8 容差校准到 `5e-2`，或从数据层直接生成交织后的 W1 量化布局。未裁决前不能
   把性能结果称作完整的正确性闭环。
3. **换机器的重新定标**：`comm_sms` / `push_sms` sweep、P2P 带宽与 RTT、单卡 GEMM
   天花板都要重测，理由见 [`docs/04_平台边界与负结果.md`](docs/04_平台边界与负结果.md) §1。
4. **baseline 调优未落地**：`tools/tune_triton_moe.py` 已写好但没上机跑过。跑完
   要（a）balanced / uniform 各出一份 JSON，（b）用
   `VLLM_TUNED_CONFIG_FOLDER` 复测 serial e2e，（c）按新 baseline 重报领先幅度，
   （d）把 docs/08 §2 的 triton 锁频数字标注为"未调优"或用调优 config 重采。
5. 剩余优化杠杆（2026-07-26 探针后重排，账见 docs/11 §9）：①重标定切片
   交织（L0 差距的 70%，寄存器零成本、逐比特等价）；②任务边界 store 后置
   （~20µs/L0）；③4×2 几何 + COL=128 + [gate16|up16]（等 cta 补丁后的 REG
   裁决）；已关闭：TASK_Q（判负）、scale 进 smem（无 stall signature，降级
   观察）、L1 引擎（已反超 CUTLASS 参照）。e2e 侧另有 act 量化暴露 ~50µs、
   L1 尾 + final reduce watermark 可重叠量，未动。

## 6. 红线（写/改 kernel 前必读）

- **死锁红线**在 `AGENTS.md`，兜底体系与审计清单在
  [`docs/06_死锁兜底体系.md`](docs/06_死锁兜底体系.md)：所有跨卡/跨块等待必须有界 + trap，
  新协议 kernel 首测必须单步隔离，worker 出错走硬退出。2026-07-16 有过一次挂死
  wedge 整机、只能重启宿主机的事故。
- **改 GEMM 模板要改正本** `kernels/tileoverlap/common/sm120_common.cuh`；
  `kernels/tk/sm120_common.cuh` 是 build 时拷过去的副本，被 gitignore，改了会被覆盖。
- **host 调度表的每个 tensor 创建点都要显式 `device="cpu"`**（worker 里
  `set_default_device(cuda)` 会劫持），`preflight_tp_cpu.py` 用 TorchFunctionMode 守住。
- 已判负的路线不要重开，除非能说清**哪个前提变了**——清单见 docs/04 §2。
