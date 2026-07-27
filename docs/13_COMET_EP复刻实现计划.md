# COMET 复刻实现计划:用本仓库 PK 原语在 PCIe 平台实现 COMET 风格 EP MoE 层(scheme `cmep`)

> 本文是给 AI 实现者的完整计划。裁决依据:`docs/paper_row/` 两篇论文
> (COMET, arXiv:2502.19811;ParallelKittens, arXiv:2511.13940)、
> `docs/02/04/07/09` 与当前 `tktp` 实现。按 AGENTS.md 要求:
> **根据计划实现,不要偏离**;每阶段完成后提交 + 更新 HANDOFF。

## 0. 可行性裁决(先读)

**结论:可行。** 但必须先明确两件事:

### 0.1 "PK 原语"在本平台的真实可用面

PK 论文的 8 个原语在本平台(4×RTX Pro 5000 sm120,PCIe 互连,无
NVLink/NVSwitch)的可用性:

| PK 原语 | 本平台可用性 | 本仓库对应物 |
|---|---|---|
| `store_async`(TMA P2P push) | ✓(经 IPC 映射的 peer HBM,PCIe posted write) | `tma::store_async` 到 `TKParallelTensor` peer plane(tktp L0 push / L1 push 已用) |
| `store_add_async`(TMA 远端原子加) | ✗ 无远端原子(docs/04 §2"远端原子 RMW 计数"判负) | 无;归约必须"本地预归约 + push + 目的端本地求和" |
| `reduce` / `all_reduce`(multimem 网内归约) | ✗ 需 NVSwitch | 无 |
| `signal` | ✓(单写者 slot + `st.release.sys`) | `pcie_sync::signal_slot` |
| `signal_all`(multicast) | ✗ 需 multicast 对象 | 循环逐卡 `signal_slot` |
| `wait` | ✓(必须有界) | `pcie_sync::wait_slot`(自带 SPIN_GUARD trap) |
| `barrier` | ✓ | `pcie_sync::pcie_barrier_all` |
| LCSC 程序模板 | sm120 无 wgmma/cluster,PK 模板未移植 | 等价物 = `grouped_gemm_sm120_fp8_dispenser` 持久 kernel + 专职 comm block + named barrier 转岗 |

即:**"用 PK 原语"落地为"用本仓库里 PK 思想的 sm120/PCIe 实现层"**
(`pcie_sync` + `TKParallelTensor/PGL` + `tma_cta` + gg8 dispenser),
而不是直接调 PK 上游 API(其网内归约与 LCSC 模板在本平台不成立)。

### 0.2 COMET 机制 → 本平台实现的映射(含强制偏离)

| COMET 机制 | 论文实现 | 本平台实现 | 状态 |
|---|---|---|---|
| 细粒度通信(token 级) | NVSHMEM UVA 远端读写 | 源端 posted write push + per-token 到达 flag + 收端本地 scatter(P2 协议) | **强制偏离**。pull/UVA 远端读在本平台两次判负:RTT 在飞并发受限(docs/07 §4)、peer TMA 进 GEMM SM 的队列 HoL(docs/07 §3) |
| 线程块特化(comm/GEMM block 分离,GEMM 与非融合版同构) | CUTLASS GEMM block + comm block 水平融合 | gg8 dispenser 持久 kernel + 专职 comm block;且**多一个论文没有的机制:comm block 完工转岗 GEMM** | 已有(tktp),直接复用 |
| L0 shared tensor 沿 M 分解 + 按源 rank 排序、本地 tile 先算 | token sort by source rank,tile 序最小化远端依赖 | 调度表(pull_order/job_order/行块计数 Gate)+ **EP 布局本地行置前** | 机制已有,EP 化是本计划新增 |
| L1 shared tensor 沿 N 分解 + 列序 GroupGEMM | 前 T_N 列算完即开始 topk 归约 | **不做**。Comet-N 在本平台已判负(docs/04 §2:PCIe push 耗 SM,与 GEMM 零和,提前算完只把 wire 暴露到尾部)。替代:沿 M 分解——行块算完即预归约 + push(tktp L1 机制),同样达成"归约先于全部 expert 完成而启动" | **强制偏离**(有既有负结果背书) |
| 自适应工作量分配(n_c 划分点,预编译 kernel 库) | 预编译多档 + profile 元数据 | 运行时 `CM_COMM_SMS` 环境旋钮 + sweep(持久 kernel 的 grid 划分本就是运行时参数,严格优于预编译库) | 已有模式(tktp),EP 需重扫 |

### 0.3 与 tktp 的关系(为什么新增的是 EP 形态)

当前 `tktp` 在 TP 形态下**已经落地了 COMET 的两大核心机制**(线程块特化
+ shared tensor 细粒度依赖解析),且在转岗、两级 tile 等处超出论文。
但 COMET 的原生场景是 **EP:dispatch/combine 由路由决定、通信与计算负载
随路由动态变化**——这正是论文 §3.1 两条 pipeline、图 5/6、图 14(负载
不均)的全部语境,而 TP 的稠密 AG/RS 没有这些动态性。

因此本计划 = **新 scheme `cmep`(Comet-style EP)**:E=64 拆 4 卡各 16
个 expert,L0 = 路由驱动的 token dispatch ⊕ W1 GroupGEMM+SwiGLU 融合,
L1 = W2 GroupGEMM ⊕ 预 combine ⊕ 回源 push 融合。与 `serial`(EP 模式)
和 `tktp`(TP)在同一 harness 下三方对照,并复刻论文的三组实验
(本地优先重排 A/B、n_c sweep、路由倾斜 sweep)。

### 0.4 不做清单(红线,均有 docs/04 判负记录或平台缺失)

1. L1 沿 N 维分解(Comet-N)——判负,重开需新前提;
2. GEMM block 内做 peer TMA / 通信 warp 化——TMA 队列 HoL 判负;
3. peer pull 数据面——RTT 判负,方向必须是源端 push;
4. Copy Engine 编排——判负;
5. 远端原子做计数/完成协议——判负,协议只用单写者 release store;
6. NVSHMEM / multimem / multicast——平台缺失;
7. 预编译 n_c kernel 库——用运行时旋钮替代,不复刻这一工程形态。

关于 docs/04 §2 第一行"早期 EP/TP dispatch 设默认 push 判负"的说明:
该判负的语境是早期 BF16 引擎、L0 GEMM-bound、pull 为默认时"强推 push
无收益",与本计划前提不同——现在 P2 push 已是 TP 默认且证明了
posted write 数据面的正确组织方式(docs/07 §5)。按 docs/04 §5 的重开
原则,改变的前提 = FP8 引擎 + P2 push 架构 + EP 稀疏 dispatch 场景。

## 1. 目标与总验收

- 新 scheme `cmep` 接入 bench(`--scheme cmep`),EP、FP8、主形状与主配置
  完全一致(仅 `parallel_mode: ep` 一项覆盖,见 §2.1)。
- 正确性:harness verify 通过(EP 参考路径已有,ADDING_IMPLEMENTATIONS.md
  "Why the cross-check is sound");W1 GLU 交织二次量化的 rel_err 口径
  (~4.28e-2)与 tktp 相同,沿用同一容差裁决,不新开口径。
- 性能:balanced + uniform 双口径,vs serial(EP)与 tktp(TP)三方报数。
  首验收线 = **balanced 下相对 serial-EP 耗时降低 ≥15%**;努力目标 =
  接近 tktp 水位(balanced 下 EP 与 TP 的通信量、计算量几乎相等,见 §2.2)。
- 论文复刻实验矩阵(M5)产出一篇结果文档。
- 全程遵守 AGENTS.md 死锁红线:每个等待点有界 + trap、首测单步隔离、
  worker 硬退出、提交信息附死锁审计。

## 2. 设计

### 2.1 并行形态与配置

- 新配置 `configs/ep_rtx_pro5000_4gpu_fp8.yaml`:从主配置复制,仅改
  `parallel_mode: ep`,文件头注明"cmep 专用,覆盖项仅此一项"。所有
  结果目录与 HANDOFF 记录该覆盖。
- 每卡 16 个 expert(`problem.expert_map` 给出 global→local 映射),
  权重不切片:W1 per expert (2I=6144, H=4096),W2 (H=4096, I=3072)。
  每卡权重 ~600MB fp8,显存无压力。
- 计算量与 TP 完全相等(每卡 M×N:TP=16384×1536 vs EP=4096×6144,
  K 同)。通信量(balanced):dispatch 每 token 期望送达 2.74 个远端
  rank(4×(1−C(48,8)/C(64,8))−0.915≈2.74)≈ TP 稠密 AG(3)的 91%;
  combine 同理 91%。**所以 balanced 下 cmep ≈ tktp 是合理预期,
  差异化信息在 uniform/倾斜口径。**

### 2.2 数据流(一句话版)

```text
rowgroup_quant_fp8(本 rank tokens)
  → [L0 融合 kernel] push 本 rank token(fp8+scales)到需要它的 peer staging
                     (每 (token,目的卡) 恰一次;到达 flag)
                     → 收端本地 scatter 到该 token 命中的本卡 expert slot(CSR,1..8 个)
                     → 行块计数满 → W1 GroupGEMM tile 开算(本地行 tile 优先)
                     → SwiGLU epilogue 写 act bf16 → comm block 转岗
rowgroup_quant_fp8(act)
  → [L1 融合 kernel] W2 GroupGEMM → 行块就绪信号
                     → job block:同 (src,token) 的本卡多 expert 行加权预 combine(fp32)
                     → 一行 bf16 partial TMA push 回源卡 staging → 水位信号
  → [final reduce]   源卡按 final_contrib 掩码等实际有贡献的 rank,本地求和 → (T,H) 输出
```

与 tktp 的全部结构性差异集中在**调度表**:dispatch 目的集合、scatter
扇出、job 空间、贡献掩码都从"稠密、路由无关"变成"稀疏、每迭代由路由
决定"。kernel 侧机制(push/flag/scatter/行块 Gate/dispenser/转岗/
watermark/final reduce)与 tktp 同构,直接复用 `sm120_common.cuh` 引擎。

### 2.3 调度表(shared tensor 依赖解析的 EP 落地)

输入:一次 packed all_gather 得到全局 `topk_ids_all (W,T,K)` +
`topk_w_all`(与 tktp 完全同款,ids+weights 拼一个 int32x2 张量)。
所有表在每个 rank 上从同一份全局路由确定性推导——源端 push 序 =
目的端消费序,零额外协商(tktp 已验证的技巧)。

L0(本 rank = d 作为收端;同时为每个 peer 计算自己作为源端的 push 序):

| 表 | 形状/语义 |
|---|---|
| `padded_rows` | 本卡 16 个 expert 各自的行数(补齐到 ROW_BLOCK;两级 tile 时带 blk_slack),前缀和给出 slot 布局。**expert 内行序:本地行(src==d)在前,远端按 src 升序**——COMET reschedule 原则的落地 |
| `recv_tok`(pull_order) | 收端消费序:本卡需要的 (src,tok) 列表。排序键 = 该 token 最早喂到的 slot 位置;本地 token 因布局天然靠前 |
| `tok_slot_ptr / tok_slot_idx` | CSR:每个 (src,tok) → 它命中的本卡 slot 列表(1..8 个;fan-out 可变,区别于 tktp 固定 TOP_K) |
| `push_order[r]` | 源端视角:我的哪些 token 需要送 rank r,按 r 的消费序。staging 行号 = 源 token id(恒等,plane 固定 T 行) |
| `blk_expert / blk_slack / blk_cnt_expected` | GEMM 任务表:行块 → expert、尾块 slack、放行阈值。任务序:**全本地行块置前,其余按"最晚到达行的到达序"升序**(近似即可,M5 做 A/B 裁决这一项的贡献) |
| `slot_src_tok` | slot → (src,tok),SwiGLU epilogue 后 L1 侧复用 |

L1:

| 表 | 形状/语义 |
|---|---|
| `job_list` | 稀疏 job 空间:每个 (dest=src rank, tok) 且本卡至少服务其 1 个 expert → 1 个 job。按 dest 分组 |
| `job_rows_ptr / job_rows_idx / job_w` | CSR:job → 本卡 expert_out 行列表 + 路由权重(预 combine 用) |
| `slot_job / slot_w` | 逆映射(行 → job),行块就绪信号驱动 job ready 计数(tktp 机制原样) |
| `push_expected_l1[r]` | 本卡发往 r 的 job 数(水位阈值) |
| `recv_from[s] / final_contrib[t,s]` | 收端:rank s 是否给我送、token t 从 s 是否有贡献。**EP 下由路由决定,必须每迭代重建**(tktp 中是常量 1) |

正确性不变式(preflight 必须逐条断言):① 每个 staging 行/每个 slot
单写者;② `Σ_d |push_order[d]|` = 本 rank token 的 (token,目的卡) 对数;
③ CSR 展开后 slot 全集恰好 = padded 布局的非 padding 行;④ job CSR 行
全集 = expert_out 非 padding 行;⑤ `final_contrib` 与对端 `job_list`
互为转置;⑥ 空 expert(0 行)产生 0 个行块;零贡献 rank 的水位不被等待。

### 2.4 L0 kernel(`cmep::l0`)

- 引擎:`grouped_gemm_sm120_fp8_dispenser` 原样(Gate = 行块计数,
  与 tktp 同);SwiGLU epilogue 原样;两级 tile 通过 blk_slack 接入
  (v1 先全满块,M4 打开)。
- comm block:前 `CM_L0_PUSH_SMS` 个做 push(逐 (token,目的卡) 领
  `push_order`,fp8 行 + scales TMA store 到 peer staging,
  `fence.sys` + per-token `st.release.sys` flag),其余做本地 scatter
  (领 `recv_tok`,acquire 等 flag——**本地内存自旋,有界**,读本地
  staging,按 CSR 写 1..8 个 slot,每写完一个 slot 给对应行块计数 +1)。
  完工后 named barrier 转岗 GEMM。
- 本地 token 不走 staging:scatter 直接从本卡量化缓冲读(等价于 tktp
  对 src==me plane 的处理,省一次拷贝;实现时看 tktp 现状对齐)。
- 空工况:0 push / 0 scatter / 0 GEMM 任务时各角色立即转岗/退出,
  不进任何等待。

### 2.5 L1 kernel(`cmep::l1`)+ final reduce

- W2 GroupGEMM(16 expert,K=3072,N=4096)+ 行块就绪信号:引擎原样。
- job block:按就绪序领 job,读本卡 expert_out 的 job 行集,fp32 加权
  预 combine 成一行,bf16 TMA push 到 dest 的 combine staging
  (staging 行号 = dest 端 token id,单写者 per (src plane, token)),
  本地计数聚合 → 达 `push_expected_l1[dest]` 的唯一线程发跨卡水位
  (`signal_slot`)。dest==本卡的 job 走本地写,不过 PCIe。
- final reduce kernel:每 token 按 `final_contrib[t,:]` 等相应 plane
  水位(有界),本地求和写输出。EPIRED(epilogue 直推 red.add)**不做**
  ——L2 原子吞吐判负(docs/04 §2),沿用行级预归约。

### 2.6 缓冲(全部 setup 分配,worst-case 尺寸,allocation-stable)

| 缓冲 | 尺寸(worst case) | 说明 |
|---|---|---|
| dispatch staging | W plane × T 行 × (H fp8 + H/128 fp32) | 每源恒等 T 行,与路由无关,≈ tktp AG staging |
| 到达 flag / 行块计数 / 水位 | 同 tktp 结构 | flag per (plane,token);seq 单调免清零 |
| padded act 输入(gathered fp8+scales) | P_max = W·T·K + 16·(RB−1) ≈ 18416 行 | 极端倾斜全落一卡的上界;每迭代实际用量由表决定 |
| act / act_fp8 / expert_out | P_max × {3072 bf16 / 3072 fp8 / 4096 bf16} | ≈ 113/38/151 MB,48GB 显存无压力 |
| combine staging | W plane × T 行 × H bf16 ≈ 16.8MB | 恒等行号 |
| 调度表 | 全部按 worst-case 长度 + 有效计数张量 | kernel 从 gmem 读计数,grid 固定 → CUDA-graph 安全 |

### 2.7 数值口径

与 tktp 逐项相同:token/act 1×128 rowgroup 量化;W1 反量化→GLU 32 列
交织→128×128 重量化(二次量化口径问题原样继承,报数时同样注明);
W2 原布局直用;fp32 累加、per-128K 重标定;combine bf16 传输 fp32
本地累加。**不引入任何新数值口径**,保证与 tktp/serial 可比。

## 3. 死锁审计(预填;实现时逐点复核并写进提交信息)

| # | 等待点 | 生产者 | 生产者永不到达的情形 | 兜底 |
|---|---|---|---|---|
| 1 | L0 到达 flag 自旋(scatter block) | peer 的 push block | peer 崩溃/被 kill | 本地内存自旋 + `PCIE_SPIN_GUARD` trap |
| 2 | L0 行块计数 Gate(GEMM producer) | 本卡 scatter block(同 kernel) | scatter 挂死(被 #1 trap 连带) | 同 kernel trap 传染;计数阈值与表同源 |
| 3 | GEMM 流水线 mbarrier | 同 kernel producer | trap 连带 | 裸 wait 允许(本地流水线,红线第 2 条豁免) |
| 4 | L1 行块就绪 → job ready | 本卡 GEMM block(同 kernel) | 同上 | 同 kernel |
| 5 | L1 水位跨卡自旋(final reduce) | peer 的 job block 完成计数唯一写者 | peer 崩溃 | `wait_slot` 有界 trap;**只等 `recv_from` 掩码内的 rank** |
| 6 | 迭代间 pcie_barrier_all | 全体 rank | 任一 rank 崩溃 | 有界 trap |

跨 rank 依赖链方向:L0 恒为"我 push → 对端消费",L1 恒为"我算完 →
push → 对端归约",**无环**。EP 特有风险(必须在 M2/M3 验收里显式验证):

- **零贡献路径**:极端倾斜下某 rank 对某 dest 零 job → dest 绝不等它
  (掩码来自全局路由,所有 rank 推导一致);某 rank 零 GEMM 任务 →
  dispenser 立即 drain,comm 角色直接转岗,barrier 照常参与。
- 空 expert → 0 行块,任务表不含其条目。
- `tools/fault_inject_kill_rank.py` 在 M3 必跑:杀 1 rank,其余有界
  退出 + nvidia-smi 存活 + 卡可复用。

## 4. 实施步骤

每步:改动文件明确、有验收命令与判据、过不了停下来找负责人,不跳步。
上机全部先 `nvidia-smi` 确认卡空闲,优先卡组 0,1,2,3;在 `/workspace`
上一级目录运行;每步完成 = 一次 git 提交(信息含死锁审计)+ HANDOFF 更新。

### M0 基线与探路(半天,只有配置和跑数,零 kernel 改动)

1. 新增 `configs/ep_rtx_pro5000_4gpu_fp8.yaml`(§2.1)。
2. serial EP 基线:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.bench \
     --config moe_bench/configs/ep_rtx_pro5000_4gpu_fp8.yaml \
     --distributed --scheme serial
   ```
   验收:verify ok;记录 balanced e2e(uniform 也各跑一份)。
3. 引擎几何探路(EP 形状下 gg8 无新惊喜):
   ```bash
   CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 16 256 4096 6144 20
   CUDA_VISIBLE_DEVICES=<空闲卡> python -m moe_bench.tools.verify_fp8_gemm 16 256 3072 4096 20
   ```
   验收:rel_err 与 1.68e-03 同量级、ptxas 零 spill、耗时录入结果目录
   (N 只是运行时循环,预期无异常;若工具不支持该形状参数,先补参数)。

### M1 调度表 host golden + preflight(开发机可完成,无 GPU)

1. 新文件 `cm_ep_scheme.py`:`CMFusedEP(DistributedScheme)` 骨架 +
   `_build_ep_schedules`(host 逐元素 golden,纯 CPU,所有 tensor 显式
   `device="cpu"`——docs/04 §4 默认 device 污染坑)+ torch 向量化版。
2. 新工具 `tools/preflight_ep_cpu.py`(仿 `preflight_tp_cpu.py`):
   golden ↔ torch 版逐元素对拍 + §2.3 六条不变式 + 构造用例:balanced、
   uniform、极端倾斜(全部 token 落一卡)、单 expert 空、某 rank 零贡献。
3. `schemes.py` lazy 注册 `cmep`(仿 tdtp 的 try/except 块)。

验收:`python moe_bench/tools/preflight_ep_cpu.py` 全绿。

### M2 L0 融合 kernel + 首测(单步隔离,红线第 3 条)

1. 新文件 `kernels/tk/cmep_moe.cu`(自包含:include `sm120_common.cuh`,
   自带 rowgroup_quant/barrier 入口副本;命名空间 `cmep_l0` 等)。
   `build.py`:把 `_build_so` 的源文件由硬编码 `tk_moe.cu` 改为随
   `module_name` 选择(缓存 key 已含 module 名,零影响 tktp)。
2. `cm_ep_scheme.py` setup:TKParallelTensor 缓冲(§2.6)、权重准备
   (W1 交织重量化复用 tktp 代码路径)、host 表上载;run:v1 先只到
   L0(quant → L0 fused),L1 用临时非融合兜底(gg8 单跑 W2 + torch
   预 combine + NCCL all_to_all/RS)以便全链路 verify 兜住 L0 正确性。
3. 新工具 `tools/run_cmep.py`(仿 `run_tktp.py`:直接读 EP YAML,
   最小覆盖并打印)。
4. 环境旋钮:`CM_COMM_SMS`(默认 24 起步)、`CM_L0_PUSH_SMS`(默认 4)、
   `CM_COMM_SMS_L1`、`CM_GPU_SCHED`(v1 固定 0,host 表;GPU/融合
   builder 到 M4)。

首测 runbook(**STEPS 只含正确性门这一步**):
```bash
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4   # 看 ptxas: 零 spill
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_cmep --iters 10
```
验收:verify ok(rel_err 与 tktp 同口径 ~4.28e-2 量级);提交信息附
§3 审计表逐点结论。失败处置:任何 rank 挂死迹象立即按 docs/04 §4 流程
取证,不重试不批量。

### M3 L1 融合 + final reduce + 全链路

1. `cmep_l1` kernel + final reduce(§2.5),替换 M2 的临时兜底。
2. 正确性门 + 故障注入:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_cmep --iters 10
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_cmep --iters 10 --dist uniform
   python -m moe_bench.tools.fault_inject_kill_rank ...   # 杀 1 rank, 其余有界退出
   ```
3. 首轮 e2e(balanced + uniform,`--no-verify`,另跑 10 次正确性)。

验收:双分布 verify ok;故障注入后 nvidia-smi 存活、卡可复用;e2e
三方数字(cmep/serial-EP/tktp)入结果目录 `tp_test_results/` + HANDOFF。

### M4 性能定标

1. `CM_COMM_SMS` × `CM_L0_PUSH_SMS` × `CM_COMM_SMS_L1` sweep
   (balanced 与 uniform 各一轮;起点 24/4/24,EP 是新工作负载,
   拐点必须重扫——docs/04 §1.2)。
2. 阶段归因工具 `tools/time_ep_stages.py`(仿 time_tp_stages:量化/
   L0/L1/final reduce/sched 分解)。
3. 打开两级 tile(blk_slack 接入;EP uniform 下尾块占比高于 TP,
   预期收益更大)。
4. 调度表下沉:torch 向量化版进 run() 计时(`CM_GPU_SCHED=1` 公平
   口径);若 sched 占比 >10% e2e,仿 `tpsched::sched_build_kernel`
   写融合版(先 golden 对拍,首 run `torch.equal` 逐表校验)。

验收:公平口径(sched 计时在内)下 balanced 相对 serial-EP ≥15%;
数字与 sweep 曲线入结果目录 + HANDOFF。

### M5 论文复刻实验 + 文档

1. **本地优先重排 A/B**(COMET §3.1 的贡献量化):调度表加开关
   `CM_LOCAL_FIRST`(0 = 朴素 expert-major 行序/任务序),A/B e2e 与
   L0 exposure——对应论文图 5 的机制验证。
2. **n_c sweep 曲线**(论文图 8 对应):M4 的 sweep 数据整理成
   "comm SM 数 → e2e"曲线,balanced/uniform 双线。
3. **路由倾斜 sweep**(论文图 14 对应):`--dist uniform --skew-alpha`
   多档,cmep vs serial-EP vs tktp 三线(tktp 天然免疫计算倾斜,是
   本平台特有的第三条参照线,论文没有)。
4. 结果沉淀 `docs/14_COMET复刻结果.md`:机制逐条对照(哪些复现、哪些
   平台强制偏离、量化差异),更新 docs/01。

## 5. 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| uniform 下 EP 计算倾斜导致 e2e 明显差于 tktp | 高(结构性,非实现缺陷) | 如实报数;这正是论文图 14 语境,三方对照本身就是结论 |
| 调度表构建成本吃掉公平口径收益 | 中 | M4 步骤 4;不达标前 `CM_GPU_SCHED=0/1` 双口径并注明 |
| CSR 可变扇出 scatter 吞吐低于 tktp 固定扇出 | 中 | scatter 内循环按 CSR 段展开,复用 P2"每 lane 一 token"组织;不够再仿 P2.5 warp 协作(注意其未定案历史,docs/07 §6) |
| P_max 缓冲让显存紧张(与 vLLM 常驻并存时) | 低 | 可按 `--dist` 口径给上界降档(balanced 时 P_max=W·T·K/W·pad);默认保守全量 |
| gg8 引擎在 N=6144 出现新的调度/寄存器行为 | 低 | M0 步骤 3 先裁决;异常则停,不带病进 M2 |

## 6. 参考文件清单(实现前必读)

- 论文:`docs/paper_row/` 两篇(COMET §3.1/3.2;PK §3.1.2-3.1.4)
- 红线与流程:`AGENTS.md`、`docs/06_死锁兜底体系.md`、`docs/04 §4-5`
- 引擎与协议:`kernels/tileoverlap/common/sm120_common.cuh`(正本,
  不要改 `kernels/tk/` 下的副本)、`docs/02_实现架构.md`
- 同构参照:`tk_tp_scheme.py`(setup/run/缓冲/调度表模式)、
  `kernels/tk/tk_moe.cu`(L0/L1/final reduce/sched 融合 kernel)
- 判负清单:`docs/04 §2`(本计划 §0.4 的依据)
- 接入规范:`ADDING_IMPLEMENTATIONS.md`(scheme 契约、EP 正确性机制)
