# HANDOFF — TK 通算融合 MoE 进度交接

## 2026-07-23：CUTLASS 参照结果——"累加税天花板"被推翻，GEMM 引擎欠账 28%

- NCU 锁频四方 + CUTLASS 参照（docs/08 §5）：**CUTLASS fp8 L0 664µs/84.9%
  tensor vs 我们 924µs/60.9%（=72%）、triton 834µs（=80%）**；L1 差距小
  （453 vs 我们 519=87%）。bf16 CUTLASS 92.3%（2.x Sm80 路径即可达）。
- docs/08 §3.4 的"60-67% 是 fp32 累加税天花板、指令级无翻盘空间"**已修正**：
  同 MMA atom 下差距在软件结构——头号项 = **per-K-block(128) 重标定摊薄**
  （我们 per-2-step → +31% 指令；CUTLASS blockwise 主循环每 128 深 K 块
  promotion 一次，数学与 serial/triton 等价，非精度赌博）。次项 = TMA
  warp-specialized 流水（深改，视第一步回收再定）。
- 注意：CUTLASS/gg8 探针均为纯 GEMM 可直比；triton 数字带路由 gather。

## 2026-07-23：CUTLASS grouped GEMM 参照探针（已跑完，结果见上一条）

- 新增子模块 `cutlass/`（NVIDIA CUTLASS 4.6.1 main @e64a913，浅克隆；远端
  `git submodule update --init --depth 1 cutlass`）。
- 动机：给 gg8/gg 引擎立"厂商可达水位"参照。sm120 支持确认：
  - fp8：`examples/87c` 即 sm120 blockwise **grouped** GEMM，口径与我们逐项
    同构（A=act fp8 RowMajor scale 1×128，B=weight fp8 ColumnMajor scale
    128×128，bf16 输出，fp32 累加，tile 128×128×128），原样编译即可；
  - bf16：sm120 的 3.x array（grouped）builder 只收 F8F6F4，**bf16 无 3.x
    grouped 路径**；用 2.x `GemmGrouped`+Sm80 mma.sync（16×8×16 bf16，与 TK
    bf16 引擎同指令）编到 sm_120a（`tools/cutlass_probe/grouped_gemm_bf16.cu`，
    example 24 拷贝改 bf16+TN layout，附带 batched GEMM 对照）。
- 构建/运行：`bash tools/cutlass_probe/build.sh`（本地无 nvcc 未编译验证，
  首编译在远端；两探针自带 CLI + 校验 + GFLOPS 输出）。形状口径：
  L0 `--groups=64 --m=256 --n=1536 --k=4096`、L1 `--n=4096 --k=768`。
- 对比时注意：CUTLASS 探针是纯 GEMM（无 GLU epilogue、无 scatter/gather、
  无 topk 加权），与 gg8 探针（tools/verify_fp8_gemm.py）比要按 docs/08 的
  口径扣除非 GEMM 成分；boost 频率下直接比,与 NCU 锁频数字不可混。

## 2026-07-23：路由不均衡劣化归因 + RB64 负结果（docs/09）

- uniform 路由 A/B（卡组 4-7）：我们 1618→1988（+22.9%），serial 2062→2167
  （+5.1%）——劣化差 3.5 倍的根因 = **ROW_BLOCK=128 padding 税**（每 expert
  256±16 行卡块边界，M 功 +25%，账与实测 +370µs 吻合）；serial 的 M 粒度
  BLOCK_M=64 已从 NCU grid 反解实锤。uniform 下领先缩到 8.3%。
- **RB64 判负**：`TK_ROW_BLOCK=64`（零代码，fp8 config 已吃宏）正确性同水位
  过门，但 balanced +249µs = 纯 B tile 重载税（行块 ×2 → 权重流量 ×2），
  uniform 2146 反而比 RB128 慢 158µs。padding 敏感度确实降（370→279）但
  底座税吞掉全部收益。正解 = 两级 tile（主体 128 + 余数行 64 尾块，未立项）。
- run_tktp 新增 `--skew-alpha`/`--active` 旋钮；build.py 修并发编译竞态
  （冷缓存 4 worker 同时 nvcc → flock + 原子改名，2d66561）。
- 建议后续报数用 balanced + uniform 双口径。verify FAIL 口径说明见 docs/09 §5。

## 2026-07-23：NCU 单 kernel 计算效率对比——GEMM 引擎与 triton 同档，e2e 领先全在重叠结构

- 单卡 NCU（锁频横比）四方对比 gg8 vs triton fused_moe：L0 我们 60.9% vs
  66.7% tensor 利用率（−10%，归因 per-2-step fp32 重标定 + dispenser 指令
  +31% 抢发射槽）；L1 我们 53.9% vs 50.2%（+3.4%，短 K 下 triton 摊薄更差）。
  两层合计 −5%，计入 GLU 融合/serial 额外 kernel 后基本平手——**e2e −21.5%
  全部来自 AG 藏匿 + fp8 AG 字节减半 + 转岗，GEMM 引擎无翻盘空间也无欠账**。
  双方指令选型相同（QMMA+fp32 累加），60-67% 即累加税下可达水位。
  详见 `docs/08_NCU单kernel计算效率_gg8_vs_triton.md`（含复现命令与坑）。
- 新增 `tools/ncu_serial_gemm_probe.py`：单卡复刻 serial 的 fused_experts
  一步（无 NCCL），形状可参数化，供 ncu kernel replay 安全采样。
- 坑（已入 docs/08 §4）：4 卡分布式下 NCU 采单 kernel 会让其余 rank 在 NCCL/
  跨卡 flag 上自旋，kill 后残留无主自旋 kernel 占卡 100% util，需
  `nvidia-smi -r` 复位；融合持久 kernel 结构性不可按 kernel 采。
- 本轮只新增工具与文档，无 kernel/等待点改动。

## 2026-07-20：新增上一周实验总结简版

- 新增 `docs/07_2026-07-13至2026-07-19实验总结_简版.md`，只保留细粒度通信计算
  overlap 的 L0/L1 数据流、persistent kernel 资源分工，以及方案 A、P1、P2、
  P2.5 的关键性能结果；详细配置、事故过程和方法论仍保留在原完整版。
- 本轮只修改文档，没有修改 kernel、同步协议或任何等待点。

## 2026-07-20：完成 2026-07-13 至 2026-07-19 实验周报

- 新增 `docs/07_2026-07-13至2026-07-19实验总结.md`，按统一口径汇总方案 A、
  P1、P2、P2.5 和 wedge/死锁兜底实验，包含原理、A/B 数据、机制裁决、正确性
  边界、性能演进和下周验证顺序。
- 周报明确区分历史迭代口径（warmup=5，T=512/1024 分别 50/30 次）与当前
  warmup=20 主配置；P2 当前最好仍为 **1629/2777µs**，P2.5 因具体 JSON 未入库且
  08p25 wedge，仅记为“功能步骤通过、性能和稳定性未定案”。
- 本轮只修改文档，没有修改 kernel 或等待协议，不需要新增死锁等待点审计。

## 2026-07-17：死锁兜底第二轮——mbarrier 有界化 + host fail-fast + 故障注入工具

- 对 7-16"已无任何无界等待"结论的复审发现三处缝隙，本轮全部收口（体系与边界
  见 **docs/06_死锁兜底体系.md**）：
  1. **mbarrier 无界等待**：跨卡 TMA pull 喂的 semaphore 在对端 rank 死亡时永不
     arrive，裸 `wait()` 不经过任何自旋 guard。新增 `pcie_sync::guarded_wait`
     （try_wait + clock64 1e11 周期 ~40s 超时 trap，唤醒延迟与 wait 相同），
     替换 tk_moe.cu 全部 14 处 + tileoverlap/02 一处。本地 GEMM 流水线 mbarrier
     保留裸 wait（生产者同 kernel、gate trap 连带回收，且是最热路径）。
  2. **tileoverlap/02 dispatch_gate 裸自旋**（7-16 审计只覆盖了 tk_moe +
     sm120_common）：补 PCIE_SPIN_GUARD。tk_scheme.py（bf16 EP）仍加载该模块。
  3. **host 侧 teardown 隐患**：worker 异常后原 finally 仍执行 synchronize +
     destroy_process_group（NCCL 跨 rank 握手 + 逐个 IPC unmap，正是 7-16 取证
     中排队在 RM/uvm 锁后面的调用类别）。distributed.py 新增 `_fail_fast_exit`：
     异常 → traceback → `os._exit(1)`，用户态 teardown 全跳过。
- **新工具 tools/fault_inject_kill_rank.py**（红线兜底的动态验证，GPU 机上跑）：
  SIGKILL 一个 rank → 断言其余进程树限时退净 + nvidia-smi 存活 + 各卡能新建
  context。用法见 docs/06 §2 第 4 层。**跑通之前，兜底只有静态审计背书**。
- AGENTS.md 红线更新：第 2 条扩为"自旋 + mbarrier 都必须有界"，新增第 4 条
  "worker 出错必须硬退出"，第 5 条挂故障注入工具。
- **性能口径不变**：guard/guarded_wait 就绪路径零或一次 clock64 读；上机后先
  `STEPS='^00_|^01_|^03f8_|^03p25_'` 单步验证编译+正确性，再跑 04 档对比确认
  1629/2777 未回退，然后按红线第 3 条单独跑一次故障注入。

## 2026-07-16：wedge 事故定位收窄 + 全量自旋加固 + 死锁红线入 AGENTS.md

- 现场收窄（tp_run_20260716_081949）：03p25 正确性/04p25/05p25 全套 sweep **全部
  通过**，挂死发生在 **08p25_time_stages_scatwarp**；且卡住的 worker 内核栈停在
  `uvm_map_external_allocation`（TKParallelTensor IPC 映射 = setup 阶段，还没跑
  benchmark）——它是排队在 RM GPU 锁后面的**受害者**。dmesg 证实 wedge 机理 =
  RM GPU 组锁被卡死持有者占住，deviceQuery/nvidia-smi/cache_mgr 全部排队 →
  8 卡"全灭"（锁是全局的，实际只用了 0-3）。元凶未收口：候选 = 前一步（05p25
  commsms_24）teardown 残留 或 08p25 自身 worker 集内的先发故障；**待复测取证**
  （需要该轮 summary.txt 全文 + 08p25 log 尾部）。
- **全量自旋加固（保证死锁有界）**：`PCIE_SPIN_GUARD_DECL/TICK` 宏（sm120_common
  正本，~32s 超时 trap）覆盖所有等待点——pcie_sync::wait_slot / pcie_barrier_all
  （L1 push/final_red/设备栅栏）、7 处 GEMM gate（disp/ddisp/tpdisp/tpdisp2/
  tpdisp8×2/prered 系）、2 处 CE flag poll、scatter_lane/scatter_warp flag 自旋。
  **代码中已无任何无界跨卡/跨块等待**：任何协议 bug ≤32s 变成干净的 kernel abort
  + 进程退出，不再 wedge 整机。
- **死锁红线已写入 AGENTS.md**（每次改 kernel 必须过）：①提交前逐等待点审计
  （信号生产者/永不到达情形/跨 rank 无环），结论写进提交信息；②所有自旋必须用
  guard 宏，禁止无界；③新协议首测单步隔离。
- **复测（重启宿主机后）**：先 `STEPS='^01_|^08p25_'` 单步复现 08p25——trap 会把
  真死锁变成 FAIL 现场；若 08p25 过了则怀疑 teardown 残留竞态，改跑完整序列取证。

## 2026-07-16：P2.5 首测整机 wedge 事故 + 自旋 trap 加固（待复测）

- 现象：P2.5 首测把容器整个卡死（nvidia-smi 挂）。机理：持久 kernel 挂死后超时
  kill 不能抢占自旋 kernel（进程 D 态），且一个 rank 死后其余 rank 经 IPC 继续
  自旋访问已死上下文显存 → 驱动通道 wedge。**非驱动损坏**，恢复：宿主机 kill -9
  → `nvidia-smi -r -i 0,1,2,3` → 容器/宿主机重启逐级升级。已入 docs/04 坑索引。
- 头号嫌疑：scatter_warp（本轮唯一首跑的新设备代码）或其与 psms=2/默认 push 的
  组合。**flag 自旋已加有界 + trap（~30s 超时杀 kernel → CUDA error → 干净退出），
  下次挂死会变成可诊断的 FAIL 而不是 wedge**（scatter_lane/scatter_warp 两处）。
- **恢复后的复测纪律**：①先 `dmesg | grep -iE "nvrm|xid"` + 上轮 summary.txt 定位
  挂死步骤；②单步隔离跑 `STEPS='^01_|^03p25_' bash tools/run_tp_all.sh`（不要挂长
  矩阵）；③trap 触发则看是哪个自旋（Xid/驱动日志 + FAIL step），协议 bug 定位后
  再放开 04p25/05p25。
- 注意：TK_L0_PUSH 默认已是 1——03f8/04f8 等所有 fp8 步骤现在默认走 push 路径
  （上轮 push 全绿，但若要排除变量可 TK_L0_PUSH=0 回滚到 lane）。

## 2026-07-16：P2 实测——push 双档全赢翻默认（1629/2777），瓶颈转移到本地 scatter

- tp_run_20260716_075314：push@24 = **1629**（vs lane 1682，−53）、T=1024 = **2777**
  （vs 2830，−53）；正确性 rel_err 4.28e-2 同水位。**TK_L0_PUSH 默认已翻 1**。
  当前最好口径 **1629/2777，vs serial fp8 2074/4030 = 耗时降低 21.5%/31.1%**。
- 关键发现：①push 2 个 SM 即饱和（psms@comm8: 2/4/6 = 1750/1875/2518），wire 彻底
  离开关键路径；②**拐点仍 24 未左移，但原因已换**——收侧 scatter 每 token 8 个 TMA
  store + store_async_wait 的 per-token 串行等待（~10µs 级/lane），scatter SM 4→20
  的 246µs 弹性与此吻合。最好实测配置 = comm24 + psms4（psms2 仅在 comm8 验证过）。
- **P2.5 已落码（TK_L0_SCAT_WARP=1，默认 0）**：`tpdisp8::scatter_warp` +
  `kernel_push<SCAT_WARP>`——scatter 块 9 个 warp 全员领 token（无 smem/semaphore），
  每 token 32 lane 各搬 128B 段写 8 槽，lane 0 acquire 等 flag + syncwarp（既有
  wait-then-sync 模式），全员 threadfence → syncwarp → lane 0..7 各发 red.release
  （signal_epilogue 模式）；push 块不变，P2.5 步骤统一 psms=2（已实测饱和）。
  脚本 03p25/04p25/05p25 sweep{4,6,8,12,16,24}/08p25。本地校验已过。
- **下一步（上机）**：`STEPS='^00_|^01_|^03f8_|^03p25_|^04f8_|^04p2_|^04p25_|^05p25_|^08p25_'
  bash tools/run_tp_all.sh`（01 必跑）。裁决：①03p25 正确性；②04p25 vs push 1629；
  ③05p25 拐点是否真正左移（warp scatter 吞吐应数倍于 lane 版；若 8 SM 档 ≤1629
  则转岗收益兑现）；④08p25 拐点档 L0_fused（目标 →~850 以下）。预期 e2e ~1530-1580。

## 2026-07-16：P2 L0 push 强路径落码（TK_L0_PUSH=1，待上机）

- 设计：源侧 push lane（TK_L0_PUSH_SMS=4 个 comm 块）按 push_order 把量化行推到
  3 个 peer 的 ag_staging 平面（单写者），每 token store_async_wait →
  threadfence_system → st.release.sys 写 per-token seq flag（免清零）；收侧
  scatter lane 严格 pull_order 领取，远端 token 本地自旋等 flag 后**从本地
  staging 读**（无 PCIe RTT——comm SM 可压的结构性原因），自己分片直读
  pre_tokens；scatter/计数/转岗与 pull 逐字节同构。依赖无环（GEMM←scatter←
  flags←push，push 只依赖本地数据）；跨迭代由既有双 pcie_device_barrier 保护。
- 落码：tpdisp8::pglobals/push_lane/scatter_lane/kernel_push/entry_push，绑定
  moe_tp_dispatch_gemm_fp8_push；scheme TK_L0_PUSH（默认 0，优先级高于 lane）+
  TK_L0_PUSH_SMS + push 缓冲（staging/sscales/flags TKParallelTensor）+
  push_order 进计时重建；stages 工具已感知。本地 py_compile/bash -n/preflight 过。
- **下一步（上机）**：`STEPS='^00_|^01_|^03f8_|^03p2_|^04f8_|^04p1_|^04p2_|^05p2|^08p2_'
  bash tools/run_tp_all.sh`（01 必跑）。裁决：①03p2 正确性；②04p2 vs lane 1681；
  ③05p2 拐点应大幅左移（目标 8 SM 档 ≤1620）；④05p2b 验证 4 push SM 够用；
  ⑤08p2 拐点档 L0 让渡税（目标 →~50-70）。预期 e2e ~1570-1610。

## 2026-07-16：P1 实测——lane 全档赢翻默认（1681/2831），拐点未左移→P2 门控触发

- tp_run_20260716_073019：lane@24 = **1681**（vs 波同步 1709，−28）、T=1024 = **2831**
  （vs 2935，−104）；lane 在每个 comm_sms 档位严格优于波同步（@4 档快 1181µs）；
  正确性 rel_err 4.27e-2 同水位。**TK_L0_LANE 默认已翻 1**（=0 回滚）。
- 核心假设证伪：拐点未左移（16 SM 即回升 1775）——pull 弱路径 RTT 受限，
  ~480 在飞并发是真实需求而非波同步低效的补偿。压 comm SM 数必须换 push
  强路径（posted write 无 RTT 往返，4 SM 打满 50.9GB/s）→ **P2 门控触发，
  下一步 = P2 L0 push 化（fp8 + lane 组织重做，不是复活 bf16 tppdisp）**。
- 待查：默认路径 08f stages 连续两轮失真（3238/3983 vs bench 1709），lane 路径
  08p1 准确（1812 vs 1808）；只影响归因工具，bench 口径不受影响。
- 当前最好口径：**e2e 1681（T=512）/ 2831（T=1024），vs serial fp8 2074/4030
  = 耗时降低 18.9% / 29.8%**。

## 2026-07-16：PK 框架路线重估——P1 comm 块 per-lane 自由化（新主线）

- 依据 PK 论文三层框架 × 本机事实完成路线重估（docs/归档/2026-07-16_PK框架下的
  路线重估.md）：①CE 判负获理论背书（≥256MB 粒度判据）；②register-op/in-network
  路线本平台物理不存在（无 NVSwitch + 远端原子不可靠）；③intra-SM 的适用前提
  （通信模式与计算 tile 流对齐）在我们 L0/L1 都不满足——方案A 判负与 PK 判据一致，
  inter-SM + 转岗的现架构方向本来就对；④L1 定量定案：PK 隐藏判据 K≥sR/2B 给出
  阈值 ≈3000 ≫ 我们的 K=768，wire 结构性藏不满，v1 不再动。
- 核心洞察：错配在"通信占的 SM 数"——PK 说 TMA 15 SM 打满 450GB/s，我们 23.5GB/s
  的 pull 却占 24 SM，根因是 dispatch_persistent 波同步低效（20 lane 等最慢者）。
  方案A 的 comm_lane_pull（per-lane 自由运转）是对的代码放错了地方，应回专职 comm 块。
- **路径**：P1 comm 块 per-lane 化 + 扫 comm_sms{4..24}（预期让渡税 183→90-120，
  e2e→~1620-1650）→ P2 由 P1 门控：push 化重估（强路径 4 SM 打满，e2e→~1570-1600）
  → P3 sched 摊薄（最大单项 ~270，e2e→~1500-1540）。
- **P1 已落码（TK_L0_LANE=1，默认 0）**：`tpdisp8::dispatch_persistent_lane`（槽线程
  stride-8 铺 5 个 warp、全局 pull_next 原子领取=严格 pull_order 消费序、零块内同步、
  非槽线程 sync(2) 等转岗）+ `kernel_lane`/`entry_lane`/绑定 `*_fp8_lane`；scheme 分支
  与 WARP/CE 互斥；time_tp_stages 已感知。脚本：03p1 门 + 04p1 e2e + 05p1 lane
  comm_sms sweep{4..24} + 05p1_wave_commsms_4 对照 + 08p1 归因。本地 py_compile/
  bash -n/preflight 已过。
- **下一步（上机）**：`STEPS='^00_|^01_|^03f8_|^03p1_|^04f8_|^04p1_|^05p1_|^08f_|^08p1_'
  bash tools/run_tp_all.sh`（01 必跑）。裁决：①03p1 正确性；②05p1 sweep 拐点是否左移
  （目标 8-12 SM 持平或超 24 SM 波同步版的 1709）；③08p1 在拐点档的 L0 让渡税
  （目标 183→90-120）；④若 12 SM 以下 L0 暴露回升 → 直接进 P2（push 化重估）。

## 2026-07-16：方案A 定位定案——per-SM TMA 队列共存税坐实，现形态判负

- 08w8_diag_warp（tp_run_20260716_070406，30 迭代全稳）给出精确账：纯 GEMM@满SM
  L0=742/L1=383；默认融合 935(+183)/556(+173)；warp 融合 964(+222)/729(+301)。
  **机制②坐实**：共存税 +226 且随 comm lane 数单调（s1 +175/s2 +205/s4 +222），
  µs 级 peer TMA 读堵住本 SM TMA 队列；**机制①排除**（gate/straggler 税 −4≈0）；
  warp 几何免费（742 vs gg8 752）。默认路径 comm 块同发 peer TMA 不拖慢 GEMM →
  per-SM 争用而非全局干扰。**平台事实：本机把 peer TMA 拉取塞进 dense GEMM 的
  SM，比让渡整块 SM 更贵**（已入 docs/04 负结果表）。
- 附带发现：86SM 纯算折算 ≈962 > 默认融合 935 → 转岗净赚，默认路径比预想更优；
  e2e 双稳 lockstep 不复现（需 rank 漂移参与），结构税已足够定案，不再单独追；
  快态账吻合（964−935≈+29 ≈ e2e min 差 +36）。
- 处置：TK_L0_WARP/TK_L1_WARP 默认 0 留档；L1 warp 无翻盘自由度定案。
- **唯一残余方向（未立项）**：comm lane 改非 TMA 向量加载（ld.global float4 绕开
  TMA 队列）；若共存税归零，L0 融合 ~790 vs 935（−145µs，e2e ~1560）。风险 =
  LSU/MSHR 换一种争用；若做必须带 e2e 04w8 复跑验证漂移稳健性。
  详见 docs/归档/2026-07-16_方案A_通信warp化设计与落地.md §-1/§定案后的处置。

## 2026-07-16：方案A 首轮实测——正确性等价，性能首轮判负，L0 双稳待复跑

- 结果（tp_run_20260716_063155，卡组 0-3）：基线 1709 | **L0 warp med 3443/min 1745
  （进程级双稳）** | L1 warp 1897（慢 11.0%，稳定）| 双开 1930（慢 12.9%）|
  双开 T=1024 3434（vs 2932 慢 17.1%）。03w8 正确性 rel_err 4.27e-2 与默认路径
  同水位——协议移植正确，问题纯在性能。
- 关键观察：L0 warp 在 bothwarp 进程里稳定处于快态（1930 ≈ 1897+33），单开进程
  多数迭代 ~3.4ms——同 kernel 进程级双稳，clocks 排除外部干扰。min=1745 仅 +36µs，
  说明快态下方案A 的账基本成立，问题是快态不可靠。机制两候选：①满负荷 mma 下
  warp 8 发射饥饿 → 已领取 token 成 straggler → 行块 gate 车队；②comm lane 的
  peer pull 与 producer 的流水加载共享同 SM TMA 队列（HoL）。教训：同 SM 混跑时
  资源耦合从"SM 数"变成"发射槽+TMA 队列"，后者不是零成本。
- L1 warp 判负机制清楚（推流人手减半 + 每 job prered 慢 8×，尾巴增长超过 GEMM
  满 SM 收益），与 docs/35 "L1 v1 近最优"一致，建议定案回 v1。
- 本轮 08f stages full_run 3983 与同 session bench 1709 严重失配，stages 数字
  本轮不采信（比 docs/41 的失真大得多，原因未查）。
- **定位工具已落码（第二批提交）**：`tools/diag_warp.py` + kernel 探针
  `moe_tp_dispatch_gemm_fp8_warp_probe`（gate_off + num_slots 旋钮），逐迭代打印，
  一次把融合税拆成 满SM纯GEMM上限 / 共存税（发射槽+TMA队列，机制②读数）/
  gate-straggler 税（机制①读数），L0/L1 各一套阶梯 + comm lane 数 sweep。
- **下一步（上机）**：`STEPS='^00_|^01_|^08w8_' bash tools/run_tp_all.sh`（01 必跑，
  tk_moe.cu 有新探针）。裁决树与修复方向映射见
  docs/归档/2026-07-16_方案A_通信warp化设计与落地.md §下一步。

## 2026-07-16：方案A 通信 warp 化落码（分支 comm_warp，待上机）

- 动机：归因显示 L0 暴露 ≈100% 是 SM 让渡税（fp8 ~264µs）、L1 让渡 +142µs（bf16 实测），
  而 comm 块的工作本质是异步 DMA 编排，不需要整块 SM。方案A 把通信角色降为 fp8 dispenser
  里 producer warp（warp 8）的闲置 lane 1..31（该 warp 只有 lane 0 发 GEMM TMA），GEMM
  拿满全部 SM；线程数/launch_bounds/寄存器分配/GEMM 模板全部零改动，协议与默认路径逐字节
  同构。依赖 sm120 ITS 让 lane 0 自旋与其余 lane 通信并发（头号裁决假设）。
- 落码：`tpdisp8::kernel_warp`（L0：lane 1..4 各管一个 in-flight token，独立领取/专属
  semaphore，无 warp 内同步）+ `tppr8::kernel_warp`（L1：31-lane comm 团队 GEMM 下流推 +
  8 consumer warp 出 GEMM 后经 named barrier 2 汇合排空，两团队共用 job dispenser）；
  绑定 `moe_tp_dispatch_gemm_fp8_warp` / `moe_tp_gemm_prered_push_fp8_warp`；scheme 开关
  `TK_L0_WARP`/`TK_L1_WARP`（默认 0，与 CE 互斥，L1 需 TK_L1_FP8=1）；脚本步骤 03w8 + 04w8
  三档 A/B + t1024。设计与风险清单：`docs/归档/2026-07-16_方案A_通信warp化设计与落地.md`。
- 本地已过：py_compile、bash -n、preflight_tp_cpu（本机无 CUDA，kernel 未编译）。
- **下一步（上机）**：`STEPS='^00_|^01_|^03f8_|^03w8_|^04f8_|^04w8_|^08f_' bash
  tools/run_tp_all.sh`——01 必跑（tk_moe.cu 变更需重编译），03w8 正确性门过了看 04w8
  阶梯 vs 04f8 基线（1708）。裁决点：①GEMM-alone 是否被 producer 分歧拖慢 >2%；
  ②L0 暴露是否回升（pull 并发 440 vs 480，不够把 WARP_SLOTS 4→6-8）；③L1 尾是否回升
  （推流线程约现役一半，wire 瓶颈则无影响）。e2e 目标 ~1500-1550。

## 2026-07-16：性能收益统一为相对 baseline 的耗时降低

- 新的唯一报数口径为 `(baseline_time - candidate_time) / baseline_time * 100%`；
  正数表示耗时降低，负数表示比 baseline 更慢，不再计算或输出加速比。
- `mb1_compute.py`、`mb3_ratio.py`、`mb4_fusion.py` 和 `verify_fp8_gemm.py` 已改为
  `time_reduction_*_pct` 字段和百分比输出；当前文档、计划和项目博客的自有性能数字已同步换算。
- `microbench/results/`、`docs/归档/` 和本文下方的旧迭代记录保留生成当时的历史字段/表述，
  不回写旧实验产物；从本条记录开始，所有新结果必须使用耗时降低百分比。

## 2026-07-16：明确“平台事实”的机器与拓扑作用域

- `docs/04_平台边界与负结果.md` 将结论拆成架构约束、机器/拓扑实测、软件栈/工作负载裁决三层。归档带宽、RTT、`comm_sms=24`、CE/Comet-N 等主要来自早期 16 卡服务器，不能无复测迁移到当前 8 卡环境。
- 新增换机最小定标清单：拓扑/P2P、四卡并发 push/pull/CE/RTT、单卡 BF16/FP8/raw-cap、主配置 comm sweep/阶段归因和关键负结果 A/B；新结果必须进入新的 run manifest。

## 2026-07-16：docs 文档收敛与历史归档

- 将原 `docs/01~46` 完整移入 `docs/归档/2026-07-08_至_2026-07-16_迭代记录/`，原始实验、负结果和复现命令均保留；历史正文中的 `docs/NN` 统一解释为归档目录中的同编号文件。
- 新增 `docs/README.md` 总入口，并按当前状态、实现架构、性能方法论、平台边界与负结果、测试调试五个长期主题完成收敛。新 session 应先读 `docs/README.md` 和本文件顶部，不再把某一篇历史轮次文档当作当前任务清单。
- 后续当前结论直接维护到主题文档；单次实验参数进入 `configs/runs/`，长过程记录进入 `docs/归档/`，避免再次形成顶层编号碎片。

## 2026-07-16：沉淀 ThunderKittens 融合 kernel profiling 方法

- 新增 `experience/13_ThunderKittens融合Kernel性能分析.md`，整理端到端 CUDA Event、torch/Chrome trace、nsys、NCU 与设备端 `TKProfiler` 的四层下钻流程，并补充多进程 `torchrun`、application replay、Source/SASS 和 marker 设计注意事项。
- 源码审计确认 TK 原生 `TKProfiler` 当前仅在 `KITTENS_SM10X`（SM100/103）编译，仓库融合 kernel 尚未接入；本项目 SM120 应先用 nsys/NCU，需要内部阶段时间戳时实现普通 global-store 的轻量后端，不能直接删除架构宏。

## 2026-07-16：核心性能矩阵复用四进程/NCCL 生命周期

- 新增 `distributed.run_distributed_suite` 和 `tools/run_tp_bench_suite.py`：主配置显式读取 `configs/tp_rtx_pro5000_4gpu_fp8.yaml`，23 个核心性能 case 只启动一次四 worker，并缓存同 shape/precision 权重；每 case 仍独立 setup/close、恢复 `TK_*` 环境并输出原 JSON 文件名。
- `tools/run_tp_all.sh` 默认 `REUSE_BENCH=1`，核心性能步骤合并为 `04_bench_suite_reuse`；`REUSE_BENCH=0` 恢复逐 case 隔离，`STEPS`/`FOCUS` 默认自动回退。正确性、schedule 和 stage attribution 保持独立；runner 默认 conda Python 和自动卡组也已对齐当前 8 卡环境（0–3/4–7，并校验 index 存在）。
- 本地已过 py_compile、23-case plan/输出唯一性/环境覆盖校验；Windows 无 vLLM/CUDA，待上机先跑 `QUICK=1`，异常时用 `REUSE_BENCH=0` 对照。当前操作说明见 `docs/05_测试与调试指南.md`，原始实现记录归档为 `docs/归档/2026-07-08_至_2026-07-16_迭代记录/46_TP性能矩阵单进程复用.md`。

## 2026-07-16：主测试配置写入协作规则

- `AGENTS.md` 已明确所有测试先以 `configs/tp_rtx_pro5000_4gpu_fp8.yaml` 为唯一默认口径；通用 benchmark 必须显式传 `--config`，专用脚本不能直接读取 YAML 时必须逐项对齐。
- A/B 只能最小化覆盖实验字段，覆盖项必须写入 run manifest、结果目录和 `HANDOFF.md`；`.gitignore` 同步忽略本地回流的 `tmp/` 结果目录。

## 2026-07-16：持久化 tp_run_20260715_124912 测试配置

- 新增 `configs/tp_rtx_pro5000_4gpu_fp8.yaml`：可由 `MoEBenchConfig` 直接加载的 RTX PRO 5000 四卡 TP FP8 正式主工作负载（H=4096、I=3072、E=64、TopK=8、T/rank=512、warmup=20、FP8 block 128x128）。
- 新增 `configs/runs/tp_run_20260715_124912.yaml`：保存该轮 git/环境/卡组/QUICK runner 参数，以及 44 步 BF16/FP8、CE、COMM SM sweep、正确性和分阶段计时矩阵；严格重放需切到记录的 commit 后执行 manifest 中的命令。
- 注意：该历史 run 使用卡组 `0,1,2,3`、未调优 vLLM serial config，且日志存在 Triton 导入错误；这些限制已写入 manifest，不能把结果直接当作推荐拓扑下的最终数值。

> 新 session 从这里接手。先读本文顶部 + `docs/README.md`；下面内容是按时间倒序保留的历史记录。

**最后更新**:2026-07-11(TP 分支 tp_test 第二轮:**首轮实测正确性全绿但性能 0.73× serial
(3613 vs 2624µs),归因为两处调度串行**(docs/20):L0 ring 拉取序使 GEMM 停在整个 AG 后 +
L1 每 job 一块在 16 comm SM 上排 128 波。已修复:pull_order(min-slot 序,链路并发+边到边算)
+ 全员 dispenser(gemm_push_kernel_tp,job_order 就绪序,comp 块跑完 GEMM 加入排空)+
分阶段归因工具 time_tp_stages(一键脚本 step 08)。**等第二轮实测**。注意:TP serial 通信占比
仅 ~23%,重叠天花板 ≈2.2ms,收益结构性低于 EP;大 NE 是相对机会(serial 随 NE 恶化)。
首轮落地记录见 docs/19。
**【FP8 收官前 2026-07-12·分支 fp8_tp·当前状态】**sched int32 兑现:
**T=512 = 1708(1.21× vs serial fp8 2074)、T=1024 = 2970(1.36×)**;
02 裁决过 int32 key 变更;08f 本次被干扰打飞(弃用,bench 稳)。距平台合理
上限 ~1660 仅剩 final_red(−20)/L1 尾(−30)——**主优化线收官**。
fp8 全程:2074(serial)→1920→1879→1756→1708;T=1024 对 serial bf16 已 1.69×。
**待办:①全量回归 `bash tools/run_tp_all.sh`;②数值口径决策(rel 4.28e-2
vs 容差 3.5e-2:放容差或消 w1 二次量化);③终数报告(双口径+平台结论汇编:
fp32 累加税/CE 判负/Comet-N 判负/让渡税/粒度纪律)**。docs/45。
**【FP8 P3 兑现·CE 判负(历史)】**
**P3 兑现:T=512 = 1756(1.18× vs serial fp8)、T=1024 = 3017(1.33×)**,
L1 fp8 净赚 −129 与预算吻合;数值 rel_err 4.28e-2(2% 元素超容差,口径决策
待定:放容差或消 w1 二次量化)。**CE 负结果定案(docs/44 §2)**:L0CE +236/
L1CE +450/双 CE +609,拐点 sweep 全程劣于 SM 版——CE 把 token 粒度消费序
流水退化成分片大块搬运,违反粒度纪律;本机结论:SM 驱动细粒度流水 > CE,
代码留档默认关。**sched int32 瘦身已落码**(三处 argsort key int64→int32,
值域校验安全,预期 268→~220)。当前账:1756,余量 sched −40~55 + final_red
−20 + L1 尾 −30 → 收官预计 ~1660-1700(1.22-1.25×)贴平台上限。
**下一步:`STEPS='^00_|^01_|^02_|^03f8_|^04f8_|^08f_' bash tools/run_tp_all.sh`
(02 必跑:builder key 变更)→ 全量回归 + bf16/fp8 双口径终数**。docs/44。
**【FP8 CE 死锁修复(历史)】**CE 首测
03c8/04c8 死锁(挂满 600s 超时):**持久 kernel 占满 110 SM 自旋等 flag,而
cudaMemsetAsync/同设备 memcpy 是小 kernel 形式,排不上队 → 互相等死**(经典
坑,已入踩坑索引)。修复三处:①自己分片不经 CE(kernel 直读本地 pre_tokens,
省一次搬运);②CE 模式 grid = sm−1(留 1 SM 给辅助 kernel);③final_reduce
改 grid-stride + 栅格限 sm−2(防同类饿死)。其余计划不变(docs/43)。
**下一步:杀掉挂着的脚本,git pull 后重跑同一 STEPS 命令**。
**【FP8 CE 通信+TK 审计(历史·死锁已修)】**用户指示:
通信改 copy engine(0 SM,打破 docs/35 零和)+ TK 原语审计。**审计(docs/43
§1)**:5 处 bar.sync asm → group<N>::sync(2) ✅;host CE = raw_ptrs_+side
streams(club 的多进程等价);pcie_sync/量化/policy 保留有据。**CE 已落码**:
ce_ag_pull(L0:CE 拉分片进本地 ag 缓冲+flag,kernel 只本地 scatter)、
ce_rs_push(L1:out_planes CE 推对端+4B watermark,final_red 零改动)、
ce_rs_fence(跨迭代护栏);TK_L0_CE/TK_L1_CE 默认 0,04c8 三档 A/B +
05c8 CE 拐点重扫 {4,8,12,16} + 08c8 归因。P3(L1 fp8)同待实测。预期:
L0 CE 回收让渡大头(935→~800),L1 CE 账面平衡由数据裁决;e2e 目标 ~1650-1700。
**下一步:`STEPS='^00_|^01_|^03f8_|^03c8_|^04f8_|^04l8_|^04c8_|^05c8_|^08f_|^08c8_'
bash tools/run_tp_all.sh`**。docs/42/43。
**【FP8 P3 落码(历史)】**量化 kernel 兑现
(tok_copy 112→24,e2e **1879/3180**)。**未隐藏通信定量账(docs/42 §1)**:
L0 AG≈0(全藏)、L1 尾 108、final_red 20、sched gather ~40、屏障 ~20 =
**~190µs(10%)**;另让渡税 264(平台结构性)。合理上限 ~1560-1600
(1.30-1.33×),余 ~300。**P3 已落码**:tppr8(fp8 dispenser W2 GEMM +
v1 push 逐行同构;**w2/w2_scale 原样可用零转置零重量化**)+ act 量化复用
rowgroup kernel;TK_L1_FP8 开关 + 04l8 A/B;stages L1 fp8 感知(cm/nb 置 0)。
预期 L1_fused 696→~560,e2e ~1770-1800(1.15-1.17×),T=1024 ~3000(1.34×)。
**下一步:`STEPS='^00_|^01_|^03f8_|^04f8_|^04l8_|^08f_' bash
tools/run_tp_all.sh`**;过了做 C(sched 双流)+ 全量回归 + 双口径终数。docs/42。
**【FP8 A 证伪+量化 kernel(历史)】**
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
平台:16×RTX Pro 5000(sm120)PCIe,用 4 卡组(优先 8,10,12,14),无 NVLink、
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

- **持久 kernel 满铺 SM + 依赖 host 侧辅助操作 = 死锁**(docs/44):
  cudaMemsetAsync 与**同设备** cudaMemcpyAsync 在部分实现里是小 kernel,
  被占满 SM 的持久 kernel 饿死 → flag 永不落地互相等(CE 首测 03c8/04c8
  hang 10min 超时)。解法:CE 模式 grid 留 1 SM、自旋型 kernel(final_red)
  用 grid-stride 限栅格、自己设备的数据直读不经 memcpy。
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
CUDA_VISIBLE_DEVICES=8,10,12,14 python -m moe_bench.tools.run_tkfused 64
# benchmark(bf16 EP;--distributed 必带,--scheme 单值,分别跑 tkfused / serial)
CUDA_VISIBLE_DEVICES=8,10,12,14 python -m moe_bench.bench --distributed --scheme tkfused \
    --mode ep --precision bf16 --world-size 4 --no-verify --num-tokens 512
# dispatch-only 隔离计时
CUDA_VISIBLE_DEVICES=8,10,12,14 TK_DISPATCH=push3 python -m moe_bench.tools.time_dispatch 256 10 50
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
