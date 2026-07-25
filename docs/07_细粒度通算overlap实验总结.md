# 细粒度通算 Overlap 实验总结（2026-07-13 至 07-19）

## 1. 范围

本文合并原来两篇 07 文档，汇总 2026-07-13（周一）至 2026-07-19（周日）四卡 TP FP8
MoE 的通信组织实验：本周主线不是继续把 GEMM 拆成 chunk 放到不同 stream 上做粗粒度
overlap，而是在"通信编排进 persistent kernel"的融合框架不变的前提下，连续裁决四种
通信组织方式——方案 A（通信 warp 化，判负）、P1（专职块 per-lane 拉取，小幅有效）、
P2（源端 push + 收端本地 scatter，当周最好且后来固化为当前实现）、P2.5（warp 协作
scatter，未定案）。期间两次整机 wedge 推动形成了死锁兜底体系。

本文数字是该周专项 A/B 实验记录（warmup=5，T=512 计 50 次/T=1024 计 30 次，性能档
关闭 verify、另跑 10 次正确性门），不是按当前主配置 `warmup=20` 重新生成的最终报告，
仅用于同轮内部比较，当前正式口径见 [01_当前状态与结论.md](01_当前状态与结论.md)。
全文统一使用耗时降低口径 `(baseline - candidate) / baseline × 100%`，正数表示候选
更快，各节的 serial 锚点按当轮 run 记录使用的数值计算，不跨轮次混用。FP8 正确性
相对误差全程约 `4.27e-2~4.28e-2`，各实验同一水位说明新协议未引入额外误差，但仍高于
旧的元素级 `3.5e-2` 容差，这条口径分歧独立于本文性能结论，详见
[04_平台边界与负结果.md](04_平台边界与负结果.md)。

## 2. 实现思路

目标是在四卡 PCIe TP MoE 中把通信直接编排进 MoE kernel，以 token 和 GEMM tile 为
粒度推进通信与计算，而不是把 GEMM 拆成 chunk 后放到不同 stream 中做粗粒度 overlap。

### L0：AllGather 与 W1 GEMM 重叠

```text
本地 FP8 token → push 到对端 staging buffer(单写者) → release flag 通知到达
  → 目标 rank 本地 scatter 到 8 个 expert slot → 更新 row-block ready counter
  → 对应 GEMM tile 立即开始计算
```

每个 rank 把本地 token/scale 按目标 rank 消费顺序直接 push 到对端 staging plane；
写完后用 system fence + release store 发布 per-token flag，目标 rank acquire 等待
后只访问本地 staging，GEMM block 只等待自己负责的 row block 就绪；通信 block 完成
搬运后转岗执行剩余 GEMM task，避免通信结束后 SM 空闲。

### L1：W2 GEMM、预归约与 ReduceScatter 重叠

```text
W2 GEMM row block → 本地 top-k pre-reduce → push 到目标 rank
  → 发布 watermark/完成信号 → 目标 rank 收到后立即最终归约
```

每个 row block 算完即可进入预归约和通信，不需要等全部 expert GEMM 结束；目标 rank
按到达顺序消费远端结果，W2 计算、本地归约、跨卡传输和最终归约因此能连续流水。

### kernel 内的角色分工

融合 kernel 中的 SM 动态承担两类角色：通信 SM 负责 token push/scatter、完成信号和
row-block counter；计算 SM 从全局 task dispenser 领取已满足依赖的 GEMM tile。通信
SM 完成任务后通过 named barrier 汇合转岗为计算 SM，不需要额外通信 kernel，也不依赖
多 stream 的 chunk 切分。

## 3. 实验一：方案 A——把通信降到 GEMM producer warp（判负）

### 3.1 原理

旧实现为 L0/L1 各保留 24 个专职通信 block，归因显示 L0 的主要暴露不是 wire 等待，
而是通信 block 占用 SM 后导致 GEMM 少用 24 个 SM 的"让渡税"。FP8 dispenser GEMM 每
block 有 9 个 warp，其中 warp 8（producer）只有 lane 0 发本地 GEMM TMA，lane 1..31
基本闲置。方案 A 把通信任务塞进这些闲置 lane（L0 用 lane 1..4 各维护一个 in-flight
token，L1 用 31 个 lane 在 GEMM 运行期间推送已完成 job），希望回收让渡税且不增加
线程数和寄存器上限。

### 3.2 首轮结果（`tp_run_20260716_063155`，卡组 0-3）

| 档位 | median (µs) | min (µs) | 相对对照 |
|---|---:|---:|---|
| 默认 FP8 | 1709 | 1703 | — |
| L0 warp | 3443 | 1745 | median 慢 101%；min 慢 2.1% |
| L1 warp | 1897 | 1883 | 慢 11.0% |
| L0+L1 warp | 1930 | 1910 | 慢 12.9% |
| L0+L1 warp，T=1024 | 3434 | 3412 | 对照 2932，慢 17.1% |

L0 单开出现进程级双稳（多数迭代约 3.4ms，少数约 1.75ms），L0+L1 双开时反而稳定在
快态；卡 0-3 满频、4-7 空闲，未见外部负载可解释该现象。

### 3.3 三分解探针裁决根因（`tp_run_20260716_070406`）

为区分"producer warp 发射饥饿"和"同 SM TMA 队列争用"，构造了 gate/slot 可控的探针
kernel，把融合耗时拆成纯 GEMM 上限、加 peer pull 后的共存税、再加 GEMM gate 后的
straggler 税三层：

| 探针档位 | median (µs) | 解释 |
|---|---:|---|
| 纯 FP8 GEMM | 752 | 跨口径参考 |
| L0 warp 几何、无 pull | 742 | warp 几何本身免费 |
| L0 warp + pull、无 gate | 968 | 共存税 +226 |
| L0 warp 完整融合，4 lane/SM | 964 | gate/straggler 税约 −4≈0 |
| L0 warp，2/1 lane/SM | 947/917 | 1 lane/SM 仍有 +175 共存税 |
| L1 纯 GEMM / warp 融合 / 默认融合 | 383/729/556 | warp 融合税 +301，默认 +173 |

结果排除了"发射饥饿→straggler→gate 车队"假设，坐实 **per-SM TMA 队列 head-of-line
blocking**：µs 级 peer TMA read 与本地 GEMM tile load 共用同一 SM 的 TMA 队列，慢
peer read 阻塞 GEMM 流水，共存税随 comm lane 数单调，1 lane/SM 仍高达 +175µs。默认
方案的通信和 GEMM 位于不同 SM，因此不受影响。平台结论：**把 peer TMA 塞进 dense
GEMM 所在 SM，比让渡专职通信 SM 更贵**；唯一可能翻盘的变体是用普通 `ld.global` 向量
加载代替 peer TMA 绕开该队列，但可能转而争用 LSU/MSHR，本周未验证。

方案 A 实现（历史开关 `TK_L0_WARP`/`TK_L1_WARP`）已随 slim 分支移除，复现需检出
`fp8_tp` 或 `comm_warp` 分支的历史提交。

## 4. 实验二：P1——专职通信 block 内 per-lane 自由领取

方案 A 证明 per-lane 组织本身可行，只是放错了资源位置。P1 保留"专职通信 SM + 完成后
转岗 GEMM"的 inter-SM 几何，把原来一波 20 个 lane 一起发 peer pull、在 block barrier
等最慢 RTT 的方式，改为每个槽线程独立 `atomicAdd` 领取全局 `pull_order`、不做块内
波同步；20 条自由循环按 stride-8 分散到 5 个 warp，避免挤在一个 warp 内串行发射。

`tp_run_20260716_073019`：

| 档位 | median (µs) | 结论 |
|---|---:|---|
| 波同步 @24 | 1709 | 对照 |
| lane @24 | 1681 | 快 28µs，耗时降低 1.6% |
| lane @16/@12/@8/@4 | 1775/1808/1904/2633 | 16 SM 已开始回升 |
| 波同步 @4 | 3814 | lane@4 快 1181µs |
| lane，T=1024 | 2831 | 对照 2935，快 104µs，降低 3.5% |

lane 在所有 `comm_sms` 档位都优于波同步，但拐点仍在 24 SM，"per-lane 化可把拐点压
到 8~12 SM"的假设被证伪：PCIe pull 是高 RTT 弱路径，单条 4KB FP8 token 行远未打满
带宽，需要约 `24 block × 20 slot = 480` 个在飞
请求才能隐藏往返延迟，24 SM 是真实并发需求而非波同步低效的补偿。本阶段最好结果相对
serial FP8 2074/4030µs 的耗时降低为 18.9%/29.8%，一度成为默认，但立即触发 P2：要减
少通信 SM 数，必须从 pull 切换到无 RTT 往返的 posted write。P1 的 per-lane pull 数据
面代码已随 slim 分支移除。

## 5. 实验三：P2——源端 posted write，收端本地 scatter

P2 把 L0 数据移动方向从 pull 改为 push（原理见第 2 节），关键点是每个源 rank 只写
目标 rank staging 中属于自己的 plane（单写者）、`push_order` 与目标消费顺序一致、
seq 单调递增无需清零、收端只读本地 HBM 移除 peer-read RTT、依赖链
`GEMM ← scatter ← flag ← peer push` 跨 rank 无环。微基准显示 posted write 强路径
约 50.9GB/s，少量 SM 即可打满。

`tp_run_20260716_075314`：

| 档位 | median (µs) | 结论 |
|---|---:|---|
| push @24，push_sms=4 | 1629 | 比 lane 1681 快 52µs |
| push，T=1024 | 2777 | 比 lane 2831 快 54µs |
| comm_sms=6/8/10/12/16 | 2513/1875/1752/1749/1731 | 拐点仍未左移 |
| comm_sms=8，push_sms=2/4/6 | 1750/1875/2518 | 2 个 push SM 已饱和 |

相对本周起点 1709/2935µs 分别快 80/158µs；相对该 run 记录采用的 serial FP8
2074/4030µs，耗时降低 **21.5%/31.1%**。`comm_sms=24, push_sms=4` 稳定推荐；
`push_sms=2` 只在 `comm_sms=8` 档验证过，不能宣称全局最优。

虽然 wire 已退出关键路径，最佳 `comm_sms` 仍是 24：根因从 PCIe RTT 变为收端本地
scatter 的组织效率——每个 lane 串行处理一个 token，每 token 需要 8 次 TMA store 再
一次 `store_async_wait`，单 lane 周期约 10µs；scatter SM 从 4 增到 20 可回收约
246µs，与实测 sweep 一致。通信方向已经正确，剩余问题是本地 HBM 搬运的并行化，不再
是难以消除的跨卡往返延迟。P2 之后被直接固化为当前 `tktp` 实现的唯一路径（不再有
`TK_L0_PUSH` 这类开关），见 [02_实现架构.md](02_实现架构.md)。

## 6. 实验四：P2.5——warp 协作式本地 scatter（未定案）

P2.5 把"每 lane 串行搬一个 token"改为"每 warp 协作搬一个 token"：9 个 warp 独立
领取 `pull_order` 中的 token，lane 0 acquire 等待到达 flag，32 个 lane 每人搬运
128B 段并行写入 8 个 expert slot，数据路径改为纯 LSU 向量读写（不用 scatter TMA、
mbarrier 和 per-token `store_async_wait`），全员 `threadfence` 后由 lane 0..7 分别
发布 8 个 slot 的行块计数，目标是把 `comm_sms` 拐点压到 8~12，预计端到端可达
1530~1580µs。

`tp_run_20260716_081949` 的正确性门、端到端步骤和 `comm_sms={4,6,8,12,16,24}`
sweep 均记录为通过，但未保存该轮具体 JSON、summary 全文或各档 latency，因而无法
判断是否优于 P2 的 1629/2777µs、最优 `comm_sms` 是否左移。随后的阶段归因步骤触发
整机 wedge（见第 7 节），根因截至当周末未闭环。正式结论：**功能步骤有通过记录，
性能和稳定性均未定案**，当默认仍是 P2 的 push + lane scatter；P2.5 实现已随 slim
分支移除，复现需检出 `fp8_tp` 分支历史提交。

## 7. wedge 事故与死锁兜底

本周出现两次相关 wedge（P2.5 首测、后续阶段归因步骤）：协议 bug 或某 rank 先退出
→ 其他 rank 的 persistent kernel 在跨卡 flag/mbarrier 上无界等待 → 超时 kill 无法
抢占 kernel、进程进 D 态 → context 销毁/IPC unmap/NCCL teardown 卡在 RM/uvm 路径 →
`nvidia-smi` 和新 CUDA 进程全部排队 → 整机只能重启。RM GPU 组锁影响全机，即使实验
只用 0-3 卡，也可能让 8 张卡全部不可用。

本周由此推动完成四层兜底：设备侧自旋有界（`PCIE_SPIN_GUARD`，约 32s trap）、跨卡
mbarrier 有界（`pcie_sync::guarded_wait`，约 40s trap）、host 侧 fail-fast
（`distributed._fail_fast_exit`，跳过 synchronize/NCCL destroy/IPC unmap）、故障
注入验证（`tools/fault_inject_kill_rank.py`）。前三层当周完成实现和静态审计，第四
层的动态验证和 P2 性能回归当周末仍待执行。完整机理、边界和审计清单见
[06_死锁兜底体系.md](06_死锁兜底体系.md)。

## 8. 一周性能演进

| 阶段 | T=512 (µs) | T=1024 (µs) | 相对 serial 2074/4030 的耗时降低 | 状态 |
|---|---:|---:|---:|---|
| 本周起点：默认 FP8 pull | 1709 | 2935 | 17.6% / 27.2% | 对照 |
| 方案 A：同 SM 通信 warp | 1930（双开） | 3434 | 6.9% / 14.8% | 判负 |
| P1：专职块 per-lane pull | 1681 | 2831 | 18.9% / 29.8% | 有效，后被 P2 取代 |
| P2：posted write + 本地 scatter | 1629 | 2777 | 21.5% / 31.1% | 当周最好，后固化为默认实现 |
| P2.5：warp scatter | 数值未入库 | 数值未入库 | 不可计算 | 未定案，代码已移除 |

从本周起点到 P2，T=512/T=1024 分别再降低 4.7%/5.4%；更重要的是瓶颈性质变化：专职
block 波同步效率 → peer pull RTT/在飞并发 → posted write 后本地 scatter 吞吐。前两
项受 PCIe 弱路径约束较强，最后一项是可继续优化的本地搬运问题，GEMM 引擎本身的后续
优化见 [10_fp8主循环P1_K-tile128与寄存器预算.md](10_fp8主循环P1_K-tile128与寄存器预算.md)。

## 9. 结论与可复用认识

1. **不要只看"占了多少 SM"判断 overlap 是否划算。** 同 SM 放置通信可能释放 SM 数，
   却引入发射槽、TMA 队列、LSU/MSHR 等更隐蔽且更贵的资源竞争。
2. **通信方向比数据字节数更关键。** FP8 把 token 行减半没有降低 pull 所需 SM（瓶颈
   是 RTT）；改成 posted write 才真正移除往返依赖。
3. **per-lane 自由化能消除波内长尾，但不能创造带宽或降低 RTT。** P1 全档优于波同步，
   却无法把 24 SM 拐点左移。
4. **必须用分解探针裁决机制。** "纯 GEMM→加通信无 gate→完整融合"的阶梯直接把
   226µs 共存税与约 0µs gate 税分开，避免围绕错误假设继续调参。
5. **失败模式必须先设计，结果资产要与结论一起持久化。** 新协议首测要单步隔离，每个
   跨卡等待点都要有界 trap；P2.5 只留下"步骤通过"记录、没有具体 JSON，导致性能结论
   无法恢复，新 run 必须保存 summary/logs/JSON。

综合结论：persistent kernel 内同时保留通信/计算角色、源端 posted write、数据一到就
scatter 并执行 GEMM tile、通信 SM 完工后转岗，这条路径（P2）相对 serial 的耗时降低
当周达到 **21.5%（T=512）/31.1%（T=1024）**，并已固化为当前 `tktp` 实现。
