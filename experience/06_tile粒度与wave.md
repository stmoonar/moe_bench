# 06 tile 粒度与 wave

> 主题：三种粒度（计算 tile / 通信 chunk / 信号粒度）如何分别选择，以及 wave 作为核心诊断指标的用法。

## 1. 三种粒度是三个独立变量

```text
计算 tile：   GEMM CTA tile（M_tile × N_tile × K_tile）——按 Tensor Core 效率选
通信 chunk：  一次传输的连续数据段——按链路效率（最小高效消息量）选
信号粒度：    一次同步覆盖的数据量——按"触发一次有意义的下游动作"选
```

- **计算 tile 永远按纯 GEMM 最优选，不被通信粒度绑架**。tile 尺寸、stage 数、cluster、调度模式（cooperative/pingpong）照常 autotune；通信只通过 prologue/epilogue/专职角色接入。
- **通信 chunk 与计算 tile 靠映射对接，不必相等**：一个 GEMM tile 可以跨多个 chunk（逐个等待），一个 chunk 也可以覆盖多个 tile。Flux 论文明确结论：comm chunk 无普适最优值，从"rank 分片"开始逐级减半向下搜索，直到接近 GEMM tile 尺寸。
- **信号粒度通常应粗于计算 tile**：per-tile 信号只有在"通信必须极早启动且信号开销确实低"时才值得；常规选择是 rank 分片（Comet layer0）、N-split 列块（Comet layer1）、wave group（FlashOverlap）、chunk（Syncopate）。

## 2. 每级粒度的量化经验

- 一次原子加对一个 128×128 tile（~32KB half）完全可忽略；信号的真实成本在**等待分支、global 流量、以及每段通信的启动开销**，随段数线性增长。
- 分段过多的下界效应：小消息区带宽曲线陡峭，段太小每段吃不满带宽——重叠收益 < 分段损失时，调优器应收敛到"不分段"（FlashOverlap 的 tuner 在小 shape 上确实收敛到单段，这是正确行为而非失败）。
- MoE 子问题粒度：每个子问题目标约 4 个 M-tile 的行数——太碎会增加 group 调度的 metadata 开销（Comet 的取值）。
- 列块（N-split）粒度：切 7~9 份量级（按 hidden 整除性选），不是越多越好。
- token 聚合粒度：凑满一个 GEMM tile 的行数（如 128 token）再放行计算。

## 3. wave：第一诊断指标

```text
单波容量 ≈ 可用 SM 数 × 每 SM 常驻 CTA 数
num_waves = ceil(tile 总数 / 单波容量)
```

用法按 num_waves 分档：

- **≤ 1 wave**：tile 级重叠基本无空间——计算没有"先完成的部分"。要么放弃融合（回退整段串行或 stream 粗重叠），要么改变切分维度制造 wave（token/sequence 切分）。小 M（decode、小 micro-batch、M 被 TP 切细）是重灾区；还会伴随次生问题：M 太小使输出写出效率崩塌（Flux 在 m=64 的 0.95× 倒退）。
- **2~4 waves**：有空间但要粗分段；首段小（尽早通信）、末段小（尾部暴露少）。
- **多 waves**：细分空间大，但尾波调度与 swizzle 的影响上升；分段边界对齐 wave 容量的整数倍。

两个必须建模的修正：

1. **让出 SM 后 wave 数会变**：给通信留 k 个 SM 后，计算 kernel 的实际 wave 数按 `ceil(T/(SM-k))` 重算——FlashOverlap 的预测模型显式含 `ceil(T/(SM-2))/ceil(T/SM)` 放大项，漏掉它模型系统性乐观。
2. **切分会增加总 wave 数**：任何切分方案先算"切分后各块 wave 数之和"是否超过原始 wave 数（TokenWeave 的 wave 对齐切分，见 03 篇）。

## 4. wave 完成顺序：验证，不要假设

- tile 的完成顺序 ≠ 内存地址顺序（swizzle 所致），也不保证每次运行一致。
- FlashOverlap 的准入协议值得照抄：**监控模式跑 10 次，记录每个 tile 的全局完成序；某 tile 10 次都落在同一 wave 窗口才判归该 wave；任何 tile 不一致 → 换 kernel 配置重试（top-5~10 候选）；全部失败 → 该 shape 判定不适用**。
- 完成顺序的可复现性来自"block→tile 映射固定 + 调度近似确定"，但会被抢占、MPS、cluster、persistent 调度改变——每换一类 kernel 结构都要重新验证。
- 若不想依赖完成序，可用"到达序驱动"的对偶方案：不预测完成序，而是让 tile 执行序跟随数据到达序（Syncopate），代价是需要 kernel 内动态取任务。

## 5. 平波手段及其代价

wave 量化严重（尾波利用率低）时的选项：

- split-K / stream-K：平衡 SM 利用，但**多 block 写同一 tile 会破坏"1 信号 = 1 tile 完成"的语义**——与 tile 级信号方案基本互斥（FlashOverlap 直接禁用），若必须共存要把信号改为"该 tile 的所有 partial 都完成"的计数语义。
- persistent 调度：任务粒度与 SM 解耦，尾波自然平滑，是与 tile 信号兼容性最好的选择。
- token/序列级切分：在更粗粒度上制造流水，不动 GEMM 内部。

## 6. 粒度选择的思想级流程

```text
1. 按纯计算 autotune 定计算 tile → 得 tile 总数与 num_waves
2. num_waves ≤ 1 ? → 换切分维度或放弃 tile 级方案
3. 按链路带宽曲线找最小高效消息量 → 定通信 chunk 下界
4. 信号粒度 = max(通信 chunk, 有意义的下游触发单位)，从 rank 分片/wave group 起步
5. 向下细化信号粒度，直到新增隐藏量 < 新增开销（每段启动 + 等待 + 带宽损失）
6. 用 10 次一致性实测验证所有关于完成序/到达序的假设
```
