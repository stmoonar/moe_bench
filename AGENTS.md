# AGENTS.md

## 目录说明

本目录下面是一个moe（混合专家）层的性能单侧，对比的baseline是vllm的串行实现，通信和计算分别lunch kernel串行实现。

## 环境说明

运行环境是一台8卡的RTX Pro5000（sm120架构），卡与卡之间通过PCIe连接。我们使用四张卡来测试。

每次开始运行测试前先看一下卡是不是空闲的，不要在有任务正在上面的进行的卡上跑测试。

尽量使用下面的四卡组合（按优先级）：
- 0,1,2,3
- 4,5,6,7

python环境(conda)：`/root/miniconda3/envs/vllm-td/bin/python`。

运行benchmark时必须在上一级目录（`/workspace`）执行 `python -m moe_bench.bench ...`，不能在 `moe_bench/` 目录内运行，否则会报 `ModuleNotFoundError`。

## 测试配置

每次开始测试时，先使用 `configs/tp_rtx_pro5000_4gpu_fp8.yaml` 作为主配置和唯一的默认口径。能够通过通用 benchmark 入口执行的测试，命令中必须显式传入 `--config moe_bench/configs/tp_rtx_pro5000_4gpu_fp8.yaml`。

专用测试脚本如果暂时不能直接读取该 YAML，运行前必须逐项核对其模型形状、并行方式、精度、每 rank token 数、warmup、迭代次数和正确性设置与主配置一致。A/B 实验需要覆盖配置时，只覆盖实验所需的最小字段，并把所有覆盖项记录到对应的 run manifest、结果目录和 `HANDOFF.md`，不能静默改变默认测试口径。

## 新增MoE实现

在bench里面新增一个MoE的实现（我们是要新增的实现是通算融合的MoE实现，不要去降级实现成GEMM kernel拆成chunk和通信在不同的stream上overlap，这样会降低计算的效率），参考ADDING_IMPLEMENTATIONS.md里的说明接入，并测试性能，需要保证结果正确性。

根据给的实现计划去做实现，不要偏离计划。

当前我们实现的版本和baseline都使用FP8的输入和权重来计算，token数512（每rank，总共2048个）。

## 死锁红线（每次写完/改完 kernel 代码必须执行，不可跳过）

2026-07-16 P2.5 首测把整机 wedge（持久 kernel 挂死 → 上下文销毁抱住 RM GPU 锁 →
nvidia-smi/新 CUDA 进程全部排队 → 只能重启宿主机）。为此立红线：

1. **提交前必须做死锁审计**：列出本次新增/修改的**每一个等待点**（自旋、named
   barrier、mbarrier、跨卡 flag/watermark/计数器），逐个回答"这个信号由谁生产、
   生产者在什么情况下永不到达"（包括对端 rank 崩溃/被 kill/超时被杀的情形）；
   跨 rank 的等待依赖必须构成无环链。审计结论写进提交信息。
2. **所有跨卡/跨块等待必须有界 + trap**：自旋用 `PCIE_SPIN_GUARD_DECL` /
   `PCIE_SPIN_GUARD_TICK`（sm120_common.cuh，~32s 超时 trap 杀 kernel →
   CUDA error → 进程干净退出）；**可能由跨卡数据喂的 mbarrier/semaphore
   （如 TMA pull 对端显存）用 `pcie_sync::guarded_wait` 代替裸 `wait()`**
   （try_wait + clock64 超时，唤醒延迟与 wait 相同）。禁止新增任何无界
   自旋或无界 mbarrier 等待。本地流水线 mbarrier（GEMM inputs/task 语义，
   生产者在同 kernel 内且会被 trap 连带杀死）可以保留裸 wait。
3. **新协议 kernel 首测必须单步隔离**：STEPS 只含该协议的正确性门一步，跑通后
   才允许进长矩阵；不允许把首测挂进多步骤批量运行。
4. **worker 出错后必须硬退出**：分布式 worker 捕获异常（含 kernel trap 后的
   CUDA error）后走 `os._exit`（distributed.py `_fail_fast_exit`），不做
   synchronize / NCCL destroy / 逐个 IPC unmap——那些调用会排队在被卡住的
   RM/uvm 锁后面。
5. 挂死后的恢复与取证流程见 `docs/04_平台边界与负结果.md` 的坑索引。死锁兜底
   的故障注入验证用 `tools/fault_inject_kill_rank.py`（杀一个 rank，验证其余
   有界退出 + nvidia-smi 存活 + 卡可复用）。

## 代码版本控制与经验持久化

在当前的分支下，每次有一些进展需要进行提交并写详细且规范的提交信息。并且在这个迭代的过程中遇到的问题和调优的经验，都可以分类沉淀到docs/下面的文档里（使用中文，且不同的问题开新的文档写，文档要整理好）。

每次的进度都持久化到HANDOFF.md文档里，方便之后新开session之后能继续接手。
