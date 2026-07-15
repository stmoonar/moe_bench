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

## 代码版本控制与经验持久化

在当前的分支下，每次有一些进展需要进行提交并写详细且规范的提交信息。并且在这个迭代的过程中遇到的问题和调优的经验，都可以分类沉淀到docs/下面的文档里（使用中文，且不同的问题开新的文档写，文档要整理好）。

每次的进度都持久化到HANDOFF.md文档里，方便之后新开session之后能继续接手。
