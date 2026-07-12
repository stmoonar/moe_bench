# AGENTS.md

## 目录说明

本目录下面是一个moe（混合专家）层的性能单侧，对比的baseline是vllm的串行实现，通信和计算分别lunch kernel串行实现。

## 环境说明

运行环境是一台16卡的RTX Pro5000（sm120架构），卡与卡之间通过PCIe连接。我们使用四张卡来测试。

每次开始运行测试前先看一下卡是不是空闲的，不要在有任务正在上面的进行的卡上跑测试。

因为连续的两张卡连在同一个PCIe Switch下，因此为了最大化性能，尽量使用下面的四卡组合（按优先级）：
- 8,10,12,14
- 9,11,13,15
- 1,3,5,7
- 0,2,4,6

python环境：`source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate`。

运行benchmark时必须在上一级目录（`/data/cinnzhang_vllm_td_test/xxy`）执行 `python -m moe_bench.bench ...`，不能在 `moe_bench/` 目录内运行，否则会报 `ModuleNotFoundError`。

## 新增MoE实现

在bench里面新增一个MoE的实现（我们是要新增的实现是通算融合的MoE实现，不要去降级实现成GEMM kernel拆成chunk和通信在不同的stream上overlap，这样会降低计算的效率），参考ADDING_IMPLEMENTATIONS.md里的说明接入，并测试性能，需要保证结果正确性。

根据给的实现计划去做实现，不要偏离计划。

当前我们实现的版本和baseline都使用FP8的输入和权重来计算，token数512（每rank，总共2048个）。

## 代码版本控制与经验持久化

在当前的分支下，每次有一些进展需要进行提交并写详细且规范的提交信息。并且在这个迭代的过程中遇到的问题和调优的经验，都可以分类沉淀到docs/下面的文档里（使用中文，且不同的问题开新的文档写，文档要整理好）。

每次的进度都持久化到HANDOFF.md文档里，方便之后新开session之后能继续接手。

