# 13 ThunderKittens 融合 Kernel 性能分析与 TKProfiler 使用

**整理日期**：2026-07-16

**源码口径**：`ThunderKittens` commit `02e9acbd8c330564357a9e2df929e938ac67d6d0`

**适用问题**：使用 TK 的 `pgl`、TMA、跨卡 signal/barrier 等原语编写通算融合 kernel 后，如何从端到端延迟逐层下钻到 kernel 内通信、等待和计算阶段。

## 1. 结论先行

ThunderKittens 仓库有三层 profiling 能力，但用途不同：

| 层级 | 仓库入口 | 能回答什么 | 不能回答什么 |
|---|---|---|---|
| 端到端计时 | `kernels/parallel/common.py::benchmark_*` | 整个融合算子延迟、TFLOPS、不同 `num_comm_sms` 的最优点 | kernel 内通信/计算各占多少 |
| Kernel 时间线 | `common.py::profile()`、`make nsys` | 每个 rank 的 kernel 起止、主 kernel 与 epilogue、跨 rank 拖尾 | 同一 kernel 内 comm CTA 与 comp CTA 的阶段拆分 |
| 单 kernel 深挖 | `make ncu`、Nsight Compute | 吞吐、occupancy、TMA/内存、warp stall，并关联 Source/SASS | 通常仍是整颗 kernel 聚合指标，不天然按 CTA 角色拆账 |
| Kernel 内插桩 | `include/pyutils/profiler.cuh::TKProfiler` | 每个 CTA/指定 leader warp 的内部时间点和等待区间 | 原生实现目前只支持 SM100/SM103，且插桩会扰动性能 |

最重要的使用纪律：

1. 先用无插桩版本确认端到端数字，再开 profiler 做归因；profile 数字不能替代 release 性能数字。
2. 先固定一个 shape、一个 `num_comm_sms`、一次目标 kernel，再做 nsys/NCU；不要直接 profile 仓库默认的完整 sweep。
3. 通信 CTA 和计算 CTA 若在同一 `main_kernel` 内通过 `blockIdx.x` 分流，nsys 只能看到一条 kernel；必须靠 NCU Source/SASS 或设备端时间戳继续拆。
4. 本项目目标机器是 RTX PRO 5000 / SM120；TK 原生 `TKProfiler` 被 `KITTENS_SM10X` 宏保护，**SM120 不能直接使用**，需要移植存储后端或写轻量时间戳缓冲。

## 2. 仓库里的 Profile 到底在哪里

### 2.1 端到端 CUDA Event 计时

`ThunderKittens/kernels/parallel/common.py` 提供：

- `benchmark_no_l2_clear()`：warmup 后用 CUDA Event 包住多次调用，返回平均毫秒数；适合稳定热缓存吞吐。
- `benchmark_l2_clear()`：每轮先刷新约 128 MiB L2 工作集，再分别计时；适合看冷缓存或降低跨轮缓存复用影响。
- `use_events=False` 时退化为 host `perf_counter + cuda synchronize`，只用于事件不适用的特殊路径。

融合调用可能连续 launch `main_kernel` 和 `epilogue_kernel`，Event 包住 Python callable 时会把两者都计入；这正是端到端算子延迟，但不是单独的主融合 kernel 延迟。

### 2.2 PyTorch profiler / Chrome trace

同一文件的 `profile()` 使用 `torch.profiler`，每个 rank 输出：

```text
rank_0_trace.json
rank_1_trace.json
...
```

可用 Perfetto 或 Chrome tracing 打开。现有 `ag_gemm/benchmark.py` 有两个容易误用的点：

- 主循环把 `do_profile=False` 写死；不开开关不会生成 trace。
- profile 分支构造的是 `(nccl_run(), tk_run())`，会把 NCCL baseline 和 TK 都放进同一 trace。只看 TK 时应把被测 callable 收窄到 `tk_run`。

这层适合找“哪颗 kernel 占大头”和“哪个 rank 拖尾”，不适合给同一 kernel 内阶段拆账。

### 2.3 Makefile 自带 nsys / NCU 目标

`ThunderKittens/kernels/common.mk` 已提供：

```bash
make nsys
make ncu
```

并默认加 `-lineinfo`，便于 NCU 把 SASS/warp stall 关联回 CUDA 源码。

但 `make ncu` 当前使用 `--replay-mode kernel --set full`，并直接包住 `torchrun`。对跨卡 barrier、peer memory 和自旋等待 kernel，有两个风险：

1. `torchrun` 的 CUDA worker 是子进程，缺少 `--target-processes all` 时可能抓不到目标进程。
2. Kernel replay 会重复单颗 kernel；如果其他 rank 没有按同样协议共同推进，可能改变通信时序甚至卡住。

因此，多 GPU 融合 kernel 应优先使用 application replay，且先只采少量 section 和一次目标 launch。

## 3. 推荐的四层下钻流程

### 3.1 第一层：无 profiler 的端到端基线

对同一 shape、同一数据类型至少测四个数：

```text
T_compute_alone
T_comm_alone
T_serial = T_compute_alone + T_comm_alone
T_fused
```

同时 sweep `num_comm_sms`。TK 的 `ag_gemm` 示例本身就在 sweep通信 SM 数，这能找到资源切分最优点，但不能说明时间花在哪个内部 wait 上。

基线测量必须满足：

- warmup 足够，barrier/信号在每轮前处于正确初值；
- 各 rank 同时进入测量区，并保存 per-rank 时间而不只看 rank 0；
- profiler、调试 store、`printf` 全关闭；
- compute-alone 尽量使用与融合版相同的计算 SM 预算，否则 overlap efficiency 会被资源差异污染。

### 3.2 第二层：torch trace / nsys 看多卡时间线

先把 benchmark 缩成单一 shape 和单一配置，再运行 `make nsys` 或等价命令。检查：

- 所有 rank 的主 kernel 是否近似同时开始；
- 是否有单 rank 明显晚结束，导致全局 barrier 拖尾；
- 主融合 kernel、epilogue/reset kernel、额外 memset/copy 的占比；
- NCCL baseline 与 TK 路径的 kernel 数量和 launch gap；
- 改 `num_comm_sms` 后主 kernel 是缩短还是计算被让出 SM 拖长。

对于 `ag_gemm`，`main_kernel` 内部按 `blockIdx.x < num_comp_sms` 分到 `comp_sm()`，其余 block 进入 `comm_sm()`。nsys 看到的是同一个 kernel span，不会画出两条 comm/comp 子时间线。

### 3.3 第三层：NCU 看瓶颈位置

一个更适合多卡 `torchrun` 的命令模板是：

```bash
ncu \
  --target-processes all \
  --devices 0 \
  --replay-mode application \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*main_kernel.*' \
  --launch-count 1 \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --force-overwrite \
  --export fused_main \
  torchrun --nproc_per_node=8 benchmark.py
```

这是思想模板，不保证 regex 与每版编译后的模板符号完全相同。应先从 nsys 报告复制实际 demangled kernel 名；TK 的 launcher 实际启动的是模板化 `global_kernel<..., main_kernel>`。

只采 GPU 0 是为了降低 profiler 干扰；其余 rank 仍需正常运行并参加通信。application replay 会重启整个应用，因此每轮必须能确定性初始化 peer buffer 和 barrier。

重点看：

| 代码角色 | 重点源码位置 | 重点指标/现象 |
|---|---|---|
| comm CTA | TMA load、peer store、`store_async_wait`、signal | L2/DRAM 吞吐、long scoreboard、memory throttle、等待写完成 |
| comp producer warp | 等远端 `g.barrier`、装载 A/B、推进 pipeline | barrier/membar stall、TMA 延迟、pipeline 空槽 |
| comp consumer warpgroup | WGMMA/MMA、`mma_async_wait`、epilogue store | Tensor throughput、eligible warps、occupancy、register/smem 限制 |
| 整颗 kernel | launch/结束与尾部 | SM Active、wave 数、最后 comm/comp CTA 拖尾 |

新版本 NCU 的 PM/Warp sampling 可以观察 stall 随 kernel 时间变化，并把 stall 关联到指令；但多数吞吐 counter 仍是整颗 kernel 聚合结果。comm/comp CTA 共用一个 kernel 时，不能把聚合的“Tensor 60%、DRAM 40%”直接解释成阶段时间占比。

不要一上来 `--set full`：它需要很多 replay pass，运行慢，且越容易放大跨卡协议和缓存状态差异。先靠少量 section 找最大瓶颈，再定向加 Scheduler、Source Counters 或 PM Sampling。

## 4. TKProfiler 的原理与正确接入方法

### 4.1 原理

`ThunderKittens/include/pyutils/profiler.cuh` 中的 `TKProfiler<TIMING_WIDTH>`：

- 在 shared memory 保存固定宽度的 `int timings[TIMING_WIDTH]`；
- `init()` 清零并读取 `%globaltimer` 作为 CTA 起点，所有线程必须在分流前调用；
- `record(i)` 由一个线程读取 `%globaltimer`，保存“当前时间 - CTA 起点”；
- `store_and_reset(global, i, j)` 把整段 shared timing buffer 写到 global memory；
- Host 用 `allocate(M, N)` 分配输出，用 `save()` 导出文本，再用 NumPy reshape。

`%globaltimer` 名义单位是纳秒，但 PTX 文档注明其行为依目标架构而定。TK 把 64 位时间差截断成有符号 `int`，因此不适合记录超过约 2.1 秒的区间；正常微秒/毫秒级 kernel 不受影响。

### 4.2 接入步骤

仅在 profile 构建打开编译宏，按以下顺序接入：

1. 显式 include `pyutils/profiler.cuh`；`kittens.cuh` 不会自动 include 它。
2. 选一个尽量小的 `TIMING_WIDTH`，并把 `Profiler::timing_t` 加到 kernel globals。
3. Host 在一次诊断 launch 前 `allocate(grid_blocks, records_per_block)`，将句柄传进 globals。
4. kernel 中声明一个 shared `Profiler`，所有线程在任何 comm/comp、producer/consumer 分流前调用 `init()`。
5. 每个时间点只允许一个明确的 leader lane 调 `record(slot)`；不同 CTA/warp 角色使用互不冲突的 slot。
6. kernel 收尾先做必要的 CTA 同步，再由一个线程 `store_and_reset()`。
7. kernel 完成后 `save()`，每个 rank 使用不同文件名；分析时对累计时间戳做相邻差分。
8. profile 完成后关闭插桩重新编译，用无插桩版本报告最终性能。

思想级伪代码：

```text
globals 增加 timing buffer
CTA 全员 profiler.init()

if comm CTA:
    leader 记录 comm_start / load_done / remote_store_done / signal_done
else:
    producer leader 记录 remote_wait_begin / remote_wait_end / tma_done
    consumer leader 记录 mma_begin / mma_done / epilogue_done

CTA 收尾同步
单线程把 timings 写到 [block_id, record_id, slot]
```

### 4.3 推荐 marker 布局

| Slot | comm CTA | comp producer | comp consumer |
|---|---|---|---|
| 0 | CTA start | CTA start | CTA start |
| 1 | local TMA issued | remote barrier wait begin | first MMA ready |
| 2 | local TMA ready | remote barrier wait end | last MMA issued |
| 3 | peer store done | A/B TMA issued | MMA wait done |
| 4 | signal done | A/B ready | epilogue store done |
| 5 | CTA end | producer end | consumer end |

不要让每个循环迭代反复覆盖同一个 slot。需要看稳态 pipeline 时，只记录选定的第一个、中间和最后一个 task；或者把 task 序号编码进独立 slot，但必须确保不超过 `TIMING_WIDTH`。

跨 warp 写同一个 shared profiler 是允许的前提是 slot 唯一；它能把 producer wait 与 consumer MMA 放在同一 CTA 时间轴上。跨 GPU 的 `%globaltimer` 不应默认严格同步，分析时先做每 rank/每 CTA 的相对时间，不直接拼绝对纳秒轴。

### 4.4 插桩本身的扰动

TKProfiler 会增加：

- static shared memory 和对齐空间；
- `%globaltimer` 读取与 shared store；
- kernel 尾部的 global memory 写；
- 为获得 CTA makespan 而加入的同步。

因此要同时保留两份数据：

```text
release build：端到端真实性能
profile build：阶段因果和相对占比
```

如果 profile build 比 release 慢很多，应减少 slot、只采样少量 CTA，或只采样指定 block（例如每类角色各一个首/中/尾 CTA）。

## 5. SM90 / SM120 的平台限制

原生 `profiler.cuh` 整体位于：

```cpp
#ifdef KITTENS_SM10X
...
#endif
```

即只覆盖 `KITTENS_SM100` 和 `KITTENS_SM103`。H100/SM90 与本项目 RTX PRO 5000/SM120 都不能直接实例化它。虽然 SM90/SM120 可以读取 `%globaltimer`，但原实现的 bulk shared-to-global store 路径没有为这些架构声明支持，不能只删除宏保护后假定正确。

本项目在 SM120 上推荐两种选择：

1. **先用 nsys + NCU Source/Warp Sampling**：零改 TK profiler，适合先定位 barrier、memory、MMA 哪类 stall 最大。
2. **实现 SM120 轻量后端**：保留 `init/record` 思路，shared memory 中存 `uint64_t` 时间戳，kernel 尾部由少量线程做普通 global store；输出按 `[rank, block, marker]` 排列。先做 correctness/canary，再评估插桩扰动。

第二种实现应放在独立 profile 编译分支或编译宏下，不进入默认 release fast path。

## 6. 从时间戳计算可行动指标

仅报告“通信用了 X 微秒”不够，还应计算：

```text
exposed_comm = max(0, T_fused - T_compute_same_budget)
overlap_efficiency =
    (T_compute_same_budget + T_comm_same_budget - T_fused)
    / min(T_compute_same_budget, T_comm_same_budget)
```

其中 `same_budget` 表示 standalone 对照尽量复现融合版的计算/通信 SM 配额。结果可能因融合额外开销而小于 0，也可能因缓存、布局或算子融合收益而大于 1；这时不能强行截断，应回到 NCU 和 marker 解释额外差异。

设备端 marker 进一步给出：

- `remote_wait_end - remote_wait_begin`：计算侧真正暴露的通信等待；
- `peer_store_done - local_TMA_ready`：peer 写和链路阶段；
- `mma_done - mma_begin`：计算主体；
- `CTA_end - max(comm_done, mma_done)`：epilogue/尾部同步；
- 各 CTA end 的 p50/p95/max：是否存在少数尾 CTA 拖慢整颗 kernel。

## 7. 最小执行清单

1. 固定单一 shape、dtype、world size、`num_comm_sms`。
2. 关闭全部 profiler，测 release 端到端和 standalone 对照。
3. 用 torch trace/nsys 验证 rank 对齐、kernel 数和主 kernel 占比。
4. NCU 用 `--target-processes all`、单 GPU、application replay、单 launch、少量 section。
5. 在 Source/SASS 中确认最大 stall 位于 comm、producer wait 还是 consumer MMA。
6. 只有 NCU 仍无法拆阶段时才开设备端 marker。
7. SM100/103 可接原生 TKProfiler；SM120 使用轻量移植后端，不能直接删除架构宏。
8. marker 文件按 rank/block 保存，分析累计时间戳的差分与尾延迟分布。
9. 关闭插桩重编译，最终数字只取 release build。

## 8. 源码与工具参考

- `../ThunderKittens/include/pyutils/profiler.cuh`：TKProfiler 定义和注释用法。
- `../ThunderKittens/kernels/parallel/common.py`：CUDA Event benchmark 与 torch profiler trace。
- `../ThunderKittens/kernels/common.mk`：`ncu`、`nsys` 目标和 `-lineinfo`。
- `../ThunderKittens/kernels/parallel/ag_gemm/`：同一 kernel 内 comm/comp CTA 分工的典型实例。
- `../ThunderKittens/kernels/parallel/moe_dispatch_gemm/`：dispatch + grouped GEMM 融合实例。
- NVIDIA Nsight Compute CLI：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>
- NVIDIA Nsight Compute Profiling Guide：<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- NVIDIA Nsight Systems User Guide：<https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- NVIDIA PTX `%globaltimer`：<https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer-globaltimer-lo-globaltimer-hi>
