# HANDOFF — TK 通算融合 MoE 进度交接

> 这份文档只记"接手要知道的当前状态"。原理与账在 [`docs/`](docs/README.md)，
> 历史过程在 git（分支 `fp8_tp` / `tk_dev` 及其提交信息）。

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
4. 剩余优化杠杆（按归因排序）：A/B 改 2D TMA descriptor 绕开 4d/5d 的驱动 syscall；
   4×2 warp 几何 + smem GLU 配对减 LDSM；scale 走 smem；e2e 侧继续压未隐藏通信。

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
