# HANDOFF — TK 通算融合 MoE 进度交接

> 这份文档只记"接手要知道的当前状态"。原理与账在 [`docs/`](docs/README.md)，
> 历史过程在 git（分支 `fp8_tp` / `tk_dev` 及其提交信息）。

## 最新（2026-07-26 深夜）：L1 EPIRED（epilogue 直推加权归约）已实现（待上机验证）

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

**流水化已实现（待上机验证）**：push_job 双 buffer 2 在飞——TK 的
`store_async` 末尾自带 `commit_group`，每 job 自成一组；槽位复用前
`store_async_wait<1>` 只等最老一组并 retire（本地计数+选举 watermark），
与新 job 的 wait_slot 同阶段并行（retire=tid0，wait_slot 改 tid1..8）；
循环末按发出序 drain 两级（wait<1>/wait<0>）。watermark 语义不变（数据
落地才计数），只延后 ~1 个 job。死锁审计：wait_group 等本线程自己发出的
store（硬件事务必定完成），与旧版 wait<0> 性质相同，不新增跨卡等待类型；
对端崩溃的 PCIe 传染路径与现状一致（docs/06 兜底）。A/B 无开关，对照 =
父提交 `64e3b0b` 的 exposure 192.2。runbook：

```bash
cd /workspace
rm -rf moe_bench/kernels/tk/build && python moe_bench/kernels/tk/build.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.run_tktp --iters 10   # 正确性门(单步隔离; rel_err 应仍 ~4.28e-2 不恶化)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.time_tp_stages 64 20 512
# 协同 sweep(流水化后拐点可能下移):
for c in 12 16 20 32; do echo "== TK_COMM_SMS_L1=$c =="; TK_COMM_SMS_L1=$c \
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m moe_bench.tools.time_tp_stages 64 20 512 | grep -E "L1_fused|L1 exposure|full_run"; done
```

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
