# 25 TP 第五轮：首次超 serial（1.11~1.15×）、push 冻结、下一步

> 第五轮（tp_run_20260711_144047）：**首次全绿**（pull+push 两路径正确性
> 全档 ok，29/32 步过，仅 time_tp_stages 因工具签名 bug 未跑成）。
> docs/20 的调度修复（pull_order + 全员 dispenser）首次真正执行并兑现。

## 1. 结果（4×RTX Pro 5000, 卡组 9,11,13,15, bf16, fair 口径）

| 配置 | serial | tktp(pull) | 加速 |
|---|---:|---:|---:|
| NE=64, T=512, comm=16 | 2630 | **2363** | **1.11×** |
| NE=64, T=512, comm=24 | — | **2290** | **1.15×** |
| T=1024 | 5024 | 4325 | **1.16×** |
| T=256 | 1670 | 1712 | 0.98× |
| NE=128 | 3279 | 3117 | 1.05× |
| NE=256 | 5285 | 5451 | 0.97× |

comm_sms sweep（pull）：4→4047, 8→2751, 16→2366, **24→2290**（未收敛）。
对比第二轮（修复前）：3613 → 2363，**docs/20 的两项修复共砍 1250µs（35%）**，
波数账的归因兑现。

push 路径（TP-T1）：正确性全过（含 NE=256、skewed），但慢于 pull：
默认 (p4,c16) 2511；(p4,c8) 4234、(p4,c12) 2869、(p2,c8) 3320。

## 2. push 为什么输给 pull（对 docs/23 预期的修正）

docs/23 依据 mb 事实（pull 并发 23.5GB/s 弱 vs push 50.9GB/s 强）判断
"push 化必选"。实测推翻，归因两条：

1. **TP 这个 shape 是 GEMM-bound，通信路径强弱不在关键路径上**。AG 只有
   12.6MB：pull 即使按并发弱路径 23.5GB/s 也只要 ~540µs，完全藏在
   ~1.5ms 的 gate+up GEMM 下。mb 的 push/pull 带宽差只在 **comm-bound**
   （EP dispatch 100MB 级）时兑现——"收益天花板 = min(T_comp, T_comm) 受
   通信占比约束"（experience/01 §10）在方案选型层再次生效：**先判 bound
   类型，再选数据面**。
2. **push 路径的 scatter 是 block-per-token 粒度**（每轮 2×__syncthreads +
   顺序单 token），又踩了 docs/21 §4 反模式的轻量变体：sweep 对 scatter
   块数极端敏感（4 块 4234 → 12 块 2505），说明瓶颈整个在 scatter 排空，
   而 pull 路径的 12-lane TMA 流水天然无 block-wide 同步。

**处置：push 冻结**（`TK_TP_DISPATCH=push` 保留，脚本留单点回归），不再
投入。若未来 shape 变为 comm-bound（fp8 前更大 H、或更小 intermediate），
再解冻并先修 scatter 粒度。

## 3. 档位分析与下一步

- **NE=256 输（0.97×）的主因是 padding**：balanced 下每 expert 64 个真实
  token 被 pad 到 128 → P=32768，**GEMM 算了 2× 的行**。serial 的 triton
  按 16/32 对齐没这个浪费。→ **T5 的 TK_ROW_BLOCK=64 开关**（P 归 16384、
  零 padding），EP 上 NE=256 净赢 ~260µs（docs/16），TP 下 GEMM 占比更高、
  预期收益更大；但警惕 docs/16 的小 tile 效率折损，实测裁决。
  脚本已加 07b（rb64 预编译防并发踩踏 + 对拍 + bench）。
- **comm_sms 拐点未到**：24 仍在降。已把默认提到 24，下轮扫 8~40。
  注意 comm 块同时服务 L0 dispatch 队列和 L1 dispenser——收益到底来自
  哪层，等 time_tp_stages 归因（签名 bug 已修：make_problem 无 device 参）。
- **T=256 打平（0.98×）**：GEMM 便宜后 sched ~200µs + 固定开销占比大。
  低优先级（产品点在 512+）。
- **TP-T3（triton 调优基线）**：报终数前必须做——当前 serial 是未调优
  triton（docs/22 P3），tktp 的 1.1~1.16× 有虚高风险。

## 4. 与预期账对数（docs/21 §3 纪律）

docs/23 §1 预测 pull 路径"难赢 serial"（预估 2.8-3.0ms）——**实测 2290，
预测偏悲观 ~600µs**。偏差来源：mb 的 23.5GB/s 是持续满负荷并发数字，我们
的 AG 是突发且与 GEMM 交错；k=16 让渡 8~16% 的 mb 数字来自纯 GEMM 工况，
融合 kernel 里 GEMM 本身有 gate 等待，让渡的边际代价更小。教训：**微基准
外推到融合工况要打折，bound 判断比带宽数字更可靠**。

## 5. 当前最优配置（记录在案）

```
TK_TP_DISPATCH=pull  TK_COMM_SMS=24  TK_GPU_SCHED=1  ROW_BLOCK=128
NE=64/T=512: 2290µs vs serial 2630 = 1.15×（fair，未调优 triton 基线）
```
