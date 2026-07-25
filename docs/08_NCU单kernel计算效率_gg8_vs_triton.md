# NCU 单 kernel 计算效率：gg8（我们）vs triton fused_moe（vLLM serial）

日期：2026-07-23。单卡（GPU 4，RTX PRO 5000，driver 580.105.08），NCU kernel
replay，默认 clock-control（SM 锁 ~1.67GHz 基础频率）。目的：把 e2e 领先
（1629 vs 2074µs，−21.5%）里"GEMM 引擎本身"的贡献从"重叠结构"的贡献中剥离。

## 1. 方法与口径

两边都用**单卡纯 GEMM 探针**，剥掉一切分布式依赖（4 卡在跑的 kernel 没法用
NCU 按 kernel 采，见 §4 教训）：

- 我们：`tools/verify_fp8_gemm.py`（`gg8::kernel`，与融合 kernel 同一套
  `gemm_config_fp8` 引擎，无 comm 角色）；
- baseline：一次性探针脚本 `ncu_serial_gemm_probe.py`（复刻 SerialNaive 的
  `fused_experts` 一步，无 NCCL；triton `fused_moe_kernel` 每迭代两颗，
  launch-skip 6/7 分别采 w13/w2，grid 3828/10208 与 N=1536/4096 对应可验）——
  该探针脚本已随 slim 分支移除，结论保留在本文，复现需检出 `fp8_tp` 分支历史提交。

形状 = 主配置 rank 0 视角，routed 行数 16384（=2048 token × topk8 = 64
expert × 256 行）：

| 层 | GEMM | K（输入维） | N（输出维） | FLOPs |
|---|---|---|---|---|
| L0 | W13 gate+up | 4096 | 1536 | 206.2 GFLOP |
| L1 | W2 down | 768 | 4096 | 103.1 GFLOP |

命令模板（`verify_fp8_gemm` 参数序 = `[E] [rows/e] [K] [N]`）：

```bash
CUDA_VISIBLE_DEVICES=<idle> ncu --replay-mode kernel --kernel-name-base demangled \
  --kernel-name 'regex:gg8' --launch-skip 4 --launch-count 1 \
  --section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats \
  --export <out> python -m moe_bench.tools.verify_fp8_gemm 64 256 4096 1536 10   # L0
  #                                                        64 256 768 4096 10   # L1
# baseline: 同 sections, -k 'regex:fused_moe_kernel', probe 默认形状,
#   --launch-skip 6(w13) / 7(w2) --launch-count 1
```

⚠️ 锁频警告：NCU 默认把 SM 锁在基础频率，绝对耗时**不能**与 bench（boost）
数字直接比；本文只做同条件横比（利用率、相对差）。

## 2. 结果（2026-07-23 实测）

| 采样 | Duration | Tensor pipe 利用率 | DRAM | 折算 TFLOP/s@锁频 | 执行指令数 | IPC |
|---|---|---|---|---|---|---|
| gg8 @ L0 | 924µs | 60.9% | 46.8% | 223 | 208.5M | 1.26 |
| triton w13 @ L0 | **834µs** | **66.7%** | 45.8% | 247 | 159.6M | 1.06 |
| gg8 @ L1 | **519µs** | **53.9%** | 50.9% | 199 | 112.7M | 1.21 |
| triton w2 @ L1 | 537µs | 50.2% | 50.1% | 192 | 110.1M | 1.10 |

## 3. 结论

1. **GEMM 引擎与 triton 同档，e2e 领先全部来自重叠结构**。L0 我们 −10%、
   L1 我们 +3.4%，两层合计 1443 vs 1371µs（−5%）；再计入我们 L0 融合 GLU
   epilogue、serial 额外付 silu_and_mul + 层间 act 量化 kernel，基本平手。
   −21.5% 的 e2e 收益应归因于：AG 藏进计算 + fp8 AG 字节减半 + 转岗。
2. **L0 的 −10% 定位在发射带宽**：指令数 +31%（208.5M vs 159.6M），来源是
   per-2-step fp32 重标定（正比于 K 深度：K=4096 → 32 个量化块）+ dispenser
   管理。IPC 更高（1.26）但 tensor 占空比反而低 6 个点 = 非 MMA 指令抢发射槽。
   旁证：L1（K=768，6 块）指令数即与 triton 持平且反超。若日后要抠这 10%，
   方向是重标定摊薄（如 per-4-step，需验精度余量），但按当前瓶颈排序
   （本地 scatter、sched）优先级靠后。
3. **短 K 形状两边都掉档，triton 掉得更狠**（66.7→50.2 vs 我们 60.9→53.9）：
   短主循环下固定开销摊薄差。定量支撑 e2e 上 L1 fp8 净赚 −129µs 的合理性。
4. **指令选型无差异**：双方均为 QMMA + fp32 累加（tensor FP 子管道，
   imma=0）。~~60-67% 即 sm120 fp32 累加税（原归档 38，已随 slim 分支移除）下的
   可达水位带，
   指令级无翻盘空间~~ **此条已被 §5 的 CUTLASS 参照推翻**：同指令
   （SM120_16x8x32_TN fp8, fp32 acc）CUTLASS 在 L0 打到 84.9%，60-67%
   不是硬件天花板，是软件流水/重标定摊薄的欠账。

## 4. 过程教训（坑）

- **4 卡分布式下 NCU 采不了任何单颗 kernel**：kernel replay 的显存
  save/restore + 多 pass 期间，其余 rank 停在 NCCL device kernel 忙等自旋；
  kill 后残留无主自旋 kernel 把卡钉在 100% util（无进程持有，
  `fuser`/`ps` 均空），需 `nvidia-smi -r` 复位。融合持久 kernel 更是结构性
  不可采（需要四 rank kernel 并发喂数据，隔离即死，自旋 guard ~32s trap 会
  把对端杀干净）。**结论：单 kernel 计算效率一律用单卡探针**；真要采活的
  融合 kernel 只剩 `--replay-mode app-range` + `torch.cuda.profiler` 标记一条路。
- **`verify_fp8_gemm` 参数序是 [E] [rows] [K] [N]**，L1 是 `768 4096` 不是
  `4096 768`——K/N 颠倒后 FLOPs 相同、duration 恰为 L0 一半，表面合理实则
  访存形状全反，第一轮 L1 因此作废重跑。校验法：K=输入维、N=输出维，
  两层互为镜像（4096→1536，768→4096）。
- 容器无真 init（entrypoint 为 `sleep infinity`）时，被 kill 的 worker 会留
  永久僵尸（无害，不占 GPU）；容器重建时加 `docker run --init` 根治。

## 5. 2026-07-23 补充：CUTLASS 参照水位（同锁频口径）

用 CUTLASS 4.6.1 官方 sm120 kernel 立厂商可达上限（探针脚本 `tools/cutlass_probe/`
和 CUTLASS 子模块本身已随 slim 分支移除，结论保留在本文，复现需检出 `fp8_tp` 分支
历史提交；fp8 = examples/87c blockwise grouped GEMM 原样编译，与我们量化口径逐项
同构；bf16 = 2.x GemmGrouped+Sm80 mma.sync，皆纯 GEMM、无路由间接寻址）：

| 锁频口径 | L0 | tensor | L1 | tensor |
|---|---|---|---|---|
| CUTLASS fp8 (87c) | **664µs** | **84.9%** | **453µs** | 63.3% |
| triton fp8 | 834µs | 66.7% | 537µs | 50.2% |
| gg8（我们） | 924µs | 60.9% | 519µs | 53.9% |
| CUTLASS bf16 (2.x) | 1203µs | 92.3% | 708µs | 77.5% |
| triton bf16 | 1382µs | 79.6% | 826µs | 66.2% |

结论修正与新知：

1. **"60-67% 是累加税天花板"被推翻**。CUTLASS 用完全相同的 MMA atom
   （SM120_16x8x32_TN fp8，fp32 累加）在 L0 打到 84.9% tensor 占空比；
   bf16 甚至 92.3%（还是走 2.x Sm80 老路径）。差距在软件结构：
   （a）**scale 重标定摊薄**——CUTLASS blockwise 主循环每 128 深 K 块做一次
   accum promotion（`MainloopSm120ArrayTmaWarpSpecializedBlockwiseScaling`），
   我们每 2 个 MMA step 重标定一次 → +31% 指令抢发射槽；
   （b）TMA warp-specialized 双 producer 流水 + LDSM/swizzle smem 供数。
2. **相对位次**：L0 我们 = CUTLASS 的 72%、triton = 80%；L1 差距收窄
   （我们 87%、triton 84%），短 K 下大家都被固定开销/访存压住（CUTLASS
   L1 tensor 也只有 63.3%）。**主要欠账集中在 L0 长 K 形状**。
3. 口径警告：CUTLASS 探针是纯 GEMM——没有 sorted_token_ids 间接寻址、
   topk 加权、GLU epilogue。gg8 探针同样是纯 GEMM（可直接比），但 triton
   的数字里带路由 gather（含真实 MoE 开销），所以"路由感知 kernel"的真实
   可达上限略低于 664µs。即便如此，L0 的 260µs 锁频差距远超该扣减。
4. 优化方向（按 ROI）：**per-K-block(128) 重标定摊薄**是头号项——数学上与
   serial/triton 的 blockwise 逐块 promotion 等价（不是精度赌博），预期消掉
   +31% 指令的大头；TMA 流水加深/warp specialization 属深改，视第一步
   回收效果再定。
