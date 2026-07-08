# 09 调参与 benchmark 方法论

## 1. 必测 baseline（缺一不可）

```text
T_compute_alone / T_comm_alone         # 上限公式的输入
T_non_overlap（串行）                   # 收益的分母
T_naive_stream_overlap                 # 两条流裸并发：暴露干扰量
T_目标方案
GEMM TFLOPS/MFU、有效通信带宽、SM 驻留、L2 命中/DRAM 带宽
```

只看端到端总时长是最常见的假阳性来源：**必须拆出 compute slowdown、comm slowdown、overlap efficiency 三个数**。overlap 效率高但计算被拖慢 20% 的方案可能比效率低但零干扰的方案更差。

## 2. 干扰的定位方法（timeline 检查点）

- 两条 stream 是否真并发？（隐式依赖、event、通信库 group 语义都可能串行化）
- 计算 kernel 是否被通信 kernel 拉长（对比 alone 时长）；反之亦然。
- GEMM 尾部是否仍裸露一段通信（分段/末段太大）。
- wait/counting kernel 的 SM 占用是否超预期。
- L2/DRAM 带宽是否达到瓶颈；通信数据是否污染缓存（读写路径是否旁路）。
- 远端写/异步 put 是否有 burst 或反压。

## 3. 性能建模的三个必备修正项

1. **让出 SM 的 wave 放大**：计算时长按 `ceil(T/(SM-k))/ceil(T/SM)` 修正（k = 通信+辅助占用的 SM）。
2. **带宽-消息量曲线**：实测一条 30~40 个点的带宽曲线（消息量按指数间隔），分段通信的每段时长用插值算，不要用峰值带宽。
3. **流水递推**：`acc_comm = max(acc_comp_i, acc_comm) + T_comm(段 i)`，末段通信全暴露要单独加。
   这样的解析模型可把预测误差做到 ~3.4%，替代分钟级在线穷举。

## 4. autotune 体系的结构

- **两级结构**：离线 profiling 生成"(算子元信息, shape) → 最优配置"的查找表 + 查不到时的启发式规则兜底。表按精确 shape 打点（业务 shape 有限），启发式覆盖长尾。
- **搜索空间分层**：计算 kernel 参数（tile/stage/cluster/调度模式/raster）与重叠参数（chunk、信号粒度、分段、swizzle、push-pull、SM 配额）分开 sweep——先固定纯计算最优，再扫重叠参数；只有出现明显互作用（如大 tile 降 stage 保 smem）才联合调。
- **配置的物化**：调好的配置落成静态注册表（编译产物或配置文件），运行时零搜索开销；同时保留运行时覆盖（如 SM 数手动覆盖）。
- **variant 管理**：动态负载（MoE 的 topk/BN/dtype/layout）用 AOT variant 笛卡尔积时要克制——组合爆炸使编译产物大、新硬件适配慢；优先把可运行时化的参数（SM 配额、分段数）从模板常量里拿出来。

## 5. 分布式 autotune 的特殊点（单卡经验之外）

1. **每次 profiling 前重置信号/缓冲**：上一轮残留计数会让下一轮测量死锁或测错。
2. **所有 rank 必须选同一配置**：各 rank 独立调优会选出不同 config → 分段不一致 → 集合通信死锁。做法：调优后全局同步、广播统一选择；配置缓存带硬件/版本指纹。
3. **确定性验证纳入调优流程**：依赖完成序的方案，把"多次运行完成序一致"作为配置的准入条件（不一致的配置直接淘汰，而不是带病上线）。
4. 多机时配置文件放共享存储，避免 rank 间版本漂移。

## 6. 推荐 sweep 空间（起点值）

```yaml
shape:
  M: [decode 级(64~512), prefill 级(1k~8k), 训练级(8k~64k)]
  并行: TP [2,4,8] × EP [8~256] × topk [1~8]
资源:
  comm_sm/CTA: [0(CE/NIC), 1, 2, 4, 8, 16, 20, 28]
  channel: [1, 2, 4, 8, 10, 16]（2 SM/channel 起步）
粒度:
  comm_chunk: [rank_slice, 1/2, 1/4, ..., ≈gemm_tile]
  信号粒度: [rank片, N-split(7~9), wave_group(1,2,4), chunk]
  切分数: [1(不切), 2(wave对齐), 4, 8]
调度:
  swizzle: [none, rank_shift, topology_ring, arrival_order]
  模式: [push, pull, hybrid] × [fused, 分kernel, CE]
```

## 7. 正确性验证（与性能同权重）

- 逐 tile/逐段 checksum 对拍（与串行参考实现比），覆盖：边界 tile（跨 chunk/跨 rank）、pad 区、复用 buffer 的第二次迭代（脏数据）、grid < SM 数的 shape。
- 原子加归约的方案要明确非确定性求和的容差策略。
- 压力项：反复调用（信号复位路径）、动态 shape 交替（缓冲复用）、故意的慢 rank（超时/屏蔽机制）。
- 自旋等待全部带超时报警；死锁调试用"超时即陷阱"的等待原语。
