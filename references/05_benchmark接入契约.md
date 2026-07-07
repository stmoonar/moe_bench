# Benchmark 接入契约

## 1. moe_bench 架构概述

moe_bench 是一个 MoE 层性能微基准测试框架，对比不同实现方案（串行 baseline vs 通算融合）在同一硬件、同一输入下的延迟和正确性。

### 1.1 两种测试模式

| 模式 | 接口 | 测量范围 | 用途 |
|---|---|---|---|
| compute-only | `MoEImplementation` | 单卡本地专家计算 | kernel 级对比 |
| distributed | `DistributedScheme` | 完整 MoE 层（含跨卡通信） | **本项目使用** |

本项目使用 **distributed 模式**，因为通算融合的核心价值在于隐藏跨卡通信延迟。

### 1.2 运行流程

1. 主进程解析 config（YAML 或命令行参数）
2. `mp.spawn` 启动 `world_size` 个 worker 进程，每个绑定一张 GPU
3. 每个 worker：初始化 NCCL 进程组 → 构建权重（`make_weights`）→ 遍历 token 数列表 → 每个点 `make_problem` → `scheme.setup` → 预热 + 计时 → （可选）正确性校验
4. 跨 rank 聚合：延迟取 max（最慢 rank 决定步骤延迟），verify 任一 rank 失败即标记 FAIL
5. rank 0 打印报告 + 可选写 JSON

### 1.3 计时方法

- 分布式模式**禁用 CUDA Graph**（跨 NCCL collective 的 graph capture 脆弱），计时为 eager 模式
- 预热 `warmup_iters`（默认 10）次，正式计时 `bench_iters`（默认 50）次
- 每次 `run()` 前后用 `torch.cuda.Event` 计时
- 报告 avg / min / med 三种延迟

---

## 2. DistributedScheme 接口

通算融合算子需实现 `DistributedScheme` 接口：

```python
class DistributedScheme(ABC):
    name: str

    def setup(self, problem: MoEProblem, ctx: DistContext) -> None:
        """一次性初始化（分配通信缓冲、准备权重/stream），不计入计时。"""

    @abstractmethod
    def run(self) -> torch.Tensor:
        """执行完整 MoE 层，返回本 rank 的 (num_tokens, hidden) 输出。
        在计时循环中被反复调用，必须无副作用、分配稳定（重用 setup 中的缓冲）。"""

    def close(self) -> None:
        """可选的清理（释放 stream/buffer）。"""
```

### 2.1 DistContext

```python
@dataclass
class DistContext:
    rank: int           # 本进程的 rank
    world_size: int     # 总 rank 数
    local_rank: int     # 本机 rank（= rank，单机场景）
    device: torch.device  # cuda 设备
    group: ProcessGroup  # torch.distributed 进程组（None = 默认 world group）
```

### 2.2 注册

在 `schemes.py` 的 `SCHEMES` 字典中注册：

```python
SCHEMES: dict[str, type[DistributedScheme]] = {
    SerialNaive.name: SerialNaive,
    "overlap": MyOverlapScheme,  # 新增
}
```

### 2.3 运行命令

```bash
python -m moe_bench.bench --distributed --mode ep --precision fp8 \
    --world-size 4 --scheme overlap
```

---

## 3. MoEProblem 数据结构

`setup` 接收的 `MoEProblem` 包含本 rank 的全部输入：

| 字段 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `config` | - | MoEBenchConfig | 配置对象 |
| `num_tokens` | 标量 | int | 本 rank 的 token 数（=512） |
| `hidden_states` | (512, 7168) | bf16 | 本 rank 的 token 分片（FP8 模式下输入仍为 bf16，需内部量化） |
| `w1` | (E_local, 2*inter, hidden) | fp8 | 上投影权重（已量化），E_local=64（EP world_size=4） |
| `w2` | (E_local, hidden, inter) | fp8 | 下投影权重（已量化） |
| `topk_ids` | (512, 8) | int32 | 每 token 的 topk 个全局专家 id |
| `topk_weights` | (512, 8) | float32 | combine 权重 |
| `expert_map` | (256,) | int32 | EP 模式：全局专家 id → 本地 slot（-1 表示不在本 rank） |
| `quant_config` | - | FusedMoEQuantConfig | FP8 block scales（w1_scale, w2_scale, block_shape） |

### 3.1 权重形状说明

EP 模式下（world_size=4, num_experts=256）：
- `E_local = num_experts / world_size = 64`（每卡 64 个专家）
- `intermediate_shard = intermediate_size = 2048`（EP 不切中间维）
- `w1` shape: `(64, 2*2048, 7168) = (64, 4096, 7168)`
- `w2` shape: `(64, 7168, 2048)`

### 3.2 权重确定性

权重生成种子来自 `(seed, global_expert_id, shard)`，不依赖 rank。这意味着：
- expert `e` 在任何 rank 上都有相同的权重
- baseline 和通算融合实现看到的是**逐字节相同的张量**
- 可以用单卡全专家参考来核对多卡输出

---

## 4. SerialNaive Baseline

串行 baseline 的数据流（`run()` 方法）：

```
1. AllGather(token 分片 + 路由) → 全量 batch (2048, 7168)
2. fused_experts(hidden_full, w1, w2, topk_weights, topk_ids, expert_map, quant_config)
   → 内部做 scatter + GroupGEMM0 + SiLU + GroupGEMM1 + gather + combine
3. ReduceScatter(结果) → 本 rank 的输出 (512, 7168)
```

通信和计算**完全串行**——AllGather 完成后才开始 GEMM，GEMM 完成后才开始 ReduceScatter。这是通算融合要超越的对象。

### 4.1 Baseline 的通信量

- AllGather: 每个 rank 发送 `512 * 7168 * 2 bytes` (bf16) + 路由元数据
- ReduceScatter: 每个 rank 接收 `512 * 7168 * 2 bytes`
- 总通信量与 token 数和 hidden_size 成正比

---

## 5. 正确性校验

### 5.1 校验流程

moe_bench 在计时前（或计时后）自动校验每个 token 数点的正确性：

1. 构建本 rank 的 golden problem：本 rank 的 token 分片过**全量专家**的完整 MoE（`make_golden_problem`）
2. 用纯 torch 参考实现（`reference_moe`）计算期望输出
3. 调用 `scheme.run()` 获取实际输出
4. `verify_output` 做 `torch.testing.assert_close` 对比

### 5.2 容差

| 精度 | atol | rtol |
|---|---|---|
| bf16 | 2e-2 | 0.0 |
| fp16 | 1e-2 | 0.0 |
| **fp8** | **3.5e-2** | **3.5e-2** |

### 5.3 参考实现细节

FP8 参考实现的关键：
- 对激活做与 kernel **相同的 block 量化**（`moe_kernel_quantize_input`）
- 逐 expert 循环做 `_block_matmul`（逐 tile 反量化 + fp32 累加）
- 携带与 kernel **相同的量化误差**，保证在紧容差内可比

### 5.4 为什么可以单卡校验多卡

分布式校验利用的性质：每 rank 的 combined 输出 = 其 token 分片过**全量专家**的完整 MoE 结果。因为：
- EP 的 ReduceScatter 返回本 rank 的 token 过所有专家的 combine 结果
- TP 的 ReduceScatter 返回本 rank 的 token 过完整专家（中间维已 all-reduce 求和）的结果
- 权重确定性保证各 rank 的专家权重一致

所以 `make_golden_problem` 只需本 rank 的 token + 全量专家权重，用单卡参考即可核对。

---

## 6. 通算融合方案的接入要点

### 6.1 需要实现的内容

通算融合 scheme 需要在 `setup` 中完成：

1. **控制面初始化**：对称缓冲分配（cudaIPC/NVSHMEM）、stream 创建（main + cp + rs 高优先级）
2. **权重预处理**：从 `problem.w1/w2` 和 `quant_config` 提取本 rank 专家的权重和 scales，按 GEMM 需要的布局准备
3. **路由元数据预计算**：从 `problem.topk_ids/topk_weights` 生成 `splits`、`scatter_index`、`gather_index`、`output_vec_scale`
4. **缓冲预分配**：`input_buffer`、`barrier`、`gemm_out`、`reduce_buffer`、`tile_flag` 等，大小按 `max(num_tokens)` 预分配

`run()` 中执行：

1. 本 rank token 分片复制到对称缓冲 + barrier 清零
2. 启动 AG（cp_stream）+ 元数据 kernel（main_stream）
3. 发射 layer0 融合 GEMM（main_stream，kernel 内等灯）
4. SiLU-mul 激活（可用 torch 或融合进 GEMM epilogue）
5. 发射 layer1 GEMM（main_stream，sm_margin）+ Reducer（rs_stream，高优先级）
6. 等待完成，返回本 rank 输出

### 6.2 与 moe_bench 契约的对齐

- **输入**：`problem.hidden_states` 是 bf16（FP8 模式下也如此），scheme 需内部做 block 量化
- **输出**：`run()` 返回 `(num_tokens, hidden_size)` 的 bf16 张量
- **无副作用**：`run()` 不能修改 `problem` 中的张量（计时循环会反复调用）
- **分配稳定**：所有缓冲在 `setup` 中分配，`run()` 中重用，不新建
- **EP 模式**：`problem.expert_map` 非空，scheme 需据此过滤本 rank 的专家

### 6.3 测试配置

```bash
# 4 卡 EP FP8
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.bench \
    --distributed --mode ep --precision fp8 \
    --world-size 4 --num-tokens 512 --scheme overlap

# 先跑 baseline
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.bench \
    --distributed --mode ep --precision fp8 \
    --world-size 4 --num-tokens 512 --scheme serial
```

---

## 7. 关键文件索引

| 文件 | 作用 |
|---|---|
| [schemes.py](../schemes.py) | `DistributedScheme` 接口定义 + `SerialNaive` baseline + scheme 注册表 |
| [data.py](../data.py) | `MoEProblem` 数据结构 + 权重/输入/路由生成 + golden problem |
| [config.py](../config.py) | `MoEBenchConfig` 配置 schema + 所有参数定义 |
| [distributed.py](../distributed.py) | 分布式 benchmark 运行器：spawn worker、计时、校验、聚合报告 |
| [reference.py](../reference.py) | 纯 torch 参考实现 + `verify_output` 校验函数 |
| [context.py](../context.py) | `DistContext` 数据类 |
| [report.py](../report.py) | 结果表格打印 + JSON 输出 |
| [ADDING_IMPLEMENTATIONS.md](../ADDING_IMPLEMENTATIONS.md) | 添加新 scheme 的完整说明 |
| [AGENTS.md](../AGENTS.md) | 运行环境约束（卡选择、空闲检查等） |
