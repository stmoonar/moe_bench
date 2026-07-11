# PLAN — SM120 + PCIe 上的 MoE tile 粒度通信计算融合

> 配套阅读：`experience/11_ThunderKittens多GPU与MoE融合实战.md`（机制）与 `experience/12_SM120与PCIe拓扑适配.md`（本机器的约束与替代方案）。本目录是这两篇的代码落地。
>
> ⚠️ **诚实声明**：以下代码在无 CUDA 环境的 Windows 开发机上编写，**从未编译过**。第一次 `make` 大概率有编译错误要修（预期修正点见 §8），逻辑经过对照仓库 H100 kernel 的逐行桌面检查，但正确性以机器上的对拍为准。

## 0. 交付物与代码结构

```text
tileoverlap/
├── PLAN.md                      ← 本文件
├── common/
│   └── sm120_common.cuh         ← ① pcie_sync: PCIe 合法的跨卡同步（slot+序号，无远端原子）
│                                   ② grouped_gemm_sm120: SM120 GEMM 模板（warp 级 mma.sync
│                                      + TMA 3 级流水 + 可注入的 Gate = 通信融合点）
├── 00_probe/                    ← Phase 0: 平台探测（纯 CUDA，单进程多卡）
├── 01_grouped_gemm/             ← Phase 1: 单卡 grouped GEMM（计算基线 + 正确性锚点）
└── 02_moe_dispatch_gemm/        ← Phase 2: dispatch ⊕ grouped GEMM 融合（EP layer0）
                                    含 pcie_device_barrier(barrier, seq) 跨卡对齐原语
```

核心设计一句话：**GEMM kernel 不拆开**——通信以三种最小侵入的形式存在：(a) grid 尾部的 dispatch block 从 peer 卡拉 token 到本地并做本地原子计数；(b) GEMM producer warp 在每个 128-token 行块前自旋等本地计数器到 128（`dispatch_gate`，全部融合就这一个函数）；(c) 层边界一个 slot+序号式的跨卡 barrier。全程零远端原子、零 multimem，符合 PCIe 红线。

## 1. 环境要求

- Linux（TK 的 IPC 用 Unix socket 传 fd，Windows/WSL1 不行；WSL2 的 P2P 大概率残废，用裸机）。
- CUDA Toolkit **12.8+**（sm_120a 需要）、与之匹配的驱动、PyTorch 2.x-cu12x、pybind11（`pip install pybind11`）。
- BIOS/内核：开启 Resizable BAR / Above 4G Decoding；**关闭或正确配置 IOMMU/ACS**（`dmesg | grep -i iommu`；虚拟化开着的话 P2P 很可能被 ACS 拦掉）。
- `export ARCH=SM120`。

## 2. Phase 0 — 平台探测（半天，一切的前提）

```bash
# 先看拓扑
nvidia-smi topo -m
cd tileoverlap/00_probe && make run
```

probe 会输出 [A]~[D] 四组结果。**把下面这张表填完再动下一步**：

| 基准 | 来源 | 期望值 | 实测 | go/no-go 判据 |
|---|---|---|---|---|
| P2P access（每对） | probe [A] | 1 | | **=0 → 全盘停止**，转 NCCL+signal 低侵入路线（08 篇级 1），本目录代码作废 |
| nativeAtomicSupported | probe [A] | 0（PCIe 预期） | | =1 的话反而要重查拓扑；=0 印证代码里禁用远端原子的设计 |
| 远端原子实测 | probe [B] | WRONG/UNSUPPORTED | | 若三种原子全 OK，可考虑简化信号协议（但别指望） |
| CE 峰值带宽 | probe [C1] | ~50 GB/s (Gen5 x16) | | <25 GB/s → 查 ACS/BAR/链路降速 (`nvidia-smi -q \| grep -i width`) |
| SM pull 带宽 & 饱和 block 数 | probe [C2] | 接近 CE 值；2~8 block 饱和 | | pull 带宽 << CE → dispatch 改走 CE+pack 路线（§7 风险 R2） |
| 信号 RTT | probe [D] | 1~3 μs | | TIMEOUT → release-store 可见性坏了，全盘停止排查平台 |

另外记录三个**外部 baseline**（后面算 overlap 效率要用）：

```bash
# 1. NCCL all-to-all / all-gather 带宽（作为 T_comm_alone 的下界参照）
git clone https://github.com/NVIDIA/nccl-tests && make -C nccl-tests
./nccl-tests/build/alltoall_perf -b 64M -e 1G -f 2 -g <NUM_GPUS>
# 2. torch bf16 稠密 GEMM 峰值（本卡计算屋顶）
python -c "import torch; ..."  # 8192^3 matmul 计时，记 TFLOPS
# 3. torch 版 MoE 参考实现耗时（02 的 benchmark 里 torch_reference 就是）
```

## 3. Phase 1 — 单卡 grouped GEMM（1~2 天，含编译调错）

```bash
cd tileoverlap/01_grouped_gemm
make          # 首次编译，预期要按 §8 修错
make run      # 正确性对拍 + TFLOPS
```

验收标准：
1. 三档 token 规模 max diff < 0.1（bf16 累加误差量级）；
2. TK / torch ≥ **70%**（v1 目标；warp mma + 3 级流水的合理起点。低于 50% 先看 `-Xptxas` 有无 spill，再查 `ncu` 的 smem 吞吐与 mma 占用）；
3. 记录 `T_comp_alone(seq)` 曲线——这是 Phase 2 算干扰系数的分母。

## 4. Phase 2 — 融合 kernel（1~2 周）

```bash
cd tileoverlap/02_moe_dispatch_gemm
make run NUM_GPUS=2        # 先 2 卡把正确性打通
make run NUM_GPUS=4        # 再扩（NUM_GPUS 同时决定编译期 pgl 尺寸和 torchrun 进程数）
```

顺序：
1. **正确性**（benchmark 第一轮自动跑）：`outputs` 和 `inputs_gathered` 双重对拍。若 gathered 错而 outputs 之前的单卡对拍是对的 → 问题在 dispatch/TMA-peer/信号，不在 GEMM。
2. **sweep `num_comm_sms`**（benchmark 自动 1/2/4/8/16）：PCIe 上预期 1~4 就够（probe [C2] 的饱和 block 数 × 一点余量）。注意这里 comm SM 只影响 dispatch 排队并发度，不是 H100 版那种大预算。
3. **09 篇三件套**：
   - `T_comp_alone`：Phase 1 数据；
   - `T_comm_alone`：把 `num_comp_sms` 的 GEMM 部分短路掉单测 dispatch（临时改法：weights/I 缩到最小），或用 NCCL all-gather 折算；
   - `T_total`：benchmark 输出；
   - overlap 效率 = (T_comp + T_comm − T_total) / min(T_comp, T_comm)，目标 ≥ 0.8；PCIe 上大概率 comm-bound，**T_total 逼近 T_comm 即算成功**。
4. `nsys` 看时间线：dispatch block 是否真的和 GEMM 交错、producer 自旋是否只出现在最前排 row block。

## 5. Phase 3 — layer1（GEMM(W2) ⊕ combine），待写

复用 `grouped_gemm_sm120`，新增两件东西（12 篇 §6 路线 B'）：

1. `03_moe_gemm_combine/`：GEMM consumer 存完一个列块后，由**每 SM 的信号 warp** 对本地 barrier 做 `red.release.gpu` 计数（列块粒度）；grid 尾部 combine block 等"该列块全部 tile 就绪"后按 combine 索引从本卡 outputs gather 行、乘 gate 权重、**pull 语义写给源卡**——注意方向反转：是**源卡的 combine block 来拉**（等 expert 卡的就绪信号，用 `pcie_sync::signal_slot` 每 expert 卡一个 slot 发到源卡），FP32 累加 top-k 份后落本地。全 pull + slot 信号，与 layer0 同构。
2. combine 索引 = layer0 pull 索引的逆映射，在生成 schedule 时顺手产出。
3. 验收锚点先行：非融合版（GEMM 完 → 独立 combine kernel）先对拍，再融合。

## 6. Phase 4 — 接入完整 MoE 层

目标形态（python 侧，推理优先）：

```python
class TKMoELayer:
    def __init__(...):   # 一次性：TKParallelTensor 常驻 buffer（pre_tokens/barrier），
                         # 容量按 token 上限分配；权重普通 tensor
    def forward(x):      # 每层：
        # 1. gating: torch topk → chosen_experts (GPU)
        # 2. schedule: 生成 pull_dispatch_indices / padded_tokens_per_expert / combine 索引
        # 3. pre_tokens.data_.copy_(x)（或让上一层直接写进 pre_tokens）
        # 4. device_barrier(seq++)   ← 保证所有卡的 pre_tokens 就绪且上一层的拉取已结束
        # 5. moe_dispatch_gemm(...)  → h = act(x@W1)  [v1: 激活在 torch 里做，后续融进 epilogue]
        # 6. (Phase 3) moe_gemm_combine(...) → y 回到原 token 顺序
```

关键工程点：
- **schedule 生成必须搬上 GPU**。benchmark 里的 Python 三重循环只配跑 benchmark；生产路径= torch 向量化（`argsort(expert_id)` + `cumsum` 就能拼出 pull 索引，~10 行）先顶上，之后再考虑 CUDA metadata kernel 消掉 D2H 同步（`num_padded_local_tokens` 回传是当前唯一一个 CPU 同步点；消法：buffer 按上限、kernel 读 device 端计数）。
- **buffer 生命周期**：`TKParallelTensor` 创建有 IPC 握手成本，**必须**层间复用（按最大 token 数一次分配）；`barrier` 一个就够（row 0 每层 epilogue 自动清零，row 1 的 seq 单调不清零，Python 侧全局计数器保证跨层单调）。
- **步骤 4 的 barrier 什么时候可以省**：若相邻两层之间本来就有 NCCL 集合通信（如 attention 的 TP allreduce），它顺带完成了对齐，`device_barrier` 可跳过——用 nsys 确认后再省。
- **gated FFN（W1/W3 双矩阵）**：v1 跑两次 `moe_dispatch_gemm`（第二次 dispatch 会命中已 gather 的 token？不会——直接对 `inputs_gathered` 调 01 的 `grouped_gemm` 即可，token 已在本地）；之后再做双 B 流水的融合版。
- **训练**：本方案 v1 只管 forward。backward 的 dispatch/combine 是转置的通信模式（combine 的反向=dispatch），协议可复用，但 autograd 接入和确定性（FP32 累加顺序）要单独设计——放最后。
- **FP8 dispatch**（comm-bound 时的头号杠杆）：token 传输改 fp8e4m3（`sv_fp8e4m3<H>` + scale），GEMM 侧 SM120 有原生 fp8 mma 路径。等 bf16 版全链路打通后做，预期 dispatch 时间近似减半。

## 7. 风险与回退

| # | 风险 | 信号 | 回退 |
|---|---|---|---|
| R1 | P2P 不可用 | probe [A]=0 | 全盘转 NCCL/copy-engine + signal 低侵入路线（08 篇级 1），本目录仅 01 可用 |
| R2 | TMA 对 peer 内存不工作或奇慢 | 02 gathered 对拍错 / probe [C2] pull 带宽塌 | dispatch 改 warp 协作 `ld.global.v4` 拉取（改动只在 `dispatch()` 一个函数）；或源端 pack + CE 大块搬运 |
| R3 | 逐 token 拉取延迟受限打不满带宽 | nsys 里 dispatch 拖尾 >> 字节数/带宽 | 加大 TOKENS_PER_BLOCK 的并发（分块 token 向量、双缓冲），或做源端按 (目标卡,expert) pack |
| R4 | GEMM 效率不达标 | Phase 1 < 50% torch | 查 spill → 缩 COL_BLOCK 到 64×2；查 smem 吞吐 → B 子块复用改载入顺序 |
| R5 | 自旋死锁 | kernel 挂死 | 检查 num_comm_sms≥1、comp blocks < SM 数；用 `careful_wait` 式超时 trap 定位哪个 row block 没等到 |

## 8. 已知未验证点（首次编译预期修正清单）

按可能性排序：
1. `warp::load(rt_bf<16,COL,col>, st_subtile)` 的 col-layout 装载路径是否对 st_subtile 全支持（不行就把 B 子块先 `subtile` 成 row 再用 `mma_ABt` 换公式）。
2. `consumers::store(gl, rt_fl, coord)` 的 group 全局存储 + float→bf16 转换组合（不行就每 warp 先 `store` 到 smem C tile 再 TMA 出去，仿 H100 版）。
3. `st.subtile<16,16>` 在 128B swizzle 父 tile 上的地址断言。
4. 寄存器压力：`-Xptxas --warn-on-spills` 有 spill 就把 `COL_BLOCK` 降 64 或 acc 拆两半。
5. `pgl` 在 `MULTICAST=false` 下的 host 构造路径（`make_pgl` 无 mc 变体）与 `barrier_pgl` 的 gl 维度推断。
6. 杂项：`kittens::warpid()` 命名、`coord` 花括号初始化维度语义、`CUDACHECK` 宏可见性。

修一轮编译 + 一轮 compute-sanitizer（`make ncu` 前先 `compute-sanitizer ./...`）是计划内工作，不是意外。
