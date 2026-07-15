# 09 优化#1c 规划：push 语义 dispatch ⊕ GEMM 融合（单写者 slot 信号，无远端原子）

**日期**：2026-07-09
**状态**：规划（未实现）
**前置阅读**：docs/08（push/push2 冻结归因）、experience/12 §3（slot+序号协议）、docs/07 §3 #1
**外部参考**：Triton-distributed（`C:\Users\stmoonar\Desktop\files\Triton-distributed`，本文 §2 为其代码考古结论）

## 0. 三个问题的直接回答

**Q1：docs/08 遇到的"PCIe 远端原子丢增量"正常吗？**
正常，是平台规格而非 bug。PCIe 平台 `cudaDevP2PAttrNativeAtomicSupported == 0`，对 peer
显存做原子 RMW（`red`/`atom`）本就不受支持、结果未定义——高并发下丢增量只是"未定义"的
具体表现形式。低并发 probe 碰巧不丢，才造成了 docs/07 的误判。旁证：Triton-distributed
在 `python/triton_dist/utils.py` 里显式探测这个属性并打 warning（"this may cause undefined
behavior"），其 `common_ops.py::barrier_all_intra_node_atomic_cas_block` 的 docstring 直言
"memory over PCI-e does not support atomic r/w. DON'T use this function on such platforms"。
**PCIe 上跨卡可靠的只有普通 load/store 的可见性（配 release/acquire），原子 RMW 不可靠**——
这条要刻进 experience/12 红线 2，不再需要 probe 复核。

**Q2：Triton-distributed 怎么解决 push 的同步？**
两阶段协议 + 单写者 slot + SET 语义信号，**intra-node 路径零远端原子**。见 §2。

**Q3：TK 上怎么做？**
我们仓库已有全部所需原语（`pcie_sync::signal_slot/wait_slot`、`comb::combine_signal_epilogue`
的"本地原子选举唯一信号者"模式、`dpush::push` 的 TMA 数据面、Gate functor 挂钩），只差把它们
按 §3 的协议拼进一个融合 kernel。方案代号 **push3**（`TK_DISPATCH=push3`）。

## 1. 问题定义（docs/08 的遗留难题）

docs/08 的教训：push2（推→barrier→GEMM 三段串行）正确但比 pull 慢 30%，因为丢了 pull 版
"dispatch 融进 gate GEMM producer 自旋"的通算 overlap。真正的提速形态必须同时满足：

1. **数据面走强路径**：源端 TMA push（~51 GB/s），不是 SM pull（~20 GB/s）；
2. **保持融合**：push 与 gate GEMM 在同一个 kernel 里，GEMM producer 边等边算，无中间 barrier；
3. **完成检测无远端原子**：gate 要精确知道"某 128-token 行块到齐"，但不能用 `red.release.sys` 计数。

难点全部集中在第 3 条：多张源卡的 token 交错落进同一个行块，"到齐"天然是个多写者计数问题。
push2 用全局 barrier 回避了它（于是丢了融合）；push3 要正面解掉它。

## 2. Triton-distributed 的解法（代码考古结论）

关键文件：`python/triton_dist/kernels/nvidia/ep_all2all_fused.py`（dispatch⊕grouped GEMM 融合
mega-kernel）、`ep_a2a_intra_node.py`（独立 kernel 版）、`common_ops.py`（barrier 原语）。

**协议 = 两阶段，把"计数"变成"定值"：**

**阶段 1（metadata）**：每 rank 把自己的 splits（发给各 expert 的 token 数）用
`putmem_signal + NVSHMEM_SIGNAL_SET` 推给所有对端；各 rank 本地 cumsum 出
`recv_buf_offset_per_expert`——**每个 token 在目的卡 buffer 里的绝对 slot 从此完全确定，
无任何竞争**。（我们的 `_build_schedules` 在 host 上做的就是这一步，且做得更细：直接产出
per-assignment 的 `(dst_dev, dst_slot)`。）

**阶段 2（数据 + 信号）**，`tile_kernel_dispatch_token_intra_node` 的核心三行：

```python
libshmem_device.putmem_warp(dst_ptr, src_ptr, bytes, expert_rank)          # push 数据（纯拷贝）
sent = atomic_add(counter_ptr + expert_idx, 1, scope="gpu", semantic="relaxed")  # 本地计数（gpu scope，合法）
if sent == tokens_this_expert - 1:                                          # 我是本卡最后一个完成者
    libshmem_device.fence()
    libshmem_device.signal_op(barriers + expert*world + rank, 1, SIGNAL_SET, expert_rank)  # 远端独占 slot，SET 非 ADD
```

消费端（同一 persistent kernel 里的 grouped GEMM tile）：

```python
while ld_acquire(barriers_ptr + expert_id * world_size + t, scope="gpu") != 1: pass  # 本地自旋，等齐 world_size 个 flag
```

**设计要点提炼**（谁写/写哪/写什么/谁等/等什么）：

| 要素 | Triton-distributed 的选择 |
|---|---|
| 计数 | 只在**源卡本地**做（gpu-scope atomic_add），跨卡零 RMW |
| 信号写者 | 本地计数选举出的"最后完成者"，每个 (src_rank, dst_expert) slot **恰好一个写者** |
| 信号原语 | `SIGNAL_SET`（普通 store）/ `st.release.sys`，写定值不累加 |
| 信号粒度 | per-(src_rank, dst_expert) chunk；two_stage 变体是 per-token 独占 slot 写 store_idx（哨兵 -1） |
| 等待方 | 消费端本地 acquire 自旋，等**已知个数**（world_size 个）的 flag 逐一到位 |
| 到齐判定 | 不靠"数到了多少"——期望值在阶段 1 已定死，只等 flag |

**附带发现（避坑）**：`ep_all2all_fused.py` 的 GEMM→combine 阶段间 rendezvous 硬编码调用了
`barrier_all_intra_node_atomic_cas_block`（依赖 P2P 原子 CAS，1271/1407/1572/1760 行），没走
它自己库里的 `supports_p2p_native_atomic()` 降级分支——说明**这套融合 kernel 在我们的 PCIe
平台上也不能直接拿来用**，它们只是 dispatch 信号协议本身是 PCIe-safe 的。我们自研仍是必要的。

**与我们的对照**：这个协议就是 experience/12 §3 的"slot+序号"协议 + docs/08 处置 §3 第一条
路线，而"本地原子选举最后完成者发信号"的模式我们已经在 layer1 的
`comb::combine_signal_epilogue`（tk_moe.cu:476-492）里实现并对拍通过——`atom.acq_rel.gpu.add`
本地计数，满额者 `pcie_sync::signal_slot` 广播。push3 只是把同一模式从"GEMM 输出侧"搬到
"dispatch 输入侧"。

## 3. push3 协议设计

### 3.1 一句话

源卡 TMA push token 到目的卡定死的 slot（复用 `push_indices` schedule）→ 每完成一个 push 在
**源卡本地** per-(dst, 行块) 计数器上 `atom.acq_rel.gpu.add` → 加到 host 预计算的期望值的那个
线程被选举为唯一信号者，`fence` 后向目的卡 `barrier[dst][SIG+my_rank][行块]` 写 seq
（`st.release.sys`，单写者）→ 目的卡 GEMM producer 的 gate 对每个行块本地自旋等
"所有有贡献的源卡 slot 都 ≥ seq"。

### 3.2 信号布局与期望值（闭式，exp/11 纪律）

```text
barrier_l0 扩为 (2 + NUM_DEVICES, bar_cols)：
  row 0            ：pull 模式的本地计数器（保留，pull 是回退路径）
  row 1            ：pcie_barrier_all 到达 slot（保留）
  row 2 + s        ：源卡 s 的 push3 完成信号，列 = 目的卡本地行块号 rb
                     值 = 单调 seq（immune to reset，同 pcie_sync 约定）

每个 slot (dst, 2+s, rb) 的写者：恰好是源卡 s 上"该 (dst,rb) 计数器最后加满的线程"——1 个。✓
gate 等待条件（目的卡 d，行块 rb）：
  ∀ s ∈ [0,N)：gate_expected[rb][s] > 0 ⇒ wait_slot(barrier, d, 2+s, rb, seq)
  gate_expected[rb][s] == 0 的源卡不等（它根本不会发信号）。
  padding 不参与：期望值只数真实 token，纯 padding 行块四个 expected 全 0，gate 直接放行。
```

期望值表全部由 host 从全局路由算出（`_build_schedules` 已 all-gather 了 `all_topk`，docs/08
第一步 instrumentation 的直方图对账证明这张表能算对）：

- `push_cnt_idx (num_push,) int32`：本卡第 i 个 outgoing assignment 对应的**本地计数器**
  扁平下标 = `dst_blk_offset[dst_dev] + dst_slot // ROW_BLOCK`（dst_blk_offset 为各目的卡
  行块数的前缀和）。
- `push_expected (total_dst_blocks,) int32`：每个本地计数器的满额值 = 本卡发往该 (dst,rb)
  的真实 token 数（>0 的才会有 assignment 指向它）。
- `gate_expected (nblk_local, NUM_DEVICES) int32`：本卡（作为目的卡）每个行块从各源卡应收
  的 token 数，gate 只用"是否 >0"。
- `local_cnt (total_dst_blocks,) int32` 普通本地 tensor，每 iteration 清零（本卡写本卡清，
  同 stream 序，无跨卡竞争——对比 push 版远端计数器"不能 reset 只能 re-seed"的别扭，这里
  天然干净）。

### 3.3 Kernel 侧（新 namespace `dpush3`，改动集中在两个 functor）

**push 块**（改造 `dpush::push`，tk_moe.cu:256）：

```cuda
// 每线程一个 outgoing assignment（与现 push 相同的 TMA 数据面）
tma::load_async(token, pre_tokens, {src_tok,0}, sem); wait(sem);
tma::store_async(G.gathered[dst_dev], token, {dst_slot, 0});
tma::store_async_wait();                                   // 本线程的远端写已完成
const int c = G.push_cnt_idx[{i,0}];
int old; asm("atom.acq_rel.gpu.global.add.s32 %0,[%1],1;" ...);  // 本地计数（合法）
if (old + 1 == G.push_expected[{c}]) {                     // 我是最后完成者（唯一）
    __threadfence_system();                                // 见 §3.4 内存序论证
    pcie_sync::signal_slot(G.barrier, dst_dev, 2 + G.dev_idx, c - dst_blk_offset, G.seq);
}
```

**gate**（替换 `dpush::push_gate`）：

```cuda
struct push3_gate {
    __device__ void operator()(int rb) const {
        for (int s = 0; s < NUM_DEVICES; s++)
            if (G.gate_expected[{rb, s}] > 0)
                pcie_sync::wait_slot(G.barrier, G.dev_idx, 2 + s, rb, G.seq);   // 本地自旋
    }
};
```

`grouped_gemm_sm120` 模板、consumer、producer 流水**零改动**——这正是 TK 的设计哲学：
通信作为 Gate/Epilogue functor 贴在标准 GEMM 模板上，计算主体不知道多 GPU 的存在。

**grid 布局**：与现 push 相同，`num_comp_sms + push_blocks`，comp 在前（自旋块 < 110 红线
不变）。push 块短命，TOKENS_PER_BLOCK=6（smem 限制）→ 4096 assignments ≈ 683 块排队消化。

**入口**：`moe_dispatch_push3(pre_tokens, gathered, w_gate, gate_out, padded, push_idx,
push_src, push_cnt_idx, push_expected, gate_expected, local_cnt, barrier, num_comm_sms,
num_padded_local, num_push, seq)`。scheme 里 `TK_DISPATCH=push3` 分支：清零 local_cnt →
（无需 seed、无需 pre-barrier，seq 单调）→ launch。

### 3.4 内存序论证（R1，本方案唯一的非平凡风险）

链条：线程 X 的 TMA 数据写（async proxy）→ X 的 `store_async_wait` → X 的
`atom.acq_rel.gpu`（generic proxy，release 侧）→ 线程 Y 的 `atom.acq_rel.gpu`（acquire 侧，
读到满额）→ Y 的 `__threadfence_system` → Y 的 `st.release.sys` 信号 → 目的卡
`ld.acquire.sys` 见信号 → 目的卡 TMA 读数据。

- 卡内（X→Y）：acq_rel 原子链 + threadfence_system，标准；与 `combine_signal_epilogue`
  完全同构（那里是 consumer 普通 store → fence.sys → atom.acq_rel → signal_slot，已对拍）。
- 跨卡（Y 的信号 vs X 的数据谁先到）：X `store_async_wait` 返回 ⇒ X 的 bulk 写已从源端
  完成提交；Y 的信号在其后才发出，同一 (src GPU→dst GPU) 路径上 PCIe posted write 不互相
  超越。push2 的"push_data 完成 → 另一 kernel 的 st.release.sys barrier → 对端读数"依赖
  同样的跨线程/跨 proxy 顺序且 256 专家 stress 通过——但 push2 有 kernel 边界兜底，push3
  没有，**这是新增风险点**，Triton-distributed 在同位置放了 `libshmem_device.fence()`，
  我们对应放 `__threadfence_system()`。
- **验证手段（必做，不许跳过）**：复用 push-only debug 模式——push3 的 push 块单独跑 +
  目的卡按 gate_expected 对账信号值 + 数据 checksum 对比（docs/08 的两步 instrumentation
  现成）。若出现"信号到了数据没到"的偶发错：升级为"signaler 重读一个自己 push 的远端字节"
  或退回每 (dst,rb) 集中到单 warp 顺序发。

### 3.5 与 Triton-distributed 方案的差异（为什么不照抄粒度）

它们的信号粒度是 per-(src, dst_expert)，我们选 per-(src, dst 行块)：512 token/rank 档
per-expert 均值 64、padding 到 128 后**绝大多数 expert 恰好 1 个行块**，两种粒度几乎重合；
但行块粒度与现有 gate/`padded_tokens_per_expert` 体系零转换成本，且大 token 档（expert 多行块）
下放行更早。等待成本 O(N)=4 个本地 load/行块，可忽略。

## 4. 收益账与止损（docs/08 的教训：别重复计入 overlap）

| 项 | pull（现默认） | push2（已冻结） | push3（预期） |
|---|---|---|---|
| 数据面 | SM pull ~20GB/s | TMA push ~51GB/s | TMA push ~51GB/s |
| 融合 overlap | ✅ | ❌（三段串行） | ✅ |
| dispatch-only 实测/预估 | **2.00 ms** | 2.65 ms | 理论数据面 57MB@51 ≈ 1.1ms + gate 尾部 |
| 端到端层 | 7133 µs | 7823 µs | 目标 ≤ 6.6ms |

诚实预期：512 token 档收益**不保证显著**——pull 的 2.00ms 里已有 overlap 掩盖，push3 的
上界收益 ≈ 0.5~0.9ms（dispatch-only 口径）。真正拉开差距要靠后续叠加：
- **(token,dst) 去重**（docs/07 P3，push 天然适合：源端每 (token,dst) 只推一次到 staging，
  目的卡本地 scatter 到多 slot，流量 4096→~1800 份，数据面再省 ~2.2×）——阶段 2；
- **FP8 传输**（docs/07 #2，字节减半）——与 push3 正交，两者可叠加。

**成功判据**：64 专家对拍 reference_moe 通过 + 256 专家 30-iter stress 无挂死/无错 +
dispatch-only < 2.00ms。**止损**：若 §3.4 验证出"信号先于数据"且两种补救均无效，或
dispatch-only 仍 ≥ pull，则 push3 冻结归档（同 docs/08 纪律），主线继续 pull + 转做
去重/FP8/schedule 上 GPU。

## 5. 实施步骤（每步有锚点，可交给子代理按步执行）

1. **host schedule 扩展**：`_build_schedules` 增产 `push_cnt_idx / push_expected /
   gate_expected / dst_blk_offset`；纯 torch 复用现有 replay 循环。锚点：与 docs/08 第一步
   的直方图对账脚本互验（expected 表行和 == 每行块真实 token 数）。
2. **barrier 扩容**：`barrier_l0` 改 (2+world, bar_cols)；pull/barrier 行号不变，无回归。
3. **kernel `dpush3`**：push 块（§3.3）+ push3_gate + entry + pybind。先写 **push3-only
   debug 入口**（只跑 push 块，无 gate，不会挂）。
4. **协议验证（先于融合跑通）**：push3-only → 目的卡信号对账 + gathered checksum 对比
   pull 版 gathered；重复 30 次抓偶发。这是 §3.4 R1 的裁决点。
5. **融合对拍**：64 专家 reference_moe；256 专家 stress。
6. **计时**：dispatch-only 隔离 + 端到端，4 卡组 9,11,13,15（跑前查卡空闲），与 pull 同口径
   对比；结果与结论写 docs/10，无论成败。
7. **视结果决定**：达标 → 设为 `TK_DISPATCH` 可选项并评估转默认；叠加去重（阶段 2）。
   不达标 → 冻结归档。
