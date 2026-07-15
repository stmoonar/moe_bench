# 13 T6-v0:combine 预归约 host schedule 表设计与对账

**日期**:2026-07-09
**任务**:docs/11 §T6 / §3 —— combine 预归约 + push 化,第一步(v0/v1 共用地基):
**host schedule 预归约表 + 对账工具**,先裁决干净再动 kernel。
**状态**:✅ host 表 + 对账工具落地,NE∈{64,128,256} 4 卡对账全过(total_failures=0)。
**范围**:纯 host 侧(`_build_prereduce_schedule` + `tools/reconcile_prereduce.py`),
**未接入** `run()`/`setup()`,与已验证的 pull/push3 路径完全隔离。

---

## 1. 预归约在做什么(数学等价)

现状 combine(已验证)每个源 token 一次性把 8 条 expert 输出行加权求和:

```
out[t] = Σ_k  w(t,k) · E[erank(t,k)][slot(t,k)]      (k = 0..TOP_K-1)
```

其中 `E[d]` 是 expert 卡 d 的 `expert_out` 缓冲,`(erank,slot,w)` 就是 `comb_idx` /
`combine_w`。这条求和里,同一个源 token 的 8 个 expert 常常有多个落在**同一张卡**上
(NE=256、8 卡时期望每卡 8·(e_local/NE) 命中)。现状是源卡逐 (token,expert) 把这些行
**逐条跨卡拉回**(docs/12 实测:58.7MB gather,~32GB/s,占 layer1 尾 94~99%)。

预归约把这条和**按 expert 卡重新分组**:

```
partial[d][(s,t)] = Σ_{k : erank(t,k)==d}  w(t,k) · E[d][slot(t,k)]     (expert 卡 d 本地做)
out[t]            = Σ_d partial[d][(rank,t)]                            (源卡 s=rank 终归约)
```

**求和项完全不变,只是换了结合顺序**(FP32 累加,结合律下数值同量级)。收益:
expert 卡 d 对命中同 token 的多条本地行**先本地 FP32 加和成一行**再发,发送行数
从 4096(每 assignment 一行)降到 unique(src_token, expert卡) ≈ 1800(docs/11 §0.3 同因子),
跨卡流量 44MB → ~25MB;v1 再把数据面从源卡 pull 反转成 expert 卡 TMA push(强路径 51GB/s)。

**关键性质:因为只是重新分组,预归约表必须能对 `comb_idx` 逐项对账**——这就是本步先做表
+ 对账、把风险(host schedule 新表)裁决干净的原因(docs/11 §3)。

## 2. 表结构(`_build_prereduce_schedule`,tk_scheme.py)

复用与 `_build_schedules` **完全相同的 ring-by-source-device 写游标**(`write_pos`),
因此这里算出的本地 slot == dispatch 落的 slot(对账工具正是靠"两套独立代码算出同一 slot"
来交叉验证 slot 映射)。新增 all_gather `topk_weights`——因为权重按业界通例(DeepEP)在
**expert 卡侧**乘,而 expert 卡需要所有源卡的权重。

**expert 卡视角**(本 rank 作为 producer,d == rank):

| 表 | 形状 | 含义 |
|---|---|---|
| `prered_dst`   | (J, 2)     int32 | 每个 partial job 的目标 (src_dev, src_tok) |
| `prered_slots` | (J, TOP_K) int32 | 本卡要 FP32 归约的本地 slot 列表(-1 填充) |
| `prered_w`     | (J, TOP_K) f32   | 与 slot 对齐的权重(0 填充) |
| `num_jobs` J   | 标量             | 本卡的 unique (src_dev, src_tok) 命中数 |

一个 job = "本卡把命中 (src_dev,src_tok) 的所有本地行加权求和成 partial[d][(src_dev,src_tok)]"。
一个 job 最多 TOP_K 条(极端:一个 token 的 8 个 expert 全落本卡),典型 1~3 条。

**源卡视角**(本 rank 作为 consumer,s == rank):

| 表 | 形状 | 含义 |
|---|---|---|
| `final_contrib` | (num_tokens, world) int32 | `[t][d]==1` iff 卡 d 持有 t 的 ≥1 个 expert |

终归约 `out[t] = Σ_{d: contrib=1} partial[d][(rank,t)]`,只加真正有贡献的卡(消 HOL:
源卡只等 ≤world 个卡的信号,不再 per-token 等 8 个 max)。

**partial 行布局**:(src_dev, src_tok) 的预归约行放在 (world, num_tokens, H) buffer 的
平面 `[src_dev][src_tok]`——v0(源卡 pull 该行)与 v1(expert 卡 TMA push 到源卡
staging[d][t])共用此布局,是 v0→v1 的平滑台阶。

## 3. 对账工具(`tools/reconcile_prereduce.py`)

spawn 4 卡,同时建 golden `comb_idx`/`combine_w`(源卡视角)与 `_build_prereduce_schedule`
(expert 卡视角),all_gather 每张 expert 卡的 job 表,每个源 rank 对自己的 token
**双向逐项对账**:

- **(a) 覆盖**:每条 golden 项 `(t, erank, slot, w)` 必须在 expert 卡 `erank` 针对
  `(src=rank, t)` 的 job 里,找到**恰好一个**未消费的 `(slot, w)` 项(消费即置位,
  防重复计数);
- **(b) 无多余**:每个针对 `(rank, *)` 的 job 项都必须被 (a) 消费掉(双向 = bijection,
  杜绝漏项/伪项);
- **(c) contrib**:`final_contrib[t][d]==1` iff 卡 d 出现在 t 的 expert 集合。

结果:**NE∈{64,128,256} 全部 total_failures=0**。matcher 单测另验证其能拒绝重复计数
与 slot/权重不匹配(非空对账)。

复现:
```bash
source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
cd /data/cinnzhang_vllm_td_test/xxy
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.reconcile_prereduce
```

## 4. 下一步(T6-v0 kernel)

地基已裁决干净。接下来按 docs/11 §3:
1. **v0 kernel**:expert 卡侧新增 partial 预归约(读本地 slot 行,FP32 加和,写
   partial buffer 平面);源卡侧终归约小 kernel(按 contrib 加 ≤world 个 partial 行)。
   数据面仍走 pull(pull 的是预归约行,行数 4096→~1800),零新协议,风险集中在本步已裁决
   的 host 表。预期 combine 尾减半(docs/11 总账 ~6.2ms)。
2. **v1**:数据面反转为 expert 侧 TMA push + 水位信号(dpush3 骨架镜像,内存序链 =
   push3 R1 同款,`validate_push3` 方法论直接搬);治好 docs/12 的"零重叠"病。

---

## 5. T6-v0 kernel 落地实测(2026-07-09,✅ 达标且超预期)

**实现**(`kernels/tk/tk_moe.cu` namespace `prered`,`tk_scheme.py` `TK_COMBINE`):

- `moe_gemm_prered_fused`:W2 GEMM ⊕ **本地**预归约,单 kernel 同 grid 两类 block:
  - GEMM block(`num_comp_sms` 个):照常算 W2 写本地 `expert_out`,epilogue 改为
    **本地**完成信号——每 row block 选举一次(`atom.acq_rel.gpu`),满额者
    `st.release.gpu` 写 `barrier_l1[1][rb]=seq`。**只用 `__threadfence()`(gpu scope),
    不用 `fence.sys`**——job block 同卡,跨卡步骤挪到了后面独立的 barrier;
  - job block(`num_jobs` 个):每个 job 等它命中的 slot 所在 row block 的本地信号,
    把本卡命中 (src_dev,src_tok) 的 ≤TOP_K 条本地行 **FP32 加权求和**成一行,写
    peer-readable `partials[dev][src_dev*nt+src_tok]`。
- `pcie_device_barrier`(复用已验证 push2 的 `pcie_barrier_all`,走 `barrier_l0`,
  与 l1 不同张量无冲突):让所有卡的 partial 对系统可见。**零新协议**。
- `moe_final_reduce`:源卡每 token 一个 block,按 `final_contrib[t][d]` 加 ≤world 个
  `partials[d][rank*nt+t]`(权重已在 expert 侧乘,这里纯求和)→ `combine_out`。

**正确性**(`tools/validate_prered.py`,4 卡):以已验证 pull combine 为 golden,
prered 全链路 30 迭代对比 `combine_out`:
- **NE∈{64,128,256} 全过 total_failures=0**,max_rel_err **6.9~7.1e-3**
  (比 pull 基线 4.3e-3 多一次 bf16 舍入:FP32 partial→bf16→FP32 终和,符合预期);
- `run_tkfused` 全链路对拍 reference_moe:prered rel_err **4.42e-3 ok**(pull 4.27e-3)。

**性能**(4 卡 9,11,13,15,512 token/rank,e2e):

| NE | pull(旧) | **prered(T6-v0)** | serial | 说明 |
|---|---|---|---|---|
| 64  | 5089µs | **3712µs** (−1377) | — | run_tkfused |
| 256 | 7131µs | **5775µs** (−1356) | 7063µs | bench,**首次超过 serial** |

NE=256 e2e 7131→5775µs(**−1356µs**),**优于 docs/11 总账 ~6.2ms 的预期**,且
tkfused 首次打过 serial(5775 vs 7063)。**prered 全档位严格优于 pull**(不像 push3 有
NE 交叉点——预归约只减字节、无 ∝ 行块数的协议开销),故**设为默认**(`TK_COMBINE=prered`);
`TK_COMBINE=pull` 保留旧路径。

**为什么 v0(仍 pull 数据面)已经拿到这么多**:docs/12 实测 combine 尾 1838µs 是
per-(token,expert) 逐行跨卡 gather 58.7MB @ ~32GB/s。预归约把它降到 unique(token,卡)
≈1800 行:(a) expert 卡本地把命中同 token 的多条行先加成一行,跨卡搬运量 44MB→~25MB;
(b) 更关键——**终归约每 token 只读 ≤world(=4)个 partial 平面,不再逐 expert 各拉一行**,
且这些 partial 行是 GEMM 完成即就绪、边算边被 barrier 后的 final_reduce 消费,HOL 大幅缓解。

**下一步(v1)**:数据面反转为 expert 侧 TMA push + 水位信号(docs/11 §3),把跨卡 pull
换成强路径 51GB/s push,并让 partial 就绪即推、贯穿 GEMM,进一步压 combine 尾。

---

## 6. T4 gate+up 合并(2026-07-09,穿插小活,✅)

docs/11 §T4。`w1` 本是 `[gate; up]` 连排 (E, 2*inter, H),`w1.transpose(1,2)` 即
`w_gateup` (E, H, 2*inter),列 `[gate | up]`。pull dispatch⊕GEMM 的 N 从 inter 变 2*inter
(`col_blocks` 由 `weights.cols()` 派生,**GEMM 模板零改动**),一次 fused GEMM 出
`gateup_out (padded, 2*inter)`,silu·mul 直接切两半;删掉独立 up `grouped_gemm`。up 的
计算藏进 comm-bound dispatch。

- **实现**:`tk_scheme.py` `TK_FUSE_GATEUP`(默认 1)。fuse 时只 materialize `w_gateup`
  (省内存);push* dispatch 自动回退双 GEMM(它们 push 进 w_gate,不支持翻倍 N)。
- **正确性**:run_tkfused 对拍 reference_moe,NE=64 rel 4.42e-3、NE=128 4.34e-3 ok
  (NE=256 的 reference dense 路径 OOM,是**既有问题**,非 T4;bench --no-verify NE=256 正常)。
- **性能累积**(4 卡,512 tok/rank,e2e):

| 里程碑 | NE=64 | NE=256 | vs serial(7063) |
|---|---|---|---|
| 原 pull combine | 5089µs | 7131µs | 慢 |
| + T6-v0 prered | 3712µs | 5775µs | 快 |
| + T4 fuse gate+up | **3234µs** | **5571µs** | **快 1.27×** |

NE=256 T4 收益 −204µs 小于 ~0.5ms 预估:N 翻倍后 dispatch 偏 compute,up 未完全隐藏——
**T5(ROW_BLOCK=64)减半 GEMM 行后 up 才完全沉入通信**(docs/11 §T4)。
