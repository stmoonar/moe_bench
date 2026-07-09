# 14 T3:schedule GPU 化并计入 run()(公平性/可信度)

**日期**:2026-07-09
**任务**:docs/11 §T3 / docs/07 P1 —— 把 host 侧的 schedule 预计算 GPU 向量化并**计入
timed run()**,消除"打平/超过 serial"结论的星号。
**状态**:✅ 落地。默认路径(pull dispatch + prered combine + fused gate+up)的 schedule
每迭代在 GPU 重算并计时;NE∈{64,128,256}×{balanced,skewed} 对 host golden **element-wise
全等**;e2e NE=256 计入后 5832µs,仍 < serial 7063µs(1.21×)。

---

## 1. 为什么要做(公平性缺口)

`_build_schedules` 是 ~65k 次 Python 循环,在 `setup()`(**不计时**)里跑。而 serial 的
`fused_experts` 每次 `run()` 都在 GPU 上做 routing metadata(`moe_align_block_size` 排序/
对齐)+ 3 次 all_gather(hidden/ids/weights,**都在计时内**)。所以"tkfused 5571 打过
serial 7063"带星号:tkfused 的 routing/schedule 成本被藏在 setup。

T3 去掉星号:把默认路径消费的 schedule 表 GPU 向量化,**每迭代在 run() 里重算并计时**,
并把 topk 的 all_gather 移进 run()。口径(用户确认):重算索引**内容**并计时;grid **尺寸**
(num_padded_local, num_jobs)仍在 setup 定(routing 每问题固定,serial 同样 pre-alloc 到
静态 max),不引入每迭代 GPU→host 同步。范围仅默认路径 6 张表:disp_idx / padded /
prered_dst / prered_slots / prered_w / final_contrib。push*/pull-combine 的实验表仍 host 建。

## 2. 排序恒等式(核心)

host `_build_schedules` 与 `_build_prereduce_schedule` 共用同一个 slot 游标:expert 卡
e_rank 按 ring index i(src_dev=(i+e_rank)%world)→ src_tok → kpos 遍历,per-expert
`write_pos` 每命中 +1(跨所有源卡共享)。于是一个 assignment 的 slot = 它在总序中的名次:

    key = (eid, ring_offset, src_tok, kpos),  ring_offset = (src_dev − e_rank) mod world

因为 (src_dev, src_tok, kpos) 是 flattened all_topk 的双射,**每个 key 唯一** → argsort 无需
稳定(CUDA argsort 不稳定,唯一性绕开)。一次全局 argsort 即复现 ring-order slot 分配。

- **padded / 游标**:`counts = scatter_add(eid)`(**不用 bincount**——它有 device sync,
  破坏 CUDA graph 捕获;scatter_add 等价且可捕获);`padded=⌈counts/128⌉·128`;
  `pos_in_expert = arange(N) − excl_cumsum(counts)[eid_s]`。
- **disp_idx**:本卡 assignment 的 `slot_abs = padded_base[el_idx] + pos_in_expert`;
  非本卡的用 masked el_idx=0 保持 in-bounds,scatter 到 trash 行 P 后丢弃(**定形**,
  无布尔压缩)。
- **prered(稠密 job 空间,列=kpos)**:见 §3。
- **final_contrib**:`scatter_(1, all_topk[rank]//e_local, 1)`。

## 3. 稠密 job 空间(让 builder 又快又可 CUDA-graph 捕获)

原始 prered 用 `unique` 压缩 job + 第二次 argsort 排 kpos 列——两者既慢(unique 194µs +
argsort2 105µs)又**破坏 graph 捕获**(动态 shape + sync)。改成**稠密 job 空间**:

- num_jobs = world·num_tokens(**常量**,与 routing 无关),job j = src_dev·T + src_tok
  覆盖所有 (src_dev,src_tok) 对,正好对齐 partials buffer 的行布局;
- 列直接用 **kpos**(FP32 求和与列序无关,无需 unique/排序/cumcount);
- prered_slots 的 flat cell n = (src_dev·T+src_tok)·top_k + kpos **==原始 flat assignment
  下标**,所以只需把每个 sorted 本卡 assignment 的 slot `scatter_(0, order, slot)` 回其原始位;
- 无命中的 job 全 -1,combine 端算出一行零、写到不被读的 partial 行(final_contrib=0);
- 代价:多 ~248 个空 job block(2048 vs ~1800),各写一行零,可忽略。

host golden `_build_prereduce_schedule` 同步改成稠密(列=kpos),保证对账仍 element-wise。

## 4. CUDA graph 捕获

builder 是纯计算(无 NCCL)、定形、定地址 → 首次 run() warmup 后用
`torch.cuda.CUDAGraph` 捕获,之后 `replay()`。这把 ~30 个 eager kernel launch 从
**660µs 压到 176µs**。all_gather 留在 graph 外(NCCL 此处不可捕获,仅 29µs)。
关键坑:`bincount` 会 sync 破坏捕获 → 换 `scatter_add`。

## 5. 实测

builder 成本(tools/time_schedule,4 卡 NE=256):

| 版本 | all_gather×2 | builder | 合计 |
|---|---|---|---|
| 原始(unique+2argsort,eager) | 29µs | 1166µs | 1195µs |
| 稠密(eager) | 29µs | 564µs | 593µs |
| **稠密 + CUDA graph** | 29µs | **176µs** | **~205µs** |

e2e(bench,4 卡,512 tok/rank,NE=256):

| 口径 | e2e | vs serial 7063 |
|---|---|---|
| schedule 在 setup(不计时) | 5603µs | 1.26× |
| **schedule 计入 run()(graph)** | **5832µs** | **1.21×** |

计入后仅 +229µs(≈ 205µs builder+ag,匹配),**结论(超过 serial)在两种口径下都成立**,
星号已去掉。NE=64 verify 3462µs、NE=128 3550µs(rel_err 4.42/4.34e-3 ok)。

## 6. 正确性裁决

- **tools/verify_schedule_gpu.py**:4 卡,同 seed,host golden vs GPU builder,6 张表
  `torch.equal` 逐元素;NE∈{64,128,256}×{balanced,skewed} **全过 total_failures=0**
  (skewed 覆盖空专家/不均 padding 的高危档)。
- reconcile_prereduce(稠密 host golden)3/3;validate_prered(prered vs pull combine)
  3/3 max_rel~7e-3;run_tkfused 全链路 rel_err 4.42e-3 ok。

`TK_GPU_SCHED=1` 默认开(仅默认路径 pull+prered 生效);`=0` 回退 setup 预计算。

## 7. 复现

```bash
source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
cd /data/cinnzhang_vllm_td_test/xxy
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.verify_schedule_gpu
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.tools.time_schedule 256 100
CUDA_VISIBLE_DEVICES=9,11,13,15 python -m moe_bench.bench --distributed --scheme tkfused \
    --mode ep --precision bf16 --world-size 4 --no-verify --num-tokens 512
```
