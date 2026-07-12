# 43 FP8:copy-engine 通信(打破 SM 零和)与 TK 原语审计

> 用户两点指示:① 通信改走 copy engine(0 SM),SM/CE 两版对测;
> ② 审计代码,凡 TK/PK 已有原语的地方改用原语,遵循其设计哲学。

## 1. TK 原语审计结论

| 手写处 | TK 对应物 | 处理 |
|---|---|---|
| `bar.sync 2` 内联 asm ×5 | `kittens::group<NUM_WARPS>::sync(2)` | ✅ 已全部替换 |
| host 侧 CE 编排 | **TKParallelTensor.raw_ptrs_(IPC 指针簿)+ 流上 cudaMemcpyAsync**;KittensClub 是单进程多卡版,我们每 rank 一进程,等价物即本进程 side streams | ✅ ce:: 按此实现 |
| pcie_sync(选举/信号/自旋) | TK 跨卡原语 = multimem/red(sm90+/NVLink),PCIe 无对应物 | 保留(docs/08 实测裁决) |
| rowgroup 量化 shfl 归约 | rv+warp 归约可写,但 elementwise kernel 属常规 CUDA 域 | 保留 |
| push_job/final_red float4 循环 | sv/rv 可重写,零收益 | 保留 |
| GLU/plain store policy | store policy 本就是模板的用户扩展点 | 保留 |

## 2. CE 通信设计(ce:: 命名空间)

动机:docs/35 的结构性结论——SM 驱动通信下重叠是零和(让渡税实测
~264µs)。CE 不占 SM,是唯一能把让渡税变成真增量的路径(experience/12
§4 的远期项)。

- **L0(ce_ag_pull)**:主流 event(在 pcie 屏障后,各 rank 量化已可见)
  → 每源卡一条 side stream:`cudaMemcpyAsync`(D2D peer,UVA 走 CE)把
  pre_tokens/pre_scales 分片拉进本地 ag 缓冲 → 同流 4B flag。kernel 侧
  comm 块只做**本地 scatter**(TMA 走 L2,分片 flag 本地轮询放行)——
  预期 comm_sms 拐点大幅左移(不再需要 24 个 SM 撑 PCIe 并发),
  05c8 重扫 {4,8,12,16};
- **L1(ce_rs_push)**:push_job<CE> 归约直写本地 out_planes(免 smem/
  TMA/选举)→ kernel 后主流 event → 每目的卡一条 stream:plane(4MB)
  CE 推到对端 staging 行段 + 同流 4B seq 写对端 barrier (2+me,0)——
  **watermark 槽位与 SM 版完全一致,final_reduce_push 零改动**;
- **跨迭代护栏**:ce_rs_fence(下一迭代 L1 kernel 写 out_planes 前,
  主流 wait 上次 CE 读完的 per-stream event);L0 侧天然安全(下迭代
  copies 等 event,event 在上迭代 kernel 后);
- 正确性关键:TMA 读走 L2(CE 写 L2 一致)✓;flag 同流尾随 copy ✓;
  开关 TK_L0_CE / TK_L1_CE(默认 0,SM 版仍是默认,数据说话)。

## 3. 预期与对测

- L0 CE:让渡税 202 大头可回收(comm 块只剩本地 scatter ~90µs 即转岗),
  L0_fused 935 → **~800-830**;拐点左移进一步放大;
- L1 CE:kernel 尾不再推线(排空只剩归约),但 plane 推送后置串行
  (4MB×3 并行 CE ≈ ~100µs)——**账面接近盈亏平衡,由 04c8 A/B 裁决**;
- e2e(L0CE+L1fp8):~1700 → 若 L1 CE 也赢 → ~1650(1.26×+)。

## 4. 下一步

```bash
STEPS='^00_|^01_|^03f8_|^03c8_|^04f8_|^04l8_|^04c8_|^05c8_|^08f_|^08c8_' bash tools/run_tp_all.sh
```

看点:03c8 正确性、04c8 的 L0CE/L1CE/双 CE 三档 vs SM 版(04f8)、
05c8 拐点、08c8 归因(L0 让渡是否消失)。
