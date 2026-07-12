# 32 TP 第十二轮计划:L1 combine 按 N 维分解(Comet layer1-N 教训)

> 用户点破 + experience/02 §2 印证:MoE GEMM1+combine 的可分解维度是 **N
> (输出列块)**,M 维明确"不可"——combine 要对同一 token 的 topk 行归约,
> 行与行强耦合。我们的 L1 v1 恰好是按 M 分解的,本轮换 N。

## 1. 为什么 v1(按 M)必然拖尾

v1 的 job = 一个 token,就绪条件 = 它 TOP_K 个 slot 的行块全部算完。
topk=8 时 8 个 slot 近似均匀散布全表,**max slot 的期望位置在 ~8/9 处
→ 约九成 job 拖到 GEMM 尾部 ~11% 才解锁**:push 挤在尾巴、L1 comm 块大部分
时间 spin(第十一轮实测 L1 暴露 186µs 的主要构成,此前误记为纯 SM 让渡)。
job_order 按 max-slot 排序只能让"先熟的先吃",改变不了几乎全都晚熟。

## 2. v2 设计(kernel: tppr2,TK_L1=v2 默认 / v1 回滚)

N 维切 combine 本身不改变行块何时完成,必须配套生产端换序,共三件:

1. **W2 GEMM 列外层扫描**:`grouped_gemm_sm120_dispenser<COL_MAJOR=true>`
   (模板新增参数,task t → rb = t%nblk, cb = t/nblk)。第 c 个列扫在
   ~(c+1)/32 的 GEMM 进度处完成——完整输出列切片提前成型;
2. **信号按列扫聚合**(docs/02 §3 计数聚合):每个 (rb,cb) tile 完成给
   cb 计数器 +1(barrier_l1 行 0,重用途),凑满 nblk = 该列全表就绪,
   单写者盖 seq 到行 1 列 cb。**一个信号放行该列全部 2048 个 token 的
   combine**;per-job 的 max-slot wait_slot 与 job_order 全部删除,
   协议净简化(job_order 也移出 v2 的 sched 计时重建,省一个 argsort);
3. **push job = (token, chunk)**:chunk = CHUNK_CB(=4)个列块 = 512 列
   = 1KB bf16。等 chunk 的 4 个列扫信号 → fp32 加权归约 TOP_K 行的列段
   → TMA 推 1KB 到源卡 staging 行内偏移(staging pgl 的 TMA 描述子改
   sv_bf<512>)。job 序 chunk-major,comm 块从 chunk 0 顺次消化——
   **push 从 GEMM ~1/8 进度开始流**,而非尾部倾泻。watermark 选举不变
   (expected × NCHUNKS);final_reduce_push 零改动。

角色结构与 v1 相同:comp 块 GEMM 完经 bar.sync 2(docs/24)转岗入 push
池;comm 块(TK_COMM_SMS_L1=24)从头消化。

## 3. 预期账与风险

- push 总量 12.6MB remote ≈ 250µs wire,v1 挤在尾部;v2 均匀摊进 GEMM 的
  ~508µs → 基本全隐藏,尾巴只剩最后一个 chunk(~31µs)+ final_red。
  **L1 暴露预期 186 → ~50-80µs;e2e 2125 → ~2000-2050(≈1.28-1.31×),
  T=1024 收益更大**;
- 风险①:1KB push 的 PCIe 效率(v1 是 8KB 整行;PCIe TLP 本就 256B 粒度,
  预计损失有限,由 stages 的 L1 暴露实测裁决;不够就把 TK_L1_CHUNK_CB
  提到 8 = 2KB,编译期常量,重编译生效);
- 风险②:列外层对 W2 GEMM 本身的 L2 复用影响(行外层复用 A 块,列外层
  复用同 expert 的 B 列片),用 08l(v1)vs 08(v2)的 L1_gemm_alone…
  注意 GEMM-alone 参考仍是行外层 plain 版,对比看 L1_fused 与暴露即可;
- 风险③:选举计数 ×8(16384 次本地原子),量级无虞。

## 4. 裁决与脚本

- 03l(TK_L1=v1)正确性门加入,失败可与 03(v2)二分定位;
- 04l A/B:L1 v1 单项回退 bench,给 N 维分解独立定价;
- 08l:v1 归因对照(看 L1 暴露 186 → ?);
- 本地 CPU 预检全绿(builder 逻辑未变,job_order 改按需构建)。
