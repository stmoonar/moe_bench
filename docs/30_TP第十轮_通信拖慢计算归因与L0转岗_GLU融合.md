# 30 TP 第十轮:通信拖慢计算的定量账、L0 comm 块转岗与 SwiGLU 融合

> 承接 docs/29。用户裁定:不再管 vLLM triton 调优,回到我们自己的融合算子。
> 本篇先给"还有多少肉 + 通信拖慢了计算多少"的定量账,再落地三项优化。

## 1. 通信把计算拖慢了多少(T=512, comm24,三轮稳定数据)

| 层 | GEMM 单独 | 融合后 | 拖慢 | 定性 |
|---|---:|---:|---:|---|
| L0(AG⊕gate+up) | 978 | 1232 | **+254µs (+26%)** | ≈100% 是 SM 让渡 |
| L1(W2⊕prered push) | 508 | 698 | **+190µs (+37%)** | ~142µs SM 让渡 + ~48µs 排空尾 |

**判据**:SM 让渡模型 110/(110−24) = +27.9%。L0 实测 +26% < 模型 → 暴露完全由
"让出 24 个 SM"解释,**数据等待 ≈ 0**(docs/20 的调度序修复已把流水贴合);
L1 实测 +37% 超出模型的部分(~48µs)才是真正的通信协议成本。

comm_sms sweep 的 U 型(8:2722 / 16:2334 / **24:2260** / 32:2412 / 40:2610)
本质是"等数据(少 SM 拉不动)vs 让 SM(多了白占)"的**静态**折衷。
结构性解法不是调这个数,而是**打破折衷:comm 块干完活转岗算 GEMM**。

## 2. 还有多少肉(T=512,当前 2260µs)

| 项 | 现值 | 可压 | 手段 |
|---|---:|---:|---|
| GEMM 地板(L0+L1) | 1486 | — | 是分母不是肉 |
| L0 SM 让渡 | ~255 | 大半 | **本轮:comm 块 AG 后转岗(v2 dispenser)** |
| 独立 silu kernel | 109 | ~全部 | **本轮:GLU 融进 L0 epilogue** |
| sched(双 AG+graph) | 248 | ~40 | **本轮:ids+权重打包单 all_gather** |
| L1 SM 让渡 | ~142 | 部分 | **本轮:TK_COMM_SMS_L1 独立预算(首扫 8/16)** |
| L1 排空尾 | ~48 | 少量 | 后续 |
| tok_copy + final_red | ~50 | 少量 | 后续 |

理论地板 ≈ 1894µs(1.39×);本轮全兑现的乐观值 ~1900-1980(1.33-1.38×),
当前 1.16×。

## 3. 本轮落地(全部带独立回滚开关)

### 3.1 L0 v2:dispenser GEMM + comm 块转岗(TK_L0=v2 默认,v1 回滚)

v1 的结构问题:GEMM 是**静态划分**(task 按 sm_idx 跨步预分配),dispatch
块是短命块(12 token/块,排队在 24 个 comm SM 上),拉完就退出——AG 排空后
24 个 SM 闲置到 L0 结束。v2(`tpdisp2` + `grouped_gemm_sm120_dispenser`):

- GEMM task(row_block × col_block)由**全局原子 dispenser** 发放,claim 序
  = 原 expert-major 序(与 pull_order 就绪序的贴合不变);
- producer warp 通过 **smem 描述环(深 2,mbarrier 握手,与 stage 流水同款
  phasebit 模式)**把 task 流给 consumer warps——3 级输入流水**跨 task 连续**,
  无逐 task block barrier、无流水排空(这是没做成"静态两池"的原因:池比例
  猜不准,dispenser 不用猜);
- comm 块改为 **num_comm_sms 个常驻块**,按 pull_order 分波拉取(每波 12
  token,mbarrier 相位翻转复用 smem),拉完 `bar.sync 2`(docs/24:专用命名
  barrier,不能用 __syncthreads)汇合后**加入 GEMM dispenser**——与 L1 的
  "comp 块跑完转岗排空 push"(docs/20)互为镜像;
- 新表 `blk_expert (nblk,)`(row block → expert,dispenser 的 B tile 索引):
  host golden + GPU builder(searchsorted,capture-safe)双实现,
  verify_tp_schedule/preflight 扩了逐元素对拍 + 非降/计数不变量。

### 3.2 SwiGLU 融进 L0 GEMM(TK_L0_GLU=1 默认,0 回滚,独立于 3.1)

权重列交织(setup 一次置换):每个 128 列 GEMM tile = [gate64 | up64],同一
组 intermediate 列的两半落在**同一个累加器**里;consumer 在 fp32 寄存器上算
silu(gate)*up,直接存 64 宽 act tile(`glu_store_policy`)。收益:

- 省掉独立 silu kernel(109µs)+ gateup_out 的 50MB 写 + 75MB 读写回;
- **精度反而更好**(silu 作用在 fp32 累加器上,而不是 rounded bf16);
- 风险点:group::store 的 warpgroup 行置换(docs/05)按推导与 tile 宽度无关
  (A 侧 store_strip 公式不变),但**以正确性门为准**——脚本里 GLU on/off、
  v1/v2 三档分开对拍,失败可直接定位。

### 3.3 sched 合并 all_gather(builder 输入改打包)

ids(int32)+ 权重 bit pattern(float32.view(int32))打包成 (T,K,2) 单张量,
**一次** NCCL all_gather(原来两次);builder 从 packed[...,0] 取 ids、
把 packed[...,1] 通过 prered_w 的 int32 视图**位拷贝**(不是 dtype cast)。
打包 copy(2 个 ~5µs 小 kernel)留在计时区内,公平口径不变。预期 −40µs。

### 3.4 L1 独立 comm 预算(TK_COMM_SMS_L1,默认=TK_COMM_SMS)

L1 的 comm 块只流推送、从不帮 GEMM(排空靠 comp 块转岗),而 push 强路径
4 SM 就能打满(docs/22)——L1 预算可能远小于 L0 的 24。新步骤 05b 首扫
{8,16}。

## 4. 脚本与裁决更新

- 03 新增 `03g(GLU off)`/`03o(v1)` 正确性门(定位 dispenser vs GLU);
- 04 新增 `04g/04o` A/B bench(隔离每项收益);05b L1 预算 sweep;
- 08 新增 `08o(v1)` 归因对照;stages 工具 v2 感知(GLU 下 silu 应≈0,
  L0_gemm_alone 换 `grouped_gemm_glu` 苹果对苹果参考);
- 本地 CPU 预检已过(builder 打包/blk_expert 逐元素 + 数据流仿真)。

## 5. 预期与看点(下一轮产物)

- `03*` 三档全绿 → 协议成立;任何一档红 → 用 03g/03o 二分;
- 08 归因:GLU 下 silu→~0、L0_fused 预期 1232→~1050-1100(转岗回收)、
  sched 248→~205;
- 04 A/B:v2+GLU vs v2 vs v1 的阶梯 → 每项优化的独立定价;
- e2e 预期 2260 → **~1950-2050(1.28-1.35×)**;
- 05 sweep:v2 下 comm_sms 拐点可能右移(comm 块不再是纯开销)——若 32
  反超 24,说明转岗生效。
