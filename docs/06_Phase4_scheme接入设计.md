# Phase 4 设计：接入 moe_bench 作为 DistributedScheme（bf16 EP v1）

**日期**：2026-07-08
**目标**：把已验证的两个融合 kernel（layer0 dispatch⊕W1、layer1 W2⊕combine）拼成完整
EP MoE 层，接入 `schemes.py` 的 `DistributedScheme`，与 `serial` baseline 对比、对拍
`reference_moe`。用户选定 **先 bf16 打通**，fp8 后续。

## 1. moe_bench 契约要点（来自 reference.py / distributed.py / data.py）

- 运行器：`mp.spawn` 起 world_size 个进程，每进程 `rank==local_rank==device index`。
- `MoEProblem`（EP，bf16）给的是本 rank 的：
  - `hidden_states (num_tokens, H)` bf16，num_tokens=512。
  - `w1 (E_local, 2*inter, H)` —— **gated FFN**：前 inter 行是 gate，后 inter 行是 up。
  - `w2 (E_local, H, inter)`。
  - `topk_ids (num_tokens, top_k)` 全局 expert id；`topk_weights (num_tokens, top_k)`。
  - `expert_map (E_global,)` 全局→本地 slot（-1 不在本 rank）。
  - EP：E_local = 256/world_size，inter_shard = inter = 2048（EP 不切中间维）。
- 输出：`run()` 返回本 rank 的 `(num_tokens, H)` bf16。
- 正确性：每 rank 输出 = 本 rank token 过**全量专家**的完整 MoE（golden）。bf16 容差
  atol=2e-2。
- **run() 必须无副作用、分配稳定**：所有 buffer 在 setup 分配，run 里 copy_ 复用。

参考数学（reference.py `_reference_dense` + combine）：
```
per (token t, its top-k expert e):
  tmp1 = a[t] @ w1[e].T            # (2*inter,)  gate|up
  tmp2 = silu(tmp1[:inter]) * tmp1[inter:]     # (inter,)
  y_e  = tmp2 @ w2[e].T            # (H,)
out[t] = Σ_k topk_weight[t,k] * y_{e(t,k)}
```
注意 reference 的 w1/w2 是 `@ w.T`（w1 存成 (2*inter, H)，w2 存成 (H, inter)）。
我们的 grouped_gemm 算 `x @ W`（W 存成 (K, N)），所以喂权重时要转置好布局。

## 2. v1 数据流（EP，bf16，用两个融合 kernel + 一次辅助 GEMM）

每 rank 同时是「源卡」（有自己的 512 token）和「专家卡」（有 E_local 个专家）。

```
setup（一次）:
  - routing → dispatch schedule（pull_dispatch_indices，按 expert 排序+128 padding+ring）
                combine schedule（combine_indices = dispatch 逆映射 + combine_weights）
  - 权重布局：w1 拆成 gate(E_local, H, inter) 和 up(E_local, H, inter)（转置成 x@W 布局）；
              w2 → (E_local, inter, H)。
  - TKParallelTensor：pre_tokens(512,H)、expert_out(inter 维中间张量按 padded_max)、
                      combine 用 expert_out2(H 维)、barrier；按 token 上限一次分配。
  - 普通 buffer：gathered(padded_local, H)、gate_out/up_out(padded_local, inter)、
                 act(padded_local, inter)、combine_out(512, H)。

run（每次，分配稳定）:
  1. pre_tokens.copy_(hidden_states)
  2. device_barrier(seq++)              # 保证所有卡 pre_tokens 就绪
  3. layer0 融合（02 kernel）: dispatch ⊕ grouped_gemm(gate) → gate_out
     （dispatch 把 token 拉到本地 gathered，同时算 gate GEMM）
  4. up GEMM（01 kernel，tokens 已本地）: gathered @ up → up_out
  5. act = silu(gate_out) * up_out       # torch，(padded_local, inter)
  6. layer1 融合（03 kernel）: grouped_gemm(W2) ⊕ combine → combine_out
     （expert 卡算 act@W2 → expert_out2，源卡 pull+FP32 combine）
  7. return combine_out                  # (512, H)
```

说明：
- 为什么 up 用单独 01 GEMM 而不再融合：dispatch 只需做一次（token 拉到本地后
  gate/up 都在本地）。gate 融进 dispatch（02），up 直接对已 gather 的 token 跑 01。
  这符合 PLAN §6「v1 跑两次 grouped_gemm，第二次直接对 inputs_gathered 调 01」。
- combine 的 expert 输出是 W2 的结果（H 维），03 fused kernel 直接产出并 combine。

## 3. 风险与验证

- **broker 与 mp.spawn**：TKParallelTensor 用固定 key 的 shm + /tmp socket，按 local_rank
  握手。harness 里 rank==local_rank==device，应兼容。**先单测**：一个最小 scheme 只做
  「pre_tokens.copy_ + device_barrier + 返回 hidden_states」跑通 spawn，确认 broker 不炸。
- **NUM_DEVICES 编译期**：TK 扩展按 world_size 编译（TK_NUM_DEVICES）。scheme setup 里按
  ctx.world_size 选择/编译对应 .so；先支持 world_size=4（默认卡组）。
- **schedule 用 topk_ids 现算**：v1 用 torch 向量化（argsort+cumsum），host 端算，D2H 一次
  同步可接受（setup 里做，不计时）。生产再上 GPU metadata kernel。
- **padding 与容差**：padding 行的 combine 不参与（combine_indices=-1）。bf16 atol=2e-2，
  我们单元测 max diff ~1e-4，接完应轻松过。

## 4. 交付顺序

1. broker × mp.spawn 最小连通性验证（trivial scheme）。
2. 把 01/02/03 三个 .so 统一成一个 scheme 可 import 的扩展（或分别 import）。
3. 实现 `TKFusedEP` scheme，先只接 bf16、world_size=4、EP。
4. 对拍 `reference_moe`（--scheme tkfused --mode ep --precision bf16 --world-size 4）。
5. 与 serial baseline 比性能。
6. 沉淀结果；fp8 作为后续。
