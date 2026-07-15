# 关键 bug：group::store 的 warpgroup 交织行映射与 consumer 输入行不匹配

**日期**：2026-07-08
**严重性**：高——静默产生约 6% 相对误差的错误结果，且骗过了 Phase 1 的容差检查。
**影响文件**：`tileoverlap/common/sm120_common.cuh`（grouped_gemm_sm120 的 consumer 循环）

## 症状

Phase 3b 融合 kernel 对拍 torch，mean diff ≈ ref mean（0.0077 vs 0.0064）——不是舍入，是错的。
逐层排查发现**W2 GEMM 那一半就错**。进一步用 01 单卡 kernel 复现：
- 逐行看误差：**第 0~15 行完全正确（err=0），第 16 行往后全错**。
- 每个 consumer warp 负责一个 16 行条带；只有 warp 0 的条带对，warp 1~7 全错。
- 相对误差恒定 ~6%，与 K、输出宽度无关。

## 根因

`kittens::group<8>::store`（`include/ops/group/memory/tile/global_to_register.cuh:117`）
在 `GROUP_WARPS % 4 == 0` 时用 **warpgroup 交织**映射决定每个 warp 写哪个行条带：

```cpp
local_warpid = warpid()/4 + (warpid()%4)*(GROUP_WARPS/4);   // 8 warp: 0,2,4,6,1,3,5,7
const int row_offset = RT::rows * local_warpid;             // 该 warp 写的行 = local_warpid*16
```

即 warp `w` 的累加器被写到**第 `local_warpid` 个 16 行条带**，不是第 `w` 个。
（这是为 SM90 wgmma 的 warpgroup 布局设计的。）

而 `grouped_gemm_sm120` 的 consumer 循环用**原始 `warp_id`** 加载输入 A 的行条带：

```cpp
auto a_sub = inputs[stage].A.template subtile<16, 16>({warp_id, kk});  // 错：用 warp_id
```

于是 warp 1 **算的是第 1 条带的输入**，却把结果**存到第 2 条带**（local_warpid=2）。
warp 0 恰好 local_warpid=0 一致，所以只有它对——完美解释「第 0~15 行对、其余全错」。

## 修复

consumer 加载 A 时用与 store **相同**的 `local_warpid`：

```cpp
constexpr int WG = cfg::CONSUMER_WARPS;
const int store_strip = (WG % 4 == 0) ? (warp_id / 4 + (warp_id % 4) * (WG / 4)) : warp_id;
auto a_sub = inputs[stage].A.template subtile<16, 16>({store_strip, kk});
```

修复后 01 的 max diff 从 0.05~0.8 降到 **~0.0002**（K、输出宽度全谱系）。

## 为什么之前没抓到

Phase 1 的 benchmark 只有一个 `assert max_diff < 0.1`。K=7168 时错误的 max diff 恰好
≈0.076~0.09，**侥幸压在 0.1 以下**（K 越大、每元素量级越小，绝对误差越小）。mean diff
早就 ≈ ref mean（0.0099 vs 0.0094）——这个信号当时被误判成「bf16 大 K 舍入」写进了
docs/02。**教训**：正确性判据要看 **mean diff / ref mean 的比值**（应 <<1），不能只看
max diff 的绝对阈值；relative error 恒定不随 K 缩小就是系统性 bug 的铁证。

## 连带修正

- docs/02_Phase1 的「坑 2（144896 token 略超阈值是 bf16 舍入）」结论**作废**——那不是舍入，
  就是这个 bug。修复后 144896 token 也 max diff 0.0002 通过。
- Phase 1（01）/ Phase 2（02）/ Phase 3a（03 anchor）/ Phase 3b（03 fused）重新全部对拍：
  max diff 均 ~1e-4 量级，全部真正正确。

## 另一处（Phase 3b 特有）：跨卡可见性需 system fence

融合 combine 的输出被 **peer 卡**通过 PCIe 读。GEMM 的 8 个 consumer warp 各写自己的行条带后，
仅靠信号 warp 的 `st.release.sys` 只能保证**它自己**的写可见。修复：epilogue 里**每个 consumer
线程**都 `__threadfence_system()` 刷自己的条带，再 `group::sync` 汇合，最后由单线程发
`signal_slot`。这样 peer 读到的 expert_outputs 一定是完整刷新的。
（layer0 dispatch 不需要，因为它读的是本地拉取的数据，gpu scope 足够。）
