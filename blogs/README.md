# 博客系列：从零写通算融合算子

面向"只会一点 GPU 编程"的读者的六篇教学系列，覆盖 ThunderKittens 写融合
算子的教程、本仓库融合算子的实现细节（结合真实代码）与全部优化经验。

| # | 文件 | 内容 |
|---|---|---|
| 1 | `01-why-fuse-comm-and-compute.md` | 为什么要通算融合：串行瓶颈、收益账、四种模式地图、两条技术路线 |
| 2 | `02-know-your-platform.md` | 平台事实：PCIe 拓扑、三条红线、push/pull 强弱路径、microbench 方法论 |
| 3 | `03-thunderkittens-multi-gpu.md` | TK 多卡编程：tile/TMA/pgl、persistent + block specialization、可挂钩子的 grouped GEMM 模板 |
| 4 | `04-pcie-sync-protocols.md` | PCIe 同步协议：单写者 slot + 单调序号、选举 + 水位、内存序链、死锁与 barrier 审计 |
| 5 | `05-fused-moe-in-practice.md` | 实战：调度表设计 + 两个融合 kernel 完整走读 + 三道正确性锚点 |
| 6 | `06-tuning-attribution-and-pitfalls.md` | 调优：公平口径、三条账归因、七轮迭代史、踩坑速查表、远程迭代实践 |

## Astro 接入

文件即 Astro content collection 格式（markdown + frontmatter）。放入任意
Astro 博客工程的 `src/content/blog/` 即可；frontmatter 字段：

```yaml
title: string          # 标题
description: string    # 摘要(列表页/SEO)
pubDate: date          # 发布日期
tags: string[]         # 标签
series: string         # 系列名(自定义字段, 可用于聚合页)
order: number          # 系列内顺序(自定义字段)
```

若工程的 content schema 未定义 `series/order`，在 `src/content/config.ts`
里加两个可选字段，或删除这两行（不影响正文）。

## 素材来源

- 教程与经验：`experience/`（12 篇平台与相关工作研读）、`docs/19~27`
  （TP 线七轮迭代记录）、`docs/01~18`（EP 线）
- 代码：`kernels/tk/tk_moe.cu`、`kernels/tileoverlap/common/sm120_common.cuh`、
  `tk_tp_scheme.py`
- 数字：`microbench/results/20260710_071848/`、`tp_test_results/`（各轮 zip）
