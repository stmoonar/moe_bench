# SPDX-License-Identifier: Apache-2.0
"""每次运行打印一行"每个 expert 分到的 token 数"。

分布式下取**全局**口径(一次 all-reduce 合并各 rank 的本地计数): TP 每张卡都要
算整批 gathered token, 只有全局计数对得上 kernel 真正处理的行数 —— 与
``tk_tp_scheme`` 建调度表时的 ``bincount(all_topk)`` 同源。

打印默认开启, 设 ``MOE_BENCH_ROUTING_STATS=0`` 关掉。
"""

from __future__ import annotations

import os
from typing import Sequence

import torch

ENV_TOGGLE = "MOE_BENCH_ROUTING_STATS"


def enabled() -> bool:
    """打印开关(``MOE_BENCH_ROUTING_STATS``, 默认开)。

    分布式下它门控着 :func:`expert_token_counts` 里的一次 all-reduce, 所以**必须
    对所有 rank 取同一个值** —— worker 由 ``mp.spawn`` 继承父进程环境, 天然满足;
    不要改成按 rank 判断(只有一个 rank 跳过就会把其余 rank 挂在集合操作上)。
    """
    return os.environ.get(ENV_TOGGLE, "1") != "0"


def expert_token_counts(
    topk_ids: torch.Tensor,
    num_experts: int,
    group=None,
    global_counts: bool = True,
) -> list[int]:
    """每个 expert 分到的 token 数(= 落到它的 (token, expert) assignment 数)。

    ``global_counts`` 且进程组已初始化时 all-reduce 成全局口径。计时区外调用。
    """
    counts = torch.bincount(
        topk_ids.reshape(-1).long(), minlength=num_experts
    )[:num_experts].clone()
    if (global_counts and torch.distributed.is_available()
            and torch.distributed.is_initialized()):
        torch.distributed.all_reduce(counts, group=group)
    return counts.tolist()


def print_expert_tokens(counts: Sequence[int], label: str = "",
                        per_line: int = 16) -> None:
    """打印 ``[x, x, x, ...]``; 超过 ``per_line`` 个折行并按列宽对齐。"""
    if not enabled():
        return
    vals = [int(c) for c in counts]
    tag = "[routing] 每 expert token 数" + (f" ({label})" if label else "")
    if not vals:
        print(f"{tag}: []")
        return
    width = max(len(str(v)) for v in vals)
    cells = [str(v).rjust(width) for v in vals]
    if len(cells) <= per_line:
        print(f"{tag}: [" + ", ".join(cells) + "]")
        return
    rows = [", ".join(cells[i:i + per_line]) for i in range(0, len(cells), per_line)]
    print(f"{tag}:")
    print("  [" + ",\n   ".join(rows) + "]")
