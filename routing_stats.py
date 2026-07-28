# SPDX-License-Identifier: Apache-2.0
"""每次跑 benchmark 都打印路由分布与 BLOCK 布局。

打两件事:

1. **每个 expert 分到的 token 数** —— `[x, x, x, ...]`, 长度 = ``num_experts``,
   元素是落到该 expert 的 (token, expert) assignment 数。分布式下是**全局**
   口径(world 张卡的 token 合起来): TP 每张卡都要算整批 gathered token, EP 的
   dispatch 也按全局路由分, 所以只有全局计数才对得上 kernel 真正处理的行数。
2. **计算用的 BLOCK 配置** —— 行(token)维的 tile / padding 粒度 ``BLOCK_M``,
   以及由它推出的每 expert 块数 `[b, b, ...]`、padding 行数、尾块/半块数。
   docs/09 的 padding 粒度税就是从这两行直接读出来的。

``BLOCK_M`` 的口径由**实现**给出(见 :class:`BlockConfig`), 因为各实现不同:

- ``tktp``  : kernel 编译期的 ``ROW_BLOCK``, 同时是 expert 的 padding 单位;
- ``serial``: vLLM triton fused_moe 查表/兜底得到的 ``BLOCK_SIZE_M``, 也就是
  ``moe_align_block_size`` 的对齐粒度。

打印默认开启, 设 ``MOE_BENCH_ROUTING_STATS=0`` 关掉(批量脚本刷屏时用)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch

from .config import Precision

ENV_TOGGLE = "MOE_BENCH_ROUTING_STATS"


@dataclass
class BlockConfig:
    """一个实现的行维 BLOCK 口径。

    ``block_m``      token 行维的 tile / padding 粒度(必填)。
    ``desc``         这个值的来源, 一句话(打印在 BLOCK_M 后面的括号里)。
    ``detail``       附加的 tile 形状 / 开关, 可空。
    ``padded_rows``  每 expert padding 之后的实际行数。实现能给就给(``tktp``
                     的 local_first 分段布局下, 每 expert 占两段, 块数**不等于**
                     ``ceil(counts/block_m)``); 给 ``None`` 则按 counts 推。
    ``block_slack``  每个行块的 padding 行数(= ``block_m`` − 真实行数)。给了就
                     用它精确统计尾块/半块, 否则同样按 counts 推。
    """

    block_m: int
    desc: str
    detail: str = ""
    padded_rows: Sequence[int] | None = None
    block_slack: Sequence[int] | None = None


def expert_token_counts(
    topk_ids: torch.Tensor,
    num_experts: int,
    group=None,
    global_counts: bool = True,
) -> list[int]:
    """每个 expert 分到的 token(assignment)数。

    ``global_counts`` 且进程组已初始化时对本地计数做一次 all-reduce, 得到全局
    口径 —— 与 ``tk_tp_scheme`` 建调度表时的 ``bincount(all_topk)`` 逐元素一致,
    但省掉一次 routing 的 all_gather。计时区外调用, 不影响任何测量。
    """
    counts = torch.bincount(
        topk_ids.reshape(-1).long(), minlength=num_experts
    )[:num_experts]
    if global_counts and torch.distributed.is_available() and torch.distributed.is_initialized():
        counts = counts.clone()
        torch.distributed.all_reduce(counts, group=group)
    return counts.tolist()


def block_layout(counts: Sequence[int], block: BlockConfig) -> dict:
    """把 counts + BLOCK 口径展开成块布局统计。

    返回 ``blocks_per_expert`` 以及总块数 / padded 行数 / padding 行数 /
    尾块(不满一个 BLOCK_M 的块)数 / 半块(真实行 ≤ BLOCK_M/2, 两级 tile 的
    受益块, docs/09 §4)数。
    """
    bm = block.block_m
    if block.padded_rows is not None:
        padded = [int(p) for p in block.padded_rows]
    else:
        padded = [(int(c) + bm - 1) // bm * bm for c in counts]
    blocks = [p // bm for p in padded]

    if block.block_slack is not None:
        real_rows = [bm - int(s) for s in block.block_slack]
    else:
        # canonical 布局: expert e 的第 b 块装 [b*bm, (b+1)*bm) 里的真实行。
        real_rows = [
            max(0, min(bm, int(c) - b * bm))
            for c, nb in zip(counts, blocks)
            for b in range(nb)
        ]

    total_real = sum(int(c) for c in counts)
    total_padded = sum(padded)
    return {
        "blocks_per_expert": blocks,
        "num_blocks": len(real_rows),
        "padded_rows": total_padded,
        "pad_rows": total_padded - total_real,
        "pad_pct": (total_padded / total_real - 1.0) * 100.0 if total_real else 0.0,
        "tail_blocks": sum(1 for r in real_rows if r != bm),
        "half_blocks": sum(1 for r in real_rows if 0 < r <= bm // 2),
        "empty_blocks": sum(1 for r in real_rows if r == 0),
    }


def format_int_list(values: Sequence[int], indent: int = 4, per_line: int = 16) -> str:
    """``[a, b, c, ...]``。

    超过 ``per_line`` 个元素时折行并按列宽对齐 —— 仍然是一个方括号列表, 只是
    E=64/128 时一行读不过来。
    """
    vals = [int(v) for v in values]
    if not vals:
        return "[]"
    width = max(len(str(v)) for v in vals)
    cells = [str(v).rjust(width) for v in vals]
    if len(cells) <= per_line:
        return "[" + ", ".join(cells) + "]"
    pad = " " * (indent + 1)
    rows = [", ".join(cells[i:i + per_line]) for i in range(0, len(cells), per_line)]
    return "[" + (",\n" + pad).join(rows) + "]"


def enabled() -> bool:
    """打印开关(``MOE_BENCH_ROUTING_STATS``, 默认开)。

    分布式下它门控着 :func:`expert_token_counts` 里的一次 all-reduce, 所以**必须
    对所有 rank 取同一个值** —— worker 由 ``mp.spawn`` 继承父进程环境, 天然满足;
    不要改成按 rank 判断(只有一个 rank 跳过就会把其余 rank 挂在集合操作上)。
    """
    return os.environ.get(ENV_TOGGLE, "1") != "0"


def print_routing_stats(
    counts: Sequence[int],
    block: BlockConfig | None,
    header: str,
    indent: int = 4,
) -> None:
    """打印一屏 "每 expert token 数 + BLOCK 配置"。

    ``block`` 为 ``None``(实现没报口径)时只打路由那半边。
    """
    if not enabled():
        return
    counts = [int(c) for c in counts]
    total = sum(counts)
    e = len(counts)
    mean = total / e if e else 0.0
    label = "  "                 # 说明行
    body = " " * indent          # 列表行

    print()
    print(f"== 路由 / BLOCK 布局 ({header}) ==")
    print(
        f"{label}每 expert token 数 (assignments={total}, "
        f"min={min(counts) if counts else 0}, max={max(counts) if counts else 0}, "
        f"mean={mean:.1f}, max/mean={(max(counts) / mean if mean else 0.0):.2f}x):"
    )
    print(f"{body}{format_int_list(counts, indent=indent)}")

    if block is None:
        print(f"{label}BLOCK 配置: n/a (实现未报告行维 tile 口径)")
        print()
        return

    detail = f"; {block.detail}" if block.detail else ""
    print(f"{label}BLOCK 配置: BLOCK_M={block.block_m} ({block.desc}{detail})")
    lay = block_layout(counts, block)
    print(
        f"{label}每 expert BLOCK 数 (blocks={lay['num_blocks']}, "
        f"padded_rows={lay['padded_rows']}, padding={lay['pad_rows']} 行 "
        f"+{lay['pad_pct']:.1f}%, 尾块 {lay['tail_blocks']} "
        f"(其中半块 {lay['half_blocks']}, 空块 {lay['empty_blocks']})):"
    )
    print(f"{body}{format_int_list(lay['blocks_per_expert'], indent=indent)}")
    print()


# --------------------------------------------------------------------------
# vLLM triton fused_moe 的 BLOCK 口径(serial baseline / naive impl 共用)
# --------------------------------------------------------------------------

# fused_experts_impl 按 CHUNK 切 M 后再取 config(envs.VLLM_FUSED_MOE_CHUNK_SIZE,
# 默认 32768)。主口径 M=2048 远小于它, 这里只是为了大 token 扫描时别报错口径。
_VLLM_CHUNK_SIZE = 32768


def vllm_triton_block_config(problem, num_rows: int) -> BlockConfig | None:
    """vLLM triton fused_moe 实际会用的 tile config。

    首选 ``try_get_optimal_moe_config`` —— 与 ``fused_experts`` 内部同一条路
    (含 ``override_config`` 和 ``VLLM_TUNED_CONFIG_FOLDER`` 查表), 所以调优过
    与没调优过会如实反映出来(docs/08 §6: 本机查表落空, 兜底 BM=64)。它取不到
    就退到 ``get_default_config``(兜底分支本身, 签名已被 tools/tune_triton_moe.py
    实证), 那时打印会标 "兜底" —— 与查表命中区分开, 别把两者混为同一个数。

    ``num_rows`` 是喂给 ``fused_experts`` 的行数: serial 是 AllGather 之后的
    ``world_size × T``, 单卡 impl 就是 ``T``。

    vLLM 内部 API 变动只影响这一行打印, 两条路都拿不到就返回 ``None`` 打 n/a,
    不让 benchmark 因为它挂掉。
    """
    cfg = problem.config
    is_fp8 = cfg.precision == Precision.FP8
    # dtype 串与 vLLM 的 _get_config_dtype_str 一致(tools/tune_triton_moe.py 同源)
    dtype_str = "fp8_w8a8" if is_fp8 else None
    block_shape = cfg.block_shape if is_fp8 else None
    m = min(int(num_rows), _VLLM_CHUNK_SIZE)
    e_local, _, n_key = problem.w2.shape

    conf, how, first_exc = None, "", None
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            try_get_optimal_moe_config,
        )

        conf = try_get_optimal_moe_config(
            tuple(problem.w1.shape), tuple(problem.w2.shape),
            cfg.topk, dtype_str, m, block_shape=block_shape,
        )
        how = "查表/兜底"
    except Exception as exc:  # noqa: BLE001  换兜底那条路再试
        first_exc = exc
    if conf is None:
        try:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                get_default_config,
            )

            conf = get_default_config(m, int(e_local), int(n_key),
                                      cfg.hidden_size, cfg.topk, dtype_str,
                                      block_shape)
            how = "兜底(未查表)"
        except Exception as exc:  # noqa: BLE001  仅影响这行打印
            print(f"  [routing-stats] 取 vLLM triton config 失败 "
                  f"({first_exc} / {exc}); BLOCK 口径记 n/a")
            return None

    return BlockConfig(
        block_m=int(conf["BLOCK_SIZE_M"]),
        desc=f"vLLM triton fused_moe {how} @ M={num_rows}, "
             f"moe_align_block_size 粒度",
        detail=" ".join(f"{k}={v}" for k, v in conf.items()),
    )
