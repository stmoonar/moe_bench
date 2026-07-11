# SPDX-License-Identifier: Apache-2.0
"""按 AGENTS.md 的优先级选一组空闲 4 卡,stdout 打印如 "9,11,13,15"。

空闲判定:GPU 利用率 < 5% 且显存占用 < 1GB。没有可用组时 exit 1。
"""
from __future__ import annotations

import subprocess
import sys

# 同一 PCIe Switch 下的连续两卡不放进同一组(AGENTS.md),优先级从高到低
GROUPS = [[9, 11, 13, 15], [8, 10, 12, 14], [1, 3, 5, 7], [0, 2, 4, 6]]


def gpu_stats() -> dict[int, tuple[int, int]]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    stats = {}
    for line in out.strip().splitlines():
        idx, util, mem = [x.strip() for x in line.split(",")]
        stats[int(idx)] = (int(util), int(mem))
    return stats


def main() -> None:
    stats = gpu_stats()
    for group in GROUPS:
        ok = all(i in stats and stats[i][0] < 5 and stats[i][1] < 1024
                 for i in group)
        if ok:
            print(",".join(map(str, group)))
            return
    print("[pick_gpus] 没有完全空闲的 4 卡组:", file=sys.stderr)
    for i, (util, mem) in sorted(stats.items()):
        print(f"  gpu{i}: util={util}% mem={mem}MB", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
