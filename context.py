# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank distributed context for the full-layer MoE schemes.

A :class:`DistContext` bundles the parallel handles a distributed scheme needs
(rank, world size, device, process group) so schemes take a single argument
instead of threading these through every call. Kept deliberately small — a
scheme that needs more (extra streams, an NVSHMEM handle) allocates it in its
own ``setup`` from ``ctx.device`` / ``ctx.group``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistContext:
    """Handles for one rank of a distributed MoE-layer benchmark."""

    rank: int
    world_size: int
    local_rank: int
    device: Any
    # Process group the scheme's collectives run over. ``None`` means the
    # default (world) group, which is what the harness sets up.
    group: Any | None = None

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0
