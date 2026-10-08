# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ragged index-score work assigned from per-row request ownership."""

from dataclasses import dataclass

from atom.mono.plan.execution import BLOCKS

KEYS_PER_CHUNK = 64


@dataclass(frozen=True)
class WalkTask:
    lead: int
    span: int
    chunk_first: int
    chunk_last: int


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def ragged_walk(
    owners: list[int] | tuple[int, ...],
    visible: list[int] | tuple[int, ...],
    *,
    ctas: int = BLOCKS,
) -> tuple[WalkTask, ...]:
    """Plan score chunks for contiguous rows belonging to the same request."""

    if len(owners) != len(visible):
        raise ValueError("owners and visible must have the same row count")
    if ctas <= 0:
        raise ValueError("ctas must be positive")

    groups: list[tuple[int, int, int]] = []
    row = 0
    while row < len(owners):
        owner = owners[row]
        end = row + 1
        while end < len(owners) and owners[end] == owner:
            end += 1
        if owner >= 0:
            chunks = max(
                (
                    _ceil_div(max(visible[i], 0), KEYS_PER_CHUNK)
                    for i in range(row, end)
                ),
                default=0,
            )
            if chunks:
                groups.append((row, end - row, chunks))
        row = end

    if not groups:
        return ()
    width = max(1, _ceil_div(sum(chunks for _, _, chunks in groups), ctas))
    while sum(_ceil_div(chunks, width) for _, _, chunks in groups) > ctas:
        width += 1

    return tuple(
        WalkTask(lead, span, first, min(first + width, chunks))
        for lead, span, chunks in groups
        for first in range(0, chunks, width)
    )
