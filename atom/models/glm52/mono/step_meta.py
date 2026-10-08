# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-row metadata staged once for a ragged speculative decode step."""

import triton
import triton.language as tl

META_WORDS = 6
OWNER, POSITION, SLOT, KEY_COUNT, INDEX_BASE, LIVE = range(META_WORDS)


def row_meta(
    owner: int, position: int, slot: int, row: int, topk: int
) -> tuple[int, ...]:
    """Reference metadata for one row."""

    live = owner >= 0 and slot >= 0
    count = min(max(position + 1, 0), topk) if live else 0
    return owner, position, slot, count, row * topk, int(live)


@triton.jit
def _write_step_meta(
    batch_ids,
    positions,
    slot_mapping,
    out,
    topk,
    rows,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    mask = row < rows
    owner = tl.load(batch_ids + row, mask=mask, other=-1).to(tl.int32)
    position = tl.load(positions + row, mask=mask, other=0).to(tl.int64)
    slot = tl.load(slot_mapping + row, mask=mask, other=-1).to(tl.int64)
    live = (owner >= 0) & (slot >= 0)
    count = tl.where(live, tl.minimum(tl.maximum(position + 1, 0), topk), 0)
    base = row * topk
    tl.store(out + row * WORDS + OWNER, owner, mask=mask)
    tl.store(out + row * WORDS + POSITION, position.to(tl.int32), mask=mask)
    tl.store(out + row * WORDS + SLOT, slot.to(tl.int32), mask=mask)
    tl.store(out + row * WORDS + KEY_COUNT, count.to(tl.int32), mask=mask)
    tl.store(out + row * WORDS + INDEX_BASE, base.to(tl.int32), mask=mask)
    tl.store(out + row * WORDS + LIVE, live.to(tl.int32), mask=mask)


def write_step_meta(batch_ids, positions, slot_mapping, out, topk: int) -> None:
    """Write contiguous int32 ``[rows, META_WORDS]`` metadata."""

    rows = positions.numel()
    assert batch_ids.numel() >= rows and slot_mapping.numel() >= rows
    assert out.shape == (rows, META_WORDS) and out.is_contiguous()
    _write_step_meta[(1,)](
        batch_ids,
        positions,
        slot_mapping,
        out,
        topk,
        rows,
        WORDS=META_WORDS,
        BLOCK=triton.next_power_of_2(rows),
    )
