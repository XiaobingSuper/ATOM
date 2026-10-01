# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense physical FP8 cache contract for Kimi-K3 MLA MonoKernel layers."""

from __future__ import annotations

import torch

from atom.model_ops.monokernel.config import FP8_MAX

PHYSICAL_BLOCK_SIZE = 16
PHYSICAL_PAGE_SIZE = 1
MLA_CACHE_ROW = 576
MLA_VALUE_ROW = 512


def physical_cache_slot(block: int, offset: int) -> int:
    """Flatten one block-16/page-1 cache location to a physical slot."""

    if block < 0:
        raise ValueError(f"block must be non-negative, got {block}")
    if not 0 <= offset < PHYSICAL_BLOCK_SIZE:
        raise ValueError(
            f"offset must be in [0, {PHYSICAL_BLOCK_SIZE}), got {offset}"
        )
    return block * PHYSICAL_BLOCK_SIZE + offset


def logical_to_physical_slot(
    block_tables: torch.Tensor,
    *,
    batch_id: int,
    position: int,
) -> int:
    """Map one request-local logical token to its physical cache slot."""

    if (
        block_tables.ndim != 2
        or block_tables.dtype is not torch.int32
        or not block_tables.is_contiguous()
    ):
        raise ValueError("block_tables must be contiguous int32 [batch, blocks]")
    if not 0 <= batch_id < block_tables.shape[0]:
        raise ValueError(f"batch_id {batch_id} is outside the block table")
    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}")
    logical_block, offset = divmod(position, PHYSICAL_BLOCK_SIZE)
    if logical_block >= block_tables.shape[1]:
        raise ValueError(f"position {position} exceeds the block-table capacity")
    return physical_cache_slot(
        int(block_tables[batch_id, logical_block]),
        offset,
    )


def validate_fp8_mla_cache(
    main_cache: torch.Tensor,
    main_scale: torch.Tensor,
    *,
    index_cache: torch.Tensor | None = None,
) -> int:
    """Validate Kimi's index-free token-major cache and return slot count."""

    if index_cache is not None:
        raise TypeError("Kimi MLA has no sparse index cache")
    fnuz = getattr(torch, "float8_e4m3fnuz", None)
    if fnuz is not None and main_cache.dtype is fnuz:
        raise ValueError("E4M3FNUZ MLA caches are unsupported")
    if (
        main_cache.ndim != 3
        or main_cache.shape[1:] != (PHYSICAL_PAGE_SIZE, MLA_CACHE_ROW)
        or main_cache.dtype is not torch.float8_e4m3fn
        or not main_cache.is_contiguous()
    ):
        raise ValueError(
            "Kimi MLA cache must be contiguous token-major E4M3FN "
            f"[slots,1,{MLA_CACHE_ROW}], got "
            f"{tuple(main_cache.shape)} {main_cache.dtype}"
        )
    if (
        main_scale.dtype is not torch.float32
        or main_scale.numel() != 1
        or not main_scale.is_contiguous()
    ):
        raise ValueError("Kimi MLA cache descale must be one contiguous FP32 scalar")
    return main_cache.shape[0]


def publish_fp8_mla_rows(
    main_cache: torch.Tensor,
    main_scale: torch.Tensor,
    *,
    slot_mapping: torch.Tensor,
    batch_ids: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Reference FP8 physical scatter; padding and negative slots are untouched."""

    slots = validate_fp8_mla_cache(main_cache, main_scale)
    rows = values.shape[0]
    if (
        values.shape != (rows, MLA_CACHE_ROW)
        or values.dtype is not torch.bfloat16
        or not values.is_contiguous()
    ):
        raise ValueError(
            f"values must be contiguous BF16 [rows,{MLA_CACHE_ROW}]"
        )
    for name, value, dtype in (
        ("slot_mapping", slot_mapping, torch.int64),
        ("batch_ids", batch_ids, torch.int32),
    ):
        if (
            value.shape != (rows,)
            or value.dtype is not dtype
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous {dtype} [rows]")
    descale = main_scale.reshape(-1)[0]
    storage = main_cache.view(torch.uint8).view(slots, MLA_CACHE_ROW)
    for row in range(rows):
        slot = int(slot_mapping[row])
        if int(batch_ids[row]) < 0 or slot < 0:
            continue
        if slot >= slots:
            raise ValueError(f"physical slot {slot} exceeds cache capacity {slots}")
        storage[slot].copy_(
            (values[row].float() / descale)
            .clamp(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    return main_cache


def dense_fp8_paged_mla_reference(
    *,
    query: torch.Tensor,
    main_cache: torch.Tensor,
    main_scale: torch.Tensor,
    positions: torch.Tensor,
    batch_ids: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    fresh_slots: torch.Tensor | None = None,
    fresh_values: torch.Tensor | None = None,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference causal dense MLA over physical rows with fresh-value bypass."""

    slots = validate_fp8_mla_cache(main_cache, main_scale)
    if (
        query.ndim != 3
        or query.shape[-1] != MLA_CACHE_ROW
        or query.dtype is not torch.bfloat16
    ):
        raise ValueError(
            f"query must be BF16 [rows, heads, {MLA_CACHE_ROW}]"
        )
    rows, heads, _ = query.shape
    for name, value, dtype in (
        ("positions", positions, torch.int64),
        ("batch_ids", batch_ids, torch.int32),
    ):
        if (
            value.shape != (rows,)
            or value.dtype is not dtype
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous {dtype} [rows]")
    if (
        context_lens.ndim != 1
        or context_lens.dtype is not torch.int32
        or not context_lens.is_contiguous()
    ):
        raise ValueError("context_lens must be contiguous int32 [batch]")
    if block_tables.shape[0] != context_lens.numel():
        raise ValueError("block table and context rows must match")
    if (fresh_slots is None) != (fresh_values is None):
        raise ValueError("fresh slots and values must be provided together")
    if fresh_slots is not None:
        assert fresh_values is not None
        if (
            fresh_slots.ndim != 1
            or fresh_slots.dtype is not torch.int64
            or fresh_values.shape != (fresh_slots.numel(), MLA_CACHE_ROW)
            or fresh_values.dtype is not torch.bfloat16
        ):
            raise ValueError(
                f"fresh values must pair int64 slots with BF16 {MLA_CACHE_ROW}-wide rows"
            )

    output = torch.zeros(
        rows,
        heads,
        MLA_VALUE_ROW,
        dtype=torch.float32,
        device=query.device,
    )
    lse = torch.full(
        (rows, heads),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    dequantized = (
        main_cache.view(slots, MLA_CACHE_ROW).float()
        * main_scale.reshape(-1)[0]
    )
    for row in range(rows):
        batch_id = int(batch_ids[row])
        if batch_id < 0:
            continue
        position = int(positions[row])
        context = int(context_lens[batch_id])
        visible = min(context, position + 1)
        physical = [
            logical_to_physical_slot(
                block_tables,
                batch_id=batch_id,
                position=logical,
            )
            for logical in range(visible)
        ]
        if any(slot < 0 or slot >= slots for slot in physical):
            raise ValueError("block table maps outside the physical cache")
        keys = dequantized[physical].clone()
        if fresh_slots is not None:
            assert fresh_values is not None
            for key_row, slot in enumerate(physical):
                matches = (fresh_slots == slot).nonzero().flatten()
                if matches.numel():
                    keys[key_row].copy_(fresh_values[matches[-1]].float())
        scores = torch.einsum("hd,kd->hk", query[row].float(), keys)
        scores.mul_(softmax_scale)
        probabilities = torch.softmax(scores, dim=-1)
        output[row].copy_(
            torch.einsum(
                "hk,kv->hv",
                probabilities,
                keys[:, :MLA_VALUE_ROW],
            )
        )
        lse[row].copy_(torch.logsumexp(scores, dim=-1))
    return output, lse


__all__ = [
    "MLA_CACHE_ROW",
    "MLA_VALUE_ROW",
    "PHYSICAL_BLOCK_SIZE",
    "dense_fp8_paged_mla_reference",
    "logical_to_physical_slot",
    "physical_cache_slot",
    "publish_fp8_mla_rows",
    "validate_fp8_mla_cache",
]
