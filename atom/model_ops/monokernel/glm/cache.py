# SPDX-License-Identifier: Apache-2.0

"""Physical FP8 paged-cache contract for the GLM full-layer kernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.model_ops.monokernel.config import FP8_MAX

PHYSICAL_BLOCK_SIZE = 16
PHYSICAL_PAGE_SIZE = 1
MAIN_CACHE_ROW_BYTES = 576
INDEX_KEY_BYTES = 128
INDEX_CACHE_ROW_BYTES = 144
INDEX_SCALE_OFFSET = INDEX_KEY_BYTES
INDEX_CACHE_BLOCK_TOKENS = 16
INDEX_CACHE_BLOCK_BYTES = INDEX_CACHE_BLOCK_TOKENS * INDEX_CACHE_ROW_BYTES
INDEX_CACHE_KEYS_BYTES = INDEX_CACHE_BLOCK_TOKENS * INDEX_KEY_BYTES


@dataclass(frozen=True)
class Fp8CacheRowPolicy:
    query_active: bool
    cache_writer: bool


@dataclass(frozen=True)
class Fp8CacheWorkPolicy:
    index_score: bool
    sparse_split: bool
    empty_output: tuple[float, float]


def physical_cache_slot(block: int, offset: int) -> int:
    """Flatten one block-16/page-1 cache location to its physical slot."""

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
    """Map one request-local logical token position to a physical cache slot."""

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


def visible_context_length(position: int, request_context: int) -> int:
    """Causal key count visible to one flattened MTP verification row."""

    if position < 0 or request_context < 0:
        raise ValueError("position and request_context must be non-negative")
    return min(request_context, position + 1)


def visible_physical_slots(
    block_tables: torch.Tensor,
    *,
    batch_id: int,
    position: int,
    request_context: int,
) -> list[int]:
    """Reference physical cache rows visible to one causal query row."""

    return [
        logical_to_physical_slot(
            block_tables,
            batch_id=batch_id,
            position=logical,
        )
        for logical in range(visible_context_length(position, request_context))
    ]


def fp8_cache_row_policy(*, batch_id: int, slot: int) -> Fp8CacheRowPolicy:
    active = batch_id >= 0
    return Fp8CacheRowPolicy(
        query_active=active,
        cache_writer=active and slot >= 0,
    )


def fp8_cache_work_policy(
    *,
    batch_id: int,
    owned_count: int,
) -> Fp8CacheWorkPolicy:
    active = batch_id >= 0 and owned_count > 0
    return Fp8CacheWorkPolicy(
        index_score=active,
        sparse_split=active,
        empty_output=(0.0, float("-inf")),
    )


def validate_fp8_paged_cache(
    main_cache: torch.Tensor,
    main_scale: torch.Tensor,
    index_cache: torch.Tensor | None,
    *,
    with_indexer: bool,
) -> int:
    """Validate production physical storage and return flattened slot count."""

    fnuz = getattr(torch, "float8_e4m3fnuz", None)
    if fnuz is not None and main_cache.dtype is fnuz:
        raise ValueError("E4M3FNUZ main caches are unsupported on the MI355 path")
    if (
        main_cache.ndim != 3
        or main_cache.shape[1:] != (PHYSICAL_PAGE_SIZE, MAIN_CACHE_ROW_BYTES)
        or not main_cache.is_contiguous()
        or main_cache.dtype is not torch.float8_e4m3fn
    ):
        raise ValueError(
            "main FP8 cache must be contiguous token-major E4M3FN [slots,1,576], "
            f"got {tuple(main_cache.shape)} {main_cache.dtype}"
        )
    slots = main_cache.shape[0]
    if (
        main_scale.dtype is not torch.float32
        or not main_scale.is_contiguous()
        or main_scale.numel() != 1
    ):
        raise ValueError("main cache descale must be one contiguous FP32 scalar")
    if with_indexer:
        if index_cache is None:
            raise ValueError("FP8 indexed cache requires index_cache")
        _validate_index_cache(index_cache, slots)
    return slots


def _validate_index_cache(index_cache: torch.Tensor, slots: int) -> None:
    if (
        index_cache.ndim != 3
        or index_cache.shape[1:]
        != (INDEX_CACHE_BLOCK_TOKENS, INDEX_CACHE_ROW_BYTES)
        or not index_cache.is_contiguous()
        or index_cache.dtype is not torch.float8_e4m3fn
    ):
        raise ValueError(
            "index FP8 cache must be contiguous block-major E4M3FN "
            f"[blocks,16,144], got {tuple(index_cache.shape)} {index_cache.dtype}"
        )
    if index_cache.shape[0] * INDEX_CACHE_BLOCK_TOKENS != slots:
        raise ValueError("main and index FP8 caches must have equal slot capacity")


def index_cache_key_byte_offset(slot: int, dim: int) -> int:
    """Address one preshuffled index-key byte in block16 production storage."""

    if slot < 0:
        raise ValueError(f"physical slot must be non-negative, got {slot}")
    if not 0 <= dim < INDEX_KEY_BYTES:
        raise ValueError(f"index dimension must be in [0,128), got {dim}")
    block, token = divmod(slot, INDEX_CACHE_BLOCK_TOKENS)
    return (
        block * INDEX_CACHE_BLOCK_BYTES
        + (dim // 16) * 256
        + token * 16
        + dim % 16
    )


def index_cache_scale_byte_offset(slot: int) -> int:
    """Address one embedded FP32 row scale in block16 production storage."""

    if slot < 0:
        raise ValueError(f"physical slot must be non-negative, got {slot}")
    block, token = divmod(slot, INDEX_CACHE_BLOCK_TOKENS)
    return block * INDEX_CACHE_BLOCK_BYTES + INDEX_CACHE_KEYS_BYTES + token * 4


def read_fp8_index_cache_row(
    index_cache: torch.Tensor,
    *,
    slot: int,
) -> torch.Tensor:
    """Pure reference dequantization of one preshuffled physical index row."""

    blocks = index_cache.shape[0] if index_cache.ndim else 0
    _validate_index_cache(index_cache, blocks * INDEX_CACHE_BLOCK_TOKENS)
    if slot >= blocks * INDEX_CACHE_BLOCK_TOKENS:
        raise ValueError("physical slot exceeds index-cache capacity")
    storage = index_cache.view(torch.uint8).reshape(-1)
    key = torch.stack(
        [storage[index_cache_key_byte_offset(slot, dim)] for dim in range(128)]
    ).view(torch.float8_e4m3fn)
    scale_at = index_cache_scale_byte_offset(slot)
    scale = storage[scale_at : scale_at + 4].clone().view(torch.float32)[0]
    return key.float() * scale


def _quantize_row(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.float()
    amax = values.abs().max()
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    quantized = (
        (values / scale)
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return quantized, scale


def publish_fp8_cache_rows(
    main_cache: torch.Tensor,
    main_scale: torch.Tensor,
    index_cache: torch.Tensor | None,
    *,
    slot: int,
    batch_id: int,
    main_bf16: torch.Tensor,
    index_bf16: torch.Tensor | None = None,
) -> bool:
    """Pure reference publication; BF16 inputs remain independent of storage."""

    policy = fp8_cache_row_policy(batch_id=batch_id, slot=slot)
    if not policy.cache_writer:
        return False
    slots = validate_fp8_paged_cache(
        main_cache,
        main_scale,
        index_cache,
        with_indexer=index_bf16 is not None,
    )
    if slot >= slots:
        raise ValueError(f"physical slot {slot} exceeds cache capacity {slots}")
    if main_bf16.numel() != MAIN_CACHE_ROW_BYTES:
        raise ValueError("main BF16 source must contain 576 values")

    descale = main_scale.reshape(-1)[0]
    main_q = (
        (main_bf16.reshape(-1).float() / descale)
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    main_cache.view(torch.uint8).view(slots, MAIN_CACHE_ROW_BYTES)[slot].copy_(
        main_q
    )
    if index_bf16 is not None:
        assert index_cache is not None
        if index_bf16.numel() != INDEX_KEY_BYTES:
            raise ValueError("index BF16 source must contain 128 values")
        index_q, index_s = _quantize_row(index_bf16.reshape(-1))
        storage = index_cache.view(torch.uint8).reshape(-1)
        for dim in range(INDEX_KEY_BYTES):
            storage[index_cache_key_byte_offset(slot, dim)] = index_q[dim]
        scale_at = index_cache_scale_byte_offset(slot)
        storage[scale_at : scale_at + 4].copy_(index_s.reshape(1).view(torch.uint8))
    return True


def fp8_paged_sparse_attention_reference(
    *,
    query: torch.Tensor,
    main_cache: torch.Tensor,
    main_scale: torch.Tensor,
    selected_slots: torch.Tensor,
    selected_counts: torch.Tensor,
    selected_indptr: torch.Tensor,
    batch_ids: torch.Tensor,
    owned_counts: torch.Tensor,
    fresh_slots: torch.Tensor | None = None,
    fresh_values: torch.Tensor | None = None,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference physical-slot attention shared by FULL and SHARED layers."""

    slots = validate_fp8_paged_cache(
        main_cache,
        main_scale,
        None,
        with_indexer=False,
    )
    if query.ndim != 3 or query.shape[-1] != MAIN_CACHE_ROW_BYTES:
        raise ValueError("query must be [rows, heads, 576]")
    rows, heads, _ = query.shape
    for name, value, size in (
        ("selected_counts", selected_counts, rows),
        ("selected_indptr", selected_indptr, rows + 1),
        ("batch_ids", batch_ids, rows),
        ("owned_counts", owned_counts, rows),
    ):
        if value.dtype is not torch.int32 or value.numel() < size:
            raise ValueError(f"{name} must contain {size} int32 values")
    if selected_slots.dtype is not torch.int32:
        raise ValueError("selected_slots must be int32")
    if (fresh_slots is None) != (fresh_values is None):
        raise ValueError("fresh slots and values must be provided together")
    fresh: dict[int, torch.Tensor] = {}
    if fresh_slots is not None and fresh_values is not None:
        if (
            fresh_slots.dtype is not torch.int64
            or fresh_values.ndim != 2
            or fresh_values.shape != (fresh_slots.numel(), MAIN_CACHE_ROW_BYTES)
        ):
            raise ValueError("fresh values must be [fresh_slots, 576]")
        fresh = {
            int(slot): fresh_values[i].float()
            for i, slot in enumerate(fresh_slots.tolist())
            if slot >= 0
        }

    output = torch.zeros(
        rows,
        heads,
        512,
        dtype=torch.float32,
        device=query.device,
    )
    lse = torch.full(
        (rows, heads),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    cache = main_cache.reshape(slots, MAIN_CACHE_ROW_BYTES)
    descale = main_scale.reshape(-1)[0]
    for row in range(rows):
        if int(batch_ids[row]) < 0 or int(owned_counts[row]) <= 0:
            continue
        begin = int(selected_indptr[row])
        end = int(selected_indptr[row + 1])
        count = min(max(end - begin, 0), max(int(selected_counts[row]), 0))
        if count == 0:
            continue
        physical = selected_slots[begin : begin + count].tolist()
        if any(not 0 <= slot < slots for slot in physical):
            raise ValueError("selected physical slot exceeds cache capacity")
        rows_kv = torch.stack(
            [
                fresh[slot]
                if slot in fresh
                else cache[slot].float() * descale
                for slot in physical
            ]
        ).to(query.device)
        scores = (
            query[row].float() @ rows_kv.transpose(0, 1)
        ) * softmax_scale
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = probabilities @ rows_kv[:, :512]
        lse[row] = torch.logsumexp(scores, dim=-1)
    return output, lse


__all__ = [
    "Fp8CacheRowPolicy",
    "Fp8CacheWorkPolicy",
    "INDEX_CACHE_ROW_BYTES",
    "INDEX_CACHE_BLOCK_BYTES",
    "INDEX_CACHE_BLOCK_TOKENS",
    "INDEX_CACHE_KEYS_BYTES",
    "INDEX_KEY_BYTES",
    "INDEX_SCALE_OFFSET",
    "MAIN_CACHE_ROW_BYTES",
    "PHYSICAL_BLOCK_SIZE",
    "PHYSICAL_PAGE_SIZE",
    "fp8_cache_row_policy",
    "fp8_cache_work_policy",
    "fp8_paged_sparse_attention_reference",
    "logical_to_physical_slot",
    "index_cache_key_byte_offset",
    "index_cache_scale_byte_offset",
    "physical_cache_slot",
    "publish_fp8_cache_rows",
    "read_fp8_index_cache_row",
    "validate_fp8_paged_cache",
    "visible_context_length",
    "visible_physical_slots",
]
