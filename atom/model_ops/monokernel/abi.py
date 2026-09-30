# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared graph-stable contracts for Agentic full-layer MonoKernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgenticDecodeShape:
    """Separate real requests, query rows, and graph capacities."""

    running_bs: int
    query_len: int
    batch_capacity: int
    row_capacity: int
    tile_rows: int = 8

    def __post_init__(self) -> None:
        if self.running_bs <= 0 or self.query_len <= 0 or self.tile_rows <= 0:
            raise ValueError("Agentic decode dimensions must be positive")
        if self.batch_capacity < self.running_bs:
            raise ValueError("batch capacity is smaller than the running batch")
        if self.row_capacity < self.batch_capacity * self.query_len:
            raise ValueError("row capacity is smaller than batch_capacity * query_len")

    @property
    def actual_rows(self) -> int:
        return self.running_bs * self.query_len

    @property
    def tiles(self) -> int:
        return (self.row_capacity + self.tile_rows - 1) // self.tile_rows

    @property
    def tail_rows(self) -> int:
        return self.row_capacity - (self.tiles - 1) * self.tile_rows

    def is_active_row(self, row: int) -> bool:
        return 0 <= row < self.actual_rows

    def row_to_request(self, row: int) -> int:
        if not self.is_active_row(row):
            raise ValueError(f"inactive row {row}")
        return row // self.query_len

    def row_to_query(self, row: int) -> int:
        if not self.is_active_row(row):
            raise ValueError(f"inactive row {row}")
        return row % self.query_len


@dataclass(frozen=True)
class DecodeRowContract:
    """Independent query, cache-write, and local-attention predicates."""

    query_active: bool
    cache_writer: bool
    local_sparse_active: bool
    safe_sparse_row: int


def classify_decode_row(
    *,
    batch_id: int,
    context_len: int,
    slot: int,
    sparse_begin: int,
    sparse_end: int,
    owned_count: int,
) -> DecodeRowContract:
    """Reference the device-side row predicates used by both model kernels."""

    if sparse_begin < 0 or sparse_end < sparse_begin or owned_count < 0:
        raise ValueError("invalid sparse row metadata")
    query_active = batch_id >= 0 and context_len > 0
    return DecodeRowContract(
        query_active=query_active,
        cache_writer=query_active and slot >= 0,
        local_sparse_active=(
            query_active and owned_count > 0 and sparse_end > sparse_begin
        ),
        safe_sparse_row=sparse_begin if sparse_end > sparse_begin else 0,
    )


__all__ = [
    "AgenticDecodeShape",
    "DecodeRowContract",
    "classify_decode_row",
]
