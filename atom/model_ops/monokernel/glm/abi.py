# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.2 TP4 full-layer Agentic shape contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.model_ops.monokernel.abi import AgenticDecodeShape
from atom.model_ops.monokernel.config import LayerConfig, glm5_shard_config

GLM_AGENTIC_MAX_ROWS = 400
GLM_AGENTIC_TILE_ROWS = 8


@dataclass(frozen=True)
class GlmRowTile:
    index: int
    start: int
    stop: int
    active_rows: int

    @property
    def capacity(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class GlmAgenticRuntimeTile:
    row: GlmRowTile
    batch_ids: torch.Tensor
    owned_counts: torch.Tensor


@dataclass(frozen=True)
class GlmAgenticRuntime:
    """Graph-stable row ownership consumed directly by the device ABI."""

    shape: "GlmAgenticShape"
    batch_ids: torch.Tensor
    owned_counts: torch.Tensor

    @classmethod
    def bind(
        cls,
        shape: "GlmAgenticShape",
        batch_ids: torch.Tensor,
        owned_counts: torch.Tensor,
    ) -> "GlmAgenticRuntime":
        capacity = shape.common.row_capacity
        for name, value in (
            ("batch_ids", batch_ids),
            ("owned_counts", owned_counts),
        ):
            if (
                value.dtype is not torch.int32
                or not value.is_contiguous()
                or value.numel() < capacity
            ):
                raise ValueError(
                    f"{name} must be contiguous int32 with at least row capacity "
                    f"{capacity}, got {tuple(value.shape)} {value.dtype}"
                )
        if batch_ids.device != owned_counts.device:
            raise ValueError("batch_ids and owned_counts must use the same device")
        return cls(
            shape=shape,
            batch_ids=batch_ids[:capacity],
            owned_counts=owned_counts[:capacity],
        )

    def tile(self, row: GlmRowTile) -> GlmAgenticRuntimeTile:
        if row not in self.shape.row_tiles:
            raise ValueError("row tile does not belong to this GLM graph shape")
        return GlmAgenticRuntimeTile(
            row=row,
            batch_ids=self.batch_ids[row.start : row.stop],
            owned_counts=self.owned_counts[row.start : row.stop],
        )


@dataclass(frozen=True)
class GlmAgenticShape:
    common: AgenticDecodeShape
    dcp_size: int
    query_replication: bool
    tp_size: int = 4

    def __post_init__(self) -> None:
        if self.dcp_size not in (1, 4):
            raise ValueError(f"GLM Agentic DCP must be 1 or 4, got {self.dcp_size}")
        if self.query_replication and self.dcp_size == 1:
            raise ValueError("query replication requires DCP")
        glm5_shard_config(self.tp_size)
        if self.common.row_capacity > GLM_AGENTIC_MAX_ROWS:
            raise ValueError(
                f"GLM Agentic row capacity must be <= {GLM_AGENTIC_MAX_ROWS}, "
                f"got {self.common.row_capacity}"
            )
        if self.common.tile_rows != GLM_AGENTIC_TILE_ROWS:
            raise ValueError(
                f"GLM Agentic row tiles must have {GLM_AGENTIC_TILE_ROWS} rows"
            )

    @property
    def config(self) -> LayerConfig:
        return glm5_shard_config(self.tp_size)

    @property
    def local_heads(self) -> int:
        return self.config.local_heads

    @property
    def expert_intermediate(self) -> int:
        return self.config.inter

    @property
    def physical_experts(self) -> int:
        return self.config.n_experts + self.config.num_shared_experts

    @property
    def query_heads(self) -> int:
        return 64 if self.query_replication else self.local_heads

    @property
    def row_tiles(self) -> tuple[GlmRowTile, ...]:
        rows = self.common.row_capacity
        tile_rows = self.common.tile_rows
        active = self.common.actual_rows
        return tuple(
            GlmRowTile(
                index=start // tile_rows,
                start=start,
                stop=min(start + tile_rows, rows),
                active_rows=max(0, min(tile_rows, active - start)),
            )
            for start in range(0, rows, tile_rows)
        )

    @property
    def work_tiles(self) -> tuple[GlmRowTile, ...]:
        """Capacity tiles that contain runtime rows; device masks remain per-row."""

        return tuple(tile for tile in self.row_tiles if tile.active_rows)

    @classmethod
    def for_graph(
        cls,
        *,
        batch_capacity: int,
        query_len: int,
        dcp_size: int,
        query_replication: bool,
        tp_size: int = 4,
    ) -> "GlmAgenticShape":
        return cls(
            common=AgenticDecodeShape(
                running_bs=batch_capacity,
                query_len=query_len,
                batch_capacity=batch_capacity,
                row_capacity=batch_capacity * query_len,
                tile_rows=GLM_AGENTIC_TILE_ROWS,
            ),
            dcp_size=dcp_size,
            query_replication=query_replication,
            tp_size=tp_size,
        )


__all__ = [
    "GLM_AGENTIC_MAX_ROWS",
    "GLM_AGENTIC_TILE_ROWS",
    "GlmAgenticRuntime",
    "GlmAgenticRuntimeTile",
    "GlmAgenticShape",
    "GlmRowTile",
]
