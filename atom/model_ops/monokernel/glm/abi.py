# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.2 TP4 full-layer Agentic shape contract."""

from __future__ import annotations

from dataclasses import dataclass

from atom.model_ops.monokernel.abi import AgenticDecodeShape


@dataclass(frozen=True)
class GlmAgenticShape:
    common: AgenticDecodeShape
    dcp_size: int
    query_replication: bool
    local_heads: int = 16
    expert_intermediate: int = 512

    def __post_init__(self) -> None:
        if self.dcp_size not in (1, 4):
            raise ValueError(f"GLM Agentic DCP must be 1 or 4, got {self.dcp_size}")
        if self.query_replication and self.dcp_size == 1:
            raise ValueError("query replication requires DCP")

    @property
    def query_heads(self) -> int:
        return 64 if self.query_replication else self.local_heads

    @classmethod
    def for_graph(
        cls,
        *,
        batch_capacity: int,
        query_len: int,
        dcp_size: int,
        query_replication: bool,
    ) -> "GlmAgenticShape":
        return cls(
            common=AgenticDecodeShape(
                running_bs=batch_capacity,
                query_len=query_len,
                batch_capacity=batch_capacity,
                row_capacity=batch_capacity * query_len,
            ),
            dcp_size=dcp_size,
            query_replication=query_replication,
        )


__all__ = ["GlmAgenticShape"]
