# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 TP8 full-layer Agentic shape contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.model_ops.monokernel.abi import AgenticDecodeShape


@dataclass(frozen=True)
class KimiAgenticShape:
    common: AgenticDecodeShape
    dcp_size: int
    replay_ssm: bool
    state_dtype: torch.dtype = torch.float16
    local_heads: int = 12

    def __post_init__(self) -> None:
        if self.dcp_size != 1:
            raise ValueError("Kimi full MonoKernel supports TP-only DCP1")
        if self.common.query_len != 8:
            raise ValueError(
                f"Kimi full Agentic MonoKernel requires q=8, got {self.common.query_len}"
            )
        if self.state_dtype is not torch.float16:
            raise ValueError("Kimi full Agentic recurrent state must use FP16")
        if self.replay_ssm:
            raise ValueError("ReplaySSM is used only by unsupported DCP bands")

    @property
    def snapshot_shape(self) -> tuple[int, int]:
        return (self.common.batch_capacity, self.common.query_len)

    @property
    def accepted_shape(self) -> tuple[int]:
        return (self.common.batch_capacity,)

    @classmethod
    def for_graph(
        cls,
        *,
        batch_capacity: int,
        query_len: int,
        dcp_size: int,
        replay_ssm: bool,
        state_dtype: torch.dtype = torch.float16,
    ) -> "KimiAgenticShape":
        return cls(
            common=AgenticDecodeShape(
                running_bs=batch_capacity,
                query_len=query_len,
                batch_capacity=batch_capacity,
                row_capacity=batch_capacity * query_len,
            ),
            dcp_size=dcp_size,
            replay_ssm=replay_ssm,
            state_dtype=state_dtype,
        )


@dataclass(frozen=True)
class KimiAgenticRuntime:
    """Graph-stable q=8 snapshot metadata for one full-layer launch."""

    shape: KimiAgenticShape
    snapshot_slots: torch.Tensor
    num_accepted_tokens: torch.Tensor

    @classmethod
    def bind(
        cls,
        shape: KimiAgenticShape,
        snapshot_slots: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> "KimiAgenticRuntime":
        if (
            snapshot_slots.shape != shape.snapshot_shape
            or snapshot_slots.dtype is not torch.int32
            or not snapshot_slots.is_contiguous()
        ):
            raise ValueError(
                "snapshot_slots must be contiguous int32 "
                f"{list(shape.snapshot_shape)}"
            )
        if (
            num_accepted_tokens.shape != shape.accepted_shape
            or num_accepted_tokens.dtype is not torch.int32
            or not num_accepted_tokens.is_contiguous()
        ):
            raise ValueError(
                "num_accepted_tokens must be contiguous int32 "
                f"{list(shape.accepted_shape)}"
            )
        if snapshot_slots.device != num_accepted_tokens.device:
            raise ValueError("Kimi Agentic metadata must use one device")
        return cls(shape, snapshot_slots, num_accepted_tokens)

    def transition_slots(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the circular snapshot transitions for reference checks."""

        query_len = self.shape.common.query_len
        columns = torch.arange(
            query_len,
            dtype=torch.int64,
            device=self.snapshot_slots.device,
        ).expand(self.snapshot_slots.shape[0], query_len).clone()
        columns[:, 0] = self.num_accepted_tokens.to(torch.int64) - 1
        columns[:, 1:] -= 1
        return self.snapshot_slots.gather(1, columns), self.snapshot_slots

    def conv_window_plan(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source rows for q=8 convolution reads and the next window.

        Source rows 0..9 address the previous production cache window and
        rows 10..17 address this forward's eight input rows.
        """

        device = self.num_accepted_tokens.device
        accepted = self.num_accepted_tokens.to(torch.int64).view(-1, 1, 1)
        token = torch.arange(8, dtype=torch.int64, device=device).view(1, 8, 1)
        history = torch.arange(3, dtype=torch.int64, device=device).view(1, 1, 3)
        logical = accepted - 1 + token + history
        first_draft = accepted + 2
        reads = torch.where(
            logical < first_draft,
            logical,
            10 + logical - first_draft,
        )
        committed = accepted.view(-1, 1) + torch.arange(
            2,
            dtype=torch.int64,
            device=device,
        )
        draft = 10 + torch.arange(8, dtype=torch.int64, device=device)
        final = torch.cat((committed, draft.expand(committed.shape[0], 8)), dim=1)
        return reads, final


@dataclass(frozen=True)
class KimiMlaAgenticRuntime:
    """Graph-stable dense MLA metadata for one TP8 q=8 layer launch.

    ``batch_ids`` distinguishes live query rows from graph padding. Physical
    writes use ``slot_mapping`` while reads map each request-local logical
    position through ``block_tables``. Kimi has no sparse-index input.
    """

    shape: KimiAgenticShape
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    batch_ids: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    block_size: int
    block_ratio: int

    @classmethod
    def bind(
        cls,
        shape: KimiAgenticShape,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        batch_ids: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        *,
        block_size: int,
        block_ratio: int,
    ) -> "KimiMlaAgenticRuntime":
        batch = shape.common.batch_capacity
        rows = shape.common.row_capacity
        if batch not in (1, 2, 4):
            raise ValueError("Kimi MLA graph buckets support B1/B2/B4 q=8")
        for name, value, dtype in (
            ("positions", positions, torch.int64),
            ("slot_mapping", slot_mapping, torch.int64),
            ("batch_ids", batch_ids, torch.int32),
        ):
            if (
                value.shape != (rows,)
                or value.dtype is not dtype
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous {dtype} [{rows}]"
                )
        if (
            context_lens.shape != (batch,)
            or context_lens.dtype is not torch.int32
            or not context_lens.is_contiguous()
        ):
            raise ValueError(
                f"context_lens must be contiguous int32 [{batch}]"
            )
        if (
            block_tables.ndim != 2
            or block_tables.shape[0] != batch
            or block_tables.shape[1] <= 0
            or block_tables.dtype is not torch.int32
            or not block_tables.is_contiguous()
        ):
            raise ValueError(
                "block_tables must be contiguous int32 "
                f"[{batch}, physical_blocks]"
            )
        tensors = (
            positions,
            slot_mapping,
            batch_ids,
            context_lens,
            block_tables,
        )
        if any(value.device != positions.device for value in tensors[1:]):
            raise ValueError("Kimi MLA metadata must use one device")
        if block_size != 128 or block_ratio != 128:
            raise ValueError(
                "Kimi MLA MonoKernel requires scheduler block_size=128 "
                "and token-page block_ratio=128"
            )
        return cls(
            shape,
            positions,
            slot_mapping,
            batch_ids,
            context_lens,
            block_tables,
            block_size,
            block_ratio,
        )

    def cache_writers(self) -> torch.Tensor:
        """Rows that may scatter their newly projected latent to cache."""

        return (self.batch_ids >= 0) & (self.slot_mapping >= 0)

    def visible_physical_slots(self, row: int) -> list[int]:
        """Reference physical cache rows visible to one causal query."""

        if not 0 <= row < self.shape.common.row_capacity:
            raise ValueError(f"row {row} is outside the graph bucket")
        batch_id = int(self.batch_ids[row])
        if batch_id < 0:
            return []
        if batch_id >= self.shape.common.batch_capacity:
            raise ValueError(f"batch id {batch_id} is outside the graph bucket")
        position = int(self.positions[row])
        context = int(self.context_lens[batch_id])
        if position < 0 or context < 0:
            raise ValueError("positions and context lengths must be non-negative")
        visible = min(context, position + 1)
        from atom.model_ops.monokernel.k3.mla_cache import (
            logical_to_physical_slot,
        )

        return [
            logical_to_physical_slot(
                self.block_tables,
                batch_id=batch_id,
                position=logical,
                block_size=self.block_size,
                block_ratio=self.block_ratio,
            )
            for logical in range(visible)
        ]


__all__ = [
    "KimiAgenticRuntime",
    "KimiAgenticShape",
    "KimiMlaAgenticRuntime",
]
