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


__all__ = ["KimiAgenticRuntime", "KimiAgenticShape"]
