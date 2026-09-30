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
        if self.common.query_len not in (1, 8):
            raise ValueError(
                f"Kimi TP-only query length must be 1 or 8, got {self.common.query_len}"
            )
        if self.state_dtype not in (torch.float16, torch.float32):
            raise ValueError("Kimi recurrent state must use FP16 or FP32")
        if self.replay_ssm:
            raise ValueError("ReplaySSM is used only by unsupported DCP bands")

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


__all__ = ["KimiAgenticShape"]
