# SPDX-License-Identifier: Apache-2.0

"""TP-only GLM IndexShare host plan and exact reference selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from atom.model_ops.monokernel.glm.cache import (
    logical_to_physical_slot,
    visible_context_length,
)


class GlmIndexShareMode(str, Enum):
    FULL = "full"
    SHARED = "shared"


@dataclass(frozen=True)
class GlmIndexSharePlan:
    modes: tuple[GlmIndexShareMode, ...]
    source_layers: tuple[int, ...]

    @classmethod
    def from_runtime_pattern(
        cls,
        pattern: tuple[str, ...] | list[str],
        *,
        allow_external_prefix: bool = False,
    ) -> "GlmIndexSharePlan":
        modes = tuple(
            GlmIndexShareMode.FULL if value.upper() == "F" else
            GlmIndexShareMode.SHARED if value.upper() == "S" else
            (_ for _ in ()).throw(ValueError(f"invalid IndexShare mode {value!r}"))
            for value in pattern
        )
        source = []
        latest_full = None
        for layer, mode in enumerate(modes):
            if mode is GlmIndexShareMode.FULL:
                latest_full = layer
            elif latest_full is None and not allow_external_prefix:
                raise ValueError("shared IndexShare layers cannot precede a full layer")
            source.append(-1 if latest_full is None else latest_full)
        return cls(modes, tuple(source))


def stable_physical_topk(
    scores: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_id: int,
    position: int,
    request_context: int,
    topk: int,
    owned_count: int = 1,
) -> tuple[torch.Tensor, int]:
    """Reference FP32 top-k with lower logical position as the exact tie break."""

    if scores.dtype is not torch.float32 or scores.ndim != 1:
        raise ValueError("scores must be one-dimensional FP32")
    if not 0 < topk <= 2048:
        raise ValueError("topk must be in [1, 2048]")
    output = torch.zeros(topk, dtype=torch.int32, device=scores.device)
    if batch_id < 0 or owned_count <= 0:
        return output, 0
    visible = visible_context_length(position, request_context)
    if scores.numel() < visible:
        raise ValueError("scores do not cover the row-visible context")
    logical = sorted(
        range(visible),
        key=lambda index: (-float(scores[index]), index),
    )[:topk]
    if logical:
        output[: len(logical)] = torch.tensor(
            [
                logical_to_physical_slot(
                    block_tables,
                    batch_id=batch_id,
                    position=index,
                )
                for index in logical
            ],
            dtype=torch.int32,
            device=scores.device,
        )
    return output, len(logical)


__all__ = [
    "GlmIndexShareMode",
    "GlmIndexSharePlan",
    "stable_physical_topk",
]
