# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared host arena for GLM Agentic full-layer row tiles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.model_ops.monokernel.config import MAX_LAYERS_PER_STEP
from atom.model_ops.monokernel.glm.abi import (
    GlmAgenticRuntime,
    GlmAgenticShape,
)
from atom.model_ops.monokernel.glm.layout import (
    GlmAgenticWorkspaceLayout,
    agentic_workspace_layout,
)
from atom.model_ops.monokernel.runtime import SymmetricPeerBuffer


@dataclass(frozen=True)
class GlmAgenticLayerBinding:
    workspace: "GlmAgenticWorkspace"
    slot: int


@dataclass
class GlmAgenticWorkspace:
    """One graph-bucket arena shared by all target decoder layers."""

    shape: GlmAgenticShape
    layout: GlmAgenticWorkspaceLayout
    scratch: torch.Tensor
    peer_buffer: object
    step: torch.Tensor
    hidden_buffers: tuple[torch.Tensor, torch.Tensor] | None = None
    runtime: GlmAgenticRuntime | None = None
    selected_slots: torch.Tensor | None = None
    selected_counts: torch.Tensor | None = None
    selected_indptr: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.layout.config != self.shape.config:
            raise ValueError("workspace layout does not match the GLM shape")
        if (
            self.scratch.dtype is not torch.uint8
            or not self.scratch.is_contiguous()
            or self.scratch.numel() < self.layout.scratch["_bytes"]
        ):
            raise ValueError("scratch storage does not cover the workspace layout")
        if (
            self.step.dtype is not torch.int32
            or self.step.numel() != 1
            or not self.step.is_contiguous()
        ):
            raise ValueError("step must be one contiguous int32 value")
        if self.hidden_buffers is not None:
            expected = (
                self.shape.common.row_capacity,
                self.shape.config.hidden,
            )
            for hidden in self.hidden_buffers:
                if (
                    hidden.shape != expected
                    or hidden.dtype is not torch.bfloat16
                    or not hidden.is_contiguous()
                    or hidden.device != self.scratch.device
                ):
                    raise ValueError(
                        f"hidden workspace must be contiguous BF16 {list(expected)} "
                        f"on {self.scratch.device}"
                    )

    @classmethod
    def allocate(
        cls,
        shape: GlmAgenticShape,
        *,
        rank: int,
        npes: int,
        group,
        sparse_attention_topk: int,
        with_indexer: bool = False,
        index_max_seq: int = 4096,
    ) -> "GlmAgenticWorkspace":
        layout = agentic_workspace_layout(
            shape,
            npes=npes,
            sparse_attention_topk=sparse_attention_topk,
            with_indexer=with_indexer,
            index_max_seq=index_max_seq,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        scratch = torch.zeros(
            layout.scratch["_bytes"],
            dtype=torch.uint8,
            device=device,
        )
        peers = SymmetricPeerBuffer(
            layout.symmetric["_bytes"],
            rank=rank,
            npes=npes,
            group=group,
        )
        step = torch.zeros(1, dtype=torch.int32, device=device)
        hidden_buffers = tuple(
            torch.empty(
                shape.common.row_capacity,
                shape.config.hidden,
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(2)
        )
        return cls(
            shape,
            layout,
            scratch,
            peers,
            step,
            hidden_buffers=hidden_buffers,
        )

    def ensure_index_share(self, topk: int) -> None:
        """Allocate graph-stable full→shared selection storage once."""

        if topk <= 0:
            raise ValueError("IndexShare top-k must be positive")
        rows = self.shape.common.row_capacity
        expected = ((rows, topk), (rows,), (rows + 1,))
        buffers = (self.selected_slots, self.selected_counts, self.selected_indptr)
        if any(buffer is not None for buffer in buffers):
            if (
                any(buffer is None for buffer in buffers)
                or tuple(
                    tuple(buffer.shape) for buffer in buffers if buffer is not None
                ) != expected
                or any(
                    buffer.dtype is not torch.int32
                    or not buffer.is_contiguous()
                    or buffer.device != self.scratch.device
                    for buffer in buffers
                    if buffer is not None
                )
            ):
                raise ValueError("existing IndexShare storage has incompatible geometry")
            return
        self.selected_slots = torch.empty(
            expected[0], dtype=torch.int32, device=self.scratch.device
        )
        self.selected_counts = torch.empty(
            expected[1], dtype=torch.int32, device=self.scratch.device
        )
        self.selected_indptr = torch.empty(
            expected[2], dtype=torch.int32, device=self.scratch.device
        )

    def bind_runtime(
        self,
        batch_ids: torch.Tensor,
        owned_counts: torch.Tensor,
    ) -> GlmAgenticRuntime:
        runtime = GlmAgenticRuntime.bind(self.shape, batch_ids, owned_counts)
        if runtime.batch_ids.device != self.scratch.device:
            raise ValueError("row ownership metadata must use the workspace device")
        self.runtime = runtime
        return runtime

    def layer(self, slot: int) -> GlmAgenticLayerBinding:
        if not 0 <= slot < MAX_LAYERS_PER_STEP:
            raise ValueError(
                f"layer slot must be in [0, {MAX_LAYERS_PER_STEP}), got {slot}"
            )
        return GlmAgenticLayerBinding(self, slot)

    def advance_step(self) -> None:
        self.step.add_(1)

    def close(self) -> None:
        close = getattr(self.peer_buffer, "close", None)
        if close is not None:
            close()


__all__ = ["GlmAgenticLayerBinding", "GlmAgenticWorkspace"]
