# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Executable graph-bucket owner for tiled GLM Agentic layer kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from atom.model_ops.monokernel.config import (
    MAX_LAYERS_PER_STEP,
    AttentionWeight,
    KvCacheLayout,
)
from atom.model_ops.monokernel.glm.cache import validate_fp8_paged_cache
from atom.model_ops.monokernel.glm.index_share import (
    GlmIndexShareMode,
    GlmIndexSharePlan,
)
from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace


@dataclass(frozen=True)
class GlmAgenticLayerInputs:
    kv_cache: torch.Tensor
    pe_cache: torch.Tensor | None
    indices: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    index_cache: torch.Tensor | None = None
    kv_cache_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class GlmAgenticLayerSpec:
    weights: Any
    inputs: GlmAgenticLayerInputs
    with_indexer: bool = False
    index_max_seq: int = 4096
    attention_weight: AttentionWeight = AttentionWeight.FP8_BLOCK128
    kv_cache_layout: KvCacheLayout = KvCacheLayout.ATOM_FP8
    uv_scale_block_m: int = 128
    index_share_mode: GlmIndexShareMode | None = None

    def validate_cache_inputs(self) -> int | None:
        if self.index_share_mode is GlmIndexShareMode.FULL and not self.with_indexer:
            raise ValueError("full IndexShare layers require indexer weights")
        if self.index_share_mode is GlmIndexShareMode.SHARED and self.with_indexer:
            raise ValueError("shared IndexShare layers must skip the indexer")
        if self.kv_cache_layout is not KvCacheLayout.ATOM_FP8:
            return None
        scale = self.inputs.kv_cache_scale
        if scale is None:
            raise ValueError("ATOM FP8 graph layer requires one FP32 scalar descale")
        return validate_fp8_paged_cache(
            self.inputs.kv_cache,
            scale,
            self.inputs.index_cache,
            with_indexer=self.with_indexer,
        )


@dataclass(frozen=True)
class GlmAgenticLayer:
    slot: int
    kernels: dict[int, Any]
    inputs: GlmAgenticLayerInputs
    packed_artifacts: Any
    index_share_mode: GlmIndexShareMode | None = None

    @property
    def workspace(self) -> GlmAgenticWorkspace:
        return next(iter(self.kernels.values())).workspace


class GlmAgenticGraphBucket:
    """Tile-major execution over full-layer kernels sharing one graph arena."""

    def __init__(
        self,
        workspace: GlmAgenticWorkspace,
        layers: tuple[GlmAgenticLayer, ...],
    ) -> None:
        if not layers:
            raise ValueError("a GLM graph bucket requires at least one layer")
        slots = [layer.slot for layer in layers]
        if len(set(slots)) != len(slots):
            raise ValueError("GLM graph-bucket layer slots must be distinct")
        if any(not 0 <= slot < MAX_LAYERS_PER_STEP for slot in slots):
            raise ValueError("GLM graph-bucket layer slot is out of range")
        tile_sizes = {tile.capacity for tile in workspace.shape.row_tiles}
        for layer in layers:
            if set(layer.kernels) != tile_sizes:
                raise ValueError(
                    f"layer {layer.slot} kernels must cover tile sizes "
                    f"{sorted(tile_sizes)}"
                )
            for size, kernel in layer.kernels.items():
                if (
                    kernel.S != size
                    or kernel.workspace is not workspace
                    or kernel.scratch is not workspace.scratch
                    or kernel.peer_buffer is not workspace.peer_buffer
                    or kernel.step is not workspace.step
                ):
                    raise ValueError(
                        f"layer {layer.slot} tile {size} does not share the bucket arena"
                    )
                if kernel.packed_artifacts is not layer.packed_artifacts:
                    raise ValueError(
                        f"layer {layer.slot} tile {size} does not share packed artifacts"
                    )
        if workspace.hidden_buffers is None:
            raise ValueError("graph bucket requires shared hidden buffers")
        self.workspace = workspace
        self.layers = layers

    @classmethod
    def build(
        cls,
        workspace: GlmAgenticWorkspace,
        specs: tuple[GlmAgenticLayerSpec, ...],
        *,
        rank: int,
        npes: int,
        group,
        topk: int,
        kernel_factory: Callable[..., Any] | None = None,
        artifact_factory: Callable[..., Any] | None = None,
    ) -> "GlmAgenticGraphBucket":
        if len(specs) > MAX_LAYERS_PER_STEP:
            raise ValueError("too many layers for the launch-tag ABI")
        explicit_modes = tuple(
            spec.index_share_mode for spec in specs
            if spec.index_share_mode is not None
        )
        if explicit_modes:
            if len(explicit_modes) != len(specs):
                raise ValueError("IndexShare mode must be explicit for every graph layer")
            GlmIndexSharePlan.from_runtime_pattern(
                tuple(
                    "F" if mode is GlmIndexShareMode.FULL else "S"
                    for mode in explicit_modes
                )
            )
            workspace.ensure_index_share(topk)
        if kernel_factory is None:
            from atom.model_ops.monokernel.glm.op import (
                Glm5MonoKernel,
                Glm5PackedArtifacts,
            )

            kernel_factory = Glm5MonoKernel
            artifact_factory = artifact_factory or Glm5PackedArtifacts.pack
        if artifact_factory is None:
            from atom.model_ops.monokernel.glm.op import Glm5PackedArtifacts

            artifact_factory = Glm5PackedArtifacts.pack
        tile_sizes = sorted(
            {tile.capacity for tile in workspace.shape.row_tiles}
        )
        layers = []
        try:
            for slot, spec in enumerate(specs):
                spec.validate_cache_inputs()
                artifacts = artifact_factory(
                    spec.weights,
                    npes=npes,
                    attention_weight=spec.attention_weight,
                    with_indexer=spec.with_indexer,
                )
                kernels = {}
                try:
                    for size in tile_sizes:
                        kernels[size] = kernel_factory(
                            spec.weights,
                            size,
                            rank=rank,
                            npes=npes,
                            group=group,
                            topk=topk,
                            launches_per_step=MAX_LAYERS_PER_STEP,
                            with_indexer=spec.with_indexer,
                            index_share=spec.index_share_mode is not None,
                            index_max_seq=spec.index_max_seq,
                            attention_weight=spec.attention_weight,
                            kv_cache_layout=spec.kv_cache_layout,
                            uv_scale_block_m=spec.uv_scale_block_m,
                            agentic_row_contract=True,
                            workspace=workspace,
                            packed_artifacts=artifacts,
                        )
                except Exception:
                    for kernel in kernels.values():
                        kernel.close()
                    raise
                layers.append(
                    GlmAgenticLayer(
                        slot=slot,
                        kernels=kernels,
                        inputs=spec.inputs,
                        packed_artifacts=artifacts,
                        index_share_mode=spec.index_share_mode,
                    )
                )
        except Exception:
            for layer in layers:
                for kernel in layer.kernels.values():
                    kernel.close()
            raise
        return cls(workspace, tuple(layers))

    def __call__(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        sparse_kv_indptr: torch.Tensor,
        batch_ids: torch.Tensor,
        owned_counts: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        shape = self.workspace.shape
        rows = shape.common.row_capacity
        hidden_shape = (rows, shape.config.hidden)
        if (
            hidden_states.shape != hidden_shape
            or hidden_states.dtype is not torch.bfloat16
            or not hidden_states.is_contiguous()
        ):
            raise ValueError(
                f"hidden states must be contiguous BF16 {list(hidden_shape)}"
            )
        for name, value, dtype, size in (
            ("positions", positions, torch.int64, rows),
            ("slot_mapping", slot_mapping, torch.int64, rows),
            ("sparse_kv_indptr", sparse_kv_indptr, torch.int32, rows + 1),
        ):
            if (
                value.dtype is not dtype
                or value.numel() < size
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous {dtype} with {size} values"
                )
        batch_capacity = shape.common.batch_capacity
        if (
            block_tables.ndim != 2
            or block_tables.dtype is not torch.int32
            or not block_tables.is_contiguous()
            or block_tables.shape[0] < batch_capacity
        ):
            raise ValueError(
                "block_tables must be contiguous int32 [batch_capacity, blocks]"
            )
        if (
            context_lens.dtype is not torch.int32
            or context_lens.ndim != 1
            or context_lens.numel() < batch_capacity
            or not context_lens.is_contiguous()
        ):
            raise ValueError(
                "context_lens must be contiguous int32 [batch_capacity]"
            )
        runtime = self.workspace.bind_runtime(batch_ids, owned_counts)
        buffers = self.workspace.hidden_buffers
        assert buffers is not None
        for tile in shape.work_tiles:
            start, stop = tile.start, tile.stop
            state = hidden_states[start:stop]
            ownership = runtime.tile(tile)
            selected_slots = self.workspace.selected_slots
            selected_counts = self.workspace.selected_counts
            selected_indptr = self.workspace.selected_indptr
            for layer_index, layer in enumerate(self.layers):
                output = buffers[layer_index % 2][start:stop]
                args = layer.inputs
                index_share = layer.index_share_mode is not None
                layer_indices = (
                    selected_slots[start:stop].reshape(-1)
                    if index_share and selected_slots is not None
                    else args.indices
                )
                layer_indptr = (
                    selected_indptr[start : stop + 1]
                    if index_share and selected_indptr is not None
                    else sparse_kv_indptr[start : stop + 1]
                )
                state = layer.kernels[tile.capacity].forward(
                    state,
                    positions[start:stop],
                    args.kv_cache,
                    args.pe_cache,
                    layer_indices,
                    args.cos,
                    args.sin,
                    x_out=output,
                    layer=layer.slot,
                    advance=False,
                    index_cache=args.index_cache,
                    positions=positions[start:stop],
                    slot_mapping=slot_mapping[start:stop],
                    sparse_kv_indptr=layer_indptr,
                    selected_counts=(
                        selected_counts[start:stop]
                        if index_share and selected_counts is not None
                        else None
                    ),
                    batch_ids=ownership.batch_ids,
                    owned_counts=ownership.owned_counts,
                    kv_cache_scale=args.kv_cache_scale,
                    block_tables=block_tables,
                    context_lens=context_lens,
                )
            self.workspace.advance_step()
        return buffers[(len(self.layers) - 1) % 2]

    def close(self) -> None:
        for layer in self.layers:
            for kernel in layer.kernels.values():
                kernel.close()
        self.workspace.close()


__all__ = [
    "GlmAgenticGraphBucket",
    "GlmAgenticLayer",
    "GlmAgenticLayerInputs",
    "GlmAgenticLayerSpec",
]
