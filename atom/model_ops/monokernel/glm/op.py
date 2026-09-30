# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import torch

from atom.model_ops.monokernel.config import (
    AttentionWeight,
    GLM5_CONFIG,
    KvCacheLayout,
    MoeMode,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    RouterWeightLayout,
    as_kv_cache_layout,
    glm5_shard_config,
    validate_shard,
)
from atom.model_ops.monokernel.glm.kernel import build_glm5_monokernel
from atom.model_ops.monokernel.glm.cache import validate_fp8_paged_cache
from atom.model_ops.monokernel.glm.layout import (
    INDEX_DIM,
    layout,
    stage_tasks,
)
from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace
from atom.model_ops.monokernel.layout import TL_COLS
from atom.model_ops.monokernel.packing import pack_bf16, pack_fp8, pack_layer_weights
from atom.model_ops.monokernel.runtime import SymmetricPeerBuffer
from atom.model_ops.monokernel.weights import LayerWeights, prepare_mxfp4_expert_storage

__all__ = ["Glm5MonoKernel", "Glm5PackedArtifacts"]


def _glm_kernel_config(weights: LayerWeights, npes: int):
    config = weights.config
    expected = glm5_shard_config(npes) if npes in (4, 8) else GLM5_CONFIG
    if config != expected:
        raise ValueError(
            f"Glm5MonoKernel TP{npes} requires {expected.local_heads} local "
            f"heads and expert width {expected.inter}, got "
            f"{config.local_heads} heads and width {config.inter}"
        )
    if weights.heads != config.local_heads:
        raise ValueError(
            f"{config.name} requires {config.local_heads} local heads, "
            f"got {weights.heads}"
        )
    physical = config.n_experts + config.num_shared_experts
    if weights.physical_experts != physical:
        raise ValueError(
            f"GLM-5.2 requires {physical} physical experts, "
            f"got {weights.physical_experts}"
        )
    return config


@dataclass(frozen=True)
class Glm5PackedArtifacts:
    """Layer-owned immutable packed weights shared by all row-tile kernels."""

    weights: LayerWeights
    tensors: Mapping[str, torch.Tensor]
    npes: int
    attention_weight: AttentionWeight
    with_indexer: bool
    expert_mxfp4: bool

    @classmethod
    def pack(
        cls,
        weights: LayerWeights,
        *,
        npes: int,
        attention_weight: AttentionWeight | str,
        with_indexer: bool,
    ) -> "Glm5PackedArtifacts":
        config = _glm_kernel_config(weights, npes)
        attention_weight = AttentionWeight(attention_weight)
        t = weights.t
        expert_mxfp4 = t["w_ug"].dtype is torch.uint8
        moe_mode = MoeMode.A16W4 if expert_mxfp4 else MoeMode.W8A8
        profile = replace(config, attention_weight=attention_weight)
        atom_experts = (
            expert_mxfp4
            and weights.mxfp4_weight_layout is Mxfp4WeightLayout.ATOM
            and weights.mxfp4_scale_layout is Mxfp4ScaleLayout.ATOM
        )
        if (
            weights.mxfp4_weight_layout is Mxfp4WeightLayout.ATOM
            or weights.mxfp4_scale_layout is Mxfp4ScaleLayout.ATOM
        ) and not atom_experts:
            raise ValueError("ATOM expert storage requires MXFP4 values and scales together")
        if atom_experts:
            packed = pack_layer_weights(t, moe_mode, profile, attention_only=True)
            packed["w_r"] = pack_bf16(t["w_r"])
            packed.update(
                dict(
                    zip(
                        ("w_ug", "s_ug", "w_dn", "s_dn"),
                        prepare_mxfp4_expert_storage(weights),
                    )
                )
            )
        else:
            packed = pack_layer_weights(
                t,
                moe_mode,
                profile,
                mxfp4_weight_layout=Mxfp4WeightLayout.NATIVE,
                mxfp4_scale_layout=Mxfp4ScaleLayout.NATIVE,
                router_weight_layout=RouterWeightLayout.NATIVE,
            )
        if with_indexer:
            required = (
                "w_index_k",
                "s_index_k",
                "w_index_w",
                "w_index_q",
                "s_index_q",
                "g_index_k",
                "b_index_k",
            )
            missing = [name for name in required if name not in t]
            if missing:
                raise ValueError(
                    f"with_indexer=True requires weights: {', '.join(missing)}"
                )
            packed["w_index_k"] = pack_fp8(t["w_index_k"])
            packed["w_index_q"] = pack_fp8(t["w_index_q"])
            packed["w_index_w"] = pack_bf16(t["w_index_w"])
        return cls(
            weights=weights,
            tensors=MappingProxyType(packed),
            npes=npes,
            attention_weight=attention_weight,
            with_indexer=with_indexer,
            expert_mxfp4=expert_mxfp4,
        )

    def validate(
        self,
        weights: LayerWeights,
        *,
        npes: int,
        attention_weight: AttentionWeight,
        with_indexer: bool,
    ) -> None:
        if self.weights is not weights:
            raise ValueError("packed artifacts belong to a different GLM layer")
        if (
            self.npes != npes
            or self.attention_weight is not attention_weight
            or self.with_indexer != with_indexer
        ):
            raise ValueError("packed artifacts do not match the GLM kernel profile")


class Glm5MonoKernel:
    """One TP rank of GLM-5's indexed decode MonoKernel.

    With ``with_indexer=True``, a single persistent launch covers index K/Q/W
    projection, index K normalization/RoPE/cache update, scoring, exact sparse
    top-k selection, MLA, routing, all expert compute, and both TP reductions.
    ``group`` is a torch.distributed group (None for npes=1).

    The symmetric buffer uses PyTorch's caching allocator and CUDA/ROCm IPC
    storage sharing. Scratch and symmetric buffers may be shared by all layers
    because every launch uses a fresh ``tag``.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        topk: int = 2048,
        launches_per_step: int = 1,
        with_indexer: bool = False,
        index_share: bool = False,
        index_max_seq: int = 4096,
        attention_weight: AttentionWeight | str = AttentionWeight.FP8_BLOCK128,
        kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
        uv_scale_block_m: int = 128,
        agentic_row_contract: bool = False,
        row_capacity: int | None = None,
        timeline=False,
        workspace: GlmAgenticWorkspace | None = None,
        packed_artifacts: Glm5PackedArtifacts | None = None,
    ):
        config = _glm_kernel_config(W, npes)
        validate_shard(samples, W.heads, rank, npes, topk, config)
        if not 1 <= launches_per_step <= 128:
            raise ValueError(f"launches_per_step must be in [1, 128], got {launches_per_step}")
        self.W, self.S, self.rank, self.npes, self.topk = W, samples, rank, npes, topk
        self.launches_per_step = launches_per_step
        self.with_indexer = with_indexer
        self.index_share = index_share or with_indexer
        self.index_max_seq = index_max_seq
        self.attention_weight = AttentionWeight(attention_weight)
        self.kv_cache_layout = as_kv_cache_layout(kv_cache_layout)
        self.agentic_row_contract = agentic_row_contract
        self.workspace = workspace
        if workspace is not None:
            if not agentic_row_contract:
                raise ValueError("shared Agentic workspace requires row ownership")
            if workspace.shape.config != config:
                raise ValueError("shared workspace geometry does not match weights")
            if samples not in {tile.capacity for tile in workspace.shape.row_tiles}:
                raise ValueError(f"row tile size {samples} is outside the graph bucket")
            if timeline:
                raise ValueError("shared Agentic workspace does not support per-op timelines")
        row_capacity = samples if row_capacity is None else row_capacity
        if row_capacity != samples and (
            workspace is None
            or not agentic_row_contract
            or samples != workspace.shape.common.tile_rows
            or row_capacity != workspace.shape.common.row_capacity
        ):
            raise ValueError(
                "persistent row capacity must match the shared Agentic workspace"
            )
        self.row_capacity = row_capacity
        t = W.t
        if packed_artifacts is None:
            packed_artifacts = Glm5PackedArtifacts.pack(
                W,
                npes=npes,
                attention_weight=self.attention_weight,
                with_indexer=with_indexer,
            )
        else:
            packed_artifacts.validate(
                W,
                npes=npes,
                attention_weight=self.attention_weight,
                with_indexer=with_indexer,
            )
        self._packed_artifacts = packed_artifacts
        self.packed = packed_artifacts.tensors
        self.expert_mxfp4 = packed_artifacts.expert_mxfp4
        self.scr_layout, self.sym_layout = layout(
            samples,
            W.heads,
            npes,
            topk,
            with_indexer,
            index_max_seq,
            config,
        )
        dev = torch.device("cuda", torch.cuda.current_device())
        self.stages = stage_tasks(
            samples,
            W.heads,
            topk,
            with_indexer,
            index_max_seq,
            self.expert_mxfp4,
            config,
        )
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None
        self.index_counts = (
            torch.empty(row_capacity, dtype=torch.int32, device=dev)
            if self.index_share else None
        )
        if with_indexer:
            index_tensors = dict(t, **self.packed)
            self.index_params = torch.tensor(
                [
                    index_tensors[name].data_ptr()
                    for name in (
                        "w_index_k",
                        "s_index_k",
                        "w_index_w",
                        "w_index_q",
                        "s_index_q",
                        "g_index_k",
                        "b_index_k",
                    )
                ]
                + [0 if self.timeline is None else self.timeline.data_ptr()],
                dtype=torch.int64,
                device=dev,
            )
        else:
            self.index_params = None
        if workspace is None:
            self.scratch = torch.zeros(
                self.scr_layout["_bytes"],
                dtype=torch.uint8,
                device=dev,
            )
            self.peer_buffer = SymmetricPeerBuffer(
                self.sym_layout["_bytes"],
                rank=rank,
                npes=npes,
                group=group,
            )
            self.step = torch.zeros(
                1,
                dtype=torch.int32,
                device=dev,
            )
            self._owns_peer_buffer = True
        else:
            if workspace.scratch.numel() < self.scr_layout["_bytes"]:
                raise ValueError("shared scratch is smaller than the tile layout")
            self.scratch = workspace.scratch
            self.peer_buffer = workspace.peer_buffer
            self.step = workspace.step
            self._owns_peer_buffer = False
        self.sym_storage = self.peer_buffer.storage
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses
        self.launch = build_glm5_monokernel(
            samples,
            W.heads,
            npes,
            topk,
            launches_per_step=launches_per_step,
            with_indexer=with_indexer,
            index_share=self.index_share,
            index_max_seq=index_max_seq,
            expert_mxfp4=self.expert_mxfp4,
            attention_weight=self.attention_weight,
            kv_cache_layout=self.kv_cache_layout,
            uv_scale_block_m=uv_scale_block_m,
            agentic_row_contract=agentic_row_contract,
            row_capacity=row_capacity,
            timeline=timeline,
            model_config=config,
        )

    @property
    def packed_artifacts(self) -> Glm5PackedArtifacts:
        """Strong reference to the layer-owned packed-weight lifetime."""

        return self._packed_artifacts

    def _validate_cache_storage(
        self,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor | None,
        index_cache: torch.Tensor | None,
        kv_cache_scale: torch.Tensor | None,
    ) -> int:
        cache_width = self.W.config.kv_lora + self.W.config.pe_dim
        if self.kv_cache_layout is KvCacheLayout.ATOM_FP8:
            if kv_cache_scale is None:
                raise ValueError("ATOM FP8 KV cache requires one FP32 scalar descale")
            return validate_fp8_paged_cache(
                kv_cache,
                kv_cache_scale,
                index_cache,
                with_indexer=self.with_indexer,
            )
        if self.kv_cache_layout is not KvCacheLayout.ATOM:
            return kv_cache.shape[0]
        if (
            kv_cache.dtype is not torch.bfloat16
            or not kv_cache.is_contiguous()
            or kv_cache.shape[-1] != cache_width
        ):
            raise ValueError(
                f"ATOM KV cache must be contiguous BF16 with {cache_width} columns"
            )
        if pe_cache is None or kv_cache.data_ptr() != pe_cache.data_ptr():
            raise ValueError(
                "ATOM KV cache layout requires the same fused tensor for "
                "kv_cache and pe_cache"
            )
        return kv_cache.numel() // cache_width

    def debug(self, name: str, shape, dtype=torch.float32, pairs=True, bf2=False) -> torch.Tensor:
        """Values of a scratch mailbox (``(value, tag)`` pairs unless ``pairs=False``;
        ``bf2``: each pair's value word packs two bf16 elements)."""
        off = self.scr_layout[name]
        n = 1
        for d in shape:
            n *= d
        if not pairs:
            return self.scratch[off : off + n * 4].view(dtype).view(shape)
        if bf2:
            words = self.scratch[off : off + n * 4].view(torch.int32).view(n // 2, 2)[:, 0].contiguous()
            return words.view(torch.bfloat16).float().view(shape)
        words = self.scratch[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()
        return words.view(dtype).view(shape)

    def forward(
        self,
        h,
        cur_pos,
        kv_cache,
        pe_cache,
        indices,
        cos,
        sin,
        x_out=None,
        layer=0,
        advance=True,
        index_cache=None,
        positions=None,
        slot_mapping=None,
        sparse_kv_indptr=None,
        batch_ids=None,
        owned_counts=None,
        kv_cache_scale=None,
        block_tables=None,
        context_lens=None,
        selected_counts=None,
    ):
        """One layer across the configured row capacity.

        Layers sharing this scratch within a model call need distinct ``layer``
        slots. Call ``advance_step`` (or pass ``advance=True``) once per model
        call; both operations are stream ordered and HIP-graph capturable.
        """
        if not 0 <= layer < self.launches_per_step:
            raise ValueError(f"layer must be in [0, {self.launches_per_step}), got {layer}")
        rows = self.row_capacity
        hidden_shape = (rows, self.W.config.hidden)
        if (
            h.shape != hidden_shape
            or h.dtype is not torch.bfloat16
            or not h.is_contiguous()
        ):
            raise ValueError(
                f"hidden states must be contiguous BF16 {list(hidden_shape)}"
            )
        if (
            self.with_indexer
            and self.kv_cache_layout is not KvCacheLayout.ATOM_FP8
        ):
            if index_cache is None:
                raise ValueError("index_cache is required when with_indexer=True")
            if index_cache.shape != (self.index_max_seq, INDEX_DIM) or index_cache.dtype is not torch.bfloat16:
                raise ValueError(
                    f"index_cache must be bf16 [{self.index_max_seq}, {INDEX_DIM}], got "
                    f"{tuple(index_cache.shape)} {index_cache.dtype}"
                )
        if self.kv_cache_layout in (KvCacheLayout.ATOM, KvCacheLayout.ATOM_FP8):
            self._validate_cache_storage(
                kv_cache,
                pe_cache,
                index_cache,
                kv_cache_scale,
            )
            if self.kv_cache_layout is KvCacheLayout.ATOM_FP8:
                if (
                    block_tables is None
                    or block_tables.dtype is not torch.int32
                    or block_tables.ndim != 2
                    or not block_tables.is_contiguous()
                ):
                    raise ValueError(
                        "block_tables must be contiguous int32 [batch, blocks]"
                    )
                if (
                    context_lens is None
                    or context_lens.dtype is not torch.int32
                    or context_lens.ndim != 1
                    or not context_lens.is_contiguous()
                ):
                    raise ValueError(
                        "context_lens must be contiguous int32 [batch]"
                    )
            for name, value, dtype, size in (
                ("positions", positions, torch.int64, rows),
                ("slot_mapping", slot_mapping, torch.int64, rows),
                ("sparse_kv_indptr", sparse_kv_indptr, torch.int32, rows + 1),
            ):
                if value is None or value.dtype is not dtype or value.numel() < size or not value.is_contiguous():
                    got = None if value is None else (tuple(value.shape), value.dtype)
                    raise ValueError(f"{name} must be contiguous {dtype} with at least {size} values, got {got}")
            if self.agentic_row_contract:
                for name, value in (
                    ("batch_ids", batch_ids),
                    ("owned_counts", owned_counts),
                ):
                    if (
                        value is None
                        or value.dtype is not torch.int32
                        or value.numel() < rows
                        or not value.is_contiguous()
                    ):
                        got = None if value is None else (tuple(value.shape), value.dtype)
                        raise ValueError(
                            f"{name} must be contiguous int32 with at least "
                            f"{rows} values, got {got}"
                        )
        t = dict(self.W.t, **self.packed)
        if self.index_share:
            if indices.dtype is not torch.int32 or indices.numel() < rows * self.topk:
                raise ValueError(
                    f"IndexShare requires {rows * self.topk} int32 slots"
                )
            if selected_counts is None:
                if not self.with_indexer:
                    raise ValueError(
                        "shared IndexShare attention requires selected_counts"
                    )
                selected_counts = self.index_counts
            if (
                selected_counts.dtype is not torch.int32
                or selected_counts.numel() < rows
                or not selected_counts.is_contiguous()
            ):
                raise ValueError(
                    f"selected_counts must contain {rows} contiguous int32 values"
                )
        if x_out is None:
            x_out = torch.empty(
                rows,
                self.W.config.hidden,
                dtype=torch.bfloat16,
                device=h.device,
            )
        elif (
            x_out.shape != hidden_shape
            or x_out.dtype is not torch.bfloat16
            or not x_out.is_contiguous()
        ):
            raise ValueError(
                f"output must be contiguous BF16 {list(hidden_shape)}"
            )
        p = lambda x: x.data_ptr()  # noqa: E731
        self.launch(
            p(h),
            p(x_out),
            p(cur_pos),
            p(cur_pos if positions is None else positions),
            p(cur_pos if slot_mapping is None else slot_mapping),
            p(cur_pos if sparse_kv_indptr is None else sparse_kv_indptr),
            p(cur_pos if batch_ids is None else batch_ids),
            p(cur_pos if owned_counts is None else owned_counts),
            p(cur_pos if block_tables is None else block_tables),
            p(cur_pos if context_lens is None else context_lens),
            0 if block_tables is None else block_tables.shape[1],
            p(kv_cache),
            p(kv_cache if pe_cache is None else pe_cache),
            p(kv_cache if kv_cache_scale is None else kv_cache_scale),
            p(indices),
            p(indices if index_cache is None else index_cache),
            p(indices if selected_counts is None else selected_counts),
            p(cos),
            p(sin),
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p(t["w_qkv_a"]),
            p(t.get("s_qkv_a", t["w_qkv_a"])),
            p(t["w_q_b"]),
            p(t.get("s_q_b", t["w_q_b"])),
            p(t["w_uk"]),
            p(t.get("s_uk", t["w_uk"])),
            p(t["w_uv"]),
            p(t.get("s_uv", t["w_uv"])),
            p(t["w_o"]),
            p(t.get("s_o", t["w_o"])),
            p(t["w_r"]),
            p(t["bias"]),
            p(t["w_ug"]),
            p(t["s_ug"]),
            p(t["w_dn"]),
            p(t["s_dn"]),
            p(self.scratch),
            self.sym,
            p(self.peers),
            p(self.index_params) if self.with_indexer else (0 if self.timeline is None else p(self.timeline)),
            p(self.step),
            self.rank,
            layer,
            stream=torch.cuda.current_stream(),
        )
        if advance:
            self.advance_step()
        return x_out

    def advance_step(self):
        self.step.add_(1)

    def close(self):
        """Release this rank's remote HIP IPC mappings."""

        if self._owns_peer_buffer:
            self.peer_buffer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def timeline_report(self) -> str:
        """Per stage, in us from launch start: [first start, median hint seen, last end]
        and median per-task phases (hint wait, payload staging, compute, epilogue)."""
        tl = self.timeline[:, :5].cpu().double() / 100.0  # s_memrealtime ticks at 100 MHz
        t0 = tl[:, 0].min()
        rows, i = [], 0
        for name, n in self.stages:
            st = tl[i : i + n].clone()
            i += n
            for c in (1, 2, 3):  # missing marks inherit the previous one
                st[:, c] = torch.where(st[:, c] > 0, st[:, c], st[:, c - 1])
            d = (st[:, 1:] - st[:, :-1]).median(0).values
            rows.append(
                f"{name:7s} x{n:4d}  [{(st[:, 0].min() - t0):6.1f} | hint {(st[:, 1].median() - t0):6.1f} | "
                f"end {(st[:, 4].max() - t0):6.1f}]  hint {d[0]:5.1f}  stage {d[1]:5.1f}  "
                f"compute {d[2]:5.1f}  epi {d[3]:5.1f}"
            )
            if name == "index_score":
                per_sample = n // self.S
                ready = [(st[s * per_sample : (s + 1) * per_sample, 4].max() - t0).item() for s in range(self.S)]
                rows.append(" " * 10 + "score-ready/sample " + " ".join(f"{v:.1f}" for v in ready))
            elif name == "index_select":
                done = [(st[s, 4] - t0).item() for s in range(self.S)]
                rows.append(" " * 10 + "select-done/sample " + " ".join(f"{v:.1f}" for v in done))
        return "\n".join(rows)

    def intermediates(self):
        S, H, config = self.S, self.W.heads, self.W.config
        result = dict(
            q_a=self.debug("q_a", (S, config.q_lora)),
            kv_a=self.debug("kv_a", (S, config.kv_lora + config.pe_dim)),
            q_nope=self.debug("q_nope", (S, H, config.nope_dim), bf2=True),
            q_pe=self.debug("q_pe", (S, H, config.pe_dim), bf2=True),
            q_lat=self.debug("q_lat", (S, H, config.kv_lora), bf2=True),
            o=self.debug("o", (S, H * config.v_dim), bf2=True),
            a=self.debug("a", (S, config.hidden), bf2=True).to(torch.bfloat16),
            scores=self.debug("scores", (S, config.n_experts)),
            sel=self.debug("sel", (S, config.moe_slots), torch.int32),
            prob=self.debug("prob", (S, config.moe_slots)),
            mid=self.debug("mid", (S, config.moe_slots, config.inter)),
            xq=self.debug("xqd", (S, config.hidden), pairs=False),
        )
        if self.with_indexer:
            result["index_q"] = self.debug("index_q", (S, 32, INDEX_DIM), bf2=True)
            result["index_w"] = self.debug("index_w", (S, 32))
            off = self.scr_layout["indices"]
            result["indices"] = self.scratch[off : off + S * self.topk * 4].view(torch.int32).view(S, self.topk)
        return result
