# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native Kimi-K3 KDA decode using ATOM-owned MonoKernel sources."""

from __future__ import annotations

import logging

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.monokernel.config import (
    KIMI_K3_CONFIG,
    ConvStateLayout,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
)
from atom.model_ops.monokernel.abi import AgenticDecodeShape
from atom.model_ops.monokernel.k3.abi import (
    KimiAgenticRuntime,
    KimiAgenticShape,
    KimiMlaAgenticRuntime,
)
from atom.model_ops.monokernel.dispatch import (
    KIMI_MLA_AGENTIC_ROWS,
    SAMPLES,
    MonoUnsupported,
    select_backend,
    tp_uniform_local_validation,
)
from atom.model_ops.monokernel.layout import symmetric_allreduce_nbytes
from atom.model_ops.monokernel.telemetry import MonoRouteStats
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    atom_mxfp4_storage_view,
    linear_bf16,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# Immutable projections/MoE weights are shared by S4/S8. Each bucket retains
# only graph-stable activation/scratch workspaces.
_PACKED_RESERVE_PER_LAYER = 256 << 20
_WORKSPACE_RESERVE_PER_LAYER_BUCKET = 32 << 20
_HIDDEN = KIMI_K3_CONFIG.hidden
_PROJECTION = KIMI_K3_CONFIG.local_heads * KIMI_K3_CONFIG.v_dim
_FUSED_PAD = 6400
_BF16_BYTES = 2
_INT64_BYTES = 8


def _mxfp8_packed_nbytes(rows: int, cols: int) -> int:
    scale_rows = (rows + 255) // 256 * 256
    return rows * cols + scale_rows * (cols // 32)


_FULL_MOE_PACKED_BYTES = (
    KIMI_K3_CONFIG.n_experts * _HIDDEN * 2
    + _mxfp8_packed_nbytes(KIMI_K3_CONFIG.routed_hidden, _HIDDEN)
    + _mxfp8_packed_nbytes(2 * KIMI_K3_CONFIG.shared_inter, _HIDDEN)
    + _mxfp8_packed_nbytes(_HIDDEN, KIMI_K3_CONFIG.shared_inter)
    + _mxfp8_packed_nbytes(
        _HIDDEN // 8,
        KIMI_K3_CONFIG.routed_hidden,
    )
)
_FULL_KDA_PACKED_BYTES = (
    _FUSED_PAD * _HIDDEN * 2
    + _HIDDEN * _PROJECTION * 2
    + _FULL_MOE_PACKED_BYTES
)
_FULL_MLA_PACKED_BYTES = (
    _HIDDEN * _PROJECTION * 2 + _FULL_MOE_PACKED_BYTES
)
_FULL_DENSE_PACKED_BYTES = (
    _FUSED_PAD * _HIDDEN * 2
    + _HIDDEN * _PROJECTION * 2
    + 2 * (33792 // 8) * _HIDDEN * 2
    + _HIDDEN * (33792 // 8) * 2
)
_FULL_SHARED_ARENA_BYTES = (
    8_969_472
    + max(KIMI_MLA_AGENTIC_ROWS) * 8 * _HIDDEN * 2
    + 4
)
_FULL_COLLECTIVE_BYTES = sum(
    symmetric_allreduce_nbytes((rows * _HIDDEN,), 8)
    + symmetric_allreduce_nbytes(
        (
            rows * KIMI_K3_CONFIG.routed_hidden,
            rows * _HIDDEN,
        ),
        8,
    )
    for rows in KIMI_MLA_AGENTIC_ROWS
)
_FULL_SHARED_BYTES = _FULL_SHARED_ARENA_BYTES + _FULL_COLLECTIVE_BYTES


def _full_layer_workspace_nbytes(backend: str, rows: int) -> int:
    """Exact persistent bucket-local bytes after shared-plan rebinding."""

    timeline = 10 * _INT64_BYTES
    if backend == "dense_full":
        return (
            6 * rows * _HIDDEN * _BF16_BYTES
            + rows * _HIDDEN
            + rows * ((_HIDDEN + 255) // 256)
            + timeline
        )
    if backend not in {"mono", "mla_full"}:
        raise ValueError(f"unsupported Kimi full backend {backend!r}")
    padded_rows = (rows + 31) // 32 * 32
    common = (
        6 * rows * _HIDDEN * _BF16_BYTES
        + padded_rows * _HIDDEN
        + padded_rows * (_HIDDEN // 32)
        + timeline
    )
    if backend == "mla_full":
        return common
    max_sorted = rows * KIMI_K3_CONFIG.top_k * 16
    max_blocks = (max_sorted + 15) // 16
    staged = (
        padded_rows * _HIDDEN
        + padded_rows * (_HIDDEN // 32)
        + 2 * max_sorted * 4
        + max_blocks * 4
        + 2 * 4
        + max_sorted * KIMI_K3_CONFIG.inter * _BF16_BYTES
        + rows * KIMI_K3_CONFIG.n_experts * (2 + 4 + 8)
        + rows * KIMI_K3_CONFIG.top_k * (4 + 8 + 4 + 4)
        + 4
        * rows
        * KIMI_K3_CONFIG.routed_hidden
        * _BF16_BYTES
        + 3
        * rows
        * KIMI_K3_CONFIG.shared_inter
        * _BF16_BYTES
        + 3 * rows * _HIDDEN * _BF16_BYTES
        + rows * (_HIDDEN // 8) * _BF16_BYTES
    )
    return common + staged


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _kda_state_pool_supported(cache, *, conv_state_length: int = 3) -> bool:
    """Validate the fixed native KDA state ABI before any device launch."""

    conv_state = getattr(cache, "k_cache", None)
    recurrent_state = getattr(cache, "v_cache", None)
    cfg = KIMI_K3_CONFIG
    if not isinstance(conv_state, torch.Tensor) or not isinstance(recurrent_state, torch.Tensor):
        return False
    slots = conv_state.shape[0] if conv_state.ndim == 3 else 0
    return (
        slots > 0
        and conv_state.shape
        == (slots, conv_state_length, 3 * cfg.local_heads * cfg.v_dim)
        and conv_state.dtype is torch.bfloat16
        and conv_state.is_contiguous()
        and recurrent_state.shape == (slots, cfg.local_heads, cfg.v_dim, cfg.v_dim)
        and recurrent_state.dtype in (torch.float16, torch.float32)
        and recurrent_state.is_contiguous()
    )


def _kda_state_dtype(model, fwd) -> torch.dtype | None:
    for layer in model.layers[model.start_layer : model.end_layer]:
        if not getattr(layer, "is_linear_attn", False):
            continue
        cache = fwd.kv_cache_data.get(f"layer_{layer.layer_idx}")
        state = None if cache is None else getattr(cache, "v_cache", None)
        if isinstance(state, torch.Tensor):
            return state.dtype
    return None


def _mla_cache_descale(layer, cache: torch.Tensor) -> torch.Tensor:
    """Bind the device scalar owned by the production attention implementation."""

    scale = layer.self_attn.attn.impl._k_scale_device
    if (
        not isinstance(scale, torch.Tensor)
        or scale.dtype is not torch.float32
        or scale.numel() != 1
        or not scale.is_contiguous()
        or scale.device != cache.device
    ):
        raise ValueError(
            "Kimi MLA cache descale must be one contiguous FP32 scalar "
            "on the cache device"
        )
    return scale


def _split_kv_b(
    weight: torch.Tensor,
    *,
    heads: int,
    nope_dim: int,
    value_dim: int,
    kv_lora: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert production KV-B rows to absorbed W_UK/W_UV matrices."""

    by_head = weight.view(heads, nope_dim + value_dim, kv_lora)
    w_uk = (
        by_head[:, :nope_dim]
        .transpose(1, 2)
        .contiguous()
        .view(heads * kv_lora, nope_dim)
    )
    w_uv = (
        by_head[:, nope_dim:]
        .contiguous()
        .view(heads * value_dim, kv_lora)
    )
    return w_uk, w_uv


def _layer_weights(layer, rank: int, npes: int) -> LayerWeights:
    attn = layer.self_attn
    if not hasattr(layer, "block_sparse_moe"):
        cfg = KIMI_K3_CONFIG
        _need(layer.layer_idx == 0, "only Kimi layer 0 may use dense FFN")
        _need(layer.is_linear_attn, "Kimi dense layer 0 must use KDA")
        dense_inter = layer.mlp.down_proj.weight.shape[1]
        _need(
            dense_inter == 33792 // npes,
            f"dense FFN shard width {dense_inter}",
        )
        tensors = {
            "g_in": layer.input_layernorm.weight,
            "g_post": layer.post_attention_layernorm.weight,
            "g_self_res": layer.self_attention_res_norm.weight,
            "w_self_res": linear_bf16(
                layer.self_attention_res_proj,
                name="self_attention_res_proj",
                logical_rows=1,
                logical_cols=cfg.hidden,
            ).view(-1),
            "g_mlp_res": layer.mlp_res_norm.weight,
            "w_mlp_res": linear_bf16(
                layer.mlp_res_proj,
                name="mlp_res_proj",
                logical_rows=1,
                logical_cols=cfg.hidden,
            ).view(-1),
            "w_dense_ug": linear_bf16(
                layer.mlp.gate_up_proj,
                name="mlp.gate_up_proj",
                logical_rows=2 * dense_inter,
                logical_cols=cfg.hidden,
            ),
            "w_dense_dn": linear_bf16(
                layer.mlp.down_proj,
                name="mlp.down_proj",
                logical_rows=cfg.hidden,
                logical_cols=dense_inter,
            ),
            "w_kda_in": linear_bf16(
                attn.in_proj,
                name="in_proj",
                logical_rows=(
                    4 * cfg.local_heads * cfg.v_dim
                    + cfg.local_heads
                    + cfg.v_dim
                ),
                logical_cols=cfg.hidden,
            ),
            "w_kda_fb": linear_bf16(
                attn.f_b_proj,
                name="f_b_proj",
                logical_rows=cfg.local_heads * cfg.v_dim,
                logical_cols=cfg.v_dim,
            ),
            "w_kda_conv": attn.conv_weight,
            "kda_a_log": attn.A_log.float().contiguous(),
            "kda_dt_bias": attn.dt_bias.view(
                cfg.local_heads,
                cfg.v_dim,
            ),
            "g_kda_out": attn.o_norm.weight,
            "w_kda_o": linear_bf16(
                attn.o_proj,
                name="o_proj",
                logical_rows=cfg.hidden,
                logical_cols=cfg.local_heads * cfg.v_dim,
            ),
        }
        return LayerWeights(
            cfg.local_heads,
            tensors,
            cfg,
            rank,
            npes,
            mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
            mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        )
    moe = layer.block_sparse_moe
    experts = moe.experts
    cfg = KIMI_K3_CONFIG
    _need(hasattr(layer, "block_sparse_moe"), f"layer {layer.layer_idx}: dense FFN")
    _need(not experts.quant_method.is_guinterleave, "ATOM_MOE_GU_ITLV must be 0")
    _need(experts.global_num_experts == cfg.n_experts, "expert count")
    _need(experts.intermediate_size_per_partition == cfg.inter, "expert width")

    w_ug = experts.w13_weight
    s_ug = experts.w13_weight_scale
    w_dn = experts.w2_weight
    s_dn = experts.w2_weight_scale
    atom_mxfp4_storage_view(
        w_ug,
        name="w_ug",
        logical_rows=cfg.n_experts * 2 * cfg.inter,
        logical_k=cfg.routed_hidden,
        scale=False,
    )
    atom_mxfp4_storage_view(
        s_ug,
        name="s_ug",
        logical_rows=cfg.n_experts * 2 * cfg.inter,
        logical_k=cfg.routed_hidden,
        scale=True,
    )
    atom_mxfp4_storage_view(
        w_dn,
        name="w_dn",
        logical_rows=cfg.n_experts * cfg.routed_hidden,
        logical_k=cfg.inter,
        scale=False,
    )
    atom_mxfp4_storage_view(
        s_dn,
        name="s_dn",
        logical_rows=cfg.n_experts * cfg.routed_hidden,
        logical_k=cfg.inter,
        scale=True,
    )
    shard = cfg.hidden // npes
    tensors = {
        "g_in": layer.input_layernorm.weight,
        "g_post": layer.post_attention_layernorm.weight,
        "g_self_res": layer.self_attention_res_norm.weight,
        "w_self_res": linear_bf16(
            layer.self_attention_res_proj,
            name="self_attention_res_proj",
            logical_rows=1,
            logical_cols=cfg.hidden,
        ).view(-1),
        "g_mlp_res": layer.mlp_res_norm.weight,
        "w_mlp_res": linear_bf16(
            layer.mlp_res_proj,
            name="mlp_res_proj",
            logical_rows=1,
            logical_cols=cfg.hidden,
        ).view(-1),
        "w_r": linear_bf16(
            moe.gate,
            name="gate",
            logical_rows=cfg.n_experts,
            logical_cols=cfg.hidden,
        ),
        "bias": moe.gate.e_score_correction_bias,
        "w_latent_down": linear_bf16(
            moe.routed_expert_down_proj,
            name="routed_expert_down_proj",
            logical_rows=cfg.routed_hidden,
            logical_cols=cfg.hidden,
        ),
        "g_latent": moe.routed_expert_norm.weight,
        "w_latent_up": linear_bf16(
            moe.routed_expert_up_proj,
            name="routed_expert_up_proj",
            logical_rows=cfg.hidden,
            logical_cols=cfg.routed_hidden,
            row_start=rank * shard,
            row_count=shard,
        ),
        "w_shared_ug": linear_bf16(
            moe.shared_experts.gate_up_proj,
            name="shared_experts.gate_up_proj",
            logical_rows=2 * cfg.shared_inter,
            logical_cols=cfg.hidden,
        ),
        "w_shared_dn": linear_bf16(
            moe.shared_experts.down_proj,
            name="shared_experts.down_proj",
            logical_rows=cfg.hidden,
            logical_cols=cfg.shared_inter,
        ),
        "w_ug": w_ug,
        "s_ug": s_ug,
        "w_dn": w_dn,
        "s_dn": s_dn,
    }
    if layer.is_linear_attn:
        tensors.update(
            {
                "w_kda_in": linear_bf16(
                    attn.in_proj,
                    name="in_proj",
                    logical_rows=(
                        4 * cfg.local_heads * cfg.v_dim
                        + cfg.local_heads
                        + cfg.v_dim
                    ),
                    logical_cols=cfg.hidden,
                ),
                "w_kda_fb": linear_bf16(
                    attn.f_b_proj,
                    name="f_b_proj",
                    logical_rows=cfg.local_heads * cfg.v_dim,
                    logical_cols=cfg.v_dim,
                ),
                "w_kda_conv": attn.conv_weight,
                "kda_a_log": attn.A_log.float().contiguous(),
                "kda_dt_bias": attn.dt_bias.view(
                    cfg.local_heads,
                    cfg.v_dim,
                ),
                "g_kda_out": attn.o_norm.weight,
                "w_kda_o": linear_bf16(
                    attn.o_proj,
                    name="o_proj",
                    logical_rows=cfg.hidden,
                    logical_cols=cfg.local_heads * cfg.v_dim,
                ),
            }
        )
    else:
        kv_b = linear_bf16(
            attn.kv_b_proj,
            name="kv_b_proj",
            logical_rows=cfg.local_heads * (cfg.nope_dim + cfg.v_dim),
            logical_cols=cfg.kv_lora,
        )
        w_uk, w_uv = _split_kv_b(
            kv_b,
            heads=cfg.local_heads,
            nope_dim=cfg.nope_dim,
            value_dim=cfg.v_dim,
            kv_lora=cfg.kv_lora,
        )
        tensors.update(
            {
                "w_qkv_a": linear_bf16(
                    attn.fused_qkv_a_proj,
                    name="fused_qkv_a_proj",
                    logical_rows=cfg.q_lora + cfg.kv_lora + cfg.pe_dim,
                    logical_cols=cfg.hidden,
                ),
                "g_q": attn.q_a_layernorm.weight,
                "g_kv": attn.kv_a_layernorm.weight,
                "w_q_b": linear_bf16(
                    attn.q_b_proj,
                    name="q_b_proj",
                    logical_rows=cfg.local_heads
                    * (cfg.nope_dim + cfg.pe_dim),
                    logical_cols=cfg.q_lora,
                ),
                "w_uk": w_uk,
                "w_uv": w_uv,
                "w_gate": linear_bf16(
                    attn.g_proj,
                    name="g_proj",
                    logical_rows=cfg.local_heads * cfg.v_dim,
                    logical_cols=cfg.hidden,
                ),
                "w_o": linear_bf16(
                    attn.o_proj,
                    name="o_proj",
                    logical_rows=cfg.hidden,
                    logical_cols=cfg.local_heads * cfg.v_dim,
                ),
            }
        )
    return LayerWeights(
        cfg.local_heads,
        tensors,
        cfg,
        rank,
        npes,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
    )


class _KimiLayerOp:
    def __init__(
        self,
        layer,
        weights: LayerWeights,
        samples: int,
        mode: str,
        *,
        attention_symmetric_allreduce=None,
        moe_symmetric_allreduce=None,
        state_dtype: torch.dtype = torch.float32,
        agentic_batch_size: int = 0,
        defer_collectives: bool = False,
        packed_artifacts: dict[str, object] | None = None,
    ) -> None:
        from atom.model_ops.monokernel.k3.op import KimiK3MonoKernel
        from atom.model_ops.monokernel.k3.staged import _KimiK3KdaStagedPath

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        tp = get_tp_group()
        if mode == "mla_full":
            from atom.model_ops.monokernel.k3.mla_full import (
                KimiK3MlaMonoKernel,
            )

            op_type = KimiK3MlaMonoKernel
        elif mode == "dense_full":
            from atom.model_ops.monokernel.k3.dense_full import (
                KimiK3DenseMonoKernel,
            )

            op_type = KimiK3DenseMonoKernel
        else:
            op_type = (
                KimiK3MonoKernel
                if mode == "mono"
                else _KimiK3KdaStagedPath
            )
        op_kwargs = dict(
            layer_idx=layer.layer_idx,
            rank=rank,
            npes=npes,
            group=tp.cpu_group,
            reduce_group=tp.device_group,
            mtp=agentic_batch_size > 0,
            conv_state_layout=ConvStateLayout.TIME_MAJOR,
            attention_symmetric_allreduce=attention_symmetric_allreduce,
            moe_symmetric_allreduce=moe_symmetric_allreduce,
            state_dtype=state_dtype,
            defer_collectives=defer_collectives,
            packed_artifacts=packed_artifacts,
        )
        if mode == "mla_full":
            for name in (
                "mtp",
                "conv_state_layout",
                "state_dtype",
            ):
                op_kwargs.pop(name)
        elif agentic_batch_size:
            if mode not in {"mono", "dense_full"}:
                raise ValueError("Kimi Agentic q=8 requires the full MonoKernel")
            op_kwargs["agentic_batch_size"] = agentic_batch_size
        self.op = op_type(weights, samples, **op_kwargs)

    def initialize_collectives(self, attention=None, moe=None):
        return self.op.initialize_collectives(attention, moe)


class KimiFullModelPlan:
    """Atomically own every DSpark full-layer bucket and shared model arena."""

    def __init__(self, runner: "KimiMonoDecode") -> None:
        self.runner = runner
        self.ops: dict[
            tuple[int, int, str, torch.dtype, int],
            _KimiLayerOp,
        ] = {}
        self.packed_artifacts: dict[int, dict[str, object]] = {}
        self.reductions: dict[
            int,
            tuple[object, object],
        ] = {}
        self.max_arena: torch.Tensor | None = None
        self.block_residual: torch.Tensor | None = None
        self.model_epoch: torch.Tensor | None = None
        self.ready = False

    @staticmethod
    def _backend(layer) -> str:
        if not hasattr(layer, "block_sparse_moe"):
            if layer.layer_idx != 0 or not layer.is_linear_attn:
                raise MonoUnsupported("unsupported dense layer")
            return "dense_full"
        return "mono" if layer.is_linear_attn else "mla_full"

    @staticmethod
    def _scratch(op):
        scratch = getattr(op, "monokernel_scratch", None)
        if scratch is not None:
            return scratch
        return getattr(getattr(op, "attention", None), "monokernel_scratch", None)

    @staticmethod
    def _bind_arena_and_epoch(op, arena, epoch) -> None:
        if hasattr(op, "monokernel_scratch"):
            op.monokernel_scratch = arena
        attention = getattr(op, "attention", None)
        if attention is not None:
            attention.monokernel_scratch = arena
            attention.step = epoch
        if hasattr(op, "step"):
            op.step = epoch

    @staticmethod
    def _track_collectives(tracked: dict[int, object], op) -> None:
        attention = getattr(op, "attention", None)
        for resource in (
            getattr(attention, "symmetric_allreduce", None),
            getattr(op, "symmetric_allreduce", None),
        ):
            if resource is not None:
                tracked[id(resource)] = resource

    def commit(
        self,
        candidates,
        artifacts,
        reductions,
        max_arena,
        block_residual,
        model_epoch,
    ) -> None:
        self.ops = candidates
        self.packed_artifacts = artifacts
        self.reductions = reductions
        self.max_arena = max_arena
        self.block_residual = block_residual
        self.model_epoch = model_epoch
        shared_artifacts = getattr(
            self.runner,
            "_packed_artifacts",
            None,
        )
        if shared_artifacts is None:
            self.runner._packed_artifacts = dict(artifacts)
        else:
            shared_artifacts.update(artifacts)
        self.ready = True

    def prepare(self) -> bool:
        if self.ready:
            return True
        runner = self.runner
        model = runner._lm.model
        layers = tuple(
            model.layers[model.start_layer : model.end_layer]
        )
        if not layers or layers[0].layer_idx != 0:
            return False
        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        tp = get_tp_group()
        candidates = {}
        artifacts: dict[int, dict[str, object]] = dict(
            getattr(runner, "_packed_artifacts", {})
        )
        reductions = {}
        tracked_collectives: dict[int, object] = {}
        max_arena = block_residual = model_epoch = None
        from atom.model_ops.monokernel.k3.kernel import (
            monokernel_scratch_nbytes,
        )

        arena_bytes = max(
            monokernel_scratch_nbytes(
                rows,
                fuse_attn_res=True,
                fuse_moe=True,
                mtp=not mla,
                mla=mla,
                dense_ffn=dense,
            )
            for rows in KIMI_MLA_AGENTIC_ROWS
            for mla, dense in ((False, False), (False, True), (True, False))
        )
        max_blocks = max(layer.layer_idx // 12 for layer in layers) + 1
        try:
            for rows in KIMI_MLA_AGENTIC_ROWS:
                batch = rows // 8
                pending = []
                for layer in layers:
                    backend = self._backend(layer)
                    weights = None
                    error = None
                    try:
                        weights = _layer_weights(layer, rank, npes)
                    except (
                        AttributeError,
                        MonoUnsupported,
                        RuntimeError,
                        ValueError,
                    ) as caught:
                        error = caught
                    tp_uniform_local_validation(
                        error,
                        group=tp.cpu_group,
                        world_size=npes,
                        context=(
                            f"Kimi-K3 full-plan layer {layer.layer_idx} "
                            "weight mapping failed"
                        ),
                    )
                    assert weights is not None
                    owned = None
                    error = None
                    try:
                        owned = _KimiLayerOp(
                            layer,
                            weights,
                            rows,
                            backend,
                            state_dtype=torch.float16,
                            agentic_batch_size=batch,
                            defer_collectives=True,
                            packed_artifacts=artifacts.get(
                                layer.layer_idx
                            ),
                        )
                    except (
                        AttributeError,
                        MonoUnsupported,
                        RuntimeError,
                        ValueError,
                    ) as caught:
                        error = caught
                    tp_uniform_local_validation(
                        error,
                        group=tp.cpu_group,
                        world_size=npes,
                        context=(
                            f"Kimi-K3 full-plan layer {layer.layer_idx} "
                            f"B{batch} construction failed"
                        ),
                    )
                    assert owned is not None
                    if max_arena is None:
                        scratch = self._scratch(owned.op)
                        allocated = None
                        error = None
                        try:
                            if scratch is None:
                                raise ValueError(
                                    "full plan has no device arena"
                                )
                            max_arena = torch.zeros(
                                arena_bytes,
                                dtype=torch.uint8,
                                device=scratch.device,
                            )
                            block_residual = torch.empty(
                                max(KIMI_MLA_AGENTIC_ROWS),
                                max_blocks,
                                KIMI_K3_CONFIG.hidden,
                                dtype=torch.bfloat16,
                                device=scratch.device,
                            )
                            model_epoch = torch.zeros(
                                1,
                                dtype=torch.int32,
                                device=scratch.device,
                            )
                            allocated = (
                                max_arena,
                                block_residual,
                                model_epoch,
                            )
                        except (RuntimeError, ValueError) as caught:
                            error = caught
                        tp_uniform_local_validation(
                            error,
                            group=tp.cpu_group,
                            world_size=npes,
                            context=(
                                "Kimi-K3 full-plan shared arena "
                                "allocation failed"
                            ),
                        )
                        assert allocated is not None
                        max_arena, block_residual, model_epoch = allocated
                    self._bind_arena_and_epoch(
                        owned.op,
                        max_arena,
                        model_epoch,
                    )
                    if layer.layer_idx not in artifacts:
                        artifacts[layer.layer_idx] = (
                            owned.op.packed_artifacts()
                        )
                    owned.op.release_packed_sources()
                    key = (
                        layer.layer_idx,
                        rows,
                        backend,
                        torch.float16,
                        batch,
                    )
                    candidates[key] = owned
                    pending.append(owned)
                shared = (None, None)
                for owned in pending:
                    initialized = None
                    error = None
                    try:
                        initialized = owned.initialize_collectives(*shared)
                    except (
                        AttributeError,
                        RuntimeError,
                        ValueError,
                    ) as caught:
                        error = caught
                    finally:
                        self._track_collectives(
                            tracked_collectives,
                            owned.op,
                        )
                        if initialized is not None:
                            for resource in initialized:
                                if resource is not None:
                                    tracked_collectives[id(resource)] = resource
                    tp_uniform_local_validation(
                        error,
                        group=tp.cpu_group,
                        world_size=npes,
                        context=(
                            "Kimi-K3 full-plan collective "
                            f"B{batch} initialization failed"
                        ),
                    )
                    assert initialized is not None
                    shared = initialized
                    reductions[rows] = shared
            assert (
                max_arena is not None
                and block_residual is not None
                and model_epoch is not None
            )
            self.commit(
                candidates,
                artifacts,
                reductions,
                max_arena,
                block_residual,
                model_epoch,
            )
            return True
        except (
            AttributeError,
            MonoUnsupported,
            RuntimeError,
            ValueError,
        ) as error:
            for resource in reversed(tuple(tracked_collectives.values())):
                resource.close()
            logger.warning(
                "Kimi-K3 full-plan fallback before launch: %s",
                error,
            )
            return False

    def lookup(
        self,
        layer,
        rows: int,
        state_dtype: torch.dtype,
        batch: int,
    ) -> _KimiLayerOp:
        backend = self._backend(layer)
        key = (layer.layer_idx, rows, backend, state_dtype, batch)
        try:
            return self.ops[key]
        except KeyError as error:
            raise MonoUnsupported("full plan is incomplete") from error

    def forward(self, callback):
        if not self.ready or self.model_epoch is None:
            raise MonoUnsupported("full plan is not ready")
        return callback()

    def close(self) -> None:
        closed = set()
        for attention, moe in reversed(tuple(self.reductions.values())):
            for resource in (moe, attention):
                if resource is None or id(resource) in closed:
                    continue
                resource.close()
                closed.add(id(resource))
        self.reductions.clear()
        self.ops.clear()
        self.packed_artifacts.clear()
        self.max_arena = None
        self.block_residual = None
        self.model_epoch = None
        self.ready = False


class KimiMonoDecode:
    """Run eligible KDA+MoE layers natively and preserve every fallback layer."""

    def __init__(self, causal_lm, atom_config, mode: str) -> None:
        self._lm = causal_lm
        self._atom_config = atom_config
        self._mode = mode
        self._ops: dict[
            tuple[int, int, str, torch.dtype, int],
            _KimiLayerOp,
        ] = {}
        self._weights: dict[int, LayerWeights] = {}
        self._packed_artifacts: dict[int, dict[str, object]] = {}
        self._reductions: dict[tuple[int, str], tuple[object, object]] = {}
        self._refused: set[tuple[int, int, str, torch.dtype, int]] = set()
        self._full_plan = KimiFullModelPlan(self)
        self._capture_full_route_decision: bool | None = None
        self._stats = MonoRouteStats("kimi_k3")
        self._enabled = mode != "off"
        if not self._enabled:
            return
        checks = (
            (atom_config.tensor_parallel_size == 8, "not TP8"),
            (atom_config.parallel_config.data_parallel_size == 1, "DP"),
            (not atom_config.enable_dp_attention, "DPA"),
            (atom_config.pipeline_parallel_size == 1, "PP"),
            (not is_plugin_mode(), "plugin mode"),
            (atom_config.kv_cache_dtype in ("bf16", "fp8"), "KV dtype"),
        )
        for ok, why in checks:
            if not ok:
                logger.info("Kimi-K3 MonoKernel off: %s", why)
                self._enabled = False
                break

    def _route_stats(self) -> MonoRouteStats:
        stats = getattr(self, "_stats", None)
        if stats is None:
            stats = MonoRouteStats("kimi_k3")
            self._stats = stats
        return stats

    def route_stats(self) -> dict[str, object]:
        return self._route_stats().snapshot()

    def _fallback(self, reason: str, samples: int) -> bool:
        self._route_stats().record_fallback(reason, samples)
        return False

    def _close_reductions(self, keys=None) -> None:
        """Close adapter-owned shared reductions in collective order."""

        keys = tuple(self._reductions) if keys is None else tuple(keys)
        closed: set[int] = set()
        for key in keys:
            reductions = self._reductions.get(key)
            if reductions is None:
                continue
            for reduction in reversed(reductions):
                if reduction is None or id(reduction) in closed:
                    continue
                reduction.close()
                closed.add(id(reduction))
            self._reductions.pop(key)

    def memory_reserve_bytes(self) -> int:
        """Reserve headroom only for native routes this mode can select."""

        if not self._enabled or self._mode not in {
            "auto",
            "mono",
            "staged",
        }:
            return 0
        model = self._lm.model
        speculative = getattr(self._atom_config, "speculative_config", None)
        dspark = getattr(speculative, "method", None) == "dspark"
        full_dspark = self._full_route_eligible()
        if self._mode in {"auto", "mono"} and dspark and not full_dspark:
            return 0
        if self._mode == "auto" and not full_dspark:
            return 0
        dcp = getattr(
            self._atom_config,
            "decode_context_parallel_size",
            1,
        ) > 1
        layers = tuple(
            model.layers[model.start_layer : model.end_layer]
        )
        if full_dspark:
            packed = 0
            workspaces = 0
            kda_layers = 0
            for layer in layers:
                backend = KimiFullModelPlan._backend(layer)
                if not hasattr(layer, "block_sparse_moe"):
                    packed += _FULL_DENSE_PACKED_BYTES
                elif layer.is_linear_attn:
                    packed += _FULL_KDA_PACKED_BYTES
                    kda_layers += 1
                else:
                    packed += _FULL_MLA_PACKED_BYTES
                workspaces += sum(
                    _full_layer_workspace_nbytes(backend, rows)
                    for rows in KIMI_MLA_AGENTIC_ROWS
                )
            reserve = (
                packed
                + workspaces
                + _FULL_SHARED_BYTES
            )
            if self._mode == "mono":
                reserve += (
                    kda_layers
                    * len(SAMPLES)
                    * _WORKSPACE_RESERVE_PER_LAYER_BUCKET
                )
            return reserve
        packed_layers: set[int] = set()
        workspace_buckets = 0
        for layer in layers:
            is_kda = bool(getattr(layer, "is_linear_attn", False))
            if is_kda:
                buckets = tuple((samples, False) for samples in SAMPLES)
            else:
                buckets = ()
            for samples, mtp in buckets:
                route = dict(
                    samples=samples,
                    tp_size=self._atom_config.tensor_parallel_size,
                    kv_cache_dtype=self._atom_config.kv_cache_dtype,
                    mtp=mtp,
                    dcp=dcp,
                    query_len=8 if mtp else 1,
                    is_kda=is_kda,
                    has_moe=hasattr(layer, "block_sparse_moe"),
                )
                if not is_kda:
                    route.update(
                        external_indexer=False,
                        cache_layout="atom_fp8",
                        segment="mla_layer",
                        replay_ssm=False,
                    )
                backend = select_backend("kimi_k3", self._mode, **route)
                if backend in {"mono", "mla_full", "staged"}:
                    packed_layers.add(layer.layer_idx)
                    workspace_buckets += 1
        return (
            len(packed_layers) * _PACKED_RESERVE_PER_LAYER
            + workspace_buckets * _WORKSPACE_RESERVE_PER_LAYER_BUCKET
        )

    def _full_route_eligible(
        self,
        *,
        samples: int | None = None,
        query_len: int = 8,
        replay_ssm: bool = False,
    ) -> bool:
        speculative = getattr(
            self._atom_config,
            "speculative_config",
            None,
        )
        if (
            not self._enabled
            or self._mode not in {"auto", "mono"}
            or self._atom_config.tensor_parallel_size != 8
            or getattr(
                self._atom_config,
                "decode_context_parallel_size",
                1,
            )
            != 1
            or self._atom_config.kv_cache_dtype != "fp8"
            or getattr(
                speculative,
                "method",
                None,
            )
            != "dspark"
            or int(
                getattr(speculative, "num_speculative_tokens", 7)
                or 0
            )
            != 7
            or envs.ATOM_ENABLE_REPLAYSSM is True
            or replay_ssm
        ):
            return False
        if samples is not None and (
            samples not in KIMI_MLA_AGENTIC_ROWS or query_len != 8
        ):
            return False
        model = getattr(getattr(self, "_lm", None), "model", None)
        if model is None:
            return False
        layers = tuple(model.layers[model.start_layer : model.end_layer])
        expected_layers = getattr(
            getattr(self._lm, "config", None),
            "num_hidden_layers",
            model.end_layer,
        )
        if (
            not layers
            or model.start_layer != 0
            or model.end_layer != expected_layers
            or len(layers) != expected_layers
            or any(
                layer.layer_idx != index
                for index, layer in enumerate(layers)
            )
        ):
            return False
        try:
            return all(
                (
                    layer.layer_idx == 0
                    and KimiFullModelPlan._backend(layer) == "dense_full"
                )
                or (
                    layer.layer_idx > 0
                    and hasattr(layer, "block_sparse_moe")
                    and KimiFullModelPlan._backend(layer)
                    in {"mono", "mla_full"}
                )
                for layer in layers
            )
        except MonoUnsupported:
            return False

    def begin_capture_lifecycle(self) -> None:
        """Reset the capture-wide all-or-baseline decision."""

        self._capture_full_route_decision = None

    def prepare_for_capture(self) -> bool:
        """Build the complete DSpark plan before any graph capture begins."""

        if not self._full_route_eligible():
            return True
        if self._capture_full_route_decision is None:
            self._capture_full_route_decision = self._full_plan.prepare()
        return self._capture_full_route_decision

    def supports(self, input_ids, positions, intermediate_tensors, inputs_embeds) -> bool:
        samples = input_ids.numel()
        self._route_stats().record_attempt(samples)
        if not self._enabled:
            return self._fallback("disabled", samples)
        if intermediate_tensors is not None:
            return self._fallback("pipeline", samples)
        if inputs_embeds is not None and (
            inputs_embeds.shape != (samples, KIMI_K3_CONFIG.hidden)
            or inputs_embeds.dtype != torch.bfloat16
            or not inputs_embeds.is_contiguous()
        ):
            return self._fallback("inputs_embeds", samples)
        fwd = get_forward_context()
        if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
            return self._fallback("forward_mode", samples)
        md = getattr(fwd.attn_metadata, "kda_metadata", None)
        if md is None:
            md = getattr(fwd.attn_metadata, "gdn_metadata", None)
        if md is None:
            return self._fallback("metadata", samples)
        agentic = md.num_spec_decodes > 0
        query_len = getattr(fwd.context, "max_seqlen_q", 1)
        replay_ssm = getattr(md, "replayssm", False)
        speculative = getattr(self._atom_config, "speculative_config", None)
        if agentic and getattr(speculative, "method", None) != "dspark":
            return self._fallback("spec_method", samples)
        full_route = agentic and self._full_route_eligible(
            samples=samples,
            query_len=query_len,
            replay_ssm=replay_ssm,
        )
        if (
            agentic
            and self._mode in {"auto", "mono"}
            and not full_route
        ):
            return self._fallback("full_route", samples)
        backend = select_backend(
            "kimi_k3",
            self._mode,
            samples=samples,
            tp_size=self._atom_config.tensor_parallel_size,
            kv_cache_dtype=self._atom_config.kv_cache_dtype,
            mtp=agentic,
            dcp=getattr(self._atom_config, "decode_context_parallel_size", 1) > 1,
            query_len=query_len,
            replay_ssm=replay_ssm,
        )
        if (
            backend is None
            or positions.dtype is not torch.int64
            or positions.numel() != samples
            or not positions.is_contiguous()
        ):
            return self._fallback("dispatch", samples)
        agentic_batch_size = 0
        if agentic:
            try:
                common = AgenticDecodeShape.from_forward_mode(
                    fwd.context,
                    batch_capacity=fwd.context.running_bs,
                    row_capacity=samples,
                )
                shape = KimiAgenticShape(
                    common=common,
                    dcp_size=getattr(
                        self._atom_config,
                        "decode_context_parallel_size",
                        1,
                    ),
                    replay_ssm=replay_ssm,
                )
                KimiAgenticRuntime.bind(
                    shape,
                    md.spec_state_indices_tensor,
                    md.num_accepted_tokens,
                )
            except (AttributeError, TypeError, ValueError):
                return self._fallback("agentic_shape", samples)
            agentic_batch_size = shape.common.batch_capacity
            supported = (
                md.num_prefills == 0
                and md.num_decodes == 0
                and 0 < md.num_spec_decodes <= agentic_batch_size
                and 0 < md.num_actual_tokens <= samples
                and md.num_spec_decode_tokens == md.num_actual_tokens
            )
        else:
            shape = None
            if hasattr(fwd.context, "running_bs"):
                try:
                    shape = AgenticDecodeShape.from_forward_mode(
                        fwd.context,
                        batch_capacity=fwd.context.running_bs,
                        row_capacity=max(
                            samples,
                            fwd.context.running_bs * fwd.context.max_seqlen_q,
                        ),
                    )
                except ValueError:
                    return self._fallback("agentic_shape", samples)
            if shape is not None and (
                shape.query_len != 1 or shape.actual_rows > samples
            ):
                return self._fallback("agentic_shape", samples)
            state_indices = getattr(md, "non_spec_state_indices_tensor", None)
            if (
                not isinstance(state_indices, torch.Tensor)
                or state_indices.shape != (samples,)
                or state_indices.dtype is not torch.int32
                or not state_indices.is_contiguous()
            ):
                return self._fallback("state_indices", samples)
            supported = (
                md.num_prefills == 0
                and md.num_decodes == md.num_actual_tokens
                and md.num_spec_decodes == 0
                and 0 < md.num_actual_tokens <= samples
                and not replay_ssm
            )
        if not supported:
            return self._fallback("decode_shape", samples)
        model = getattr(getattr(self, "_lm", None), "model", None)
        if model is None:
            return True
        mla_layers = [
            layer
            for layer in model.layers[model.start_layer : model.end_layer]
            if not getattr(layer, "is_linear_attn", True)
        ]
        state_dtype = None
        for layer in model.layers[model.start_layer : model.end_layer]:
            if not getattr(layer, "is_linear_attn", True):
                continue
            try:
                cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"]
            except KeyError:
                return self._fallback("state_cache", samples)
            if not _kda_state_pool_supported(
                cache,
                conv_state_length=10 if agentic else 3,
            ):
                return self._fallback("state_layout", samples)
            if state_dtype is None:
                state_dtype = cache.v_cache.dtype
            elif cache.v_cache.dtype is not state_dtype:
                return self._fallback("mixed_state_dtype", samples)
        if agentic and state_dtype is not torch.float16:
            return self._fallback("state_dtype", samples)
        if agentic and self._atom_config.kv_cache_dtype != "fp8":
            return self._fallback("kv_dtype", samples)
        if agentic and mla_layers:
            attention_metadata = fwd.attn_metadata
            try:
                mla_runtime = KimiMlaAgenticRuntime.bind(
                    shape,
                    positions,
                    attention_metadata.slot_mapping[:samples],
                    attention_metadata.batch_id_per_q_token[:samples],
                    attention_metadata.context_lens,
                    attention_metadata.block_tables,
                    block_size=attention_metadata.block_size,
                    block_ratio=attention_metadata.block_ratio,
                )
                del mla_runtime
                for layer in mla_layers:
                    cache = fwd.kv_cache_data[
                        f"layer_{layer.layer_idx}"
                    ].k_cache
                    scale = _mla_cache_descale(layer, cache)
                    from atom.model_ops.monokernel.k3.mla_cache import (
                        validate_fp8_mla_cache,
                    )

                    validate_fp8_mla_cache(cache, scale)
            except (AttributeError, KeyError, TypeError, ValueError):
                return self._fallback("mla_metadata", samples)
        prepared = False
        if state_dtype is not None:
            full_plan = getattr(self, "_full_plan", None)
            if full_route and full_plan is not None:
                if (
                    torch.cuda.is_current_stream_capturing()
                    and not full_plan.ready
                ):
                    return self._fallback("capture_prepare", samples)
                decision = getattr(
                    self,
                    "_capture_full_route_decision",
                    None,
                )
                if decision is False:
                    return self._fallback("capture_prepare", samples)
                prepared = (
                    decision
                    if decision is not None
                    else full_plan.prepare()
                )
            else:
                prepared = (
                    self._prepare(
                        samples,
                        state_dtype,
                        agentic_batch_size,
                    )
                    if agentic
                    else self._prepare(samples, state_dtype)
                )
        if not prepared:
            return self._fallback("prepare", samples)
        return True

    def _layer_specs(
        self,
        samples: int,
        state_dtype: torch.dtype = torch.float32,
        agentic_batch_size: int = 0,
    ):
        model = self._lm.model
        specs = []
        for layer in model.layers[model.start_layer : model.end_layer]:
            route = dict(
                samples=samples,
                tp_size=8,
                kv_cache_dtype=self._atom_config.kv_cache_dtype,
                mtp=agentic_batch_size > 0,
                dcp=getattr(
                    self._atom_config,
                    "decode_context_parallel_size",
                    1,
                )
                > 1,
                query_len=8 if agentic_batch_size else 1,
                is_kda=layer.is_linear_attn,
                has_moe=hasattr(layer, "block_sparse_moe"),
            )
            if not getattr(layer, "is_linear_attn", True):
                route.update(
                    external_indexer=False,
                    cache_layout="atom_fp8",
                    segment="mla_layer",
                    replay_ssm=False,
                )
            backend = select_backend(
                "kimi_k3",
                self._mode,
                **route,
            )
            if backend is not None:
                specs.append(
                    (
                        layer,
                        backend,
                        (
                            layer.layer_idx,
                            samples,
                            backend,
                            state_dtype,
                            agentic_batch_size,
                        ),
                    )
                )
        return specs

    def _prepare(
        self,
        samples: int,
        state_dtype: torch.dtype,
        agentic_batch_size: int = 0,
    ) -> bool:
        specs = self._layer_specs(samples, state_dtype, agentic_batch_size)
        if not specs:
            return False
        if all(key in self._ops for _, _, key in specs):
            return True
        if any(key in self._refused for _, _, key in specs):
            return False
        if torch.cuda.is_current_stream_capturing():
            return False

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        tp = get_tp_group()
        pending = []
        try:
            for layer, backend, key in specs:
                if key in self._ops:
                    continue
                validation_error = None
                weights = None
                try:
                    weights = self._weights.get(layer.layer_idx)
                    if weights is None:
                        weights = _layer_weights(layer, rank, npes)
                except (AttributeError, MonoUnsupported, RuntimeError, ValueError) as error:
                    validation_error = error
                tp_uniform_local_validation(
                    validation_error,
                    group=tp.cpu_group,
                    world_size=npes,
                    context=f"Kimi-K3 layer {layer.layer_idx} weight mapping failed",
                )
                assert weights is not None
                self._weights[layer.layer_idx] = weights
                owned = None
                construction_error = None
                packed_artifacts = self._packed_artifacts.get(layer.layer_idx)
                try:
                    owned = _KimiLayerOp(
                        layer,
                        weights,
                        samples,
                        backend,
                        state_dtype=state_dtype,
                        agentic_batch_size=agentic_batch_size,
                        defer_collectives=True,
                        packed_artifacts=packed_artifacts,
                    )
                except (AttributeError, MonoUnsupported, RuntimeError, ValueError) as error:
                    construction_error = error
                tp_uniform_local_validation(
                    construction_error,
                    group=tp.cpu_group,
                    world_size=npes,
                    context=f"Kimi-K3 layer {layer.layer_idx} rank-local construction failed",
                )
                assert owned is not None
                if packed_artifacts is None:
                    self._packed_artifacts[layer.layer_idx] = (
                        owned.op.packed_artifacts()
                    )
                owned.op.release_packed_sources()
                self._weights.pop(layer.layer_idx, None)
                pending.append((key, backend, owned))

            grouped = {}
            for key, backend, owned in pending:
                family = (
                    "full"
                    if backend in {"mono", "mla_full"}
                    else backend
                )
                grouped.setdefault((samples, family), []).append((key, owned))
            for reduction_key, group in grouped.items():
                shared = self._reductions.get(reduction_key, (None, None))
                for key, owned in group:
                    initialized = None
                    initialization_error = None
                    try:
                        initialized = owned.initialize_collectives(*shared)
                    except (AttributeError, RuntimeError, ValueError) as error:
                        initialization_error = error
                    tp_uniform_local_validation(
                        initialization_error,
                        group=tp.cpu_group,
                        world_size=npes,
                        context=f"Kimi-K3 layer {key[0]} collective initialization failed",
                    )
                    assert initialized is not None
                    shared = initialized
                self._reductions[reduction_key] = shared

            self._ops.update((key, owned) for key, _, owned in pending)
        except (AttributeError, MonoUnsupported, RuntimeError, ValueError) as error:
            closed = set()
            for _, _, owned in reversed(pending):
                reductions = (
                    getattr(owned.op, "symmetric_allreduce", None),
                    getattr(owned.op.attention, "symmetric_allreduce", None),
                )
                for reduction in reductions:
                    if reduction is None or id(reduction) in closed:
                        continue
                    reduction.close()
                    closed.add(id(reduction))
            for reduction_key in {
                (
                    samples,
                    "full"
                    if backend in {"mono", "mla_full"}
                    else backend,
                )
                for _, backend, _ in pending
            }:
                self._reductions.pop(reduction_key, None)
            for layer, _, _ in specs:
                self._weights.pop(layer.layer_idx, None)
            self._refused.update(key for _, _, key in specs)
            logger.warning("Kimi-K3 MonoKernel fallback before launch: %s", error)
            return False
        logger.info(
            "Kimi-K3 MonoKernel ready: backend=%s S=%d layers=%d",
            specs[0][1],
            samples,
            len(specs),
        )
        return True

    def _op(
        self,
        layer,
        samples: int,
        state_dtype: torch.dtype,
        agentic_batch_size: int = 0,
    ) -> _KimiLayerOp:
        route = dict(
            samples=samples,
            tp_size=8,
            kv_cache_dtype=self._atom_config.kv_cache_dtype,
            mtp=agentic_batch_size > 0,
            dcp=getattr(
                self._atom_config,
                "decode_context_parallel_size",
                1,
            )
            > 1,
            query_len=8 if agentic_batch_size else 1,
            is_kda=layer.is_linear_attn,
            has_moe=hasattr(layer, "block_sparse_moe"),
        )
        if not layer.is_linear_attn:
            route.update(
                external_indexer=False,
                cache_layout="atom_fp8",
                segment="mla_layer",
                replay_ssm=False,
            )
        backend = select_backend(
            "kimi_k3",
            self._mode,
            **route,
        )
        if backend is None:
            raise MonoUnsupported("backend unavailable")
        key = (
            layer.layer_idx,
            samples,
            backend,
            state_dtype,
            agentic_batch_size,
        )
        try:
            return self._ops[key]
        except KeyError as error:
            raise MonoUnsupported("layer was not prepared before launch") from error

    @staticmethod
    def _publish_layer_output(layer, positions, layer_input, post_hidden):
        """Preserve decoder forward-hook and DSpark aux-output semantics."""

        output = (post_hidden, None, None, None)
        original = output
        for hook in getattr(layer, "_forward_hooks", {}).values():
            if hasattr(hook, "_atom_native_output_buffer"):
                continue
            hooked = hook(layer, (positions, layer_input), output)
            if hooked is not None:
                output = hooked
        if output is original:
            return post_hidden
        if isinstance(output, tuple):
            return layer.aux_hidden_state(output)
        return output

    @staticmethod
    def _native_output_target(layer, fwd, samples):
        for hook in getattr(layer, "_forward_hooks", {}).values():
            buffer = getattr(hook, "_atom_native_output_buffer", None)
            if buffer is None:
                continue
            offset = getattr(fwd.context, "ubatch_token_offset", 0)
            return buffer[offset : offset + samples]
        return None

    def _forward_layers(
        self,
        hidden,
        positions,
        fwd,
        md,
        state_dtype,
        agentic_batch_size,
        full_plan,
    ):
        model = self._lm.model
        samples = hidden.shape[0]
        blocks = (
            full_plan.block_residual[:samples]
            if full_plan is not None
            else hidden.new_zeros(samples, 0, hidden.shape[-1])
        )
        pending = pending2 = None
        layers = tuple(
            model.layers[model.start_layer : model.end_layer]
        )
        for layer_index, layer in enumerate(layers):
            advance_epoch = (
                full_plan is None or layer_index == len(layers) - 1
            )
            layer_input = hidden
            output_target = self._native_output_target(
                layer,
                fwd,
                samples,
            )
            if full_plan is not None:
                owned = full_plan.lookup(
                    layer,
                    samples,
                    state_dtype,
                    agentic_batch_size,
                )
            else:
                try:
                    owned = self._op(
                        layer,
                        samples,
                        state_dtype,
                        agentic_batch_size,
                    )
                except MonoUnsupported:
                    hidden, pending, pending2, blocks = layer(
                        positions,
                        hidden,
                        blocks,
                        pending_add=pending,
                        pending_add2=pending2,
                    )
                    continue
            for add in (pending, pending2):
                if add is not None:
                    hidden = hidden + add
            pending = pending2 = None
            block_idx = owned.op.block_write_idx
            if full_plan is None and block_idx >= blocks.shape[1]:
                extra = hidden.new_zeros(samples, block_idx + 1 - blocks.shape[1], hidden.shape[-1])
                blocks = torch.cat((blocks, extra), dim=1)
            cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"]
            if not getattr(layer, "is_linear_attn", True):
                attention_metadata = fwd.attn_metadata
                runtime = KimiMlaAgenticRuntime.bind(
                    KimiAgenticShape(
                        common=AgenticDecodeShape.from_forward_mode(
                            fwd.context,
                            batch_capacity=agentic_batch_size,
                            row_capacity=samples,
                        ),
                        dcp_size=1,
                        replay_ssm=False,
                    ),
                    positions,
                    attention_metadata.slot_mapping[:samples],
                    attention_metadata.batch_id_per_q_token[:samples],
                    attention_metadata.context_lens,
                    attention_metadata.block_tables,
                    block_size=attention_metadata.block_size,
                    block_ratio=attention_metadata.block_ratio,
                )
                scale = _mla_cache_descale(layer, cache.k_cache)
                hidden = owned.op.forward(
                    hidden,
                    blocks,
                    runtime,
                    cache.k_cache,
                    scale,
                    x_out=output_target,
                    epoch_layer=layer.layer_idx,
                    advance=advance_epoch,
                )
                hidden = self._publish_layer_output(
                    layer,
                    positions,
                    layer_input,
                    hidden,
                )
                continue
            state_indices = (
                md.spec_state_indices_tensor
                if agentic_batch_size
                else md.non_spec_state_indices_tensor[:samples]
            )
            forward_kwargs = {"epoch_layer": layer.layer_idx}
            if agentic_batch_size:
                forward_kwargs["num_accepted_tokens"] = (
                    md.num_accepted_tokens
                )
            forward_kwargs["advance"] = advance_epoch
            hidden = owned.op.forward(
                hidden,
                blocks,
                state_indices,
                cache.k_cache,
                cache.v_cache,
                x_out=output_target,
                **forward_kwargs,
            )
            hidden = self._publish_layer_output(
                layer,
                positions,
                layer_input,
                hidden,
            )
        hidden, _ = model.output_attn_res(hidden, blocks, pending, pending2)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self._lm.model
        fwd = get_forward_context()
        md = getattr(fwd.attn_metadata, "kda_metadata", None)
        if md is None:
            md = fwd.attn_metadata.gdn_metadata
        samples = input_ids.numel()
        state_dtype = _kda_state_dtype(model, fwd) or torch.float32
        agentic_batch_size = (
            md.spec_state_indices_tensor.shape[0]
            if md.num_spec_decodes > 0
            else 0
        )
        hidden = (
            model.get_input_embeddings(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        configured_full_plan = getattr(self, "_full_plan", None)
        full_plan = (
            configured_full_plan
            if agentic_batch_size
            and self._mode in {"auto", "mono"}
            and configured_full_plan is not None
            and configured_full_plan.ready
            else None
        )
        if full_plan is not None:
            hidden = full_plan.forward(
                lambda: self._forward_layers(
                    hidden,
                    positions,
                    fwd,
                    md,
                    state_dtype,
                    agentic_batch_size,
                    full_plan,
                )
            )
            backend = "full_model"
        else:
            hidden = self._forward_layers(
                hidden,
                positions,
                fwd,
                md,
                state_dtype,
                agentic_batch_size,
                None,
            )
            backend = "staged"
        self._route_stats().record_hit(backend, samples)
        return hidden

    def close(self) -> None:
        full_plan = getattr(self, "_full_plan", None)
        if full_plan is not None:
            full_plan.close()
        self._close_reductions()
        self._ops.clear()
        self._weights.clear()
        getattr(self, "_packed_artifacts", {}).clear()
        self._refused.clear()
