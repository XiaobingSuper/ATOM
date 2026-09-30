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
from atom.model_ops.monokernel.dispatch import (
    MonoUnsupported,
    select_backend,
    tp_uniform_local_validation,
)
from atom.model_ops.monokernel.telemetry import MonoRouteStats
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    atom_mxfp4_storage_view,
    linear_bf16,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

_NATIVE_BUCKETS = 2  # S4 and S8
# Immutable projections/MoE weights are shared by S4/S8. Each bucket retains
# only graph-stable activation/scratch workspaces.
_PACKED_RESERVE_PER_LAYER = 256 << 20
_WORKSPACE_RESERVE_PER_LAYER_BUCKET = 32 << 20


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _kda_state_pool_supported(cache) -> bool:
    """Validate the fixed native KDA state ABI before any device launch."""

    conv_state = getattr(cache, "k_cache", None)
    recurrent_state = getattr(cache, "v_cache", None)
    cfg = KIMI_K3_CONFIG
    if not isinstance(conv_state, torch.Tensor) or not isinstance(recurrent_state, torch.Tensor):
        return False
    slots = conv_state.shape[0] if conv_state.ndim == 3 else 0
    return (
        slots > 0
        and conv_state.shape == (slots, 3, 3 * cfg.local_heads * cfg.v_dim)
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


def _layer_weights(layer, rank: int, npes: int) -> LayerWeights:
    attn = layer.self_attn
    moe = layer.block_sparse_moe
    experts = moe.experts
    cfg = KIMI_K3_CONFIG
    _need(layer.is_linear_attn, f"layer {layer.layer_idx}: not KDA")
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
        "w_kda_in": linear_bf16(
            attn.in_proj,
            name="in_proj",
            logical_rows=4 * cfg.local_heads * cfg.v_dim + cfg.local_heads + cfg.v_dim,
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
        "kda_dt_bias": attn.dt_bias.view(cfg.local_heads, cfg.v_dim),
        "g_kda_out": attn.o_norm.weight,
        "w_kda_o": linear_bf16(
            attn.o_proj,
            name="o_proj",
            logical_rows=cfg.hidden,
            logical_cols=cfg.local_heads * cfg.v_dim,
        ),
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
        defer_collectives: bool = False,
        packed_artifacts: dict[str, object] | None = None,
    ) -> None:
        from atom.model_ops.monokernel.k3.op import KimiK3MonoKernel
        from atom.model_ops.monokernel.k3.staged import _KimiK3KdaStagedPath

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        tp = get_tp_group()
        op_type = KimiK3MonoKernel if mode == "mono" else _KimiK3KdaStagedPath
        self.op = op_type(
            weights,
            samples,
            layer_idx=layer.layer_idx,
            rank=rank,
            npes=npes,
            group=tp.cpu_group,
            reduce_group=tp.device_group,
            mtp=False,
            conv_state_layout=ConvStateLayout.TIME_MAJOR,
            attention_symmetric_allreduce=attention_symmetric_allreduce,
            moe_symmetric_allreduce=moe_symmetric_allreduce,
            state_dtype=state_dtype,
            defer_collectives=defer_collectives,
            packed_artifacts=packed_artifacts,
        )

    def initialize_collectives(self, attention=None, moe=None):
        return self.op.initialize_collectives(attention, moe)


class KimiMonoDecode:
    """Run eligible KDA+MoE layers natively and preserve every fallback layer."""

    def __init__(self, causal_lm, atom_config, mode: str) -> None:
        self._lm = causal_lm
        self._atom_config = atom_config
        self._mode = mode
        self._ops: dict[tuple[int, int, str, torch.dtype], _KimiLayerOp] = {}
        self._weights: dict[int, LayerWeights] = {}
        self._packed_artifacts: dict[int, dict[str, object]] = {}
        self._reductions: dict[tuple[int, str], tuple[object, object]] = {}
        self._refused: set[tuple[int, int, str, torch.dtype]] = set()
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
        """Reserve KV-budget headroom for lazy S4/S8 packed layer artifacts."""

        if not self._enabled:
            return 0
        return (
            len(self._layer_specs(4))
            * (
                _PACKED_RESERVE_PER_LAYER
                + _NATIVE_BUCKETS * _WORKSPACE_RESERVE_PER_LAYER_BUCKET
            )
        )

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
        backend = select_backend(
            "kimi_k3",
            self._mode,
            samples=samples,
            tp_size=self._atom_config.tensor_parallel_size,
            kv_cache_dtype=self._atom_config.kv_cache_dtype,
            dcp=getattr(self._atom_config, "decode_context_parallel_size", 1) > 1,
        )
        if (
            backend is None
            or positions.dtype is not torch.int64
            or positions.numel() != samples
            or not positions.is_contiguous()
        ):
            return self._fallback("dispatch", samples)
        fwd = get_forward_context()
        if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
            return self._fallback("forward_mode", samples)
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
            if shape.query_len != 1 or shape.actual_rows > samples:
                return self._fallback("agentic_shape", samples)
        md = getattr(fwd.attn_metadata, "kda_metadata", None)
        if md is None:
            md = getattr(fwd.attn_metadata, "gdn_metadata", None)
        if md is None:
            return self._fallback("metadata", samples)
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
            and not getattr(md, "replayssm", False)
        )
        if not supported:
            return self._fallback("decode_shape", samples)
        model = getattr(getattr(self, "_lm", None), "model", None)
        if model is None:
            return True
        state_dtype = None
        for layer in model.layers[model.start_layer : model.end_layer]:
            if not layer.is_linear_attn or not hasattr(layer, "block_sparse_moe"):
                continue
            try:
                cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"]
            except KeyError:
                return self._fallback("state_cache", samples)
            if not _kda_state_pool_supported(cache):
                return self._fallback("state_layout", samples)
            if state_dtype is None:
                state_dtype = cache.v_cache.dtype
            elif cache.v_cache.dtype is not state_dtype:
                return self._fallback("mixed_state_dtype", samples)
        if state_dtype is None or not self._prepare(samples, state_dtype):
            return self._fallback("prepare", samples)
        return True

    def _layer_specs(
        self,
        samples: int,
        state_dtype: torch.dtype = torch.float32,
    ):
        model = self._lm.model
        specs = []
        for layer in model.layers[model.start_layer : model.end_layer]:
            backend = select_backend(
                "kimi_k3",
                self._mode,
                samples=samples,
                tp_size=8,
                kv_cache_dtype=self._atom_config.kv_cache_dtype,
                dcp=getattr(self._atom_config, "decode_context_parallel_size", 1) > 1,
                is_kda=layer.is_linear_attn,
                has_moe=hasattr(layer, "block_sparse_moe"),
            )
            if backend is not None:
                specs.append(
                    (
                        layer,
                        backend,
                        (layer.layer_idx, samples, backend, state_dtype),
                    )
                )
        return specs

    def _prepare(self, samples: int, state_dtype: torch.dtype) -> bool:
        specs = self._layer_specs(samples, state_dtype)
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
                grouped.setdefault((samples, backend), []).append((key, owned))
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
                (samples, backend) for _, backend, _ in pending
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
    ) -> _KimiLayerOp:
        backend = select_backend(
            "kimi_k3",
            self._mode,
            samples=samples,
            tp_size=8,
            kv_cache_dtype=self._atom_config.kv_cache_dtype,
            dcp=getattr(self._atom_config, "decode_context_parallel_size", 1) > 1,
            is_kda=layer.is_linear_attn,
            has_moe=hasattr(layer, "block_sparse_moe"),
        )
        if backend is None:
            raise MonoUnsupported("layer fallback")
        key = (layer.layer_idx, samples, backend, state_dtype)
        try:
            return self._ops[key]
        except KeyError as error:
            raise MonoUnsupported("layer was not prepared before launch") from error

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
        backend = select_backend(
            "kimi_k3",
            self._mode,
            samples=samples,
            tp_size=8,
            kv_cache_dtype=self._atom_config.kv_cache_dtype,
        )
        self._route_stats().record_hit(backend or "baseline", samples)
        hidden = model.get_input_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        blocks = hidden.new_zeros(samples, 0, hidden.shape[-1])
        pending = pending2 = None
        for layer in model.layers[model.start_layer : model.end_layer]:
            try:
                owned = self._op(layer, samples, state_dtype)
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
            if block_idx >= blocks.shape[1]:
                extra = hidden.new_zeros(samples, block_idx + 1 - blocks.shape[1], hidden.shape[-1])
                blocks = torch.cat((blocks, extra), dim=1)
            cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"]
            state_indices = md.non_spec_state_indices_tensor[:samples]
            hidden = owned.op.forward(
                hidden,
                blocks,
                state_indices,
                cache.k_cache,
                cache.v_cache,
                epoch_layer=layer.layer_idx,
            )
        hidden, _ = model.output_attn_res(hidden, blocks, pending, pending2)
        return hidden

    def close(self) -> None:
        self._close_reductions()
        self._ops.clear()
        self._weights.clear()
        getattr(self, "_packed_artifacts", {}).clear()
        self._refused.clear()
