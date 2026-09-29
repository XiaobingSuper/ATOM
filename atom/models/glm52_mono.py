# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native GLM-5.2 sparse-MLA decode using ATOM-owned MonoKernel sources."""

from __future__ import annotations

import logging

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.layernorm import rmsnorm2d_fwd_
from atom.model_ops.monokernel.config import (
    EPS,
    GLM5_CONFIG,
    AttentionWeight,
    KvCacheLayout,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
)
from atom.model_ops.monokernel.dispatch import (
    MonoUnsupported,
    is_flat_atom_cache_page_size,
    select_backend,
    tp_uniform_local_validation,
)
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    atom_mxfp4_storage_view,
    linear_bf16,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _bf16_vector(tensor: torch.Tensor, name: str, size: int) -> torch.Tensor:
    _need(tensor.dtype is torch.bfloat16, f"{name} must be BF16, got {tensor.dtype}")
    _need(tensor.shape == (size,) and tensor.is_contiguous(), f"{name} layout")
    return tensor


def _split_kv_b(
    weight: torch.Tensor,
    *,
    heads: int,
    nope_dim: int,
    value_dim: int,
    kv_lora: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (heads * (nope_dim + value_dim), kv_lora)
    _need(weight.shape == expected, f"kv_b_proj shape {tuple(weight.shape)} != {expected}")
    by_head = weight.view(heads, nope_dim + value_dim, kv_lora)
    w_uk = by_head[:, :nope_dim].transpose(1, 2).contiguous().view(heads * kv_lora, nope_dim)
    w_uv = by_head[:, nope_dim:].contiguous().view(heads * value_dim, kv_lora)
    return w_uk, w_uv


def _atom_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    view = tensor.view(torch.uint8)
    view.is_shuffled = bool(getattr(tensor, "is_shuffled", False))
    return view


def _layer_weights(layer, rank: int, npes: int) -> LayerWeights:
    cfg = GLM5_CONFIG
    attn = layer.self_attn
    moe = layer.mlp
    experts = moe.experts
    physical_experts = cfg.n_experts + cfg.num_shared_experts

    _need(experts.global_num_experts == physical_experts, "shared expert is not fused into ATOM expert storage")
    _need(experts.num_fused_shared_experts == cfg.num_shared_experts, "fused shared-expert count")
    _need(not experts.use_ep, "expert parallelism")
    _need(not experts.quant_method.is_guinterleave, "ATOM_MOE_GU_ITLV must be 0")
    _need(experts.intermediate_size_per_partition == cfg.inter, "expert width")
    for name, norm, size in (
        ("input_layernorm", layer.input_layernorm, cfg.hidden),
        ("q_a_layernorm", attn.q_a_layernorm, cfg.q_lora),
        ("kv_a_layernorm", attn.kv_a_layernorm, cfg.kv_lora),
        ("post_attention_layernorm", layer.post_attention_layernorm, cfg.hidden),
    ):
        _need(norm.eps == EPS, f"{name} epsilon {norm.eps} != {EPS}")
        _bf16_vector(norm.weight, f"{name}.weight", size)

    w_ug = _atom_byte_view(experts.w13_weight)
    s_ug = experts.w13_weight_scale.view(torch.uint8)
    w_dn = _atom_byte_view(experts.w2_weight)
    s_dn = experts.w2_weight_scale.view(torch.uint8)
    atom_mxfp4_storage_view(
        w_ug,
        name="w_ug",
        logical_rows=physical_experts * 2 * cfg.inter,
        logical_k=cfg.hidden,
        scale=False,
    )
    atom_mxfp4_storage_view(
        s_ug,
        name="s_ug",
        logical_rows=physical_experts * 2 * cfg.inter,
        logical_k=cfg.hidden,
        scale=True,
    )
    atom_mxfp4_storage_view(
        w_dn,
        name="w_dn",
        logical_rows=physical_experts * cfg.hidden,
        logical_k=cfg.inter,
        scale=False,
    )
    atom_mxfp4_storage_view(
        s_dn,
        name="s_dn",
        logical_rows=physical_experts * cfg.hidden,
        logical_k=cfg.inter,
        scale=True,
    )

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
    bias = moe.gate.e_score_correction_bias
    _need(bias is not None and bias.dtype is torch.float32, "router correction bias must be FP32")
    _need(bias.shape == (cfg.n_experts,) and bias.is_contiguous(), "router correction bias layout")

    tensors = {
        "g_in": _bf16_vector(layer.input_layernorm.weight, "input_layernorm.weight", cfg.hidden),
        "g_q": _bf16_vector(attn.q_a_layernorm.weight, "q_a_layernorm.weight", cfg.q_lora),
        "g_kv": _bf16_vector(attn.kv_a_layernorm.weight, "kv_a_layernorm.weight", cfg.kv_lora),
        "g_post": _bf16_vector(layer.post_attention_layernorm.weight, "post_attention_layernorm.weight", cfg.hidden),
        "w_qkv_a": linear_bf16(
            attn.fused_qkv_a_proj,
            name="fused_qkv_a_proj",
            logical_rows=cfg.qkv_a_rows,
            logical_cols=cfg.hidden,
        ),
        "w_q_b": linear_bf16(
            attn.q_b_proj,
            name="q_b_proj",
            logical_rows=cfg.local_heads * (cfg.nope_dim + cfg.pe_dim),
            logical_cols=cfg.q_lora,
        ),
        "w_uk": w_uk,
        "w_uv": w_uv,
        "w_o": linear_bf16(
            attn.o_proj,
            name="o_proj",
            logical_rows=cfg.hidden,
            logical_cols=cfg.local_heads * cfg.v_dim,
        ),
        "w_r": linear_bf16(
            moe.gate,
            name="gate",
            logical_rows=cfg.n_experts,
            logical_cols=cfg.hidden,
        ),
        "bias": bias,
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
        physical_experts=physical_experts,
    )


def _attention_impl(layer):
    wrapped = layer.self_attn.mla_attn
    return getattr(wrapped, "impl", wrapped)


def _shared_sparse_buffer(layers) -> torch.Tensor:
    shared = None
    for layer in layers:
        impl = _attention_impl(layer)
        current = getattr(impl, "sparse_kv_indices_buffer", None)
        _need(current is not None and current.dtype is torch.int32 and current.is_contiguous(), "sparse index buffer")
        if shared is None:
            shared = current
        else:
            _need(current.data_ptr() == shared.data_ptr(), "IndexShare layers do not share physical sparse indices")
        indexer = getattr(layer.self_attn, "indexer", None)
        if indexer is not None:
            _need(
                indexer.sparse_kv_indices_buffer.data_ptr() == shared.data_ptr(),
                "full IndexShare layer is not bound to the shared sparse indices",
            )
    _need(shared is not None, "no sparse index buffer")
    return shared


class _GlmLayerOp:
    def __init__(self, weights: LayerWeights, samples: int, topk: int) -> None:
        from atom.model_ops.monokernel.glm.op import Glm5MonoKernel

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        self.op = Glm5MonoKernel(
            weights,
            samples,
            rank=rank,
            npes=npes,
            group=get_tp_group().cpu_group,
            topk=topk,
            launches_per_step=1,
            with_indexer=False,
            attention_weight=AttentionWeight.BF16,
            kv_cache_layout=KvCacheLayout.ATOM,
        )

    def close(self) -> None:
        self.op.close()


class Glm52MonoDecode:
    """Run eligible GLM-5.2 MoE layers while retaining ATOM's external indexer."""

    def __init__(self, causal_lm, atom_config, mode: str) -> None:
        self._lm = causal_lm
        self._atom_config = atom_config
        self._mode = mode
        self._ops: dict[tuple[int, int], _GlmLayerOp] = {}
        self._refused: set[int] = set()
        self._enabled = mode != "off"
        if not self._enabled:
            return

        config = atom_config.hf_config
        checks = (
            (getattr(config, "model_type", None) == "glm_moe_dsa", "model type"),
            (atom_config.tensor_parallel_size == 8, "not TP8"),
            (atom_config.parallel_config.data_parallel_size == 1, "DP"),
            (not atom_config.enable_dp_attention, "DPA"),
            (atom_config.decode_context_parallel_size == 1, "DCP"),
            (atom_config.prefill_context_parallel_size == 1, "PCP"),
            (atom_config.pipeline_parallel_size == 1, "PP"),
            (not atom_config.enable_expert_parallel, "EP"),
            (not atom_config.enable_tbo and not atom_config.enable_tbo_decode, "TBO"),
            (atom_config.speculative_config is None, "MTP/speculative decode"),
            (atom_config.kv_cache_dtype == "bf16", "KV dtype"),
            (not is_plugin_mode(), "plugin mode"),
            (not envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV, "shuffled MLA cache"),
            (is_flat_atom_cache_page_size(envs.ATOM_MLA_PAGE_SIZE), "segmented MLA cache"),
        )
        for ok, why in checks:
            if not ok:
                logger.info("GLM-5.2 MonoKernel off: %s", why)
                self._enabled = False
                return

        expected = {
            "hidden_size": GLM5_CONFIG.hidden,
            "q_lora_rank": GLM5_CONFIG.q_lora,
            "kv_lora_rank": GLM5_CONFIG.kv_lora,
            "qk_rope_head_dim": GLM5_CONFIG.pe_dim,
            "qk_nope_head_dim": GLM5_CONFIG.nope_dim,
            "v_head_dim": GLM5_CONFIG.v_dim,
            "n_routed_experts": GLM5_CONFIG.n_experts,
            "num_experts_per_tok": GLM5_CONFIG.top_k,
            "n_shared_experts": GLM5_CONFIG.num_shared_experts,
            "index_topk": 2048,
            "routed_scaling_factor": GLM5_CONFIG.route_scale,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "rms_norm_eps": EPS,
        }
        for name, value in expected.items():
            if getattr(config, name, None) != value:
                logger.info("GLM-5.2 MonoKernel off: %s=%r", name, getattr(config, name, None))
                self._enabled = False
                return
        if config.moe_intermediate_size // 8 != GLM5_CONFIG.inter:
            logger.info("GLM-5.2 MonoKernel off: expert width")
            self._enabled = False
            return

        layers = list(causal_lm.model.layers[causal_lm.model.start_layer : causal_lm.model.end_layer])
        first_moe = next((i for i, layer in enumerate(layers) if hasattr(layer.mlp, "experts")), len(layers))
        if first_moe == len(layers) or any(not hasattr(layer.mlp, "experts") for layer in layers[first_moe:]):
            logger.info("GLM-5.2 MonoKernel off: MoE layers are not a suffix")
            self._enabled = False
            return
        seen_full_indexer = False
        for layer in layers:
            if layer.self_attn.indexer is not None and not layer.self_attn.skip_topk:
                seen_full_indexer = True
            elif not seen_full_indexer:
                logger.info("GLM-5.2 MonoKernel off: shared IndexShare layer precedes a full layer")
                self._enabled = False
                return

    def _mono_layers(self):
        model = self._lm.model
        return [
            layer
            for layer in model.layers[model.start_layer : model.end_layer]
            if hasattr(layer.mlp, "experts")
        ]

    def _prepare(self, samples: int) -> bool:
        if all((layer.layer_idx, samples) in self._ops for layer in self._mono_layers()):
            return True
        if samples in self._refused or torch.cuda.is_current_stream_capturing():
            return False

        mapped = []
        validation_error = None
        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        try:
            _bf16_vector(
                self._lm.model.norm.weight,
                "model.norm.weight",
                GLM5_CONFIG.hidden,
            )
            for layer in self._mono_layers():
                backend = select_backend(
                    "glm52",
                    self._mode,
                    samples=samples,
                    tp_size=8,
                    kv_cache_dtype="bf16",
                    has_moe=True,
                    external_indexer=True,
                    cache_layout="atom",
                )
                _need(backend == "mono", "layer backend")
                mapped.append((layer, _layer_weights(layer, rank, npes)))
        except (MonoUnsupported, ValueError) as error:
            validation_error = error
        try:
            tp_uniform_local_validation(
                validation_error,
                group=get_tp_group().cpu_group,
                world_size=npes,
                context="GLM-5.2 weight mapping failed",
            )
        except MonoUnsupported as error:
            self._refused.add(samples)
            logger.warning("GLM-5.2 MonoKernel fallback before launch: %s", error)
            return False

        made = []
        try:
            for layer, weights in mapped:
                owned = _GlmLayerOp(weights, samples, self._atom_config.hf_config.index_topk)
                self._ops[layer.layer_idx, samples] = owned
                made.append((layer.layer_idx, samples))
        except (MonoUnsupported, ValueError) as error:
            for key in made:
                self._ops.pop(key).close()
            self._refused.add(samples)
            logger.warning("GLM-5.2 MonoKernel fallback before launch: %s", error)
            return False
        return True

    def supports(self, input_ids, positions, intermediate_tensors, inputs_embeds) -> bool:
        if not self._enabled or intermediate_tensors is not None:
            return False
        if self._lm.model.aux_hidden_state_layers:
            return False
        samples = input_ids.numel()
        if select_backend(
            "glm52",
            self._mode,
            samples=samples,
            tp_size=self._atom_config.tensor_parallel_size,
            kv_cache_dtype=self._atom_config.kv_cache_dtype,
            mtp=self._atom_config.speculative_config is not None,
            dpa=self._atom_config.enable_dp_attention,
            dcp=self._atom_config.decode_context_parallel_size > 1,
            plugin=is_plugin_mode(),
        ) != "mono":
            return False
        if positions.dtype is not torch.int64 or positions.numel() != samples or not positions.is_contiguous():
            return False
        if inputs_embeds is not None and (
            inputs_embeds.shape != (samples, GLM5_CONFIG.hidden)
            or inputs_embeds.dtype is not torch.bfloat16
            or not inputs_embeds.is_contiguous()
        ):
            return False

        fwd = get_forward_context()
        context = fwd.context
        metadata = fwd.attn_metadata
        if (
            context is None
            or metadata is None
            or context.is_prefill
            or fwd.ubatch_slices is not None
            or metadata.max_seqlen_q != 1
            or metadata.slot_mapping.dtype is not torch.int64
            or metadata.slot_mapping.numel() < samples
            or not metadata.slot_mapping.is_contiguous()
            or metadata.sparse_kv_indptr.dtype is not torch.int32
            or metadata.sparse_kv_indptr.numel() < samples + 1
            or not metadata.sparse_kv_indptr.is_contiguous()
        ):
            return False

        try:
            shared = _shared_sparse_buffer(self._lm.model.layers)
            _need(shared.numel() >= samples, "sparse index capacity")
            for layer in self._mono_layers():
                cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"].k_cache
                _need(
                    cache.dtype is torch.bfloat16
                    and cache.is_contiguous()
                    and cache.shape[-1] == GLM5_CONFIG.kv_lora + GLM5_CONFIG.pe_dim,
                    f"layer {layer.layer_idx} fused BF16 cache",
                )
        except (KeyError, MonoUnsupported):
            return False
        return self._prepare(samples)

    @staticmethod
    def _refresh_indexer(layer, state: torch.Tensor, positions: torch.Tensor) -> None:
        attn = layer.self_attn
        indexer = attn.indexer
        if indexer is None or attn.skip_topk:
            return
        normalized = rmsnorm2d_fwd_(
            state,
            layer.input_layernorm.weight,
            layer.input_layernorm.eps,
            GLM5_CONFIG.hidden,
        )
        qkv = attn.fused_qkv_a_proj(normalized)
        q_c = qkv[:, : GLM5_CONFIG.q_lora]
        qr = rmsnorm2d_fwd_(
            q_c,
            attn.q_a_layernorm.weight,
            attn.q_a_layernorm.eps,
            GLM5_CONFIG.q_lora,
        )
        indexer(normalized, qr, None, positions, attn.indexer_rope_emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self._lm.model
        fwd = get_forward_context()
        metadata = fwd.attn_metadata
        samples = input_ids.numel()
        hidden = model.get_input_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        shared_indices = _shared_sparse_buffer(model.layers)
        slot_mapping = metadata.slot_mapping[:samples]
        sparse_indptr = metadata.sparse_kv_indptr[: samples + 1]

        for layer in model.layers[model.start_layer : model.end_layer]:
            owned = self._ops.get((layer.layer_idx, samples))
            if owned is None:
                hidden, residual = layer(positions, hidden, residual)
                continue
            state = hidden if residual is None else hidden + residual
            self._refresh_indexer(layer, state, positions)
            attn = layer.self_attn
            cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"].k_cache
            hidden = owned.op.forward(
                state,
                positions,
                cache,
                cache,
                shared_indices,
                attn.rotary_emb.cos_cache,
                attn.rotary_emb.sin_cache,
                layer=0,
                positions=positions,
                slot_mapping=slot_mapping,
                sparse_kv_indptr=sparse_indptr,
            )
            residual = None
        state = hidden if residual is None else hidden + residual
        return rmsnorm2d_fwd_(
            state,
            model.norm.weight,
            model.norm.eps,
            GLM5_CONFIG.hidden,
        )

    def close(self) -> None:
        for owned in self._ops.values():
            owned.close()
        self._ops.clear()
        self._refused.clear()
