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
from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.model_ops.monokernel.abi import AgenticDecodeShape
from atom.model_ops.monokernel.config import (
    EPS,
    GLM5_CONFIG,
    AttentionWeight,
    KvCacheLayout,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    glm5_shard_config,
)
from atom.model_ops.monokernel.dispatch import (
    MonoUnsupported,
    is_flat_atom_cache_page_size,
    select_backend,
    tp_uniform_local_validation,
)
from atom.model_ops.monokernel.glm.layout import (
    INDEX_KEYS_PER_TASK,
    INDEX_MAX_LOGICAL_CONTEXT,
)
from atom.model_ops.monokernel.telemetry import MonoRouteStats
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    atom_mxfp4_storage_view,
    linear_bf16,
    linear_fp8_block128,
    quantize_fp8_blocks,
    quantize_fp8_block128,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")
GLM52_AGENTIC_BATCH_LADDERS = {
    6: (1, 2, 4, 8, 12, 16),
    5: (1, 2, 4, 8, 12, 16, 20),
}
GLM52_AGENTIC_BUCKETS = tuple(
    (batch, query_len)
    for query_len, batches in GLM52_AGENTIC_BATCH_LADDERS.items()
    for batch in batches
)
GLM52_NUM_LAYERS = 78
GLM52_FIRST_MOE_LAYER = 3
GLM52_FIRST_FULL_LAYER = 6


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _agentic_index_max_seq(atom_config) -> int:
    max_seq = int(getattr(atom_config, "max_model_len", 0))
    _need(max_seq > 0, "max_model_len must be positive")
    _need(
        max_seq <= INDEX_MAX_LOGICAL_CONTEXT,
        f"max_model_len exceeds Agentic cap {INDEX_MAX_LOGICAL_CONTEXT}",
    )
    _need(
        max_seq % INDEX_KEYS_PER_TASK == 0,
        f"max_model_len must align to {INDEX_KEYS_PER_TASK} index keys",
    )
    return max_seq


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


def _physical_expert_count(experts) -> int:
    _need(experts.global_num_experts == GLM5_CONFIG.n_experts, "routed expert count")
    _need(
        experts.num_fused_shared_experts == GLM5_CONFIG.num_shared_experts,
        "fused shared-expert count",
    )
    return GLM5_CONFIG.n_experts + GLM5_CONFIG.num_shared_experts


def _mxfp4_expert_tensors(experts, config, physical_experts: int) -> dict[str, torch.Tensor]:
    tensors = {
        "w_ug": _atom_byte_view(experts.w13_weight),
        "s_ug": experts.w13_weight_scale.view(torch.uint8),
        "w_dn": _atom_byte_view(experts.w2_weight),
        "s_dn": experts.w2_weight_scale.view(torch.uint8),
    }
    for name, logical_rows, logical_k, scale in (
        ("w_ug", physical_experts * 2 * config.inter, config.hidden, False),
        ("s_ug", physical_experts * 2 * config.inter, config.hidden, True),
        ("w_dn", physical_experts * config.hidden, config.inter, False),
        ("s_dn", physical_experts * config.hidden, config.inter, True),
    ):
        atom_mxfp4_storage_view(
            tensors[name],
            name=name,
            logical_rows=logical_rows,
            logical_k=logical_k,
            scale=scale,
        )
    return tensors


def _layer_weights(layer, rank: int, npes: int) -> LayerWeights:
    cfg = glm5_shard_config(npes)
    attn = layer.self_attn
    moe = layer.mlp
    experts = moe.experts
    physical_experts = _physical_expert_count(experts)
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

    expert_tensors = _mxfp4_expert_tensors(experts, cfg, physical_experts)

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
        **expert_tensors,
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


def _agentic_layer_weights(layer, rank: int, npes: int) -> LayerWeights:
    """Map one production TP4 layer without reinterpreting quant scales."""

    cfg = glm5_shard_config(npes)
    _need(npes == 4, "Agentic full-layer path requires TP4")
    attn = layer.self_attn
    base = _layer_weights(layer, rank, npes)
    tensors = dict(base.t)
    for name, linear, rows, cols in (
        ("qkv_a", attn.fused_qkv_a_proj, cfg.qkv_a_rows, cfg.hidden),
        (
            "q_b",
            attn.q_b_proj,
            cfg.local_heads * (cfg.nope_dim + cfg.pe_dim),
            cfg.q_lora,
        ),
        ("o", attn.o_proj, cfg.hidden, cfg.local_heads * cfg.v_dim),
    ):
        weight, scale = linear_fp8_block128(
            linear,
            name=f"{name}_proj",
            logical_rows=rows,
            logical_cols=cols,
        )
        tensors[f"w_{name}"] = weight
        tensors[f"s_{name}"] = scale

    kv_b = linear_bf16(
        attn.kv_b_proj,
        name="kv_b_proj",
        logical_rows=cfg.local_heads * (cfg.nope_dim + cfg.v_dim),
        logical_cols=cfg.kv_lora,
    )
    uk, uv = _split_kv_b(
        kv_b,
        heads=cfg.local_heads,
        nope_dim=cfg.nope_dim,
        value_dim=cfg.v_dim,
        kv_lora=cfg.kv_lora,
    )
    tensors["w_uk"], tensors["s_uk"] = quantize_fp8_blocks(uk, block_k=64)
    tensors["w_uv"], tensors["s_uv"] = quantize_fp8_block128(uv)

    indexer = attn.indexer
    if indexer is not None and not attn.skip_topk:
        tensors["w_index_q"], tensors["s_index_q"] = linear_fp8_block128(
            indexer.wq_b,
            name="indexer.wq_b",
            logical_rows=32 * 128,
            logical_cols=cfg.q_lora,
        )
        if indexer.use_wk_weights_proj_fusion:
            fused = indexer.wk_weights_proj.weight
            _need(
                fused.dtype is torch.bfloat16
                and fused.shape == (128 + 32, cfg.hidden)
                and fused.is_contiguous(),
                "fused indexer wk/weights_proj layout",
            )
            index_k = fused[:128]
            index_w = fused[128:]
        else:
            index_k = linear_bf16(
                indexer.wk,
                name="indexer.wk",
                logical_rows=128,
                logical_cols=cfg.hidden,
            )
            index_w = linear_bf16(
                indexer.weights_proj,
                name="indexer.weights_proj",
                logical_rows=32,
                logical_cols=cfg.hidden,
            )
        tensors["w_index_k"], tensors["s_index_k"] = quantize_fp8_block128(
            index_k
        )
        tensors["w_index_w"] = index_w
        _need(
            indexer.k_norm.weight.dtype is torch.float32
            and indexer.k_norm.bias.dtype is torch.float32,
            "indexer norm parameters must be FP32",
        )
        tensors["g_index_k"] = indexer.k_norm.weight
        tensors["b_index_k"] = indexer.k_norm.bias
    return LayerWeights(
        cfg.local_heads,
        tensors,
        cfg,
        rank,
        npes,
        mxfp4_weight_layout=base.mxfp4_weight_layout,
        mxfp4_scale_layout=base.mxfp4_scale_layout,
        physical_experts=base.physical_experts,
    )


def _staged_moe_weights(moe, rank: int, npes: int) -> LayerWeights:
    cfg = glm5_shard_config(npes)
    experts = moe.experts
    physical_experts = _physical_expert_count(experts)
    _need(npes == 4, "staged MoE requires TP4")
    _need(not experts.use_ep, "expert parallelism")
    _need(not experts.quant_method.is_guinterleave, "ATOM_MOE_GU_ITLV must be 0")
    _need(experts.intermediate_size_per_partition == cfg.inter, "expert width")

    expert_tensors = _mxfp4_expert_tensors(experts, cfg, physical_experts)
    return LayerWeights(
        cfg.local_heads,
        expert_tensors,
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


def _agentic_index_modes(causal_lm, config=None) -> tuple[object, ...]:
    """Validate the global 78-layer plan, including the external prefix."""

    if config is None:
        config = causal_lm.config
    pattern = getattr(config, "indexer_types", None)
    _need(
        pattern is not None and len(pattern) == GLM52_NUM_LAYERS,
        "indexer_types must describe all 78 layers",
    )
    layers = list(causal_lm.model.layers)
    _need(len(layers) == GLM52_NUM_LAYERS, "GLM-5.2 must expose all 78 layers")
    _need(
        all(not hasattr(layer.mlp, "experts") for layer in layers[:3])
        and all(hasattr(layer.mlp, "experts") for layer in layers[3:]),
        "dense-prefix/MoE layer boundary",
    )
    from atom.model_ops.monokernel.glm.index_share import (
        GlmIndexShareMode,
        GlmIndexSharePlan,
    )

    modes = tuple(
        GlmIndexShareMode.FULL
        if str(value).lower() in ("full", "f")
        else GlmIndexShareMode.SHARED
        if str(value).lower() in ("shared", "s")
        else (_ for _ in ()).throw(MonoUnsupported(f"invalid indexer_types[{i}]={value!r}"))
        for i, value in enumerate(pattern[GLM52_FIRST_MOE_LAYER:], GLM52_FIRST_MOE_LAYER)
    )
    _need(
        modes[:3] == (GlmIndexShareMode.SHARED,) * 3
        and modes[3] is GlmIndexShareMode.FULL,
        "layers 3-5 must share layer 2; layer 6 must be FULL",
    )
    GlmIndexSharePlan.from_runtime_pattern(
        tuple("F" if mode is GlmIndexShareMode.FULL else "S" for mode in modes),
        allow_external_prefix=True,
    )
    return modes


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


class _GlmMoeBinding:
    def __init__(self, owner, layer, key: str, original) -> None:
        self.owner = owner
        self.layer = layer
        self.key = key
        self.original = original

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.owner._staged_moe_forward(self, hidden_states)


class Glm52MonoDecode:
    """Run eligible GLM-5.2 MoE layers while retaining ATOM's external indexer."""

    def __init__(self, causal_lm, atom_config, mode: str) -> None:
        self._lm = causal_lm
        self._atom_config = atom_config
        self._mode = mode
        self._ops: dict[tuple[int, int], _GlmLayerOp] = {}
        self._refused: set[int] = set()
        self._staged_bindings: list[_GlmMoeBinding] = []
        self._staged_workspaces: dict[int, object] = {}
        self._staged_ops: dict[tuple[int, int], object] = {}
        self._staged_refused: set[int] = set()
        self._staged_enabled = False
        self._agentic_modes: tuple[object, ...] = ()
        self._agentic_weights: dict[int, LayerWeights] = {}
        self._agentic_artifacts: dict[int, object] = {}
        self._agentic_buckets: dict[tuple[int, int], object] = {}
        self._agentic_ready_q: set[int] = set()
        self._agentic_ready = False
        self._agentic_refused = False
        self._stats = MonoRouteStats("glm52")
        self._enabled = mode != "off"
        if not self._enabled:
            return

        config = atom_config.hf_config
        speculative = atom_config.speculative_config
        agentic_checks = (
            mode in ("auto", "mono"),
            getattr(config, "model_type", None) == "glm_moe_dsa",
            atom_config.tensor_parallel_size == 4,
            atom_config.parallel_config.data_parallel_size == 1,
            not atom_config.enable_dp_attention,
            atom_config.decode_context_parallel_size == 1,
            atom_config.prefill_context_parallel_size == 1,
            atom_config.pipeline_parallel_size == 1,
            not atom_config.enable_expert_parallel,
            not atom_config.enable_tbo and not atom_config.enable_tbo_decode,
            speculative is not None and speculative.method == "mtp",
            atom_config.kv_cache_dtype == "fp8",
            getattr(atom_config, "index_cache_dtype", None) == "fp8",
            not is_plugin_mode(),
            is_flat_atom_cache_page_size(envs.ATOM_MLA_PAGE_SIZE),
        )
        if all(agentic_checks):
            try:
                for name, value in {
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
                }.items():
                    _need(getattr(config, name, None) == value, f"{name} mismatch")
                _need(
                    config.moe_intermediate_size // 4
                    == glm5_shard_config(4).inter,
                    "TP4 expert width",
                )
                self._agentic_modes = _agentic_index_modes(causal_lm, config)
            except (AttributeError, MonoUnsupported, ValueError) as error:
                logger.warning("GLM-5.2 full Agentic path off: %s", error)
            else:
                logger.info(
                    "GLM-5.2 full Agentic candidate: TP4 layers=%d",
                    len(self._agentic_modes),
                )
                return

        staged_checks = (
            mode == "staged",
            getattr(config, "model_type", None) == "glm_moe_dsa",
            atom_config.tensor_parallel_size == 4,
            atom_config.parallel_config.data_parallel_size == 1,
            not atom_config.enable_dp_attention,
            atom_config.prefill_context_parallel_size == 1,
            atom_config.pipeline_parallel_size == 1,
            not atom_config.enable_expert_parallel,
            not atom_config.enable_tbo and not atom_config.enable_tbo_decode,
            atom_config.kv_cache_dtype == "fp8",
            not is_plugin_mode(),
        )
        if all(staged_checks):
            install_error = None
            try:
                self._install_staged_moe()
            except (AttributeError, MonoUnsupported, RuntimeError, ValueError) as error:
                install_error = error
            try:
                tp_uniform_local_validation(
                    install_error,
                    group=get_tp_group().cpu_group,
                    world_size=get_tensor_model_parallel_world_size(),
                    context="GLM-5.2 staged MoE installation failed",
                )
            except MonoUnsupported as error:
                self._uninstall_staged_moe()
                logger.warning("GLM-5.2 staged MoE off: %s", error)
            if install_error is None and self._staged_bindings:
                self._staged_enabled = True
                self._enabled = False
                logger.info(
                    "GLM-5.2 staged MoE installed: TP4 layers=%d",
                    len(self._staged_bindings),
                )
                return

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

    def _install_staged_moe(self) -> None:
        from atom.model_ops.monokernel.glm.staged_moe import (
            install_staged_moe_forward,
        )

        context = self._atom_config.compilation_config.static_forward_context
        layers = self._mono_layers()
        _need(bool(layers), "no MoE layers")
        for layer in layers:
            _physical_expert_count(layer.mlp.experts)
        try:
            for layer in layers:
                key = f"glm52_staged_moe.layer_{layer.layer_idx}"
                _need(key not in context, f"duplicate staged key {key}")
                original = install_staged_moe_forward(layer.mlp, key)
                binding = _GlmMoeBinding(self, layer, key, original)
                context[key] = binding
                self._staged_bindings.append(binding)
        except Exception:
            for binding in self._staged_bindings:
                binding.layer.mlp.forward = binding.original
                if context.get(binding.key) is binding:
                    context.pop(binding.key)
            self._staged_bindings.clear()
            raise

    def _uninstall_staged_moe(self) -> None:
        context = self._atom_config.compilation_config.static_forward_context
        for binding in self._staged_bindings:
            binding.layer.mlp.forward = binding.original
            if context.get(binding.key) is binding:
                context.pop(binding.key)
        self._staged_bindings.clear()

    def _prepare_staged_moe(self, rows: int) -> bool:
        if all((binding.layer.layer_idx, rows) in self._staged_ops for binding in self._staged_bindings):
            return True
        if rows in self._staged_refused or torch.cuda.is_current_stream_capturing():
            return False

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        mapped = []
        error = None
        try:
            mapped = [
                (binding, _staged_moe_weights(binding.layer.mlp, rank, npes))
                for binding in self._staged_bindings
            ]
        except (AttributeError, MonoUnsupported, RuntimeError, ValueError) as caught:
            error = caught
        try:
            tp_uniform_local_validation(
                error,
                group=get_tp_group().cpu_group,
                world_size=npes,
                context="GLM-5.2 staged MoE weight mapping failed",
            )
        except MonoUnsupported as caught:
            self._staged_refused.add(rows)
            logger.warning("GLM-5.2 staged MoE fallback before launch: %s", caught)
            return False

        workspace = None
        stages = {}
        construction_error = None
        try:
            from atom.model_ops.monokernel.glm.staged_moe import (
                Glm52MoeWorkspace,
                Glm52Tp4MoeStage,
            )

            workspace = Glm52MoeWorkspace(
                rows,
                torch.device("cuda", torch.cuda.current_device()),
            )
            stages = {
                (binding.layer.layer_idx, rows): Glm52Tp4MoeStage(
                    weights,
                    binding.layer.mlp.gate,
                    binding.layer.mlp.gate.e_score_correction_bias,
                    workspace,
                    binding.layer.mlp.experts,
                    reduce_results=binding.layer.mlp.reduce_results,
                )
                for binding, weights in mapped
            }
        except (AttributeError, RuntimeError, ValueError) as caught:
            construction_error = caught
        try:
            tp_uniform_local_validation(
                construction_error,
                group=get_tp_group().cpu_group,
                world_size=npes,
                context="GLM-5.2 staged MoE construction failed",
            )
        except MonoUnsupported as caught:
            self._staged_refused.add(rows)
            logger.warning("GLM-5.2 staged MoE fallback before launch: %s", caught)
            return False

        assert workspace is not None
        self._staged_workspaces[rows] = workspace
        self._staged_ops.update(stages)
        logger.info(
            "GLM-5.2 MonoKernel ready: backend=staged_moe rows=%d layers=%d",
            rows,
            len(stages),
        )
        return True

    def _staged_moe_forward(
        self,
        binding: _GlmMoeBinding,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        rows = hidden_states.shape[0]
        fwd = get_forward_context()
        context = fwd.context
        shape = None
        has_agentic_shape = context is not None and hasattr(context, "running_bs")
        if has_agentic_shape:
            try:
                shape = AgenticDecodeShape.from_forward_mode(
                    context,
                    batch_capacity=context.running_bs,
                    row_capacity=max(
                        rows,
                        context.running_bs * context.max_seqlen_q,
                    ),
                )
            except ValueError:
                shape = None
        backend = select_backend(
            "glm52",
            self._mode,
            samples=rows,
            tp_size=self._atom_config.tensor_parallel_size,
            kv_cache_dtype=self._atom_config.kv_cache_dtype,
            mtp=self._atom_config.speculative_config is not None,
            dpa=self._atom_config.enable_dp_attention,
            dcp=self._atom_config.decode_context_parallel_size > 1,
            plugin=is_plugin_mode(),
            segment="moe",
        )
        if (
            backend != "staged_moe"
            or context is None
            or (has_agentic_shape and shape is None)
            or (shape is not None and shape.actual_rows != rows)
            or context.is_prefill
            or fwd.ubatch_slices is not None
            or hidden_states.ndim != 2
            or hidden_states.shape[1] != GLM5_CONFIG.hidden
            or hidden_states.dtype is not torch.bfloat16
            or not hidden_states.is_contiguous()
            or not self._prepare_staged_moe(rows)
        ):
            self._route_stats().record_fallback("staged_moe", rows)
            return binding.original(hidden_states)
        self._route_stats().record_hit(backend, rows)
        return self._staged_ops[binding.layer.layer_idx, rows](hidden_states)

    def _route_stats(self) -> MonoRouteStats:
        stats = getattr(self, "_stats", None)
        if stats is None:
            stats = MonoRouteStats("glm52")
            self._stats = stats
        return stats

    def route_stats(self) -> dict[str, object]:
        return self._route_stats().snapshot()

    def _fallback(self, reason: str, samples: int) -> bool:
        self._route_stats().record_fallback(reason, samples)
        return False

    def _mono_layers(self):
        model = self._lm.model
        return [
            layer
            for layer in model.layers[model.start_layer : model.end_layer]
            if hasattr(layer.mlp, "experts")
        ]

    def _close_agentic_buckets(self, buckets=None) -> None:
        owned = self._agentic_buckets if buckets is None else buckets
        for key in sorted(tuple(owned), reverse=True):
            try:
                owned.pop(key).close()
            except Exception:
                logger.exception("failed closing GLM Agentic bucket %s", key)
        if buckets is None:
            self._agentic_ready = False
            getattr(self, "_agentic_ready_q", set()).clear()

    def _agentic_bucket_keys(self, query_len: int) -> tuple[tuple[int, int], ...]:
        allowed = GLM52_AGENTIC_BATCH_LADDERS.get(query_len)
        _need(allowed is not None, f"unsupported Agentic query length {query_len}")
        configured = tuple(sorted(set(self._atom_config.capture_sizes)))
        _need(configured, "empty CUDAGraph capture ladder")
        _need(
            all(batch in allowed for batch in configured),
            f"capture ladder {configured} is unsupported for q={query_len}",
        )
        return tuple((batch, query_len) for batch in configured)

    @staticmethod
    def _runtime_query_len(context, metadata) -> int:
        query_len = int(getattr(metadata, "max_seqlen_q", 0))
        mode = getattr(context, "forward_mode", None)
        if mode is not None and int(mode.max_seqlen_q) != query_len:
            return 0
        return query_len

    def _agentic_specs(self, fwd):
        from atom.model_ops.monokernel.glm.graph import (
            GlmAgenticLayerInputs,
            GlmAgenticLayerSpec,
        )

        layers = list(self._lm.model.layers)
        external = _shared_sparse_buffer(layers[2:])
        specs = []
        index_max_seq = _agentic_index_max_seq(self._atom_config)
        for layer, mode in zip(layers[3:], self._agentic_modes):
            cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"]
            main = cache.k_cache
            _need(
                main.dtype is torch.float8_e4m3fn
                and main.is_contiguous()
                and main.ndim == 3
                and main.shape[1:] == (1, 576),
                f"layer {layer.layer_idx} main FP8 cache must be [slots,1,576]",
            )
            slots = main.shape[0]
            _need(slots > 0, "cache slot capacity must be positive")
            full = str(getattr(mode, "value", mode)) == "full"
            index_cache = cache.index_cache if full else None
            if full:
                _need(
                    index_cache is not None
                    and index_cache.dtype is torch.float8_e4m3fn
                    and index_cache.is_contiguous()
                    and index_cache.ndim == 3
                    and index_cache.shape[1:] == (16, 144)
                    and index_cache.shape[0] * 16 == slots,
                    f"layer {layer.layer_idx} index cache must be [blocks,16,144]",
                )
            impl = _attention_impl(layer)
            scale = getattr(impl, "_k_scale_device", None)
            _need(
                scale is not None
                and scale.dtype is torch.float32
                and scale.numel() == 1
                and scale.is_contiguous(),
                f"layer {layer.layer_idx} main cache scalar descale",
            )
            inputs = GlmAgenticLayerInputs(
                kv_cache=main,
                pe_cache=None,
                indices=external,
                cos=layer.self_attn.rotary_emb.cos_cache,
                sin=layer.self_attn.rotary_emb.sin_cache,
                index_cache=index_cache,
                kv_cache_scale=scale,
            )
            specs.append(
                GlmAgenticLayerSpec(
                    self._agentic_weights[layer.layer_idx],
                    inputs,
                    with_indexer=full,
                    index_max_seq=index_max_seq,
                    attention_weight=AttentionWeight.FP8_BLOCK128,
                    kv_cache_layout=KvCacheLayout.ATOM_FP8,
                    index_share_mode=mode,
                )
            )
        return tuple(specs), index_max_seq

    def _prepare_agentic(self, fwd, query_len: int) -> bool:
        if query_len in self._agentic_ready_q:
            return True
        if self._agentic_refused or torch.cuda.is_current_stream_capturing():
            return False
        from atom.model_ops.monokernel.glm.graph import GlmAgenticGraphBucket
        from atom.model_ops.monokernel.glm.op import Glm5PackedArtifacts
        from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
        from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        group = get_tp_group().cpu_group
        local_error = None
        try:
            keys = self._agentic_bucket_keys(query_len)
            if not self._agentic_weights:
                for layer in self._lm.model.layers[3:]:
                    weights = _agentic_layer_weights(layer, rank, npes)
                    self._agentic_weights[layer.layer_idx] = weights
                    full = self._agentic_modes[layer.layer_idx - 3].value == "full"
                    self._agentic_artifacts[layer.layer_idx] = (
                        Glm5PackedArtifacts.pack(
                            weights,
                            npes=npes,
                            attention_weight=AttentionWeight.FP8_BLOCK128,
                            with_indexer=full,
                        )
                    )
            specs, index_max_seq = self._agentic_specs(fwd)
        except (AttributeError, KeyError, MonoUnsupported, RuntimeError, ValueError) as error:
            local_error = error
        try:
            tp_uniform_local_validation(
                local_error,
                group=group,
                world_size=npes,
                context="GLM-5.2 Agentic mapping failed",
            )
        except MonoUnsupported as error:
            self._close_agentic_buckets()
            self._agentic_refused = True
            self._agentic_weights.clear()
            self._agentic_artifacts.clear()
            logger.warning("GLM-5.2 Agentic fallback before allocation: %s", error)
            return False

        workspaces = {}
        assert local_error is None
        try:
            for batch_capacity, bucket_q in keys:
                shape = GlmAgenticShape.for_graph(
                    batch_capacity=batch_capacity,
                    query_len=bucket_q,
                    dcp_size=1,
                    query_replication=False,
                )
                workspace = GlmAgenticWorkspace.allocate_local(
                    shape,
                    npes=npes,
                    sparse_attention_topk=self._atom_config.hf_config.index_topk,
                    with_indexer=True,
                    index_max_seq=index_max_seq,
                )
                workspace.ensure_index_share(self._atom_config.hf_config.index_topk)
                workspaces[(batch_capacity, bucket_q)] = workspace
        except (AttributeError, RuntimeError, ValueError) as caught:
            local_error = caught
        try:
            tp_uniform_local_validation(
                local_error,
                group=group,
                world_size=npes,
                context=f"GLM-5.2 Agentic q={query_len} local allocation failed",
            )
        except MonoUnsupported as caught:
            for workspace in workspaces.values():
                workspace.close()
            self._close_agentic_buckets()
            self._agentic_refused = True
            self._agentic_weights.clear()
            self._agentic_artifacts.clear()
            logger.warning("GLM-5.2 Agentic rank-atomic allocation fallback: %s", caught)
            return False

        for key in keys:
            error = None
            try:
                workspaces[key].initialize_collective(
                    rank=rank,
                    npes=npes,
                    group=group,
                )
            except (AttributeError, RuntimeError, ValueError) as caught:
                error = caught
            try:
                tp_uniform_local_validation(
                    error,
                    group=group,
                    world_size=npes,
                    context=f"GLM-5.2 Agentic symmetric bucket {key} failed",
                )
            except MonoUnsupported as caught:
                for workspace in workspaces.values():
                    workspace.close()
                self._close_agentic_buckets()
                self._agentic_refused = True
                self._agentic_weights.clear()
                self._agentic_artifacts.clear()
                logger.warning("GLM-5.2 Agentic symmetric fallback: %s", caught)
                return False

        by_weight = {
            id(weights): self._agentic_artifacts[layer_idx]
            for layer_idx, weights in self._agentic_weights.items()
        }

        def artifacts(weights, **_kwargs):
            return by_weight[id(weights)]

        built = {}
        local_error = None
        try:
            for key in keys:
                built[key] = GlmAgenticGraphBucket.build(
                    workspaces[key],
                    specs,
                    rank=rank,
                    npes=npes,
                    group=group,
                    topk=self._atom_config.hf_config.index_topk,
                    artifact_factory=artifacts,
                )
        except (
            AttributeError,
            KeyError,
            MonoUnsupported,
            RuntimeError,
            ValueError,
        ) as caught:
            local_error = caught
        try:
            tp_uniform_local_validation(
                local_error,
                group=group,
                world_size=npes,
                context=f"GLM-5.2 Agentic q={query_len} construction failed",
            )
        except MonoUnsupported as caught:
            for key, workspace in workspaces.items():
                if key in built:
                    built[key].close()
                else:
                    workspace.close()
            self._close_agentic_buckets()
            self._agentic_refused = True
            self._agentic_weights.clear()
            self._agentic_artifacts.clear()
            logger.warning("GLM-5.2 Agentic all-bucket fallback: %s", caught)
            return False

        self._agentic_buckets.update(built)
        self._agentic_ready_q.add(query_len)
        self._agentic_ready = True
        logger.info(
            "GLM-5.2 Agentic ready: q=%d buckets=%s layers=%d",
            query_len,
            keys,
            len(self._agentic_modes),
        )
        return True

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
        except (AttributeError, MonoUnsupported, RuntimeError, ValueError) as error:
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
        except (AttributeError, MonoUnsupported, RuntimeError, ValueError) as error:
            for key in made:
                self._ops.pop(key).close()
            self._refused.add(samples)
            logger.warning("GLM-5.2 MonoKernel fallback before launch: %s", error)
            return False
        logger.info(
            "GLM-5.2 MonoKernel ready: backend=mono S=%d layers=%d",
            samples,
            len(mapped),
        )
        return True

    def supports(self, input_ids, positions, intermediate_tensors, inputs_embeds) -> bool:
        samples = input_ids.numel()
        if getattr(self, "_staged_enabled", False):
            return False
        self._route_stats().record_attempt(samples)
        if not self._enabled:
            return self._fallback("disabled", samples)
        if getattr(self, "_agentic_modes", ()):
            fwd = get_forward_context()
            context = fwd.context
            metadata = fwd.attn_metadata
            query_len = (
                0
                if context is None or metadata is None
                else self._runtime_query_len(context, metadata)
            )
            key = (
                getattr(context, "running_bs", 0),
                query_len,
            )
            try:
                configured_keys = self._agentic_bucket_keys(query_len)
            except MonoUnsupported:
                configured_keys = ()
            if (
                select_backend(
                    "glm52",
                    self._mode,
                    samples=samples,
                    tp_size=self._atom_config.tensor_parallel_size,
                    kv_cache_dtype=self._atom_config.kv_cache_dtype,
                    mtp=True,
                    dpa=False,
                    dcp=False,
                    plugin=False,
                    external_indexer=False,
                    cache_layout="atom_fp8",
                    segment="agentic_layer",
                )
                != "agentic_full"
                or key not in configured_keys
                or samples != key[0] * key[1]
                or intermediate_tensors is not None
                or self._lm.model.aux_hidden_state_layers
                or (
                    inputs_embeds is not None
                    and (
                        inputs_embeds.shape != (samples, GLM5_CONFIG.hidden)
                        or inputs_embeds.dtype is not torch.bfloat16
                        or not inputs_embeds.is_contiguous()
                    )
                )
                or context is None
                or metadata is None
                or context.is_prefill
                or fwd.ubatch_slices is not None
                or positions.dtype is not torch.int64
                or positions.numel() != samples
                or not positions.is_contiguous()
                or metadata.batch_id_per_q_token is None
                or metadata.batch_id_per_q_token.dtype is not torch.int32
                or metadata.batch_id_per_q_token.numel() < samples
                or metadata.block_tables is None
                or metadata.context_lens is None
                or metadata.sparse_kv_indptr is None
                or metadata.sparse_kv_indptr.numel() < samples + 1
                or not self._prepare_agentic(fwd, query_len)
                or not self._agentic_ready
                or any(bucket_key not in self._agentic_buckets for bucket_key in configured_keys)
            ):
                return self._fallback("agentic_full", samples)
            metadata.glm_agentic_owned_counts = self._agentic_buckets[
                key
            ].workspace.publish_owned_counts(metadata.sparse_kv_indptr)
            return True
        if intermediate_tensors is not None:
            return self._fallback("pipeline", samples)
        if self._lm.model.aux_hidden_state_layers:
            return self._fallback("aux_hidden_states", samples)
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
            return self._fallback("dispatch", samples)
        if positions.dtype is not torch.int64 or positions.numel() != samples or not positions.is_contiguous():
            return self._fallback("positions", samples)
        if inputs_embeds is not None and (
            inputs_embeds.shape != (samples, GLM5_CONFIG.hidden)
            or inputs_embeds.dtype is not torch.bfloat16
            or not inputs_embeds.is_contiguous()
        ):
            return self._fallback("inputs_embeds", samples)

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
            return self._fallback("metadata", samples)

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
            return self._fallback("cache_layout", samples)
        if not self._prepare(samples):
            return self._fallback("prepare", samples)
        return True

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
        if getattr(self, "_agentic_modes", ()):
            return self._forward_agentic(input_ids, positions, inputs_embeds)
        model = self._lm.model
        fwd = get_forward_context()
        metadata = fwd.attn_metadata
        samples = input_ids.numel()
        self._route_stats().record_hit("mono", samples)
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

    def _forward_agentic(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        model = self._lm.model
        fwd = get_forward_context()
        metadata = fwd.attn_metadata
        key = (
            fwd.context.running_bs,
            self._runtime_query_len(fwd.context, metadata),
        )
        bucket = self._agentic_buckets[key]
        hidden = (
            model.get_input_embeddings(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        residual = None
        for layer in model.layers[:GLM52_FIRST_MOE_LAYER]:
            hidden, residual = layer(positions, hidden, residual)
        dense_mlp = model.layers[GLM52_FIRST_MOE_LAYER - 1].mlp
        reduce_results = getattr(
            dense_mlp,
            "reduce_results",
            getattr(getattr(dense_mlp, "down_proj", None), "reduce_results", True),
        )
        if not reduce_results:
            hidden = tensor_model_parallel_all_reduce(hidden)
        state = hidden if residual is None else hidden + residual
        output = bucket(
            state,
            positions=positions,
            slot_mapping=metadata.slot_mapping,
            sparse_kv_indptr=metadata.sparse_kv_indptr,
            batch_ids=metadata.batch_id_per_q_token,
            owned_counts=metadata.glm_agentic_owned_counts,
            block_tables=metadata.block_tables,
            context_lens=metadata.context_lens,
        )
        self._route_stats().record_hit("agentic_full", input_ids.numel())
        return rmsnorm2d_fwd_(
            output,
            model.norm.weight,
            model.norm.eps,
            GLM5_CONFIG.hidden,
        )

    def close(self) -> None:
        if hasattr(self, "_agentic_buckets"):
            self._close_agentic_buckets()
            self._agentic_weights.clear()
            self._agentic_artifacts.clear()
        self._uninstall_staged_moe()
        self._staged_ops.clear()
        self._staged_workspaces.clear()
        self._staged_refused.clear()
        for owned in self._ops.values():
            owned.close()
        self._ops.clear()
        self._refused.clear()
