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
from atom.model_ops.monokernel.dispatch import (
    MonoUnsupported,
    select_backend,
    tp_uniform_local_validation,
)
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    atom_mxfp4_storage_view,
    linear_bf16,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


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
    def __init__(self, layer, weights: LayerWeights, samples: int, mode: str) -> None:
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
        )

    def close(self) -> None:
        self.op.close()


class KimiMonoDecode:
    """Run eligible KDA+MoE layers natively and preserve every fallback layer."""

    def __init__(self, causal_lm, atom_config, mode: str) -> None:
        self._lm = causal_lm
        self._atom_config = atom_config
        self._mode = mode
        self._ops: dict[tuple[int, int, str], _KimiLayerOp] = {}
        self._weights: dict[int, LayerWeights] = {}
        self._refused: set[tuple[int, int, str]] = set()
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

    def supports(self, input_ids, positions, intermediate_tensors, inputs_embeds) -> bool:
        if not self._enabled or intermediate_tensors is not None:
            return False
        samples = input_ids.numel()
        if inputs_embeds is not None and (
            inputs_embeds.shape != (samples, KIMI_K3_CONFIG.hidden)
            or inputs_embeds.dtype != torch.bfloat16
            or not inputs_embeds.is_contiguous()
        ):
            return False
        backend = select_backend(
            "kimi_k3",
            self._mode,
            samples=samples,
            tp_size=self._atom_config.tensor_parallel_size,
            kv_cache_dtype=self._atom_config.kv_cache_dtype,
        )
        if backend is None or positions.numel() != samples:
            return False
        fwd = get_forward_context()
        if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
            return False
        md = getattr(fwd.attn_metadata, "kda_metadata", None)
        if md is None:
            md = getattr(fwd.attn_metadata, "gdn_metadata", None)
        if md is None:
            return False
        return (
            md.num_prefills == 0
            and md.num_decodes > 0
            and md.num_spec_decodes == 0
            and 0 < md.num_actual_tokens <= samples
            and not getattr(md, "replayssm", False)
        )

    def _op(self, layer, samples: int) -> _KimiLayerOp:
        backend = select_backend(
            "kimi_k3",
            self._mode,
            samples=samples,
            tp_size=8,
            kv_cache_dtype=self._atom_config.kv_cache_dtype,
            is_kda=layer.is_linear_attn,
            has_moe=hasattr(layer, "block_sparse_moe"),
        )
        if backend is None:
            raise MonoUnsupported("layer fallback")
        key = (layer.layer_idx, samples, backend)
        if key in self._refused:
            raise MonoUnsupported("layer construction refused")
        if key not in self._ops:
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("cannot construct during graph capture")
            rank = get_tensor_model_parallel_rank()
            npes = get_tensor_model_parallel_world_size()
            tp = get_tp_group()
            weights = self._weights.get(layer.layer_idx)
            validation_error = None
            if weights is None:
                try:
                    weights = _layer_weights(layer, rank, npes)
                except (MonoUnsupported, ValueError) as error:
                    validation_error = error
            try:
                tp_uniform_local_validation(
                    validation_error,
                    group=tp.cpu_group,
                    world_size=npes,
                    context=f"layer {layer.layer_idx} weight mapping failed",
                )
                assert weights is not None
                self._weights[layer.layer_idx] = weights
                self._ops[key] = _KimiLayerOp(layer, weights, samples, backend)
            except (MonoUnsupported, ValueError) as error:
                self._refused.add(key)
                logger.warning(
                    "Kimi-K3 MonoKernel layer %d fallback: %s",
                    layer.layer_idx,
                    error,
                )
                raise MonoUnsupported(str(error)) from error
        return self._ops[key]

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
        hidden = model.get_input_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        blocks = hidden.new_zeros(samples, 0, hidden.shape[-1])
        pending = pending2 = None
        for layer in model.layers[model.start_layer : model.end_layer]:
            try:
                owned = self._op(layer, samples)
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
        for owned in self._ops.values():
            owned.close()
        self._ops.clear()
        getattr(self, "_weights", {}).clear()
        self._refused.clear()
