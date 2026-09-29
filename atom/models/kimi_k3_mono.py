# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native Kimi-K3 C1 decode using ATOM-owned MonoKernel sources."""

from __future__ import annotations

import logging

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.attention_residual_contract import resolve_attn_res_prefix
from atom.model_ops.monokernel.config import (
    KIMI_K3_CONFIG,
    ConvStateLayout,
    KimiDecodeGeometry,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
)
from atom.model_ops.monokernel.dispatch import (
    MonoUnsupported,
    select_backend,
    tp_uniform_local_validation,
)
from atom.model_ops.monokernel.k3.prepared import (
    KimiK3PreparedWeights,
    prepare_kimi_k3_weights,
)
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    atom_mxfp4_storage_view,
    linear_bf16,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")
_C1_Q = 8
_C1_CONV_ROWS = 10


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def build_kimi_state_chains(
    output: torch.Tensor,
    resume_columns: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Build ``resume + q outputs`` for every request without a host read."""

    groups, q = spec_state_indices.shape
    if output.shape != (groups, q + 1) or output.dtype is not torch.int32:
        raise ValueError(f"output must be int32 [{groups}, {q + 1}]")
    if resume_columns.shape != (groups,) or resume_columns.dtype is not torch.int64:
        raise ValueError(f"resume_columns must be int64 [{groups}]")
    if spec_state_indices.dtype is not torch.int32 or not spec_state_indices.is_contiguous():
        raise ValueError("spec_state_indices must be contiguous int32")
    if (
        num_accepted_tokens.shape != (groups,)
        or num_accepted_tokens.dtype is not torch.int32
        or not num_accepted_tokens.is_contiguous()
    ):
        raise ValueError(f"num_accepted_tokens must be contiguous int32 [{groups}]")
    if any(
        tensor.device != spec_state_indices.device
        for tensor in (output, resume_columns, num_accepted_tokens)
    ):
        raise ValueError("state-chain tensors must share one device")

    output[:, 1:].copy_(spec_state_indices)
    resume_columns.copy_(num_accepted_tokens)
    resume_columns.sub_(1)
    torch.gather(
        spec_state_indices,
        1,
        resume_columns.view(groups, 1),
        out=output[:, :1],
    )
    return output


def _tail_weights(layer, rank: int, npes: int) -> LayerWeights:
    moe = layer.block_sparse_moe
    experts = moe.experts
    cfg = KIMI_K3_CONFIG
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
    return LayerWeights(
        cfg.local_heads,
        tensors,
        cfg,
        rank,
        npes,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
    )


def _layer_weights(layer, rank: int, npes: int) -> LayerWeights:
    _need(layer.is_linear_attn, f"layer {layer.layer_idx}: not KDA")
    weights = _tail_weights(layer, rank, npes)
    attn = layer.self_attn
    cfg = KIMI_K3_CONFIG
    weights.t.update(
        {
            "w_kda_in": linear_bf16(
                attn.in_proj,
                name="in_proj",
                logical_rows=(
                    4 * cfg.local_heads * cfg.v_dim + cfg.local_heads + cfg.v_dim
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
            "kda_dt_bias": attn.dt_bias.view(cfg.local_heads, cfg.v_dim),
            "g_kda_out": attn.o_norm.weight,
            "w_kda_o": linear_bf16(
                attn.o_proj,
                name="o_proj",
                logical_rows=cfg.hidden,
                logical_cols=cfg.local_heads * cfg.v_dim,
            ),
        }
    )
    return weights


def _run_layer_hooks(layer, args: tuple, kwargs: dict, output: tuple) -> tuple:
    """Publish native results through the layer-local hook ABI used by DSpark."""

    hooks_with_kwargs = getattr(layer, "_forward_hooks_with_kwargs", {})
    for hook_id, hook in tuple(getattr(layer, "_forward_hooks", {}).items()):
        if hook_id in hooks_with_kwargs:
            result = hook(layer, args, kwargs, output)
        else:
            result = hook(layer, args, output)
        if result is not None:
            output = result
    if not isinstance(output, tuple) or len(output) != 4:
        raise RuntimeError("a Kimi decoder forward hook changed the layer output ABI")
    return output


class _KimiLayerOp:
    def __init__(
        self,
        layer,
        weights: LayerWeights,
        prepared_weights: KimiK3PreparedWeights | None,
        geometry: KimiDecodeGeometry,
        backend: str,
        kind: str,
    ) -> None:
        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        tp = get_tp_group()
        common = {
            "layer_idx": layer.layer_idx,
            "rank": rank,
            "npes": npes,
            "group": tp.cpu_group,
            "reduce_group": tp.device_group,
            "mtp": geometry.q > 1,
            "conv_state_layout": ConvStateLayout.TIME_MAJOR,
            "conv_state_rows": geometry.conv_state_rows,
            "state_dtype": torch.float16,
        }
        if kind == "kda":
            from atom.model_ops.monokernel.k3.staged import _KimiK3KdaStagedPath

            self.op = _KimiK3KdaStagedPath(
                weights,
                geometry.q,
                prepared_weights=prepared_weights,
                **common,
            )
        else:
            from atom.model_ops.monokernel.k3.staged import _KimiK3FusedTail

            self.op = _KimiK3FusedTail(weights, geometry.q, **common)
        self.q = geometry.q
        self.kind = kind
        self._outputs: dict[int, torch.Tensor] = {}

    def output(self, geometry: KimiDecodeGeometry, hidden: torch.Tensor) -> torch.Tensor:
        output = self._outputs.get(geometry.groups)
        if output is None:
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("cannot allocate grouped output during graph capture")
            output = torch.empty_like(hidden)
            self._outputs[geometry.groups] = output
        return output

    def close(self) -> None:
        self.op.close()
        self._outputs.clear()


class KimiMonoDecode:
    """Exact AgentX C1 decode with staged KDA and a baseline-MLA fused tail."""

    def __init__(self, causal_lm, atom_config, mode: str) -> None:
        self._lm = causal_lm
        self._atom_config = atom_config
        self._mode = mode
        self._ops: dict[tuple[int, int, str, str], _KimiLayerOp] = {}
        self._weights: dict[tuple[int, str], LayerWeights] = {}
        self._prepared: dict[tuple[int, str, bool], KimiK3PreparedWeights] = {}
        self._chains: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._refused: set[tuple[int, int, str, str]] = set()
        self._announced: set[tuple[str, str, int]] = set()
        self._prebuilt = False
        self._enabled = mode != "off"
        if not self._enabled:
            return

        speculative = atom_config.speculative_config
        checks = (
            (atom_config.tensor_parallel_size == 8, "not TP8"),
            (atom_config.parallel_config.data_parallel_size == 1, "DP"),
            (not atom_config.enable_dp_attention, "DPA"),
            (atom_config.decode_context_parallel_size == 1, "not DCP1"),
            (atom_config.pipeline_parallel_size == 1, "PP"),
            (not is_plugin_mode(), "plugin mode"),
            (atom_config.kv_cache_dtype == "fp8", "not FP8 KV"),
            (speculative is not None, "no DSpark"),
            (getattr(speculative, "method", None) == "dspark", "not DSpark"),
            (
                getattr(speculative, "num_speculative_tokens", None) == 7,
                "not DSpark7",
            ),
        )
        for ok, why in checks:
            if not ok:
                logger.info("Kimi-K3 MonoKernel off: %s", why)
                self._enabled = False
                break

    @staticmethod
    def _metadata(fwd):
        metadata = getattr(fwd.attn_metadata, "kda_metadata", None)
        if metadata is None:
            metadata = getattr(fwd.attn_metadata, "gdn_metadata", None)
        return metadata

    def _geometry(self, samples: int, metadata) -> KimiDecodeGeometry | None:
        if (
            metadata is None
            or metadata.num_prefills != 0
            or getattr(metadata, "replayssm", False)
            or not 0 < metadata.num_actual_tokens <= samples
        ):
            return None
        if metadata.num_spec_decodes > 0:
            slots = metadata.spec_state_indices_tensor
            accepted = metadata.num_accepted_tokens
            if slots is None or slots.ndim != 2 or slots.shape[1] != _C1_Q:
                raise RuntimeError("Kimi C1 requires an eight-slot DSpark state table")
            if samples % _C1_Q:
                raise RuntimeError(f"Kimi C1 token rows {samples} are not divisible by q=8")
            groups = samples // _C1_Q
            if slots.shape[0] < groups or accepted is None or accepted.numel() < groups:
                raise RuntimeError("Kimi C1 metadata does not cover the graph batch")
            return KimiDecodeGeometry(
                groups=groups,
                q=_C1_Q,
                conv_state_rows=_C1_CONV_ROWS,
                state_dtype="fp16",
                replay_mode=False,
            )
        if metadata.num_decodes > 0:
            slots = metadata.non_spec_state_indices_tensor
            if slots is None or slots.numel() < samples:
                raise RuntimeError("Kimi C1 decode slots do not cover the graph batch")
            return KimiDecodeGeometry(
                groups=samples,
                q=1,
                conv_state_rows=_C1_CONV_ROWS,
                state_dtype="fp16",
                replay_mode=False,
            )
        return None

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
        fwd = get_forward_context()
        if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
            return False
        geometry = self._geometry(samples, self._metadata(fwd))
        if geometry is None or positions.numel() != samples:
            return False
        return (
            select_backend(
                "kimi_k3",
                self._mode,
                samples=samples,
                tp_size=self._atom_config.tensor_parallel_size,
                kv_cache_dtype=self._atom_config.kv_cache_dtype,
                mtp=geometry.q > 1,
                q=geometry.q,
            )
            is not None
        )

    def _spec_chains(self, geometry: KimiDecodeGeometry, metadata) -> torch.Tensor:
        key = (geometry.groups, geometry.q)
        buffers = self._chains.get(key)
        if buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("cannot allocate state chains during graph capture")
            slots = metadata.spec_state_indices_tensor
            buffers = (
                torch.empty(
                    geometry.groups,
                    geometry.q + 1,
                    dtype=torch.int32,
                    device=slots.device,
                ),
                torch.empty(
                    geometry.groups,
                    dtype=torch.int64,
                    device=slots.device,
                ),
            )
            self._chains[key] = buffers
        chains, resume_columns = buffers
        return build_kimi_state_chains(
            chains,
            resume_columns,
            metadata.spec_state_indices_tensor[: geometry.groups],
            metadata.num_accepted_tokens[: geometry.groups],
        )

    def _op(
        self,
        layer,
        geometry: KimiDecodeGeometry,
        kind: str,
    ) -> _KimiLayerOp:
        backend = "tail"
        if kind == "kda":
            selected = select_backend(
                "kimi_k3",
                self._mode,
                samples=geometry.tokens,
                tp_size=8,
                kv_cache_dtype=self._atom_config.kv_cache_dtype,
                mtp=geometry.q > 1,
                q=geometry.q,
                is_kda=True,
                has_moe=True,
            )
            if selected is None:
                raise RuntimeError("expected Kimi C1 KDA backend is unavailable")
            backend = selected
        key = (layer.layer_idx, geometry.q, backend, kind)
        if key in self._refused:
            raise RuntimeError(f"layer {layer.layer_idx} native construction was refused")
        if key in self._ops:
            return self._ops[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot construct Kimi C1 native layers during graph capture")

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        tp = get_tp_group()
        weight_key = (layer.layer_idx, kind)
        weights = self._weights.get(weight_key)
        validation_error = None
        if weights is None:
            try:
                mapper = _layer_weights if kind == "kda" else _tail_weights
                weights = mapper(layer, rank, npes)
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
            self._weights[weight_key] = weights
            prepared = None
            if kind == "kda":
                prepared_key = (layer.layer_idx, backend, geometry.q > 1)
                prepared = self._prepared.get(prepared_key)
                preparation_error = None
                if prepared is None:
                    try:
                        prepared = prepare_kimi_k3_weights(
                            weights,
                            backend,
                            mtp=geometry.q > 1,
                        )
                    except (MonoUnsupported, ValueError) as error:
                        preparation_error = error
                    tp_uniform_local_validation(
                        preparation_error,
                        group=tp.cpu_group,
                        world_size=npes,
                        context=f"layer {layer.layer_idx} weight preparation failed",
                    )
                    assert prepared is not None
                    self._prepared[prepared_key] = prepared
                else:
                    prepared.validate_source(weights, backend)
            owned = _KimiLayerOp(
                layer,
                weights,
                prepared,
                geometry,
                backend,
                kind,
            )
            self._ops[key] = owned
            announcement = (kind, backend, geometry.q)
            if rank == 0 and announcement not in self._announced:
                logger.info(
                    "Kimi-K3 MonoKernel on: path=%s backend=%s q=%d",
                    kind,
                    backend,
                    geometry.q,
                )
                self._announced.add(announcement)
            return owned
        except (MonoUnsupported, ValueError) as error:
            self._refused.add(key)
            raise RuntimeError(
                f"Kimi C1 layer {layer.layer_idx} cannot use {kind}: {error}"
            ) from error

    def prepare(self) -> None:
        """Construct exact-C1 weights before KV memory is budgeted."""

        if not self._enabled or self._prebuilt:
            return
        geometry = KimiDecodeGeometry(
            groups=1,
            q=_C1_Q,
            conv_state_rows=_C1_CONV_ROWS,
            state_dtype="fp16",
            replay_mode=False,
        )
        model = self._lm.model
        for layer in model.layers[model.start_layer : model.end_layer]:
            if not hasattr(layer, "block_sparse_moe"):
                continue
            self._op(layer, geometry, "kda" if layer.is_linear_attn else "tail")
        self._prebuilt = True

    @staticmethod
    def _require_blocks(blocks: torch.Tensor | None) -> torch.Tensor:
        if blocks is None:
            raise RuntimeError("Kimi C1 native layers require attention-residual blocks")
        return blocks

    @staticmethod
    def _ensure_block_capacity(
        hidden: torch.Tensor,
        blocks: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        if block_index < blocks.shape[1]:
            return blocks
        extra = hidden.new_zeros(
            hidden.shape[0],
            block_index + 1 - blocks.shape[1],
            hidden.shape[1],
        )
        return torch.cat((blocks, extra), dim=1)

    def _run_kda(
        self,
        owned: _KimiLayerOp,
        geometry: KimiDecodeGeometry,
        hidden: torch.Tensor,
        blocks: torch.Tensor,
        cache,
        state_indices: torch.Tensor,
        accepted: torch.Tensor | None,
        layer_idx: int,
    ) -> torch.Tensor:
        output = owned.output(geometry, hidden)
        for group in range(geometry.groups):
            start = group * geometry.q
            end = start + geometry.q
            indices = state_indices[group] if geometry.q > 1 else state_indices[start:end]
            accepted_row = None if accepted is None else accepted[group : group + 1]
            owned.op.forward(
                hidden[start:end],
                blocks[start:end],
                indices,
                cache.k_cache,
                cache.v_cache,
                num_accepted_tokens=accepted_row,
                x_out=output[start:end],
                epoch_layer=layer_idx,
            )
        return output

    def _run_tail(
        self,
        owned: _KimiLayerOp,
        geometry: KimiDecodeGeometry,
        prefix_sum: torch.Tensor | None,
        blocks: torch.Tensor,
        attention_delta: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        output = owned.output(geometry, attention_delta)
        for group in range(geometry.groups):
            start = group * geometry.q
            end = start + geometry.q
            prefix = None if prefix_sum is None else prefix_sum[start:end]
            owned.op.forward_from_attention(
                prefix,
                blocks[start:end],
                attention_delta[start:end],
                x_out=output[start:end],
                epoch_layer=layer_idx,
            )
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self._lm.model
        fwd = get_forward_context()
        metadata = self._metadata(fwd)
        samples = input_ids.numel()
        geometry = self._geometry(samples, metadata)
        if geometry is None:
            raise RuntimeError("Kimi C1 native forward lost its decode geometry")
        hidden = (
            model.get_input_embeddings(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        blocks = hidden.new_zeros(samples, 0, hidden.shape[-1])
        pending = pending2 = None
        state_indices = (
            self._spec_chains(geometry, metadata)
            if geometry.q > 1
            else metadata.non_spec_state_indices_tensor[:samples]
        )
        accepted = (
            metadata.num_accepted_tokens[: geometry.groups]
            if geometry.q > 1
            else None
        )

        for layer in model.layers[model.start_layer : model.end_layer]:
            has_moe = hasattr(layer, "block_sparse_moe")
            if not has_moe:
                hidden, pending, pending2, blocks = layer(
                    positions,
                    hidden,
                    blocks,
                    pending_add=pending,
                    pending_add2=pending2,
                )
                continue

            hook_input = hidden
            hook_kwargs = {"pending_add": pending, "pending_add2": pending2}
            if layer.is_linear_attn:
                hidden, pending, pending2 = resolve_attn_res_prefix(
                    hidden, pending, pending2
                )
                blocks = self._require_blocks(blocks)
                owned = self._op(layer, geometry, "kda")
                blocks = self._ensure_block_capacity(
                    hidden, blocks, owned.op.block_write_idx
                )
                cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"]
                hidden = self._run_kda(
                    owned,
                    geometry,
                    hidden,
                    blocks,
                    cache,
                    state_indices,
                    accepted,
                    layer.layer_idx,
                )
            else:
                attention_input, prefix_sum = layer.self_attention_attn_res(
                    hidden, blocks, pending, pending2
                )
                blocks, prefix_sum = layer.self_attention_attn_res.maybe_close_block(
                    prefix_sum, blocks
                )
                blocks = self._require_blocks(blocks)
                attention_delta = layer.self_attn(positions, attention_input)
                if attention_delta.shape != hidden.shape:
                    raise RuntimeError(
                        f"MLA layer {layer.layer_idx} changed graph rows from "
                        f"{hidden.shape[0]} to {attention_delta.shape[0]}"
                    )
                owned = self._op(layer, geometry, "tail")
                hidden = self._run_tail(
                    owned,
                    geometry,
                    prefix_sum,
                    blocks,
                    attention_delta,
                    layer.layer_idx,
                )
            pending = pending2 = None
            hidden, pending, pending2, blocks = _run_layer_hooks(
                layer,
                (positions, hook_input, blocks),
                hook_kwargs,
                (hidden, pending, pending2, blocks),
            )

        hidden, _ = model.output_attn_res(hidden, blocks, pending, pending2)
        return hidden

    def close(self) -> None:
        for owned in self._ops.values():
            owned.close()
        self._ops.clear()
        self._prepared.clear()
        self._weights.clear()
        self._chains.clear()
        self._refused.clear()
        self._announced.clear()
        self._prebuilt = False
