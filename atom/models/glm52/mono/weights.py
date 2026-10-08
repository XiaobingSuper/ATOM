# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Production GLM weight views consumed in place by the retained layer math."""

import torch

from atom.model_ops.monokernel.config import FP8_MAX
from atom.model_ops.monokernel.glm.layout import INDEX_DIM, INDEX_HEADS
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    _unshuffle_linear_weight,
    linear_bf16,
)
from atom.mono.runtime.consensus import MonoUnsupported


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise MonoUnsupported(reason)


def _quantize_fp8_block128(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    _need(cols % 128 == 0, f"FP8 block K={cols} must be divisible by 128")
    row_blocks = (rows + 127) // 128
    padded = torch.zeros(
        row_blocks * 128,
        cols,
        dtype=torch.float32,
        device=weight.device,
    )
    padded[:rows].copy_(weight.float())
    blocks = padded.view(row_blocks, 128, cols // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3))
    scale = torch.where(scale > 0, scale / FP8_MAX, torch.ones_like(scale))
    quantized = (
        (blocks / scale[:, None, :, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(row_blocks * 128, cols)[:rows]
        .contiguous()
    )
    return quantized, scale.contiguous()


def _linear_fp8_block128(
    linear,
    *,
    name: str,
    logical_rows: int,
    logical_cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant_name = getattr(getattr(linear, "quant_type", None), "name", None)
    weight = linear.weight
    if quant_name == "per_1x128":
        _need(
            weight.dtype is torch.float8_e4m3fn,
            f"{name} must use E4M3FN, got {weight.dtype}",
        )
        _need(
            weight.shape == (logical_rows, logical_cols),
            f"{name} shape {tuple(weight.shape)}",
        )
        scale = getattr(linear, "weight_scale", None)
        expected = ((logical_rows + 127) // 128, logical_cols // 128)
        _need(
            scale is not None
            and scale.dtype is torch.float32
            and scale.shape == expected,
            f"{name} block scale must be FP32 {expected}",
        )
        return _unshuffle_linear_weight(weight), scale.contiguous()
    _need(
        quant_name == "per_Token",
        f"{name} requires per_1x128 or PTPC FP8 storage, got {quant_name}",
    )
    bf16 = linear_bf16(
        linear,
        name=name,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )
    return _quantize_fp8_block128(bf16)


def bind_layer(layer, rank: int, npes: int, *, with_indexer: bool) -> LayerWeights:
    """Bind one loaded layer without copying its established attention/MoE views."""

    from atom.models.glm52_mono import _layer_weights

    base = _layer_weights(layer, rank, npes)
    if not with_indexer:
        return base

    attn = layer.self_attn
    indexer = attn.indexer
    _need(indexer is not None and not attn.skip_topk, "layer has no full indexer")
    tensors = dict(base.t)
    tensors["w_index_q"], tensors["s_index_q"] = _linear_fp8_block128(
        indexer.wq_b,
        name="indexer.wq_b",
        logical_rows=INDEX_HEADS * INDEX_DIM,
        logical_cols=base.config.q_lora,
    )
    if indexer.use_wk_weights_proj_fusion:
        fused = indexer.wk_weights_proj.weight
        expected = (INDEX_DIM + INDEX_HEADS, base.config.hidden)
        _need(
            fused.dtype is torch.bfloat16
            and tuple(fused.shape) == expected
            and fused.is_contiguous(),
            "fused indexer wk/weights_proj layout",
        )
        index_k, index_w = fused[:INDEX_DIM], fused[INDEX_DIM:]
    else:
        index_k = linear_bf16(
            indexer.wk,
            name="indexer.wk",
            logical_rows=INDEX_DIM,
            logical_cols=base.config.hidden,
        )
        index_w = linear_bf16(
            indexer.weights_proj,
            name="indexer.weights_proj",
            logical_rows=INDEX_HEADS,
            logical_cols=base.config.hidden,
        )
    tensors["w_index_k"], tensors["s_index_k"] = _quantize_fp8_block128(index_k)
    tensors["w_index_w"] = index_w
    _need(
        indexer.k_norm.weight.dtype is torch.float32
        and indexer.k_norm.bias.dtype is torch.float32,
        "indexer norm parameters must be FP32",
    )
    tensors["g_index_k"] = indexer.k_norm.weight
    tensors["b_index_k"] = indexer.k_norm.bias
    return LayerWeights(
        base.heads,
        tensors,
        base.config,
        rank,
        npes,
        mxfp4_weight_layout=base.mxfp4_weight_layout,
        mxfp4_scale_layout=base.mxfp4_scale_layout,
        physical_experts=base.physical_experts,
    )


def bind_index_cache(indexer, index_max_seq: int) -> torch.Tensor:
    """Bind the BF16 flat cache ABI used by the retained fused-indexer core."""

    cache = indexer.k_cache.kv_cache[0]
    _need(
        cache.dtype is torch.bfloat16
        and tuple(cache.shape) == (index_max_seq, INDEX_DIM)
        and cache.is_contiguous(),
        "batch mono needs a flat BF16 index cache; production paged FP8 "
        "[..., 132] needs the new ragged score stage",
    )
    return cache
