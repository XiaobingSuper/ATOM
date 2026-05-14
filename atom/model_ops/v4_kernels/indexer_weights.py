# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small Indexer helper kernels."""

import torch
import triton
import triton.language as tl
from aiter import dtypes


@triton.jit
def _scale_indexer_weights_kernel(
    weights_ptr,  # [T, H] fp32/bf16
    q_scale_ptr,  # [T, H, 1] fp32, flattened as [T * H]
    out_ptr,  # [T, H] fp32
    n_elements,
    n_cols: tl.constexpr,
    weights_stride_t: tl.constexpr,
    weights_stride_h: tl.constexpr,
    q_scale_stride_t: tl.constexpr,
    q_scale_stride_h: tl.constexpr,
    weights_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    rows = offsets // n_cols
    cols = offsets % n_cols
    weights_offsets = rows * weights_stride_t + cols * weights_stride_h
    q_scale_offsets = rows * q_scale_stride_t + cols * q_scale_stride_h

    weights = tl.load(weights_ptr + weights_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    q_scale = tl.load(q_scale_ptr + q_scale_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(out_ptr + offsets, weights * q_scale * weights_scale, mask=mask)


def scale_indexer_weights(
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    weights_scale: float,
    block_size: int = 1024,
) -> torch.Tensor:
    """Apply `weights * q_scale.squeeze(-1) * weights_scale` in one Triton launch."""
    assert weights.dim() == 2, f"weights must be [T, H], got {tuple(weights.shape)}"
    assert q_scale.shape == (
        weights.size(0),
        weights.size(1),
        1,
    ), (
        f"q_scale shape {tuple(q_scale.shape)} incompatible with weights "
        f"{tuple(weights.shape)}"
    )

    n_elements = weights.numel()
    out = torch.empty(weights.shape, device=weights.device, dtype=torch.float32)
    if n_elements == 0:
        return out

    grid = (triton.cdiv(n_elements, block_size),)
    _scale_indexer_weights_kernel[grid](
        weights,
        q_scale,
        out,
        n_elements,
        weights.size(1),
        weights.stride(0),
        weights.stride(1),
        q_scale.stride(0),
        q_scale.stride(1),
        weights_scale,
        BLOCK_SIZE=block_size,
    )
    return out


@triton.jit
def _quant_indexer_q_and_scale_weights_kernel(
    q_ptr,  # [T * H, D] bf16/fp16/fp32
    weights_ptr,  # [T, H] bf16/fp32
    q_out_ptr,  # [T, H, D] fp8
    weights_out_ptr,  # [T, H] fp32
    n_heads: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_d: tl.constexpr,
    weights_stride_t: tl.constexpr,
    weights_stride_h: tl.constexpr,
    weights_scale: tl.constexpr,
    fp8_max: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(q_ptr + row * q_stride_m + offs_d * q_stride_d).to(tl.float32)
    amax = tl.max(tl.abs(q), axis=0)
    scale = tl.maximum(amax, 1.0e-10) / fp8_max
    q_fp8 = tl.clamp(q / scale, -fp8_max, fp8_max).to(q_out_ptr.dtype.element_ty)
    tl.store(q_out_ptr + row * BLOCK_D + offs_d, q_fp8)

    token = row // n_heads
    head = row % n_heads
    weights = tl.load(
        weights_ptr + token * weights_stride_t + head * weights_stride_h,
    ).to(tl.float32)
    tl.store(weights_out_ptr + row, weights * scale * weights_scale)


def quant_indexer_q_and_scale_weights(
    q: torch.Tensor,
    weights: torch.Tensor,
    weights_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize Indexer q and fold q_scale into weights in one Triton launch."""
    assert q.dim() == 2, f"q must be [T * H, D], got {tuple(q.shape)}"
    assert weights.dim() == 2, f"weights must be [T, H], got {tuple(weights.shape)}"
    n_tokens, n_heads = weights.shape
    head_dim = q.size(1)
    assert q.size(0) == n_tokens * n_heads, (
        f"q rows {q.size(0)} incompatible with weights shape {tuple(weights.shape)}"
    )
    assert head_dim == 128, f"only head_dim=128 is supported, got {head_dim}"

    q_fp8 = torch.empty(
        (n_tokens, n_heads, head_dim),
        dtype=dtypes.fp8,
        device=q.device,
    )
    weights_out = torch.empty(weights.shape, device=weights.device, dtype=torch.float32)
    if q.numel() == 0:
        return q_fp8, weights_out

    grid = (q.size(0),)
    _quant_indexer_q_and_scale_weights_kernel[grid](
        q,
        weights,
        q_fp8,
        weights_out,
        n_heads,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights_scale,
        torch.finfo(dtypes.fp8).max,
        BLOCK_D=head_dim,
    )
    return q_fp8, weights_out
