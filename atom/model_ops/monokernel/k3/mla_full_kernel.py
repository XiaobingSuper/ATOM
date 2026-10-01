# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense Kimi-K3 MLA specialization of the shared full-layer MonoKernel."""

from __future__ import annotations

from atom.model_ops.monokernel.config import MAX_LAYERS_PER_STEP
from atom.model_ops.monokernel.k3.kernel import (
    build_kimi_k3_monokernel,
    monokernel_layout,
)


def mla_full_layout(rows: int, *, fuse_moe: bool = True) -> dict[str, int]:
    """Return the shared KDA-tail arena plus dense MLA frontend mailboxes."""

    if rows not in (8, 16, 32):
        raise ValueError(f"Kimi MLA rows must be 8, 16, or 32, got {rows}")
    return monokernel_layout(
        rows,
        fuse_attn_res=True,
        fuse_moe=fuse_moe,
        mtp=False,
        mla=True,
    )


def build_kimi_k3_mla_full_monokernel(
    rows: int,
    *,
    npes: int = 8,
    launches_per_step: int = MAX_LAYERS_PER_STEP,
    attn_res_blocks: int,
    block_write_idx: int,
    atom_expert_layout: bool = True,
):
    """Build one index-free dense-MLA + current K3 latent-MoE launch."""

    return build_kimi_k3_monokernel(
        rows,
        npes,
        launches_per_step,
        attn_res_blocks,
        block_write_idx,
        fuse_moe=True,
        mtp=False,
        atom_expert_layout=atom_expert_layout,
        mla=True,
    )


__all__ = ["build_kimi_k3_mla_full_monokernel", "mla_full_layout"]
