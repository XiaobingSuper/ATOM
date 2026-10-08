# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exact-width builds of the retained GLM whole-layer math."""

from dataclasses import dataclass, field

from atom.model_ops.monokernel.config import (
    AttentionWeight,
    KvCacheLayout,
    glm5_tp_config,
)
from atom.model_ops.monokernel.glm.abi import GLM5_ABI
from atom.models.glm52.mono.config import (
    LEGACY_CORE_WIDTHS,
    MTP_QUERY_LENGTHS,
    TARGET_WIDTHS,
    TP,
)
from atom.models.glm52.mono.sources import SOURCES
from atom.mono.plan.build_key import key_tuple
from atom.mono.runtime.consensus import MonoUnsupported

ABI = GLM5_ABI


@dataclass(frozen=True)
class LayerBuild:
    tokens: int = field(metadata={"sym": "s"})
    with_indexer: bool = field(metadata={"sym": "ix"})
    mtp_speculative_tokens: int = field(metadata={"sym": "mtp"})
    index_max_seq: int = 4096
    topk: int = 2048
    tp: int = TP
    heads: int = glm5_tp_config(TP).local_heads
    inter: int = glm5_tp_config(TP).inter
    timeline: bool = False

    def __post_init__(self) -> None:
        if self.tokens not in TARGET_WIDTHS:
            raise ValueError(
                f"tokens must be one of {TARGET_WIDTHS}, got {self.tokens}"
            )
        if self.mtp_speculative_tokens not in MTP_QUERY_LENGTHS:
            raise ValueError("GLM-5.2 mono supports MTP4 and MTP5")


def build_layer(key: LayerBuild):
    """Build one persistent layer launcher, without launching it."""

    if key.tokens not in LEGACY_CORE_WIDTHS:
        raise MonoUnsupported(
            f"S{key.tokens} needs the row-batched K4 core; retained GLM LDS is "
            f"bounded to {LEGACY_CORE_WIDTHS}"
        )
    from atom.model_ops.monokernel.glm.kernel import build_glm5_monokernel

    return build_glm5_monokernel(
        key.tokens,
        key.heads,
        key.tp,
        key.topk,
        launches_per_step=128,
        with_indexer=key.with_indexer,
        index_max_seq=key.index_max_seq,
        expert_mxfp4=True,
        atom_experts=True,
        attention_weight=AttentionWeight.FP8_PTPC,
        kv_cache_layout=KvCacheLayout.ATOM,
        kv_cache_dtype="fp8",
        inter=key.inter,
        output_heads=key.heads,
        dcp_size=1,
        native_fp4_mfma=True,
        timeline=key.timeline,
        jit_key=key_tuple(key, SOURCES),
        internal_indices=True,
    )
