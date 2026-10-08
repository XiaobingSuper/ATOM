# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Static mailbox and compile-only contracts for GLM-5.2 mono builds."""

import sys

from atom.model_ops.monokernel.glm.layout import layout
from atom.models.glm52.mono.config import LEGACY_CORE_WIDTHS
from atom.models.glm52.mono.kernel import ABI, LayerBuild, build_layer
from atom.mono.plan.check import RegionDecl
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import compile_kernels, scratch_bytes

_SCRATCH_PRODUCERS = {
    "q_a": "qkv_a",
    "q_an": "q_norm",
    "kv_a": "qkv_a",
    "kvnew": "cache",
    "penew": "cache",
    "q_nope": "q_b",
    "q_pe": "q_b",
    "q_lat": "uk",
    "sp_acc": "split",
    "sp_m": "split",
    "sp_l": "split",
    "o": "uv",
    "a": "o",
    "scores": "router",
    "xq": "router",
    "xqs": "router",
    "sel": "router",
    "prob": "router",
    "mid": "ug",
    "ugp": "ug",
    "xqd": "router",
}
_INDEX_PRODUCERS = {
    "index_k": "qkv_a",
    "index_k_new": "cache",
    "index_ready": "cache",
    "index_w": "qkv_a",
    "index_q": "index_q",
    "index_scores": "index_score",
    "indices": "index_select",
    "indices_ready": "index_select",
}


def regions(key: LayerBuild) -> list[RegionDecl]:
    producers = dict(_SCRATCH_PRODUCERS)
    if key.with_indexer:
        producers.update(_INDEX_PRODUCERS)
    return [
        *(
            RegionDecl(name, Space.SCRATCH, producer)
            for name, producer in producers.items()
        ),
        RegionDecl("attn", Space.PEER, "o", exchange=True),
        RegionDecl("ffn", Space.PEER, "down", exchange=True),
    ]


def check_layout(key: LayerBuild) -> None:
    scratch, symmetric = layout(
        key.tokens,
        key.heads,
        key.tp,
        key.topk,
        key.with_indexer,
        key.index_max_seq,
        inter=key.inter,
        output_heads=key.heads,
        dcp_size=1,
        native_fp4_mfma=True,
    )
    missing = [
        declaration.name
        for declaration in regions(key)
        if declaration.name
        not in (scratch if declaration.space is Space.SCRATCH else symmetric)
    ]
    if missing:
        raise RuntimeError(f"GLM mailbox layout is missing {missing}")


def check_build(key: LayerBuild) -> None:
    """Compile one build and reject private per-lane scratch."""

    check_layout(key)
    launcher = build_layer(key)
    compile_kernels([(launcher, ABI)])
    private = scratch_bytes(launcher)
    if private:
        raise RuntimeError(f"S{key.tokens} needs {private} B private scratch per lane")


if __name__ == "__main__":
    sizes = [int(value) for value in sys.argv[1:]] or list(LEGACY_CORE_WIDTHS)
    for tokens in sizes:
        for mtp in (4, 5):
            for with_indexer in (False, True):
                check_build(LayerBuild(tokens, with_indexer, mtp))
        print(f"S={tokens}: mailbox and zero-scratch contracts hold")
