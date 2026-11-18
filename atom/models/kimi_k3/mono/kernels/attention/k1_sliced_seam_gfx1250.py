# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The column-sliced AttnRes seam inside the resident gfx1250 K1
(docs/kimi_k3_k1_sliced_seam.md), on ``K1LocalBuildSpec.sliced_seam``.

Every K1 cluster (``TDM_CLUSTER`` CTAs, the launch's own) is a seam home: in
the projection's ``prologue``, with the first weight slabs in flight, its CTAs
mix their token's column slices with cluster-scope exchanges only
(``attn_res_sliced_gfx1250``'s pass), store x and publish one device-scope
ready pair a CTA. The TDM projection's token-copy wave then polls, right
before the first slab that reads a rank's columns, that rank's pairs of every
home holding a token, and copies x at device scope: no acquire fence."""

import flydsl.expr as fx

from atom.models.kimi_k3.mono.kernels.attention import kda_gfx1250 as gfx_backend
from atom.models.kimi_k3.mono.kernels.attention.k1_sliced_seam_layout import K1SeamPlan
from atom.models.kimi_k3.mono.kernels.stages import attn_res_sliced_gfx1250 as sliced
from atom.mono.device.ops import CM_DEV, rsrc, traced
from atom.mono.plan.architecture import GFX1250


@traced
def publish_slice(plan: K1SeamPlan, x_ready, rank, home, generation, tid):
    """This CTA's slice of its home's x is stored: drain, rendezvous, publish."""
    gfx_backend.mailbox_publish_device(
        x_ready, plan.ready_index(rank, home), fx.Int32(0), generation, tid == 0
    )


def gate_slab(plan: K1SeamPlan, x_ready, slab, tokens: int, generation, lane):
    """On the wave about to copy token slab ``slab`` (a trace-time int, or a
    runtime index in the rolled slab loop): if it is the first slab reading
    some rank's columns, wait until every home holding a token has published
    that rank's slice. A lane a home, every poll of a lane in flight at once."""
    homes = plan.geometry.clusters(tokens)
    for first, rank in plan.polls():
        if not isinstance(slab, int):
            _poll_at(plan, x_ready, slab, first, rank, homes, generation, lane)
        elif slab == first:
            _poll_rank(plan, x_ready, rank, homes, generation, lane)


@traced
def _poll_at(plan, x_ready, slab, first, rank, homes, generation, lane):
    if fx.Int32(slab) == fx.Int32(first):
        _poll_rank(plan, x_ready, rank, homes, generation, lane)


def _poll_rank(plan, x_ready, rank, homes, generation, lane):
    gfx_backend.mailbox_poll_many_device(
        [
            (
                x_ready,
                plan.ready_index(
                    rank, fx.min(lane + step * GFX1250.wave_size, homes - 1)
                ),
            )
            for step in range(-(-homes // GFX1250.wave_size))
        ],
        generation,
    )


def seam_pass(
    plan: K1SeamPlan,
    mix: sliced.SlicedMix,
    red,
    tokens: int,
    blocks,
    blk_sm,
    blk_sr,
    prefix,
    delta,
    ares_nw,
    ares_qk,
    in_nw,
    x,
    records,
    x_ready,
    generation,
):
    """This CTA's part of ``attn_res_sliced_gfx1250``'s pass on K1's operands:
    the score weight ``ares_nw * ares_qk`` a column, the output gain ``in_nw``,
    ``prefix + delta`` back into ``prefix``, x to K1's scratch row at device
    scope; then its ready pair. ``red`` holds ``mix.lds_words()``."""
    lds = {}
    offset = 0
    for name, words in mix.lds_words().items():
        lds[name] = red + offset
        offset += words
    block = fx.Int32(fx.block_idx.x)
    sliced.emit_sliced_mix(
        mix,
        lds,
        block,
        tokens,
        blocks,
        blk_sm,
        blk_sr,
        prefix,
        [delta],
        lambda column: _score_weight(ares_nw, ares_qk, column),
        in_nw,
        x,
        prefix,
        records,
        out_cache=CM_DEV,
    )
    publish_slice(
        plan,
        x_ready,
        block % plan.cluster,
        block // plan.cluster,
        generation,
        fx.Int32(fx.thread_idx.x),
    )


def _score_weight(norm_weight, query, column):
    return [
        n * q
        for n, q in zip(
            sliced._eight(rsrc(norm_weight), column),
            sliced._eight(rsrc(query), column),
        )
    ]
