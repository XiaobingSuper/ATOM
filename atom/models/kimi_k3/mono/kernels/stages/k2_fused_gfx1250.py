# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K2 past the replicated front as one resident 256-CTA launch: the
lite finish (``k2_finish_persistent_gfx1250``: routing, HERD's finalize, the
shared down), the fused MoE (``moe_tdm_gfx1250.build_moe``) and, with a
tail, the gathering all-reduce, routed RMSNorm and finalize
(``k2_tail_gfx1250``), each CTA running the three kernels' bodies in turn
on one LDS block, bit for bit their separate launches' outputs.

A kernel boundary used to order each hand-off; here a flag a dependency
does, tagged with the fused MoE's epoch (one a launch, never repeated, so
the flags are zeroed once). The writer stores its payload at device scope,
drains it (``s_wait_storecnt 0``), meets its CTA at a barrier, then stores
the tag at device scope; the reader polls with ``s_sleep`` and then loads
the payload at device scope or through the TDM -- no acquire fence, whose
device-wide invalidate costs microseconds a CTA. The flags (``FLAGS``):

- ``ROUTE_AT + t``: token ``t``'s routes are in ``route``/``tile_route``
  (without HERD; the MoE's wave 0 polls its tokens', the latent's TDM copy
  already in flight from the issuer waves).
- ``HERD_AT``: HERD's kept routes are in the finish scratch (with HERD,
  where the MoE reads them).
- ``SHARED_AT + b``: CTA ``b``'s shared down rows are in ``shared_partial``.
- ``MOE_AT + b``: CTA ``b``'s down tiles are in ``grouped``.

The tail's routed CTAs wait on every ``MOE_AT`` flag and its shared CTAs on
every ``SHARED_AT`` one, a thread a flag, after their first up-weight slabs
are in flight. Every wait on a flag here gives up past ``POLL_LIMIT`` polls
and records itself in ``flags[RECORD]`` (0: none), as the MoE's own do; the
tail's waits on its peers' lines and its CTAs' flags are its own, unbounded
as in its launch: they wait on the other ranks, which may be late."""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import Int32, Int64

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    FINISH_PACKED_FRONT_ABI,
    THREADS,
    build_packed_finish,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_tail_gfx1250 import (
    build_k2_tail,
    tail_grid,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import (
    POLL_LIMIT,
    _lds_barrier,
    build_moe,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import CM_DEV, ld_dev, rsrc, st_dev, traced, uniform
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

GRID = 256
RECORD = 0
ROUTE_AT = 16
HERD_AT = ROUTE_AT + GFX1250_MAX_TOKENS
SHARED_AT = HERD_AT + 16
MOE_AT = SHARED_AT + GRID
FLAGS = MOE_AT + GRID

FUSED_ABI = KernelAbi(
    FINISH_PACKED_FRONT_ABI.names
    + (
        "moe_ids",
        "moe_w13s",
        "moe_w2s",
        "moe_inter",
        "moe_grouped",
        "moe_flags",
        "moe_state",
        "fused_flags",
        "tail_data",
        "tail_route_weights",
        "norm_weight",
        "up_weight",
        "output",
        "tail_flags",
        "peers",
        "epochs",
        "rank",
    )
)


@dataclass(frozen=True)
class K2FusedKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    scratch_bytes: int
    tail: bool
    abi: KernelAbi = FUSED_ABI
    launch_count: int = 1


@traced
def _wait_word(flags, word, tag, code):
    """Until ``flags[word]`` carries ``tag``; past POLL_LIMIT polls ``code``
    goes to ``flags[RECORD]`` and the wait goes on."""
    polls = fx.Int32(0)
    flag = fx.Int32(ld_dev(flags, word))
    while (flag != tag) & (polls < POLL_LIMIT):
        rocdl.s_sleep(1)
        polls = polls + 1
        flag = fx.Int32(ld_dev(flags, word))
    if polls >= POLL_LIMIT:
        st_dev(code, flags, RECORD)


def build_k2_fused(
    plan: Gfx1250KimiBuildPlan,
    *,
    herd: bool,
    fp8_shared_down: bool,
    tail_tp: int | None = None,
    tail_eps: float | None = None,
) -> K2FusedKernel:
    """The grouped-MoE lite finish after a replicated front (``herd``: with
    HERD's precomputed routes) and the fused MoE at ``plan.tokens`` rows,
    their rolled code, and with ``tail_tp`` the identity-route K2 tail of
    ``tail_eps``. Arguments: the finish's, then the MoE's own (its ids --
    HERD's kept or the finish's routes -- scales, ``inter``, ``grouped``,
    flags and state; its latent and weights are the finish's), ``fused_flags``
    (``FLAGS`` i32, zeroed once), then the tail's (``tail_data`` its
    ``data``, ``tail_route_weights`` its ``route_weights``; unread without a
    tail)."""
    rows = plan.tokens
    if plan.blocks != GRID or plan.threads != THREADS:
        raise ValueError("the fused K2 is a 256-CTA, 256-thread launch")
    if rows > GFX1250_MAX_TOKENS:
        raise ValueError(f"the fused K2 routes at most {GFX1250_MAX_TOKENS} rows")
    finish = build_packed_finish(
        plan,
        gu_interleave=True,
        include_routed=False,
        replicated_front=True,
        herd=herd,
        fp8_shared_down=fp8_shared_down,
        precompute_herd=herd,
        rolled=True,
    )
    moe_part = build_moe(rows, rolled=True)
    with_tail = tail_tp is not None
    tail_part = (
        build_k2_tail(tail_tp, rows, tail_eps, identity_routes=True)
        if with_tail
        else None
    )
    tail_ctas = tail_grid(rows) if with_tail else 0
    if tail_ctas > GRID:
        raise ValueError(f"the K2 tail's {tail_ctas} CTAs exceed the fused grid")

    if with_tail:

        @fx.union
        class Lds:
            lite: finish.lds
            moe: moe_part.lds
            tail: tail_part.lds

    else:

        @fx.union
        class Lds:
            lite: finish.lds
            moe: moe_part.lds

    name = (
        f"k3_k2_gfx1250_fused_s{rows}"
        + ("_herd" if herd else "")
        + ("_f8" if fp8_shared_down else "")
        + (f"_t{tail_tp}" if with_tail else "")
    )
    sources = SOURCES
    launch_attrs = (
        {"value_attrs": gemv_tdm.CLUSTER_LAUNCH["value_attrs"]} if with_tail else {}
    )
    cluster = {"cluster": gemv_tdm.CLUSTER_LAUNCH["cluster"]} if with_tail else {}

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        front_gathered: Int64,
        router_bias: Int64,
        shared_gate_up: Int64,
        shared_down_weight: Int64,
        shared_down_scale: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        shared_hidden: Int64,
        route: Int64,
        tile_route: Int64,
        flags: Int64,
        experts: Int64,
        kof: Int64,
        union_sizes: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        routed_intermediate_fp4: Int64,
        routed_intermediate_scales: Int64,
        routed_partial: Int64,
        shared_partial: Int64,
        scratch: Int64,
        moe_ids: Int64,
        moe_w13s: Int64,
        moe_w2s: Int64,
        moe_inter: Int64,
        moe_grouped: Int64,
        moe_flags: Int64,
        moe_state: Int64,
        fused_flags: Int64,
        tail_data: Int64,
        tail_route_weights: Int64,
        norm_weight: Int64,
        up_weight: Int64,
        output: Int64,
        tail_flags: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        storage = fx.SharedAllocator().allocate(Lds)
        # The MoE's epoch, moved on by its last CTA once every CTA is past
        # its MoE part, so every CTA reads this launch's.
        epoch = uniform(fx.Int32(ld_dev(moe_state, 2)))
        tag = epoch + 1

        def put(pointer, index, value):
            bo.buffer_store(value, rsrc(pointer), index, cache_modifier=CM_DEV)

        def publish(word):
            rocdl.s_wait_storecnt(0)
            _lds_barrier()
            if tid == 0:
                st_dev(tag, fused_flags, word)

        def route_done(token):
            if const_expr(not herd):
                publish(ROUTE_AT + token)

        def herd_done():
            publish(HERD_AT)

        def shared_done():
            publish(SHARED_AT + bid)

        def wait_routes():
            if const_expr(herd):
                if tid == 0:
                    _wait_word(fused_flags, HERD_AT, tag, (1 << 28) | (bid << 12))
            else:
                if tid < rows:
                    _wait_word(
                        fused_flags,
                        ROUTE_AT + tid,
                        tag,
                        (1 << 28) | (bid << 12) | (ROUTE_AT + tid),
                    )
            _lds_barrier()

        def down_done(moe_tag):
            if tid == 0:
                st_dev(moe_tag, fused_flags, MOE_AT + bid)

        def wait_inputs():
            # Thread t the flag of CTA t: a routed line's picks may be any
            # CTA's down tiles, a shared line any CTA's shared rows.
            first = (bid < 2 * rows).select(fx.Int32(MOE_AT), fx.Int32(SHARED_AT))
            _wait_word(
                fused_flags,
                first + tid,
                tag,
                (2 << 28) | (bid << 12) | (first + tid),
            )
            _lds_barrier()

        finish.body(
            storage.lite,
            front_gathered,
            router_bias,
            shared_gate_up,
            shared_down_weight,
            shared_down_scale,
            w13,
            w13s,
            w2,
            w2s,
            shared_hidden,
            route,
            tile_route,
            flags,
            experts,
            kof,
            union_sizes,
            latent_fp4,
            latent_scales,
            routed_intermediate_fp4,
            routed_intermediate_scales,
            routed_partial,
            shared_partial,
            scratch,
            fused={
                "put": put,
                "route_done": route_done,
                "herd_done": herd_done,
                "shared_done": shared_done,
            },
        )
        # The finish's LDS is the MoE's from here.
        _lds_barrier()
        moe_part.body(
            storage.moe.peek(),
            moe_ids,
            latent_fp4,
            latent_scales,
            w13,
            moe_w13s,
            w2,
            moe_w2s,
            moe_inter,
            moe_grouped,
            moe_flags,
            moe_state,
            fused={"epoch": epoch, "wait_routes": wait_routes, "down_done": down_done},
        )
        if const_expr(with_tail):
            if bid < tail_ctas:
                tail_part.body(
                    storage.tail,
                    tail_data,
                    moe_grouped,
                    tail_route_weights,
                    tail_route_weights,
                    norm_weight,
                    up_weight,
                    output,
                    tail_flags,
                    peers,
                    epochs,
                    rank,
                    fused={"wait_inputs": wait_inputs},
                )

    @flyc.jit
    def launch(
        front_gathered: Int64,
        router_bias: Int64,
        shared_gate_up: Int64,
        shared_down_weight: Int64,
        shared_down_scale: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        shared_hidden: Int64,
        route: Int64,
        tile_route: Int64,
        flags: Int64,
        experts: Int64,
        kof: Int64,
        union_sizes: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        routed_intermediate_fp4: Int64,
        routed_intermediate_scales: Int64,
        routed_partial: Int64,
        shared_partial: Int64,
        scratch: Int64,
        moe_ids: Int64,
        moe_w13s: Int64,
        moe_w2s: Int64,
        moe_inter: Int64,
        moe_grouped: Int64,
        moe_flags: Int64,
        moe_state: Int64,
        fused_flags: Int64,
        tail_data: Int64,
        tail_route_weights: Int64,
        norm_weight: Int64,
        up_weight: Int64,
        output: Int64,
        tail_flags: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            front_gathered,
            router_bias,
            shared_gate_up,
            shared_down_weight,
            shared_down_scale,
            w13,
            w13s,
            w2,
            w2s,
            shared_hidden,
            route,
            tile_route,
            flags,
            experts,
            kof,
            union_sizes,
            latent_fp4,
            latent_scales,
            routed_intermediate_fp4,
            routed_intermediate_scales,
            routed_partial,
            shared_partial,
            scratch,
            moe_ids,
            moe_w13s,
            moe_w2s,
            moe_inter,
            moe_grouped,
            moe_flags,
            moe_state,
            fused_flags,
            tail_data,
            tail_route_weights,
            norm_weight,
            up_weight,
            output,
            tail_flags,
            peers,
            epochs,
            rank,
            **launch_attrs,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream, **cluster)

    FUSED_ABI.check(kernel, launch)
    return K2FusedKernel(
        name, launch, (GRID,), (THREADS,), finish.scratch_bytes, with_tail
    )
