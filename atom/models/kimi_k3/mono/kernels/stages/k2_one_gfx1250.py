# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K2 of a KDA layer as one resident 256-CTA launch: o_proj and its
all-reduce, the MLP-side AttnRes mix and the replicated front
(``k2_oproj_seam_front_gfx1250``), then the lite finish, the fused MoE and
the K2 tail (``k2_fused_gfx1250``), each CTA running the parts' bodies in
turn on one LDS block, bit for bit their separate launches' outputs.

The o_proj seam front keeps its own hand-offs (all-reduce lines, the mix
flags), tagged with the K2 all-reduce counters' next epoch; the tail's
all-reduce takes the same epoch on its own peer buffer, and the launch
moves the counters on once, at its end. Past the front, hand-offs are
``k2_fused_gfx1250``'s flags, tagged with the fused MoE's epoch, plus one a
front CTA (``FRONT_AT + pair``): its router logits, latent or shared hidden
rows stored at device scope and drained. A routing CTA waits on the router
pairs' flags, a shared-down CTA on the shared pairs' once its down weights
are in flight, and a MoE CTA on the latent pairs' before its latent copy;
each then reads at device scope, with no acquire fence. These waits give up
past ``POLL_LIMIT`` polls and record themselves in ``flags[RECORD]``, as
``k2_fused_gfx1250``'s do.

Known to fail at TP > 1: every wait past the front sits behind the
cross-rank all-reduce, so a rank that waits on its peers (a launch skew of
more than ``POLL_LIMIT`` polls) gives up and reads stale flags. The fix: the
waits downstream of the all-reduce -- ``wait_front``, ``wait_routes``,
``wait_inputs`` here and the MoE's and tail's own bounded waits -- poll
without a limit, as the seam's all-reduce polls do (or with a limit sized for
the ranks' skew, its record kept as a diagnostic); the MoE and tail bodies
need a hook for that.

A ``profile`` build stamps each CTA's 100 MHz clock (``STAMPS`` words a CTA
past ``FLAGS``): start, its o_proj seam front part done, router logits in,
shared rows in, finish done, latent in, routes in, MoE done, tail inputs
in, end; its o_proj seam front part stamps into ``sync`` as that kernel's
profile build does."""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages import k2_fused_gfx1250 as kf
from atom.models.kimi_k3.mono.kernels.stages import (
    k2_oproj_seam_front_gfx1250 as osf,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    THREADS,
    build_packed_finish,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_tail_gfx1250 import (
    build_k2_tail,
    tail_grid,
    tail_max_rows,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import (
    _lds_barrier,
    _realtime,
    build_moe,
)
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import CM_DEV, ld_dev, rsrc, st_dev, uniform
from atom.mono.runtime.abi import KernelAbi

GRID = kf.GRID
RECORD = kf.RECORD
FRONT_AT = kf.FLAGS
FLAGS = FRONT_AT + -(-front.PAIRS // 16) * 16
STAMPS = 10
PROFILE_WORDS = GRID * STAMPS
(
    STAMP_START,
    STAMP_SEAM,
    STAMP_ROUTER,
    STAMP_SHARED,
    STAMP_FINISH,
    STAMP_LATENT,
    STAMP_ROUTES,
    STAMP_MOE,
    STAMP_TAIL,
    STAMP_END,
) = range(STAMPS)

K2_ONE_ABI = KernelAbi(
    (
        # the o_proj seam front's
        "core",
        "w_o",
        "w_o_scale",
        "attn_out",
        "oproj_peers",
        "epochs",
        "rank",
        "blocks",
        "prefix",
        "score_weight",
        "out_weight",
        "x",
        "prefix_out",
        "token_stride",
        "sync",
        "router_weight",
        "latent_weight",
        "shared_weight",
        "shared_scale",
        "router_logits",
        "latent_fp4",
        "latent_scales",
        "shared_hidden",
        # the lite finish's past those
        "router_bias",
        "shared_gate_up",
        "shared_down_weight",
        "shared_down_scale",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "route",
        "tile_route",
        "flags",
        "experts",
        "kof",
        "union_sizes",
        "routed_intermediate_fp4",
        "routed_intermediate_scales",
        "routed_partial",
        "shared_partial",
        "scratch",
        # the fused MoE's
        "moe_ids",
        "moe_w13s",
        "moe_w2s",
        "moe_inter",
        "moe_grouped",
        "moe_flags",
        "moe_state",
        "one_flags",
        # the tail's
        "tail_data",
        "tail_route_weights",
        "norm_weight",
        "up_weight",
        "output",
        "tail_flags",
        "tail_peers",
    )
)


@dataclass(frozen=True)
class K2OneKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    scratch_bytes: int
    abi: KernelAbi = K2_ONE_ABI
    launch_count: int = 1


def build_k2_one(
    plan: Gfx1250KimiBuildPlan,
    tp: int,
    proj: int,
    count: int,
    addends: int,
    eps: float,
    out_eps: float,
    *,
    herd: bool,
    fp8_shared: bool,
    fp8_shared_down: bool,
    tail_eps: float,
    profile: bool = False,
) -> K2OneKernel:
    """``build_oproj_seam_front(plan, tp, proj, count, addends, eps,
    out_eps, fp8_shared)`` then ``build_k2_fused(plan, herd=herd,
    fp8_shared_down=fp8_shared_down, tail_tp=tp, tail_eps=tail_eps)`` as one
    launch. Arguments (``K2_ONE_ABI``): the o_proj seam front's (its
    ``peers`` as ``oproj_peers``), the finish's not among them (its
    ``front_gathered`` is ``router_logits``), the fused MoE's, ``one_flags``
    (``FLAGS`` i32, ``PROFILE_WORDS`` more in a ``profile`` build, zeroed
    once) and the tail's not among them (its ``peers`` as ``tail_peers``)."""
    rows = plan.tokens
    if plan.blocks != GRID or plan.threads != THREADS:
        raise ValueError("the one-launch K2 is a 256-CTA, 256-thread launch")
    if rows > tail_max_rows() or tail_grid(rows) > GRID:
        raise ValueError(
            f"the one-launch K2's tail takes at most {tail_max_rows()} rows"
        )
    seam_part = osf.build_oproj_seam_front(
        plan,
        tp,
        proj,
        count,
        addends,
        eps,
        out_eps,
        fp8_shared=fp8_shared,
        profile=profile,
    )
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
    tail_part = build_k2_tail(tp, rows, tail_eps, identity_routes=True)
    tail_ctas = tail_grid(rows)
    # the seam's mix CTAs, before its projection CTAs
    mixers = osf.sf._clusters(rows * seam_part.split)

    @fx.union
    class Lds:
        seam: seam_part.lds
        lite: finish.lds
        moe: moe_part.lds
        tail: tail_part.lds

    name = (
        f"k3_k2_gfx1250_one_tp{tp}_s{rows}_b{count}_a{addends}"
        + ("_herd" if herd else "")
        + ("_f8" if fp8_shared_down else "")
        + ("_profile" if profile else "")
    )
    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        core: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        attn_out: Int64,
        oproj_peers: Int64,
        epochs: Int64,
        rank: Int32,
        blocks: Int64,
        prefix: Int64,
        score_weight: Int64,
        out_weight: Int64,
        x: Int64,
        prefix_out: Int64,
        token_stride: Int32,
        sync: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        shared_scale: Int64,
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_hidden: Int64,
        router_bias: Int64,
        shared_gate_up: Int64,
        shared_down_weight: Int64,
        shared_down_scale: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        route: Int64,
        tile_route: Int64,
        flags: Int64,
        experts: Int64,
        kof: Int64,
        union_sizes: Int64,
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
        one_flags: Int64,
        tail_data: Int64,
        tail_route_weights: Int64,
        norm_weight: Int64,
        up_weight: Int64,
        output: Int64,
        tail_flags: Int64,
        tail_peers: Int64,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        storage = fx.SharedAllocator().allocate(Lds)
        # The all-reduce epoch, the o_proj's and the tail's; the counters
        # move on at the end, every CTA's its own.
        ar_epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        # The MoE's epoch, moved on by its last CTA once every CTA is past
        # its MoE part, so every CTA reads this launch's.
        epoch = uniform(fx.Int32(ld_dev(moe_state, 2)))
        tag = epoch + 1

        def stamp(index):
            if const_expr(profile):  # noqa: SIM102
                if tid == 0:
                    st_dev(
                        fx.Int32(_realtime()),
                        one_flags,
                        FLAGS + bid * STAMPS + index,
                    )

        def put(pointer, index, value):
            bo.buffer_store(value, rsrc(pointer), index, cache_modifier=CM_DEV)

        def publish(word):
            rocdl.s_wait_storecnt(0)
            _lds_barrier()
            if tid == 0:
                st_dev(tag, one_flags, word)

        def wait_front(first, pairs, code):
            # thread t the flag of front pair ``first + t``
            if tid < pairs:
                kf._wait_word(
                    one_flags,
                    FRONT_AT + first + tid,
                    tag,
                    (code << 28) | (bid << 12) | (first + tid),
                )
            _lds_barrier()

        def front_done(pair):
            publish(FRONT_AT + pair)

        def wait_router():
            wait_front(0, front.LATENT_PAIR0, 3)
            stamp(STAMP_ROUTER)

        def wait_shared():
            wait_front(front.SHARED_PAIR0, front.PAIRS - front.SHARED_PAIR0, 4)
            stamp(STAMP_SHARED)

        def wait_latent():
            wait_front(front.LATENT_PAIR0, front.SHARED_PAIR0 - front.LATENT_PAIR0, 5)
            stamp(STAMP_LATENT)

        def poll(requests):
            values = []
            for pointer, index, _tag in requests:
                word = fx.Int32(ld_dev(pointer, index))
                values.append((word, fx.Int32(1)))
            return values

        def route_done(token):
            if const_expr(not herd):
                publish(kf.ROUTE_AT + token)

        def herd_done():
            publish(kf.HERD_AT)

        def shared_done():
            publish(kf.SHARED_AT + bid)

        def wait_routes():
            if const_expr(herd):
                if tid == 0:
                    kf._wait_word(one_flags, kf.HERD_AT, tag, (1 << 28) | (bid << 12))
            else:
                if tid < rows:
                    kf._wait_word(
                        one_flags,
                        kf.ROUTE_AT + tid,
                        tag,
                        (1 << 28) | (bid << 12) | (kf.ROUTE_AT + tid),
                    )
            _lds_barrier()
            stamp(STAMP_ROUTES)

        def down_done(moe_tag):
            if tid == 0:
                st_dev(moe_tag, one_flags, kf.MOE_AT + bid)
            stamp(STAMP_MOE)

        def wait_inputs():
            # Thread t the flag of CTA t: a routed line's picks may be any
            # CTA's down tiles, a shared line any CTA's shared rows.
            first = (bid < 2 * rows).select(fx.Int32(kf.MOE_AT), fx.Int32(kf.SHARED_AT))
            kf._wait_word(
                one_flags,
                first + tid,
                tag,
                (2 << 28) | (bid << 12) | (first + tid),
            )
            _lds_barrier()
            stamp(STAMP_TAIL)

        stamp(STAMP_START)
        seam_part.body(
            storage.seam.peek(),
            core,
            w_o,
            w_o_scale,
            attn_out,
            oproj_peers,
            epochs,
            rank,
            blocks,
            prefix,
            score_weight,
            out_weight,
            x,
            prefix_out,
            token_stride,
            sync,
            router_weight,
            latent_weight,
            shared_weight,
            shared_scale,
            router_logits,
            latent_fp4,
            latent_scales,
            shared_hidden,
            fused={
                "epoch": ar_epoch,
                "front_done": lambda: front_done(bid - mixers),
            },
        )
        stamp(STAMP_SEAM)
        # Every cluster peer past its GEMVs: no multicast still lands in
        # this CTA's LDS, the finish's from here.
        cluster.cluster_barrier()
        finish.body(
            storage.lite,
            router_logits,
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
                "wait_router": wait_router,
                "wait_shared": wait_shared,
                "poll": poll,
            },
        )
        stamp(STAMP_FINISH)
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
            fused={
                "epoch": epoch,
                "wait_routes": wait_routes,
                "down_done": down_done,
                "wait_latent": wait_latent,
            },
        )
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
                tail_peers,
                epochs,
                rank,
                fused={"wait_inputs": wait_inputs, "epoch": ar_epoch},
            )
        gpu.barrier()
        if tid == 0:
            bo.buffer_store(ar_epoch, rsrc(epochs), bid)
        stamp(STAMP_END)

    @flyc.jit
    def launch(
        core: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        attn_out: Int64,
        oproj_peers: Int64,
        epochs: Int64,
        rank: Int32,
        blocks: Int64,
        prefix: Int64,
        score_weight: Int64,
        out_weight: Int64,
        x: Int64,
        prefix_out: Int64,
        token_stride: Int32,
        sync: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        shared_scale: Int64,
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_hidden: Int64,
        router_bias: Int64,
        shared_gate_up: Int64,
        shared_down_weight: Int64,
        shared_down_scale: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        route: Int64,
        tile_route: Int64,
        flags: Int64,
        experts: Int64,
        kof: Int64,
        union_sizes: Int64,
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
        one_flags: Int64,
        tail_data: Int64,
        tail_route_weights: Int64,
        norm_weight: Int64,
        up_weight: Int64,
        output: Int64,
        tail_flags: Int64,
        tail_peers: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            core,
            w_o,
            w_o_scale,
            attn_out,
            oproj_peers,
            epochs,
            rank,
            blocks,
            prefix,
            score_weight,
            out_weight,
            x,
            prefix_out,
            token_stride,
            sync,
            router_weight,
            latent_weight,
            shared_weight,
            shared_scale,
            router_logits,
            latent_fp4,
            latent_scales,
            shared_hidden,
            router_bias,
            shared_gate_up,
            shared_down_weight,
            shared_down_scale,
            w13,
            w13s,
            w2,
            w2s,
            route,
            tile_route,
            flags,
            experts,
            kof,
            union_sizes,
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
            one_flags,
            tail_data,
            tail_route_weights,
            norm_weight,
            up_weight,
            output,
            tail_flags,
            tail_peers,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(GRID,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    K2_ONE_ABI.check(kernel, launch)
    return K2OneKernel(name, launch, (GRID,), (THREADS,), finish.scratch_bytes)
