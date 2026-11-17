# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's head in one gfx1250 launch: o_proj and its TP all-reduce, the
MLP-side AttnRes mix and the replicated front, the front split over K.

``k2_oproj_seam_front_gfx1250`` with its front swapped for
``k2_front_tdm_gfx1250.front_split_tasks``: CTAs ``[0, front.SPLIT_CTAS)``
project in TDM clusters, each CTA a quarter of K of its cluster's tasks. A
front CTA issues its weights first -- its whole share where the LDS holds it
(``front_split_plan(resident=True)``), else as many slab stages as fit -- and
only then sums and lands its all-reduce duty and waits for the mix, so the
weights stream while the all-reduce is pending and once ``x`` is out only its
quarter of ``x`` is left to copy. The cluster then sums the partials in CTA
order: another reduction order than the full-K front, so its outputs differ
from the separate launches' in the last bits.

The mix CTAs follow the front, up to ``GRID - front.SPLIT_CTAS`` of them, and
the CTAs below ``k2_oproj_ar_gfx1250.CTAS`` run the o_proj all-reduce's CTA
first, as in the o_proj seam front. The mix is a ``HeadMix``
(``row_mix``: ``k2_seam_front_gfx1250``'s, a token's columns over
``split`` CTAs): its CTAs poll their attention row from the all-reduce's
sums, mix and publish one flag each; the front waits for every flag.

A poll reads the line it waits for at system scope (device scope at TP1), a
flag at device scope and ``x`` through the TDM at device scope: no acquire
fence.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import attn_res_gfx1250 as attn_res
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_seam_front_gfx1250 as osf
from atom.models.kimi_k3.mono.kernels.stages import k2_seam_front_gfx1250 as sf
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    tdm_split_scratch_bytes,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import CM_DEV, lds_bytes, load_ptr64, rsrc, st_dev, traced
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = osf.THREADS
GRID = sf.MAX_CTAS
FRONT_CTAS = front.SPLIT_CTAS
MIX_CTAS = GRID - FRONT_CTAS
assert FRONT_CTAS % sf.CLUSTER == 0 and MIX_CTAS > 0
# ``sync``: the o_proj seam front's words; a profile build's stamps past them
SYNC_WORDS = osf.SYNC_WORDS
PROFILE_WORDS = osf.PROFILE_WORDS
STAMPS = osf.STAMPS
# the o_proj seam front's stamps, and a front CTA's weights issued
STAMP_START = 0
STAMP_LANDED = 1
STAMP_X_READY = 2
STAMP_FRONT_END = 3
STAMP_MIX_PUBLISHED = 4
STAMP_ISSUED = 5
STAMP_PUSHED = osf.STAMP_PUSHED
STAMP_REDUCED = osf.STAMP_REDUCED
STAMP_MIX_POLLED = osf.STAMP_LANDED
assert STAMP_ISSUED not in (STAMP_PUSHED, STAMP_REDUCED, STAMP_MIX_POLLED)

# The launcher's arguments: the o_proj seam front's.
K2_HEAD_ABI = osf.OPROJ_SEAM_FRONT_ABI
# The kernel's: those, then the front's f32 partials (kept by the kernel).
_KERNEL_ABI = KernelAbi((*K2_HEAD_ABI.names, "front_scratch"))


@dataclass(frozen=True)
class HeadMix:
    """A mix of the head: ``ctas`` CTAs past the front (whole clusters),
    ``flags`` of which publish one ``sync`` flag each once their ``x`` columns
    are out. ``gemv_words``/``mix_words``: the i32 LDS words it stages in the
    head's ``gemv`` region and the f32 words of its own ``mix`` region.
    ``body(lds, index, ctx)`` is mix CTA ``index``'s part, ``idle``
    included; ``ctx`` (``_MixContext``) the head's view."""

    name: str
    ctas: int
    flags: int
    gemv_words: int
    mix_words: int
    body: object


@dataclass(frozen=True)
class _MixContext:
    reduce: object
    sync: object
    epoch: object
    tid: object
    lane: object
    wave: object
    stamp: object
    # (blocks, prefix, score_weight, out_weight, x, prefix_out, token_stride)
    pointers: tuple


def row_mix(rows, count, addends, eps, out_eps, ctas=MIX_CTAS, split=None):
    """``k2_seam_front_gfx1250``'s mix: token ``t`` on mix CTAs
    ``[t * split, (t + 1) * split)``, the most CTAs a token that ``ctas``
    hold (``split``: that many)."""
    if split is None:
        split = next((s for s in sf.SPLITS if sf._clusters(rows * s) <= ctas), None)
    if split is None or sf._clusters(rows * split) > ctas:
        raise ValueError(f"the head's {ctas} mix CTAs hold no mix of {rows} rows")
    layout = sf._MixLayout(count, addends, eps, out_eps, split)
    attention = count + addends
    flags = rows * split
    reductions = layout.reductions

    @traced
    def body(lds, index, c):
        blocks, prefix, score_weight, out_weight, x, prefix_out, stride = c.pointers
        if index < flags:
            token = index // split
            part = index % split
            column0 = part * layout.width
            regions = layout.regions(lds.gemv.ptr)
            pointers = (blocks, prefix, prefix, prefix, score_weight, out_weight)
            sf._stage(
                regions,
                token,
                column0,
                c.wave,
                pointers,
                stride,
                layout,
                skip=attention,
                wait=False,
            )
            osf._sum_duty(c.reduce, fx.Int32(fx.block_idx.x), c.tid)
            c.stamp(STAMP_REDUCED)
            sf._mix_token(
                lds.mix.ptr,
                lds.gemv.ptr,
                c.sync,
                token,
                part,
                c.tid,
                c.lane,
                c.wave,
                (*pointers, x, prefix_out),
                stride,
                layout,
                c.stamp,
                stage=lambda: osf._attention_row(
                    c.reduce,
                    token,
                    column0,
                    layout,
                    regions[0] + attention * layout.row_words,
                    c.tid,
                    c.stamp,
                ),
            )
            sf._publish(c.sync, index, c.epoch, c.tid)
            c.stamp(STAMP_MIX_PUBLISHED)
            osf._land_duty(c.reduce, fx.Int32(fx.block_idx.x), c.tid)
        if index >= flags:
            osf._sum_duty(c.reduce, fx.Int32(fx.block_idx.x), c.tid)
            sf._idle_mix(split)
            osf._land_duty(c.reduce, fx.Int32(fx.block_idx.x), c.tid)

    return HeadMix(
        name=f"m{split}",
        ctas=sf._clusters(flags),
        flags=flags,
        gemv_words=sf.mix_lds_words(count, addends, split),
        mix_words=sf.MIX_WAVES * reductions + reductions + sf.MIX_WAVES,
        body=body,
    )


@dataclass(frozen=True)
class K2HeadKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    mix: HeadMix
    # {kind: (format, stages, resident)} of the split front
    split: dict
    # the front's f32 partials, kept as long as the launcher
    scratch: object
    abi: KernelAbi = K2_HEAD_ABI
    launch_count: int = 1


def build_k2_head(
    plan: Gfx1250KimiBuildPlan,
    tp: int,
    proj: int,
    count: int,
    addends: int,
    eps: float,
    out_eps: float,
    fp8_shared: bool = False,
    resident: bool = True,
    mix: HeadMix | None = None,
    poll_scope: int | None = None,
    profile: bool = False,
) -> K2HeadKernel:
    """``k2_oproj_seam_front_gfx1250.build_oproj_seam_front(plan, tp, proj,
    count, addends, eps, out_eps, fp8_shared)`` with the split-K front: the
    same arguments and ``sync`` (``SYNC_WORDS`` i32, ``PROFILE_WORDS`` more in
    a ``profile`` build, zeroed once and used by no other kernel). ``mix``:
    a ``HeadMix`` (``row_mix`` by default). ``resident``: each front
    projection whose whole weight share fits the LDS is copied whole."""
    import torch

    if plan.architecture is not GFX1250:
        raise ValueError("the K2 head requires gfx1250")
    if plan.threads != THREADS or plan.waves != sf.WAVES:
        raise ValueError("the K2 head requires 256 threads/eight waves")
    if FRONT_CTAS > plan.blocks or GRID > plan.blocks:
        raise ValueError(f"the K2 head needs {GRID} resident CTAs")
    if not 1 <= count < attn_res.MAX_CANDIDATES:
        raise ValueError(
            f"the AttnRes mix holds 1 to {attn_res.MAX_CANDIDATES - 1} blocks"
        )
    if addends not in (0, 1):
        raise ValueError("the K2 head's mix folds at most its attention row")
    if HIDDEN % (tp * oar.CTA_COLUMNS):
        raise ValueError(f"a rank's columns must be whole CTAs at TP{tp}")
    rows = plan.tokens
    if not 1 <= rows <= GFX1250_MAX_TOKENS:
        raise ValueError(f"the K2 head runs 1 to {GFX1250_MAX_TOKENS} rows")
    if oar.triples(rows) * oar.CTA_PAIRS > THREADS:
        raise ValueError(f"the o_proj all-reduce cuts {rows} rows into a line a thread")
    if mix is None:
        mix = row_mix(rows, count, addends, eps, out_eps)
    if mix.ctas > MIX_CTAS or mix.ctas % sf.CLUSTER:
        raise ValueError(f"the K2 head's mix takes whole clusters of {MIX_CTAS} CTAs")
    if mix.flags > THREADS:
        raise ValueError("the front waits on a mix flag a thread")
    if poll_scope is None:
        poll_scope = osf.SCOPE_DEV if tp == 1 else ar.SCOPE_SYS | ar.VOLATILE
    grid = max(FRONT_CTAS + mix.ctas, osf.OPROJ_CTAS)
    assert grid % sf.CLUSTER == 0 and grid <= GRID
    tiles = (rows + front.ROWS - 1) // front.ROWS
    group_words = rows * 16
    budget = (plan.lds_bytes - 1024) // 4 - group_words - mix.mix_words
    split = front.front_split_plan(rows, fp8_shared, False, budget, resident=resident)
    if split is None:
        raise ValueError(f"the split-K front does not fit {rows} rows")
    gemv_words = max(
        front.front_split_lds_words(rows, split),
        mix.gemv_words,
        osf.oproj_lds_words(rows),
    )
    words_a_bank = oar.bank_words(tp, rows)

    @fx.struct
    class Lds:
        # the o_proj's slabs, then the split front's or a mix CTA's
        gemv: fx.Array[fx.Int32, gemv_words, 16]
        group: fx.Array[fx.Int32, group_words, 16]
        mix: fx.Array[fx.Float32, mix.mix_words, 16]

    if lds_bytes(Lds) > plan.lds_bytes:
        raise ValueError(f"the K2 head needs {lds_bytes(Lds)} B of LDS")
    stages = "".join("r" if whole else str(depth) for _, depth, whole in split.values())
    name = (
        f"k3_k2_gfx1250_head_tp{tp}_b{count}_a{addends}_s{rows}"
        f"{'_f8' if fp8_shared else ''}_{mix.name}_k{stages}_q{poll_scope & 0xFF}"
        f"{'_profile' if profile else ''}"
    )
    scratch = torch.empty(
        tdm_split_scratch_bytes(FRONT_CTAS, rows),
        dtype=torch.uint8,
        # a CPU build is never launched
        device=(
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        ),
    )
    flags = mix.flags
    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        core: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        attn_out: Int64,
        peers: Int64,
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
        front_scratch: Int64,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        lds = fx.SharedAllocator().allocate(Lds).peek()
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        mine = load_ptr64(peers, rank)
        reduce = osf._Reduce(
            rows,
            tp,
            rsrc(mine),
            [load_ptr64(peers, p) for p in range(tp)],
            attn_out,
            rank,
            (epoch & 1) * words_a_bank,
            epoch,
            rsrc(mine, 2 * words_a_bank * 4),
            poll_scope,
        )

        def stamp(index):
            if const_expr(profile):  # noqa: SIM102
                if tid == 0:
                    st_dev(
                        fx.Int32(_realtime()),
                        sync,
                        SYNC_WORDS + bid * STAMPS + index,
                    )

        def prologue():
            # the front's weights are in flight from here
            stamp(STAMP_ISSUED)
            osf._sum_duty(reduce, bid, tid)
            stamp(STAMP_REDUCED)
            osf._land_duty(reduce, bid, tid)
            stamp(STAMP_LANDED)
            sf._await_mix(sync, epoch, flags, tid)
            stamp(STAMP_X_READY)

        stamp(STAMP_START)
        if bid < osf.OPROJ_CTAS:
            osf._oproj_partials(
                lds.gemv.ptr, core, w_o, w_o_scale, proj, reduce, bid, tid, lane, wave
            )
            stamp(STAMP_PUSHED)
        if bid < FRONT_CTAS:
            front.front_split_tasks(
                lds,
                x,
                (
                    router_weight,
                    latent_weight,
                    latent_weight,
                    shared_weight,
                    shared_scale,
                ),
                (router_logits, latent_fp4, latent_scales, shared_hidden),
                tid,
                lane,
                wave,
                rows,
                tiles,
                front_scratch,
                split,
                prologue=prologue,
                x_cache_policy=CM_DEV,
            )
            stamp(STAMP_FRONT_END)
        if (bid >= FRONT_CTAS) & (bid < FRONT_CTAS + mix.ctas):
            mix.body(
                lds,
                bid - FRONT_CTAS,
                _MixContext(
                    reduce,
                    sync,
                    epoch,
                    tid,
                    lane,
                    wave,
                    stamp,
                    (
                        blocks,
                        prefix,
                        score_weight,
                        out_weight,
                        x,
                        prefix_out,
                        token_stride,
                    ),
                ),
            )
        if bid >= FRONT_CTAS + mix.ctas:
            osf._sum_duty(reduce, bid, tid)
            osf._land_duty(reduce, bid, tid)
        gpu.barrier()
        counter = bid + tid * grid
        if counter < ar.MAX_CTAS:
            bo.buffer_store(epoch, rsrc(epochs), counter)

    @flyc.jit
    def launch(
        core: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        attn_out: Int64,
        peers: Int64,
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
        front_scratch: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            core,
            w_o,
            w_o_scale,
            attn_out,
            peers,
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
            front_scratch,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(grid,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    _KERNEL_ABI.check(kernel, launch)
    scratch_address = scratch.data_ptr()

    def launcher(*args, **kwargs):
        return launch(*args, scratch_address, **kwargs)

    return K2HeadKernel(name, launcher, (grid,), (THREADS,), mix, split, scratch)
