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
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

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
    TDM_SPLIT_SLAB,
    tdm_split_scratch_bytes,
    tdm_split_slabs,
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

# The launcher's arguments: the o_proj seam front's, then (an MXFP4 latent,
# its planes in ``latent_weight``) the latent's E8M0 plane.
K2_HEAD_ABI = osf.OPROJ_SEAM_FRONT_ABI
K2_HEAD_FP4_ABI = KernelAbi((*K2_HEAD_ABI.names, "latent_scale"))
# The kernel's: those, then the front's f32 partials (kept by the kernel).
_KERNEL_ABI = KernelAbi((*K2_HEAD_FP4_ABI.names, "front_scratch"))


@dataclass(frozen=True)
class HeadMix:
    """A mix of the head: ``ctas`` CTAs past the front (whole clusters),
    ``flags`` of which publish one ``sync`` flag each once their ``x`` columns
    are out. ``gemv_words``/``mix_words``: the i32 LDS words it stages in the
    head's ``gemv`` region and the f32 words of its own ``mix`` region.
    ``body(lds, index, reduce, sync, epoch, tid, lane, wave, stamp, blocks,
    prefix, score_weight, out_weight, x, prefix_out, token_stride)`` is mix
    CTA ``index``'s part, ``idle`` included: flat, as a traced call takes
    its values."""

    name: str
    ctas: int
    flags: int
    gemv_words: int
    mix_words: int
    body: object


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
    def body(
        lds, index, reduce, sync, epoch, tid, lane, wave, stamp, blocks, prefix,
        score_weight, out_weight, x, prefix_out, stride,
    ):  # fmt: skip
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
                wave,
                pointers,
                stride,
                layout,
                skip=attention,
                wait=False,
            )
            osf._sum_duty(reduce, fx.Int32(fx.block_idx.x), tid)
            stamp(STAMP_REDUCED)
            sf._mix_token(
                lds.mix.ptr,
                lds.gemv.ptr,
                sync,
                token,
                part,
                tid,
                lane,
                wave,
                (*pointers, x, prefix_out),
                stride,
                layout,
                stamp,
                stage=lambda: osf._attention_row(
                    reduce,
                    token,
                    column0,
                    layout,
                    regions[0] + attention * layout.row_words,
                    tid,
                    stamp,
                ),
            )
            sf._publish(sync, index, epoch, tid)
            stamp(STAMP_MIX_PUBLISHED)
            osf._land_duty(reduce, fx.Int32(fx.block_idx.x), tid)
        if index >= flags:
            osf._sum_duty(reduce, fx.Int32(fx.block_idx.x), tid)
            sf._idle_mix(split)
            osf._land_duty(reduce, fx.Int32(fx.block_idx.x), tid)

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


PREFETCH_LINE = 128


def _prefetch(base, count, offset, tid, scope):
    """Prefetch hints for the ``count`` lines ``offset(item)`` bytes past
    ``base``, a line a thread a round (a round's spare threads repeat the
    last line)."""
    for round_index in range_constexpr(-(-count // THREADS)):
        item = fx.min(tid + THREADS * round_index, count - 1)
        address = fx.Int64(base) + offset(item)
        rocdl.global_prefetch(
            llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(address)).result,
            cache_policy=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), scope),
        )


def _streamed(split, kind, slabs):
    """(first, count): the slabs of a CTA's quarter of K that ``kind``'s
    projection streams once ``x`` is in, at most ``slabs`` of them."""
    _, stages, whole = split[kind]
    quarter = tdm_split_slabs(HIDDEN)
    return stages, 0 if whole else max(0, min(quarter - stages, slabs))


def _prefetch_rows(base, groups, row_bytes, slab_bytes, window, rank, tid, scope):
    """Row-major weights (BF16, or MXFP4 pairs): slabs ``window`` of
    ``rank``'s quarter of K over the 64-row groups from ``groups``."""
    first, count = window
    quarter = tdm_split_slabs(HIDDEN)
    lines = slab_bytes // PREFETCH_LINE
    per_row = lines * count

    def offset(item):
        row = item // per_row
        rest = item % per_row
        group_row = groups[0]
        if const_expr(len(groups) == 2):
            group_row = (row < 64).select(groups[0], groups[1])
        return (
            fx.Int64(group_row + row % 64) * fx.Int64(row_bytes)
            + fx.Int64(rank * quarter + first + rest // lines) * fx.Int64(slab_bytes)
            + fx.Int64(rest % lines * PREFETCH_LINE)
        )

    _prefetch(base, 64 * len(groups) * per_row, offset, tid, scope)


def _prefetch_fp8(base, groups, window, rank, tid, scope):
    """AITER's preshuffled FP8: a 16-row task's slab is one 16 x 256 B
    block, its group's four tasks 16 rows of K apart."""
    first, count = window
    quarter = tdm_split_slabs(HIDDEN)
    block = TDM_SPLIT_SLAB * front.ROWS
    lines = block // PREFETCH_LINE
    per_task = lines * count

    def offset(item):
        task = item // per_task
        rest = item % per_task
        group_row = (task < 4).select(groups[0], groups[1])
        return (
            fx.Int64(group_row + task % 4 * front.ROWS) * fx.Int64(HIDDEN)
            + fx.Int64(rank * quarter + first + rest // lines) * fx.Int64(block)
            + fx.Int64(rest % lines * PREFETCH_LINE)
        )

    _prefetch(base, 8 * per_task, offset, tid, scope)


def build_k2_head(
    plan: Gfx1250KimiBuildPlan,
    tp: int,
    proj: int,
    count: int,
    addends: int,
    eps: float,
    out_eps: float,
    fp8_shared: bool = False,
    fp4_latent: bool = False,
    resident: bool = True,
    mix: HeadMix | None = None,
    poll_scope: int | None = None,
    prefetch: int = 0,
    prefetch_scope: int = 0,
    profile: bool = False,
) -> K2HeadKernel:
    """``k2_oproj_seam_front_gfx1250.build_oproj_seam_front(plan, tp, proj,
    count, addends, eps, out_eps, fp8_shared)`` with the split-K front: the
    same arguments and ``sync`` (``SYNC_WORDS`` i32, ``PROFILE_WORDS`` more in
    a ``profile`` build, zeroed once and used by no other kernel). ``mix``:
    a ``HeadMix`` (``row_mix`` by default). ``resident``: each front
    projection whose whole weight share fits the LDS is copied whole.
    ``fp4_latent``: the latent weight is its MXFP4 planes, the launcher's
    ``latent_weight`` and its trailing ``latent_scale``
    (``K2_HEAD_FP4_ABI``). ``prefetch``: of the slabs each front CTA streams
    once ``x`` is in, that many hinted into the cache (``prefetch_scope``)
    while it waits."""
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
    head_mix = mix
    if head_mix is None:
        head_mix = row_mix(rows, count, addends, eps, out_eps)
    if head_mix.ctas > MIX_CTAS or head_mix.ctas % sf.CLUSTER:
        raise ValueError(f"the K2 head's mix takes whole clusters of {MIX_CTAS} CTAs")
    if head_mix.flags > THREADS:
        raise ValueError("the front waits on a mix flag a thread")
    if poll_scope is None:
        poll_scope = osf.SCOPE_DEV if tp == 1 else ar.SCOPE_SYS | ar.VOLATILE
    grid = max(FRONT_CTAS + head_mix.ctas, osf.OPROJ_CTAS)
    assert grid % sf.CLUSTER == 0 and grid <= GRID
    tiles = (rows + front.ROWS - 1) // front.ROWS
    group_words = rows * 16
    budget = (plan.lds_bytes - 1024) // 4 - group_words - head_mix.mix_words
    split = front.front_split_plan(
        rows, fp8_shared, fp4_latent, budget, resident=resident
    )
    if split is None:
        raise ValueError(f"the split-K front does not fit {rows} rows")
    gemv_words = max(
        front.front_split_lds_words(rows, split),
        head_mix.gemv_words,
        osf.oproj_lds_words(rows),
    )
    words_a_bank = oar.bank_words(tp, rows)

    @fx.struct
    class Lds:
        # the o_proj's slabs, then the split front's or a mix CTA's
        gemv: fx.Array[fx.Int32, gemv_words, 16]
        group: fx.Array[fx.Int32, group_words, 16]
        mix: fx.Array[fx.Float32, head_mix.mix_words, 16]

    if lds_bytes(Lds) > plan.lds_bytes:
        raise ValueError(f"the K2 head needs {lds_bytes(Lds)} B of LDS")
    stages = "".join("r" if whole else str(depth) for _, depth, whole in split.values())
    name = (
        f"k3_k2_gfx1250_head_tp{tp}_b{count}_a{addends}_s{rows}"
        f"{'_f8' if fp8_shared else ''}{'_f4' if fp4_latent else ''}_{head_mix.name}_k{stages}_q{poll_scope & 0xFF}"
        f"{f'_p{prefetch}c{prefetch_scope}' if prefetch else ''}{'_profile' if profile else ''}"
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
    flags = head_mix.flags
    sources = SOURCES
    streamed = {kind: _streamed(split, kind, prefetch) for kind in split}
    latent_bytes = 2 if split["latent"][0] == "bf16" else 1 / 2

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
        latent_scale: Int64,
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

        def hint_streamed():
            cluster = bid // gemv_tdm.TDM_CLUSTER
            rank = bid % gemv_tdm.TDM_CLUSTER
            latent_cluster = cluster - front.SPLIT_ROUTER_CLUSTERS
            shared_cluster = latent_cluster - front.SPLIT_LATENT_CLUSTERS
            shared_rows = [
                shared_cluster * 64,
                front.SHARED_COLUMNS + shared_cluster * 64,
            ]
            if const_expr(streamed["router"][1] > 0):  # noqa: SIM102
                if cluster < front.SPLIT_ROUTER_CLUSTERS:
                    _prefetch_rows(
                        router_weight, [cluster * 64], HIDDEN * 2, TDM_SPLIT_SLAB * 2,
                        streamed["router"], rank, tid, prefetch_scope,
                    )  # fmt: skip
            if const_expr(streamed["latent"][1] > 0):  # noqa: SIM102
                if (latent_cluster >= 0) & (shared_cluster < 0):
                    _prefetch_rows(
                        latent_weight, [latent_cluster * 128, latent_cluster * 128 + 64],
                        int(HIDDEN * latent_bytes), int(TDM_SPLIT_SLAB * latent_bytes),
                        streamed["latent"], rank, tid, prefetch_scope,
                    )  # fmt: skip
            if const_expr(streamed["shared"][1] > 0):  # noqa: SIM102
                if shared_cluster >= 0:
                    if const_expr(fp8_shared):
                        _prefetch_fp8(
                            shared_weight, shared_rows, streamed["shared"], rank, tid,
                            prefetch_scope,
                        )  # fmt: skip
                    else:
                        _prefetch_rows(
                            shared_weight, shared_rows, HIDDEN * 2, TDM_SPLIT_SLAB * 2,
                            streamed["shared"], rank, tid, prefetch_scope,
                        )  # fmt: skip

        def prologue():
            # the front's weights are in flight from here
            stamp(STAMP_ISSUED)
            if const_expr(prefetch > 0):
                hint_streamed()
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
                    latent_scale,
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
        if (bid >= FRONT_CTAS) & (bid < FRONT_CTAS + head_mix.ctas):
            head_mix.body(
                lds, bid - FRONT_CTAS, reduce, sync, epoch, tid, lane, wave, stamp,
                blocks, prefix, score_weight, out_weight, x, prefix_out, token_stride,
            )  # fmt: skip
        if bid >= FRONT_CTAS + head_mix.ctas:
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
        latent_scale: Int64,
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
            latent_scale,
            front_scratch,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(grid,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    _KERNEL_ABI.check(kernel, launch)
    tail = (scratch.data_ptr(),) if fp4_latent else (0, scratch.data_ptr())
    return K2HeadKernel(
        name,
        front._TailedLauncher(launch, tail),
        (grid,),
        (THREADS,),
        head_mix,
        split,
        scratch,
        K2_HEAD_FP4_ABI if fp4_latent else K2_HEAD_ABI,
    )
