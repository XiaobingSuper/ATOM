# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's o_proj, its TP all-reduce, the MLP-side AttnRes mix and the replicated
front in one gfx1250 launch: ``k2_oproj_ar_gfx1250`` then
``k2_seam_front_gfx1250``, bit for bit.

The grid is the seam front's (mix CTAs, then the front's ``PAIRS``), at least
``k2_oproj_ar_gfx1250.CTAS`` wide. CTA ``b`` below that runs the o_proj
all-reduce's CTA ``b`` first: its 32 output columns' FP8 GEMV and the push of
their BF16 partials to the owning rank. Then each CTA takes its seam role:

- a mix CTA issues the TDM copies of its token's block rows (and prefix) and
  the two weights, sums the lines it owns and pushes them to every rank, then
  polls its token's summed lines in its own rank's region -- the attention
  row, straight from the owners' pushes -- into LDS: a token mixes once its
  own row is reduced, not the whole step. It lands its o_proj columns of the
  attention output once its ``x`` is out;
- a projection CTA issues its first weight slabs, sums and lands its o_proj
  columns as the all-reduce does, then waits for the mix CTAs' flags;
- the CTAs past both only sum and land.

The all-reduce's lines and the mix's flags carry one tag: the K2 all-reduce
counters' next epoch, as every all-reduce launch takes it. A poll reads the
line it waits for at system scope, a flag at device scope and ``x`` through
the TDM at device scope: no acquire fence.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import attn_res_gfx1250 as attn_res
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import k2_seam_front_gfx1250 as sf
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import tdm_lds_words
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import (
    _lds_barrier,
    _realtime,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    bf16_round,
    lds_bytes,
    load_ptr64,
    rsrc,
    st_dev,
    traced,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = sf.THREADS
WAVES = sf.WAVES
OPROJ_CTAS = oar.CTAS
LINE_WORDS = oar.LINE_WORDS
LINE_TOKENS = oar.LINE_TOKENS
assert sf.MAX_CTAS == ar.MAX_CTAS
# ``sync``: the seam front's flags and exchange words (its epoch words unused),
# then a profile build's stamps
# gfx12 CPol SCOPE_DEV: one rank's sums polled past its own CU's caches
SCOPE_DEV = 2 << 3
SYNC_WORDS = sf.SYNC_WORDS
PROFILE_WORDS = sf.PROFILE_WORDS
STAMPS = sf.STAMPS
# stamps past the seam front's
STAMP_PUSHED = 13
STAMP_REDUCED = 14
# a mix CTA's attention row polled in
STAMP_LANDED = 15
assert STAMP_LANDED < STAMPS

OPROJ_SEAM_FRONT_ABI = KernelAbi(
    (
        "core",
        "w_o",
        "w_o_scale",
        "attn_out",
        "peers",
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
    )
)


@dataclass(frozen=True)
class K2OprojSeamFrontKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    split: int
    token_buffers: int
    abi: KernelAbi = OPROJ_SEAM_FRONT_ABI
    launch_count: int = 1
    # its LDS struct and kernel body, for a launch that runs it as a part
    lds: object = None
    body: object = None


def oproj_lds_words(rows: int) -> int:
    """i32 LDS words of the o_proj: its FP8 GEMV, then (over its consumed
    weight slabs) the accumulators and the pairs cut into lines."""
    tiles = (rows + oar.ROWS - 1) // oar.ROWS
    gemv = tdm_lds_words(rows, 2, True, depth=gemv_tdm.FP8_DEPTH)
    epilogue = (
        2 * tiles * oar.ACCUMULATOR_WORDS
        + oar.triples(rows) * LINE_TOKENS * oar.CTA_PAIRS
    )
    return max(sum(gemv), gemv[0] + epilogue)


@dataclass(frozen=True)
class _Reduce:
    """A launch's all-reduce geometry and this CTA's view of it."""

    rows: int
    tp: int
    mine: object
    bases: list
    out: object
    rank: object
    bank: object
    epoch: object
    # ``mine`` bounded to this width's banks
    region: object
    scope: int

    @property
    def own(self):
        return oar.own_pairs(self.tp)

    @property
    def sums(self):
        """The sums' first word in a bank-relative region."""
        return self.bank + LINE_WORDS * self.tp * oar.triples(self.rows) * self.own


@traced
def _oproj_partials(lds, core, w_o, w_o_scale, proj, reduce, bid, tid, lane, wave):
    """``k2_oproj_ar_gfx1250``'s CTA ``bid`` up to its partials' push; the
    CTA's LDS free on return."""
    rows, tp, own = reduce.rows, reduce.tp, reduce.own
    tiles = (rows + oar.ROWS - 1) // oar.ROWS
    count = oar.triples(rows)
    row0 = bid * oar.CTA_COLUMNS
    owner = bid // (OPROJ_CTAS // tp)
    owner_base = reduce.bases[0]
    for p in range_constexpr(1, tp):
        owner_base = (owner == p).select(reduce.bases[p], owner_base)
    accumulators = gemv_tdm.project(
        lds,
        core,
        proj,
        rows,
        w_o,
        HIDDEN,
        proj,
        [row0, row0 + oar.ROWS],
        lane,
        wave,
        w_scale=w_o_scale,
        depth=gemv_tdm.FP8_DEPTH,
    )
    # Every peer past its waits: their token multicasts have landed here too.
    cluster.cluster_barrier()
    # the accumulators as words: the LDS region is i32
    red = lds + tdm_lds_words(rows, 2, True, depth=gemv_tdm.FP8_DEPTH)[0]
    stage = red + 2 * tiles * oar.ACCUMULATOR_WORDS
    front._stage_accumulators(accumulators, red, tiles, wave, lane)
    gpu.barrier()
    task = tid // (oar.ROWS * 8)
    local_token = tid % (oar.ROWS * 8) // 8
    row_pair = tid % 8
    for tile in range_constexpr(tiles):
        slot = red + (2 * tile + task) * oar.ACCUMULATOR_WORDS
        token = tile * oar.ROWS + local_token
        low = bf16_round(front._row_sum(slot, 2 * row_pair, local_token))
        high = bf16_round(front._row_sum(slot, 2 * row_pair + 1, local_token))
        word = (
            fx.Vector.from_elements([low, high], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        oar._store_stage(
            local_token < min(oar.ROWS, rows - tile * oar.ROWS),
            stage + token * oar.CTA_PAIRS + task * oar.ROWS // 2 + row_pair,
            word,
        )
    spare = (count * LINE_TOKENS - rows) * oar.CTA_PAIRS
    if const_expr(spare > 0):  # noqa: SIM102
        if tid < spare:
            fx.ptr_store(fx.Int32(0), stage + rows * oar.CTA_PAIRS + tid)
    gpu.barrier()
    if tid < count * oar.CTA_PAIRS:
        triple = tid // oar.CTA_PAIRS
        pair = tid % oar.CTA_PAIRS
        words = [
            fx.ptr_load(stage + (triple * LINE_TOKENS + j) * oar.CTA_PAIRS + pair)
            for j in range(LINE_TOKENS)
        ]
        owned = (row0 - owner * (2 * own)) // 2 + pair
        at = (
            fx.Int64(
                reduce.bank
                + LINE_WORDS * ((reduce.rank * count + triple) * own + owned)
            )
            * 4
        )
        ar._store_line_sys(
            owner_base + at,
            fx.Vector.from_elements([*words, reduce.epoch], fx.Int32),
        )
    gpu.barrier()


@traced
def _sum_duty(reduce, bid, tid):
    """The all-reduce's sums this CTA computes and pushes to every rank (an
    owner below ``SPREAD_ROWS`` rows lands them too)."""
    rows, tp = reduce.rows, reduce.tp
    args = (reduce.mine, reduce.bases)
    if bid < OPROJ_CTAS:
        if const_expr(rows > oar.SPREAD_ROWS):
            if tid < oar.SUM_THREADS:
                oar._reduce_share(
                    *args, reduce.rank, rows, tp, reduce.own, reduce.bank,
                    reduce.epoch, bid, tid,
                )  # fmt: skip
        else:
            if bid // (OPROJ_CTAS // tp) == reduce.rank:
                oar._reduce_owned(
                    *args, reduce.out, reduce.rank, rows, tp, reduce.own,
                    reduce.bank, reduce.epoch, bid, tid,
                )  # fmt: skip


@traced
def _land_duty(reduce, bid, tid):
    """This CTA's o_proj columns of the attention output, from their owner's
    sums, where ``_sum_duty`` did not land them."""
    rows, tp = reduce.rows, reduce.tp
    land = (
        reduce.mine, reduce.out, rows, tp, reduce.own, reduce.bank, reduce.epoch,
        bid, tid, THREADS,
    )  # fmt: skip
    if bid < OPROJ_CTAS:
        if const_expr(rows > oar.SPREAD_ROWS):
            oar._land_sums(*land)
        else:
            if bid // (OPROJ_CTAS // tp) != reduce.rank:
                oar._land_sums(*land)


@traced
def _store_word(live, address, word):
    if live:
        fx.ptr_store(word, address)


def _load_line(region, word, scope, late=None):
    return fx.Vector(
        bo.buffer_load(
            region,
            word,
            vec_width=LINE_WORDS,
            dtype=T.i32,
            mask=late,
            cache_modifier=scope,
        )
    )


@traced
def _poll_stale(region, words, epoch, scope):
    """``k2_oproj_ar_gfx1250._poll_lines``, re-loading only the lines still
    stale: a row of lines polled whole would stream it at every poll.
    ``region`` is bounded, so a fresh line's masked load stays off memory."""
    lines = [_load_line(region, word, scope) for word in words]
    while oar._stale(lines, epoch):
        rocdl.s_sleep(1)
        fresh = []
        for word, line in zip(words, lines):
            late = fx.Int32(line[LINE_TOKENS]) != epoch
            again = _load_line(region, word, scope, late)
            fresh.append(
                fx.Vector.from_elements(
                    [
                        late.select(fx.Int32(again[j]), fx.Int32(line[j]))
                        for j in range(LINE_WORDS)
                    ],
                    fx.Int32,
                )
            )
        lines = fresh
    return [[fx.Int32(line[j]) for j in range(LINE_TOKENS)] for line in lines]


@traced
def _attention_row(reduce, token, column0, layout, row, tid, stamp):
    """The token's attention row, this CTA's ``layout.width`` columns from
    ``column0``, polled from the summed lines into the LDS row ``row``; then
    the mix's own copies waited and the row visible to every wave."""
    pairs = layout.row_words
    triple = token // LINE_TOKENS
    which = token - triple * LINE_TOKENS
    picks = []
    for k in range_constexpr(-(-pairs // THREADS)):
        index = tid + k * THREADS
        picks.append((index < pairs, fx.min(index, pairs - 1)))
    lines = _poll_stale(
        reduce.region,
        [
            reduce.sums + LINE_WORDS * (triple * (HIDDEN // 2) + column0 // 2 + pair)
            for _, pair in picks
        ],
        reduce.epoch,
        reduce.scope,
    )
    for (live, pair), words in zip(picks, lines):
        word = (which == 0).select(words[0], (which == 1).select(words[1], words[2]))
        _store_word(live, row + pair, word)
    stamp(STAMP_LANDED)
    rocdl.s_wait_tensorcnt(0)
    _lds_barrier()


def build_oproj_seam_front(
    plan: Gfx1250KimiBuildPlan,
    tp: int,
    proj: int,
    count: int,
    addends: int,
    eps: float,
    out_eps: float,
    fp8_shared: bool = False,
    wide: bool = True,
    depth: int = 4,
    token_buffers: int | None = None,
    split: int | None = None,
    prefetch: int | None = None,
    poll_scope: int | None = None,
    profile: bool = False,
) -> K2OprojSeamFrontKernel:
    """``k2_oproj_ar_gfx1250.build_oproj_allreduce(plan, tp, proj)`` then
    ``k2_seam_front_gfx1250.build_seam_front(plan, count, addends, ...)``
    with the all-reduce's output as the mix's attention row: its prefix
    with no addend (a block's first layer), else its one addend after the
    prefix. ``attn_out`` is landed as the all-reduce lands it; ``sync`` is
    ``SYNC_WORDS`` i32 (``PROFILE_WORDS`` more in a ``profile`` build)
    zeroed once and used by no other kernel."""
    if plan.architecture is not GFX1250:
        raise ValueError("the o_proj seam front requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("the o_proj seam front requires 256 threads/eight waves")
    if not 1 <= count < attn_res.MAX_CANDIDATES:
        raise ValueError(
            f"the AttnRes mix holds 1 to {attn_res.MAX_CANDIDATES - 1} blocks"
        )
    if addends not in (0, 1):
        raise ValueError("the o_proj seam front's mix folds at most its attention row")
    if HIDDEN % (tp * oar.CTA_COLUMNS):
        raise ValueError(f"a rank's columns must be whole CTAs at TP{tp}")
    rows = plan.tokens
    if not 1 <= rows <= GFX1250_MAX_TOKENS:
        raise ValueError(f"the o_proj seam front runs 1 to {GFX1250_MAX_TOKENS} rows")
    if oar.triples(rows) * oar.CTA_PAIRS > THREADS:
        raise ValueError(f"the o_proj all-reduce cuts {rows} rows into a line a thread")
    split = sf.mix_split(rows) if split is None else split
    if split not in sf.SPLITS:
        raise ValueError(f"a token's mix takes {sf.SPLITS} CTAs, not {split}")
    if poll_scope is None:
        # one rank's sums are this device's own stores; a peer's only a
        # system-scope load is guaranteed to see
        poll_scope = SCOPE_DEV if tp == 1 else ar.SCOPE_SYS | ar.VOLATILE
    mixers = sf._clusters(rows * split)
    projections = mixers + front.PAIRS
    grid = max(projections, OPROJ_CTAS)
    if grid > min(plan.blocks, sf.MAX_CTAS):
        raise ValueError(f"the o_proj seam front needs {grid} resident CTAs")
    layout = sf._MixLayout(count, addends, eps, out_eps, split)
    attention = count + addends
    tiles = (rows + front.ROWS - 1) // front.ROWS
    reductions = layout.reductions
    mix_words = sf.MIX_WAVES * reductions + reductions + sf.MIX_WAVES
    words_a_bank = oar.bank_words(tp, rows)

    def staged_words(buffers):
        return sum(tdm_lds_words(rows, 2, depth=depth, token_buffers=buffers))

    def gemv_words(buffers):
        return max(
            staged_words(buffers),
            sf.mix_lds_words(count, addends, split),
            oproj_lds_words(rows),
        )

    if token_buffers is None:
        token_buffers = next(
            (
                b
                for b in (4, 3)
                if 4 * (gemv_words(b) + rows * 16 + mix_words) <= plan.lds_bytes
            ),
            2,
        )
    staged = staged_words(token_buffers)
    token_words = tdm_lds_words(rows, 2, token_buffers=token_buffers)[0]
    assert staged - token_words >= 2 * tiles * front.ACCUMULATOR_WORDS
    assert (
        sum(tdm_lds_words(rows, 2, True, gemv_tdm.FP8_DEPTH, token_buffers)) <= staged
    )
    assert mixers % sf.CLUSTER == 0 and front.SHARED_PAIR0 % sf.CLUSTER == 0
    assert OPROJ_CTAS % sf.CLUSTER == 0 and grid % sf.CLUSTER == 0

    @fx.struct
    class Lds:
        # the o_proj's slabs, then the projection's or a mix CTA's columns
        gemv: fx.Array[fx.Int32, gemv_words(token_buffers), 16]
        group: fx.Array[fx.Int32, rows * 16, 16]
        mix: fx.Array[fx.Float32, mix_words, 16]

    if lds_bytes(Lds) > plan.lds_bytes:
        raise ValueError(f"the o_proj seam front needs {lds_bytes(Lds)} B of LDS")
    name = (
        f"k3_k2_gfx1250_oproj_seam_front_tp{tp}_b{count}_a{addends}"
        f"{'_w' if wide else ''}_d{depth}_s{rows}{'_f8' if fp8_shared else ''}"
        f"_m{split}_t{token_buffers}{'' if prefetch is None else f'_p{prefetch}'}"
        f"_q{poll_scope & 0xFF}"
        f"{'_profile' if profile else ''}"
    )
    flags = rows * split
    sources = SOURCES

    @traced
    def body(
        lds,
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
        fused=None,
    ):
        """The kernel on ``lds`` (an ``Lds``). ``fused``: the launch's first
        part (``k2_one_gfx1250``): ``epoch`` the launch's all-reduce epoch,
        whose counters the launch stores itself; the front's outputs stored
        at device scope, then ``front_done()`` on each projection CTA."""
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        if const_expr(fused is None):
            epoch = (
                fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32))
                + 1
            )
        else:
            epoch = fused["epoch"]
        mine = load_ptr64(peers, rank)
        reduce = _Reduce(
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
            _sum_duty(reduce, bid, tid)
            stamp(STAMP_REDUCED)
            _land_duty(reduce, bid, tid)
            stamp(1)
            sf._await_mix(sync, epoch, flags, tid)
            stamp(2)

        def on_slab(slab):
            if slab in sf.PROFILED_SLABS:
                stamp(4 + sf.PROFILED_SLABS.index(slab))

        stamp(0)
        if bid < OPROJ_CTAS:
            _oproj_partials(
                lds.gemv.ptr, core, w_o, w_o_scale, proj, reduce, bid, tid, lane, wave
            )
            stamp(STAMP_PUSHED)
        if bid < flags:
            token = bid // split
            part = bid % split
            column0 = part * layout.width
            regions = layout.regions(lds.gemv.ptr)
            pointers = (blocks, prefix, prefix, prefix, score_weight, out_weight)
            sf._stage(
                regions,
                token,
                column0,
                wave,
                pointers,
                token_stride,
                layout,
                skip=attention,
                wait=False,
            )
            _sum_duty(reduce, bid, tid)
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
                token_stride,
                layout,
                stamp,
                stage=lambda: _attention_row(
                    reduce,
                    token,
                    column0,
                    layout,
                    regions[0] + attention * layout.row_words,
                    tid,
                    stamp,
                ),
            )
            sf._publish(sync, bid, epoch, tid)
            stamp(4)
            _land_duty(reduce, bid, tid)
        if (bid >= flags) & (bid < mixers):
            _sum_duty(reduce, bid, tid)
            sf._idle_mix(split)
            _land_duty(reduce, bid, tid)
        if (bid >= mixers) & (bid < projections):
            front._front_tasks(
                lds,
                x,
                (router_weight, latent_weight, shared_weight, shared_scale),
                (router_logits, latent_fp4, latent_scales, shared_hidden),
                bid - mixers,
                tid,
                lane,
                wave,
                rows,
                tiles,
                fp8_shared,
                prologue=prologue,
                x_cache_policy=CM_DEV,
                wide=wide,
                depth=depth,
                on_slab=on_slab if profile else None,
                token_buffers=token_buffers,
                prefetch=prefetch,
                store_cm=0 if fused is None else CM_DEV,
            )
            stamp(3)
            if const_expr(fused is not None):
                fused["front_done"]()
        if bid >= projections:
            _sum_duty(reduce, bid, tid)
            _land_duty(reduce, bid, tid)
        if const_expr(fused is None):
            gpu.barrier()
            counter = bid + tid * grid
            if counter < ar.MAX_CTAS:
                bo.buffer_store(epoch, rsrc(epochs), counter)

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
    ):
        _ = sources
        body(
            fx.SharedAllocator().allocate(Lds).peek(),
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
        )

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
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(grid,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    OPROJ_SEAM_FRONT_ABI.check(kernel, launch)
    return K2OprojSeamFrontKernel(
        name, launch, (grid,), (THREADS,), split, token_buffers, lds=Lds, body=body
    )
