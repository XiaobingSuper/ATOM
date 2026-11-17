# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MLP-side AttnRes mix and the replicated K2 front in one gfx1250 launch.

The front's ``PAIRS`` CTAs leave CUs over: the first ``mix_ctas(rows)`` CTAs
(whole clusters) mix the tokens, ``mix_split(rows)`` CTAs a token within one
cluster, and the rest run ``k2_front_tdm_gfx1250``'s task pairs.

The mix is ``attn_res_gfx1250``'s, bit for bit. Its 1024 threads' columns
fall into four turns of 256 threads -- this kernel's thread ``t`` stands for
``t + 256 v`` in turn ``v`` -- and every reduction keeps that kernel's lanes
and waves. A token's CTAs take its turns between them: each copies its
columns of the token's rows and of the two weights to LDS through the TDM and
loops over its turns there, so a cold launch fetches about one turn's code.
Its waves' sums, and later its squares, go to its cluster peers through the
L2 the cluster shares, across a cluster barrier. It writes its columns of
``x`` (and of ``prefix_out``) with one TDM store, ``x``'s at device scope,
then tags its flag with the launch's tag.

A projection CTA issues its first weight slabs, waits for every mix CTA's
flag (a lane a flag, the loads of a poll in flight together) and reads ``x``
through the TDM at device scope as the front does, ``token_buffers - 1``
slabs ahead: ``x`` lines just written land slower than the front's. No
acquire fence: the flag and ``x`` are both read past the caches that could
hold them stale, and a device-scope invalidate costs microseconds a CTA.

Tags: ``MAX_CTAS`` epoch words, all equal: CTA ``b`` reads word ``b`` at
launch, tags with one past it and stores that back into every word congruent
to ``b`` modulo the grid when done (``k2_allreduce_gfx1250``'s scheme), so a
launch's tag is the last one's plus one, whatever the width, with no reset.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import attn_res_gfx1250 as attn_res
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import tdm_lds_words
from atom.models.kimi_k3.mono.kernels.stages.k2_allreduce_gfx1250 import _pack
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import (
    _lds_barrier,
    _realtime,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    div_rn,
    exp,
    fresh,
    hw_rsq,
    ld_dev,
    lds_bytes,
    rsrc,
    st_dev,
    traced,
    uniform,
    xred,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = front.THREADS
WAVES = front.WAVES
ELEMENTS = attn_res.ELEMENTS
# attn_res_gfx1250's threads a mix thread stands for, and their waves
TURNS = attn_res.THREADS // THREADS
MIX_WAVES = attn_res.WAVES
assert TURNS * THREADS * ELEMENTS == HIDDEN
# a thread's turn's mixed values in LDS, eight words apart
STASH_WORDS = 8
CLUSTER = gemv_tdm.CLUSTER_LAUNCH["cluster"][0]
MAX_CTAS = 256
# the CTAs a token's mix may take, the most that fit first
SPLITS = (4, 2, 1)
assert all(TURNS % split == 0 and CLUSTER % split == 0 for split in SPLITS)
# gfx12 CPol SCOPE_SE: through the L2 a cluster's CTAs share
SCOPE_SE = 1 << 3
# ``sync``: the epoch words, a ready flag a mix CTA, then a token's exchanged
# wave sums and squares; a profile build's stamps (``STAMPS`` a CTA) past it.
EPOCH0 = 0
FLAGS0 = MAX_CTAS
EXCHANGE0 = FLAGS0 + MAX_CTAS
EXCHANGE_WORDS = MIX_WAVES * 2 * attn_res.MAX_CANDIDATES + MIX_WAVES
SYNC_WORDS = EXCHANGE0 + GFX1250_MAX_TOKENS * EXCHANGE_WORDS
STAMPS = 16
PROFILE_WORDS = MAX_CTAS * STAMPS
# the projection slabs a profile build stamps once landed
PROFILED_SLABS = (0, 1, 4, 8, 13)

SEAM_FRONT_ABI = KernelAbi(
    (
        "blocks",
        "prefix",
        "add",
        "add2",
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
class K2SeamFrontKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    split: int
    token_buffers: int
    abi: KernelAbi = SEAM_FRONT_ABI
    launch_count: int = 1


def _clusters(ctas: int) -> int:
    return -(-ctas // CLUSTER) * CLUSTER


def mix_split(rows: int) -> int:
    """The CTAs a token's mix takes at ``rows`` tokens: the most whose whole
    clusters still leave the front's ``PAIRS`` CTAs resident."""
    return next((s for s in SPLITS if _clusters(rows * s) + front.PAIRS <= MAX_CTAS), 0)


def mix_ctas(rows: int) -> int:
    return _clusters(rows * mix_split(rows))


def mix_lds_words(count: int, addends: int, split: int) -> int:
    """i32 LDS words a mix CTA stages for its columns: the token's ``count``
    block rows, its prefix and addends, the FP32 score and BF16 out weights,
    its turns' stash and its BF16 ``prefix_out`` and ``x`` columns."""
    row_words = HIDDEN // split // 2
    return (
        (count + 1 + addends) * row_words
        + 2 * row_words
        + row_words
        + TURNS // split * THREADS * STASH_WORDS
        + 2 * row_words
    )


def _sum(a, b):
    return a + b


def _xsum(value, offset):
    """``xred(value, offset, +)`` for GFX1250: inside a 16-lane row one
    v_add_f32 with DPP row_xmask, the same sum (addition commutes); as asm,
    so LLVM cannot pair it into a packed add the DPP does not fold into."""
    if offset >= 16:
        return xred(value, offset, _sum, GFX1250)
    return fx.Float32(
        llvm.inline_asm(
            T.f32,
            [fx.Float32(value).ir_value()],
            f"v_add_f32_dpp $0, $1, $1 row_xmask:{offset} row_mask:0xf bank_mask:0xf",
            "=v,v",
        )
    )


def _wave_sums(values):
    """``attn_res_gfx1250._wave_sums`` of ``values``, bit for bit."""
    for offset in GFX1250.reduction_offsets:
        values = [_xsum(v, offset) for v in values]
    return values


def _selectors(tid):
    """v_perm_b32 selectors of ``_lds_seven``'s even and odd values: a
    thread's seven columns start odd where its index is odd, so its value
    ``j`` is the BF16 ``j + 1`` of the four words, else ``j``."""
    odd = (tid % 2) == 1
    return (
        odd.select(fx.Int32(0x03020C0C), fx.Int32(0x01000C0C)),
        odd.select(fx.Int32(0x05040C0C), fx.Int32(0x03020C0C)),
    )


def _perm(high, low, selector):
    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.perm",
            [as_ir_value(high), as_ir_value(low), as_ir_value(selector)],
            [],
            [],
        )
    )


def _lds_seven(row, column, selectors):
    """``k2_allreduce_gfx1250._seven`` of an LDS BF16 row at ``row``: each
    FP32 value one v_perm_b32 of the BF16 into the high half."""
    words = [fx.Int32(fx.ptr_load(row + column // 2 + w)) for w in range_constexpr(4)]
    return [
        _perm(words[min(j // 2 + 1, 3)], words[j // 2], selectors[j % 2]).bitcast(
            fx.Float32
        )
        for j in range_constexpr(ELEMENTS)
    ]


@traced
def _lds_put_seven(row, column, values):
    """``k2_allreduce_gfx1250._store_seven`` of seven BF16 into an LDS row in
    words: a column's word shared with the next thread (lane ^ 1, whose seven
    start odd) is the even thread's, with that thread's first value."""
    first = xred(values[0], 1, lambda a, b: b, GFX1250)
    if column % 2 == 0:
        pairs = [(values[2 * w], values[2 * w + 1]) for w in range_constexpr(3)]
        for w in range_constexpr(4):
            low, high = pairs[w] if w < 3 else (values[6], first)
            fx.ptr_store(_pack(low, high), row + column // 2 + w)
    else:
        for w in range_constexpr(3):
            fx.ptr_store(
                _pack(values[2 * w + 1], values[2 * w + 2]),
                row + (column + 1) // 2 + w,
            )


@traced
def _lane0_store(lane, values, pointer, offsets):
    """``values[i]`` at ``pointer + offsets[i]``, from lane 0."""
    if lane == 0:
        for i in range_constexpr(len(values)):
            fx.ptr_store(values[i], pointer + offsets[i])


@dataclass(frozen=True)
class _MixLayout:
    """A mix CTA's LDS (``staged`` onward) and its share of the token."""

    count: int
    addends: int
    eps: float
    out_eps: float
    split: int

    @property
    def width(self):
        return HIDDEN // self.split

    @property
    def row_words(self):
        return self.width // 2

    @property
    def turns(self):
        return TURNS // self.split

    @property
    def reductions(self):
        return 2 * (self.count + 1)

    def regions(self, staged):
        rows = staged
        scores = rows + (self.count + 1 + self.addends) * self.row_words
        gains = scores + self.width
        stash = gains + self.row_words
        folded = stash + self.turns * THREADS * STASH_WORDS
        normed = folded + self.row_words
        return rows, scores, gains, stash, folded, normed


def _candidates(rows, column, layout, selectors):
    """The token's block rows then its prefix with the addends folded in,
    seven columns from (CTA-local) ``column``: attn_res_gfx1250's ``rows``."""
    count, step = layout.count, layout.row_words
    candidates = [
        _lds_seven(rows + b * step, column, selectors) for b in range_constexpr(count)
    ]
    last = _lds_seven(rows + count * step, column, selectors)
    for addend in range_constexpr(layout.addends):
        more = _lds_seven(rows + (count + 1 + addend) * step, column, selectors)
        last = [a + b for a, b in zip(last, more)]
    return candidates + [last]


def _partials(regions, partial, first_turn, tid, lane, wave, turn, layout, selectors):
    """Turn ``turn``'s two sums of every candidate, over its wave, at its
    attn_res_gfx1250 wave's row of ``partial``; the folded prefix to LDS."""
    rows, scores, _, _, folded, _ = regions
    reductions = layout.reductions
    turn = fx.Int32(turn)
    column = (turn * THREADS + tid) * ELEMENTS
    candidates = _candidates(rows, column, layout, selectors)
    if layout.addends > 0:
        _lds_put_seven(folded, column, candidates[-1])
    weight = [
        fx.Int32(fx.ptr_load(scores + column + j)).bitcast(fx.Float32)
        for j in range_constexpr(ELEMENTS)
    ]
    sums = []
    for row in candidates:
        square = row[0] * row[0]
        dot = row[0] * weight[0]
        for j in range_constexpr(1, ELEMENTS):
            square = square + row[j] * row[j]
            dot = dot + row[j] * weight[j]
        sums += [square, dot]
    sums = _wave_sums(sums)
    base = ((first_turn + turn) * WAVES + wave) * reductions
    _lane0_store(lane, sums, partial, [base + k for k in range_constexpr(reductions)])


def _mixed(regions, squares, odds, division, first_turn, tid, lane, wave, turn, layout, selectors):  # fmt: skip
    """Turn ``turn``'s mixed columns to the stash and their squares' sum over
    the wave at its attn_res_gfx1250 wave's word of ``squares``.
    ``division``: the softmax denominator and its reciprocal."""
    rows, _, _, stash, _, _ = regions
    denominator, reciprocal = division
    turn = fx.Int32(turn)
    column = (turn * THREADS + tid) * ELEMENTS
    candidates = _candidates(rows, column, layout, selectors)
    mixed = []
    for j in range_constexpr(ELEMENTS):
        total = odds[0] * candidates[0][j]
        for c in range_constexpr(1, layout.count + 1):
            total = total + odds[c] * candidates[c][j]
        mixed.append(div_rn(total, denominator, reciprocal))
    square = mixed[0] * mixed[0]
    for value in mixed[1:]:
        square = square + value * value
    square = _wave_sums([square])
    _lane0_store(lane, square, squares, [(first_turn + turn) * WAVES + wave])
    words = [m.bitcast(fx.Int32) for m in mixed] + [fx.Int32(0)]
    slot = stash + (turn * THREADS + tid) * STASH_WORDS
    for half in range_constexpr(2):
        fx.ptr_store(
            fx.Vector.from_elements(words[4 * half : 4 * half + 4], fx.Int32),
            slot + 4 * half,
        )


def _output(regions, inverse, tid, turn, selectors):
    """Turn ``turn``'s columns of ``x`` to LDS: the stashed mix, normed."""
    _, _, gains, stash, _, normed = regions
    turn = fx.Int32(turn)
    column = (turn * THREADS + tid) * ELEMENTS
    slot = stash + (turn * THREADS + tid) * STASH_WORDS
    words = []
    for half in range_constexpr(2):
        words.extend(
            fx.Vector(
                fx.ptr_load(
                    slot + 4 * half, result_type=fx.Vector.make_type(4, fx.Int32)
                )
            )[i]
            for i in range_constexpr(4)
        )
    mixed = [fx.Int32(w).bitcast(fx.Float32) for w in words[:ELEMENTS]]
    gain = _lds_seven(gains, column, selectors)
    _lds_put_seven(normed, column, [m * inverse * g for m, g in zip(mixed, gain)])


@traced
def _tdm_row(wave, issuer, address, row, width, row_stride, rows, cache_policy, store):
    """Wave ``issuer``'s TDM copy between ``rows`` rows of ``width`` BF16 at
    ``address`` (``row_stride`` apart) and packed LDS rows at ``row``."""
    if wave == issuer:
        gemv_tdm.load_rows(
            address,
            row_stride,
            row,
            rows,
            rows,
            padded=False,
            cache_policy=cache_policy,
            whole=True,
            width=width,
            store=store,
        )


def _stage(regions, token, column0, wave, pointers, token_stride, layout, skip=None, wait=True):  # fmt: skip
    """The CTA's columns of the token's rows and the two weights to LDS, a
    TDM copy a wave. ``skip``: a candidate row (``layout.count`` the prefix,
    past it the addends) the caller lands itself; ``wait``: the copies done
    on return."""
    blocks, prefix, add, add2, score_weight, out_weight = pointers
    rows, scores, gains = regions[:3]
    width, step = layout.width, layout.row_words
    singles = [prefix, add, add2][: 1 + layout.addends]
    copies = [
        (blocks + fx.Int64(token * token_stride + column0) * 2, rows, layout.count)
    ]
    copies += [
        (
            source + fx.Int64(token * HIDDEN + column0) * 2,
            rows + (layout.count + s) * step,
            1,
        )
        for s, source in enumerate(singles)
        if layout.count + s != skip
    ]
    for issuer, (address, row, count) in enumerate(copies):
        _tdm_row(wave, issuer, address, row, width, HIDDEN, count, 0, False)
    issuer = len(copies)
    _tdm_row(
        wave,
        issuer,
        score_weight + fx.Int64(column0) * 4,
        scores,
        2 * width,
        2 * HIDDEN,
        1,
        0,
        False,
    )
    _tdm_row(
        wave,
        issuer + 1,
        out_weight + fx.Int64(column0) * 2,
        gains,
        width,
        HIDDEN,
        1,
        0,
        False,
    )
    if const_expr(wait):
        rocdl.s_wait_tensorcnt(0)
        _lds_barrier()


@traced
def _exchange(sync, base, lds, own0, own, total, tid):
    """The CTA's ``own`` words of ``lds`` from ``own0`` out to its cluster
    peers and theirs of the ``total`` in, across a cluster barrier."""
    for i in range_constexpr(-(-own // THREADS)):
        k = tid + i * THREADS
        if k < own:
            bo.buffer_store(
                fx.ptr_load(lds + own0 + k),
                rsrc(sync),
                base + own0 + k,
                cache_modifier=SCOPE_SE,
            )
    cluster.cluster_barrier()
    for i in range_constexpr(-(-total // THREADS)):
        k = tid + i * THREADS
        if (k < total) & ((k < own0) | (k >= own0 + own)):
            fx.ptr_store(
                fx.Float32(
                    bo.buffer_load(
                        rsrc(sync),
                        base + k,
                        vec_width=1,
                        dtype=T.f32,
                        cache_modifier=SCOPE_SE,
                    )
                ),
                lds + k,
            )
    _lds_barrier()


@traced
def _mix_token(mix, staged, sync, token, part, tid, lane, wave, pointers, token_stride, layout, stamp, stage=None):  # fmt: skip
    """``attn_res_gfx1250``'s mix of ``token``, part ``part`` of its columns
    (``layout.turns`` turns of this CTA's); this part's ``x`` columns stored
    at device scope, the TDM store done, on return. ``stage()``: the part's
    rows landed in LDS by the caller, else ``_stage``'s copies."""
    blocks, prefix, add, add2, score_weight, out_weight, x, prefix_out = pointers
    split, reductions = layout.split, layout.reductions
    candidates = layout.count + 1
    squares = mix
    partial = squares + MIX_WAVES
    total = partial + MIX_WAVES * reductions
    regions = _MixLayout.regions(layout, staged)
    folded, normed = regions[4:]
    first_turn = part * layout.turns
    column0 = part * layout.width
    exchange = EXCHANGE0 + token * EXCHANGE_WORDS
    # opaque, so LLVM keeps each loop of turns one turn's code
    turns = uniform(fresh(layout.turns))
    selectors = _selectors(tid)
    if const_expr(stage is None):
        _stage(
            regions,
            token,
            column0,
            wave,
            (blocks, prefix, add, add2, score_weight, out_weight),
            token_stride,
            layout,
        )
    else:
        stage()
    stamp(1)
    for turn in range(turns):
        _partials(
            regions, partial, first_turn, tid, lane, wave, turn, layout, selectors
        )
        stamp(5 + fx.Int32(turn))
    _lds_barrier()
    if const_expr(layout.addends > 0):
        _tdm_row(
            wave,
            WAVES - 1,
            prefix_out + fx.Int64(token * HIDDEN + column0) * 2,
            folded,
            layout.width,
            HIDDEN,
            1,
            0,
            True,
        )
    if const_expr(split > 1):
        own = layout.turns * WAVES * reductions
        _exchange(sync, exchange, partial, part * own, own, MIX_WAVES * reductions, tid)
    for r in range_constexpr(-(-reductions // WAVES)):
        _reduce_round(partial, total, reductions, wave + r * WAVES, lane)
    _lds_barrier()
    stamp(2)
    totals = [fx.ptr_load(total + k) for k in range_constexpr(reductions)]
    weighted = [
        totals[2 * c + 1]
        * hw_rsq(totals[2 * c] * fx.Float32(1.0 / HIDDEN) + fx.Float32(layout.eps))
        for c in range_constexpr(candidates)
    ]
    top = weighted[0]
    for score in weighted[1:]:
        top = fx.max(top, score)
    odds = [exp(score - top) for score in weighted]
    denominator = odds[0]
    for p in odds[1:]:
        denominator = denominator + p
    division = (denominator, fx.Float32(1.0) / denominator)
    for turn in range(turns):
        _mixed(
            regions, squares, odds, division, first_turn, tid, lane, wave, turn,
            layout, selectors,
        )  # fmt: skip
        stamp(9 + fx.Int32(turn))
    _lds_barrier()
    if const_expr(split > 1):
        own = layout.turns * WAVES
        _exchange(
            sync,
            exchange + MIX_WAVES * reductions,
            squares,
            part * own,
            own,
            MIX_WAVES,
            tid,
        )
    quads = fx.Vector(
        fx.ptr_load(squares, result_type=fx.Vector.make_type(MIX_WAVES, fx.Float32))
    )
    norm = fx.Float32(quads[0])
    for w in range_constexpr(1, MIX_WAVES):
        norm = norm + fx.Float32(quads[w])
    inverse = hw_rsq(norm * fx.Float32(1.0 / HIDDEN) + fx.Float32(layout.out_eps))
    for turn in range(turns):
        _output(regions, inverse, tid, turn, selectors)
    _lds_barrier()
    _tdm_row(
        wave,
        0,
        x + fx.Int64(token * HIDDEN + column0) * 2,
        normed,
        layout.width,
        HIDDEN,
        1,
        CM_DEV,
        True,
    )
    rocdl.s_wait_tensorcnt(0)
    stamp(3)


@traced
def _reduce_round(partial, total, reductions, k, lane):
    """attn_res_gfx1250's second round for reduction ``k``: its 32 waves'
    sums, a lane each, butterflied."""
    if k < reductions:
        value = _wave_sums([fx.ptr_load(partial + lane * reductions + k)])[0]
        if lane == 0:
            fx.ptr_store(value, total + k)


@traced
def _idle_mix(split):
    """A cluster's spare mix CTA: its peers' cluster barriers, no mix."""
    if const_expr(split > 1):
        cluster.cluster_barrier()
        cluster.cluster_barrier()


@traced
def _publish(sync, flag, tag, tid):
    """The CTA's flag, once its ``x`` columns are at device scope: the TDM
    store's wave waited on its SCOPE_DEV completion before this barrier."""
    rocdl.s_wait_storecnt(0)
    _lds_barrier()
    if tid == 0:
        st_dev(tag, sync, FLAGS0 + flag)


@traced
def _await_mix(sync, tag, flags, tid):
    """Until every mix CTA's flag carries ``tag``: a lane a flag, the loads
    of a poll in flight together."""
    if tid < flags:
        flag = fx.Int32(ld_dev(sync, FLAGS0 + tid))
        while flag != tag:
            rocdl.s_sleep(1)
            flag = fx.Int32(ld_dev(sync, FLAGS0 + tid))
    gpu.barrier()


def build_seam_front(
    plan: Gfx1250KimiBuildPlan,
    count: int,
    addends: int,
    eps: float,
    out_eps: float,
    fp8_shared: bool = False,
    wide: bool = True,
    depth: int = 4,
    token_buffers: int | None = None,
    split: int | None = None,
    profile: bool = False,
) -> K2SeamFrontKernel:
    """``attn_res_gfx1250.build_attn_res(count, addends, True, eps, out_eps)``
    then ``k2_front_tdm_gfx1250.build_replicated_front(plan, fp8_shared,
    wide, depth)`` as one launch: the mix's arguments then ``x`` (the mix's
    output, the front's input), ``sync`` (``SYNC_WORDS`` i32 zeroed once, and
    ``PROFILE_WORDS`` more in a ``profile`` build) and the front's weights and
    outputs. ``token_buffers``: the projection's token slabs, by default the
    most of 4 that fit; ``split``: a token's mix CTAs, ``mix_split``'s by
    default."""
    if plan.architecture is not GFX1250:
        raise ValueError("the seam front requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("the seam front requires 256 threads/eight waves")
    if not 1 <= count < attn_res.MAX_CANDIDATES:
        raise ValueError(
            f"the AttnRes mix holds 1 to {attn_res.MAX_CANDIDATES - 1} blocks"
        )
    if addends not in (0, 1, 2):
        raise ValueError("the AttnRes mix folds up to two addends")
    rows = plan.tokens
    if not 1 <= rows <= GFX1250_MAX_TOKENS:
        raise ValueError(f"the seam front mixes 1 to {GFX1250_MAX_TOKENS} rows")
    split = mix_split(rows) if split is None else split
    if split not in SPLITS:
        raise ValueError(f"a token's mix takes {SPLITS} CTAs, not {split}")
    mixers = _clusters(rows * split)
    grid = mixers + front.PAIRS
    if grid > min(plan.blocks, MAX_CTAS):
        raise ValueError(f"the seam front needs {grid} resident CTAs")
    layout = _MixLayout(count, addends, eps, out_eps, split)
    tiles = (rows + front.ROWS - 1) // front.ROWS
    reductions = layout.reductions
    mix_words = MIX_WAVES * reductions + reductions + MIX_WAVES

    def staged_words(buffers):
        return sum(tdm_lds_words(rows, 2, depth=depth, token_buffers=buffers))

    def gemv_words(buffers):
        return max(staged_words(buffers), mix_lds_words(count, addends, split))

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
    # Mix CTAs fill whole clusters, so the projection pairs keep the front's
    # cluster boundaries.
    assert mixers % CLUSTER == 0 and front.SHARED_PAIR0 % CLUSTER == 0

    @fx.struct
    class Lds:
        # the projection's slabs, or a mix CTA's staged columns
        gemv: fx.Array[fx.Int32, gemv_words(token_buffers), 16]
        group: fx.Array[fx.Int32, rows * 16, 16]
        mix: fx.Array[fx.Float32, mix_words, 16]

    if lds_bytes(Lds) > plan.lds_bytes:
        raise ValueError(f"the seam front needs {lds_bytes(Lds)} B of LDS")
    name = (
        f"k3_k2_gfx1250_seam_front_b{count}_a{addends}"
        f"{'_w' if wide else ''}_d{depth}_s{rows}{'_f8' if fp8_shared else ''}"
        f"_m{split}_t{token_buffers}{'_profile' if profile else ''}"
    )
    flags = rows * split
    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        blocks: Int64,
        prefix: Int64,
        add: Int64,
        add2: Int64,
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
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        lds = fx.SharedAllocator().allocate(Lds).peek()
        tag = fx.Int32(ld_dev(sync, EPOCH0 + bid)) + 1

        def stamp(index):
            if const_expr(profile):  # noqa: SIM102
                if tid == 0:
                    st_dev(
                        fx.Int32(_realtime()),
                        sync,
                        SYNC_WORDS + bid * STAMPS + index,
                    )

        def prologue():
            stamp(1)
            _await_mix(sync, tag, flags, tid)
            stamp(2)

        def on_slab(slab):
            if slab in PROFILED_SLABS:
                stamp(4 + PROFILED_SLABS.index(slab))

        stamp(0)
        if bid < flags:
            _mix_token(
                lds.mix.ptr,
                lds.gemv.ptr,
                sync,
                bid // split,
                bid % split,
                tid,
                lane,
                wave,
                (
                    blocks,
                    prefix,
                    add,
                    add2,
                    score_weight,
                    out_weight,
                    x,
                    prefix_out,
                ),
                token_stride,
                layout,
                stamp,
            )
            _publish(sync, bid, tag, tid)
            stamp(4)
        if (bid >= flags) & (bid < mixers):
            _idle_mix(split)
        if bid >= mixers:
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
            )
            stamp(3)
        counter = bid + tid * grid
        if counter < MAX_CTAS:
            st_dev(tag, sync, EPOCH0 + counter)

    @flyc.jit
    def launch(
        blocks: Int64,
        prefix: Int64,
        add: Int64,
        add2: Int64,
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
            blocks,
            prefix,
            add,
            add2,
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

    SEAM_FRONT_ABI.check(kernel, launch)
    return K2SeamFrontKernel(name, launch, (grid,), (THREADS,), split, token_buffers)
