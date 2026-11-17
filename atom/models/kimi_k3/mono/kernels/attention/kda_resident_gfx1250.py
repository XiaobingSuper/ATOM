# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-launch resident gfx1250 Kimi-K3 K1."""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import GFX1250_BLOCKS
from atom.models.kimi_k3.mono.kernels.attention import kda as base
from atom.models.kimi_k3.mono.kernels.attention import (
    kda_gfx1250 as gfx_backend,
)
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    HIDDEN,
    K1_ABI_NAMES,
    PROJECTION_TASKS_PER_CTA,
    SEAM_TOKEN_BUFFERS,
    K1KernelDeclaration,
    K1LocalBuildSpec,
    local_scratch_layout,
)
from atom.models.kimi_k3.mono.kernels.attention.protocol import (
    ATTENTION_CTAS,
    HEAD_DIM,
    HEADS,
    NPROJ,
    PROJECTION_TASKS,
    STATE_CHUNKS,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    fp8_pack4,
    hw_rcp,
    ld_i32,
    rsrc,
    traced,
    uniform,
    wave_max,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.plan.build_key import key_tuple
from atom.mono.runtime.abi import KernelAbi

FUSED_ROWS = 6284
ROWS = 16
PROJECTION_ROWS_PER_CTA = PROJECTION_TASKS_PER_CTA
VALUE_CHUNK = 16
PROJ = HEADS * HEAD_DIM
PTPC_FP8_MAX = 448.0
_STREAM = fx.Stream(None)
_BUILDS: dict[tuple[object, ...], object] = {}
LOCAL_ABI = KernelAbi(K1_ABI_NAMES)


@dataclass(frozen=True)
class ResidentBuildKey:
    tokens: int
    qlen: int
    nblocks: int
    threads: int
    blocks: int
    lds_bytes: int
    delta: bool
    state_len: int
    eps: float
    out_eps: float
    onorm_eps: float
    lower_bound: float
    write_idx: int
    quant_core: bool
    debug: bool
    kda0: int
    gemv_unit: int
    external_seam: bool
    tdm_projection: bool
    fp8_projection: bool


@dataclass(frozen=True)
class ResidentK1EntryPoint:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    abi: KernelAbi = LOCAL_ABI
    launch_count: int = 1
    complete: bool = True


def _require_contract(
    plan: Gfx1250KimiBuildPlan,
    declaration: K1KernelDeclaration,
    spec: K1LocalBuildSpec,
) -> None:
    spec.validate(plan)
    if plan.architecture is not GFX1250:
        raise ValueError("resident K1 requires gfx1250")
    if (plan.threads, plan.blocks) != (declaration.threads, declaration.blocks):
        raise ValueError("resident K1 declaration geometry drifted from its plan")
    if declaration.unsupported_device_protocol:
        raise ValueError("resident local K1 must not contain peer transport")
    if declaration.threads != 256 or declaration.waves != 8:
        raise ValueError("resident K1 requires 256 threads/eight wave32 waves")
    if declaration.blocks != GFX1250_BLOCKS:
        raise ValueError(f"resident K1 requires {GFX1250_BLOCKS} all-resident CTAs")
    if declaration.gemv_unit != spec.gemv_unit:
        raise ValueError("resident K1 GEMV unit drifted from its declaration")


# The thread that arrives: the last wave's first lane, whose first TDM copy in
# ``gemv_tdm.project``'s issue order is a weight slab several slabs ahead, so
# the atomic's round trip holds up no copy slab 0 waits on.
_ARRIVER = 7 * GFX1250.wave_size


@traced
def _arrive(sync, slots, blocks):
    """The launch's rendezvous without a wait: one thread counts its CTA in
    and keeps the generation past the one it found, and the last arrival
    opens that one for the next launch. ``sync`` is the 64-bit word
    {arrivals, generation}, so one fetch-and-add both counts the CTA in and
    finds the generation, and the last arrival's reset follows every other
    one; the previous launch, stream-ordered, reads no pair any more, so the
    launch may publish under its generation at once. Nothing is acquired or
    released: kernel boundaries order the launches, and on gfx1250 either
    would invalidate or write back the caches under every CTA's first loads."""

    tid = fx.thread_idx.x
    if tid == _ARRIVER:
        found = gfx_backend.epoch_arrive_device_atomic(sync, fx.Int64(1))
        previous = fx.Int32(found >> fx.Int64(32))
        if fx.Int32(found) == fx.Int32(blocks - 1):
            gfx_backend.mailbox_store_device(
                sync, 0, fx.Int32(0), previous + fx.Int32(1)
            )
        fx.ptr_store(fx.Int64(previous + fx.Int32(1)), slots)


@traced
def _arrive_unless_seam(sync, slots, blocks, block, tokens):
    if block >= tokens:
        _arrive(sync, slots, blocks)


@traced
def _generation(slots):
    """The generation ``_arrive`` kept, once its thread has."""

    gpu.barrier()
    return fx.Int32(fx.ptr_load(slots))


@traced
def _await_tokens(x_ready, tokens, tag):
    """Every seam token's ``x`` row published under ``tag``. No acquire
    fence: on gfx1250 it invalidates the XCD's L2, which every projection CTA
    would do in turn (and the threads past ``tokens`` at once, under the
    seam's loads); ``x`` is read at device scope instead."""

    if fx.thread_idx.x < tokens:
        gfx_backend.mailbox_poll_device(x_ready, fx.thread_idx.x, tag)
    gpu.barrier()


def _task_start(block, first, blocks):
    return (block - first + blocks) % blocks


def build_resident_k1(
    plan: Gfx1250KimiBuildPlan,
    declaration: K1KernelDeclaration,
    spec: K1LocalBuildSpec,
) -> ResidentK1EntryPoint:
    """Emit the complete rank-local K1 as one all-resident launch."""

    _require_contract(plan, declaration, spec)
    cache_key = (
        declaration.tokens,
        declaration.qlen,
        declaration.nblocks,
        declaration.threads,
        declaration.blocks,
        plan.lds_bytes,
        spec,
    )
    cached = _BUILDS.get(cache_key)
    if cached is not None:
        return cached

    backend = plan.kda_backend()
    compute = declaration.bind_compute_backend(backend)
    device = declaration.bind_device_protocol(backend)
    lds_types = device["emit_role_lds_types"](declaration.lds_type_spec)
    layout = local_scratch_layout(declaration)
    slabs = declaration.lds.arena.gemv_slabs[0]
    tile_starts = tuple(tile.row_start for tile in declaration.token_tiles)
    tile_rows = tuple(tile.rows for tile in declaration.token_tiles)
    if not spec.external_seam and declaration.tokens > ATTENTION_CTAS:
        raise ValueError("resident K1's seam tokens outnumber its attention CTAs")
    group_columns = plan.kda_matrix_contract().group_columns
    s = declaration.tokens
    qlen = declaration.qlen
    sources = declaration.nblocks + 1
    task_count = (s // qlen) * HEADS * STATE_CHUNKS
    keyed = key_tuple(
        ResidentBuildKey(
            s,
            qlen,
            declaration.nblocks,
            declaration.threads,
            declaration.blocks,
            plan.lds_bytes,
            spec.delta,
            spec.state_len,
            spec.eps,
            spec.out_eps,
            spec.onorm_eps,
            spec.lower_bound,
            spec.write_idx,
            spec.quant_core,
            spec.debug,
            spec.kda0,
            spec.gemv_unit,
            spec.external_seam,
            spec.tdm_projection,
            spec.fp8_projection,
        ),
        SOURCES,
    )
    name = (
        f"k3_k1_gfx1250_resident_s{s}_l{qlen}_nb{declaration.nblocks}"
        f"_u{spec.gemv_unit}_q{int(spec.quant_core)}"
        f"{'_xs' if spec.external_seam else ''}{'_t' if spec.tdm_projection else ''}"
        f"{'_f8' if spec.fp8_projection else ''}"
    )

    @traced
    def stage_x_tile(x, xl, tid, tile_start, rows, first_group, groups):
        columns = groups * group_columns
        words = rows * columns // 8
        for iteration in range_constexpr(
            (words + declaration.threads - 1) // declaration.threads
        ):
            work = tid + declaration.threads * iteration
            if work < words:
                local_token = work // (columns // 8)
                column = (work % (columns // 8)) * 8
                global_token = tile_start + local_token
                global_column = first_group * group_columns + column
                value = fx.Vector(
                    bo.buffer_load(
                        rsrc(x),
                        (global_token * HIDDEN + global_column) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                fx.ptr_store(
                    value,
                    xl + (local_token * declaration.xrow + column) // 2,
                )
        gpu.barrier()

    @traced
    def conv_metadata(context, request, head, chunk):
        tid = context["tid"]
        args = context["args"]
        metadata = {}
        metadata["acc_idx"] = uniform(ld_i32(args["num_acc"], request)) - 1
        metadata["slot_c"] = uniform(
            ld_i32(args["st_idx"], request * args["st_stride"])
        )
        channel_thread = fx.min(tid, 3 * VALUE_CHUNK - 1)
        metadata["part"] = channel_thread // VALUE_CHUNK
        metadata["chan"] = (
            metadata["part"] * (HEADS * HEAD_DIM)
            + head * HEAD_DIM
            + chunk * VALUE_CHUNK
            + channel_thread % VALUE_CHUNK
        )
        metadata["cs"] = (
            args["conv_st"]
            + fx.Int64(fx.max(metadata["slot_c"], 0)) * fx.Int64(args["cs_seq"]) * 2
        )
        metadata["co"] = metadata["chan"] * args["cs_dim"]
        metadata["cw"] = fx.Vector(
            bo.buffer_load(
                rsrc(args["conv_w"]),
                metadata["chan"] * 4,
                vec_width=4,
                dtype=T.f32,
            )
        )
        metadata["win"] = [
            base.ld_bf(
                metadata["cs"],
                metadata["co"] + (metadata["acc_idx"] + index) * args["cs_tok"],
            )
            for index in range(3)
        ]
        metadata["old"] = [
            base.ld_bf(
                metadata["cs"],
                metadata["co"] + (metadata["acc_idx"] + 1 + index) * args["cs_tok"],
            )
            for index in range(spec.state_len - qlen)
        ]
        return metadata

    @traced
    def gate_metadata(context, head):
        args = context["args"]
        metadata = {}
        metadata["fb"], metadata["dtb"] = compute["kda_gate_loads"](
            context,
            head,
        )
        metadata["aa"] = base.hw_exp2(
            base.ld_f32(args["a_log"], head) * fx.Float32(base.LOG2E)
        )
        return metadata

    @traced
    def load_state_metadata(context, metadata, request, head, chunk):
        tid = context["tid"]
        args = context["args"]
        metadata["acc_idx"] = uniform(ld_i32(args["num_acc"], request)) - 1
        metadata["slot0"] = uniform(
            ld_i32(
                args["st_idx"],
                request * args["st_stride"] + metadata["acc_idx"],
            )
        )
        if tid < qlen:
            fx.ptr_store(
                ld_i32(
                    args["st_idx"],
                    request * args["st_stride"] + tid,
                ),
                context["slots"] + tid,
            )
        gpu.barrier()
        compute["kda_state_loads"](context, metadata, head, chunk)
        return metadata

    @traced
    def quantize_core(token, tid, lane, wave, red, core_out, core_q):
        values = []
        maximum = fx.Float32(0.0)
        for iteration in range_constexpr(
            (PROJ + declaration.threads * 4 - 1) // (declaration.threads * 4)
        ):
            element = (iteration * declaration.threads + tid) * 4
            live = element < PROJ
            # Other CTAs' device-scope stores, read at device scope: no fence.
            words = fx.Vector(
                bo.buffer_load(
                    rsrc(core_out),
                    (token * PROJ + fx.min(element, PROJ - 4)) // 2,
                    vec_width=2,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            loaded = [
                bf_lo(words[0]),
                bf_hi(words[0]),
                bf_lo(words[1]),
                bf_hi(words[1]),
            ]
            loaded = [live.select(value, fx.Float32(0.0)) for value in loaded]
            values.extend(loaded)
            for value in loaded:
                maximum = fx.max(maximum, fmath.absf(value))

        maximum = wave_max(maximum, GFX1250)
        gpu.barrier()
        if lane == 0:
            fx.ptr_store(maximum, red + wave)
        gpu.barrier()
        maximum = fx.ptr_load(red)
        for other_wave in range_constexpr(1, declaration.waves):
            maximum = fx.max(maximum, fx.ptr_load(red + other_wave))
        scale = maximum * fx.Float32(1.0 / PTPC_FP8_MAX)
        if tid == 0:
            bo.buffer_store(
                scale,
                rsrc(core_q + fx.Int64(s * PROJ)),
                token,
            )
        inverse = hw_rcp(scale)

        for iteration in range_constexpr(len(values) // 4):
            element = (iteration * declaration.threads + tid) * 4
            if element < PROJ:
                quantized = [
                    fx.max(
                        fx.min(
                            values[iteration * 4 + offset] * inverse,
                            fx.Float32(PTPC_FP8_MAX),
                        ),
                        fx.Float32(-PTPC_FP8_MAX),
                    )
                    for offset in range(4)
                ]
                bo.buffer_store(
                    fp8_pack4(*quantized),
                    rsrc(core_q),
                    (token * PROJ + element) // 4,
                    cache_modifier=CM_DEV,
                )
        device["wait_vector_stores"]()

    def open_generation(sync, slots, block):
        """The arrival of every CTA but a seam token's, which arrives once its
        loads are in flight; the generation is read once a CTA needs it, so no
        barrier holds the first copies back."""
        if const_expr(spec.external_seam):
            _arrive(sync, slots, declaration.blocks)
        else:
            _arrive_unless_seam(sync, slots, declaration.blocks, block, s)

    @flyc.kernel(name=name, known_block_size=[declaration.threads, 1, 1])
    def resident(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_in_scale: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        core_q: Int64,
        scratch: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        allocator = fx.SharedAllocator()
        smem = allocator.allocate(lds_types.smem).peek()
        role = allocator.allocate(lds_types.role)
        attention = role.attn.peek()
        projection = role.gemv.peek()
        recurrence = role.kda.peek()
        sync = scratch + fx.Int64(layout["sync"][0])
        open_generation(sync, smem.tls.ptr, bid)
        part = scratch + fx.Int64(layout["part"][0])
        x_ready = scratch + fx.Int64(layout["xrdy"][0])
        x = prefix if spec.external_seam else scratch + fx.Int64(layout["x"][0])
        projected = scratch + fx.Int64(layout["proj"][0])
        # The seam's PART handoff is dead once every projection is visible.
        # Reuse its much larger tagged-pair region for post-KDA core readiness
        # instead of keeping another base pointer live through the resident
        # kernel.
        core_ready = part

        if const_expr(not spec.external_seam):  # noqa: SIM102
            # Token t's whole seam on CTA t, which no projection task owns.
            if bid < s:
                compute["attention_token"](
                    {
                        "NS": sources,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "matrix_threads": declaration.threads,
                        "red": smem.red.ptr,
                        "vals": attention.vals.ptr,
                        "eps": spec.eps,
                        "out_eps": spec.out_eps,
                        "delta": spec.delta,
                        "write_idx": spec.write_idx,
                        "xbuf": x,
                        "xrdy": x_ready,
                        "arrive": lambda: _arrive(
                            sync, smem.tls.ptr, declaration.blocks
                        ),
                        "tag": lambda: _generation(smem.tls.ptr),
                        "args": {
                            "blocks": blocks,
                            "blk_sm": blk_sm,
                            "blk_sr": blk_sr,
                            "prefix": prefix,
                            "delta": delta,
                            "ares_nw": ares_nw,
                            "ares_qk": ares_qk,
                            "in_nw": in_nw,
                        },
                    },
                    bid,
                )

        if bid >= ATTENTION_CTAS:
            # Every token's x before the first token copy: on the TDM path once
            # the first weight slabs are in flight (their copies count on the
            # tensor counter, not the loads a poll retires behind), else before
            # any weights, which a buffer poll would wait behind every spin.
            def await_x():
                if const_expr(not spec.external_seam):
                    _await_tokens(x_ready, s, _generation(smem.tls.ptr))

            if const_expr(not spec.tdm_projection):
                await_x()
            first_task = (bid - ATTENTION_CTAS) * PROJECTION_ROWS_PER_CTA
            if const_expr(spec.tdm_projection):
                accumulators0, accumulators1 = gemv_tdm.project(
                    projection.xl.ptr,
                    x,
                    HIDDEN,
                    s,
                    w_in,
                    FUSED_ROWS,
                    HIDDEN,
                    [
                        fx.min(first_task + task, PROJECTION_TASKS - 1) * ROWS
                        for task in range(PROJECTION_ROWS_PER_CTA)
                    ],
                    lane,
                    wave,
                    **(
                        {"w_scale": w_in_scale, "depth": gemv_tdm.FP8_DEPTH}
                        if spec.fp8_projection
                        else {}
                    ),
                    prologue=None if spec.external_seam else await_x,
                    x_cache_policy=0 if spec.external_seam else CM_DEV,
                    token_buffers=(
                        SEAM_TOKEN_BUFFERS
                        if spec.fp8_projection and not spec.external_seam
                        else 2
                    ),
                )
                gpu.barrier()
            else:
                accumulators0 = [compute["kda_zero"]() for _ in range(len(tile_starts))]
                accumulators1 = [compute["kda_zero"]() for _ in range(len(tile_starts))]
                projection_context = {
                    "S": ROWS,
                    "lane": lane,
                    "wave": wave,
                    "matrix_waves": declaration.waves,
                    "gemv_tasks": PROJECTION_TASKS,
                    "fused_rows": FUSED_ROWS,
                    "xrow": declaration.xrow,
                    "xl": projection.xl.ptr,
                    "args": {"w_in": w_in},
                }
                for slab in range_constexpr(len(slabs)):
                    first_group = slabs[slab][0]
                    groups = slabs[slab][1]
                    operands0 = compute["kda_gemv_loads"](
                        projection_context,
                        first_task,
                        range(first_group, first_group + groups),
                    )
                    for tile in range_constexpr(len(tile_starts)):
                        projection_context["S"] = tile_rows[tile]
                        stage_x_tile(
                            x,
                            projection.xl.ptr,
                            tid,
                            tile_starts[tile],
                            tile_rows[tile],
                            first_group,
                            groups,
                        )
                        accumulators0[tile] = compute["kda_gemv_slab"](
                            projection_context,
                            operands0,
                            groups,
                            accumulators0[tile],
                        )
                        if const_expr(tile == 0):
                            operands1 = compute["kda_gemv_loads"](
                                projection_context,
                                first_task + 1,
                                range(first_group, first_group + groups),
                            )
                        accumulators1[tile] = compute["kda_gemv_slab"](
                            projection_context,
                            operands1,
                            groups,
                            accumulators1[tile],
                        )
                        gpu.barrier()
            # Every accumulator first, then one cross-wave reduction of them all.
            slot_words = plan.kda_accumulator_lds_elements
            for task_offset in range_constexpr(PROJECTION_ROWS_PER_CTA):
                accumulators = (
                    accumulators0 if const_expr(task_offset == 0) else accumulators1
                )
                for tile in range_constexpr(len(tile_starts)):
                    compute["kda_gemv_store"](
                        accumulators[tile],
                        projection.acc.ptr
                        + (task_offset * len(tile_starts) + tile) * slot_words,
                        wave,
                        lane,
                    )
            gpu.barrier()
            projected_tag = _generation(smem.tls.ptr)
            for task_offset in range_constexpr(PROJECTION_ROWS_PER_CTA):
                task = first_task + task_offset
                if task < PROJECTION_TASKS:
                    for tile in range_constexpr(len(tile_starts)):
                        slot = (
                            projection.acc.ptr
                            + (task_offset * len(tile_starts) + tile) * slot_words
                        )
                        work_items = tile_rows[tile] * ROWS
                        for iteration in range_constexpr(
                            (work_items + declaration.threads - 1)
                            // declaration.threads
                        ):
                            work = tid + declaration.threads * iteration
                            if work < work_items:
                                local_token = work // ROWS
                                row = work % ROWS
                                value = fx.Float32(
                                    compute["kda_gemv_row_sum"](slot, row, local_token)
                                ).to(fx.BFloat16)
                                value = fx.Float32(value)
                                global_token = tile_starts[tile] + local_token
                                gfx_backend.mailbox_store_device(
                                    projected,
                                    global_token * NPROJ + task * ROWS + row,
                                    value,
                                    projected_tag,
                                )
            gpu.barrier()
            device["wait_vector_stores"]()

        generation = _generation(smem.tls.ptr)

        recurrence_context = {
            "S": s,
            "L": qlen,
            "SL": spec.state_len,
            "tid": tid,
            "lane": lane,
            "wave": wave,
            "matrix_waves": declaration.waves,
            "matrix_threads": declaration.threads,
            "red": smem.red.ptr,
            "tls": smem.tls.ptr,
            "fa": recurrence.fa.ptr,
            "qkl": recurrence.qkl.ptr,
            "decay": recurrence.decay.ptr,
            "vloc": recurrence.vloc.ptr,
            "ol": recurrence.ol.ptr,
            "beta": recurrence.beta.ptr,
            "slots": recurrence.slots.ptr,
            "head_onrm_lds": recurrence.onrm.ptr,
            "scratch": scratch,
            "proj_offset": layout["proj"][0],
            "qk_offset": layout["qk"][0],
            "onrm_offset": layout["onrm"][0],
            "tag": generation,
            "on": False,
            "xp": 0,
            "lb": spec.lower_bound,
            "onorm_eps": spec.onorm_eps,
            "args": {
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "on_w": on_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
                "core_out": core_out,
            },
        }
        first = _task_start(bid, spec.kda0, declaration.blocks)
        # A chunk task spreads one request/head over eight CTAs; once those
        # tasks need more than one pass of the grid, a whole head per CTA is
        # faster.
        if const_expr(qlen == 1 or task_count > declaration.blocks):
            head_tasks = (s // qlen) * HEADS
            for task in range(first, head_tasks, declaration.blocks):
                request = task // HEADS
                head = task % HEADS
                compute["kda_head"](
                    recurrence_context,
                    request,
                    head,
                )
                if const_expr(spec.quant_core):
                    device["wait_vector_stores"]()
                    gpu.barrier()
                    if tid == 0:
                        for chunk in range_constexpr(STATE_CHUNKS):
                            gfx_backend.mailbox_store_device(
                                core_ready,
                                (request * HEADS + head) * STATE_CHUNKS + chunk,
                                fx.Int32(1),
                                generation + fx.Int32(1),
                            )
                    gpu.barrier()
        else:
            for task in range(first, task_count, declaration.blocks):
                request = task // (HEADS * STATE_CHUNKS)
                head = (task // STATE_CHUNKS) % HEADS
                chunk = task % STATE_CHUNKS
                conv_values = conv_metadata(
                    recurrence_context,
                    request,
                    head,
                    chunk,
                )
                compute["kda_conv_inputs"](
                    recurrence_context,
                    request,
                    head,
                    chunk,
                )
                compute["kda_conv_compute"](
                    recurrence_context,
                    conv_values,
                    request,
                    head,
                    chunk,
                )
                recurrent_values = load_state_metadata(
                    recurrence_context,
                    gate_metadata(recurrence_context, head),
                    request,
                    head,
                    chunk,
                )
                compute["kda_gates"](
                    recurrence_context,
                    recurrent_values,
                    request,
                    head,
                )
                compute["kda_recur"](
                    recurrence_context,
                    recurrent_values,
                    request,
                    head,
                    chunk,
                )
                compute["kda_norm"](
                    recurrence_context,
                    request,
                    head,
                    chunk,
                )
                if const_expr(spec.quant_core):
                    device["wait_vector_stores"]()
                    gpu.barrier()
                    if tid == 0:
                        gfx_backend.mailbox_store_device(
                            core_ready,
                            (request * HEADS + head) * STATE_CHUNKS + chunk,
                            fx.Int32(1),
                            generation + fx.Int32(1),
                        )
                    gpu.barrier()
        device["wait_vector_stores"]()

        if const_expr(spec.quant_core):  # noqa: SIM102
            if (bid >= ATTENTION_CTAS) & (bid < ATTENTION_CTAS + s):
                token = bid - ATTENTION_CTAS
                request = token // qlen
                if tid < HEADS * STATE_CHUNKS:
                    gfx_backend.mailbox_poll_device(
                        core_ready,
                        request * HEADS * STATE_CHUNKS + tid,
                        generation + fx.Int32(1),
                    )
                gpu.barrier()
                quantize_core(
                    token,
                    tid,
                    lane,
                    wave,
                    smem.red.ptr,
                    core_out,
                    core_q,
                )

    @flyc.jit
    def launch(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_in_scale: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        core_q: Int64,
        scratch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        with CompilationContext.compile_hints({"waves_per_eu": 2}):
            call = resident(
                prefix,
                delta,
                blocks,
                blk_sm,
                blk_sr,
                ares_nw,
                ares_qk,
                in_nw,
                w_in,
                w_in_scale,
                w_fb,
                conv_w,
                conv_st,
                cs_seq,
                cs_dim,
                cs_tok,
                a_log,
                dt_bias,
                on_w,
                rstate,
                rs_seq,
                st_idx,
                st_stride,
                num_acc,
                core_out,
                core_q,
                scratch,
                **(
                    {"value_attrs": gemv_tdm.CLUSTER_LAUNCH["value_attrs"]}
                    if spec.tdm_projection
                    else {}
                ),
            )
            call.launch(
                grid=(declaration.blocks,),
                block=(declaration.threads,),
                stream=stream,
                **(
                    {"cluster": gemv_tdm.CLUSTER_LAUNCH["cluster"]}
                    if spec.tdm_projection
                    else {}
                ),
            )

    LOCAL_ABI.check(resident, launch)
    entry = ResidentK1EntryPoint(
        name,
        launch,
        (declaration.blocks,),
        (declaration.threads,),
    )
    _BUILDS[cache_key] = entry
    return entry
