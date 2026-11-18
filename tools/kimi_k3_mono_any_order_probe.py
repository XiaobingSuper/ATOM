# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""hipExtAnyOrderLaunch on FlyDSL launchers in a torch CUDA graph (gfx1250).

PAIRS producer/consumer launches captured in one graph, twice over the same
buffers: every launch in order, then each consumer inside
``any_order_launch.any_order()`` (its AQL packet's barrier bit clear). A
producer CTA runs BUSY_US (TAIL_US more on the last TAIL_CTAS CTAs, the
stragglers), stores its output words at device scope, waits for the stores,
then stores its epoch-tagged flag. A consumer CTA streams PRO_KB of
independent weights, waits for every producer flag (DEP=all) or its own CTA's
(DEP=one), and checks the words. Per-CTA 100 MHz stamps give per XCD (CTA bid
% NXCD; the clock's offset differs between XCDs): the gap from the producer's
last CTA end to the consumer's first CTA entry, how much later the consumer
ends, and the producer stragglers' duration; the consumer -> next producer
boundary (always in order) is the control.

    python tools/kimi_k3_mono_any_order_probe.py
    AMD_LOG_LEVEL=4 AMD_LOG_MASK=0x8 PAIRS=2 REPS=1 REPLAYS=1 \\
        python tools/kimi_k3_mono_any_order_probe.py 2>&1 | grep "Dispatch Header"
    env: GRID (256), PAIRS (8), BUSY_US (4), TAIL_US (0), TAIL_CTAS (8),
         PRO_KB (0), DEP (all | one), REPS (8), REPLAYS (20), NXCD (8)
"""

import json
import os
import statistics

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono import any_order_launch
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime
from atom.mono.device.ops import ld_dev, rsrc, spin_pause, st_dev

THREADS = 256
WORDS = 4  # a thread's 16 B of a weight round
STAMPS = 4  # low words of the 100 MHz clock: entry, flags seen, end; errors
ENTRY, SEEN, END, ERRORS = range(STAMPS)
# which producer waves poll the clock while busy: all, or wave 0 (the others
# wait at a barrier); wave 0's s_sleep between polls
BUSY_WAVES = os.environ.get("BUSY_WAVES", "all")
BUSY_SLEEP = int(os.environ.get("BUSY_SLEEP", "1"))
# which consumer waves poll the producer's flags (DEP=all): all, or wave 0
POLL_WAVES = os.environ.get("POLL_WAVES", "all")
# dynamic LDS a CTA of each kernel asks for (KB): enough of it keeps a
# consumer CTA off a WGP until a producer CTA there exits
LDS_KB = int(os.environ.get("LDS_KB", "0"))


def tag(epoch, cta, word):
    return epoch * 1000003 + cta * THREADS + word


def build(grid: int, dep: str, pro_rounds: int):
    @flyc.kernel(
        name=f"k3_any_order_probe_producer_g{grid}_{BUSY_WAVES}_s{BUSY_SLEEP}",
        known_block_size=[THREADS, 1, 1],
    )
    def producer(
        counter: Int64, flags: Int64, data: Int64, stamps: Int64,
        busy: Int32, tail: Int32, tail_ctas: Int32,
    ):  # fmt: skip
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        entry = _realtime()
        epoch = fx.Int32(ld_dev(counter, bid)) + 1
        ticks = busy + (bid + tail_ctas >= grid).select(tail, fx.Int32(0))
        deadline = entry + fx.Int64(ticks)
        if const_expr(BUSY_WAVES == "all"):
            now = _realtime()
            while now < deadline:
                rocdl.s_sleep(1)
                now = _realtime()
        else:
            if tid < 32:
                t = _realtime()
                while t < deadline:
                    rocdl.s_sleep(BUSY_SLEEP)
                    t = _realtime()
            gpu.barrier()
        st_dev(tag(epoch, bid, tid), data, bid * THREADS + tid)
        rocdl.s_wait_storecnt(0)
        gpu.barrier()
        if tid == 0:
            st_dev(epoch, flags, bid)
            st_dev(epoch, counter, bid)
            out = rsrc(stamps)
            bo.buffer_store(fx.Int32(entry), out, bid * STAMPS + ENTRY)
            bo.buffer_store(fx.Int32(_realtime()), out, bid * STAMPS + END)

    @flyc.kernel(
        name=f"k3_any_order_probe_consumer_g{grid}_{dep}_p{pro_rounds}_{POLL_WAVES}",
        known_block_size=[THREADS, 1, 1],
    )
    def consumer(
        counter: Int64,
        flags: Int64,
        data: Int64,
        weights: Int64,
        stamps: Int64,
        sink: Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        entry = _realtime()
        epoch = fx.Int32(ld_dev(counter, bid)) + 1
        acc = fx.Int32(0)
        w = rsrc(weights)
        for r in range_constexpr(pro_rounds):
            v = fx.Vector(
                bo.buffer_load(
                    w, ((bid * pro_rounds + r) * THREADS + tid) * WORDS, vec_width=WORDS,
                    dtype=T.i32,
                )
            )  # fmt: skip
            acc = (
                acc ^ fx.Int32(v[0]) ^ fx.Int32(v[1]) ^ fx.Int32(v[2]) ^ fx.Int32(v[3])
            )
        if const_expr(dep == "all" and POLL_WAVES == "wave0"):
            if tid < 32:
                for p0 in range_constexpr(0, grid, 32):
                    if tid + p0 < grid:
                        flag = fx.Int32(ld_dev(flags, tid + p0))
                        while flag < epoch:
                            spin_pause()
                            flag = fx.Int32(ld_dev(flags, tid + p0))
        elif const_expr(dep == "all"):
            if tid < grid:
                flag = fx.Int32(ld_dev(flags, tid))
                while flag < epoch:
                    spin_pause()
                    flag = fx.Int32(ld_dev(flags, tid))
        else:
            if tid == 0:
                flag = fx.Int32(ld_dev(flags, bid))
                while flag < epoch:
                    spin_pause()
                    flag = fx.Int32(ld_dev(flags, bid))
        gpu.barrier()
        seen = _realtime()
        if const_expr(dep == "all"):
            word = bid % THREADS
            cta = fx.min(tid, grid - 1)
            got = fx.Int32(ld_dev(data, cta * THREADS + word))
            if (tid < grid) & (got != tag(epoch, cta, word)):
                st_dev(fx.Int32(1), stamps, bid * STAMPS + ERRORS)
        else:
            got = fx.Int32(ld_dev(data, bid * THREADS + tid))
            if got != tag(epoch, bid, tid):
                st_dev(fx.Int32(1), stamps, bid * STAMPS + ERRORS)
        if acc == 0x5EED5EED:
            bo.buffer_store(acc, rsrc(sink), bid)
        gpu.barrier()
        if tid == 0:
            st_dev(epoch, counter, bid)
            out = rsrc(stamps)
            bo.buffer_store(fx.Int32(entry), out, bid * STAMPS + ENTRY)
            bo.buffer_store(fx.Int32(seen), out, bid * STAMPS + SEEN)
            bo.buffer_store(fx.Int32(_realtime()), out, bid * STAMPS + END)

    @flyc.jit
    def launch_producer(
        counter: Int64, flags: Int64, data: Int64, stamps: Int64,
        busy: Int32, tail: Int32, tail_ctas: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):  # fmt: skip
        producer(counter, flags, data, stamps, busy, tail, tail_ctas).launch(
            grid=(grid,), block=(THREADS,), smem=LDS_KB * 1024, stream=stream
        )

    @flyc.jit
    def launch_consumer(
        counter: Int64, flags: Int64, data: Int64, weights: Int64, stamps: Int64, sink: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):  # fmt: skip
        consumer(counter, flags, data, weights, stamps, sink).launch(
            grid=(grid,), block=(THREADS,), smem=LDS_KB * 1024, stream=stream
        )

    return launch_producer, launch_consumer


def delta_us(a, b):
    """(a - b) of two low clock words, in us."""
    return (((int(a) - int(b) + (1 << 31)) % (1 << 32)) - (1 << 31)) / 100.0


def main():
    import torch

    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    any_order_launch.install()
    grid = int(os.environ.get("GRID", "256"))
    pairs = int(os.environ.get("PAIRS", "8"))
    busy = int(float(os.environ.get("BUSY_US", "4")) * 100)
    tail = int(float(os.environ.get("TAIL_US", "0")) * 100)
    tail_ctas = int(os.environ.get("TAIL_CTAS", "8"))
    pro_rounds = int(os.environ.get("PRO_KB", "0")) * 1024 // (WORDS * 4 * THREADS)
    dep = os.environ.get("DEP", "all")
    reps = int(os.environ.get("REPS", "8"))
    replays = int(os.environ.get("REPLAYS", "20"))
    nxcd = int(os.environ.get("NXCD", "8"))
    launch_producer, launch_consumer = build(grid, dep, pro_rounds)

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.int32, device=dev)

    counters = zeros(2, pairs, grid)
    flags, data = zeros(pairs, grid), zeros(pairs, grid * THREADS)
    weights = torch.randint(
        0, 1 << 30, (pairs, max(grid * pro_rounds * THREADS * WORDS, 1)),
        dtype=torch.int32, device=dev,
    )  # fmt: skip
    sink = zeros(grid)
    stamps = {name: zeros(2, pairs, grid, STAMPS) for name in ("in_order", "any_order")}

    def run(name):
        s = stamps[name]
        stream = torch.cuda.current_stream()
        for i in range(pairs):
            launch_producer(
                counters[0, i].data_ptr(), flags[i].data_ptr(), data[i].data_ptr(),
                s[0, i].data_ptr(), busy, tail, tail_ctas, stream=stream,
            )  # fmt: skip
            with any_order_launch.any_order(name == "any_order"):
                launch_consumer(
                    counters[1, i].data_ptr(), flags[i].data_ptr(), data[i].data_ptr(),
                    weights[i].data_ptr(), s[1, i].data_ptr(), sink.data_ptr(),
                    stream=stream,
                )  # fmt: skip

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run("in_order")  # compiles both launchers, after install()
    torch.cuda.synchronize()
    graphs = {}
    for name in stamps:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            run(name)
        graphs[name] = g
    for _ in range(3):
        for g in graphs.values():
            g.replay()
    torch.cuda.synchronize()
    times = {name: [] for name in graphs}
    for _ in range(reps):
        for name, g in graphs.items():
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(replays):
                g.replay()
            b.record()
            b.synchronize()
            times[name].append(a.elapsed_time(b) * 1000 / (replays * pairs))
    timed_errors = {name: int(s[1, ..., ERRORS].sum()) for name, s in stamps.items()}
    for name, g in graphs.items():
        stamps[name].zero_()
        g.replay()
    torch.cuda.synchronize()

    for name in graphs:
        s = stamps[name].cpu().tolist()  # [role][pair][cta][word]
        gap, lag, control, straggler, spread, span = [], [], [], [], [], []
        for i in range(1, pairs):
            for x in range(nxcd):
                ctas = range(x, grid, nxcd)
                p_end = max(
                    (s[0][i][c][END] for c in ctas),
                    key=lambda t: delta_us(t, s[0][i][x][ENTRY]),
                )
                entries = [delta_us(s[0][i][c][ENTRY], s[0][i][x][ENTRY]) for c in ctas]
                spread.append(max(entries) - min(entries))
                span.append(delta_us(p_end, s[0][i][x][ENTRY]) - min(entries))
                gap.append(min(delta_us(s[1][i][c][ENTRY], p_end) for c in ctas))
                lag.append(max(delta_us(s[1][i][c][END], p_end) for c in ctas))
                if i + 1 < pairs:
                    c_end = max(
                        (s[1][i][c][END] for c in ctas),
                        key=lambda t: delta_us(t, p_end),
                    )
                    control.append(
                        min(delta_us(s[0][i + 1][c][ENTRY], c_end) for c in ctas)
                    )
            for c in range(grid - tail_ctas, grid):
                straggler.append(delta_us(s[0][i][c][END], s[0][i][c][ENTRY]))
        errors = timed_errors[name] + sum(
            s[1][i][c][ERRORS] for i in range(pairs) for c in range(grid)
        )
        print(json.dumps({
            "config": name, "grid": grid, "pairs": pairs, "dep": dep,
            "busy_us": busy / 100, "tail_us": tail / 100, "tail_ctas": tail_ctas,
            "lds_kb": LDS_KB, "pro_kb": pro_rounds * WORDS * 4 * THREADS // 1024,
            "pair_us_median": round(statistics.median(times[name]), 2),
            "gap_us_p50": round(statistics.median(gap), 3),
            "gap_us_min": round(min(gap), 3), "gap_us_max": round(max(gap), 3),
            "lag_us_p50": round(statistics.median(lag), 3),
            "control_gap_us_p50": round(statistics.median(control), 3) if control else None,
            "straggler_us_p50": round(statistics.median(straggler), 3) if straggler else None,
            "producer_entry_spread_us_p50": round(statistics.median(spread), 3),
            "producer_span_us_p50": round(statistics.median(span), 3),
            "errors": errors, "launch_failures": len(any_order_launch.failures),
        }), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
