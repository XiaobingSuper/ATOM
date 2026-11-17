# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 weight prefetch into the last-level caches.

``global_prefetch_b8`` pulls the 128-byte line holding a lane's address into
the caches down to its cache policy's scope and returns nothing, so a wave
issues a range's lines and goes on. A communication kernel's waves issue the
next compute kernel's weights before they poll for their peers' lines: the
weights stream from HBM into the MALL (and the L2) while the kernel waits, and
the next kernel reads them from there. A range is split into one contiguous
share per issuing CTA, a line a thread a round; no line past the range is
issued (a prefetch past an allocation faults).

``ATOM_KIMI_MONO_PREFETCH`` (read at build time) adds a kernel's ranges to the
o_proj all-reduce (the front's weights) and the K2 tail (the next layer's K1
in-projection); unset, both kernels build as before.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int64, as_ir_value

from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime
from atom.mono.device.ops import st_dev, traced
from atom.mono.runtime.abi import KernelAbi
from atom.utils import envs

LINE = 128
# gfx12 CPol: TH in bits [2:0], scope in bits [4:3]
SCOPE_CU, SCOPE_SE, SCOPE_DEV, SCOPE_SYS = 0, 8, 16, 24
CACHE_POLICY = SCOPE_DEV


def enabled(prefetch: bool | None) -> bool:
    """``prefetch``, or ATOM_KIMI_MONO_PREFETCH where None."""
    return envs.ATOM_KIMI_MONO_PREFETCH if prefetch is None else prefetch


def range_names(count: int) -> tuple[str, ...]:
    """A prefetching kernel's extra arguments: each range's address and bytes."""
    return tuple(
        name for r in range(count) for name in (f"prefetch{r}", f"prefetch{r}_bytes")
    )


def _prefetch_line(address, cache_policy):
    rocdl_dialect.global_prefetch(
        llvm.IntToPtrOp(
            ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(address))
        ).result,
        cache_policy=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_policy),
    )


@traced
def prefetch_range(
    base, nbytes, part, parts, slot, slots, cache_policy=CACHE_POLICY, line_bytes=LINE
):
    """Thread ``slot`` of ``slots`` in CTA ``part`` of ``parts``: its lines of
    CTA ``part``'s share of the ``nbytes`` at ``base`` (none where 0)."""
    shift = line_bytes.bit_length() - 1
    assert line_bytes == 1 << shift
    # lines as i32: a range under 256 GiB
    lines = fx.Int32((fx.Int64(nbytes) + (line_bytes - 1)) >> shift)
    share = (lines + (parts - 1)) // parts
    first = fx.Int32(part) * share
    end = fx.min(first + share, lines)
    rounds = (fx.max(end - first, 0) + (slots - 1)) // slots
    # rolled: the issuing kernels run cold, where code is fetched
    for k in range(fx.Int32(0), rounds, unroll=1):
        line = fx.min(first + k * slots + slot, end - 1)
        _prefetch_line(fx.Int64(base) + (fx.Int64(line) << shift), cache_policy)


def prefetch_all(ranges, part, parts, slot, slots):
    """``prefetch_range`` of each ``(address, bytes)`` of ``ranges``."""
    for base, nbytes in ranges:
        prefetch_range(base, nbytes, part, parts, slot, slots)


def prefetch_args(ranges: int, tensors) -> list[int]:
    """The launch arguments of ``range_names(ranges)`` for ``tensors`` (None
    for none): each tensor's address and bytes, zeros past them."""
    args = []
    tensors = [t for t in tensors if t is not None][:ranges]
    for t in tensors:
        args += [t.data_ptr(), t.numel() * t.element_size()]
    return args + [0, 0] * (ranges - len(tensors))


PREFETCH_ABI = KernelAbi(("base", "nbytes", "stamps"))


@dataclass(frozen=True)
class PrefetchKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    abi: KernelAbi = PREFETCH_ABI


def build_prefetch(
    ctas: int,
    threads: int = 256,
    cache_policy: int = CACHE_POLICY,
    wait: bool = False,
    line_bytes: int = LINE,
) -> PrefetchKernel:
    """A launch that only prefetches ``nbytes`` at ``base``, for tests and
    measurements. With ``stamps``, thread 0 of CTA ``b`` stores its 100 MHz
    clock before and after (``stamps[2 b]``, ``stamps[2 b + 1]``), past a
    barrier with ``wait``: whether the prefetches hold a wave's loads."""
    name = f"k3_gfx1250_prefetch_c{ctas}_t{threads}_p{cache_policy}_l{line_bytes}" + (
        "_w" if wait else ""
    )

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(base: Int64, nbytes: Int64, stamps: Int64):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        start = fx.Int32(_realtime())
        prefetch_range(base, nbytes, bid, ctas, tid, threads, cache_policy, line_bytes)
        if const_expr(wait):
            gpu.barrier()
        if (stamps != 0) & (tid == 0):
            st_dev(start, stamps, 2 * bid)
            st_dev(fx.Int32(_realtime()), stamps, 2 * bid + 1)

    @flyc.jit
    def launch(
        base: Int64,
        nbytes: Int64,
        stamps: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(base, nbytes, stamps).launch(
            grid=(ctas,), block=(threads,), stream=stream
        )

    PREFETCH_ABI.check(kernel, launch)
    return PrefetchKernel(name, launch, (ctas,), (threads,))
