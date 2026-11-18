# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ready flags between a producer kernel and an any-order consumer launched
right after it (``any_order_launch``, ``ATOM_KIMI_MONO_ANY_ORDER``).

One buffer of ``WORDS`` int32 a boundary: the producer's per-token ready flags
(``[0, TOKENS)``), the consumer's per-CTA completion counts (``DONE`` on), and
an error word. A consumer CTA's epoch is its own completion count plus one;
the producer's, the count of consumer CTA 0, which no consumer CTA can bump
before every producer CTA has published (each waits for all the flags), and
which the producer reads after the previous consumer completed (every launch
between them is in order). So the pairs count up together whatever the
widths launched, as long as every producer launch is followed by exactly one
consumer launch. The producer stores what it publishes at device scope, waits
for the stores, then stores its flag; the consumer reads the flags and the
data at device scope. The wait is bounded: past ``SPINS`` polls a CTA goes on,
having set the error word, rather than hang its GPU.
"""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl

from atom.mono.device.ops import ld_dev, st_dev, traced

TOKENS = 1024
DONE = TOKENS
CTAS = 256
ERROR = DONE + CTAS
WORDS = ERROR + 1
SPINS = 1 << 20


def producer_epoch(flags):
    """The epoch a producer CTA publishes."""
    return fx.Int32(ld_dev(flags, DONE)) + 1


def consumer_epoch(flags, cta):
    """The epoch consumer CTA ``cta`` waits for."""
    return fx.Int32(ld_dev(flags, DONE + cta)) + 1


@traced
def publish(flags, token, tid, epoch):
    """Token ``token``'s flag, once this CTA's device-scope stores land."""
    rocdl.s_wait_storecnt(0)
    gpu.barrier()
    if tid == 0:
        st_dev(epoch, flags, token)


@traced
def wait_ready(flags, tokens: int, tid, epoch):
    """Wave 0 until tokens ``[0, tokens)`` carry ``epoch``, then the CTA."""
    if tid < 32:
        for first in range_constexpr(0, tokens, 32):
            if tid + first < tokens:
                flag = fx.Int32(ld_dev(flags, tid + first))
                spins = fx.Int32(0)
                while (flag != epoch) & (spins < SPINS):
                    rocdl.s_sleep(1)
                    flag = fx.Int32(ld_dev(flags, tid + first))
                    spins = spins + 1
                if flag != epoch:
                    st_dev(fx.Int32(1), flags, ERROR)
    gpu.barrier()


@traced
def done(flags, cta, tid, epoch):
    """Consumer CTA ``cta`` has read everything the producer published."""
    if tid == 0:
        st_dev(epoch, flags, DONE + cta)
