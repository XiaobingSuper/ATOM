# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ATOM_KIMI_MONO_ENERGY``: how the gfx1250 mono kernels' waits idle. Under
sustained decode every GPU runs at its socket power cap, so a spinning wave's
power is clock taken from the waves doing the work. A "."-separated set,
read when a kernel is built (empty, the default: the waits as they were):

  wait<N>  ``s_sleep N`` (64-clock units) between polls in the intra-GPU waits
           that last microseconds: K1's head waits on the projection, the
           tail's all-reduce flags, the fused MoE's gate/up flags, HERD's
           candidates (``s_sleep 1`` by default), the lite's grid barrier (no
           sleep by default)
  ar<N>    ``s_sleep N`` between the cross-GPU all-reduce polls (o_proj AR,
           gathering AR, the tail's), in place of ``spin_pause``
  lead     K1's head waits: wave 0 polls alone while the other waves wait at
           a barrier, then every wave loads its words (by then published)

A non-default build's kernels carry a name suffix."""

import re

from flydsl.expr import rocdl

from atom.mono.device.ops import spin_pause
from atom.utils import envs


def _knobs() -> dict[str, int]:
    knobs = {}
    for item in envs.ATOM_KIMI_MONO_ENERGY.split("."):
        if not item:
            continue
        match = re.fullmatch(r"(wait|ar)(\d+)|lead", item)
        if match is None:
            raise ValueError(f"ATOM_KIMI_MONO_ENERGY: unknown item {item!r}")
        if item == "lead":
            knobs["lead"] = 1
            continue
        units = int(match.group(2))
        if not 1 <= units <= 127:
            raise ValueError(f"ATOM_KIMI_MONO_ENERGY: {item!r} needs 1..127 units")
        knobs[match.group(1)] = units
    return knobs


def wait_pause(default: int | None = 1) -> None:
    """Between two polls of a microseconds-long intra-GPU wait: ``s_sleep`` of
    the ``wait`` units, else of ``default`` (None: nothing)."""
    units = _knobs().get("wait", default)
    if units:
        rocdl.s_sleep(units)


def ar_pause() -> None:
    """Between two polls of a cross-GPU all-reduce."""
    units = _knobs().get("ar")
    if units:
        rocdl.s_sleep(units)
    else:
        spin_pause()


def lead() -> bool:
    return "lead" in _knobs()


def suffix() -> str:
    knobs = _knobs()
    if not knobs:
        return ""
    return "_en" + "".join(f"{k[0]}{v}" if k != "lead" else "l" for k, v in sorted(knobs.items()))
