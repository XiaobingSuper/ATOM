# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 o_proj seam front (o_proj all-reduce,
MLP AttnRes mix and front in one launch)."""

import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_seam_front_gfx1250 as fused
from atom.models.kimi_k3.mono.kernels.stages import k2_seam_front_gfx1250 as seam
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS

PROJ = {1: 12288, 4: 3072, 8: 1536}


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=4
    )


@pytest.mark.parametrize("tp", (1, 4, 8))
@pytest.mark.parametrize("rows", (1, 10, 17, 20, 34, 40, 48))
def test_one_launch_covers_the_oproj_ctas_and_the_seam_fronts(tp, rows):
    kernel = fused.build_oproj_seam_front(
        _plan(rows), tp, PROJ[tp], 4, 1, 1e-6, 1e-5, fp8_shared=True
    )
    grid = max(seam.mix_ctas(rows) + front.PAIRS, oar.CTAS)
    assert kernel.grid == (grid,)
    assert grid % seam.CLUSTER == 0
    assert kernel.block == (256,)
    assert kernel.launch_count == 1
    assert kernel.split == seam.mix_split(rows)
    assert kernel.token_buffers in (2, 3, 4)
    assert kernel.abi is fused.OPROJ_SEAM_FRONT_ABI
    assert kernel.name.startswith(f"k3_k2_gfx1250_oproj_seam_front_tp{tp}_")


def test_the_abi_is_the_oproj_allreduce_then_the_seam_front_without_its_addends():
    names = fused.OPROJ_SEAM_FRONT_ABI.names
    assert names[: len(oar.OPROJ_AR_ABI.names)] == tuple(
        "attn_out" if name == "out" else name for name in oar.OPROJ_AR_ABI.names
    )
    rest = tuple(n for n in seam.SEAM_FRONT_ABI.names if n not in ("add", "add2"))
    assert names[len(oar.OPROJ_AR_ABI.names) :] == rest


def test_the_oproj_epilogue_fits_past_its_token_slabs():
    for rows in range(1, GFX1250_MAX_TOKENS + 1):
        assert fused.oproj_lds_words(rows) * 4 <= 320 * 1024


@pytest.mark.parametrize(
    ("count", "addends", "rows"),
    ((0, 1, 8), (4, 2, 8), (4, 1, GFX1250_MAX_TOKENS + 1)),
)
def test_refuses_what_it_cannot_hold(count, addends, rows):
    with pytest.raises(ValueError):
        fused.build_oproj_seam_front(
            _plan(rows), 4, PROJ[4], count, addends, 1e-6, 1e-5
        )
