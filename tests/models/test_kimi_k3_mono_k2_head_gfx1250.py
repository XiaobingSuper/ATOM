# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 K2 head (o_proj all-reduce, MLP AttnRes
mix and the split-K front in one launch)."""

import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages import k2_head_gfx1250 as head
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_seam_front_gfx1250 as osf
from atom.models.kimi_k3.mono.kernels.stages import k2_seam_front_gfx1250 as seam

PROJ = {1: 12288, 4: 3072, 8: 1536}


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=4
    )


def test_the_front_and_the_mix_share_the_grid():
    assert head.FRONT_CTAS == front.SPLIT_CTAS
    assert head.FRONT_CTAS + head.MIX_CTAS == head.GRID == 256
    assert head.FRONT_CTAS % seam.CLUSTER == 0


@pytest.mark.parametrize("tp", (1, 4, 8))
@pytest.mark.parametrize("rows", (1, 10, 17, 20, 34, 40))
def test_one_launch_covers_the_front_the_mix_and_the_oproj_ctas(tp, rows):
    kernel = head.build_k2_head(
        _plan(rows), tp, PROJ[tp], 4, 1, 1e-6, 1e-5, fp8_shared=True
    )
    split = kernel.mix.flags // rows
    assert split in seam.SPLITS
    assert kernel.mix.ctas == seam._clusters(rows * split) <= head.MIX_CTAS
    # the most CTAs a token that the mix CTAs hold
    assert all(
        seam._clusters(rows * s) > head.MIX_CTAS for s in seam.SPLITS if s > split
    )
    grid = max(head.FRONT_CTAS + kernel.mix.ctas, oar.CTAS)
    assert kernel.grid == (grid,)
    assert grid % seam.CLUSTER == 0
    assert kernel.block == (256,)
    assert kernel.launch_count == 1
    assert kernel.abi is osf.OPROJ_SEAM_FRONT_ABI
    assert set(kernel.split) == {"router", "latent", "shared"}
    assert kernel.split["shared"][0] == "fp8"
    assert kernel.name.startswith(f"k3_k2_gfx1250_head_tp{tp}_")


def test_small_widths_hold_the_router_and_shared_shares_whole():
    kernel = head.build_k2_head(
        _plan(20), 4, PROJ[4], 4, 1, 1e-6, 1e-5, fp8_shared=True
    )
    assert kernel.split["router"][2] and kernel.split["shared"][2]


@pytest.mark.parametrize("rows", (20, 40))
def test_an_mxfp4_latent_takes_the_scale_plane_and_fits_whole(rows):
    kernel = head.build_k2_head(
        _plan(rows), 4, PROJ[4], 4, 1, 1e-6, 1e-5, fp8_shared=True, fp4_latent=True
    )
    assert kernel.abi is head.K2_HEAD_FP4_ABI
    assert kernel.abi.names[-1] == "latent_scale"
    assert kernel.split["latent"][0] == "fp4" and kernel.split["latent"][2]
    assert "_f4_" in kernel.name


def test_prefetch_hints_only_the_slabs_streamed_after_x():
    kernel = head.build_k2_head(
        _plan(40), 4, PROJ[4], 4, 1, 1e-6, 1e-5, fp8_shared=True, prefetch=2
    )
    quarter = head.tdm_split_slabs(head.HIDDEN)
    for kind, (_, stages, whole) in kernel.split.items():
        first, count = head._streamed(kernel.split, kind, 2)
        assert first == stages
        assert count == (0 if whole else min(quarter - stages, 2))
    assert (
        head._streamed(kernel.split, "latent", 99)[1]
        == quarter - (kernel.split["latent"][1])
    )
    assert "_p2c0" in kernel.name


@pytest.mark.parametrize(
    ("count", "addends", "rows"),
    ((0, 1, 8), (4, 2, 8), (4, 1, 41)),
)
def test_refuses_what_it_cannot_hold(count, addends, rows):
    with pytest.raises(ValueError):
        head.build_k2_head(_plan(rows), 4, PROJ[4], count, addends, 1e-6, 1e-5)
