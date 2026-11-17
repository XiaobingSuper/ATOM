# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 seam front (MLP AttnRes mix + front)."""

import pytest

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import attn_res_gfx1250 as mix
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages import k2_seam_front_gfx1250 as seam
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import tdm_lds_words
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=4
    )


def test_a_mix_thread_stands_for_four_of_attn_res_and_keeps_their_waves():
    assert seam.TURNS * seam.THREADS == mix.THREADS
    assert seam.MIX_WAVES == mix.WAVES
    assert seam.TURNS * seam.WAVES == seam.MIX_WAVES


@pytest.mark.parametrize(
    ("rows", "split"), ((1, 4), (17, 4), (18, 2), (34, 2), (35, 1), (48, 1))
)
def test_a_token_takes_the_most_ctas_that_leave_the_front_resident(rows, split):
    assert seam.mix_split(rows) == split
    assert seam.mix_ctas(rows) % seam.CLUSTER == 0
    assert seam.mix_ctas(rows) + front.PAIRS <= seam.MAX_CTAS


def test_every_width_has_a_split_whose_ctas_share_a_cluster():
    for rows in range(1, GFX1250_MAX_TOKENS + 1):
        split = seam.mix_split(rows)
        assert split in seam.SPLITS
        assert seam.CLUSTER % split == 0


@pytest.mark.parametrize("split", seam.SPLITS)
def test_a_split_tiles_the_token_in_whole_turns(split):
    layout = seam._MixLayout(4, 1, 1e-6, 1e-5, split)
    assert layout.turns * split == seam.TURNS
    assert layout.width * split == HIDDEN
    assert layout.width == layout.turns * seam.THREADS * seam.ELEMENTS


@pytest.mark.parametrize("rows", (1, 17, 18, 34, 35, 40, 48))
def test_one_launch_of_the_mix_clusters_then_the_front_pairs(rows):
    kernel = seam.build_seam_front(_plan(rows), 4, 1, 1e-6, 1e-5, fp8_shared=True)
    assert kernel.grid == (seam.mix_ctas(rows) + front.PAIRS,)
    assert kernel.block == (256,)
    assert kernel.launch_count == 1
    assert kernel.split == seam.mix_split(rows)
    assert kernel.token_buffers in (2, 3, 4)
    assert kernel.abi is seam.SEAM_FRONT_ABI
    assert kernel.name.endswith(f"_s{rows}_f8_m{kernel.split}_t{kernel.token_buffers}")


def test_the_abi_is_the_mix_then_sync_then_the_fronts_weights_and_outputs():
    names = seam.SEAM_FRONT_ABI.names
    mix_names = tuple(
        "x" if name == "out" else name
        for name in mix.ATTN_RES_ABI.names
        if name != "tokens"
    )
    assert names[: len(mix_names)] == mix_names
    assert names[len(mix_names)] == "sync"
    assert names[len(mix_names) + 1 :] == front.REPLICATED_FRONT_ABI.names[1:]


def test_sync_holds_a_flag_a_cta_and_every_tokens_exchange():
    assert seam.FLAGS0 >= seam.EPOCH0 + seam.MAX_CTAS
    assert seam.EXCHANGE0 >= seam.FLAGS0 + seam.MAX_CTAS
    widest = 2 * (mix.MAX_CANDIDATES - 1 + 1)
    assert seam.EXCHANGE_WORDS >= seam.MIX_WAVES * widest + seam.MIX_WAVES
    assert seam.SYNC_WORDS >= (
        seam.EXCHANGE0 + GFX1250_MAX_TOKENS * seam.EXCHANGE_WORDS
    )


def test_token_buffers_widen_only_the_token_slabs():
    tokens, weights = tdm_lds_words(40, 2)
    assert tdm_lds_words(40, 2, token_buffers=3) == (tokens * 3 // 2, weights)


@pytest.mark.parametrize(
    ("count", "addends", "rows", "split"),
    (
        (0, 1, 8, None),
        (mix.MAX_CANDIDATES, 1, 8, None),
        (4, 3, 8, None),
        (4, 1, GFX1250_MAX_TOKENS + 1, None),
        (4, 1, 40, 2),
        (4, 1, 8, 3),
    ),
)
def test_refuses_what_it_cannot_hold(count, addends, rows, split):
    with pytest.raises(ValueError):
        seam.build_seam_front(_plan(rows), count, addends, 1e-6, 1e-5, split=split)
