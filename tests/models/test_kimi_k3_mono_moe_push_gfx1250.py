# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 fused MoE pushing the K2 tail's routed
all-reduce lines (``build_moe(push_tp=)``, ``build_k2_tail(pushed=True)``)."""

import pytest

from atom.models.kimi_k3.mono.config import LATENT, TOPK
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tail
from atom.models.kimi_k3.mono.kernels.stages import moe_tdm_gfx1250 as moe
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS
from atom.mono.plan.architecture import GFX1250

TAIL_WIDTHS = range(1, tail.tail_max_rows() + 1)


def _fits(rows, **kwargs):
    try:
        return moe.build_moe(rows, **kwargs)
    except ValueError:
        return None


def test_a_half_row_is_the_tails_two_cta_split_of_whole_down_tiles():
    # the tail's routed CTA 2t + h holds line h * THREADS + tid of token t
    assert ar.LINE7_ROUTED == 2 * ar.THREADS
    assert moe.HALF_TILES * moe.TASK_ROWS * 2 == LATENT
    assert moe.HALF_TILES * moe.TASK_ROWS == ar.THREADS * ar.LINE7_ELEMENTS
    # every route of a token adds each of its half's tiles once
    assert moe.PUSH_TILES == TOPK * moe.HALF_TILES == 112


def test_counters_hold_a_word_a_half_row_at_every_width():
    assert moe.push_counter_words() == 2 * GFX1250_MAX_TOKENS
    assert 2 * tail.tail_max_rows() <= moe.push_counter_words()


@pytest.mark.parametrize("rows", TAIL_WIDTHS)
@pytest.mark.parametrize("tp", (1, 4, 8))
def test_the_pushing_moe_builds_wherever_the_moe_does(rows, tp):
    base = _fits(rows)
    pushing = _fits(rows, push_tp=tp)
    assert (base is None) == (pushing is None)
    if base is None:
        return
    assert base.push_tp is None and "_push" not in base.name
    assert pushing.push_tp == tp and pushing.grid == (256,)
    assert pushing.name == base.name + f"_push{tp}"


def test_the_push_suffix_follows_the_rolled_one():
    assert moe.build_moe(20, rolled=True, push_tp=4).name.endswith("_rl_push4")
    assert moe.build_moe(20, rolled=False, push_tp=4).name.endswith("_u15_push4")


@pytest.mark.parametrize("rows", TAIL_WIDTHS)
def test_pushed_tail_layouts_fit_and_prefetch_at_most_pushed_depth(rows):
    layout = tail.tail_layout(rows, max_depth=tail.PUSHED_DEPTH)
    assert layout.words * 4 <= GFX1250.max_workgroup_lds_bytes
    assert 1 <= layout.depth <= tail.PUSHED_DEPTH
    assert layout.depth >= tail.tail_layout(rows).depth


@pytest.mark.parametrize("rows", (1, 20, 40, 42))
@pytest.mark.parametrize("tp", (1, 4))
def test_pushed_tail_keeps_its_grid_and_abi(rows, tp):
    gathering = tail.build_k2_tail(tp, rows, 1e-5, identity_routes=True)
    pushed = tail.build_k2_tail(tp, rows, 1e-5, identity_routes=True, pushed=True)
    assert pushed.abi is tail.TAIL_ABI
    assert pushed.grid == gathering.grid and pushed.ar_ctas == gathering.ar_ctas
    assert pushed.name.endswith("_id_pm")
    assert "_pm" not in gathering.name


def test_pushed_tail_takes_identity_routes_alone():
    with pytest.raises(ValueError):
        tail.build_k2_tail(4, 20, 1e-5, pushed=True)


def test_poll_limit_names_its_test_build():
    bounded = tail.build_k2_tail(1, 20, 1e-5, identity_routes=True, poll_limit=1 << 20)
    assert bounded.name.endswith(f"_pl{1 << 20}")
    assert "_pl" not in tail.build_k2_tail(1, 20, 1e-5, identity_routes=True).name


def test_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_MOE_PUSH", raising=False)
    assert envs.ATOM_KIMI_MONO_MOE_PUSH is False
    monkeypatch.setenv("ATOM_KIMI_MONO_MOE_PUSH", "1")
    assert envs.ATOM_KIMI_MONO_MOE_PUSH is True
