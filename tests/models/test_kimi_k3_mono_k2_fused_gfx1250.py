# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 fused K2 (lite finish, fused MoE and,
optionally, the K2 tail in one launch)."""

import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_fused_gfx1250 as fused
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tail
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    FINISH_PACKED_FRONT_ABI,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS


def _plan(rows, blocks=256):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=blocks,
        lds_bytes=320 * 1024,
        tokens=rows,
        nblocks=4,
    )


def test_flags_give_every_dependency_its_own_words():
    assert fused.RECORD < fused.ROUTE_AT
    assert fused.ROUTE_AT + GFX1250_MAX_TOKENS <= fused.HERD_AT
    assert fused.HERD_AT < fused.SHARED_AT
    assert fused.SHARED_AT + fused.GRID == fused.MOE_AT
    assert fused.MOE_AT + fused.GRID == fused.FLAGS


def test_abi_extends_the_finish_abi():
    names = fused.FUSED_ABI.names
    assert names[: len(FINISH_PACKED_FRONT_ABI.names)] == FINISH_PACKED_FRONT_ABI.names
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("rows", (1, 20, 40))
@pytest.mark.parametrize("herd", (False, True))
@pytest.mark.parametrize("tail_tp", (None, 1, 8))
def test_builder_is_one_resident_launch(rows, herd, tail_tp):
    kernel = fused.build_k2_fused(
        _plan(rows),
        herd=herd,
        fp8_shared_down=True,
        tail_tp=tail_tp,
        tail_eps=None if tail_tp is None else 1e-5,
    )
    assert kernel.grid == (fused.GRID,) and kernel.block == (256,)
    assert kernel.launch_count == 1 and kernel.abi is fused.FUSED_ABI
    assert kernel.tail is (tail_tp is not None)
    if tail_tp is not None:
        assert tail.tail_grid(rows) <= fused.GRID
    suffix = ("_herd" if herd else "") + "_f8" + ("" if tail_tp is None else f"_t{tail_tp}")
    assert kernel.name == f"k3_k2_gfx1250_fused_s{rows}{suffix}"


def test_builder_refuses_other_grids():
    with pytest.raises(ValueError):
        fused.build_k2_fused(_plan(20, blocks=128), herd=False, fp8_shared_down=True)


def test_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_K2_FUSED", raising=False)
    assert envs.ATOM_KIMI_MONO_K2_FUSED is False
    monkeypatch.setenv("ATOM_KIMI_MONO_K2_FUSED", "1")
    assert envs.ATOM_KIMI_MONO_K2_FUSED is True
