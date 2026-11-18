# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU/static proofs for the gfx1250 Kimi K2 front-half substrate."""

import math

import pytest

from atom.models.kimi_k3.mono.kernels import execution_plan as ep
from atom.models.kimi_k3.mono.kernels.stages import moe_layout as ml
from atom.mono.plan.architecture import GFX950, GFX1250


def _plan(**changes):
    values = {
        "threads": 256,
        "blocks": 256,
        "lds_bytes": 320 * 1024,
        "tokens": 16,
        "nblocks": 6,
    }
    values.update(changes)
    return ep.Gfx1250KimiBuildPlan(**values)


@pytest.mark.parametrize("architecture", [GFX950, GFX1250])
def test_router_lane_ownership_is_complete_without_overlap(architecture):
    contract = ml.router_contract_for(architecture)
    owned = [contract.experts_for_lane(lane) for lane in range(architecture.wave_size)]
    flattened = [expert for lane in owned for expert in lane]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == list(range(ml.EXPERTS))
    assert contract.experts_per_lane == ml.EXPERTS // architecture.wave_size


def test_router_topk_ties_nans_signed_zero_and_order_are_explicit():
    scores = [-100.0] * ml.EXPERTS
    sigmoid = [float(expert + 1) for expert in range(ml.EXPERTS)]
    for expert in range(32):
        scores[expert] = 3.0
    scores[0] = float("nan")
    sigmoid[0] = float("nan")
    scores[1] = -0.0
    scores[2] = 0.0

    reference = ml.router_contract_for(GFX950).select(scores, sigmoid)
    wave32 = ml.router_contract_for(GFX1250).select(scores, sigmoid)
    assert wave32 == reference
    assert reference[0] == tuple(range(3, 19))
    assert ml.ordered_f32_key(0.0) > ml.ordered_f32_key(-0.0)
    assert all(math.isfinite(weight) for weight in reference[1])


def test_router_equal_keys_choose_lower_experts_and_sum_in_rank_order():
    scores = [1.0] * ml.EXPERTS
    sigmoid = [0.1 + expert * 1.0e-5 for expert in range(ml.EXPERTS)]
    experts, weights = ml.router_contract_for(GFX1250).select(scores, sigmoid)
    assert experts == tuple(range(ml.TOPK))
    total = ml.f32_round(0.0)
    for expert in experts:
        total = ml.f32_round(total + ml.f32_round(sigmoid[expert]))
    inverse = ml.f32_round(1.0 / total)
    assert weights == tuple(
        ml.f32_round(ml.f32_round(sigmoid[expert]) * inverse) for expert in experts
    )


@pytest.mark.parametrize(
    ("architecture", "threads"),
    [(GFX950, 512), (GFX1250, 32), (GFX1250, 128), (GFX1250, 256)],
)
def test_route_union_thread_ownership_covers_every_expert(architecture, threads):
    contract = ml.RouteUnionContract(architecture, threads, 16)
    owned = [contract.experts_for_thread(tid) for tid in range(threads)]
    flattened = [expert for thread in owned for expert in thread]
    assert flattened == list(range(ml.EXPERTS))
    assert len(flattened) == len(set(flattened))
    assert contract.scan_words == threads // architecture.wave_size


@pytest.mark.parametrize("tokens", range(1, 17))
def test_route_union_slots_and_kof_are_complete_and_nonoverlapping(tokens):
    routes = tuple(
        tuple((token * 7 + topk * 3) % ml.EXPERTS for topk in range(ml.TOPK))
        for token in range(tokens)
    )
    contract = ml.RouteUnionContract(GFX1250, 256, tokens)
    union = contract.build(routes)
    assert union.experts == tuple(sorted(set().union(*map(set, routes))))
    assert len(union.experts) == len(set(union.experts))
    for slot, expert in enumerate(union.experts):
        assert union.slot_of[expert] == slot
    for token, row in enumerate(routes):
        for topk, expert in enumerate(row):
            assert union.kof[union.slot_of[expert]][token] == topk


def test_route_union_refuses_duplicate_or_out_of_range_picks():
    contract = ml.RouteUnionContract(GFX1250, 256, 1)
    with pytest.raises(ValueError, match="unique"):
        contract.build((tuple([0] * ml.TOPK),))
    with pytest.raises(ValueError, match="outside"):
        contract.build((tuple(range(ml.TOPK - 1)) + (ml.EXPERTS,),))


@pytest.mark.parametrize("tokens", range(1, 17))
@pytest.mark.parametrize("threads", [32, 128, 256, 512])
def test_latent_quant_group_ownership_and_layout(tokens, threads):
    contract = ml.LatentQuantContract(GFX1250, threads, tokens)
    owned = [contract.blocks_for_thread(tid) for tid in range(threads)]
    flattened = [block for thread in owned for block in thread]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == list(range(contract.blocks))

    payload = []
    scales = []
    for block in range(contract.blocks):
        words, scale = contract.output_offsets(block)
        assert len(words) == ml.MX_PACKED_WORDS
        payload.extend(words)
        scales.append(scale)
    assert sorted(payload) == list(range(tokens * ml.LATENT // 8))
    assert sorted(scales) == list(range(tokens * ml.LATENT // ml.MX_GROUP))


def test_latent_quant_scale_contract_is_capped_roundup():
    contract = ml.LatentQuantContract(GFX1250, 256, 1)
    assert contract.e8m0_roundup(0.0) == 0
    assert contract.e8m0_roundup(6.0) == 127
    assert contract.e8m0_roundup(6.01) == 128
    assert contract.e8m0_roundup(float("inf")) == 254


@pytest.mark.parametrize("tokens", range(1, 17))
@pytest.mark.parametrize("threads", [32, 128, 256])
def test_k2_candidate_lds_is_bounded_across_wave32_inputs(tokens, threads):
    plan = _plan(tokens=tokens, threads=threads)
    lds = plan.lds_plan()
    assert lds.used_bytes <= lds.capacity_bytes
    route = ml.RouteUnionContract(GFX1250, threads, tokens)
    latent = ml.LatentQuantContract(GFX1250, threads, tokens)
    assert route.scan_words == threads // 32
    assert latent.blocks == tokens * ml.LATENT // 32


@pytest.mark.parametrize("rows", (10, 40))
@pytest.mark.parametrize("rolled", (False, True))
def test_a_rolled_fused_moe_is_a_build_of_its_own(rows, rolled):
    from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import build_moe

    kernel = build_moe(rows, rolled=rolled)
    assert kernel.name.startswith(f"k3_gfx1250_moe_s{rows}_")
    assert kernel.name.endswith("_rl") == rolled


@pytest.mark.parametrize("rows", (10, 40))
@pytest.mark.parametrize("rolled", (False, True))
def test_an_early_fused_moe_is_a_build_of_its_own(rows, rolled):
    from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import build_moe

    base = build_moe(rows, rolled=rolled, early=False)
    early = build_moe(rows, rolled=rolled, early=True)
    assert early.name == base.name + "_ea"
    assert early.grid == base.grid


def test_the_early_moe_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_MOE_EARLY", raising=False)
    assert envs.ATOM_KIMI_MONO_MOE_EARLY is False
    monkeypatch.setenv("ATOM_KIMI_MONO_MOE_EARLY", "1")
    assert envs.ATOM_KIMI_MONO_MOE_EARLY is True


@pytest.mark.parametrize("rows", (10, 40))
def test_folded_and_one_issuer_fused_moes_are_builds_of_their_own(rows):
    from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import build_moe

    base = build_moe(rows, rolled=True, folded=False, one_issuer=False)
    folded = build_moe(rows, rolled=True, folded=True, one_issuer=False)
    one_issuer = build_moe(rows, rolled=True, folded=False, one_issuer=True)
    assert folded.name == base.name + "_fo" and folded.folded
    assert one_issuer.name == base.name + "_oi" and not one_issuer.folded
    assert not base.folded
    assert build_moe(rows, rolled=True, early=True, folded=True).name == base.name + "_ea_fo"


@pytest.mark.parametrize("name", ("ATOM_KIMI_MONO_MOE_FOLDED", "ATOM_KIMI_MONO_MOE_ONE_ISSUER"))
def test_the_folded_and_one_issuer_moe_switches_default_off(monkeypatch, name):
    from atom.utils import envs

    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True


def test_the_folded_gate_up_block_is_the_bytes_its_two_copies_read():
    import torch

    from atom.models.kimi_k3.mono.kernels.stages import moe_tdm_gfx1250 as mt

    experts, tiles = 3, ml.LATENT // mt.K_TILE
    gen = torch.Generator().manual_seed(0)
    w13 = torch.randint(0, 256, (experts, 2 * 384, ml.LATENT // 2), dtype=torch.uint8, generator=gen)
    w13s = torch.randint(0, 256, (experts, 2 * 384, ml.LATENT // 32), dtype=torch.uint8, generator=gen)
    folded = mt.fold_gate_up(w13, w13s, chunk=2).reshape(-1)
    weights, scales = w13.reshape(-1), w13s.reshape(-1)
    group = tiles * mt.GROUP_TILE_BYTES
    scale_row = tiles * mt.STEPS * 128
    for expert in range(experts):
        for task in range(mt.UP_BLOCKS):
            for tile in range(tiles):
                block = ((expert * mt.UP_BLOCKS + task) * tiles + tile) * mt.FOLD_BYTES
                for g in range(mt.UP_GROUPS):
                    src = (expert * mt.UP_BLOCKS * mt.UP_GROUPS + task * mt.UP_GROUPS + g) * group
                    src += tile * mt.GROUP_TILE_BYTES
                    dst = block + g * mt.GROUP_TILE_BYTES
                    assert torch.equal(
                        folded[dst : dst + mt.GROUP_TILE_BYTES], weights[src : src + mt.GROUP_TILE_BYTES]
                    )
                for w in range(mt.UP_WAVES):
                    src = (expert * mt.UP_BLOCKS * mt.UP_WAVES + task * mt.UP_WAVES + w) * scale_row
                    src += tile * mt.STEPS * 128
                    dst = block + mt.SLOT_BYTES + w * mt.STEPS * 128
                    assert torch.equal(folded[dst : dst + mt.STEPS * 128], scales[src : src + mt.STEPS * 128])


def test_the_folded_down_tile_is_its_weights_then_its_scales():
    import torch

    from atom.models.kimi_k3.mono.kernels.stages import moe_tdm_gfx1250 as mt

    experts = 2
    gen = torch.Generator().manual_seed(1)
    w2 = torch.randint(0, 256, (experts, ml.LATENT, 384 // 2), dtype=torch.uint8, generator=gen)
    w2s = torch.randint(0, 256, (experts, ml.LATENT, 384 // 32), dtype=torch.uint8, generator=gen)
    folded = mt.fold_down(w2, w2s).reshape(-1)
    weights, scales = w2.reshape(-1), w2s.reshape(-1)
    scale_bytes = mt.SCALE_WORDS * 4
    for expert in range(experts):
        for tile in range(mt.DOWN_TILES):
            index = expert * mt.DOWN_TILES + tile
            block = index * mt.FOLD_BYTES
            assert torch.equal(
                folded[block : block + mt.SLOT_BYTES], weights[index * mt.SLOT_BYTES : (index + 1) * mt.SLOT_BYTES]
            )
            assert torch.equal(
                folded[block + mt.SLOT_BYTES : block + mt.FOLD_BYTES],
                scales[index * scale_bytes : (index + 1) * scale_bytes],
            )
