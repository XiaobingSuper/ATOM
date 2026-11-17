# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only: the gfx1250 weight prefetch's arguments, the prefetching o_proj
all-reduce's and K2 tail's ABIs, and the runner passing the next kernel's
weights."""

from types import SimpleNamespace

import pytest
import torch

import atom.models.kimi_k3.mono.runner_gfx1250 as runner_module
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tail
from atom.models.kimi_k3.mono.kernels.stages import prefetch_gfx1250 as pf
from atom.models.kimi_k3.mono.runner_gfx1250 import K3Gfx1250MonoDecodeRunner
from atom.mono.plan.architecture import GFX1250


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=GFX1250.max_workgroup_lds_bytes,
        tokens=rows,
        nblocks=1,
    )


def test_range_names_pair_each_address_with_its_bytes():
    assert pf.range_names(2) == (
        "prefetch0",
        "prefetch0_bytes",
        "prefetch1",
        "prefetch1_bytes",
    )


def test_prefetch_args_skip_absent_tensors_and_pad_with_empty_ranges():
    a = torch.zeros(5, dtype=torch.bfloat16)
    b = torch.zeros(3, dtype=torch.int32)
    assert pf.prefetch_args(3, (a, None, b)) == [
        a.data_ptr(),
        10,
        b.data_ptr(),
        12,
        0,
        0,
    ]
    assert pf.prefetch_args(0, (a,)) == []
    assert pf.prefetch_args(2, ()) == [0, 0, 0, 0]


def test_prefetch_follows_the_environment_unless_given(monkeypatch):
    monkeypatch.setenv("ATOM_KIMI_MONO_PREFETCH", "1")
    assert pf.enabled(None) and not pf.enabled(False)
    monkeypatch.setenv("ATOM_KIMI_MONO_PREFETCH", "0")
    assert not pf.enabled(None) and pf.enabled(True)


@pytest.mark.parametrize("rows", (1, 20, 40))
def test_oproj_all_reduce_takes_the_prefetch_ranges_only_with_prefetch(rows):
    off = oar.build_oproj_allreduce(_plan(rows), 8, 1536, prefetch=False)
    assert off.abi is oar.OPROJ_AR_ABI and off.prefetch_ranges == 0
    on = oar.build_oproj_allreduce(_plan(rows), 8, 1536, prefetch=True)
    assert on.abi is oar.OPROJ_AR_PREFETCH_ABI
    assert on.prefetch_ranges == oar.PREFETCH_RANGES
    assert on.abi.names == off.abi.names + pf.range_names(oar.PREFETCH_RANGES)
    assert on.name == off.name + "_pf" and on.grid == off.grid


@pytest.mark.parametrize("rows", (1, 20, 40))
def test_tail_takes_the_prefetch_ranges_only_with_prefetch(rows):
    off = tail.build_k2_tail(8, rows, 1e-5, identity_routes=True, prefetch=False)
    assert off.abi is tail.TAIL_ABI and off.prefetch_ranges == 0
    on = tail.build_k2_tail(8, rows, 1e-5, identity_routes=True, prefetch=True)
    assert on.abi is tail.TAIL_PREFETCH_ABI
    assert on.prefetch_ranges == tail.PREFETCH_RANGES
    assert on.abi.names == off.abi.names + pf.range_names(tail.PREFETCH_RANGES)
    assert on.name == off.name + "_pf" and on.grid == off.grid


def _oproj_runner():
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    runner.oproj_weights = {9: (torch.zeros(4), torch.zeros(2))}
    runner.oproj_peers = torch.zeros(8, dtype=torch.int64)
    runner.k2_peers = SimpleNamespace(epochs=torch.zeros(4, dtype=torch.int32))
    runner.rank = 3
    runner.routed = {
        9: SimpleNamespace(
            w_gate=torch.zeros(6, dtype=torch.bfloat16),
            w_ld=torch.zeros(10, dtype=torch.bfloat16),
            w_sgu=torch.zeros(7, dtype=torch.uint8),
            w_sgu_scale=None,
        )
    }
    return runner


def test_prefetching_o_proj_all_reduce_passes_the_front_weights(monkeypatch):
    monkeypatch.setattr(runner_module.torch.cuda, "current_stream", lambda: None)
    runner = _oproj_runner()
    calls = []

    def launch(*args, stream):
        calls.append(args)

    kernel = SimpleNamespace(launcher=launch, prefetch_ranges=oar.PREFETCH_RANGES)
    runner._oproj(kernel, 9, torch.zeros(5, 6, dtype=torch.bfloat16))

    (args,) = calls
    routed = runner.routed[9]
    assert args[7:] == (
        routed.w_gate.data_ptr(),
        12,
        routed.w_ld.data_ptr(),
        20,
        routed.w_sgu.data_ptr(),
        7,
        0,
        0,
    )


def test_next_k1_weights_are_the_next_kda_layers_run_here():
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    weights = {
        layer: SimpleNamespace(w_in=f"w_in{layer}", w_in_scale=f"s{layer}")
        for layer in (1, 2, 4)
    }

    def plan(layer):
        return SimpleNamespace(layer_id=layer, kda=weights.get(layer))

    runner.plans = [plan(layer) for layer in range(6)]
    runner.routed = {1: 0, 2: 0, 3: 0, 4: 0}
    runner.ffn_layers = {3}
    # layer 4 is a KDA layer the runner does not run (not routed): none
    runner.routed.pop(4)

    assert runner._next_k1_weights(0) == ("w_in1", "s1")
    assert runner._next_k1_weights(1) == ("w_in2", "s2")
    assert runner._next_k1_weights(2) == ()
    assert runner._next_k1_weights(3) == ()
    assert runner._next_k1_weights(5) == ()
