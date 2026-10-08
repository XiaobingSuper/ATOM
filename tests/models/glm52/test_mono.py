# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
from pathlib import Path

import pytest

from atom.models.glm52.mono.config import (
    LEGACY_CORE_WIDTHS,
    PRODUCTION_WIDTHS,
    TARGET_WIDTHS,
    deployment_refusal,
    query_length,
)
from atom.models.glm52.mono.contract import check_layout, regions
from atom.models.glm52.mono.kernel import ABI, LayerBuild, build_layer
from atom.models.glm52.mono.sources import SOURCES
from atom.models.glm52.mono.step_meta import row_meta
from atom.models.glm52.mono.walk_plan import KEYS_PER_CHUNK, ragged_walk
from atom.mono.plan.trace import Space
from atom.mono.runtime.consensus import MonoUnsupported


def test_deployment_and_width_gates_leave_product_unchanged():
    assert TARGET_WIDTHS == (5, 6, 10, 12, 16, 24, 48)
    assert LEGACY_CORE_WIDTHS == (5, 6, 10, 12, 16)
    assert PRODUCTION_WIDTHS == ()
    assert query_length(4) == 5 and query_length(5) == 6
    assert (
        deployment_refusal(
            tp=4, dcp=1, mtp_speculative_tokens=4, kv_cache_dtype="fp8"
        )
        is None
    )
    assert (
        deployment_refusal(
            tp=8, dcp=1, mtp_speculative_tokens=4, kv_cache_dtype="fp8"
        )
        == "TP 8"
    )


def test_per_row_metadata_keeps_padding_and_causality_local():
    assert row_meta(0, 9, 42, 3, 8) == (0, 9, 42, 8, 24, 1)
    assert row_meta(1, 2, 7, 4, 8) == (1, 2, 7, 3, 32, 1)
    assert row_meta(-1, 999, -1, 5, 8) == (-1, 999, -1, 0, 40, 0)


@pytest.mark.parametrize(
    "owners",
    [
        [0] * 5,
        [0] * 6,
        [0] * 5 + [1] * 5,
        [0] * 6 + [1] * 6,
        [0] * 5 + [1] * 5 + [2] * 6,
        [owner for owner in range(4) for _ in range(6)],
        [owner for owner in range(8) for _ in range(6)],
    ],
)
def test_ragged_walk_covers_each_request_without_crossing_owners(owners):
    visible = [4096 + row for row in range(len(owners))]
    tasks = ragged_walk(owners, visible)
    assert len(owners) in TARGET_WIDTHS
    for task in tasks:
        assert len(set(owners[task.lead : task.lead + task.span])) == 1
        assert 0 <= task.chunk_first < task.chunk_last
    for owner in set(owners):
        rows = [row for row, value in enumerate(owners) if value == owner]
        expected = (
            max(visible[row] for row in rows) + KEYS_PER_CHUNK - 1
        ) // KEYS_PER_CHUNK
        spans = sorted(
            (task.chunk_first, task.chunk_last)
            for task in tasks
            if owners[task.lead] == owner
        )
        assert spans[0][0] == 0 and spans[-1][1] == expected
        assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))


def test_mailbox_and_build_contracts_are_canonical():
    assert len(ABI.names) == len(set(ABI.names)) == 38
    assert len(SOURCES) == 16
    for rows in LEGACY_CORE_WIDTHS:
        for with_indexer in (False, True):
            key = LayerBuild(rows, with_indexer, 4)
            check_layout(key)
            declared = regions(key)
            assert any(region.space is Space.PEER for region in declared)
            names = {region.name for region in declared}
            assert ("indices" in names) is with_indexer
    for rows in (24, 48):
        with pytest.raises(MonoUnsupported, match="row-batched K4"):
            build_layer(LayerBuild(rows, False, 5))


def test_width_runner_has_no_host_chunk_loop():
    runner = (
        Path(__file__).parents[3]
        / "atom"
        / "models"
        / "glm52"
        / "mono"
        / "runner.py"
    )
    tree = ast.parse(runner.read_text())
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "launch"
    )
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(launch))
    calls = [
        node
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "launcher"
    ]
    assert len(calls) == 1
