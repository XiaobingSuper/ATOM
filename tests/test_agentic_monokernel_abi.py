# SPDX-License-Identifier: MIT

import pytest


def test_agentic_shape_separates_batch_query_and_flattened_rows():
    from atom.model_ops.monokernel.abi import AgenticDecodeShape

    shape = AgenticDecodeShape(
        running_bs=3,
        query_len=5,
        batch_capacity=4,
        row_capacity=20,
        tile_rows=8,
    )

    assert shape.actual_rows == 15
    assert shape.tiles == 3
    assert shape.tail_rows == 4
    assert shape.row_to_request(14) == 2
    assert shape.row_to_query(14) == 4
    assert not shape.is_active_row(15)


def test_non_owner_query_is_active_but_cannot_write_or_contribute_locally():
    from atom.model_ops.monokernel.abi import classify_decode_row

    row = classify_decode_row(
        batch_id=2,
        context_len=4096,
        slot=-1,
        sparse_begin=120,
        sparse_end=121,
        owned_count=0,
    )

    assert row.query_active
    assert not row.cache_writer
    assert not row.local_sparse_active
    assert row.safe_sparse_row == 120


def test_padding_row_is_inactive_even_with_safe_dummy_sparse_slot():
    from atom.model_ops.monokernel.abi import classify_decode_row

    row = classify_decode_row(
        batch_id=-1,
        context_len=0,
        slot=-1,
        sparse_begin=120,
        sparse_end=121,
        owned_count=0,
    )

    assert not row.query_active
    assert not row.cache_writer
    assert not row.local_sparse_active
    assert row.safe_sparse_row == 120


@pytest.mark.parametrize(
    "kwargs",
    (
        {"running_bs": 0},
        {"query_len": 0},
        {"batch_capacity": 2},
        {"row_capacity": 14},
    ),
)
def test_agentic_shape_rejects_inconsistent_capacity(kwargs):
    from atom.model_ops.monokernel.abi import AgenticDecodeShape

    values = dict(
        running_bs=3,
        query_len=5,
        batch_capacity=4,
        row_capacity=20,
        tile_rows=8,
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        AgenticDecodeShape(**values)


def test_glm_uv_scale_offsets_support_64_and_128_row_layouts():
    from atom.model_ops.monokernel.glm.layout import fp8_scale_offset

    assert fp8_scale_offset(4, 0, k_size=512, block_m=64) == 4
    assert fp8_scale_offset(4, 0, k_size=512, block_m=128) == 0
    assert fp8_scale_offset(8, 0, k_size=512, block_m=128) == 4


def test_glm_agentic_tp4_dcp4_qrep_geometry():
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape

    shape = GlmAgenticShape.for_graph(
        batch_capacity=80,
        query_len=5,
        dcp_size=4,
        query_replication=True,
    )

    assert shape.common.row_capacity == 400
    assert shape.local_heads == 16
    assert shape.query_heads == 64
    assert shape.expert_intermediate == 512


@pytest.mark.parametrize(
    ("batch_capacity", "query_len", "dcp_size", "rows"),
    ((32, 8, 1, 256), (32, 4, 8, 128), (160, 1, 8, 160)),
)
def test_kimi_agentic_recipe_capacity_geometry(
    batch_capacity,
    query_len,
    dcp_size,
    rows,
):
    from atom.model_ops.monokernel.k3.abi import KimiAgenticShape

    shape = KimiAgenticShape.for_graph(
        batch_capacity=batch_capacity,
        query_len=query_len,
        dcp_size=dcp_size,
        replay_ssm=query_len == 4,
    )

    assert shape.common.row_capacity == rows
    assert shape.local_heads == 12
