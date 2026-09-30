# SPDX-License-Identifier: MIT

from types import SimpleNamespace

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


def test_agentic_shape_preserves_ragged_runtime_row_count():
    from atom.model_ops.monokernel.abi import AgenticDecodeShape

    shape = AgenticDecodeShape(
        running_bs=3,
        query_len=5,
        running_rows=12,
        batch_capacity=4,
        row_capacity=20,
        tile_rows=8,
    )

    assert shape.actual_rows == 12
    assert not shape.is_active_row(12)
    with pytest.raises(ValueError, match="cu_q"):
        shape.row_to_request(11)


def test_agentic_shape_consumes_forward_mode_without_rederiving_rows():
    from atom.model_ops.monokernel.abi import AgenticDecodeShape

    shape = AgenticDecodeShape.from_forward_mode(
        SimpleNamespace(running_bs=3, running_tokens=12, max_seqlen_q=5),
        batch_capacity=4,
        row_capacity=20,
    )

    assert shape.actual_rows == 12


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


def test_active_new_request_can_write_cache_with_zero_prior_context():
    from atom.model_ops.monokernel.abi import classify_decode_row

    row = classify_decode_row(
        batch_id=0,
        context_len=0,
        slot=42,
        sparse_begin=0,
        sparse_end=0,
        owned_count=0,
    )

    assert row.query_active
    assert row.cache_writer
    assert not row.local_sparse_active


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
    ("concurrency", "query_len", "rows", "tiles", "tail_rows"),
    (
        (2, 6, 24, 3, 8),
        (4, 6, 48, 6, 8),
        (8, 6, 96, 12, 8),
        (10, 5, 100, 13, 4),
        (16, 5, 160, 20, 8),
        (24, 5, 240, 30, 8),
        (32, 5, 320, 40, 8),
        (40, 5, 400, 50, 8),
        (48, 4, 384, 48, 8),
    ),
)
def test_glm_agentic_recipe_capacities_are_row_tiled(
    concurrency,
    query_len,
    rows,
    tiles,
    tail_rows,
):
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape

    shape = GlmAgenticShape.for_graph(
        batch_capacity=concurrency * 2,
        query_len=query_len,
        dcp_size=4 if concurrency >= 16 else 1,
        query_replication=concurrency >= 16,
    )

    assert shape.common.row_capacity == rows
    assert shape.common.tiles == tiles
    assert shape.common.tail_rows == tail_rows
    assert [(tile.start, tile.stop) for tile in shape.row_tiles][-1] == (
        rows - tail_rows,
        rows,
    )
    assert all(0 < tile.capacity <= 8 for tile in shape.row_tiles)


def test_glm_agentic_geometry_preserves_legacy_tp8():
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape

    shape = GlmAgenticShape.for_graph(
        batch_capacity=8,
        query_len=1,
        dcp_size=1,
        query_replication=False,
        tp_size=8,
    )

    assert shape.local_heads == 8
    assert shape.expert_intermediate == 256
    assert shape.physical_experts == 257


def test_glm_agentic_workspace_layout_reuses_one_tile_across_layer_rows():
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.layout import agentic_workspace_layout

    shape = GlmAgenticShape.for_graph(
        batch_capacity=80,
        query_len=5,
        dcp_size=4,
        query_replication=True,
    )
    workspace = agentic_workspace_layout(shape, npes=4, sparse_attention_topk=2048)

    assert workspace.row_capacity == 400
    assert workspace.tile_rows == 8
    assert workspace.tile_count == 50
    assert workspace.physical_experts == 257
    assert workspace.config.local_heads == 16
    assert workspace.config.inter == 512
    assert workspace.scratch["_bytes"] > 0
    assert workspace.symmetric["_bytes"] > 0


def test_glm_agentic_runtime_metadata_slices_tail_without_repacking():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.abi import (
        GlmAgenticRuntime,
        GlmAgenticShape,
    )

    shape = GlmAgenticShape.for_graph(
        batch_capacity=20,
        query_len=5,
        dcp_size=1,
        query_replication=False,
    )
    batch_ids = torch.arange(100, dtype=torch.int32)
    owned_counts = torch.arange(100, dtype=torch.int32)
    runtime = GlmAgenticRuntime.bind(shape, batch_ids, owned_counts)

    tail = runtime.tile(shape.row_tiles[-1])
    assert tail.batch_ids.tolist() == [96, 97, 98, 99]
    assert tail.owned_counts.tolist() == [96, 97, 98, 99]
    assert tail.batch_ids.data_ptr() == batch_ids[96:].data_ptr()
    with pytest.raises(ValueError, match="row capacity"):
        GlmAgenticRuntime.bind(shape, batch_ids[:-1], owned_counts)


def test_glm_agentic_work_queue_skips_inactive_capacity_tiles():
    from atom.model_ops.monokernel.abi import AgenticDecodeShape
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape

    shape = GlmAgenticShape(
        common=AgenticDecodeShape(
            running_bs=3,
            query_len=6,
            running_rows=13,
            batch_capacity=4,
            row_capacity=24,
        ),
        dcp_size=1,
        query_replication=False,
    )

    assert [(tile.start, tile.active_rows) for tile in shape.work_tiles] == [
        (0, 8),
        (8, 5),
    ]


def test_glm_agentic_workspace_is_shared_by_layer_bindings():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.layout import agentic_workspace_layout
    from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace

    shape = GlmAgenticShape.for_graph(
        batch_capacity=4,
        query_len=6,
        dcp_size=1,
        query_replication=False,
    )
    plan = agentic_workspace_layout(
        shape,
        npes=4,
        sparse_attention_topk=64,
    )
    workspace = GlmAgenticWorkspace(
        shape=shape,
        layout=plan,
        scratch=torch.empty(plan.scratch["_bytes"], dtype=torch.uint8),
        peer_buffer=SimpleNamespace(),
        step=torch.zeros(1, dtype=torch.int32),
    )

    first = workspace.layer(7)
    shared = workspace.layer(38)
    assert first.workspace is workspace
    assert shared.workspace is workspace
    assert (first.slot, shared.slot) == (7, 38)
    with pytest.raises(ValueError, match="layer slot"):
        workspace.layer(128)


def test_glm_graph_bucket_reuses_workspace_and_launches_s4_tail():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.graph import (
        GlmAgenticGraphBucket,
        GlmAgenticLayer,
        GlmAgenticLayerInputs,
    )
    from atom.model_ops.monokernel.glm.layout import agentic_workspace_layout
    from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace

    shape = GlmAgenticShape.for_graph(
        batch_capacity=20,
        query_len=5,
        dcp_size=1,
        query_replication=False,
    )
    plan = agentic_workspace_layout(
        shape,
        npes=4,
        sparse_attention_topk=64,
    )
    hidden_buffers = (
        torch.empty(100, shape.config.hidden, dtype=torch.bfloat16),
        torch.empty(100, shape.config.hidden, dtype=torch.bfloat16),
    )
    workspace = GlmAgenticWorkspace(
        shape=shape,
        layout=plan,
        scratch=torch.empty(plan.scratch["_bytes"], dtype=torch.uint8),
        peer_buffer=SimpleNamespace(
            storage=object(),
            local_address=11,
            addresses=torch.empty(4, dtype=torch.int64),
        ),
        step=torch.zeros(1, dtype=torch.int32),
        hidden_buffers=hidden_buffers,
    )
    calls = []

    class FakeKernel:
        def __init__(self, samples, slot, packed_artifacts):
            self.S = samples
            self.workspace = workspace
            self.slot = slot
            self.packed_artifacts = packed_artifacts
            self.scratch = workspace.scratch
            self.peer_buffer = workspace.peer_buffer
            self.step = workspace.step

        def forward(self, h, _cur_pos, *_args, x_out, layer, advance, **kwargs):
            calls.append(
                (
                    self.slot,
                    self.S,
                    layer,
                    advance,
                    kwargs["batch_ids"].clone(),
                    kwargs["owned_counts"].clone(),
                )
            )
            x_out.copy_(h + 1)
            return x_out

    inputs = GlmAgenticLayerInputs(
        kv_cache=torch.empty(1),
        pe_cache=torch.empty(1),
        indices=torch.empty(1, dtype=torch.int32),
        cos=torch.empty(1),
        sin=torch.empty(1),
    )
    layers = []
    for slot in (3, 9):
        packed_artifacts = object()
        layers.append(
            GlmAgenticLayer(
                slot=slot,
                kernels={
                    size: FakeKernel(size, slot, packed_artifacts)
                    for size in (4, 8)
                },
                inputs=inputs,
                packed_artifacts=packed_artifacts,
            )
        )
    layers = tuple(layers)
    bucket = GlmAgenticGraphBucket(workspace, layers)
    hidden = torch.zeros(100, shape.config.hidden, dtype=torch.bfloat16)
    positions = torch.arange(100, dtype=torch.int64)
    slot_mapping = positions.clone()
    sparse_indptr = torch.arange(101, dtype=torch.int32)
    batch_ids = torch.arange(100, dtype=torch.int32)
    batch_ids[-2:] = -1
    owned_counts = torch.ones(100, dtype=torch.int32)
    owned_counts[-3:] = 0

    output = bucket(
        hidden,
        positions=positions,
        slot_mapping=slot_mapping,
        sparse_kv_indptr=sparse_indptr,
        batch_ids=batch_ids,
        owned_counts=owned_counts,
    )

    assert output.data_ptr() == hidden_buffers[1].data_ptr()
    assert torch.equal(output, hidden + 2)
    assert [(slot, size) for slot, size, *_ in calls] == [
        pair
        for size in ([8] * 12 + [4])
        for pair in ((3, size), (9, size))
    ]
    assert all(layer == slot and not advance for slot, _, layer, advance, *_ in calls)
    assert calls[-1][-2].tolist() == [96, 97, -1, -1]
    assert calls[-1][-1].tolist() == [1, 0, 0, 0]
    assert workspace.step.item() == 13
    assert all(layer.workspace is workspace for layer in layers)


def test_glm_graph_bucket_builds_s8_and_s4_kernels_on_shared_storage():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.graph import (
        GlmAgenticGraphBucket,
        GlmAgenticLayerInputs,
        GlmAgenticLayerSpec,
    )
    from atom.model_ops.monokernel.glm.layout import agentic_workspace_layout
    from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace

    shape = GlmAgenticShape.for_graph(
        batch_capacity=20,
        query_len=5,
        dcp_size=1,
        query_replication=False,
    )
    plan = agentic_workspace_layout(
        shape,
        npes=4,
        sparse_attention_topk=64,
    )
    workspace = GlmAgenticWorkspace(
        shape=shape,
        layout=plan,
        scratch=torch.empty(plan.scratch["_bytes"], dtype=torch.uint8),
        peer_buffer=SimpleNamespace(
            storage=object(),
            local_address=11,
            addresses=torch.empty(4, dtype=torch.int64),
        ),
        step=torch.zeros(1, dtype=torch.int32),
        hidden_buffers=(
            torch.empty(100, shape.config.hidden, dtype=torch.bfloat16),
            torch.empty(100, shape.config.hidden, dtype=torch.bfloat16),
        ),
    )
    made = []
    packed = []

    class FakeKernel:
        def __init__(self, weights, samples, **kwargs):
            made.append((weights, samples, kwargs))
            self.S = samples
            self.workspace = kwargs["workspace"]
            self.packed_artifacts = kwargs["packed_artifacts"]
            self.scratch = self.workspace.scratch
            self.peer_buffer = self.workspace.peer_buffer
            self.step = self.workspace.step

    def pack_once(weights, **kwargs):
        owner = SimpleNamespace(weights=weights, token=torch.empty(1))
        packed.append((owner, kwargs))
        return owner

    args = GlmAgenticLayerInputs(*(torch.empty(1) for _ in range(5)))
    specs = (
        GlmAgenticLayerSpec(object(), args),
        GlmAgenticLayerSpec(object(), args),
    )
    bucket = GlmAgenticGraphBucket.build(
        workspace,
        specs,
        rank=0,
        npes=4,
        group=object(),
        topk=64,
        kernel_factory=FakeKernel,
        artifact_factory=pack_once,
    )

    assert [samples for _, samples, _ in made] == [4, 8, 4, 8]
    assert all(kwargs["workspace"] is workspace for _, _, kwargs in made)
    assert len(packed) == 2
    assert made[0][2]["packed_artifacts"] is made[1][2]["packed_artifacts"]
    assert made[2][2]["packed_artifacts"] is made[3][2]["packed_artifacts"]
    assert made[0][2]["packed_artifacts"] is not made[2][2]["packed_artifacts"]
    assert [layer.packed_artifacts for layer in bucket.layers] == [
        packed[0][0],
        packed[1][0],
    ]
    assert [layer.slot for layer in bucket.layers] == [0, 1]
    assert all(layer.workspace is workspace for layer in bucket.layers)


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
