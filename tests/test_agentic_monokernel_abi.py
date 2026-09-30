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


def test_glm_fp8_cache_flattens_block16_page1_slots():
    import torch

    from atom.model_ops.monokernel.glm.cache import (
        logical_to_physical_slot,
        physical_cache_slot,
    )

    assert physical_cache_slot(0, 0) == 0
    assert physical_cache_slot(0, 15) == 15
    assert physical_cache_slot(1, 0) == 16
    assert physical_cache_slot(7, 15) == 127
    block_tables = torch.tensor([[7, 2], [5, 9]], dtype=torch.int32)
    assert logical_to_physical_slot(block_tables, batch_id=0, position=0) == 112
    assert logical_to_physical_slot(block_tables, batch_id=0, position=15) == 127
    assert logical_to_physical_slot(block_tables, batch_id=0, position=16) == 32
    assert logical_to_physical_slot(block_tables, batch_id=1, position=17) == 145
    with pytest.raises(ValueError, match="offset"):
        physical_cache_slot(1, 16)


def test_glm_mtp_visible_context_excludes_future_physical_slots():
    import torch

    from atom.model_ops.monokernel.glm.cache import (
        visible_context_length,
        visible_physical_slots,
    )

    block_tables = torch.tensor([[7, 2], [5, 9]], dtype=torch.int32)
    rows = (
        (0, 7, 10, 8, 119),
        (0, 8, 10, 9, 120),
        (0, 9, 10, 10, 121),
        (1, 16, 19, 17, 144),
        (1, 17, 19, 18, 145),
        (1, 18, 19, 19, 146),
    )

    for batch_id, position, request_context, expected, last_slot in rows:
        assert visible_context_length(position, request_context) == expected
        slots = visible_physical_slots(
            block_tables,
            batch_id=batch_id,
            position=position,
            request_context=request_context,
        )
        assert len(slots) == expected
        assert slots[-1] == last_slot
        future = visible_physical_slots(
            block_tables,
            batch_id=batch_id,
            position=request_context - 1,
            request_context=request_context,
        )[expected:]
        assert not set(slots).intersection(future)


def test_glm_fp8_cache_publishes_exact_main_and_index_bytes():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.config import FP8_MAX
    from atom.model_ops.monokernel.glm.cache import (
        INDEX_CACHE_ROW_BYTES,
        INDEX_KEY_BYTES,
        MAIN_CACHE_ROW_BYTES,
        publish_fp8_cache_rows,
    )

    main = torch.zeros(
        (2, 16, MAIN_CACHE_ROW_BYTES), dtype=torch.float8_e4m3fn
    )
    main_scale = torch.tensor([0.125], dtype=torch.float32)
    scale_before = main_scale.clone()
    index = torch.full((2, 16, INDEX_CACHE_ROW_BYTES), 0x5A, dtype=torch.uint8)
    main_row = torch.linspace(-8.0, 8.0, MAIN_CACHE_ROW_BYTES)
    index_row = torch.linspace(-2.0, 3.0, INDEX_KEY_BYTES)

    assert publish_fp8_cache_rows(
        main,
        main_scale,
        index,
        slot=16,
        batch_id=3,
        main_bf16=main_row.to(torch.bfloat16),
        index_bf16=index_row.to(torch.bfloat16),
    )

    main_values = main_row.to(torch.bfloat16).float()
    expected_main = (
        (main_values / main_scale)
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    index_values = index_row.to(torch.bfloat16).float()
    expected_index_scale = index_values.abs().max() / FP8_MAX
    expected_index = (
        (index_values / expected_index_scale)
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )

    assert torch.equal(
        main.view(torch.uint8).view(-1, MAIN_CACHE_ROW_BYTES)[16],
        expected_main,
    )
    assert torch.equal(main_scale, scale_before)
    index_flat = index.view(-1, INDEX_CACHE_ROW_BYTES)[16]
    assert torch.equal(index_flat[:INDEX_KEY_BYTES], expected_index)
    assert (
        index_flat[INDEX_KEY_BYTES : INDEX_KEY_BYTES + 4]
        .clone()
        .view(torch.float32)
        .item()
        == pytest.approx(expected_index_scale.item())
    )
    assert torch.all(index_flat[INDEX_KEY_BYTES + 4 :] == 0x5A)
    assert torch.all(main.view(torch.uint8)[0] == 0)


@pytest.mark.parametrize(
    ("batch_id", "slot", "query_active", "writes"),
    ((2, -1, True, False), (-1, 17, False, False), (0, 17, True, True)),
)
def test_glm_fp8_cache_query_and_publication_ownership(
    batch_id,
    slot,
    query_active,
    writes,
):
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.cache import (
        INDEX_CACHE_ROW_BYTES,
        MAIN_CACHE_ROW_BYTES,
        fp8_cache_row_policy,
        publish_fp8_cache_rows,
    )

    main = torch.zeros(
        (32, MAIN_CACHE_ROW_BYTES), dtype=torch.float8_e4m3fn
    )
    scale = torch.tensor([0.25], dtype=torch.float32)
    index = torch.full((32, INDEX_CACHE_ROW_BYTES), 9, dtype=torch.uint8)
    before = (main.clone(), scale.clone(), index.clone())
    policy = fp8_cache_row_policy(batch_id=batch_id, slot=slot)
    published = publish_fp8_cache_rows(
        main,
        scale,
        index,
        slot=slot,
        batch_id=batch_id,
        main_bf16=torch.ones(MAIN_CACHE_ROW_BYTES, dtype=torch.bfloat16),
        index_bf16=torch.ones(128, dtype=torch.bfloat16),
    )

    assert policy.query_active is query_active
    assert policy.cache_writer is writes
    assert published is writes
    if not writes:
        assert torch.equal(main, before[0])
        assert torch.equal(scale, before[1])
        assert torch.equal(index, before[2])


def test_glm_fp8_cache_validates_physical_storage():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.cache import validate_fp8_paged_cache

    main = torch.empty(2, 16, 576, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, dtype=torch.float32)
    index = torch.empty(2, 16, 144, dtype=torch.uint8)

    assert validate_fp8_paged_cache(main, scale, index, with_indexer=True) == 32
    with pytest.raises(ValueError, match="576"):
        validate_fp8_paged_cache(main[..., :-1], scale, index, with_indexer=True)
    with pytest.raises(ValueError, match="FP32"):
        validate_fp8_paged_cache(main, scale.half(), index, with_indexer=True)
    with pytest.raises(ValueError, match="144"):
        validate_fp8_paged_cache(main, scale, index[..., :-1], with_indexer=True)
    with pytest.raises(ValueError, match="E4M3FN"):
        validate_fp8_paged_cache(
            main.view(torch.uint8), scale, index, with_indexer=True
        )
    if hasattr(torch, "float8_e4m3fnuz"):
        with pytest.raises(ValueError, match="FNUZ"):
            validate_fp8_paged_cache(
                main.view(torch.float8_e4m3fnuz),
                scale,
                index,
                with_indexer=True,
            )


@pytest.mark.parametrize(
    ("batch_id", "owned_count", "index_work", "split_work"),
    ((-1, 8, False, False), (0, 0, False, False), (2, 7, True, True)),
)
def test_glm_fp8_cache_work_policy_gates_padding_and_zero_owned(
    batch_id,
    owned_count,
    index_work,
    split_work,
):
    from atom.model_ops.monokernel.glm.cache import fp8_cache_work_policy

    policy = fp8_cache_work_policy(
        batch_id=batch_id,
        owned_count=owned_count,
    )

    assert policy.index_score is index_work
    assert policy.sparse_split is split_work
    if not split_work:
        assert policy.empty_output == (0.0, float("-inf"))


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


def test_glm_agentic_tp4_small_concurrency_geometry():
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape

    shape = GlmAgenticShape.for_graph(
        batch_capacity=20,
        query_len=5,
        dcp_size=1,
        query_replication=False,
    )

    assert shape.common.row_capacity == 100
    assert shape.local_heads == 16
    assert shape.query_heads == 16
    assert shape.expert_intermediate == 512


def test_glm_full_monokernel_rejects_dcp():
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape

    with pytest.raises(ValueError, match="DCP1"):
        GlmAgenticShape.for_graph(
            batch_capacity=32,
            query_len=5,
            dcp_size=4,
            query_replication=True,
        )


@pytest.mark.parametrize(
    ("concurrency", "query_len", "rows", "tiles", "tail_rows"),
    (
        (2, 6, 24, 3, 8),
        (4, 6, 48, 6, 8),
        (8, 6, 96, 12, 8),
        (10, 5, 100, 13, 4),
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
        dcp_size=1,
        query_replication=False,
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
        batch_capacity=20,
        query_len=5,
        dcp_size=1,
        query_replication=False,
    )
    workspace = agentic_workspace_layout(shape, npes=4, sparse_attention_topk=2048)

    assert workspace.row_capacity == 100
    assert workspace.tile_rows == 8
    assert workspace.tile_count == 13
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
                    kwargs["kv_cache_scale"],
                    kwargs["block_tables"],
                    kwargs["context_lens"],
                    kwargs["positions"].clone(),
                    kwargs["slot_mapping"].clone(),
                )
            )
            x_out.copy_(h + 1)
            return x_out

    cache_scale = torch.empty(1, dtype=torch.float32)
    inputs = GlmAgenticLayerInputs(
        kv_cache=torch.empty(1),
        pe_cache=torch.empty(1),
        indices=torch.empty(1, dtype=torch.int32),
        cos=torch.empty(1),
        sin=torch.empty(1),
        kv_cache_scale=cache_scale,
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
    row_ids = torch.arange(100, dtype=torch.int64)
    positions = row_ids % 5 + (row_ids // 5) * 17
    slot_mapping = (row_ids * 37) % 320
    sparse_indptr = torch.arange(101, dtype=torch.int32)
    batch_ids = torch.arange(100, dtype=torch.int32) // 5
    batch_ids[-2:] = -1
    owned_counts = torch.ones(100, dtype=torch.int32)
    owned_counts[-3:] = 0
    block_tables = torch.arange(40, dtype=torch.int32).view(20, 2).flip(1)
    context_lens = torch.arange(1, 21, dtype=torch.int32) * 17

    output = bucket(
        hidden,
        positions=positions,
        slot_mapping=slot_mapping,
        sparse_kv_indptr=sparse_indptr,
        batch_ids=batch_ids,
        owned_counts=owned_counts,
        block_tables=block_tables,
        context_lens=context_lens,
    )

    assert output.data_ptr() == hidden_buffers[1].data_ptr()
    assert torch.equal(output, hidden + 2)
    assert [(slot, size) for slot, size, *_ in calls] == [
        pair
        for size in ([8] * 12 + [4])
        for pair in ((3, size), (9, size))
    ]
    assert all(layer == slot and not advance for slot, _, layer, advance, *_ in calls)
    assert calls[-1][-7].tolist() == [19, 19, -1, -1]
    assert calls[-1][-6].tolist() == [1, 0, 0, 0]
    assert all(call[-5] is cache_scale for call in calls)
    assert all(call[-4] is block_tables for call in calls)
    assert all(call[-3] is context_lens for call in calls)
    assert calls[-1][-2].tolist() == positions[-4:].tolist()
    assert calls[-1][-1].tolist() == slot_mapping[-4:].tolist()
    assert workspace.step.item() == 13
    assert all(layer.workspace is workspace for layer in layers)


def test_glm_graph_bucket_builds_s8_and_s4_kernels_on_shared_storage():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.config import KvCacheLayout
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
        GlmAgenticLayerSpec(
            object(), args, kv_cache_layout=KvCacheLayout.ATOM
        ),
        GlmAgenticLayerSpec(
            object(), args, kv_cache_layout=KvCacheLayout.ATOM
        ),
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


def test_glm_graph_layer_spec_validates_fp8_physical_cache():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.config import KvCacheLayout
    from atom.model_ops.monokernel.glm.graph import (
        GlmAgenticLayerInputs,
        GlmAgenticLayerSpec,
    )

    args = GlmAgenticLayerInputs(
        kv_cache=torch.empty(1, 16, 576, dtype=torch.float8_e4m3fn),
        pe_cache=None,
        indices=torch.empty(1, dtype=torch.int32),
        cos=torch.empty(1),
        sin=torch.empty(1),
        index_cache=torch.empty(1, 16, 144, dtype=torch.uint8),
        kv_cache_scale=torch.empty(1, dtype=torch.float32),
    )
    spec = GlmAgenticLayerSpec(
        object(),
        args,
        with_indexer=True,
        kv_cache_layout=KvCacheLayout.ATOM_FP8,
    )

    assert spec.validate_cache_inputs() == 16
    missing_scale = GlmAgenticLayerSpec(
        object(),
        GlmAgenticLayerInputs(
            args.kv_cache,
            None,
            args.indices,
            args.cos,
            args.sin,
            index_cache=args.index_cache,
        ),
        with_indexer=True,
        kv_cache_layout=KvCacheLayout.ATOM_FP8,
    )
    with pytest.raises(ValueError, match="descale"):
        missing_scale.validate_cache_inputs()


def test_glm_indexshare_stable_topk_publishes_physical_slots():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.index_share import stable_physical_topk

    block_tables = torch.tensor([[7, 2]], dtype=torch.int32)
    scores = torch.tensor([1.0, 4.0, 4.0, -2.0, 3.0], dtype=torch.float32)

    slots, count = stable_physical_topk(
        scores,
        block_tables,
        batch_id=0,
        position=4,
        request_context=5,
        topk=8,
    )

    assert count == 5
    assert slots.tolist() == [113, 114, 116, 112, 115, 0, 0, 0]


def test_glm_indexshare_device_publication_uses_index_cache_pointer():
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "atom/model_ops/monokernel/glm/kernel.py"
    ).read_text()
    start = source.index("# Index keys use LayerNorm")
    publication = source[start : source.index('stamp("cache", t, 4)', start)]

    assert "r_index_cache = _rsrc(index_cache)" in publication
    assert "r_index_cache = _rsrc(indices)" not in publication


def test_glm_indexshare_split_acquires_before_compact_metadata_load():
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "atom/model_ops/monokernel/glm/kernel.py"
    ).read_text()
    split = source[source.index("def split_keys(t, s):"):]
    split = split[: split.index("def gather_old_kv")]

    ready = split.index('get(mb("indices_ready"), s)')
    acquire = split.index("fx.memory_fence", ready)
    bounds = split.index("row_index_bounds(s)")
    assert ready < acquire < bounds


def test_glm_indexshare_device_has_bounded_score_ordering_phase():
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "atom/model_ops/monokernel/glm/kernel.py"
    ).read_text()
    selection = source[source.index("threshold = prefix"):]
    selection = selection[: selection.index('stamp("index_select", s, 4)')]

    assert "sort_keys" in selection
    assert "sort_logical" in selection
    assert 'INDEX_LDS["sort_keys"].start' in selection
    assert "(score == 0.0).select" in source
    assert "key_a > key_b" in selection
    assert "logical_a < logical_b" in selection


def test_glm_indexshare_histogram_preserves_all_index_head_weights():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.layout import index_selection_lds_regions

    regions = index_selection_lds_regions(2048, histogram_offset=8192)
    weights = regions["index_weights"]
    histogram = regions["radix_histogram"]
    assert weights.stop <= histogram.start or histogram.stop <= weights.start

    arena = torch.full((histogram.stop + 1,), -1.0)
    expected = torch.arange(32, dtype=torch.float32) + 0.25
    arena[weights.start : weights.stop] = expected
    # Four threshold digits plus greater/equal collection passes each clear
    # and reuse the complete radix histogram.
    for _ in range(6):
        arena[histogram.start : histogram.stop] = 0
        assert torch.equal(arena[weights.start : weights.stop], expected)


def test_glm_indexshare_bounded_2048_order_matches_fp32_reference():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.index_share import stable_physical_topk

    scores = torch.arange(2304, dtype=torch.float32).remainder(37)
    blocks = torch.arange(144, dtype=torch.int32).flip(0).view(1, -1)
    slots, count = stable_physical_topk(
        scores,
        blocks,
        batch_id=0,
        position=2303,
        request_context=2304,
        topk=2048,
    )
    expected_logical = sorted(
        range(2304), key=lambda i: (-float(scores[i]), i)
    )[:2048]
    expected = [
        int(blocks[0, logical // 16]) * 16 + logical % 16
        for logical in expected_logical
    ]

    assert count == 2048
    assert slots.tolist() == expected
    signed_zero, _ = stable_physical_topk(
        torch.tensor([-0.0, 0.0, 1.0], dtype=torch.float32),
        torch.tensor([[4]], dtype=torch.int32),
        batch_id=0,
        position=2,
        request_context=3,
        topk=3,
    )
    assert signed_zero.tolist() == [66, 64, 65]
    with pytest.raises(ValueError, match="2048"):
        stable_physical_topk(
            scores,
            blocks,
            batch_id=0,
            position=2303,
            request_context=2304,
            topk=2049,
        )


def test_glm_indexshare_workspace_is_bounded_at_one_million_context():
    from atom.model_ops.monokernel.config import glm5_shard_config
    from atom.model_ops.monokernel.glm.layout import layout

    config = glm5_shard_config(4)
    small, _ = layout(
        8,
        config.local_heads,
        4,
        2048,
        with_indexer=True,
        index_max_seq=4096,
        model_config=config,
    )
    million, _ = layout(
        8,
        config.local_heads,
        4,
        2048,
        with_indexer=True,
        index_max_seq=1_000_000,
        model_config=config,
    )

    assert million["_bytes"] == small["_bytes"]
    assert "index_scores" not in million


def test_glm_indexshare_mtp_visibility_padding_and_zero_owned_rows():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.index_share import stable_physical_topk

    blocks = torch.tensor([[9, 2], [5, 12]], dtype=torch.int32)
    scores = torch.arange(32, dtype=torch.float32)
    first, first_count = stable_physical_topk(
        scores,
        blocks,
        batch_id=0,
        position=2,
        request_context=20,
        topk=8,
    )
    later, later_count = stable_physical_topk(
        scores,
        blocks,
        batch_id=1,
        position=17,
        request_context=18,
        topk=8,
    )
    padded, padded_count = stable_physical_topk(
        scores,
        blocks,
        batch_id=-1,
        position=31,
        request_context=32,
        topk=8,
    )
    unowned, unowned_count = stable_physical_topk(
        scores,
        blocks,
        batch_id=0,
        position=31,
        request_context=32,
        topk=8,
        owned_count=0,
    )

    assert (first_count, later_count) == (3, 8)
    assert first[:3].tolist() == [146, 145, 144]
    assert later.tolist() == [193, 192, 95, 94, 93, 92, 91, 90]
    assert padded_count == unowned_count == 0
    assert not padded.any() and not unowned.any()


def test_glm_indexshare_plan_requires_full_before_three_shared():
    from atom.model_ops.monokernel.glm.index_share import (
        GlmIndexShareMode,
        GlmIndexSharePlan,
    )

    plan = GlmIndexSharePlan.from_runtime_pattern(("F", "S", "S", "S", "F"))

    assert plan.modes == (
        GlmIndexShareMode.FULL,
        GlmIndexShareMode.SHARED,
        GlmIndexShareMode.SHARED,
        GlmIndexShareMode.SHARED,
        GlmIndexShareMode.FULL,
    )
    assert plan.source_layers == (0, 0, 0, 0, 4)
    with pytest.raises(ValueError, match="precede"):
        GlmIndexSharePlan.from_runtime_pattern(("S", "F"))


def test_glm_indexshare_full_then_three_shared_reuses_exact_publication():
    torch = pytest.importorskip("torch")
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.graph import (
        GlmAgenticGraphBucket,
        GlmAgenticLayer,
        GlmAgenticLayerInputs,
    )
    from atom.model_ops.monokernel.glm.index_share import GlmIndexShareMode
    from atom.model_ops.monokernel.glm.layout import agentic_workspace_layout
    from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace

    shape = GlmAgenticShape.for_graph(
        batch_capacity=1,
        query_len=4,
        dcp_size=1,
        query_replication=False,
    )
    plan = agentic_workspace_layout(shape, npes=4, sparse_attention_topk=4)
    workspace = GlmAgenticWorkspace(
        shape,
        plan,
        torch.empty(plan.scratch["_bytes"], dtype=torch.uint8),
        SimpleNamespace(),
        torch.zeros(1, dtype=torch.int32),
        hidden_buffers=tuple(
            torch.empty(4, shape.config.hidden, dtype=torch.bfloat16)
            for _ in range(2)
        ),
    )
    workspace.ensure_index_share(4)
    observed = []

    class FakeKernel:
        def __init__(self, mode):
            self.S = 4
            self.workspace = workspace
            self.scratch = workspace.scratch
            self.peer_buffer = workspace.peer_buffer
            self.step = workspace.step
            self.packed_artifacts = mode
            self.mode = mode

        def forward(
            self, h, _cur_pos, _kv, _pe, indices, _cos, _sin, *,
            x_out, sparse_kv_indptr, selected_counts, **_kwargs
        ):
            if self.mode is GlmIndexShareMode.FULL:
                indices[:5].copy_(torch.tensor([29, 3, 71, 0, 0]))
                selected_counts[:2].copy_(torch.tensor([3, 2]))
                sparse_kv_indptr[:3].copy_(torch.tensor([0, 3, 5]))
            observed.append(
                (
                    self.mode,
                    indices.data_ptr(),
                    sparse_kv_indptr.data_ptr(),
                    indices.clone(),
                    sparse_kv_indptr.clone(),
                )
            )
            x_out.copy_(h)
            return x_out

    args = GlmAgenticLayerInputs(*(torch.empty(1) for _ in range(5)))
    modes = (GlmIndexShareMode.FULL,) + (GlmIndexShareMode.SHARED,) * 3
    layers = tuple(
        GlmAgenticLayer(
            slot,
            {4: FakeKernel(mode)},
            args,
            mode,
            index_share_mode=mode,
        )
        for slot, mode in enumerate(modes)
    )
    bucket = GlmAgenticGraphBucket(workspace, layers)
    rows = shape.common.row_capacity
    bucket(
        torch.zeros(rows, shape.config.hidden, dtype=torch.bfloat16),
        positions=torch.arange(rows, dtype=torch.int64),
        slot_mapping=torch.arange(rows, dtype=torch.int64),
        sparse_kv_indptr=torch.zeros(rows + 1, dtype=torch.int32),
        batch_ids=torch.tensor([0, 0, -1, -1], dtype=torch.int32),
        owned_counts=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        context_lens=torch.tensor([2], dtype=torch.int32),
    )

    assert [item[0] for item in observed] == list(modes)
    assert len({item[1] for item in observed}) == 1
    assert len({item[2] for item in observed}) == 1
    assert all(item[3][:5].tolist() == [29, 3, 71, 0, 0] for item in observed)
    assert all(item[4][:3].tolist() == [0, 3, 5] for item in observed)


@pytest.mark.parametrize(
    ("batch_capacity", "query_len", "dcp_size", "rows"),
    ((2, 8, 1, 16), (4, 8, 1, 32), (8, 8, 1, 64)),
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
        replay_ssm=False,
    )

    assert shape.common.row_capacity == rows
    assert shape.local_heads == 12


def test_kimi_full_monokernel_rejects_dcp_and_replayssm():
    from atom.model_ops.monokernel.k3.abi import KimiAgenticShape

    with pytest.raises(ValueError, match="DCP1"):
        KimiAgenticShape.for_graph(
            batch_capacity=32,
            query_len=4,
            dcp_size=8,
            replay_ssm=True,
        )
