# SPDX-License-Identifier: Apache-2.0

"""Real gfx950 launch coverage for TP4 GLM FULL→SHARED IndexShare."""

from __future__ import annotations

import socket
from types import MappingProxyType, SimpleNamespace

import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _packed_zero_weights(device):
    import torch

    from atom.model_ops.monokernel.config import (
        AttentionWeight,
        glm5_shard_config,
    )
    from atom.model_ops.monokernel.glm.op import Glm5PackedArtifacts
    from atom.model_ops.monokernel.weights import LayerWeights

    config = glm5_shard_config(4)
    heads = config.local_heads
    matrices = {
        "qkv_a": (config.qkv_a_rows, config.hidden),
        "q_b": (heads * (config.nope_dim + config.pe_dim), config.q_lora),
        "uk": (heads * config.kv_lora, config.nope_dim),
        "uv": (heads * config.v_dim, config.kv_lora),
        "o": (config.hidden, heads * config.v_dim),
        "index_k": (128, config.hidden),
        "index_q": (32 * 128, config.q_lora),
    }
    tensors = {
        "g_in": torch.ones(config.hidden, dtype=torch.bfloat16, device=device),
        "g_q": torch.ones(config.q_lora, dtype=torch.bfloat16, device=device),
        "g_kv": torch.ones(config.kv_lora, dtype=torch.bfloat16, device=device),
        "g_post": torch.ones(config.hidden, dtype=torch.bfloat16, device=device),
        "w_index_w": torch.zeros(
            32 * config.hidden, dtype=torch.bfloat16, device=device
        ),
        "g_index_k": torch.ones(128, dtype=torch.float32, device=device),
        "b_index_k": torch.zeros(128, dtype=torch.float32, device=device),
        "w_r": torch.zeros(
            config.n_experts * config.hidden,
            dtype=torch.bfloat16,
            device=device,
        ),
        "bias": torch.zeros(
            config.n_experts, dtype=torch.float32, device=device
        ),
    }
    for name, (rows, cols) in matrices.items():
        tensors[f"w_{name}"] = torch.zeros(
            rows * cols, dtype=torch.uint8, device=device
        )
        tensors[f"s_{name}"] = torch.zeros(
            max(1, ((rows + 127) // 128) * ((cols + 63) // 64)),
            dtype=torch.float32,
            device=device,
        )

    experts = config.n_experts + config.num_shared_experts
    tensors["w_ug"] = torch.zeros(
        experts * 2 * config.inter * config.hidden // 2,
        dtype=torch.uint8,
        device=device,
    )
    tensors["s_ug"] = torch.zeros(
        experts * 2 * config.inter * (config.hidden // 32),
        dtype=torch.uint8,
        device=device,
    )
    tensors["w_dn"] = torch.zeros(
        experts * config.hidden * config.inter // 2,
        dtype=torch.uint8,
        device=device,
    )
    tensors["s_dn"] = torch.zeros(
        experts * config.hidden * (config.inter // 32),
        dtype=torch.uint8,
        device=device,
    )
    weights = LayerWeights(
        heads=heads,
        t=tensors,
        config=config,
        npes=4,
        physical_experts=experts,
    )

    def artifacts(with_indexer: bool):
        return Glm5PackedArtifacts(
            weights=weights,
            tensors=MappingProxyType(tensors),
            npes=4,
            attention_weight=AttentionWeight.FP8_BLOCK128,
            with_indexer=with_indexer,
            expert_mxfp4=True,
        )

    return weights, artifacts


def _tp4_worker(
    rank: int,
    device_offset: int,
    port: int,
    batch_capacity: int,
    query_len: int,
) -> None:
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.config import KvCacheLayout
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.graph import (
        GlmAgenticGraphBucket,
        GlmAgenticLayer,
    )
    from atom.model_ops.monokernel.glm.index_share import GlmIndexShareMode
    from atom.model_ops.monokernel.glm.op import Glm5MonoKernel
    from atom.model_ops.monokernel.telemetry import MonoRouteStats
    import atom.models.glm52_mono as glm_adapter

    device = torch.device("cuda", device_offset + rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
    )
    bucket = full = shared = workspace = None
    try:
        weights, artifacts = _packed_zero_weights(device)
        shape = GlmAgenticShape.for_graph(
            batch_capacity=batch_capacity,
            query_len=query_len,
            dcp_size=1,
            query_replication=False,
        )

        gen = torch.Generator(device=device).manual_seed(917)
        rows = shape.common.row_capacity
        tail_row = rows - (rows % 8 or 8)
        hidden = torch.randn(
            rows, weights.config.hidden, generator=gen, device=device
        ).to(torch.bfloat16)
        positions = torch.zeros(rows, dtype=torch.int64, device=device)
        positions[:2] = torch.tensor([16, 17], dtype=torch.int64, device=device)
        positions[tail_row] = 16
        slot_mapping = torch.full(
            (rows,), -1, dtype=torch.int64, device=device
        )
        slot_mapping[:2] = torch.tensor([0, 1], dtype=torch.int64, device=device)
        slot_mapping[tail_row] = 16
        batch_ids = torch.full((rows,), -1, dtype=torch.int32, device=device)
        batch_ids[:2] = 0
        batch_ids[tail_row] = batch_capacity - 1
        owned_counts = torch.zeros(rows, dtype=torch.int32, device=device)
        owned_counts[:2] = 1
        owned_counts[tail_row] = 1
        block_tables = torch.zeros(
            batch_capacity, 4, dtype=torch.int32, device=device
        )
        block_tables[0] = torch.tensor([3, 0, 2, 1], device=device)
        block_tables[-1] = torch.tensor([2, 1, 3, 0], device=device)
        context_lens = torch.zeros(
            batch_capacity, dtype=torch.int32, device=device
        )
        context_lens[0] = 19
        context_lens[-1] = 17
        cache = torch.zeros(
            64, 1, 576, dtype=torch.float8_e4m3fn, device=device
        )
        cache[0:2, 0, :512] = 7
        for physical in range(32, 48):
            cache[physical, 0, :512] = physical - 31
        for physical in range(48, 64):
            cache[physical, 0, :512] = physical - 47
        cache_scale = torch.ones(1, dtype=torch.float32, device=device)
        index_cache = torch.zeros(
            4, 16, 144, dtype=torch.float8_e4m3fn, device=device
        )
        cos = torch.ones(64, 32, dtype=torch.float32, device=device)
        sin = torch.zeros_like(cos)
        external_indptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                owned_counts.cumsum(0, dtype=torch.int32),
            )
        )

        class DensePrefix:
            mlp = SimpleNamespace(reduce_results=True)

            def __call__(self, _positions, state, residual):
                return state, residual

        shared_sparse = torch.zeros(2048 * rows, dtype=torch.int32, device=device)

        def sparse_attention(indexer):
            impl = SimpleNamespace(
                sparse_kv_indices_buffer=shared_sparse,
                _k_scale_device=cache_scale,
            )
            return SimpleNamespace(
                mla_attn=SimpleNamespace(impl=impl),
                indexer=indexer,
                rotary_emb=SimpleNamespace(cos_cache=cos, sin_cache=sin),
            )

        dense = [DensePrefix(), DensePrefix(), DensePrefix()]
        dense[2].self_attn = sparse_attention(
            SimpleNamespace(sparse_kv_indices_buffer=shared_sparse)
        )
        indexer_types = ["shared"] * 78
        indexer_types[6] = "full"
        target_layers = []
        kv_cache_data = {}
        for layer_idx in range(3, 78):
            full_mode = indexer_types[layer_idx] == "full"
            indexer = (
                SimpleNamespace(sparse_kv_indices_buffer=shared_sparse)
                if full_mode
                else None
            )
            layer = SimpleNamespace(
                layer_idx=layer_idx,
                self_attn=sparse_attention(indexer),
                mlp=SimpleNamespace(experts=object()),
            )
            target_layers.append(layer)
            kv_cache_data[f"layer_{layer_idx}"] = SimpleNamespace(
                k_cache=torch.zeros(
                    64, 1, 576, dtype=torch.float8_e4m3fn, device=device
                ),
                index_cache=(
                    torch.zeros(
                        4,
                        16,
                        144,
                        dtype=torch.float8_e4m3fn,
                        device=device,
                    )
                    if full_mode
                    else None
                ),
            )
        # Keep the observability cache on the FULL layer used by the device gate.
        kv_cache_data["layer_6"].k_cache = cache
        kv_cache_data["layer_6"].index_cache = index_cache
        for layer_idx in (3, 4, 5, 7):
            kv_cache_data[f"layer_{layer_idx}"].k_cache = cache

        hf_config = SimpleNamespace(
            index_topk=2048,
            indexer_types=indexer_types,
            max_model_len=64,
        )
        runner = object.__new__(glm_adapter.Glm52MonoDecode)
        runner._lm = SimpleNamespace(
            model=SimpleNamespace(
                layers=dense + target_layers,
                aux_hidden_state_layers=[],
                norm=SimpleNamespace(
                    weight=torch.ones(
                        weights.config.hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    eps=1e-6,
                ),
            ),
            config=hf_config,
        )
        runner._agentic_modes = glm_adapter._agentic_index_modes(
            runner._lm, hf_config
        )
        runner._agentic_buckets = {}
        runner._agentic_weights = {}
        runner._agentic_artifacts = {}
        runner._agentic_ready = False
        runner._agentic_ready_q = set()
        runner._agentic_refused = False
        runner._staged_enabled = False
        runner._enabled = True
        runner._mode = "mono"
        capture_sizes = (
            [1, 2, 4, 8, 12, 16]
            if query_len == 6
            else [1, 2, 4, 8, 12, 16, 20]
        )
        runner._atom_config = SimpleNamespace(
            tensor_parallel_size=4,
            kv_cache_dtype="fp8",
            capture_sizes=capture_sizes,
            max_model_len=64,
            hf_config=hf_config,
        )
        runner._stats = MonoRouteStats("glm52-device")
        metadata = SimpleNamespace(
            max_seqlen_q=query_len,
            sparse_kv_indptr=external_indptr,
            glm_agentic_owned_counts=owned_counts,
            slot_mapping=slot_mapping,
            batch_id_per_q_token=batch_ids,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        glm_adapter.get_forward_context = lambda: SimpleNamespace(
            context=SimpleNamespace(
                running_bs=batch_capacity,
                is_prefill=False,
                forward_mode=SimpleNamespace(max_seqlen_q=query_len),
            ),
            attn_metadata=metadata,
            ubatch_slices=None,
            kv_cache_data=kv_cache_data,
        )
        glm_adapter.rmsnorm2d_fwd_ = lambda state, *_args: state
        input_ids = torch.zeros(rows, dtype=torch.int64, device=device)

        from atom.model_ops.monokernel.config import MAX_LAYERS_PER_STEP
        from atom.model_ops.monokernel.glm.op import Glm5PackedArtifacts
        from atom.model_ops.monokernel.weights import LayerWeights

        mapping_calls = []
        real_mapping_calls = []
        artifact_calls = []
        build_calls = []

        def ptpc(rows, cols):
            return SimpleNamespace(
                input_size=cols,
                output_size=rows,
                quant_type=SimpleNamespace(name="per_Token"),
                params_dtype=torch.float8_e4m3fn,
                weight=torch.zeros(
                    rows, cols, dtype=torch.float8_e4m3fn, device=device
                ),
                weight_scale=torch.ones(
                    rows, 1, dtype=torch.float32, device=device
                ),
                is_output_padded=False,
            )

        cfg = weights.config
        probe_indexer = SimpleNamespace(
            wq_b=ptpc(32 * 128, cfg.q_lora),
            use_wk_weights_proj_fusion=True,
            wk_weights_proj=SimpleNamespace(
                weight=torch.zeros(
                    128 + 32,
                    cfg.hidden,
                    dtype=torch.bfloat16,
                    device=device,
                )
            ),
            k_norm=SimpleNamespace(
                weight=torch.ones(128, dtype=torch.float32, device=device),
                bias=torch.zeros(128, dtype=torch.float32, device=device),
            ),
        )
        probe_attention = SimpleNamespace(
            fused_qkv_a_proj=ptpc(cfg.qkv_a_rows, cfg.hidden),
            q_b_proj=ptpc(
                cfg.local_heads * (cfg.nope_dim + cfg.pe_dim), cfg.q_lora
            ),
            o_proj=ptpc(cfg.hidden, cfg.local_heads * cfg.v_dim),
            kv_b_proj=ptpc(
                cfg.local_heads * (cfg.nope_dim + cfg.v_dim), cfg.kv_lora
            ),
            indexer=probe_indexer,
            skip_topk=False,
        )
        original_base_mapper = glm_adapter._layer_weights
        glm_adapter._layer_weights = lambda *_args: weights
        mapped_template = glm_adapter._agentic_layer_weights(
            SimpleNamespace(self_attn=probe_attention),
            rank,
            4,
        )
        glm_adapter._layer_weights = original_base_mapper
        real_mapping_calls.append("ptpc")
        del probe_attention, probe_indexer

        def map_layer(layer, mapped_rank, mapped_npes):
            assert (mapped_rank, mapped_npes) == (rank, 4)
            mapping_calls.append(layer.layer_idx)
            return LayerWeights(
                mapped_template.heads,
                mapped_template.t,
                mapped_template.config,
                rank,
                4,
                mxfp4_weight_layout=mapped_template.mxfp4_weight_layout,
                mxfp4_scale_layout=mapped_template.mxfp4_scale_layout,
                physical_experts=mapped_template.physical_experts,
            )

        def pack_layer(mapped, *, with_indexer, **_kwargs):
            artifact_calls.append((id(mapped), with_indexer))
            base = artifacts(with_indexer)
            return Glm5PackedArtifacts(
                weights=mapped,
                tensors=base.tensors,
                npes=base.npes,
                attention_weight=base.attention_weight,
                with_indexer=base.with_indexer,
                expert_mxfp4=base.expert_mxfp4,
            )

        class PreparedBucket:
            def __init__(self, prepared_workspace):
                self.workspace = prepared_workspace

            def close(self):
                self.workspace.close()

        def build_bucket(
            prepared_workspace,
            specs,
            *,
            artifact_factory,
            **_kwargs,
        ):
            nonlocal full, shared
            prepared_shape = prepared_workspace.shape.common
            key = (prepared_shape.batch_capacity, prepared_shape.query_len)
            build_calls.append((key, len(specs)))
            assert len(specs) == 75
            assert [spec.index_share_mode.value for spec in specs[:5]] == [
                "shared",
                "shared",
                "shared",
                "full",
                "shared",
            ]
            if key != (batch_capacity, query_len):
                return PreparedBucket(prepared_workspace)
            layers = []
            for slot, spec in enumerate(specs[:5]):
                packed = artifact_factory(spec.weights)
                kernel = Glm5MonoKernel(
                    spec.weights,
                    8,
                    rank=rank,
                    npes=4,
                    group=None,
                    topk=2048,
                    launches_per_step=MAX_LAYERS_PER_STEP,
                    with_indexer=spec.with_indexer,
                    index_share=True,
                    index_max_seq=64,
                    kv_cache_layout=KvCacheLayout.ATOM_FP8,
                    agentic_row_contract=True,
                    row_capacity=prepared_shape.row_capacity,
                    workspace=prepared_workspace,
                    packed_artifacts=packed,
                )
                if spec.index_share_mode is GlmIndexShareMode.FULL:
                    full = kernel
                if slot == 4:
                    shared = kernel
                layers.append(
                    GlmAgenticLayer(
                        slot,
                        {8: kernel},
                        spec.inputs,
                        packed,
                        index_share_mode=spec.index_share_mode,
                    )
                )
            return GlmAgenticGraphBucket(prepared_workspace, tuple(layers))

        glm_adapter._agentic_layer_weights = map_layer
        glm_adapter.get_tensor_model_parallel_rank = lambda: rank
        glm_adapter.get_tensor_model_parallel_world_size = lambda: 4
        glm_adapter.get_tp_group = lambda: SimpleNamespace(cpu_group=None)
        Glm5PackedArtifacts.pack = staticmethod(pack_layer)
        GlmAgenticGraphBucket.build = staticmethod(build_bucket)

        def production_launch():
            assert runner.supports(input_ids, positions, None, hidden)
            return runner.forward(input_ids, positions, hidden)

        assert runner.supports(input_ids, positions, None, hidden)
        assert real_mapping_calls == ["ptpc"]
        assert mapping_calls == list(range(3, 78))
        assert len(artifact_calls) == 75
        assert build_calls == [
            ((capacity, query_len), 75) for capacity in capture_sizes
        ]
        assert set(runner._agentic_buckets) == {
            (capacity, query_len) for capacity in capture_sizes
        }
        bucket = runner._agentic_buckets[batch_capacity, query_len]
        workspace = bucket.workspace
        selected = workspace.selected_slots
        counts = workspace.selected_counts
        indptr = workspace.selected_indptr
        assert selected is not None and counts is not None and indptr is not None
        selected = selected.reshape(-1)
        selected.fill_(-1)
        counts.fill_(-1)
        indptr.fill_(-1)
        metadata.glm_agentic_owned_counts = workspace.publish_owned_counts(
            external_indptr
        )

        # Start rank 0 late so faster ranks can enter SHARED while rank 0 may
        # still be consuming FULL's TP slot. There is no inter-layer host sync.
        if rank == 0:
            torch.cuda._sleep(5_000_000)
        eager_out = production_launch()
        torch.cuda.synchronize(device)
        assert workspace.step.item() == 1

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if rank == 0:
                torch.cuda._sleep(5_000_000)
            graph_out = production_launch()
        graph.replay()
        torch.cuda.synchronize(device)
        assert workspace.step.item() == 2

        expected = torch.tensor(
            list(range(48, 64))
            + [0]
            + list(range(48, 64))
            + [0, 1]
            + list(range(32, 48))
            + [16],
            dtype=torch.int32,
            device=device,
        )
        expected_counts = torch.zeros(rows, dtype=torch.int32, device=device)
        expected_counts[:2] = torch.tensor([17, 18], device=device)
        expected_counts[tail_row] = 17
        expected_indptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                expected_counts.cumsum(0),
            )
        )
        assert torch.equal(counts, expected_counts)
        assert torch.equal(indptr, expected_indptr)
        assert torch.equal(selected[:52], expected), (
            selected[:52].cpu().tolist(),
            expected.cpu().tolist(),
        )
        assert not selected[52:].any()

        nsplit = 2048 // 64
        acc_shape = (8, nsplit, weights.config.local_heads, weights.config.kv_lora)
        shared_acc = shared.debug("sp_acc", acc_shape, bf2=True)
        # The final four-row tile is live only at global row 96. Scores are
        # deterministically zero, so P=1 and sp_acc is the explicit sum of its
        # selected physical values; slot 16 is a same-launch fresh zero.
        reference_sum = torch.arange(
            1, 17, dtype=torch.float32, device=device
        ).sum()
        reference_acc = torch.zeros(acc_shape, dtype=torch.float32, device=device)
        reference_acc[0, 0] = reference_sum
        torch.testing.assert_close(shared_acc, reference_acc, atol=0.5, rtol=0)
        assert shared_acc[0, 0].abs().sum() > 0
        assert not shared_acc[1:].any()

        for output in (eager_out, graph_out):
            torch.testing.assert_close(
                output.float(),
                hidden.float(),
                atol=4e-2,
                rtol=1e-2,
            )
            assert torch.isfinite(output).all()
        assert not cache[torch.tensor([0, 1, 16], device=device)].float().abs().max()
        selected_experts = shared.debug(
            "sel", (8, weights.config.moe_slots), torch.int32
        )
        assert torch.equal(
            selected_experts[:4, 0],
            torch.full(
                (4,),
                weights.config.shared_expert,
                dtype=torch.int32,
                device=device,
            ),
        )
        stats = runner.route_stats()
        assert stats["hits"][f"agentic_full:s{rows}"] == 2
        assert stats["fallbacks"] == {}
        for key in sorted(tuple(runner._agentic_buckets), reverse=True):
            runner._agentic_buckets.pop(key).close()
        assert runner._agentic_buckets == {}
        bucket = full = shared = workspace = None
    finally:
        if bucket is not None:
            bucket.close()
        else:
            if shared is not None:
                shared.close()
            if full is not None:
                full.close()
            if workspace is not None:
                workspace.close()
        dist.destroy_process_group()


@pytest.mark.parametrize(("batch_capacity", "query_len"), ((16, 6), (20, 5)))
def test_glm_tp4_full_to_shared_real_rocm_launch(batch_capacity, query_len):
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available():
        pytest.skip("ROCm device is unavailable")
    if torch.cuda.device_count() < 4:
        pytest.skip("TP4 device launch requires four visible ROCm devices")
    try:
        from flydsl.runtime.device import get_rocm_arch
    except ImportError:
        pytest.skip("FlyDSL ROCm compiler support is unavailable")
    if str(get_rocm_arch() or "") != "gfx950":
        pytest.skip(f"GLM TP4 MonoKernel requires gfx950, got {get_rocm_arch()}")

    import torch.multiprocessing as mp

    # Keep the two capacity-specialized process groups on disjoint devices when
    # an 8-GPU node is available. ROCm can retain graph/JIT state after a spawned
    # group exits; reusing those devices immediately made the second capacity
    # hang despite both capacities passing in isolation.
    offset = (
        0
        if query_len == 6 or torch.cuda.device_count() < 8
        else torch.cuda.device_count() - 4
    )
    mp.spawn(
        _tp4_worker,
        args=(
            offset,
            _free_port(),
            batch_capacity,
            query_len,
        ),
        nprocs=4,
        join=True,
    )
