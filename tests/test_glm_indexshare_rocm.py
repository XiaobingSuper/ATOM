# SPDX-License-Identifier: Apache-2.0

"""Real gfx950 launch coverage for TP4 GLM FULL→SHARED IndexShare."""

from __future__ import annotations

import socket
from types import MappingProxyType

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


def _tp4_worker(rank: int, device_offset: int, port: int) -> None:
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.config import KvCacheLayout
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.graph import (
        GlmAgenticGraphBucket,
        GlmAgenticLayer,
        GlmAgenticLayerInputs,
    )
    from atom.model_ops.monokernel.glm.index_share import GlmIndexShareMode
    from atom.model_ops.monokernel.glm.op import Glm5MonoKernel
    from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace

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
            batch_capacity=20,
            query_len=5,
            dcp_size=1,
            query_replication=False,
        )
        workspace = GlmAgenticWorkspace.allocate(
            shape,
            rank=rank,
            npes=4,
            group=None,
            sparse_attention_topk=2048,
            with_indexer=True,
            index_max_seq=64,
        )
        workspace.ensure_index_share(2048)
        common = dict(
            rank=rank,
            npes=4,
            group=None,
            topk=2048,
            launches_per_step=2,
            index_max_seq=64,
            kv_cache_layout=KvCacheLayout.ATOM_FP8,
            agentic_row_contract=True,
            row_capacity=100,
            workspace=workspace,
        )
        full_artifacts = artifacts(True)
        shared_artifacts = artifacts(False)
        full = Glm5MonoKernel(
            weights,
            8,
            with_indexer=True,
            index_share=True,
            packed_artifacts=full_artifacts,
            **common,
        )
        shared = Glm5MonoKernel(
            weights,
            8,
            with_indexer=False,
            index_share=True,
            packed_artifacts=shared_artifacts,
            **common,
        )

        gen = torch.Generator(device=device).manual_seed(917)
        hidden = torch.randn(
            100, weights.config.hidden, generator=gen, device=device
        ).to(torch.bfloat16)
        positions = torch.zeros(100, dtype=torch.int64, device=device)
        positions[:2] = torch.tensor([16, 17], dtype=torch.int64, device=device)
        positions[96] = 16
        slot_mapping = torch.full(
            (100,), -1, dtype=torch.int64, device=device
        )
        slot_mapping[:2] = torch.tensor([0, 1], dtype=torch.int64, device=device)
        slot_mapping[96] = 16
        batch_ids = torch.full((100,), -1, dtype=torch.int32, device=device)
        batch_ids[:2] = 0
        batch_ids[96] = 19
        owned_counts = torch.zeros(100, dtype=torch.int32, device=device)
        owned_counts[:2] = 1
        owned_counts[96] = 1
        block_tables = torch.zeros(20, 4, dtype=torch.int32, device=device)
        block_tables[0] = torch.tensor([3, 0, 2, 1], device=device)
        block_tables[19] = torch.tensor([2, 1, 3, 0], device=device)
        context_lens = torch.zeros(20, dtype=torch.int32, device=device)
        context_lens[0] = 19
        context_lens[19] = 17
        selected = workspace.selected_slots
        counts = workspace.selected_counts
        indptr = workspace.selected_indptr
        assert selected is not None and counts is not None and indptr is not None
        selected = selected.reshape(-1)
        selected.fill_(-1)
        counts.fill_(-1)
        indptr.fill_(-1)
        cache = torch.zeros(
            64, 576, dtype=torch.float8_e4m3fn, device=device
        )
        cache[0:2, :512] = 7
        for physical in range(32, 48):
            cache[physical, :512] = physical - 31
        for physical in range(48, 64):
            cache[physical, :512] = physical - 47
        cache_scale = torch.ones(1, dtype=torch.float32, device=device)
        index_cache = torch.zeros(64, 144, dtype=torch.uint8, device=device)
        cos = torch.ones(64, 32, dtype=torch.float32, device=device)
        sin = torch.zeros_like(cos)
        full_inputs = GlmAgenticLayerInputs(
            cache,
            None,
            selected,
            cos,
            sin,
            index_cache=index_cache,
            kv_cache_scale=cache_scale,
        )
        shared_inputs = GlmAgenticLayerInputs(
            cache,
            None,
            selected,
            cos,
            sin,
            kv_cache_scale=cache_scale,
        )
        bucket = GlmAgenticGraphBucket(
            workspace,
            (
                GlmAgenticLayer(
                    0,
                    {8: full},
                    full_inputs,
                    full_artifacts,
                    index_share_mode=GlmIndexShareMode.FULL,
                ),
                GlmAgenticLayer(
                    1,
                    {8: shared},
                    shared_inputs,
                    shared_artifacts,
                    index_share_mode=GlmIndexShareMode.SHARED,
                ),
            ),
        )

        launch = dict(
            hidden_states=hidden,
            positions=positions,
            slot_mapping=slot_mapping,
            sparse_kv_indptr=indptr,
            batch_ids=batch_ids,
            owned_counts=owned_counts,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        # Start rank 0 late so faster ranks can enter SHARED while rank 0 may
        # still be consuming FULL's TP slot. There is no inter-layer host sync.
        if rank == 0:
            torch.cuda._sleep(5_000_000)
        eager_out = bucket(**launch)
        torch.cuda.synchronize(device)
        assert workspace.step.item() == 1

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if rank == 0:
                torch.cuda._sleep(5_000_000)
            graph_out = bucket(**launch)
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
        expected_counts = torch.zeros(100, dtype=torch.int32, device=device)
        expected_counts[:2] = torch.tensor([17, 18], device=device)
        expected_counts[96] = 17
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


def test_glm_tp4_full_to_shared_real_rocm_launch():
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

    offset = 0
    mp.spawn(
        _tp4_worker,
        args=(offset, _free_port()),
        nprocs=4,
        join=True,
    )
