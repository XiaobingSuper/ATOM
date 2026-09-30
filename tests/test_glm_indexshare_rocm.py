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
    from atom.model_ops.monokernel.glm.op import Glm5MonoKernel

    device = torch.device("cuda", device_offset + rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
    )
    full = shared = None
    try:
        weights, artifacts = _packed_zero_weights(device)
        common = dict(
            rank=rank,
            npes=4,
            group=None,
            topk=2048,
            launches_per_step=2,
            index_max_seq=64,
            kv_cache_layout=KvCacheLayout.ATOM_FP8,
            agentic_row_contract=True,
        )
        full = Glm5MonoKernel(
            weights,
            4,
            with_indexer=True,
            index_share=True,
            packed_artifacts=artifacts(True),
            **common,
        )
        shared = Glm5MonoKernel(
            weights,
            4,
            with_indexer=False,
            index_share=True,
            packed_artifacts=artifacts(False),
            **common,
        )

        gen = torch.Generator(device=device).manual_seed(917)
        hidden = torch.randn(
            4, weights.config.hidden, generator=gen, device=device
        ).to(torch.bfloat16)
        positions = torch.tensor([16, 17, 18, 0], dtype=torch.int64, device=device)
        slot_mapping = torch.tensor([0, 1, -1, -1], dtype=torch.int64, device=device)
        batch_ids = torch.tensor([0, 0, 0, -1], dtype=torch.int32, device=device)
        owned_counts = torch.tensor([1, 1, 0, 0], dtype=torch.int32, device=device)
        block_tables = torch.tensor(
            [[3, 0, 2, 1]], dtype=torch.int32, device=device
        )
        context_lens = torch.tensor([19], dtype=torch.int32, device=device)
        selected = torch.full(
            (4 * 2048,), -1, dtype=torch.int32, device=device
        )
        counts = torch.full((4,), -1, dtype=torch.int32, device=device)
        indptr = torch.full((5,), -1, dtype=torch.int32, device=device)
        cache = torch.zeros(
            64, 576, dtype=torch.float8_e4m3fn, device=device
        )
        cache[0:2, :512] = 7
        for physical in range(48, 64):
            cache[physical, :512] = physical - 47
        cache_scale = torch.ones(1, dtype=torch.float32, device=device)
        index_cache = torch.zeros(64, 144, dtype=torch.uint8, device=device)
        cos = torch.ones(64, 32, dtype=torch.float32, device=device)
        sin = torch.zeros_like(cos)

        launch = dict(
            positions=positions,
            slot_mapping=slot_mapping,
            sparse_kv_indptr=indptr,
            batch_ids=batch_ids,
            owned_counts=owned_counts,
            kv_cache_scale=cache_scale,
            block_tables=block_tables,
            context_lens=context_lens,
            selected_counts=counts,
        )
        full_out = full.forward(
            hidden,
            positions,
            cache,
            None,
            selected,
            cos,
            sin,
            index_cache=index_cache,
            layer=0,
            advance=False,
            **launch,
        )
        # No host synchronization or intervening host read: SHARED must acquire
        # FULL's graph-stable publication through device/stream ordering.
        shared_out = shared.forward(
            full_out,
            positions,
            cache,
            None,
            selected,
            cos,
            sin,
            layer=1,
            advance=False,
            **launch,
        )
        torch.cuda.synchronize(device)

        expected = torch.tensor(
            list(range(48, 64))
            + [0]
            + list(range(48, 64))
            + [0, 1],
            dtype=torch.int32,
            device=device,
        )
        assert torch.equal(counts, torch.tensor([17, 18, 0, 0], device=device))
        assert torch.equal(indptr, torch.tensor([0, 17, 35, 35, 35], device=device))
        assert torch.equal(selected[:35], expected), (
            selected[:35].cpu().tolist(),
            expected.cpu().tolist(),
        )
        assert not selected[35:].any()

        nsplit = 2048 // 64
        acc_shape = (4, nsplit, weights.config.local_heads, weights.config.kv_lora)
        full_acc = full.debug("sp_acc", acc_shape, bf2=True)
        shared_acc = shared.debug("sp_acc", acc_shape, bf2=True)
        # Scores are deterministically zero, so P=1 and sp_acc is the explicit
        # sum of selected physical values. Slots 0/1 are same-launch fresh zeros,
        # replacing stale cache values of seven.
        reference_sum = torch.arange(
            1, 17, dtype=torch.float32, device=device
        ).sum()
        reference_acc = torch.zeros(acc_shape, dtype=torch.float32, device=device)
        reference_acc[0, 0] = reference_sum
        reference_acc[1, 0] = reference_sum
        torch.testing.assert_close(full_acc, reference_acc, atol=0.5, rtol=0)
        torch.testing.assert_close(shared_acc, reference_acc, atol=0.5, rtol=0)
        assert full_acc[0, 0].abs().sum() > 0
        assert not full_acc[2:].any() and not shared_acc[2:].any()

        guarded_counts = counts.clone()
        guarded_counts[0] = 1
        guarded_selected = selected.clone()
        guarded_selected[1:17] = 100_000
        guarded_out = shared.forward(
            full_out,
            positions,
            cache,
            None,
            guarded_selected,
            cos,
            sin,
            layer=1,
            advance=False,
            **dict(launch, selected_counts=guarded_counts),
        )
        torch.cuda.synchronize(device)
        guarded_acc = shared.debug("sp_acc", acc_shape, bf2=True)
        guarded_reference = reference_acc.clone()
        guarded_reference[0, 0] = 1
        torch.testing.assert_close(
            guarded_acc, guarded_reference, atol=0.5, rtol=0
        )

        for output in (full_out, shared_out, guarded_out):
            torch.testing.assert_close(
                output.float(),
                hidden.float(),
                atol=4e-2,
                rtol=1e-2,
            )
            assert torch.isfinite(output).all()
        assert not cache[:2].float().abs().max()
        selected_experts = full.debug(
            "sel", (4, weights.config.moe_slots), torch.int32
        )
        assert torch.equal(
            selected_experts[:, 0],
            torch.full(
                (4,),
                weights.config.shared_expert,
                dtype=torch.int32,
                device=device,
            ),
        )
    finally:
        if shared is not None:
            shared.close()
        if full is not None:
            full.close()
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

    # Prefer an idle four-device tail on shared eight-GPU development nodes.
    offset = torch.cuda.device_count() - 4
    mp.spawn(
        _tp4_worker,
        args=(offset, _free_port()),
        nprocs=4,
        join=True,
    )
