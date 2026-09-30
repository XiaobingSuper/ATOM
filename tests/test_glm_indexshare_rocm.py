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


def _preshuffled_mxfp4(source):
    import torch

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from atom.model_ops.monokernel.formats import dequantize_mxfp4

    packed, scale = dynamic_mxfp4_quant(source)
    raw = packed.view(torch.uint8).view(source.shape[0], -1)
    rows, packed_cols = raw.shape
    shuffled_weight = (
        raw.view(rows // 16, 16, packed_cols // 32, 2, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape_as(raw)
    )
    groups = scale.shape[1]
    shuffled_scale = (
        scale.reshape(rows // 32, 2, 16, groups // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .reshape_as(scale)
    )
    return (
        shuffled_weight.view(torch.uint8),
        shuffled_scale.view(torch.uint8),
        dequantize_mxfp4(packed, scale).to(torch.bfloat16),
    )


def _packed_test_weights(device, rank, *, npes=4, bf16_attention=False):
    import torch

    from atom.model_ops.monokernel.config import (
        AttentionWeight,
        Mxfp4ScaleLayout,
        Mxfp4WeightLayout,
        glm5_shard_config,
    )
    from atom.model_ops.monokernel.glm.op import Glm5PackedArtifacts
    from atom.model_ops.monokernel.packing import pack_bf16, pack_fp8
    from atom.model_ops.monokernel.weights import LayerWeights

    config = glm5_shard_config(npes)
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
    attention_reference = {}
    for matrix_index, (name, (rows, cols)) in enumerate(matrices.items()):
        weight = torch.zeros(
            rows, cols, dtype=torch.float32, device=device
        )
        row = torch.arange(rows, device=device)
        logical_row = row
        if name == "q_b":
            logical_row = row % (config.nope_dim + config.pe_dim)
        elif name == "uk":
            logical_row = row % config.kv_lora
        elif name == "uv":
            logical_row = row % config.v_dim
        weight[
            row, (logical_row * (matrix_index + 3) + 7) % cols
        ] = (
            0.03125 * (rank + 1)
            if name in {"uv", "o"}
            else 0.125
        )
        if name == "qkv_a":
            kv_row = row - config.q_lora
            keep = (row < config.q_lora) | (
                (kv_row < config.kv_lora) & (kv_row % 2 == 1)
            ) | (
                (kv_row >= config.kv_lora)
                & (kv_row < config.kv_lora + 4)
                & (kv_row >= config.kv_lora + 2)
            )
            weight[~keep] = 0
        elif name == "q_b":
            local = row % (config.nope_dim + config.pe_dim)
            weight[local >= config.nope_dim + 2] = 0
        elif name == "uk":
            weight[row % 2 == 1] = 0
        weight = weight.to(
            torch.bfloat16 if bf16_attention else torch.float8_e4m3fn
        )
        tensors[f"w_{name}"] = (
            pack_bf16(weight) if bf16_attention else pack_fp8(weight)
        )
        if not bf16_attention:
            tensors[f"s_{name}"] = torch.ones(
                max(1, ((rows + 127) // 128) * ((cols + 63) // 64)),
                dtype=torch.float32,
                device=device,
            )
        attention_reference[name] = weight.float()

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
    expert_reference = {}
    ug_w_stride = 2 * config.inter * config.hidden // 2
    ug_s_stride = 2 * config.inter * (config.hidden // 32)
    dn_w_stride = config.hidden * config.inter // 2
    dn_s_stride = config.hidden * (config.inter // 32)
    for expert, seed, factor in (
        (0, 1701, 1.0 + rank * 0.125),
        (config.shared_expert, 2718, 1.0 + rank * 0.25),
    ):
        gen = torch.Generator(device=device).manual_seed(seed)
        ug_source = (
            torch.randn(
                2 * config.inter,
                config.hidden,
                generator=gen,
                device=device,
            )
            * (0.006 * factor)
        ).to(torch.bfloat16)
        dn_source = (
            torch.randn(
                config.hidden,
                config.inter,
                generator=gen,
                device=device,
            )
            * (0.003 * factor)
        ).to(torch.bfloat16)
        ug_weight, ug_scale, ug_reference = _preshuffled_mxfp4(ug_source)
        dn_weight, dn_scale, dn_reference = _preshuffled_mxfp4(dn_source)
        tensors["w_ug"][
            expert * ug_w_stride : (expert + 1) * ug_w_stride
        ].copy_(ug_weight.reshape(-1))
        tensors["s_ug"][
            expert * ug_s_stride : (expert + 1) * ug_s_stride
        ].copy_(ug_scale.reshape(-1))
        tensors["w_dn"][
            expert * dn_w_stride : (expert + 1) * dn_w_stride
        ].copy_(dn_weight.reshape(-1))
        tensors["s_dn"][
            expert * dn_s_stride : (expert + 1) * dn_s_stride
        ].copy_(dn_scale.reshape(-1))
        expert_reference[expert] = (ug_reference, dn_reference)
    router = torch.zeros(
        config.n_experts,
        config.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    router[0, :64] = 0.25
    tensors["w_r"] = pack_bf16(router)
    tensors["bias"][0] = 10.0
    weights = LayerWeights(
        heads=heads,
        t=tensors,
        config=config,
        npes=npes,
        physical_experts=experts,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
    )

    def artifacts(with_indexer: bool):
        return Glm5PackedArtifacts(
            weights=weights,
            tensors=MappingProxyType(tensors),
            npes=npes,
            attention_weight=(
                AttentionWeight.BF16
                if bf16_attention
                else AttentionWeight.FP8_BLOCK128
            ),
            with_indexer=with_indexer,
            expert_mxfp4=True,
            atom_expert_layout=True,
        )

    return weights, artifacts, attention_reference, expert_reference


def _mapped_tp4_test_weights(
    device,
    rank,
    weights,
    attention_reference,
    *,
    ranked_index=False,
):
    import torch

    import atom.models.glm52_mono as glm_adapter

    cfg = weights.config
    source_ptpc = {}

    def ptpc(name, rows, cols, source=None):
        weight = (
            torch.zeros(
                rows,
                cols,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            if source is None
            else source.to(torch.float8_e4m3fn).clone()
        )
        scale = torch.ones(rows, 1, dtype=torch.float32, device=device)
        if source is not None:
            scale.copy_(
                (
                    0.5
                    + (torch.arange(rows, device=device) % 7).float() * 0.125
                ).view(-1, 1)
            )
            source_ptpc[name] = (weight.clone(), scale.clone())
        return SimpleNamespace(
            input_size=cols,
            output_size=rows,
            quant_type=SimpleNamespace(name="per_Token"),
            params_dtype=torch.float8_e4m3fn,
            weight=weight,
            weight_scale=scale,
            is_output_padded=False,
        )

    kv_b_rows = cfg.local_heads * (cfg.nope_dim + cfg.v_dim)
    kv_b_row = torch.arange(kv_b_rows, device=device)
    kv_b_raw = torch.zeros(
        kv_b_rows,
        cfg.kv_lora,
        dtype=torch.float32,
        device=device,
    )
    kv_b_raw[
        kv_b_row,
        (kv_b_row * 11 + rank * 3 + 5) % cfg.kv_lora,
    ] = 0.125
    kv_b_proj = ptpc("kv_b", kv_b_rows, cfg.kv_lora, kv_b_raw)
    kv_b_weight, kv_b_scale = source_ptpc["kv_b"]
    kv_b_source = (kv_b_weight.float() * kv_b_scale).to(torch.bfloat16)
    kv_b_by_head = kv_b_source.view(
        cfg.local_heads,
        cfg.nope_dim + cfg.v_dim,
        cfg.kv_lora,
    )
    uk_source = (
        kv_b_by_head[:, : cfg.nope_dim]
        .transpose(1, 2)
        .contiguous()
        .view(cfg.local_heads * cfg.kv_lora, cfg.nope_dim)
    )
    uv_source = (
        kv_b_by_head[:, cfg.nope_dim :]
        .contiguous()
        .view(cfg.local_heads * cfg.v_dim, cfg.kv_lora)
    )

    def scalar_fp8(source):
        scale = source.float().abs().max() / 448.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (
            (source.float() / scale)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn),
            scale.reshape(()).contiguous(),
        )

    wk, wk_scale = scalar_fp8(
        uk_source.view(cfg.local_heads, cfg.kv_lora, cfg.nope_dim)
    )
    wv, wv_scale = scalar_fp8(
        uv_source.view(cfg.local_heads, cfg.v_dim, cfg.kv_lora)
    )
    index_fused = torch.zeros(
        128 + 32,
        cfg.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    index_fused[:128].copy_(
        attention_reference["index_k"].to(torch.bfloat16)
    )
    if ranked_index:
        index_fused[128:, :64] = 1.0 / 64.0
    probe_indexer = SimpleNamespace(
        wq_b=ptpc(
            "index_q",
            32 * 128,
            cfg.q_lora,
            attention_reference["index_q"],
        ),
        use_wk_weights_proj_fusion=True,
        wk_weights_proj=SimpleNamespace(weight=index_fused),
        k_norm=SimpleNamespace(
            weight=torch.ones(128, dtype=torch.float32, device=device),
            bias=torch.zeros(128, dtype=torch.float32, device=device),
        ),
    )
    probe_attention = SimpleNamespace(
        fused_qkv_a_proj=ptpc(
            "qkv_a",
            cfg.qkv_a_rows,
            cfg.hidden,
            attention_reference["qkv_a"],
        ),
        q_b_proj=ptpc(
            "q_b",
            cfg.local_heads * (cfg.nope_dim + cfg.pe_dim),
            cfg.q_lora,
            attention_reference["q_b"],
        ),
        o_proj=ptpc(
            "o",
            cfg.hidden,
            cfg.local_heads * cfg.v_dim,
            attention_reference["o"],
        ),
        kv_b_proj=kv_b_proj,
        indexer=probe_indexer,
        skip_topk=False,
        mla_attn=SimpleNamespace(
            impl=SimpleNamespace(
                W_K=wk,
                W_K_scale=wk_scale,
                W_V=wv,
                W_V_scale=wv_scale,
            )
        ),
    )
    original_base_mapper = glm_adapter._layer_weights
    glm_adapter._layer_weights = lambda *_args: weights
    try:
        mapped = glm_adapter._agentic_layer_weights(
            SimpleNamespace(self_attn=probe_attention),
            rank,
            4,
        )
    finally:
        glm_adapter._layer_weights = original_base_mapper
    source_reference = {
        name: (weight.float() * scale).to(torch.bfloat16).float()
        for name, (weight, scale) in source_ptpc.items()
    }
    source_reference.update(
        uk=uk_source.float(),
        uv=uv_source.float(),
        index_k=index_fused[:128].float(),
        index_w=index_fused[128:].float(),
    )
    return mapped, source_reference


def _tp4_worker(
    rank: int,
    device_offset: int,
    port: int,
    batch_capacity: int,
    query_len: int,
) -> None:
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.config import KvCacheLayout, SOFTMAX_SCALE
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.graph import (
        GlmAgenticGraphBucket,
        GlmAgenticLayer,
    )
    from atom.model_ops.monokernel.glm.index_share import GlmIndexShareMode
    from atom.model_ops.monokernel.glm.layout import sparse_keys_per_task
    from atom.model_ops.monokernel.glm.op import Glm5MonoKernel
    from atom.model_ops.monokernel.packing import pack_bf16, pack_fp8
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
        weights, artifacts, attention_reference, expert_reference = (
            _packed_test_weights(device, rank)
        )
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
        slot_mapping = torch.full(
            (rows,), -1, dtype=torch.int64, device=device
        )
        batch_ids = torch.full((rows,), -1, dtype=torch.int32, device=device)
        owned_counts = torch.zeros(rows, dtype=torch.int32, device=device)
        if rows < 8:
            positions.copy_(
                torch.arange(40, 40 + rows, dtype=torch.int64, device=device)
            )
            slot_mapping.copy_(
                torch.arange(40, 40 + rows, dtype=torch.int64, device=device)
            )
            batch_ids.zero_()
            owned_counts.fill_(1)
            active_rows = list(range(rows))
        else:
            positions[:2] = torch.tensor(
                [16, 17], dtype=torch.int64, device=device
            )
            positions[tail_row] = 16
            slot_mapping[:2] = torch.tensor(
                [0, 1], dtype=torch.int64, device=device
            )
            slot_mapping[tail_row] = 16
            batch_ids[:2] = 0
            batch_ids[tail_row] = batch_capacity - 1
            owned_counts[:2] = 1
            owned_counts[tail_row] = 1
            active_rows = [0, 1, tail_row]
        block_tables = torch.zeros(
            batch_capacity, 4, dtype=torch.int32, device=device
        )
        block_tables[0] = torch.tensor([3, 0, 2, 1], device=device)
        block_tables[-1] = torch.tensor([2, 1, 3, 0], device=device)
        context_lens = torch.zeros(
            batch_capacity, dtype=torch.int32, device=device
        )
        context_lens[0] = 40 + rows if rows < 8 else 19
        context_lens[-1] = 17
        cache = torch.zeros(
            64, 1, 576, dtype=torch.float8_e4m3fn, device=device
        )
        cache[0:2, 0, 1:512:2] = 7
        for physical in range(32, 48):
            cache[physical, 0, 1:512:2] = physical - 31
        for physical in range(48, 64):
            cache[physical, 0, 1:512:2] = physical - 47
        cache_before = cache.clone()
        cache_scale = torch.ones(1, dtype=torch.float32, device=device)
        index_cache = torch.zeros(
            4, 16, 144, dtype=torch.float8_e4m3fn, device=device
        )
        index_cache_before = index_cache.clone()
        angles = (
            torch.arange(64, device=device, dtype=torch.float32)[:, None]
            * torch.linspace(0.001, 0.031, 32, device=device)[None, :]
        )
        cos = angles.cos().to(torch.bfloat16).contiguous()
        sin = angles.sin().to(torch.bfloat16).contiguous()
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

        cfg = weights.config
        mapped_template, source_reference = _mapped_tp4_test_weights(
            device,
            rank,
            weights,
            attention_reference,
        )
        attention_reference.update(
            {
                name: source_reference[name]
                for name in ("q_b", "o", "uk", "uv")
            }
        )
        qkv_a_oracle_weight = source_reference["qkv_a"]
        real_mapping_calls.append("ptpc")

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

        def pack_layer(mapped, *, with_indexer, attention_weight, **_kwargs):
            artifact_calls.append((id(mapped), with_indexer))
            base = artifacts(with_indexer)
            packed = dict(base.tensors)
            for name in ("qkv_a", "q_b", "o"):
                packed[f"w_{name}"] = pack_fp8(mapped.t[f"w_{name}"])
                packed[f"s_{name}"] = mapped.t[f"s_{name}"]
            for name in ("uk", "uv"):
                packed[f"w_{name}"] = pack_fp8(mapped.t[f"w_{name}"])
                packed[f"s_{name}"] = mapped.t[f"s_{name}"]
            if with_indexer:
                packed["w_index_k"] = pack_fp8(mapped.t["w_index_k"])
                packed["s_index_k"] = mapped.t["s_index_k"]
                packed["w_index_w"] = pack_bf16(mapped.t["w_index_w"])
                packed["w_index_q"] = pack_fp8(mapped.t["w_index_q"])
                packed["s_index_q"] = mapped.t["s_index_q"]
            return Glm5PackedArtifacts(
                weights=mapped,
                tensors=MappingProxyType(packed),
                npes=base.npes,
                attention_weight=attention_weight,
                with_indexer=base.with_indexer,
                expert_mxfp4=base.expert_mxfp4,
                atom_expert_layout=base.atom_expert_layout,
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
                    attention_weight=spec.attention_weight,
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
        assert full is not None and "_a1_" in full.launch.func.__name__
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
        assert torch.equal(
            metadata.glm_agentic_owned_counts,
            owned_counts,
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

        expected_counts = torch.zeros(rows, dtype=torch.int32, device=device)
        expected_parts = []
        expected_by_row = {}
        for row in active_rows:
            request = int(batch_ids[row].item())
            visible = min(
                int(positions[row].item()) + 1,
                int(context_lens[request].item()),
            )
            physical = torch.tensor(
                [
                    int(block_tables[request, logical // 16].item()) * 16
                    + logical % 16
                    for logical in range(visible)
                ],
                dtype=torch.int32,
                device=device,
            )
            expected_counts[row] = visible
            expected_parts.append(physical)
            expected_by_row[row] = physical
        expected = torch.cat(expected_parts)
        expected_indptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                expected_counts.cumsum(0),
            )
        )
        assert torch.equal(counts, expected_counts)
        assert torch.equal(indptr, expected_indptr)
        assert torch.equal(selected[: expected.numel()], expected), (
            selected[: expected.numel()].cpu().tolist(),
            expected.cpu().tolist(),
        )
        assert not selected[expected.numel() :].any()

        nsplit = 2048 // sparse_keys_per_task(8)
        acc_shape = (8, nsplit, weights.config.local_heads, weights.config.kv_lora)

        def quantize_activation(value):
            blocks = value.float().reshape(value.shape[0], -1, 128)
            scale = blocks.abs().amax(-1, keepdim=True) / 448.0
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            return (
                (blocks / scale)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .float()
                * scale
            ).reshape_as(value)

        inter = shared.intermediates()
        shared_acc = shared.debug("sp_acc", acc_shape, bf2=True)
        split_l = shared.debug(
            "sp_l",
            (8, nsplit, weights.config.local_heads),
        )
        kv_new = shared.debug("kvnew", (8, weights.config.kv_lora))
        pe_new = shared.debug("penew", (8, weights.config.pe_dim))
        tile_base = (rows - 1) // 8 * 8
        probe_global = active_rows[-1]
        probe = probe_global - tile_base
        position = int(positions[probe_global].item())

        local_active_rows = [
            row - tile_base
            for row in active_rows
            if tile_base <= row < tile_base + 8
        ]
        layer_inputs = workspace.hidden_buffers[1][
            [tile_base + row for row in local_active_rows]
        ].float()
        normalized_inputs = layer_inputs * torch.rsqrt(
            layer_inputs.square().mean(-1, keepdim=True) + 1e-6
        )
        qkv_reference = torch.nn.functional.linear(
            quantize_activation(normalized_inputs).float(),
            qkv_a_oracle_weight,
        )
        torch.testing.assert_close(
            inter["q_a"][local_active_rows],
            qkv_reference[:, : weights.config.q_lora],
            atol=0.08,
            rtol=3e-2,
        )
        torch.testing.assert_close(
            inter["kv_a"][local_active_rows],
            qkv_reference[:, weights.config.q_lora :],
            atol=0.08,
            rtol=3e-2,
        )
        assert qkv_reference.abs().max() > 0
        qkv_by_local_row = dict(zip(local_active_rows, qkv_reference))

        def kv_cache_reference(qkv_row, row_position):
            kv_raw = qkv_row[weights.config.q_lora :].float()
            kv = (
                kv_raw[: weights.config.kv_lora]
                * torch.rsqrt(
                    kv_raw[: weights.config.kv_lora].square().mean() + 1e-6
                )
            ).to(torch.bfloat16)
            raw_kpe = kv_raw[weights.config.kv_lora :]
            row_cos = cos[row_position].float()
            row_sin = sin[row_position].float()
            kpe = torch.empty_like(raw_kpe)
            kpe[0::2] = (
                raw_kpe[0::2] * row_cos - raw_kpe[1::2] * row_sin
            )
            kpe[1::2] = (
                raw_kpe[0::2] * row_sin + raw_kpe[1::2] * row_cos
            )
            return kv, kpe.to(torch.bfloat16)

        for local_row, qkv_row in zip(local_active_rows, qkv_reference):
            global_row = tile_base + local_row
            qa_row = qkv_row[: weights.config.q_lora]
            qa_row = (
                qa_row
                * torch.rsqrt(qa_row.square().mean() + 1e-6)
            ).to(torch.bfloat16)
            qb_row = torch.nn.functional.linear(
                qa_row.float(), attention_reference["q_b"]
            ).reshape(
                weights.config.local_heads,
                weights.config.nope_dim + weights.config.pe_dim,
            )
            raw_qpe_row = qb_row[:, weights.config.nope_dim :]
            row_position = int(positions[global_row].item())
            row_cos = cos[row_position].float()
            row_sin = sin[row_position].float()
            qpe_row = torch.empty_like(raw_qpe_row)
            qpe_row[:, 0::2] = (
                raw_qpe_row[:, 0::2] * row_cos
                - raw_qpe_row[:, 1::2] * row_sin
            )
            qpe_row[:, 1::2] = (
                raw_qpe_row[:, 0::2] * row_sin
                + raw_qpe_row[:, 1::2] * row_cos
            )
            torch.testing.assert_close(
                inter["q_pe"][local_row],
                qpe_row.to(torch.bfloat16).float(),
                atol=2e-2,
                rtol=2e-2,
            )

        qa = qkv_reference[local_active_rows.index(probe)].float()[
            : weights.config.q_lora
        ]
        qa = (
            qa
            * torch.rsqrt(qa.square().mean() + 1e-6)
        ).to(torch.bfloat16)
        qb = torch.nn.functional.linear(
            qa.float(), attention_reference["q_b"]
        ).reshape(
            weights.config.local_heads,
            weights.config.nope_dim + weights.config.pe_dim,
        )
        raw_qpe = qb[:, weights.config.nope_dim :]
        rope_c = cos[position].float()
        rope_s = sin[position].float()
        qpe_reference = torch.empty_like(raw_qpe)
        qpe_reference[:, 0::2] = (
            raw_qpe[:, 0::2] * rope_c - raw_qpe[:, 1::2] * rope_s
        )
        qpe_reference[:, 1::2] = (
            raw_qpe[:, 0::2] * rope_s + raw_qpe[:, 1::2] * rope_c
        )
        qpe_reference = qpe_reference.to(torch.bfloat16).float()
        torch.testing.assert_close(
            inter["q_pe"][probe],
            qpe_reference,
            atol=2e-2,
            rtol=2e-2,
        )
        assert qpe_reference.abs().max() > 0

        kv_reference, kpe_reference = kv_cache_reference(
            qkv_by_local_row[probe], position
        )
        torch.testing.assert_close(
            kv_new[probe], kv_reference.float(), atol=0.3, rtol=0.12
        )
        torch.testing.assert_close(
            pe_new[probe], kpe_reference.float(), atol=0.08, rtol=5e-2
        )
        assert kpe_reference.abs().max() > 0

        physical = expected_by_row[probe_global].tolist()
        keys, pes = [], []
        request = int(batch_ids[probe_global].item())
        for slot in physical:
            fresh = None
            for global_row in active_rows:
                if (
                    int(batch_ids[global_row].item()) == request
                    and int(slot_mapping[global_row].item()) == slot
                    and int(positions[global_row].item()) <= position
                    and tile_base <= global_row < tile_base + 8
                ):
                    fresh = global_row - tile_base
                    break
            if fresh is None:
                row = cache[slot, 0].float() * cache_scale[0]
                keys.append(row[: weights.config.kv_lora])
                pes.append(row[weights.config.kv_lora :])
            else:
                fresh_global = tile_base + fresh
                fresh_kv, fresh_pe = kv_cache_reference(
                    qkv_by_local_row[fresh],
                    int(positions[fresh_global].item()),
                )
                keys.append(fresh_kv.float())
                pes.append(fresh_pe.float())
        key = torch.stack(keys)
        key_pe = torch.stack(pes)
        q_lat_reference = torch.stack(
            [
                torch.nn.functional.linear(
                    qb[head, : weights.config.nope_dim]
                    .to(torch.bfloat16)
                    .float(),
                    source_reference["uk"][
                        head * weights.config.kv_lora :
                        (head + 1) * weights.config.kv_lora
                    ],
                )
                for head in range(weights.config.local_heads)
            ]
        )
        torch.testing.assert_close(
            inter["q_lat"][probe],
            q_lat_reference,
            atol=0.12,
            rtol=5e-2,
        )
        scores = (
            torch.einsum("hd,kd->hk", q_lat_reference, key)
            + torch.einsum("hd,kd->hk", qpe_reference, key_pe)
        ) * SOFTMAX_SCALE
        probabilities = torch.exp(scores - scores.max(-1, keepdim=True).values)
        acc_reference = torch.einsum("hk,kd->hd", probabilities, key)
        torch.testing.assert_close(
            shared_acc[probe, 0],
            acc_reference,
            atol=0.75,
            rtol=3e-2,
            msg=(
                f"attention row={probe_global}, "
                f"actual_acc={shared_acc[probe, 0].norm().item()}, "
                f"reference_acc={acc_reference.norm().item()}, "
                f"actual_lse={split_l[probe, 0].tolist()}, "
                f"reference_lse={probabilities.sum(-1).tolist()}"
            ),
        )
        torch.testing.assert_close(
            split_l[probe, 0],
            probabilities.sum(-1),
            atol=0.1,
            rtol=3e-2,
        )
        assert acc_reference.abs().max() > 0

        merged = acc_reference / probabilities.sum(-1, keepdim=True)
        uv_reference = torch.cat(
            [
                torch.nn.functional.linear(
                    merged[head],
                    source_reference["uv"][
                        head * weights.config.v_dim :
                        (head + 1) * weights.config.v_dim
                    ],
                )
                for head in range(weights.config.local_heads)
            ]
        )
        torch.testing.assert_close(
            inter["o"][probe],
            uv_reference,
            atol=0.15,
            rtol=5e-2,
        )
        local_attention = torch.nn.functional.linear(
            uv_reference.to(torch.bfloat16).float(),
            source_reference["o"],
        )
        attention_norm = torch.tensor([local_attention.norm().item()])
        attention_norms = [
            torch.zeros_like(attention_norm) for _ in range(4)
        ]
        dist.all_gather(attention_norms, attention_norm)
        assert len({round(value.item(), 6) for value in attention_norms}) == 4
        attention_sum = local_attention.cpu()
        dist.all_reduce(attention_sum)
        attention_sum = attention_sum.to(device)
        layer_input = workspace.hidden_buffers[1][probe_global].float()
        attention_state = (layer_input + attention_sum).to(torch.bfloat16)
        torch.testing.assert_close(
            inter["a"][probe],
            attention_state,
            atol=0.12,
            rtol=5e-2,
        )
        assert attention_sum.abs().max() > 0

        normalized = (
            attention_state.float()
            * torch.rsqrt(
                attention_state.float().square().mean() + 1e-6
            )
        ).to(torch.bfloat16)
        activation = quantize_activation(normalized.unsqueeze(0))[0]
        selected_experts = inter["sel"][probe]
        assert selected_experts[0] == weights.config.shared_expert
        routed_slots = torch.where(selected_experts == 0)[0]
        assert routed_slots.numel() == 1
        routed_slot = int(routed_slots[0].item())
        contributions = {}
        for name, expert, slot in (
            ("shared", weights.config.shared_expert, 0),
            ("routed", 0, routed_slot),
        ):
            ug_reference, dn_reference = expert_reference[expert]
            projected = torch.nn.functional.linear(
                activation.float(), ug_reference.float()
            )
            gate, up = projected.split(weights.config.inter)
            mid_reference = (
                torch.nn.functional.silu(gate) * up
            ).to(torch.bfloat16)
            torch.testing.assert_close(
                inter["mid"][probe, slot],
                mid_reference.float(),
                atol=0.08,
                rtol=5e-2,
            )
            coefficient = (
                1.0 if name == "shared" else inter["prob"][probe, slot]
            )
            local = (
                torch.nn.functional.linear(
                    mid_reference.float(), dn_reference.float()
                )
                * coefficient
            )
            local_norm = torch.tensor([local.norm().item()])
            gathered_norms = [torch.zeros_like(local_norm) for _ in range(4)]
            dist.all_gather(gathered_norms, local_norm)
            assert len({round(value.item(), 6) for value in gathered_norms}) == 4
            total = local.cpu()
            dist.all_reduce(total)
            contributions[name] = total.to(device)
            assert contributions[name].abs().max() > 0
        reference = (
            attention_state.float()
            + contributions["shared"]
            + contributions["routed"]
        ).to(torch.bfloat16)
        for output in (eager_out, graph_out):
            torch.testing.assert_close(
                output[probe_global],
                reference,
                atol=0.15,
                rtol=5e-2,
            )
            device_delta = (
                output[probe_global].float() - attention_state.float()
            )
            torch.testing.assert_close(
                device_delta - contributions["shared"],
                contributions["routed"],
                atol=0.15,
                rtol=5e-2,
            )
            torch.testing.assert_close(
                device_delta - contributions["routed"],
                contributions["shared"],
                atol=0.15,
                rtol=5e-2,
            )
            assert torch.isfinite(output).all()
        assert not torch.equal(
            contributions["shared"], contributions["routed"]
        )
        if rows < 8:
            expected_cache = cache_before.clone()
            for local_row, global_row in enumerate(active_rows):
                slot = int(slot_mapping[global_row].item())
                expected_kv, expected_pe = kv_cache_reference(
                    qkv_by_local_row[local_row],
                    int(positions[global_row].item()),
                )
                expected_cache[slot, 0] = torch.cat(
                    (expected_kv, expected_pe)
                ).to(torch.float8_e4m3fn)
            torch.testing.assert_close(
                cache.float(),
                expected_cache.float(),
                atol=0.5,
                rtol=0.12,
            )
            active_slots = torch.zeros(
                cache.shape[0], dtype=torch.bool, device=device
            )
            active_slots[slot_mapping[active_rows]] = True
            assert torch.equal(
                cache[~active_slots].view(torch.uint8),
                expected_cache[~active_slots].view(torch.uint8),
            )
            last_slot = int(slot_mapping[active_rows[-1]].item())
            torch.testing.assert_close(
                cache[last_slot].float(),
                expected_cache[last_slot].float(),
                atol=0.5,
                rtol=0.12,
            )
            assert torch.equal(
                index_cache[0, rows:8],
                index_cache_before[0, rows:8],
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


def _tp4_selector_worker(rank: int, device_offset: int, port: int) -> None:
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.config import (
        AttentionWeight,
        KvCacheLayout,
    )
    from atom.model_ops.monokernel.glm.cache import (
        index_cache_key_byte_offset,
        index_cache_scale_byte_offset,
    )
    from atom.model_ops.monokernel.glm.index_share import (
        stable_physical_topk,
    )
    from atom.model_ops.monokernel.glm.op import (
        Glm5MonoKernel,
        Glm5PackedArtifacts,
    )
    from atom.model_ops.monokernel.packing import pack_bf16, pack_fp8

    device = torch.device("cuda", device_offset + rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    kernel = None
    try:
        weights, artifacts, attention, _experts = _packed_test_weights(
            device,
            rank,
        )
        mapped, source = _mapped_tp4_test_weights(
            device,
            rank,
            weights,
            attention,
            ranked_index=True,
        )
        base = artifacts(True)
        packed = dict(base.tensors)
        for name in ("qkv_a", "q_b", "uk", "uv", "o"):
            packed[f"w_{name}"] = pack_fp8(mapped.t[f"w_{name}"])
            packed[f"s_{name}"] = mapped.t[f"s_{name}"]
        packed["w_index_k"] = pack_fp8(mapped.t["w_index_k"])
        packed["s_index_k"] = mapped.t["s_index_k"]
        packed["w_index_w"] = pack_bf16(mapped.t["w_index_w"])
        packed["w_index_q"] = pack_fp8(mapped.t["w_index_q"])
        packed["s_index_q"] = mapped.t["s_index_q"]
        packed_artifacts = Glm5PackedArtifacts(
            weights=mapped,
            tensors=MappingProxyType(packed),
            npes=4,
            attention_weight=AttentionWeight.FP8_PER_ROW,
            with_indexer=True,
            expert_mxfp4=base.expert_mxfp4,
            atom_expert_layout=base.atom_expert_layout,
        )
        kernel = Glm5MonoKernel(
            mapped,
            1,
            rank=rank,
            npes=4,
            group=None,
            topk=2048,
            with_indexer=True,
            index_max_seq=4096,
            cache_slots=4096,
            attention_weight=AttentionWeight.FP8_PER_ROW,
            kv_cache_layout=KvCacheLayout.ATOM_FP8,
            agentic_row_contract=True,
            packed_artifacts=packed_artifacts,
        )
        cfg = mapped.config
        hidden = torch.randn(
            1,
            cfg.hidden,
            generator=torch.Generator(device=device).manual_seed(1204),
            dtype=torch.bfloat16,
            device=device,
        )
        position = 2111
        positions = torch.tensor([position], dtype=torch.int64, device=device)
        slot_mapping = torch.full(
            (1,), -1, dtype=torch.int64, device=device
        )
        batch_ids = torch.zeros(1, dtype=torch.int32, device=device)
        owned_counts = torch.ones(1, dtype=torch.int32, device=device)
        sparse_indptr = torch.zeros(2, dtype=torch.int32, device=device)
        indices = torch.full(
            (2048,), -1, dtype=torch.int32, device=device
        )
        block_tables = (
            torch.arange(256, dtype=torch.int32, device=device) * 73
        ).remainder(256).view(1, -1).contiguous()
        context_lens = torch.tensor(
            [position + 1], dtype=torch.int32, device=device
        )
        cache = torch.zeros(
            4096,
            1,
            576,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        cache_scale = torch.ones(1, dtype=torch.float32, device=device)
        index_cache = torch.zeros(
            256,
            16,
            144,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        logical = torch.arange(position + 1, device=device)
        physical = (
            block_tables[0, logical // 16] * 16 + logical % 16
        ).to(torch.int64)
        key_values = (
            ((logical * 977).remainder(position + 1) + 1).float() / 256.0
        )
        storage = index_cache.view(torch.uint8).reshape(-1)
        key_offsets = torch.tensor(
            [
                index_cache_key_byte_offset(int(slot), 0)
                for slot in physical.cpu().tolist()
            ],
            dtype=torch.int64,
            device=device,
        )
        fp8_max_byte = (
            torch.tensor(
                448.0,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            .view(torch.uint8)
            .item()
        )
        storage[key_offsets] = fp8_max_byte
        scales = (key_values / 448.0).contiguous()
        scale_offsets = torch.tensor(
            [
                index_cache_scale_byte_offset(int(slot))
                for slot in physical.cpu().tolist()
            ],
            dtype=torch.int64,
            device=device,
        )
        byte_offsets = scale_offsets[:, None] + torch.arange(
            4, dtype=torch.int64, device=device
        )
        storage[byte_offsets] = scales.view(torch.uint8).view(-1, 4)
        cos = torch.ones(
            4096, 32, dtype=torch.bfloat16, device=device
        )
        sin = torch.zeros_like(cos)

        kernel.forward(
            hidden,
            torch.tensor([position], dtype=torch.int32, device=device),
            cache,
            cache,
            indices,
            cos,
            sin,
            index_cache=index_cache,
            positions=positions,
            slot_mapping=slot_mapping,
            sparse_kv_indptr=sparse_indptr,
            batch_ids=batch_ids,
            owned_counts=owned_counts,
            kv_cache_scale=cache_scale,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        torch.cuda.synchronize(device)
        index_k_actual = kernel.debug("index_k", (1, 128))
        index_q_actual = kernel.debug(
            "index_q",
            (1, 32, 128),
            bf2=True,
        )
        index_weights_actual = kernel.debug("index_w", (1, 32))

        def block_round(value):
            blocks = value.float().reshape(value.shape[0], -1, 128)
            scale = blocks.abs().amax(-1, keepdim=True) / 448.0
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            return (
                (blocks / scale)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .float()
                * scale
            ).reshape_as(value)

        normalized = (
            hidden.float()
            * torch.rsqrt(
                hidden.float().square().mean(-1, keepdim=True) + 1e-6
            )
        ).to(torch.bfloat16)
        qkv = torch.nn.functional.linear(
            block_round(normalized.float()),
            source["qkv_a"],
        )
        index_k_reference = torch.nn.functional.linear(
            normalized.float(),
            source["index_k"],
        )
        torch.testing.assert_close(
            index_k_actual,
            index_k_reference,
            atol=0.002,
            rtol=2e-3,
        )
        qa = qkv[:, : cfg.q_lora]
        qa = (
            qa
            * torch.rsqrt(qa.square().mean(-1, keepdim=True) + 1e-6)
        ).to(torch.bfloat16)
        index_q_reference = torch.nn.functional.linear(
            block_round(qa),
            source["index_q"],
        ).view(1, 32, 128)
        index_q_reference = index_q_reference.to(torch.bfloat16).float()
        torch.testing.assert_close(
            index_q_actual,
            index_q_reference,
            atol=0.08,
            rtol=3e-2,
        )
        index_weights = torch.nn.functional.linear(
            normalized.float(),
            source["index_w"],
        )
        torch.testing.assert_close(
            index_weights_actual,
            index_weights,
            atol=0.002,
            rtol=2e-3,
        )
        index_weights = index_weights[0]
        key_dequant = (448.0 * scales).to(torch.bfloat16).float()
        scores = (
            torch.relu(index_q_reference[0, :, 0, None] * key_dequant)
            * index_weights[:, None]
        ).sum(0)
        expected, expected_count = stable_physical_topk(
            scores.cpu(),
            block_tables.cpu(),
            batch_id=0,
            position=position,
            request_context=position + 1,
            topk=2048,
        )
        assert expected_count == 2048
        assert kernel.index_counts is not None
        assert kernel.index_counts[0].item() == expected_count
        assert sparse_indptr.cpu().tolist() == [0, expected_count]
        assert torch.equal(indices.cpu(), expected)
    finally:
        if kernel is not None:
            kernel.close()
        dist.destroy_process_group()


def _tp8_worker(rank: int, port: int) -> None:
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.config import (
        AttentionWeight,
        KvCacheLayout,
        SOFTMAX_SCALE,
    )
    from atom.model_ops.monokernel.glm.op import Glm5MonoKernel

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=90),
    )
    kernel = None
    try:
        weights, artifacts, attention, experts = _packed_test_weights(
            device,
            rank,
            npes=8,
            bf16_attention=True,
        )
        kernel = Glm5MonoKernel(
            weights,
            1,
            rank=rank,
            npes=8,
            group=dist.group.WORLD,
            topk=64,
            attention_weight=AttentionWeight.BF16,
            kv_cache_layout=KvCacheLayout.ATOM,
            agentic_row_contract=True,
            packed_artifacts=artifacts(False),
        )
        config = weights.config
        generator = torch.Generator(device=device).manual_seed(407)
        hidden = torch.randn(
            1,
            config.hidden,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        position = 11
        positions = torch.tensor([position], dtype=torch.int64, device=device)
        slot_mapping = positions.clone()
        batch_ids = torch.zeros(1, dtype=torch.int32, device=device)
        owned_counts = torch.ones(1, dtype=torch.int32, device=device)
        sparse_indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
        indices = torch.zeros(64, dtype=torch.int32, device=device)
        indices[:2] = torch.tensor([3, position], dtype=torch.int32, device=device)
        cache = torch.zeros(
            64,
            config.kv_lora + config.pe_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        cache[3] = torch.linspace(
            -0.25,
            0.25,
            config.kv_lora + config.pe_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        angles = (
            torch.arange(64, device=device, dtype=torch.float32)[:, None]
            * torch.linspace(0.003, 0.071, 32, device=device)[None, :]
        )
        cos = angles.cos().to(torch.bfloat16).contiguous()
        sin = angles.sin().to(torch.bfloat16).contiguous()

        output = kernel.forward(
            hidden,
            torch.tensor([position], dtype=torch.int32, device=device),
            cache,
            cache,
            indices,
            cos,
            sin,
            positions=positions,
            slot_mapping=slot_mapping,
            sparse_kv_indptr=sparse_indptr,
            batch_ids=batch_ids,
            owned_counts=owned_counts,
        )
        torch.cuda.synchronize(device)
        inter = kernel.intermediates()

        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + 1e-6
        )
        qkv = torch.nn.functional.linear(normalized, attention["qkv_a"])
        torch.testing.assert_close(
            inter["q_a"][0],
            qkv[0, : config.q_lora],
            atol=0.06,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            inter["kv_a"][0],
            qkv[0, config.q_lora :],
            atol=0.06,
            rtol=2e-2,
        )
        qa = qkv[0, : config.q_lora]
        qa = (qa * torch.rsqrt(qa.square().mean() + 1e-6)).to(torch.bfloat16)
        qb = torch.nn.functional.linear(
            qa.float(), attention["q_b"]
        ).reshape(
            config.local_heads,
            config.nope_dim + config.pe_dim,
        )
        raw_qpe = qb[:, config.nope_dim :]
        qpe = torch.empty_like(raw_qpe)
        qpe[:, 0::2] = (
            raw_qpe[:, 0::2] * cos[position].float()
            - raw_qpe[:, 1::2] * sin[position].float()
        )
        qpe[:, 1::2] = (
            raw_qpe[:, 0::2] * sin[position].float()
            + raw_qpe[:, 1::2] * cos[position].float()
        )
        qpe = qpe.to(torch.bfloat16).float()
        torch.testing.assert_close(
            inter["q_pe"][0], qpe, atol=2e-2, rtol=2e-2
        )
        assert qpe.abs().max() > 0

        kv_raw = qkv[0, config.q_lora :]
        kv = kv_raw[: config.kv_lora]
        kv = (kv * torch.rsqrt(kv.square().mean() + 1e-6)).to(torch.bfloat16)
        raw_kpe = kv_raw[config.kv_lora :]
        kpe = torch.empty_like(raw_kpe)
        kpe[0::2] = (
            raw_kpe[0::2] * cos[position].float()
            - raw_kpe[1::2] * sin[position].float()
        )
        kpe[1::2] = (
            raw_kpe[0::2] * sin[position].float()
            + raw_kpe[1::2] * cos[position].float()
        )
        kpe = kpe.to(torch.bfloat16)
        torch.testing.assert_close(
            cache[position],
            torch.cat((kv, kpe)),
            atol=2e-2,
            rtol=1e-2,
        )

        keys = torch.stack(
            (cache[3, : config.kv_lora], cache[position, : config.kv_lora])
        ).float()
        key_pe = torch.stack(
            (cache[3, config.kv_lora :], cache[position, config.kv_lora :])
        ).float()
        q_lat_reference = torch.stack(
            [
                torch.nn.functional.linear(
                    qb[head, : config.nope_dim].to(torch.bfloat16).float(),
                    attention["uk"][
                        head * config.kv_lora :
                        (head + 1) * config.kv_lora
                    ],
                )
                for head in range(config.local_heads)
            ]
        )
        torch.testing.assert_close(
            inter["q_lat"][0],
            q_lat_reference,
            atol=0.08,
            rtol=3e-2,
        )
        q_lat = inter["q_lat"][0].float()
        scores = (
            torch.einsum("hd,kd->hk", q_lat, keys)
            + torch.einsum("hd,kd->hk", inter["q_pe"][0].float(), key_pe)
        ) * SOFTMAX_SCALE
        probability = torch.softmax(scores, dim=-1)
        merged = torch.einsum("hk,kd->hd", probability, keys)
        uv = torch.cat(
            [
                torch.nn.functional.linear(
                    merged[head],
                    attention["uv"][
                        head * config.v_dim : (head + 1) * config.v_dim
                    ],
                )
                for head in range(config.local_heads)
            ]
        )
        torch.testing.assert_close(
            inter["o"][0],
            uv,
            atol=0.08 * (rank + 1),
            rtol=8e-2,
        )
        assert uv.abs().max() > 0
        local_attention = torch.nn.functional.linear(
            inter["o"][0].to(torch.bfloat16).float(),
            attention["o"],
        ).cpu()
        dist.all_reduce(local_attention)
        attention_state = (hidden[0].float().cpu() + local_attention).to(
            device=device,
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(
            inter["a"][0], attention_state, atol=0.12, rtol=5e-2
        )

        expert_input = inter["a"][0].float()
        expert_input *= torch.rsqrt(expert_input.square().mean() + 1e-6)
        blocks = expert_input.reshape(-1, 128)
        scale = blocks.abs().amax(-1, keepdim=True) / 448.0
        expert_input = (
            (blocks / torch.where(scale > 0, scale, torch.ones_like(scale)))
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .float()
            * torch.where(scale > 0, scale, torch.ones_like(scale))
        ).reshape(-1)
        contributions = {}
        selected = inter["sel"][0]
        routed_slot = int(torch.where(selected == 0)[0][0].item())
        for name, expert, slot in (
            ("shared", config.shared_expert, 0),
            ("routed", 0, routed_slot),
        ):
            ug, dn = experts[expert]
            gate, up = torch.nn.functional.linear(
                expert_input, ug.float()
            ).split(config.inter)
            mid = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            torch.testing.assert_close(
                inter["mid"][0, slot],
                mid.float(),
                atol=0.08,
                rtol=5e-2,
            )
            coefficient = (
                1.0 if name == "shared" else inter["prob"][0, slot]
            )
            contribution = (
                torch.nn.functional.linear(mid.float(), dn.float())
                * coefficient
            ).cpu()
            dist.all_reduce(contribution)
            contributions[name] = contribution.to(device)
            assert contributions[name].abs().max() > 0
        expected = (
            inter["a"][0].float()
            + contributions["shared"]
            + contributions["routed"]
        ).to(torch.bfloat16)
        torch.testing.assert_close(
            output[0], expected, atol=0.15, rtol=5e-2
        )
    finally:
        if kernel is not None:
            kernel.close()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("batch_capacity", "query_len"),
    ((1, 5), (1, 6), (16, 6), (20, 5)),
)
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


def test_glm_tp4_full_selector_real_rocm_launch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available():
        pytest.skip("ROCm device is unavailable")
    if torch.cuda.device_count() < 4:
        pytest.skip("TP4 selector launch requires four visible ROCm devices")
    try:
        from flydsl.runtime.device import get_rocm_arch
    except ImportError:
        pytest.skip("FlyDSL ROCm compiler support is unavailable")
    if str(get_rocm_arch() or "") != "gfx950":
        pytest.skip(f"GLM TP4 MonoKernel requires gfx950, got {get_rocm_arch()}")

    import torch.multiprocessing as mp

    offset = max(0, torch.cuda.device_count() - 4)
    mp.spawn(
        _tp4_selector_worker,
        args=(offset, _free_port()),
        nprocs=4,
        join=True,
    )


def test_glm_tp8_bf16_legacy_real_rocm_launch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available():
        pytest.skip("ROCm device is unavailable")
    if torch.cuda.device_count() < 8:
        pytest.skip("TP8 device launch requires eight visible ROCm devices")
    try:
        from flydsl.runtime.device import get_rocm_arch
    except ImportError:
        pytest.skip("FlyDSL ROCm compiler support is unavailable")
    if str(get_rocm_arch() or "") != "gfx950":
        pytest.skip(f"GLM TP8 MonoKernel requires gfx950, got {get_rocm_arch()}")

    import torch.multiprocessing as mp

    mp.spawn(
        _tp8_worker,
        args=(_free_port(),),
        nprocs=8,
        join=True,
    )
