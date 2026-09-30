# SPDX-License-Identifier: Apache-2.0

"""Real TP8 eager/HIP-graph gate for the Kimi q=8 full MonoKernel."""

from __future__ import annotations

import math
import socket

import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _zero_weights(device, rank):
    import torch

    from atom.model_ops.monokernel.config import (
        KIMI_K3_CONFIG,
        Mxfp4ScaleLayout,
        Mxfp4WeightLayout,
    )
    from atom.model_ops.monokernel.weights import LayerWeights

    config = KIMI_K3_CONFIG
    hidden = config.hidden
    routed = config.routed_hidden
    shared = config.shared_inter
    assert routed is not None and shared is not None
    projection = config.local_heads * config.v_dim
    fused = 4 * projection + config.local_heads + config.v_dim
    shard = hidden // 8

    def bf16(*shape, value=0):
        return torch.full(shape, value, dtype=torch.bfloat16, device=device)

    tensors = {
        "w_r": bf16(config.n_experts, hidden),
        "bias": torch.zeros(config.n_experts, dtype=torch.float32, device=device),
        "w_latent_down": bf16(routed, hidden),
        "g_latent": bf16(routed, value=1),
        "w_latent_up": bf16(shard, routed),
        "w_shared_ug": bf16(2 * shared, hidden),
        "w_shared_dn": bf16(hidden, shared),
        "w_kda_in": bf16(fused, hidden),
        "w_kda_fb": bf16(projection, config.v_dim),
        "w_kda_conv": bf16(3 * projection, 4),
        "kda_a_log": torch.zeros(
            config.local_heads,
            dtype=torch.float32,
            device=device,
        ),
        "kda_dt_bias": bf16(config.local_heads, config.v_dim),
        "g_kda_out": bf16(config.v_dim, value=1),
        "w_kda_o": bf16(hidden, projection),
    }
    for name in (
        "g_self_res",
        "w_self_res",
        "g_in",
        "g_mlp_res",
        "w_mlp_res",
        "g_post",
    ):
        tensors[name] = bf16(hidden, value=1)

    experts = config.n_experts
    ug_rows = experts * 2 * config.inter
    dn_rows = experts * routed
    tensors["w_ug"] = torch.zeros(
        ug_rows * routed // 2,
        dtype=torch.uint8,
        device=device,
    )
    tensors["w_dn"] = torch.zeros(
        dn_rows * config.inter // 2,
        dtype=torch.uint8,
        device=device,
    )
    tensors["w_ug"].is_shuffled = True
    tensors["w_dn"].is_shuffled = True

    def scale(rows, width):
        return torch.zeros(
            math.ceil(rows / 256) * 256,
            math.ceil((width // 32) / 8) * 8,
            dtype=torch.uint8,
            device=device,
        )

    tensors["s_ug"] = scale(ug_rows, routed)
    tensors["s_dn"] = scale(dn_rows, config.inter)
    return LayerWeights(
        heads=config.local_heads,
        t=tensors,
        config=config,
        rank=rank,
        npes=8,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
    )


def _apply_reference(conv, recurrent, snapshots, accepted):
    import torch

    next_conv = conv.clone()
    next_recurrent = recurrent.clone()
    decay = math.exp(-2.5)
    for request, count in enumerate(accepted.tolist()):
        conv_slot = snapshots[request, 0].item()
        next_conv[conv_slot, :2] = conv[conv_slot, count : count + 2]
        next_conv[conv_slot, 2:].zero_()
        for token in range(8):
            source = (
                snapshots[request, count - 1]
                if token == 0
                else snapshots[request, token - 1]
            ).item()
            target = snapshots[request, token].item()
            next_recurrent[target] = (
                next_recurrent[source].float() * decay
            ).to(torch.float16)
    return next_conv, next_recurrent


def _check_state(actual_conv, actual_recurrent, expected_conv, expected_recurrent):
    import torch

    assert torch.equal(actual_conv, expected_conv)
    torch.testing.assert_close(
        actual_recurrent,
        expected_recurrent,
        atol=2e-3,
        rtol=2e-3,
    )


def _exercise_batch(op, batch, device):
    import torch

    from atom.model_ops.monokernel.config import KIMI_K3_CONFIG

    rows = batch * 8
    slots = rows
    snapshots = torch.arange(
        slots,
        dtype=torch.int32,
        device=device,
    ).view(batch, 8)
    conv_initial = torch.arange(
        slots * 10 * 3 * KIMI_K3_CONFIG.local_heads * KIMI_K3_CONFIG.v_dim,
        dtype=torch.int64,
        device=device,
    ).remainder_(251).to(torch.bfloat16).view(
        slots,
        10,
        3 * KIMI_K3_CONFIG.local_heads * KIMI_K3_CONFIG.v_dim,
    )
    recurrent_initial = torch.empty(
        slots,
        KIMI_K3_CONFIG.local_heads,
        KIMI_K3_CONFIG.v_dim,
        KIMI_K3_CONFIG.v_dim,
        dtype=torch.float16,
        device=device,
    )
    for slot in range(slots):
        recurrent_initial[slot].fill_((slot + 1) / 64)

    conv = conv_initial.clone()
    recurrent = recurrent_initial.clone()
    prefix = torch.zeros(
        rows,
        KIMI_K3_CONFIG.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    blocks = torch.zeros(
        rows,
        1,
        KIMI_K3_CONFIG.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.full_like(prefix, float("nan"))
    accepted = torch.empty(batch, dtype=torch.int32, device=device)
    pointers = (conv.data_ptr(), recurrent.data_ptr(), accepted.data_ptr())

    eager_counts = (
        (4,) if batch == 1 else (1, 3, 6, 8),
        (7,) if batch == 1 else (8, 2, 5, 1),
    )
    expected_conv = conv_initial.clone()
    expected_recurrent = recurrent_initial.clone()
    for counts in eager_counts:
        accepted.copy_(torch.tensor(counts, dtype=torch.int32, device=device))
        output.fill_(float("nan"))
        op.forward(
            prefix,
            blocks,
            snapshots,
            conv,
            recurrent,
            x_out=output,
            num_accepted_tokens=accepted,
        )
        torch.cuda.synchronize(device)
        expected_conv, expected_recurrent = _apply_reference(
            expected_conv,
            expected_recurrent,
            snapshots,
            accepted,
        )
        _check_state(conv, recurrent, expected_conv, expected_recurrent)
        assert torch.isfinite(output).all()

    conv.copy_(conv_initial)
    recurrent.copy_(recurrent_initial)
    expected_conv.copy_(conv_initial)
    expected_recurrent.copy_(recurrent_initial)
    graph_counts = (
        (2,) if batch == 1 else (2, 7, 1, 5),
        (8,) if batch == 1 else (8, 1, 4, 6),
        (3,) if batch == 1 else (3, 3, 8, 2),
    )
    accepted.copy_(torch.tensor(graph_counts[0], dtype=torch.int32, device=device))
    op.forward(
        prefix,
        blocks,
        snapshots,
        conv,
        recurrent,
        x_out=output,
        num_accepted_tokens=accepted,
    )
    conv.copy_(conv_initial)
    recurrent.copy_(recurrent_initial)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.forward(
            prefix,
            blocks,
            snapshots,
            conv,
            recurrent,
            x_out=output,
            num_accepted_tokens=accepted,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected_conv, expected_recurrent = _apply_reference(
        expected_conv,
        expected_recurrent,
        snapshots,
        accepted,
    )
    _check_state(conv, recurrent, expected_conv, expected_recurrent)
    assert torch.isfinite(output).all()

    for counts in graph_counts[1:]:
        accepted.copy_(torch.tensor(counts, dtype=torch.int32, device=device))
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        expected_conv, expected_recurrent = _apply_reference(
            expected_conv,
            expected_recurrent,
            snapshots,
            accepted,
        )
        _check_state(conv, recurrent, expected_conv, expected_recurrent)
        assert torch.isfinite(output).all()
        assert pointers == (
            conv.data_ptr(),
            recurrent.data_ptr(),
            accepted.data_ptr(),
        )


def _tp8_worker(rank: int, port: int) -> None:
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.config import ConvStateLayout
    from atom.model_ops.monokernel.k3.op import KimiK3MonoKernel

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
    )
    try:
        weights = _zero_weights(device, rank)
        packed = None
        for batch in (1, 4):
            op = KimiK3MonoKernel(
                weights,
                batch * 8,
                layer_idx=0,
                rank=rank,
                npes=8,
                group=None,
                reduce_group=dist.group.WORLD,
                mtp=True,
                agentic_batch_size=batch,
                conv_state_layout=ConvStateLayout.TIME_MAJOR,
                state_dtype=torch.float16,
                packed_artifacts=packed,
            )
            packed = op.packed_artifacts()
            _exercise_batch(op, batch, device)
            op.close()
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_kimi_agentic_tp8_q8_eager_and_graph_real_rocm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip("Kimi Agentic device gate requires eight ROCm devices")

    import torch.multiprocessing as mp
    from flydsl.runtime.device import get_rocm_arch

    if str(get_rocm_arch() or "") != "gfx950":
        pytest.skip(f"Kimi Agentic MonoKernel requires gfx950, got {get_rocm_arch()}")
    mp.spawn(_tp8_worker, args=(_free_port(),), nprocs=8, join=True)
