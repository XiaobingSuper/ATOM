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


def _deterministic_weights(device, rank):
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
        "kda_dt_bias": bf16(
            config.local_heads,
            config.v_dim,
            value=-10.375,
        ),
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

    rows = torch.arange(fused, device=device)
    tensors["w_kda_in"][rows, rows.remainder(16)] = (
        0.015625 + rows.remainder(7).to(torch.float32) / 1024
    ).to(torch.bfloat16)
    tensors["w_kda_conv"][:, 0] = 0.0625
    tensors["w_kda_conv"][:, 1] = 0.125
    tensors["w_kda_conv"][:, 2] = 0.25
    tensors["w_kda_conv"][:, 3] = 0.5
    gate_rows = torch.arange(projection, device=device)
    tensors["w_kda_fb"][gate_rows, gate_rows.remainder(config.v_dim)] = 0.03125
    output_rows = torch.arange(hidden, device=device)
    rank_scale = (rank + 1) / 1024
    tensors["w_kda_o"][
        output_rows, output_rows.remainder(projection)
    ] = rank_scale
    tensors["w_r"][0, :16] = torch.linspace(
        -0.125, 0.125, 16, dtype=torch.bfloat16, device=device
    )
    latent_rows = torch.arange(routed, device=device)
    tensors["w_latent_down"][latent_rows, latent_rows.remainder(16)] = 0.03125
    shared_rows = torch.arange(2 * shared, device=device)
    tensors["w_shared_ug"][shared_rows, shared_rows.remainder(16)] = 0.03125
    tensors["w_shared_dn"][
        output_rows, output_rows.remainder(shared)
    ] = rank_scale
    shard_rows = torch.arange(shard, device=device)
    tensors["w_latent_up"][
        shard_rows, shard_rows.remainder(routed)
    ] = rank_scale

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
    tensors["w_ug"].fill_(0x11)
    tensors["w_dn"].fill_(0x11)
    tensors["s_ug"].fill_(120)
    tensors["s_dn"].fill_(120)
    return LayerWeights(
        heads=config.local_heads,
        t=tensors,
        config=config,
        rank=rank,
        npes=8,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
    )


def _deterministic_mla_weights(device, rank):
    import torch

    from atom.model_ops.monokernel.config import KIMI_K3_CONFIG

    weights = _deterministic_weights(device, rank)
    config = KIMI_K3_CONFIG
    hidden = config.hidden
    projection = config.local_heads * config.v_dim

    def bf16(*shape, value=0):
        return torch.full(
            shape,
            value,
            dtype=torch.bfloat16,
            device=device,
        )

    tensors = weights.t
    tensors["w_qkv_a"] = bf16(config.qkv_a_rows, hidden)
    tensors["g_q"] = bf16(config.q_lora, value=1)
    tensors["g_kv"] = bf16(config.kv_lora, value=1)
    tensors["w_q_b"] = bf16(
        config.local_heads * (config.nope_dim + config.pe_dim),
        config.q_lora,
    )
    tensors["w_uk"] = bf16(
        config.local_heads * config.kv_lora,
        config.nope_dim,
    )
    tensors["w_uv"] = bf16(projection, config.kv_lora)
    tensors["w_gate"] = bf16(projection, hidden)
    tensors["w_o"] = bf16(hidden, projection)

    qkv_rows = torch.arange(config.qkv_a_rows, device=device)
    tensors["w_qkv_a"][
        qkv_rows,
        qkv_rows.remainder(32),
    ] = 0.03125
    q_rows = torch.arange(
        config.local_heads * (config.nope_dim + config.pe_dim),
        device=device,
    )
    tensors["w_q_b"][
        q_rows,
        q_rows.remainder(config.q_lora),
    ] = 0.0625
    uk_rows = torch.arange(
        config.local_heads * config.kv_lora,
        device=device,
    )
    tensors["w_uk"][
        uk_rows,
        uk_rows.remainder(config.nope_dim),
    ] = 0.0625
    uv_rows = torch.arange(projection, device=device)
    tensors["w_uv"][
        uv_rows,
        uv_rows.remainder(config.kv_lora),
    ] = 0.0625
    tensors["w_gate"][
        uv_rows,
        uv_rows.remainder(32),
    ] = 0.03125
    output_rows = torch.arange(hidden, device=device)
    tensors["w_o"][
        output_rows,
        output_rows.remainder(projection),
    ] = (rank + 1) / 1024
    return weights


def _projected_input(prefix, weights):
    import torch

    fused = weights["w_kda_in"].shape[0]
    rows = torch.arange(fused, device=prefix.device)
    columns = rows.remainder(16)
    return (
        prefix[:, columns].float()
        * weights["w_kda_in"][rows, columns].float().unsqueeze(0)
    ).to(torch.bfloat16)


def _apply_reference(
    conv,
    recurrent,
    snapshots,
    accepted,
    projected,
    weights,
):
    import torch
    import torch.nn.functional as functional

    from atom.model_ops.monokernel.config import KIMI_K3_CONFIG
    from atom.model_ops.fla_ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    config = KIMI_K3_CONFIG
    projection = config.local_heads * config.v_dim
    next_conv = conv.clone()
    next_recurrent = recurrent.float()
    query = torch.empty(
        snapshots.shape[0],
        8,
        config.local_heads,
        config.v_dim,
        dtype=torch.bfloat16,
        device=conv.device,
    )
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    gate = torch.empty_like(query)
    beta = torch.empty(
        snapshots.shape[0],
        8,
        config.local_heads,
        dtype=torch.bfloat16,
        device=conv.device,
    )
    for request, count in enumerate(accepted.tolist()):
        conv_slot = snapshots[request, 0].item()
        old_window = conv[conv_slot].clone()
        draft = projected[request * 8 : (request + 1) * 8, : 3 * projection]
        next_conv[conv_slot, :2] = old_window[count : count + 2]
        next_conv[conv_slot, 2:] = draft
        for token in range(8):
            history = torch.cat((old_window[count - 1 : count + 2], draft[:token]))
            conv_input = torch.cat((history[-3:], draft[token : token + 1]))
            qkv = functional.silu(
                (
                    conv_input.float()
                    * weights["w_kda_conv"].float().transpose(0, 1)
                ).sum(dim=0)
            ).to(torch.bfloat16)
            row = projected[request * 8 + token]
            f_a = row[
                4 * projection
                + config.local_heads : 4 * projection
                + config.local_heads
                + config.v_dim
            ].float()
            for head in range(config.local_heads):
                start = head * config.v_dim
                query[request, token, head] = qkv[
                    start : start + config.v_dim
                ]
                key[request, token, head] = qkv[
                    projection + start : projection + start + config.v_dim
                ]
                value[request, token, head] = qkv[
                    2 * projection
                    + start : 2 * projection
                    + start
                    + config.v_dim
                ]
                gate[request, token, head] = (
                    weights["w_kda_fb"][
                        start : start + config.v_dim
                    ].float()
                    @ f_a
                ).to(torch.bfloat16)
                beta[request, token, head] = row[4 * projection + head]
    _, next_recurrent = fused_sigmoid_gating_delta_rule_update(
        weights["kda_a_log"],
        gate,
        beta,
        weights["kda_dt_bias"],
        query,
        key,
        value,
        initial_state=next_recurrent,
        inplace_final_state=True,
        ssm_state_indices=snapshots,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
    )
    final_fp32 = next_recurrent[snapshots[:, -1].to(torch.int64)].clone()
    pairwise_final = torch.empty_like(final_fp32)
    for request, count in enumerate(accepted.tolist()):
        state = recurrent[
            snapshots[request, count - 1].item()
        ].float().unsqueeze(0)
        for pair in range(4):
            begin = pair * 2
            _, pair_states = fused_sigmoid_gating_delta_rule_update(
                weights["kda_a_log"],
                gate[request : request + 1, begin : begin + 2],
                beta[request : request + 1, begin : begin + 2],
                weights["kda_dt_bias"],
                query[request : request + 1, begin : begin + 2],
                key[request : request + 1, begin : begin + 2],
                value[request : request + 1, begin : begin + 2],
                initial_state=state,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
                is_kda=True,
                lower_bound=-5.0,
            )
            state = pair_states[-1:].to(torch.float16).float()
        pairwise_final[request] = pair_states[-1]
    return (
        next_conv,
        next_recurrent.to(torch.float16),
        final_fp32,
        pairwise_final,
    )


def _check_state(actual_conv, actual_recurrent, expected_conv, expected_recurrent):
    import torch

    if not torch.equal(actual_conv, expected_conv):
        mismatch = (actual_conv != expected_conv).nonzero()[0].tolist()
        index = tuple(mismatch)
        raise AssertionError(
            "conv mismatch at "
            f"{index}: actual={actual_conv[index].item()} "
            f"expected={expected_conv[index].item()}"
        )
    try:
        torch.testing.assert_close(
            actual_recurrent,
            expected_recurrent,
            atol=1e-3,
            rtol=1e-3,
        )
    except AssertionError as error:
        delta = (actual_recurrent.float() - expected_recurrent.float()).abs()
        slot_max = delta.flatten(1).amax(1)
        raise AssertionError(
            f"{error}\nper-slot max={slot_max.tolist()}"
        ) from error


def _check_fp32_handoff(op, batch, expected, pairwise):
    import torch

    from atom.model_ops.monokernel.k3.kernel import monokernel_layout

    layout = monokernel_layout(
        batch * 8,
        fuse_attn_res=True,
        fuse_moe=True,
        mtp=True,
    )
    elements = batch * 12 * 128 * 128
    actual = (
        op.attention.monokernel_scratch[
            layout["mtp_state_handoff"] : layout["mtp_state_handoff"]
            + elements * 4
        ]
        .view(torch.float32)
        .view(batch, 12, 128, 128)
    )
    continuous_error = (actual - expected).abs().amax()
    pairwise_error = (actual - pairwise).abs().amax()
    assert continuous_error < pairwise_error, (
        f"continuous error {continuous_error.item()} is not below "
        f"pairwise-FP16 error {pairwise_error.item()}"
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def _check_moe_and_tp_output(op, rows, output):
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.k3.kernel import monokernel_layout

    layout = monokernel_layout(
        rows,
        fuse_attn_res=True,
        fuse_moe=True,
        mtp=True,
    )
    scratch = op.attention.monokernel_scratch
    regions = (
        ("router", rows * 896 * 4),
        ("shared_mid", rows * 768 * 4),
        ("routed", rows * 3584 * 2),
    )
    for name, size in regions:
        region = scratch[layout[name] : layout[name] + size]
        assert torch.count_nonzero(region) > 0, f"{name} path was not observable"

    checksum = torch.stack(
        (
            output.float().sum(),
            output.float().square().sum(),
            output.float().abs().sum(),
        )
    )
    gathered = [torch.empty_like(checksum) for _ in range(8)]
    dist.all_gather(gathered, checksum)
    for peer in gathered[1:]:
        torch.testing.assert_close(peer, gathered[0], atol=1e-2, rtol=1e-5)


def _slot_table(batch, slots, shift, device):
    import torch

    rows = batch * 8
    return (
        torch.arange(rows, dtype=torch.int32, device=device) * 7 + shift
    ).remainder(slots).view(batch, 8)


def _exercise_batch(op, batch, device, weights):
    import torch

    from atom.model_ops.monokernel.config import KIMI_K3_CONFIG

    rows = batch * 8
    slots = {1: 13, 4: 37, 8: 67}[batch]
    snapshots = _slot_table(batch, slots, 1, device)
    conv_initial = torch.linspace(
        -0.125,
        0.125,
        slots * 10 * 3 * KIMI_K3_CONFIG.local_heads * KIMI_K3_CONFIG.v_dim,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16).view(
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
        recurrent_initial[slot].copy_(
            torch.linspace(
                0.75 + slot / 1024,
                1.25 + slot / 1024,
                recurrent_initial[slot].numel(),
                dtype=torch.float32,
                device=device,
            ).view_as(recurrent_initial[slot])
        )

    conv = conv_initial.clone()
    recurrent = recurrent_initial.clone()
    prefix = torch.zeros(
        rows,
        KIMI_K3_CONFIG.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    prefix[:, :16] = torch.linspace(
        -0.75,
        0.75,
        rows * 16,
        dtype=torch.float32,
        device=device,
    ).view(rows, 16).to(torch.bfloat16)
    blocks = torch.zeros(
        rows,
        1,
        KIMI_K3_CONFIG.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.full_like(prefix, float("nan"))
    accepted = torch.empty(batch, dtype=torch.int32, device=device)
    pointers = (
        conv.data_ptr(),
        recurrent.data_ptr(),
        snapshots.data_ptr(),
        accepted.data_ptr(),
    )

    eager_counts = (
        (4,)
        if batch == 1
        else (
            (1, 3, 6, 8)
            if batch == 4
            else (1, 3, 6, 8, 2, 7, 4, 5)
        ),
        (7,)
        if batch == 1
        else (
            (8, 2, 5, 1)
            if batch == 4
            else (8, 2, 5, 1, 7, 4, 6, 3)
        ),
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
        projected = _projected_input(op.pre_attn, weights.t)
        (
            expected_conv,
            expected_recurrent,
            expected_fp32,
            pairwise_fp32,
        ) = _apply_reference(
            expected_conv,
            expected_recurrent,
            snapshots,
            accepted,
            projected,
            weights.t,
        )
        _check_state(conv, recurrent, expected_conv, expected_recurrent)
        _check_fp32_handoff(op, batch, expected_fp32, pairwise_fp32)
        assert torch.isfinite(output).all()
        assert output.abs().max() > 0
        assert not torch.equal(output, prefix)
        _check_moe_and_tp_output(op, rows, output)

    conv.copy_(conv_initial)
    recurrent.copy_(recurrent_initial)
    expected_conv.copy_(conv_initial)
    expected_recurrent.copy_(recurrent_initial)
    graph_counts = (
        (2,)
        if batch == 1
        else (
            (2, 7, 1, 5)
            if batch == 4
            else (2, 7, 1, 5, 8, 3, 6, 4)
        ),
        (8,)
        if batch == 1
        else (
            (8, 1, 4, 6)
            if batch == 4
            else (8, 1, 4, 6, 3, 5, 2, 7)
        ),
        (3,)
        if batch == 1
        else (
            (3, 3, 8, 2)
            if batch == 4
            else (3, 3, 8, 2, 6, 1, 7, 5)
        ),
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
    for replay, counts in enumerate(graph_counts):
        snapshots.copy_(
            _slot_table(batch, slots, 3 + replay * 5, device)
        )
        accepted.copy_(torch.tensor(counts, dtype=torch.int32, device=device))
        eager_conv = conv.clone()
        eager_recurrent = recurrent.clone()
        eager_output = torch.full_like(output, float("nan"))
        op.forward(
            prefix,
            blocks,
            snapshots,
            eager_conv,
            eager_recurrent,
            x_out=eager_output,
            num_accepted_tokens=accepted,
        )
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        projected = _projected_input(op.pre_attn, weights.t)
        (
            expected_conv,
            expected_recurrent,
            expected_fp32,
            pairwise_fp32,
        ) = _apply_reference(
            expected_conv,
            expected_recurrent,
            snapshots,
            accepted,
            projected,
            weights.t,
        )
        _check_state(conv, recurrent, expected_conv, expected_recurrent)
        _check_fp32_handoff(op, batch, expected_fp32, pairwise_fp32)
        torch.testing.assert_close(output, eager_output, atol=2e-2, rtol=2e-2)
        assert torch.isfinite(output).all() and output.abs().max() > 0
        _check_moe_and_tp_output(op, rows, output)
        assert pointers == (
            conv.data_ptr(),
            recurrent.data_ptr(),
            snapshots.data_ptr(),
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
        weights = _deterministic_weights(device, rank)
        packed = None
        for batch in (1, 4, 8):
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
            _exercise_batch(op, batch, device, weights)
            op.close()
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _exercise_mla_batch(op, batch, device) -> None:
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.abi import AgenticDecodeShape
    from atom.model_ops.monokernel.k3.abi import (
        KimiAgenticShape,
        KimiMlaAgenticRuntime,
    )

    rows = batch * 8
    hidden = op.config.hidden
    torch.manual_seed(917)
    prefix = torch.randn(
        rows,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    blocks = torch.zeros(
        rows,
        1,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.arange(8, dtype=torch.int64, device=device).repeat(
        batch
    )
    batch_ids = torch.arange(
        batch,
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(8)
    physical_blocks = torch.arange(
        batch - 1,
        -1,
        -1,
        dtype=torch.int32,
        device=device,
    )
    block_tables = physical_blocks.view(batch, 1).contiguous()
    slot_mapping = (
        physical_blocks.to(torch.int64).repeat_interleave(8) * 16
        + positions
    ).contiguous()
    context_lens = torch.full(
        (batch,),
        8,
        dtype=torch.int32,
        device=device,
    )
    shape = KimiAgenticShape(
        common=AgenticDecodeShape(
            running_bs=batch,
            query_len=8,
            batch_capacity=batch,
            row_capacity=rows,
        ),
        dcp_size=1,
        replay_ssm=False,
    )
    runtime = KimiMlaAgenticRuntime.bind(
        shape,
        positions,
        slot_mapping,
        batch_ids,
        context_lens,
        block_tables,
    )
    cache = torch.zeros(
        batch * 16,
        1,
        576,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    scale = torch.tensor([0.015625], dtype=torch.float32, device=device)
    rope_cos = torch.ones(rows, 1, 1, 32, device=device)
    rope_sin = torch.zeros_like(rope_cos)
    output = torch.empty_like(prefix)

    op.forward(
        prefix,
        blocks,
        runtime,
        cache,
        scale,
        rope_cos,
        rope_sin,
        x_out=output,
    )
    torch.cuda.synchronize(device)
    assert torch.isfinite(output).all() and output.abs().max() > 0
    written = cache.view(-1, 576)[slot_mapping]
    assert (written.view(torch.uint8) != 0).any()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.forward(
            prefix,
            blocks,
            runtime,
            cache,
            scale,
            rope_cos,
            rope_sin,
            x_out=output,
        )

    for replay in range(3):
        eager_output = torch.empty_like(output)
        op.forward(
            prefix,
            blocks,
            runtime,
            cache,
            scale,
            rope_cos,
            rope_sin,
            x_out=eager_output,
        )
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            output,
            eager_output,
            atol=2e-2,
            rtol=2e-2,
        )
        assert torch.isfinite(output).all() and output.abs().max() > 0
        gathered = [
            torch.empty_like(output) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered, output)
        for peer in gathered[1:]:
            torch.testing.assert_close(
                output,
                peer,
                atol=2e-2,
                rtol=2e-2,
            )
        if rows > 8 and int(batch_ids[-1]) < 0:
            assert torch.count_nonzero(output[-8:]) == 0

        if replay == 1 and rows > 8:
            batch_ids[-8:].fill_(-1)
            slot_mapping[-8:].fill_(-1)
        elif replay == 2 and rows > 8:
            batch_ids[-8:].fill_(batch - 1)
            slot_mapping[-8:].copy_(
                physical_blocks[-1].to(torch.int64) * 16 + positions[-8:]
            )


def _tp8_mla_worker(rank: int, port: int) -> None:
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.k3.mla_full import KimiK3MlaMonoKernel

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
    )
    try:
        weights = _deterministic_mla_weights(device, rank)
        packed = None
        for batch in (1, 2, 4):
            op = KimiK3MlaMonoKernel(
                weights,
                batch * 8,
                layer_idx=0,
                rank=rank,
                npes=8,
                group=None,
                reduce_group=dist.group.WORLD,
                defer_collectives=False,
                packed_artifacts=packed,
            )
            packed = op.packed_artifacts()
            _exercise_mla_batch(op, batch, device)
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


def test_kimi_mla_tp8_q8_eager_and_graph_real_rocm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip("Kimi MLA device gate requires eight ROCm devices")

    import torch.multiprocessing as mp
    from flydsl.runtime.device import get_rocm_arch

    if str(get_rocm_arch() or "") != "gfx950":
        pytest.skip(f"Kimi MLA MonoKernel requires gfx950, got {get_rocm_arch()}")
    mp.spawn(_tp8_mla_worker, args=(_free_port(),), nprocs=8, join=True)
