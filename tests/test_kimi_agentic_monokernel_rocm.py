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
    tensors["w_qkv_a"] = bf16(
        config.q_lora + config.kv_lora + config.pe_dim,
        hidden,
    )
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

    qkv_rows = torch.arange(
        config.q_lora + config.kv_lora + config.pe_dim,
        device=device,
    )
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


def _deterministic_dense_weights(device, rank):
    import torch

    from atom.model_ops.monokernel.config import KIMI_K3_CONFIG

    weights = _deterministic_weights(device, rank)
    for name in (
        "w_r",
        "bias",
        "w_latent_down",
        "g_latent",
        "w_latent_up",
        "w_shared_ug",
        "w_shared_dn",
        "w_ug",
        "s_ug",
        "w_dn",
        "s_dn",
    ):
        weights.t.pop(name)
    hidden = KIMI_K3_CONFIG.hidden
    intermediate = 33792 // 8
    gate_up = torch.zeros(
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    gate_up_rows = torch.arange(2 * intermediate, device=device)
    gate_up[
        gate_up_rows,
        gate_up_rows.remainder(hidden),
    ] = 0.03125
    down = torch.zeros(
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device=device,
    )
    down_rows = torch.arange(hidden, device=device)
    down[
        down_rows,
        down_rows.remainder(intermediate),
    ] = (rank + 1) / 1024
    weights.t["w_dense_ug"] = gate_up
    weights.t["w_dense_dn"] = down
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
    from types import SimpleNamespace

    if hasattr(op, "monokernel_scratch"):
        from atom.model_ops.monokernel.k3.mla_full_kernel import mla_full_layout

        layout = mla_full_layout(rows)
        scratch = op.monokernel_scratch
    else:
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


def _check_mla_attention_reference(
    op,
    *,
    cache,
    scale,
    positions,
    batch_ids,
    context_lens,
    block_tables,
    slot_mapping,
):
    import torch

    from atom.model_ops.monokernel.k3.mla_cache import (
        dense_fp8_paged_mla_reference,
    )
    from atom.model_ops.monokernel.k3.mla_full_kernel import mla_full_layout

    rows = positions.numel()
    heads = op.config.local_heads
    scratch = op.monokernel_scratch
    layout = mla_full_layout(rows)

    def fp32(name, count):
        start = layout[name]
        return scratch[start : start + count * 4].view(torch.float32)

    q = fp32("mla_q", rows * heads * 192).view(rows, heads, 192)
    q_latent = fp32("mla_qlat", rows * heads * 512).view(rows, heads, 512)
    query = torch.cat((q_latent, q[..., 128:]), dim=-1).to(torch.bfloat16)
    fresh = fp32("mla_fresh", rows * 576).view(rows, 576).to(torch.bfloat16)
    expected, _ = dense_fp8_paged_mla_reference(
        query=query,
        main_cache=cache,
        main_scale=scale,
        positions=positions,
        batch_ids=batch_ids,
        context_lens=context_lens,
        block_tables=block_tables,
        block_size=128,
        block_ratio=128,
        fresh_slots=slot_mapping,
        fresh_values=fresh,
        softmax_scale=192**-0.5,
    )
    actual = fp32("mla_dense_acc", rows * heads * 512).view(rows, heads, 512)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def _full_mla_staged_oracle(
    op,
    weights,
    *,
    prefix,
    blocks,
    cache,
    scale,
    positions,
    batch_ids,
    context_lens,
    block_tables,
    slot_mapping,
    rank,
    reduce_group,
):
    """Independent PyTorch MLA frontend plus staged router/MoE/TP tail."""

    import torch
    import torch.distributed as dist
    from types import SimpleNamespace

    from atom.model_ops.monokernel.k3.mla_cache import (
        dense_fp8_paged_mla_reference,
    )
    from atom.model_ops.monokernel.k3.staged import _KimiK3MlaPath

    class ReferenceTail(_KimiK3MlaPath):
        def __init__(self, arena, *args, **kwargs):
            self._arena = arena
            super().__init__(*args, **kwargs)

        def _build_attention(self, *_args, **_kwargs):
            return self._arena

    t = weights.t
    ref_blocks = blocks.clone()

    def bf16_mm(value, weight):
        return (value.float() @ weight.float().t()).to(torch.bfloat16)

    def rms(value, gain):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + 1.0e-6
        )
        return (normalized * gain.float()).to(torch.bfloat16)

    def attn_res(prefix_value, delta, norm_gain, mix_weight, output_gain, count):
        updated = (
            prefix_value
            if delta is None
            else (prefix_value.float() + delta.float()).to(torch.bfloat16)
        )
        sources = torch.cat((ref_blocks[:, :count], updated[:, None]), dim=1)
        normalized = sources.float() * torch.rsqrt(
            sources.float().square().mean(-1, keepdim=True) + 1.0e-6
        )
        logits = (
            normalized * norm_gain.float() * mix_weight.float()
        ).sum(-1)
        mixed = (
            torch.softmax(logits, dim=-1)[..., None] * sources.float()
        ).sum(1).to(torch.bfloat16)
        return rms(mixed, output_gain), updated

    pre_attn, _ = attn_res(
        prefix,
        None,
        t["g_self_res"],
        t["w_self_res"],
        t["g_in"],
        op.previous_valid_blocks,
    )

    qkv = bf16_mm(pre_attn, t["w_qkv_a"])[
        :, : op.config.q_lora + op.config.kv_lora + op.config.pe_dim
    ]
    q_raw = qkv[:, : op.config.q_lora]
    kv_raw = qkv[:, op.config.q_lora :]

    q_norm = rms(q_raw, t["g_q"])
    kv_norm = rms(kv_raw[:, : op.config.kv_lora], t["g_kv"])
    key_pe = kv_raw[:, op.config.kv_lora :]
    q = bf16_mm(q_norm, t["w_q_b"]).view(
        prefix.shape[0],
        op.config.local_heads,
        op.config.nope_dim + op.config.pe_dim,
    )
    # The gate uses identity RoPE (cos=1, sin=0), so key_pe and q[...,128:]
    # are already the rotated values.
    fresh = (
        torch.cat((kv_norm, key_pe), dim=-1)
        .to(torch.bfloat16)
        .reshape(prefix.shape[0], 576)
        .contiguous()
    )
    uk = t["w_uk"].view(
        op.config.local_heads,
        op.config.kv_lora,
        op.config.nope_dim,
    )
    q_latent = torch.einsum(
        "rhd,hkd->rhk",
        q[..., : op.config.nope_dim].float(),
        uk.float(),
    ).to(torch.bfloat16)
    query = torch.cat((q_latent, q[..., op.config.nope_dim :]), dim=-1)
    dense, _ = dense_fp8_paged_mla_reference(
        query=query.contiguous(),
        main_cache=cache,
        main_scale=scale,
        positions=positions,
        batch_ids=batch_ids,
        context_lens=context_lens,
        block_tables=block_tables,
        block_size=128,
        block_ratio=128,
        fresh_slots=slot_mapping.reshape(-1).to(torch.int64).contiguous(),
        fresh_values=fresh,
        softmax_scale=192**-0.5,
    )
    uv = t["w_uv"].view(
        op.config.local_heads,
        op.config.v_dim,
        op.config.kv_lora,
    )
    attended = torch.einsum(
        "rhk,hvk->rhv",
        dense,
        uv.float(),
    )
    gate = bf16_mm(pre_attn, t["w_gate"]).view_as(attended)
    attended.mul_(torch.sigmoid(gate.float()))
    local_attention = bf16_mm(
        attended.to(torch.bfloat16).flatten(1),
        t["w_o"],
    )
    peer_attention = [
        torch.empty_like(local_attention) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(peer_attention, local_attention, group=reduce_group)
    attention_delta = torch.zeros_like(local_attention, dtype=torch.float32)
    for peer in peer_attention:
        attention_delta.add_(peer.float())
    attention_delta = attention_delta.to(torch.bfloat16)

    post_prefix = attention_delta if op.is_block_write_layer else prefix
    post_delta = None if op.is_block_write_layer else attention_delta
    moe_input, residual = attn_res(
        post_prefix,
        post_delta,
        t["g_mlp_res"],
        t["w_mlp_res"],
        t["g_post"],
        op.previous_valid_blocks + int(op.is_block_write_layer),
    )
    output = torch.empty_like(prefix)
    for start in range(0, prefix.shape[0], 8):
        arena = SimpleNamespace(
            step=torch.zeros(1, dtype=torch.int32, device=prefix.device),
            packed_artifacts=lambda: {},
            close=lambda: None,
        )
        reference = ReferenceTail(
            arena,
            weights,
            8,
            layer_idx=op.layer_idx,
            rank=rank,
            npes=8,
            group=None,
            reduce_group=reduce_group,
            fuse_attn_res=False,
            fuse_router=True,
            fuse_shared_experts=True,
            packed_artifacts=op.packed_artifacts(),
        )
        reference.t = dict(reference.t)
        reference.t["bias"] = reference.t["bias"].to(torch.bfloat16)
        chunk = slice(start, start + 8)
        reference.latent_projection.quantize_input(moe_input[chunk])
        reference._moe(
            moe_input[chunk],
            op.layer_idx,
            residual[chunk],
            output[chunk],
        )
        reference.symmetric_allreduce.close()
    output[batch_ids < 0] = 0
    return output


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
    slots = {1: 13, 2: 23, 4: 37}[batch]
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
        else (1, 6)
        if batch == 2
        else (1, 3, 6, 8),
        (7,)
        if batch == 1
        else (8, 2)
        if batch == 2
        else (8, 2, 5, 1),
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
        else (2, 7)
        if batch == 2
        else (2, 7, 1, 5),
        (8,)
        if batch == 1
        else (8, 1)
        if batch == 2
        else (8, 1, 4, 6),
        (3,)
        if batch == 1
        else (3, 3)
        if batch == 2
        else (3, 3, 8, 2),
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


def _dense_layer0_oracle(op, weights, reduce_group):
    import torch
    import torch.distributed as dist

    gate_up = (
        op.moe_input.float() @ weights.t["w_dense_ug"].float().t()
    ).to(torch.bfloat16)
    gate, up = gate_up.float().chunk(2, dim=-1)
    middle = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    ).to(torch.bfloat16)
    local = (
        middle.float() @ weights.t["w_dense_dn"].float().t()
    ).to(torch.bfloat16)
    peers = [
        torch.empty_like(local) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(peers, local, group=reduce_group)
    reduced = torch.zeros_like(local, dtype=torch.float32)
    for peer in peers:
        reduced.add_(peer.float())
    return (op.updated_prefix.float() + reduced).to(torch.bfloat16)


def _exercise_dense_layer0(op, batch, device, weights, reduce_group):
    import torch

    rows = batch * 8
    slots = {1: 13, 2: 23, 4: 37}[batch]
    snapshots = _slot_table(batch, slots, 1, device)
    accepted = torch.full((batch,), 8, dtype=torch.int32, device=device)
    conv = torch.zeros(
        slots,
        10,
        3 * op.config.local_heads * op.config.v_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    recurrent = torch.zeros(
        slots,
        op.config.local_heads,
        op.config.v_dim,
        op.config.v_dim,
        dtype=torch.float16,
        device=device,
    )
    prefix = torch.zeros(
        rows,
        op.config.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    prefix[:, :32] = torch.linspace(
        -0.5,
        0.5,
        rows * 32,
        device=device,
    ).view(rows, 32).to(torch.bfloat16)
    blocks = torch.empty(
        rows,
        1,
        op.config.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.empty_like(prefix)

    before = int(op.step.item())
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
    expected = _dense_layer0_oracle(op, weights, reduce_group)
    torch.testing.assert_close(output, expected, atol=5e-2, rtol=5e-2)
    assert int(op.step.item()) == before + 1

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
    prefix[:, :32].mul_(0.75)
    before = int(op.step.item())
    graph.replay()
    torch.cuda.synchronize(device)
    expected = _dense_layer0_oracle(op, weights, reduce_group)
    torch.testing.assert_close(output, expected, atol=5e-2, rtol=5e-2)
    assert int(op.step.item()) == before + 1


def _tp8_worker(rank: int, port: int) -> None:
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.k3 import kda as kda_module
    from atom.model_ops.monokernel.k3 import staged as staged_module
    from atom.model_ops.monokernel.config import ConvStateLayout
    from atom.model_ops.monokernel.k3.op import KimiK3MonoKernel
    from atom.models.kimi_k3_mono import (
        _full_layer_source_nbytes,
        _full_layer_workspace_nbytes,
    )

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
    )
    try:
        pack_calls = {"quantize": 0, "expert": 0, "attention": 0}
        staged_quantize = staged_module.quantize_mxfp8
        staged_expert = staged_module.prepare_mxfp4_expert_storage
        attention_quantize = kda_module.quantize_mxfp8

        def count_quantize(*args, **kwargs):
            pack_calls["quantize"] += 1
            return staged_quantize(*args, **kwargs)

        def count_expert(*args, **kwargs):
            pack_calls["expert"] += 1
            return staged_expert(*args, **kwargs)

        def count_attention(*args, **kwargs):
            pack_calls["attention"] += 1
            return attention_quantize(*args, **kwargs)

        staged_module.quantize_mxfp8 = count_quantize
        staged_module.prepare_mxfp4_expert_storage = count_expert
        kda_module.quantize_mxfp8 = count_attention
        weights = _deterministic_weights(device, rank)
        packed = None
        canonical_moe = None
        for batch in (1, 2, 4):
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
            assert sum(
                tensor.numel() * tensor.element_size()
                for tensor in op.full_plan_workspace_tensors()
            ) == _full_layer_workspace_nbytes("mono", batch * 8)
            packed = op.packed_artifacts()
            if canonical_moe is None:
                canonical_moe = packed["moe_packed"]
            assert packed["moe_packed"] is canonical_moe
            assert op.moe_packed is canonical_moe
            assert op.attention.moe_packed is canonical_moe
            assert all(
                op.attention.moe_packed[name] is tensor
                for name, tensor in canonical_moe.items()
            )
            op.release_packed_sources()
            retained = {
                tensor.data_ptr(): tensor
                for mapping in (op.t, op.attention.t)
                for tensor in mapping.values()
            }
            retained_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in retained.values()
            )
            expected_source_bytes = _full_layer_source_nbytes("mono")
            assert retained_bytes == expected_source_bytes, (
                retained_bytes,
                expected_source_bytes,
                {
                    name: tensor.numel() * tensor.element_size()
                    for mapping in (op.t, op.attention.t)
                    for name, tensor in mapping.items()
                },
            )
            assert op.t["g_in"] is weights.t["g_in"]
            _exercise_batch(op, batch, device, weights)
            op.close()
            assert all(tensor.numel() for tensor in canonical_moe.values())
            dist.barrier()
        assert pack_calls == {
            "quantize": 4,
            "expert": 1,
            "attention": 0,
        }
    finally:
        dist.destroy_process_group()


def _tp8_dense_worker(rank: int, port: int) -> None:
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.k3.dense_full import (
        KimiK3DenseMonoKernel,
    )
    from atom.models.kimi_k3_mono import (
        _full_layer_source_nbytes,
        _full_layer_workspace_nbytes,
    )

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
    )
    try:
        weights = _deterministic_dense_weights(device, rank)
        packed = None
        for batch in (1, 2, 4):
            op = KimiK3DenseMonoKernel(
                weights,
                batch * 8,
                layer_idx=0,
                rank=rank,
                npes=8,
                group=None,
                reduce_group=dist.group.WORLD,
                state_dtype=torch.float16,
                agentic_batch_size=batch,
                packed_artifacts=packed,
            )
            assert sum(
                tensor.numel() * tensor.element_size()
                for tensor in op.full_plan_workspace_tensors()
            ) == _full_layer_workspace_nbytes(
                "dense_full",
                batch * 8,
            )
            packed = op.packed_artifacts()
            op.release_packed_sources()
            retained = {
                tensor.data_ptr(): tensor
                for mapping in (op.t, op.attention.t)
                for tensor in mapping.values()
            }
            assert sum(
                tensor.numel() * tensor.element_size()
                for tensor in retained.values()
            ) == _full_layer_source_nbytes("dense_full")
            assert op.t["g_in"] is weights.t["g_in"]
            _exercise_dense_layer0(
                op,
                batch,
                device,
                weights,
                dist.group.WORLD,
            )
            op.symmetric_allreduce.close()
            op.attention.close()
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _exercise_mla_batch(op, batch, device, weights) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace

    from atom.models.kimi_k3_mono import KimiMonoDecode
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    from atom.model_ops.monokernel.config import KIMI_K3_CONFIG
    from atom.model_ops.monokernel.telemetry import MonoRouteStats
    from atom.utils import CpuGpuBuffer
    from atom.utils.forward_context import get_forward_context

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
    # Kimi-K3 is NoPE. Production keeps only a one-row BF16 placeholder cache,
    # so this maximum checkpoint position proves the full kernel never indexes
    # that placeholder. Physical fresh rows still straddle block 128 below.
    positions = torch.full(
        (rows,),
        1_048_575,
        dtype=torch.int64,
        device=device,
    )
    fresh_positions = torch.arange(
        125, 133, dtype=torch.int64, device=device
    ).repeat(batch)
    metadata_builder = object.__new__(AiterMLAMetadataBuilder)
    metadata_builder.model_runner = SimpleNamespace(
        forward_vars={
            "batch_id_per_q_token": CpuGpuBuffer(
                rows,
                dtype=torch.int32,
                device=device,
            )
        }
    )
    batch_ids = metadata_builder.publish_batch_ids(
        np.full(batch, 8, dtype=np.int32),
        pad_to=rows,
    )
    physical_blocks = torch.arange(
        2 * batch - 1,
        -1,
        -1,
        dtype=torch.int32,
        device=device,
    )
    block_tables = physical_blocks.view(batch, 2).contiguous()
    logical_blocks = fresh_positions.view(batch, 8).div(
        128, rounding_mode="floor"
    )
    offsets = fresh_positions.view(batch, 8).remainder(128)
    slot_mapping = (
        block_tables.gather(1, logical_blocks.to(torch.int64)).to(torch.int64) * 128
        + offsets
    ).reshape(-1).contiguous()
    context_lens = torch.full(
        (batch,),
        133,
        dtype=torch.int32,
        device=device,
    )
    cache = torch.zeros(
        2 * batch * 128,
        1,
        576,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    cache_rows = cache.view(-1, 576)
    for request in range(batch):
        for logical_position in range(125):
            logical_block, offset = divmod(logical_position, 128)
            physical = int(block_tables[request, logical_block]) * 128 + offset
            values = (
                torch.arange(576, device=device, dtype=torch.float32)
                .remainder(31)
                .sub(15)
                .mul((request + 1) / 128)
            )
            cache_rows[physical].copy_(values.to(torch.float8_e4m3fn))
    scale = torch.tensor([0.015625], dtype=torch.float32, device=device)
    rope_cos = torch.full(
        (1, 32),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    rope_sin = torch.full_like(rope_cos, float("nan"))
    output = torch.empty_like(prefix)

    class PassThroughKda:
        block_write_idx = 0

        @staticmethod
        def forward(hidden_states, block_residual, *_args, **_kwargs):
            block_residual[:, 0].copy_(hidden_states)
            return hidden_states

    kda_layer = SimpleNamespace(
        layer_idx=0,
        is_linear_attn=True,
        block_sparse_moe=object(),
    )
    mla_layer = SimpleNamespace(
        layer_idx=1,
        is_linear_attn=False,
        block_sparse_moe=object(),
        self_attn=SimpleNamespace(
            attn=SimpleNamespace(impl=SimpleNamespace(_k_scale_device=scale)),
            rotary_emb=SimpleNamespace(cos_cache=rope_cos, sin_cache=rope_sin),
        ),
    )
    model = SimpleNamespace(
        layers=[kda_layer, mla_layer],
        start_layer=0,
        end_layer=2,
        get_input_embeddings=lambda _ids: pytest.fail("inputs_embeds must be used"),
        output_attn_res=lambda value, *_args: (value, None),
    )
    adapter = object.__new__(KimiMonoDecode)
    adapter._lm = SimpleNamespace(model=model)
    adapter._mode = "mono"
    adapter._enabled = True
    adapter._atom_config = SimpleNamespace(
        tensor_parallel_size=8,
        kv_cache_dtype="fp8",
        decode_context_parallel_size=1,
        speculative_config=SimpleNamespace(method="dspark"),
    )
    adapter._stats = MonoRouteStats("kimi_k3_mla_gate")
    adapter._refused = set()
    adapter._weights = {}
    adapter._packed_artifacts = {}
    adapter._reductions = {}
    adapter._full_route_eligible = lambda **_kwargs: True
    key_kda = (0, rows, "mono", torch.float16, batch)
    key_mla = (1, rows, "mla_full", torch.float16, batch)
    adapter._ops = {
        key_kda: SimpleNamespace(op=PassThroughKda()),
        key_mla: SimpleNamespace(op=op),
    }
    snapshots = torch.arange(rows, dtype=torch.int32, device=device).view(batch, 8)
    accepted = torch.full((batch,), 8, dtype=torch.int32, device=device)
    kda_metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=batch,
        num_spec_decode_tokens=rows,
        num_actual_tokens=rows,
        replayssm=False,
        spec_state_indices_tensor=snapshots,
        num_accepted_tokens=accepted,
    )
    metadata = SimpleNamespace(
        kda_metadata=kda_metadata,
        slot_mapping=slot_mapping,
        batch_id_per_q_token=batch_ids,
        context_lens=context_lens,
        block_tables=block_tables,
        block_size=128,
        block_ratio=128,
    )
    conv = torch.zeros(
        rows,
        10,
        3 * KIMI_K3_CONFIG.local_heads * KIMI_K3_CONFIG.v_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    recurrent = torch.zeros(
        rows,
        KIMI_K3_CONFIG.local_heads,
        KIMI_K3_CONFIG.v_dim,
        KIMI_K3_CONFIG.v_dim,
        dtype=torch.float16,
        device=device,
    )
    fwd = get_forward_context()
    fwd.context = SimpleNamespace(
        is_prefill=False,
        running_bs=batch,
        running_tokens=rows,
        max_seqlen_q=8,
    )
    fwd.ubatch_slices = None
    fwd.attn_metadata = metadata
    fwd.kv_cache_data = {
        "layer_0": SimpleNamespace(k_cache=conv, v_cache=recurrent),
        "layer_1": SimpleNamespace(k_cache=cache),
    }
    input_ids = torch.arange(rows, dtype=torch.int64, device=device)
    assert adapter.supports(input_ids, positions, None, prefix)
    # All keys are already populated, but this still exercises the production
    # atomic prepare path used by supports().
    assert adapter._prepare(rows, torch.float16, batch)
    blocks[:, 0].copy_(prefix)
    expected_output = _full_mla_staged_oracle(
        op,
        weights,
        prefix=prefix,
        blocks=blocks,
        cache=cache,
        scale=scale,
        positions=positions,
        batch_ids=batch_ids,
        context_lens=context_lens,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        rank=dist.get_rank(),
        reduce_group=dist.group.WORLD,
    )
    output.copy_(adapter.forward(input_ids, positions, prefix))
    torch.cuda.synchronize(device)
    assert torch.isfinite(output).all() and output.abs().max() > 0
    torch.testing.assert_close(output, expected_output, atol=8e-2, rtol=8e-2)
    _check_mla_attention_reference(
        op,
        cache=cache,
        scale=scale,
        positions=positions,
        batch_ids=batch_ids,
        context_lens=context_lens,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
    )
    _check_moe_and_tp_output(op, rows, output)
    written = cache.view(-1, 576)[slot_mapping]
    assert (written.view(torch.uint8) != 0).any()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output.copy_(adapter.forward(input_ids, positions, prefix))

    padded_cache_before = None
    for replay in range(3):
        expected_eager = _full_mla_staged_oracle(
            op,
            weights,
            prefix=prefix,
            blocks=blocks,
            cache=cache,
            scale=scale,
            positions=positions,
            batch_ids=batch_ids,
            context_lens=context_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            rank=dist.get_rank(),
            reduce_group=dist.group.WORLD,
        )
        eager_output = adapter.forward(input_ids, positions, prefix).clone()
        torch.testing.assert_close(
            eager_output,
            expected_eager,
            atol=8e-2,
            rtol=8e-2,
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
            assert torch.equal(cache, padded_cache_before)

        if replay == 1 and rows > 8:
            batch_ids[-8:].fill_(-1)
            slot_mapping[-8:].fill_(-1)
            padded_cache_before = cache.clone()
        elif replay == 2 and rows > 8:
            batch_ids[-8:].fill_(batch - 1)
            slot_mapping[-8:].copy_(
                block_tables[-1].gather(
                    0, fresh_positions[-8:].div(128, rounding_mode="floor")
                ).to(torch.int64)
                * 128
                + fresh_positions[-8:].remainder(128)
            )


def _tp8_mla_worker(rank: int, port: int) -> None:
    import torch
    import torch.distributed as dist

    from atom.model_ops.monokernel.k3.mla_full import KimiK3MlaMonoKernel
    from atom.models.kimi_k3_mono import (
        _full_layer_source_nbytes,
        _full_layer_workspace_nbytes,
    )

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
            previous_w_o = (
                None if packed is None else packed["w_o_packed"]
            )
            op = KimiK3MlaMonoKernel(
                weights,
                batch * 8,
                layer_idx=1,
                rank=rank,
                npes=8,
                group=None,
                reduce_group=dist.group.WORLD,
                defer_collectives=False,
                packed_artifacts=packed,
            )
            if previous_w_o is not None:
                assert op.w_o_packed is previous_w_o
            assert sum(
                tensor.numel() * tensor.element_size()
                for tensor in op.full_plan_workspace_tensors()
            ) == _full_layer_workspace_nbytes(
                "mla_full",
                batch * 8,
            )
            packed = op.packed_artifacts()
            op.release_packed_sources()
            retained = {
                tensor.data_ptr(): tensor for tensor in op.t.values()
            }
            retained_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in retained.values()
            )
            expected_source_bytes = _full_layer_source_nbytes("mla_full")
            assert retained_bytes == expected_source_bytes, (
                retained_bytes,
                expected_source_bytes,
                {
                    name: tensor.numel() * tensor.element_size()
                    for name, tensor in op.t.items()
                },
            )
            assert op.t["g_in"] is weights.t["g_in"]
            _exercise_mla_batch(op, batch, device, weights)
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


def test_kimi_dense_layer0_tp8_q8_eager_graph_oracle_real_rocm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip("Kimi dense layer-0 gate requires eight ROCm devices")

    import torch.multiprocessing as mp
    from flydsl.runtime.device import get_rocm_arch

    if str(get_rocm_arch() or "") != "gfx950":
        pytest.skip(f"Kimi dense MonoKernel requires gfx950, got {get_rocm_arch()}")
    mp.spawn(_tp8_dense_worker, args=(_free_port(),), nprocs=8, join=True)


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
