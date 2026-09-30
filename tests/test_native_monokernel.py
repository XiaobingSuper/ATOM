# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.model_ops.monokernel.config import (
    ConvStateLayout,
    conv_state_offset,
    conv_state_shape,
)
from atom.model_ops.monokernel.dispatch import select_backend
from atom.model_ops.monokernel.glm import layout as glm_layout


def _kimi_mono_module():
    import importlib

    from tests.aiter_stub import stubbed_aiter

    with stubbed_aiter():
        return importlib.import_module("atom.models.kimi_k3_mono")


def _glm_mono_module():
    import importlib

    from tests.aiter_stub import stubbed_aiter

    with stubbed_aiter():
        return importlib.import_module("atom.models.glm52_mono")


def _preshuffle_linear_weight(weight):
    import torch

    raw = weight.view(torch.uint8)
    *lead, rows, packed_cols = raw.shape
    lane_k = 16 // raw.element_size()
    tiled = raw.reshape(
        *lead,
        rows // 16,
        16,
        packed_cols // 32,
        32 // lane_k,
        lane_k,
    )
    nlead = len(lead)
    order = list(range(nlead)) + [nlead, nlead + 2, nlead + 3, nlead + 1, nlead + 4]
    return tiled.permute(*order).contiguous().reshape_as(raw).view(weight.dtype)


def _preshuffled_mxfp4_linear(source):
    import torch

    from atom.model_ops.monokernel.formats import dequantize_mxfp4, quantize_mxfp4

    packed, scale = quantize_mxfp4(source)
    fp4_dtype = torch.float4_e2m1fn_x2
    shuffled_weight = _preshuffle_linear_weight(packed.view(fp4_dtype))
    shuffled_weight.is_shuffled = True
    rows = source.shape[0]
    groups = scale.shape[1]
    shuffled_scale = (
        scale.reshape(rows // 32, 2, 16, groups // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .reshape(rows, groups)
    )
    linear = SimpleNamespace(
        input_size=source.shape[1],
        output_size=source.shape[0],
        quant_type=SimpleNamespace(name="per_1x32"),
        params_dtype=fp4_dtype,
        weight=shuffled_weight,
        weight_scale=shuffled_scale,
    )
    expected = dequantize_mxfp4(packed, scale).to(torch.bfloat16)
    return linear, expected


def _preshuffled_per_token_fp8_linear(source):
    import torch

    fp8_dtype = torch.float8_e4m3fn
    scale = source.abs().amax(dim=1, keepdim=True).float() / 448.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    native = (source.float() / scale).clamp(-448, 448).to(fp8_dtype)
    shuffled = _preshuffle_linear_weight(native)
    shuffled.is_shuffled = True
    linear = SimpleNamespace(
        input_size=source.shape[1],
        output_size=source.shape[0],
        quant_type=SimpleNamespace(name="per_Token"),
        params_dtype=fp8_dtype,
        weight=shuffled,
        weight_scale=scale,
        is_output_padded=False,
    )
    return linear, (native.float() * scale).to(torch.bfloat16)


def _kimi_decode_context(samples, actual_tokens=None, state_indices=None):
    import torch

    actual_tokens = samples if actual_tokens is None else actual_tokens
    if state_indices is None:
        state_indices = torch.arange(samples, dtype=torch.int32)
        state_indices[actual_tokens:] = -1
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=actual_tokens,
        num_spec_decodes=0,
        num_actual_tokens=actual_tokens,
        replayssm=False,
        non_spec_state_indices_tensor=state_indices,
    )
    return SimpleNamespace(
        context=SimpleNamespace(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=SimpleNamespace(kda_metadata=metadata),
        kv_cache_data={},
    )


def test_public_package_import_is_lazy():
    code = """
import sys
import atom.model_ops.monokernel
assert 'atom.model_ops.monokernel.glm.kernel' not in sys.modules
assert 'atom.model_ops.monokernel.k3.kernel' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_peer_buffer_allocation_failure_is_collective(monkeypatch):
    import torch

    from atom.model_ops.monokernel import runtime

    gathers = []
    monkeypatch.setattr(runtime.torch.cuda, "current_device", lambda: 0)

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("rank 0 allocation failed")

    def all_gather_object(output, local, group):
        gathers.append((local, group))
        output[:] = [local, (None, (b"peer-handle", 0))]

    monkeypatch.setattr(runtime.torch, "zeros", fail_allocation)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda **_kwargs: pytest.fail("failed readiness must skip the final barrier"),
    )

    group = object()
    with pytest.raises(RuntimeError, match="allocation/export failed.*rank 0 allocation failed"):
        runtime.SymmetricPeerBuffer(256, rank=0, npes=2, group=group)

    assert len(gathers) == 1
    assert gathers[0][1] is group


def test_peer_buffer_remote_open_failure_is_collective(monkeypatch):
    import torch

    from atom.model_ops.monokernel import runtime

    class Storage:
        device = torch.device("cuda", 0)

        @staticmethod
        def data_ptr():
            return 0x1000

    gathers = []
    closed = []
    open_calls = []
    monkeypatch.setattr(runtime.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(runtime.torch, "zeros", lambda *_args, **_kwargs: Storage())
    monkeypatch.setattr(runtime, "get_allocation_base", lambda _address: 0x1000)
    monkeypatch.setattr(runtime, "get_ipc_handle", lambda _base: b"local-handle")

    def open_ipc_handle(handle):
        open_calls.append(handle)
        if len(open_calls) == 2:
            raise RuntimeError("rank 0 open failed")
        return 0x2000

    monkeypatch.setattr(runtime, "open_ipc_handle", open_ipc_handle)
    monkeypatch.setattr(runtime, "close_ipc_handle", closed.append)

    def all_gather_object(output, local, group):
        gathers.append((local, group))
        if len(gathers) == 1:
            output[:] = [
                local,
                (None, (b"peer-1-handle", 16)),
                (None, (b"peer-2-handle", 32)),
            ]
        else:
            output[:] = [local, None, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda **_kwargs: pytest.fail("failed open must skip the final barrier"),
    )
    monkeypatch.setattr(
        runtime.torch,
        "tensor",
        lambda *_args, **_kwargs: pytest.fail("failed open must skip address publication"),
    )

    group = object()
    with pytest.raises(RuntimeError, match="open failed.*rank 0 open failed"):
        runtime.SymmetricPeerBuffer(256, rank=0, npes=3, group=group)

    assert len(gathers) == 2
    assert all(call_group is group for _, call_group in gathers)
    assert closed == [0x2000]


def test_tp_validation_consensus_propagates_peer_failure(monkeypatch):
    import torch

    from atom.model_ops.monokernel.dispatch import (
        MonoUnsupported,
        tp_uniform_local_validation,
    )

    def all_gather_object(output, local, group=None):
        assert local is None
        assert group is tp_group
        output[:] = [None, "ValueError: rank-local layout"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    tp_group = object()
    with pytest.raises(MonoUnsupported, match="rank 1: ValueError: rank-local layout"):
        tp_uniform_local_validation(
            None,
            group=tp_group,
            world_size=2,
            context="weight mapping failed",
        )


def test_bundled_aiter_compatibility_imports():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available():
        pytest.skip("AITER import requires a visible ROCm device")
    pytest.importorskip("aiter")

    import atom.model_ops.monokernel.k3.staged  # noqa: F401


def test_kimi_attn_res_compile_cache_preserves_delta_specialization():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available():
        pytest.skip("Kimi-K3 AttnRes requires a visible ROCm device")

    from atom.model_ops.monokernel.k3.attn_res import KimiK3AttnRes

    samples, hidden = 1, 7168
    prefix = torch.randn(samples, hidden, dtype=torch.bfloat16, device="cuda")
    delta = torch.randn_like(prefix)
    blocks = torch.randn(samples, 1, hidden, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")
    updated = torch.empty_like(prefix)
    output = torch.empty_like(prefix)

    KimiK3AttnRes(samples, hidden, 1, False, -1)(
        prefix,
        delta,
        blocks,
        weight,
        weight,
        weight,
        updated,
        output,
    )
    KimiK3AttnRes(samples, hidden, 1, True, -1)(
        prefix,
        delta,
        blocks,
        weight,
        weight,
        weight,
        updated,
        output,
    )
    torch.cuda.synchronize()

    expected = (prefix.float() + delta.float()).to(torch.bfloat16)
    assert torch.equal(updated, expected)


def test_kimi_monokernel_compile_cache_keys_layer_geometry():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available():
        pytest.skip("Kimi-K3 MonoKernel requires a visible ROCm device")

    from atom.model_ops.monokernel.k3.kernel import build_kimi_k3_monokernel

    launches = (
        build_kimi_k3_monokernel(4, attn_res_blocks=1, fuse_moe=True),
        build_kimi_k3_monokernel(4, attn_res_blocks=2, fuse_moe=True),
        build_kimi_k3_monokernel(8, attn_res_blocks=1, fuse_moe=True),
        build_kimi_k3_monokernel(
            4,
            attn_res_blocks=1,
            fuse_moe=True,
            atom_expert_layout=True,
        ),
    )

    assert len({launch.func.__name__ for launch in launches}) == len(launches)


def test_scaled_mfma_uses_flydsl_v0341_operand_abi():
    root = Path(__file__).parents[1] / "atom" / "model_ops" / "monokernel"
    paths = (
        root / "mxfp8_linear.py",
        root / "k3" / "kernel.py",
        root / "k3" / "router_projection.py",
    )
    scaled_calls = []
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "gemm":
                continue
            assert not isinstance(node.args[2], (ast.List, ast.Tuple))
            assert not isinstance(node.args[3], (ast.List, ast.Tuple))
            keywords = {keyword.arg for keyword in node.keywords}
            if {"scale_a", "scale_b"} <= keywords:
                scaled_calls.append(node)
    assert scaled_calls


def test_native_decode_flag(monkeypatch):
    from atom.utils import envs

    name = "ATOM_NATIVE_DECODE_MONOKERNEL"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) == "off"
    for value in ("off", "auto", "mono", "staged"):
        monkeypatch.setenv(name, value.upper())
        assert getattr(envs, name) == value
    monkeypatch.setenv(name, "on")
    with pytest.raises(ValueError, match="off, auto, mono, staged"):
        getattr(envs, name)


@pytest.mark.parametrize(
    "override",
    [
        {"mode": "off"},
        {"samples": 1},
        {"tp_size": 4},
        {"kv_cache_dtype": "auto"},
        {"native": False},
        {"decode": False},
        {"mtp": True},
        {"dpa": True},
        {"dcp": True},
        {"plugin": True},
        {"kv_cache_dtype": "fp8"},
        {"external_indexer": False},
        {"cache_layout": "split"},
    ],
)
def test_unsupported_glm_forward_falls_back(override):
    args = dict(
        model="glm52",
        mode="auto",
        samples=4,
        tp_size=8,
        kv_cache_dtype="bf16",
    )
    args.update(override)
    assert select_backend(**args) is None


def test_kimi_conv_state_layout_contracts():
    channels = 10
    channel = 7
    assert [
        conv_state_offset(ConvStateLayout.CHANNEL_MAJOR, channel, time, channels)
        for time in range(3)
    ] == [21, 22, 23]
    assert [
        conv_state_offset(ConvStateLayout.TIME_MAJOR, channel, time, channels)
        for time in range(3)
    ] == [7, 17, 27]
    assert conv_state_shape(ConvStateLayout.CHANNEL_MAJOR, 5, channels) == (5, 10, 3)
    assert conv_state_shape(ConvStateLayout.TIME_MAJOR, 5, channels) == (5, 3, 10)


@pytest.mark.parametrize("mode", ("staged", "mono"))
def test_kimi_adapter_constructs_time_major_ops(monkeypatch, mode):
    module = _kimi_mono_module()
    captured = []

    class FakeOp:
        def __init__(self, *_args, **kwargs):
            captured.append(kwargs)

    op_module = types.ModuleType("atom.model_ops.monokernel.k3.op")
    op_module.KimiK3MonoKernel = FakeOp
    staged_module = types.ModuleType("atom.model_ops.monokernel.k3.staged")
    staged_module._KimiK3KdaStagedPath = FakeOp
    monkeypatch.setitem(sys.modules, op_module.__name__, op_module)
    monkeypatch.setitem(sys.modules, staged_module.__name__, staged_module)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(
        module,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object(), device_group=object()),
    )
    attention_reduce = object()
    moe_reduce = object()
    module._KimiLayerOp(
        SimpleNamespace(layer_idx=1),
        weights=object(),
        samples=8,
        mode=mode,
        attention_symmetric_allreduce=attention_reduce,
        moe_symmetric_allreduce=moe_reduce,
    )

    assert captured[0]["conv_state_layout"] is ConvStateLayout.TIME_MAJOR
    assert "launches_per_step" not in captured[0]
    assert captured[0]["attention_symmetric_allreduce"] is attention_reduce
    assert captured[0]["moe_symmetric_allreduce"] is moe_reduce


def test_kimi_mono_accepts_shared_reduction_dependencies():
    source = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "monokernel"
        / "k3"
        / "op.py"
    ).read_text()
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KimiK3MonoKernel"
    )
    init = next(
        node
        for node in kernel.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    dependencies = {
        "attention_symmetric_allreduce",
        "moe_symmetric_allreduce",
    }
    parameters = {argument.arg for argument in init.args.args + init.args.kwonlyargs}
    forwarded = {
        keyword.arg
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    }

    assert dependencies <= parameters
    assert dependencies <= forwarded


def test_kimi_prepare_defers_collectives_until_rank_consensus():
    source = (
        Path(__file__).parents[1] / "atom" / "models" / "kimi_k3_mono.py"
    ).read_text()
    prepare = source[source.index("    def _prepare(") : source.index("    def _op(")]

    assert "defer_collectives=True" in prepare
    assert prepare.index("rank-local construction failed") < prepare.index(
        "initialize_collectives"
    )


def test_model_specific_backend_selection():
    common = dict(samples=8, tp_size=8, kv_cache_dtype="fp8")
    assert select_backend("glm52", "auto", **common) is None
    assert select_backend("glm52", "staged", **common) is None
    glm_bf16 = dict(samples=8, tp_size=8, kv_cache_dtype="bf16")
    assert select_backend("glm52", "auto", **glm_bf16) == "mono"
    assert select_backend("glm52", "mono", **glm_bf16) == "mono"
    assert select_backend("glm52", "staged", **glm_bf16) is None
    assert select_backend("kimi_k3", "auto", **common) == "staged"
    assert select_backend("kimi_k3", "mono", **common) == "mono"
    assert select_backend("kimi_k3", "auto", **common, dcp=True) == "staged"
    assert select_backend("kimi_k3", "auto", **common, is_kda=False) is None
    assert select_backend("kimi_k3", "auto", **common, has_moe=False) is None


@pytest.mark.parametrize(("rows", "mtp", "dcp"), ((48, True, False), (80, True, True)))
def test_glm_agentic_selects_moe_stage_for_flattened_rows(rows, mtp, dcp):
    common = dict(
        samples=rows,
        tp_size=4,
        kv_cache_dtype="fp8",
        mtp=mtp,
        dcp=dcp,
        segment="moe",
    )

    assert select_backend("glm52", "auto", **common) is None
    assert select_backend("glm52", "staged", **common) == "staged_moe"
    assert select_backend("glm52", "mono", **common) is None
    assert select_backend("glm52", "auto", **common, plugin=True) is None


def test_glm_tp4_staged_moe_workspace_tracks_flattened_rows():
    from atom.model_ops.monokernel.glm.staged_moe import workspace_shapes

    shapes = workspace_shapes(48)

    assert shapes["routes"] == (48, 9)
    assert shapes["output"] == (48, 6144)


def test_glm_staged_moe_wrapper_preserves_parameter_names():
    import torch

    from atom.model_ops.monokernel.glm.staged_moe import (
        install_staged_moe_forward,
    )

    class FakeMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, hidden_states):
            return hidden_states

    moe = FakeMoe()
    names_before = tuple(name for name, _ in moe.named_parameters())

    original = install_staged_moe_forward(moe, "glm52.stage.3")

    assert tuple(name for name, _ in moe.named_parameters()) == names_before
    assert original.__self__ is moe
    assert moe._glm52_staged_key == "glm52.stage.3"


def test_glm_staged_moe_uses_fused_fp32_router():
    source = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "monokernel"
        / "glm"
        / "staged_moe.py"
    ).read_text()

    assert "biased_grouped_topk_hip(" in source
    assert "torch.topk(" not in source


def test_glm_staged_moe_publishes_only_after_rank_consensus():
    source = (
        Path(__file__).parents[1] / "atom" / "models" / "glm52_mono.py"
    ).read_text()
    prepare = source[
        source.index("    def _prepare_staged_moe(") : source.index(
            "    def _staged_moe_forward("
        )
    ]

    assert "staged MoE weight mapping failed" in prepare
    assert "staged MoE construction failed" in prepare
    assert prepare.index("staged MoE construction failed") < prepare.index(
        "self._staged_ops.update"
    )


def test_glm_shard_geometry_derives_from_tensor_parallel_size():
    from atom.model_ops.monokernel.config import (
        glm5_decode_shape,
        glm5_shard_config,
    )

    tp8 = glm5_shard_config(8)
    tp4 = glm5_shard_config(4)
    assert (tp8.local_heads, tp8.inter) == (8, 256)
    assert (tp4.local_heads, tp4.inter) == (16, 512)
    with pytest.raises(ValueError, match="TP size"):
        glm5_shard_config(3)

    c4_mtp5 = glm5_decode_shape(running_bs=8, query_len=6)
    assert (c4_mtp5.rows, c4_mtp5.tiles, c4_mtp5.tail_rows) == (48, 6, 8)
    c10_mtp4 = glm5_decode_shape(running_bs=20, query_len=5)
    assert (c10_mtp4.rows, c10_mtp4.tiles, c10_mtp4.tail_rows) == (100, 13, 4)


def test_glm_layout_covers_only_requested_decode_batches():
    for samples in (4, 8):
        scratch, symmetric = glm_layout.layout(samples, 8, 8, 2048)
        assert scratch["_bytes"] > 0
        assert symmetric["_bytes"] > 0


def test_glm_layout_uses_tp4_heads_and_expert_width():
    from atom.model_ops.monokernel.config import glm5_shard_config

    config = glm5_shard_config(4)
    scratch, symmetric = glm_layout.layout(
        8,
        config.local_heads,
        4,
        2048,
        model_config=config,
    )
    stages = dict(
        glm_layout.stage_tasks(
            8,
            config.local_heads,
            2048,
            expert_mxfp4=True,
            model_config=config,
        )
    )
    one_row_stages = dict(
        glm_layout.stage_tasks(
            1,
            config.local_heads,
            2048,
            expert_mxfp4=True,
            model_config=config,
        )
    )

    assert scratch["ugp"] - scratch["mid"] >= 8 * 9 * config.inter * 8
    assert symmetric["_part_stride"] == 4 * 8 * config.hidden * 8
    assert stages["router"] == 8 * (config.n_experts // 8)
    assert stages["ug"] == 8 * 2 * 256
    assert stages["down"] == config.hidden // 32
    assert one_row_stages["split"] == 2 * (2048 // 64)


def test_glm_kernel_builder_accepts_tp4_agentic_tile_geometry():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available():
        pytest.skip("GLM-5.2 MonoKernel requires a visible ROCm device")

    from atom.model_ops.monokernel.config import (
        AttentionWeight,
        KvCacheLayout,
        glm5_shard_config,
    )
    from atom.model_ops.monokernel.glm.kernel import build_glm5_monokernel

    config = glm5_shard_config(4)
    launch = build_glm5_monokernel(
        8,
        config.local_heads,
        4,
        expert_mxfp4=True,
        attention_weight=AttentionWeight.BF16,
        kv_cache_layout=KvCacheLayout.ATOM,
        agentic_row_contract=True,
        model_config=config,
    )

    assert callable(launch)


def test_glm_host_geometry_accepts_tp4_and_physical_shared_expert():
    pytest.importorskip("flydsl")
    from atom.model_ops.monokernel.config import glm5_shard_config
    from atom.model_ops.monokernel.glm.op import _glm_kernel_config
    from atom.model_ops.monokernel.weights import LayerWeights

    config = glm5_shard_config(4)
    weights = LayerWeights(
        heads=config.local_heads,
        t={},
        config=config,
        npes=4,
        physical_experts=257,
    )

    assert _glm_kernel_config(weights, 4) == config
    weights.physical_experts = 256
    with pytest.raises(ValueError, match="physical experts"):
        _glm_kernel_config(weights, 4)
    weights.physical_experts = None
    with pytest.raises(ValueError, match="physical experts"):
        _glm_kernel_config(weights, 4)


def test_glm_tp8_geometry_and_builder_regression():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available():
        pytest.skip("GLM-5.2 MonoKernel requires a visible ROCm device")

    from atom.model_ops.monokernel.config import (
        AttentionWeight,
        KvCacheLayout,
        glm5_shard_config,
    )
    from atom.model_ops.monokernel.glm.kernel import build_glm5_monokernel
    from atom.model_ops.monokernel.glm.op import _glm_kernel_config
    from atom.model_ops.monokernel.weights import LayerWeights

    config = glm5_shard_config(8)
    weights = LayerWeights(
        heads=8,
        t={},
        config=config,
        npes=8,
        physical_experts=257,
    )
    stages = dict(
        glm_layout.stage_tasks(
            8,
            8,
            2048,
            model_config=config,
        )
    )

    assert _glm_kernel_config(weights, 8) == config
    assert stages["split"] == 8 * (2048 // 32)
    assert stages["ug"] == 8 * 256
    launch = build_glm5_monokernel(
        8,
        8,
        8,
        attention_weight=AttentionWeight.BF16,
        kv_cache_layout=KvCacheLayout.ATOM,
        agentic_row_contract=True,
        model_config=config,
    )
    assert callable(launch)


def test_glm_op_binds_shared_workspace_without_owning_peer_buffer(monkeypatch):
    import torch

    pytest.importorskip("flydsl")
    from atom.model_ops.monokernel.config import glm5_shard_config
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.layout import agentic_workspace_layout
    from atom.model_ops.monokernel.glm.op import Glm5MonoKernel
    from atom.model_ops.monokernel.glm import op as glm_op
    from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace
    from atom.model_ops.monokernel.weights import LayerWeights

    config = glm5_shard_config(4)
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

    class Peer:
        storage = object()
        local_address = 17
        addresses = torch.empty(4, dtype=torch.int64)
        closes = 0

        def close(self):
            self.closes += 1

    peer = Peer()
    workspace = GlmAgenticWorkspace(
        shape=shape,
        layout=plan,
        scratch=torch.empty(plan.scratch["_bytes"], dtype=torch.uint8),
        peer_buffer=peer,
        step=torch.zeros(1, dtype=torch.int32),
        hidden_buffers=(
            torch.empty(100, config.hidden, dtype=torch.bfloat16),
            torch.empty(100, config.hidden, dtype=torch.bfloat16),
        ),
    )
    tensors = {
        name: torch.empty(1, dtype=torch.bfloat16)
        for name in (
            "g_in",
            "g_q",
            "g_kv",
            "g_post",
            "w_qkv_a",
            "w_q_b",
            "w_uk",
            "w_uv",
            "w_o",
            "w_r",
            "bias",
            "w_ug",
            "s_ug",
            "w_dn",
            "s_dn",
        )
    }
    weights = LayerWeights(
        heads=16,
        t=tensors,
        config=config,
        npes=4,
        physical_experts=257,
    )
    monkeypatch.setattr(glm_op, "pack_layer_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(glm_op, "build_glm5_monokernel", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(glm_op.torch.cuda, "current_device", lambda: 0)

    kernel = Glm5MonoKernel(
        weights,
        4,
        rank=0,
        npes=4,
        topk=64,
        launches_per_step=128,
        agentic_row_contract=True,
        workspace=workspace,
    )

    assert kernel.workspace is workspace
    assert kernel.scratch is workspace.scratch
    assert kernel.peer_buffer is peer
    assert kernel.step is workspace.step
    kernel.close()
    assert peer.closes == 0


def test_glm_s4_s8_reuse_one_packed_artifact_owner(monkeypatch):
    import torch

    pytest.importorskip("flydsl")
    from atom.model_ops.monokernel.config import glm5_shard_config
    from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
    from atom.model_ops.monokernel.glm.layout import agentic_workspace_layout
    from atom.model_ops.monokernel.glm.op import (
        Glm5MonoKernel,
        Glm5PackedArtifacts,
    )
    from atom.model_ops.monokernel.glm import op as glm_op
    from atom.model_ops.monokernel.glm.workspace import GlmAgenticWorkspace
    from atom.model_ops.monokernel.weights import LayerWeights

    config = glm5_shard_config(4)
    shape = GlmAgenticShape.for_graph(
        batch_capacity=20,
        query_len=5,
        dcp_size=1,
        query_replication=False,
    )
    plan = agentic_workspace_layout(shape, npes=4, sparse_attention_topk=64)

    class Peer:
        storage = object()
        local_address = 17
        addresses = torch.empty(4, dtype=torch.int64)

        def close(self):
            raise AssertionError("tile kernels do not own the shared peer buffer")

    workspace = GlmAgenticWorkspace(
        shape=shape,
        layout=plan,
        scratch=torch.empty(plan.scratch["_bytes"], dtype=torch.uint8),
        peer_buffer=Peer(),
        step=torch.zeros(1, dtype=torch.int32),
        hidden_buffers=(
            torch.empty(100, config.hidden, dtype=torch.bfloat16),
            torch.empty(100, config.hidden, dtype=torch.bfloat16),
        ),
    )
    tensors = {
        name: torch.empty(1, dtype=torch.bfloat16)
        for name in (
            "g_in",
            "g_q",
            "g_kv",
            "g_post",
            "w_qkv_a",
            "w_q_b",
            "w_uk",
            "w_uv",
            "w_o",
            "w_r",
            "bias",
            "w_ug",
            "s_ug",
            "w_dn",
            "s_dn",
        )
    }
    weights = LayerWeights(
        heads=16,
        t=tensors,
        config=config,
        npes=4,
        physical_experts=257,
    )
    packed_tensors = {
        "packed_probe": torch.empty(3, dtype=torch.uint8),
        "packed_router": torch.empty(5, dtype=torch.bfloat16),
    }
    pack_calls = []

    def pack(*_args, **_kwargs):
        pack_calls.append(1)
        return packed_tensors.copy()

    monkeypatch.setattr(glm_op, "pack_layer_weights", pack)
    monkeypatch.setattr(
        glm_op, "build_glm5_monokernel", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(glm_op.torch.cuda, "current_device", lambda: 0)

    artifacts = Glm5PackedArtifacts.pack(
        weights,
        npes=4,
        attention_weight="fp8_block128",
        with_indexer=False,
    )
    s4 = Glm5MonoKernel(
        weights,
        4,
        npes=4,
        topk=64,
        launches_per_step=128,
        agentic_row_contract=True,
        workspace=workspace,
        packed_artifacts=artifacts,
    )
    s8 = Glm5MonoKernel(
        weights,
        8,
        npes=4,
        topk=64,
        launches_per_step=128,
        agentic_row_contract=True,
        workspace=workspace,
        packed_artifacts=artifacts,
    )

    assert pack_calls == [1]
    assert s4.packed_artifacts is artifacts
    assert s8.packed_artifacts is artifacts
    assert s4.packed is s8.packed
    for name, packed in packed_tensors.items():
        assert s4.packed[name] is packed
        assert s8.packed[name].data_ptr() == packed.data_ptr()
    with pytest.raises(TypeError):
        s4.packed["replacement"] = torch.empty(1)
    s4.close()
    assert s8.packed["packed_probe"].data_ptr() == packed_tensors[
        "packed_probe"
    ].data_ptr()


def test_glm_full_layer_weight_mapping_uses_tp4_geometry(monkeypatch):
    import torch

    layernorm = types.ModuleType("atom.model_ops.layernorm")
    layernorm.rmsnorm2d_fwd_ = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "atom.model_ops.layernorm", layernorm)
    module = _glm_mono_module()
    config = module.glm5_shard_config(4)
    calls = {}

    monkeypatch.setattr(
        module,
        "_bf16_vector",
        lambda _tensor, _name, _size: torch.empty(1, dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        module,
        "_mxfp4_expert_tensors",
        lambda *_args: {
            name: torch.empty(1, dtype=torch.uint8)
            for name in ("w_ug", "s_ug", "w_dn", "s_dn")
        },
    )

    def linear(_linear, *, name, logical_rows, logical_cols, **_kwargs):
        calls[name] = (logical_rows, logical_cols)
        if name == "kv_b_proj":
            return torch.empty(logical_rows, logical_cols, dtype=torch.bfloat16)
        return torch.empty(1, dtype=torch.bfloat16)

    monkeypatch.setattr(module, "linear_bf16", linear)
    norm = SimpleNamespace(eps=module.EPS, weight=torch.empty(1))
    experts = SimpleNamespace(
        global_num_experts=config.n_experts,
        num_fused_shared_experts=config.num_shared_experts,
        use_ep=False,
        quant_method=SimpleNamespace(is_guinterleave=False),
        intermediate_size_per_partition=config.inter,
    )
    gate = SimpleNamespace(
        e_score_correction_bias=torch.zeros(config.n_experts, dtype=torch.float32)
    )
    attention = SimpleNamespace(
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        kv_b_proj=object(),
        fused_qkv_a_proj=object(),
        q_b_proj=object(),
        o_proj=object(),
    )
    layer = SimpleNamespace(
        input_layernorm=norm,
        post_attention_layernorm=norm,
        self_attn=attention,
        mlp=SimpleNamespace(experts=experts, gate=gate),
    )

    weights = module._layer_weights(layer, rank=0, npes=4)

    assert weights.config == config
    assert weights.heads == 16
    assert weights.physical_experts == 257
    assert calls["q_b_proj"][0] == 16 * (config.nope_dim + config.pe_dim)
    assert calls["o_proj"][1] == 16 * config.v_dim


def test_glm_flat_atom_cache_requires_page_one():
    from atom.model_ops.monokernel.dispatch import is_flat_atom_cache_page_size

    assert is_flat_atom_cache_page_size(1)
    assert not is_flat_atom_cache_page_size(2)
    assert not is_flat_atom_cache_page_size(None)


def test_glm_flat_and_paged_sparse_indices_are_physical_rows():
    indices = [90, 91, 92, 93, 40, 41, 42, 43]
    assert list(
        glm_layout.sparse_cache_rows(
            indices,
            sample=0,
            topk=4,
            cur_pos=1,
        )
    ) == [0, 1]
    assert list(
        glm_layout.sparse_cache_rows(
            indices,
            sample=1,
            topk=4,
            cur_pos=8,
        )
    ) == [40, 41, 42, 43]

    physical = [17, 3, 99, 8, 70]
    indptr = [0, 2, 5]
    assert glm_layout.sparse_cache_rows(
        physical,
        sample=0,
        topk=4,
        sparse_kv_indptr=indptr,
    ) == [17, 3]
    assert glm_layout.sparse_cache_rows(
        physical,
        sample=1,
        topk=4,
        sparse_kv_indptr=indptr,
    ) == [99, 8, 70]


def test_glm_padded_rows_use_safe_physical_row_without_cache_store():
    physical = [17]
    indptr = [0, 1, 1, 1, 1]
    slots = [7, -1, -1, -1]

    active = glm_layout.paged_row_contract(physical, indptr, 0, slots)
    assert active == {
        "active": True,
        "context": 1,
        "index_base": 0,
        "safe_row": 17,
        "write_cache": True,
    }
    for sample in range(1, 4):
        padded = glm_layout.paged_row_contract(physical, indptr, sample, slots)
        assert padded == {
            "active": False,
            "context": 0,
            "index_base": 0,
            "safe_row": 0,
            "write_cache": False,
        }
        assert glm_layout.sparse_cache_rows(
            physical,
            sample=sample,
            topk=4,
            sparse_kv_indptr=indptr,
        ) == []


def test_glm_non_owner_row_attends_without_writing_cache():
    contract = glm_layout.paged_row_contract([17], [0, 1], 0, [-1])

    assert contract == {
        "active": True,
        "context": 1,
        "index_base": 0,
        "safe_row": 17,
        "write_cache": False,
    }


def test_glm_device_local_sparse_activity_requires_a_nonempty_row():
    source = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "monokernel"
        / "glm"
        / "kernel.py"
    ).read_text()
    predicate = source[
        source.index("        def row_local_sparse_active(") : source.index(
            "        def row_position("
        )
    ]

    assert "return row_active(s) & (count > 0) & (end > begin)" in predicate


def test_glm_empty_sparse_merge_avoids_divide_by_zero():
    source = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "monokernel"
        / "glm"
        / "kernel.py"
    ).read_text()

    assert "def row_writes_cache(s):" in source
    assert "inv_den = (den > 0.0).select(_rcp(den), fx.Float32(0.0))" in source


def test_glm_router_preserves_full_fp32_score_bits():
    source = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "monokernel"
        / "glm"
        / "kernel.py"
    ).read_text()

    assert "ok & fx.Int32(-256)" not in source
    assert "winner = wave_umax_dpp" in source
    assert "expert = 255 - winner" in source
    assert source.count("fx.min(fx.Int32(topk), bound)") >= 2


def test_glm_full_and_shared_layers_use_one_sparse_buffer():
    import torch

    module = _glm_mono_module()
    shared = torch.arange(16, dtype=torch.int32)
    full_indexer = SimpleNamespace(sparse_kv_indices_buffer=shared)
    full = SimpleNamespace(
        self_attn=SimpleNamespace(
            mla_attn=SimpleNamespace(
                impl=SimpleNamespace(sparse_kv_indices_buffer=shared)
            ),
            indexer=full_indexer,
        )
    )
    reused = SimpleNamespace(
        self_attn=SimpleNamespace(
            mla_attn=SimpleNamespace(
                impl=SimpleNamespace(sparse_kv_indices_buffer=shared)
            ),
            indexer=None,
        )
    )

    assert module._shared_sparse_buffer([full, reused]).data_ptr() == shared.data_ptr()
    reused.self_attn.mla_attn.impl.sparse_kv_indices_buffer = shared.clone()
    with pytest.raises(module.MonoUnsupported, match="do not share"):
        module._shared_sparse_buffer([full, reused])


def test_glm_bf16_attention_layout_mapping():
    import torch

    module = _glm_mono_module()
    from atom.model_ops.monokernel.weights import linear_bf16

    weight = torch.arange(2 * 5 * 3, dtype=torch.float32).to(torch.bfloat16).view(10, 3)
    linear = SimpleNamespace(
        input_size=3,
        output_size=10,
        weight=weight,
        quant_type=SimpleNamespace(name="No"),
        params_dtype=torch.bfloat16,
    )
    assert (
        linear_bf16(
            linear,
            name="kv_b_proj",
            logical_rows=10,
            logical_cols=3,
        ).data_ptr()
        == weight.data_ptr()
    )

    w_uk, w_uv = module._split_kv_b(
        weight,
        heads=2,
        nope_dim=2,
        value_dim=3,
        kv_lora=3,
    )
    by_head = weight.view(2, 5, 3)
    assert torch.equal(w_uk.view(2, 3, 2), by_head[:, :2].transpose(1, 2))
    assert torch.equal(w_uv.view(2, 3, 3), by_head[:, 2:])

    linear.weight = weight.float()
    with pytest.raises(module.MonoUnsupported, match="BF16 weight has dtype"):
        linear_bf16(
            linear,
            name="kv_b_proj",
            logical_rows=10,
            logical_cols=3,
        )


@pytest.mark.skipif(
    not hasattr(__import__("torch"), "float8_e4m3fn"),
    reason="E4M3 FP8 dtype is unavailable",
)
def test_glm_recipe_per_token_fp8_attention_mapping():
    import torch

    module = _glm_mono_module()
    from atom.model_ops.monokernel.weights import linear_bf16

    source = torch.randn(16, 64, dtype=torch.bfloat16)
    linear, expected = _preshuffled_per_token_fp8_linear(source)
    restored = linear_bf16(
        linear,
        name="kv_b_proj",
        logical_rows=16,
        logical_cols=64,
    )

    assert restored.is_contiguous()
    assert torch.equal(restored, expected)
    w_uk, w_uv = module._split_kv_b(
        restored,
        heads=2,
        nope_dim=2,
        value_dim=6,
        kv_lora=64,
    )
    by_head = expected.view(2, 8, 64)
    assert torch.equal(w_uk.view(2, 64, 2), by_head[:, :2].transpose(1, 2))
    assert torch.equal(w_uv.view(2, 6, 64), by_head[:, 2:])


def test_glm_default_off_does_not_inspect_runtime_config():
    runner = _glm_mono_module().Glm52MonoDecode(None, object(), "off")
    assert runner._enabled is False
    assert runner._ops == {}


def test_glm_graph_warmup_dispatch_counts_padded_rows(monkeypatch):
    import torch

    module = _glm_mono_module()
    samples = 4
    runner = object.__new__(module.Glm52MonoDecode)
    runner._enabled = True
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(
        tensor_parallel_size=8,
        kv_cache_dtype="bf16",
        speculative_config=None,
        enable_dp_attention=False,
        decode_context_parallel_size=1,
    )
    runner._lm = SimpleNamespace(
        model=SimpleNamespace(aux_hidden_state_layers=[], layers=[])
    )
    runner._mono_layers = lambda: []
    runner._prepare = lambda _samples: True
    metadata = SimpleNamespace(
        max_seqlen_q=1,
        slot_mapping=torch.tensor([7, -1, -1, -1], dtype=torch.int64),
        sparse_kv_indptr=torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
    )
    context = SimpleNamespace(is_prefill=False, scheduled_bs=1)
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=context,
            attn_metadata=metadata,
            ubatch_slices=None,
            kv_cache_data={},
        ),
    )
    monkeypatch.setattr(module, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(
        module,
        "_shared_sparse_buffer",
        lambda _layers: torch.arange(samples, dtype=torch.int32),
    )

    args = (
        torch.arange(samples),
        torch.arange(samples, dtype=torch.int64),
        None,
        None,
    )
    assert runner.supports(*args)

    metadata.slot_mapping = torch.arange(samples * 2, dtype=torch.int64)[::2]
    assert not runner.supports(*args)
    metadata.slot_mapping = torch.tensor([7, -1, -1, -1], dtype=torch.int64)
    metadata.sparse_kv_indptr = torch.tensor(
        [0, 99, 1, 99, 1, 99, 1, 99, 1, 99],
        dtype=torch.int32,
    )[::2]
    assert not runner.supports(*args)


def test_shared_linear_weight_unshuffle_round_trip():
    import torch

    from atom.model_ops.monokernel.weights import _unshuffle_linear_weight

    native = torch.arange(16 * 64, dtype=torch.int32).to(torch.uint8).view(16, 64)
    shuffled = _preshuffle_linear_weight(native)
    shuffled.is_shuffled = True
    restored = _unshuffle_linear_weight(shuffled)
    assert torch.equal(restored, native)

    from atom.model_ops.monokernel.packing import pack_a16w4_weight

    assert torch.equal(pack_a16w4_weight(restored), shuffled.reshape(-1))


def test_shared_linear_scale_unshuffle_round_trip():
    import torch

    from atom.model_ops.monokernel.weights import _unshuffle_linear_scale

    native = torch.arange(2 * 256 * 16, dtype=torch.int32).to(torch.uint8).view(512, 16)
    shuffled = (
        native.view(16, 2, 16, 2, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view_as(native)
    )
    restored = _unshuffle_linear_scale(shuffled, experts=2, rows=256, groups=16)
    assert torch.equal(restored, native.view(2, 256, 16))

    from atom.model_ops.monokernel.packing import pack_a16w4_scale

    assert torch.equal(pack_a16w4_scale(restored), shuffled.reshape(-1))


def test_kimi_runner_closes_shared_reductions_once():
    KimiMonoDecode = _kimi_mono_module().KimiMonoDecode

    class Owned:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1

    runner = object.__new__(KimiMonoDecode)
    attention_reduce = Owned()
    moe_reduce = Owned()
    runner._ops = {(1, 4, "staged"): object()}
    runner._weights = {}
    runner._reductions = {
        (4, "staged"): (attention_reduce, moe_reduce),
    }
    runner._refused = set()

    runner.close()
    runner.close()

    assert attention_reduce.calls == 1
    assert moe_reduce.calls == 1


def test_kimi_failed_prepare_preserves_existing_shared_reductions():
    KimiMonoDecode = _kimi_mono_module().KimiMonoDecode

    class Owned:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1

    runner = object.__new__(KimiMonoDecode)
    existing = (Owned(), Owned())
    created = (Owned(), Owned())
    runner._reductions = {
        (4, "staged"): existing,
        (8, "staged"): created,
    }

    runner._close_reductions([(8, "staged")])

    assert [owned.calls for owned in existing] == [0, 0]
    assert [owned.calls for owned in created] == [1, 1]
    assert runner._reductions == {(4, "staged"): existing}


def test_kimi_memory_reserve_tracks_persistent_packed_artifacts():
    module = _kimi_mono_module()
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._layer_specs = lambda _samples: [object(), object(), object()]

    assert runner.memory_reserve_bytes() == 3 * (
        (256 << 20) + 2 * (32 << 20)
    )


def test_kimi_s4_s8_share_bucket_independent_packed_artifacts():
    source = (
        Path(__file__).parents[1] / "atom" / "models" / "kimi_k3_mono.py"
    ).read_text()

    assert "self._packed_artifacts" in source
    assert "packed_artifacts=packed_artifacts" in source


def test_kimi_default_off_does_not_inspect_runtime_config():
    KimiMonoDecode = _kimi_mono_module().KimiMonoDecode
    runner = KimiMonoDecode(None, object(), "off")
    assert runner._enabled is False
    assert runner._ops == {}


@pytest.mark.parametrize(
    "override",
    (
        {"speculative_config": object()},
        {"decode_context_parallel_size": 2},
    ),
)
def test_kimi_constructor_defers_agentic_gates_to_each_forward(monkeypatch, override):
    module = _kimi_mono_module()
    config = SimpleNamespace(
        tensor_parallel_size=8,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_dp_attention=False,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
        speculative_config=None,
        kv_cache_dtype="bf16",
    )
    for name, value in override.items():
        setattr(config, name, value)
    monkeypatch.setattr(module, "is_plugin_mode", lambda: False)

    runner = module.KimiMonoDecode(None, config, "auto")

    assert runner._enabled is True


@pytest.mark.parametrize("samples", (4, 8))
def test_kimi_inputs_embeds_are_eligible(monkeypatch, samples):
    import torch

    module = _kimi_mono_module()
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(tensor_parallel_size=8, kv_cache_dtype="bf16")
    monkeypatch.setattr(module, "get_forward_context", lambda: _kimi_decode_context(samples))
    input_ids = torch.arange(samples)
    positions = torch.arange(samples)
    inputs_embeds = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    assert runner.supports(input_ids, positions, None, inputs_embeds)
    assert not runner.supports(input_ids, positions, object(), inputs_embeds)
    assert not runner.supports(input_ids, positions, None, inputs_embeds.float())
    assert not runner.supports(input_ids, positions, None, inputs_embeds[:, :-1])
    assert not runner.supports(input_ids, positions, None, inputs_embeds.T.contiguous().T)


def test_kimi_supports_requires_contiguous_int32_decode_slots(monkeypatch):
    import torch

    module = _kimi_mono_module()
    samples = 4
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(tensor_parallel_size=8, kv_cache_dtype="bf16")
    inputs = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)
    positions = torch.arange(samples, dtype=torch.int64)

    for invalid in (
        torch.arange(samples, dtype=torch.int64),
        torch.arange(samples - 1, dtype=torch.int32),
        torch.arange(samples * 2, dtype=torch.int32)[::2],
    ):
        monkeypatch.setattr(
            module,
            "get_forward_context",
            lambda invalid=invalid: _kimi_decode_context(
                samples,
                state_indices=invalid,
            ),
        )
        assert not runner.supports(torch.arange(samples), positions, None, inputs)


def test_kimi_supports_rejects_multi_token_decode(monkeypatch):
    import torch

    module = _kimi_mono_module()
    samples = 4
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(tensor_parallel_size=8, kv_cache_dtype="bf16")
    context = _kimi_decode_context(samples)
    context.attn_metadata.kda_metadata.num_decodes = 1
    monkeypatch.setattr(module, "get_forward_context", lambda: context)

    assert not runner.supports(
        torch.arange(samples),
        torch.arange(samples, dtype=torch.int64),
        None,
        torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16),
    )


def test_kimi_state_pool_contract_accepts_agentic_fp16_recurrence():
    import torch

    module = _kimi_mono_module()
    config = module.KIMI_K3_CONFIG
    cache = SimpleNamespace(
        k_cache=torch.zeros(
            2,
            3,
            3 * config.local_heads * config.v_dim,
            dtype=torch.bfloat16,
        ),
        v_cache=torch.zeros(
            2,
            config.local_heads,
            config.v_dim,
            config.v_dim,
            dtype=torch.float32,
        ),
    )

    assert module._kda_state_pool_supported(cache)
    cache.v_cache = cache.v_cache.to(torch.float16)
    assert module._kda_state_pool_supported(cache)
    cache.v_cache = cache.v_cache.to(torch.bfloat16)
    assert not module._kda_state_pool_supported(cache)
    cache.v_cache = torch.zeros(
        2,
        config.local_heads,
        config.v_dim,
        config.v_dim,
        dtype=torch.float32,
    )
    cache.k_cache = torch.zeros(
        2,
        4,
        3 * config.local_heads * config.v_dim,
        dtype=torch.bfloat16,
    )
    assert not module._kda_state_pool_supported(cache)


def test_kimi_state_dtype_skips_mla_cache_without_v_tensor():
    import torch

    module = _kimi_mono_module()
    layers = [
        SimpleNamespace(layer_idx=4, is_linear_attn=False),
        SimpleNamespace(layer_idx=5, is_linear_attn=True),
    ]
    model = SimpleNamespace(layers=layers, start_layer=0, end_layer=2)
    fwd = SimpleNamespace(
        kv_cache_data={
            "layer_4": SimpleNamespace(v_cache=None),
            "layer_5": SimpleNamespace(
                v_cache=torch.empty(1, dtype=torch.float16)
            ),
        }
    )

    assert module._kda_state_dtype(model, fwd) is torch.float16


def test_kimi_supports_requires_atomic_eager_prepare(monkeypatch):
    import torch

    module = _kimi_mono_module()
    samples = 4
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(tensor_parallel_size=8, kv_cache_dtype="bf16")
    layer = SimpleNamespace(
        layer_idx=1,
        is_linear_attn=True,
        block_sparse_moe=object(),
    )
    runner._lm = SimpleNamespace(
        model=SimpleNamespace(layers=[layer], start_layer=0, end_layer=1)
    )
    context = _kimi_decode_context(samples)
    context.kv_cache_data = {
        "layer_1": SimpleNamespace(v_cache=torch.empty(1, dtype=torch.float32))
    }
    monkeypatch.setattr(module, "_kda_state_pool_supported", lambda _cache: True)
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    calls = []
    runner._prepare = lambda rows, dtype: calls.append((rows, dtype)) or False
    args = (
        torch.arange(samples),
        torch.arange(samples, dtype=torch.int64),
        None,
        torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16),
    )

    assert not runner.supports(*args)
    assert calls == [(samples, torch.float32)]
    runner._prepare = lambda rows, dtype: calls.append((rows, dtype)) or True
    assert runner.supports(*args)
    assert calls == [(samples, torch.float32), (samples, torch.float32)]


@pytest.mark.parametrize(("samples", "actual_tokens"), ((4, 3), (8, 5)))
def test_kimi_graph_padding_dispatches(monkeypatch, samples, actual_tokens):
    import torch

    module = _kimi_mono_module()
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(tensor_parallel_size=8, kv_cache_dtype="bf16")
    state_indices = torch.cat(
        (
            torch.arange(actual_tokens, dtype=torch.int32),
            torch.full((samples - actual_tokens,), -1, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: _kimi_decode_context(samples, actual_tokens, state_indices),
    )

    assert runner.supports(
        torch.arange(samples),
        torch.arange(samples),
        None,
        torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("samples", (4, 8))
def test_kimi_padded_forward_propagates_negative_slots_without_state_mutation(
    monkeypatch, samples
):
    import torch

    module = _kimi_mono_module()
    actual_tokens = samples - 1
    state_indices = torch.cat(
        (
            torch.arange(actual_tokens, dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
        )
    )
    conv_state = torch.randn(4, 3, dtype=torch.bfloat16)
    recurrent_state = torch.randn(4, 3, dtype=torch.float32)
    before = (conv_state.clone(), recurrent_state.clone())
    seen = []

    class FakeOp:
        block_write_idx = 0

        @staticmethod
        def forward(hidden, _blocks, indices, conv, recurrent, **_kwargs):
            seen.append(indices)
            assert conv.data_ptr() == conv_state.data_ptr()
            assert recurrent.data_ptr() == recurrent_state.data_ptr()
            return hidden

    layer = SimpleNamespace(layer_idx=1)
    model = SimpleNamespace(
        get_input_embeddings=lambda _ids: pytest.fail("inputs_embeds must be used"),
        layers=[layer],
        start_layer=0,
        end_layer=1,
        output_attn_res=lambda hidden, *_args: (hidden, None),
    )
    runner = object.__new__(module.KimiMonoDecode)
    runner._lm = SimpleNamespace(model=model)
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(kv_cache_dtype="bf16")
    runner._op = lambda *_args: SimpleNamespace(op=FakeOp())
    context = _kimi_decode_context(samples, actual_tokens, state_indices)
    context.kv_cache_data = {
        "layer_1": SimpleNamespace(k_cache=conv_state, v_cache=recurrent_state)
    }
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    inputs_embeds = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    runner.forward(torch.arange(samples), torch.arange(samples), inputs_embeds)

    assert len(seen) == 1
    assert torch.equal(seen[0], state_indices)
    assert seen[0][-1].item() < 0
    assert torch.equal(conv_state, before[0])
    assert torch.equal(recurrent_state, before[1])


def test_kimi_negative_slot_device_guards_cover_staged_and_mono_paths():
    root = Path(__file__).parents[1] / "atom" / "model_ops" / "monokernel" / "k3"
    recurrence = (root / "kda_recurrence.py").read_text()
    mono = (root / "kernel.py").read_text()

    assert "if slot >= 0:\n            decode()\n        else:\n            zero_output()" in recurrence
    assert "if (input_slot >= 0) & (output_slot >= 0):" in mono
    assert mono.count("valid_state = (input_slot >= 0) & (output_slot >= 0)") >= 3


def test_per_layer_mailboxes_alternate_between_decode_steps():
    root = Path(__file__).parents[1] / "atom" / "model_ops" / "monokernel"
    sources = {
        "symmetric": (root / "symmetric_allreduce.py").read_text(),
        "tail": (root / "k3" / "tail.py").read_text(),
        "k3": (root / "k3" / "kernel.py").read_text(),
        "glm": (root / "glm" / "kernel.py").read_text(),
        "gemm": (root / "gemm_a16w16.py").read_text(),
    }

    assert sources["symmetric"].count("slot = step_value & 1") == 3
    assert "slot = step_value & 1" in sources["tail"]
    assert "slot = step_value & 1" in sources["k3"]
    assert "peer_slot = step_value & 1" in sources["glm"]
    assert "slot = step_value & 1" in sources["gemm"]
    for source in sources.values():
        assert "(step_value * LAYER_SLOTS + layer) & 1" not in source
        assert "(step_value * launches_per_step + layer) & 1" not in source


def test_kimi_mono_respects_declared_mxfp4_layout():
    source = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "monokernel"
        / "k3"
        / "kda.py"
    ).read_text()

    assert "prepare_mxfp4_expert_storage(self.W)" in source
    assert 'pack_mxfp4(self.t["w_ug"])' not in source
    assert 'pack_mxfp4(self.t["w_dn"])' not in source


def test_native_route_stats_preserve_bounded_fallback_reasons():
    from atom.model_ops.monokernel.telemetry import MonoRouteStats

    stats = MonoRouteStats("kimi_k3")
    stats.record_attempt(4)
    stats.record_hit("staged", 4)
    stats.record_attempt(8)
    stats.record_fallback("state_layout", 8)

    assert stats.snapshot() == {
        "model": "kimi_k3",
        "attempts": 2,
        "hits": {"staged:s4": 1},
        "fallbacks": {"state_layout:s8": 1},
    }


def test_close_model_monokernels_is_deduplicated_and_idempotent():
    from atom.model_ops.monokernel import (
        close_model_monokernels,
        model_monokernel_memory_reserve,
    )

    class Owned:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1

    owned = Owned()
    glm_owned = Owned()
    modules = [
        SimpleNamespace(_mono=owned),
        SimpleNamespace(_mono=owned),
        SimpleNamespace(_glm52_mono=glm_owned),
        SimpleNamespace(),
    ]
    model = SimpleNamespace(modules=lambda: modules)

    close_model_monokernels(model)
    close_model_monokernels(model)

    assert owned.calls == 1
    assert glm_owned.calls == 1
    owned.memory_reserve_bytes = lambda: 123
    glm_owned.memory_reserve_bytes = lambda: 456
    assert model_monokernel_memory_reserve(model) == 579


def test_close_model_monokernels_retries_failed_close():
    from atom.model_ops.monokernel import close_model_monokernels

    class Flaky:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("close failed")

    owned = Flaky()
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(_mono=owned)])

    with pytest.raises(RuntimeError, match="close failed"):
        close_model_monokernels(model)
    close_model_monokernels(model)

    assert owned.calls == 2


def test_model_runner_closes_monokernels_before_distributed_teardown():
    source = (
        Path(__file__).parents[1] / "atom" / "model_engine" / "model_runner.py"
    ).read_text()

    close_at = source.index("close_model_monokernels(self.model)")
    destroy_at = source.index("destroy_dist_env()", close_at)
    assert close_at < destroy_at
    assert "model_monokernel_memory_reserve(self.model)" in source


def test_offline_profiler_forwards_tokenizer_remote_code_trust():
    source = (
        Path(__file__).parents[1] / "atom" / "examples" / "profile_offline.py"
    ).read_text()

    assert (
        "AutoTokenizer.from_pretrained(\n"
        "        args.model, trust_remote_code=args.trust_remote_code\n"
        "    )"
    ) in source


def test_kimi_forward_uses_inputs_embeds(monkeypatch):
    import torch

    module = _kimi_mono_module()
    samples = 4
    inputs_embeds = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    def unexpected_embedding(_input_ids):
        raise AssertionError("inputs_embeds must bypass token embedding")

    model = SimpleNamespace(
        get_input_embeddings=unexpected_embedding,
        layers=[],
        start_layer=0,
        end_layer=0,
        output_attn_res=lambda hidden, *_args: (hidden, None),
    )
    runner = object.__new__(module.KimiMonoDecode)
    runner._lm = SimpleNamespace(model=model)
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(kv_cache_dtype="bf16")
    monkeypatch.setattr(module, "get_forward_context", lambda: _kimi_decode_context(samples))

    output = runner.forward(torch.arange(samples), torch.arange(samples), inputs_embeds)

    assert output is inputs_embeds


def test_kimi_kda_a_log_is_fp32(monkeypatch):
    import torch

    module = _kimi_mono_module()
    tensor = torch.zeros(1, dtype=torch.bfloat16)
    a_log = torch.zeros(12, dtype=torch.bfloat16)
    attn = SimpleNamespace(
        in_proj=SimpleNamespace(weight=tensor),
        f_b_proj=SimpleNamespace(weight=tensor),
        conv_weight=tensor,
        A_log=a_log,
        dt_bias=torch.zeros(12, 128, dtype=torch.bfloat16),
        o_norm=SimpleNamespace(weight=tensor),
        o_proj=SimpleNamespace(weight=tensor),
    )
    experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_guinterleave=False),
        global_num_experts=module.KIMI_K3_CONFIG.n_experts,
        intermediate_size_per_partition=module.KIMI_K3_CONFIG.inter,
        w13_weight=tensor,
        w2_weight=tensor,
        w13_weight_scale=tensor,
        w2_weight_scale=tensor,
    )
    moe = SimpleNamespace(
        experts=experts,
        routed_expert_up_proj=SimpleNamespace(
            weight=torch.zeros(module.KIMI_K3_CONFIG.hidden // 8, 1, dtype=torch.bfloat16)
        ),
        routed_expert_down_proj=SimpleNamespace(weight=tensor),
        routed_expert_norm=SimpleNamespace(weight=tensor),
        gate=SimpleNamespace(weight=tensor, e_score_correction_bias=tensor),
        shared_experts=SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=tensor),
            down_proj=SimpleNamespace(weight=tensor),
        ),
    )
    layer = SimpleNamespace(
        layer_idx=1,
        is_linear_attn=True,
        self_attn=attn,
        block_sparse_moe=moe,
        input_layernorm=SimpleNamespace(weight=tensor),
        post_attention_layernorm=SimpleNamespace(weight=tensor),
        self_attention_res_norm=SimpleNamespace(weight=tensor),
        self_attention_res_proj=SimpleNamespace(weight=tensor),
        mlp_res_norm=SimpleNamespace(weight=tensor),
        mlp_res_proj=SimpleNamespace(weight=tensor),
    )
    monkeypatch.setattr(module, "linear_bf16", lambda *_args, **_kwargs: tensor)
    monkeypatch.setattr(module, "atom_mxfp4_storage_view", lambda *_args, **_kwargs: tensor)

    weights = module._layer_weights(layer, rank=0, npes=8)

    assert a_log.dtype == torch.bfloat16
    assert weights.t["kda_a_log"].dtype == torch.float32
    assert weights.t["kda_a_log"].is_contiguous()
    assert weights.t["w_ug"] is experts.w13_weight
    assert weights.t["s_ug"] is experts.w13_weight_scale
    assert weights.mxfp4_weight_layout is module.Mxfp4WeightLayout.ATOM
    assert weights.mxfp4_scale_layout is module.Mxfp4ScaleLayout.ATOM


@pytest.mark.skipif(
    not hasattr(__import__("torch"), "float4_e2m1fn_x2"),
    reason="MXFP4 dtype is unavailable",
)
def test_shared_mxfp4_linear_round_trip():
    import copy

    import torch

    from atom.model_ops.monokernel.dispatch import MonoUnsupported
    from atom.model_ops.monokernel.weights import linear_bf16

    source = torch.randn(256, 256, dtype=torch.bfloat16)
    linear, expected = _preshuffled_mxfp4_linear(source)

    restored = linear_bf16(
        linear,
        name="routed_expert_down_proj",
        logical_rows=256,
        logical_cols=256,
    )

    assert torch.equal(restored, expected)
    linear.weight.is_shuffled = False
    with pytest.raises(MonoUnsupported, match="must be preshuffled"):
        linear_bf16(
            linear,
            name="routed_expert_down_proj",
            logical_rows=256,
            logical_cols=256,
        )
    linear.weight.is_shuffled = True
    bad_scale = copy.copy(linear)
    bad_scale.weight_scale = linear.weight_scale.reshape(-1)[:-1]
    with pytest.raises(MonoUnsupported, match="weight_scale shape .* expected"):
        linear_bf16(
            bad_scale,
            name="routed_expert_down_proj",
            logical_rows=256,
            logical_cols=256,
        )


@pytest.mark.skipif(
    not hasattr(__import__("torch"), "float4_e2m1fn_x2"),
    reason="MXFP4 dtype is unavailable",
)
def test_shared_mxfp4_linear_dequantizes_only_requested_rows(monkeypatch):
    import torch

    from atom.model_ops.monokernel import weights

    source = torch.randn(256, 256, dtype=torch.bfloat16)
    linear, expected = _preshuffled_mxfp4_linear(source)
    dequantize = weights.dequantize_mxfp4
    seen = []

    def record_dequantize(packed, scale):
        seen.append((packed.shape, scale.shape))
        return dequantize(packed, scale)

    monkeypatch.setattr(weights, "dequantize_mxfp4", record_dequantize)
    restored = weights.linear_bf16(
        linear,
        name="routed_expert_up_proj",
        logical_rows=256,
        logical_cols=256,
        row_start=96,
        row_count=32,
    )

    assert seen == [(torch.Size([32, 128]), torch.Size([32, 8]))]
    assert torch.equal(restored, expected[96:128])


def test_shared_bf16_linear_is_unchanged():
    import torch

    from atom.model_ops.monokernel.weights import linear_bf16

    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    linear = SimpleNamespace(
        input_size=64,
        output_size=32,
        quant_type=SimpleNamespace(name="No"),
        params_dtype=torch.bfloat16,
        weight=weight,
    )

    full = linear_bf16(
        linear,
        name="routed_expert_down_proj",
        logical_rows=32,
        logical_cols=64,
    )
    shard = linear_bf16(
        linear,
        name="routed_expert_up_proj",
        logical_rows=32,
        logical_cols=64,
        row_start=16,
        row_count=16,
    )

    assert full is weight
    assert torch.equal(shard, weight[16:])


def test_atom_expert_storage_is_zero_copy(monkeypatch):
    import torch

    from atom.model_ops.monokernel.config import Mxfp4ScaleLayout, Mxfp4WeightLayout
    from atom.model_ops.monokernel.weights import LayerWeights, prepare_mxfp4_expert_storage
    from atom.model_ops.monokernel import packing

    config = SimpleNamespace(name="test", n_experts=2, inter=128, routed_hidden=128)
    tensors = {
        "w_ug": torch.zeros(2 * 2 * 128 * 128 // 2, dtype=torch.uint8),
        "s_ug": torch.zeros(512, 8, dtype=torch.uint8),
        "w_dn": torch.zeros(2 * 128 * 128 // 2, dtype=torch.uint8),
        "s_dn": torch.zeros(256, 8, dtype=torch.uint8),
    }
    tensors["w_ug"].is_shuffled = True
    tensors["w_dn"].is_shuffled = True
    weights = LayerWeights(
        heads=1,
        t=tensors,
        config=config,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
    )

    def unexpected_pack(_tensor):
        raise AssertionError("ATOM storage must bypass packing")

    monkeypatch.setattr(packing, "pack_a16w4_weight", unexpected_pack)
    monkeypatch.setattr(packing, "pack_a16w4_scale", unexpected_pack)
    prepared = prepare_mxfp4_expert_storage(weights)

    for output, name in zip(prepared, ("w_ug", "s_ug", "w_dn", "s_dn")):
        assert output.data_ptr() == tensors[name].data_ptr()
    defaults = LayerWeights(heads=1, t={}, config=config)
    assert defaults.mxfp4_weight_layout is Mxfp4WeightLayout.NATIVE
    assert defaults.mxfp4_scale_layout is Mxfp4ScaleLayout.NATIVE


def test_glm_fused_shared_expert_storage_is_zero_copy():
    import torch

    from atom.model_ops.monokernel.config import Mxfp4ScaleLayout, Mxfp4WeightLayout
    from atom.model_ops.monokernel.weights import LayerWeights, prepare_mxfp4_expert_storage

    config = SimpleNamespace(
        name="glm-test",
        n_experts=2,
        num_shared_experts=1,
        hidden=128,
        inter=128,
        routed_hidden=None,
    )
    physical_experts = 3
    tensors = {
        "w_ug": torch.zeros(physical_experts * 2 * 128 * 128 // 2, dtype=torch.uint8),
        "s_ug": torch.zeros(768, 8, dtype=torch.uint8),
        "w_dn": torch.zeros(physical_experts * 128 * 128 // 2, dtype=torch.uint8),
        "s_dn": torch.zeros(512, 8, dtype=torch.uint8),
    }
    tensors["w_ug"].is_shuffled = True
    tensors["w_dn"].is_shuffled = True
    weights = LayerWeights(
        heads=1,
        t=tensors,
        config=config,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=physical_experts,
    )

    prepared = prepare_mxfp4_expert_storage(weights)

    for output, name in zip(prepared, ("w_ug", "s_ug", "w_dn", "s_dn")):
        assert output.data_ptr() == tensors[name].data_ptr()


def test_glm_fused_shared_expert_keeps_routed_count_separate():
    module = _glm_mono_module()
    experts = SimpleNamespace(
        global_num_experts=module.GLM5_CONFIG.n_experts,
        num_fused_shared_experts=module.GLM5_CONFIG.num_shared_experts,
    )

    assert module._physical_expert_count(experts) == 257
    experts.global_num_experts += 1
    with pytest.raises(module.MonoUnsupported, match="routed expert count"):
        module._physical_expert_count(experts)
