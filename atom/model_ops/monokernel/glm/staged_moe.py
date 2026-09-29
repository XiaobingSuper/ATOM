# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""TP4 staged MXFP4 MoE for GLM-5.2 Agentic decode."""

from __future__ import annotations

from types import MethodType

import torch
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
)
from aiter.ops.flydsl.kernels.moe_sorting_kernel import moe_sorting_flydsl

from atom.config import get_current_atom_config
from atom.model_ops.monokernel.config import glm5_shard_config
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    prepare_mxfp4_expert_storage,
)
from atom.utils import mark_spliting_op

_TP_SIZE = 4
_PHYSICAL_EXPERTS = 257
_ROUTED_TOPK = 8
_TOPK = _ROUTED_TOPK + 1
_SORT_TILE = 16


def _glm52_staged_moe_fake(
    hidden_states: torch.Tensor,
    layer_key: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


@mark_spliting_op(
    is_custom=True,
    gen_fake=_glm52_staged_moe_fake,
    mutates_args=[],
)
def glm52_staged_moe(
    hidden_states: torch.Tensor,
    layer_key: str,
) -> torch.Tensor:
    binding = get_current_atom_config().compilation_config.static_forward_context[
        layer_key
    ]
    return binding(hidden_states)


def _staged_moe_forward(module, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.ops.aiter.glm52_staged_moe(
        hidden_states,
        module._glm52_staged_key,
    )


def install_staged_moe_forward(module, key: str):
    """Route one existing MoE through the opaque staged op without reparenting it."""

    original = module.forward
    module._glm52_staged_key = key
    module.forward = MethodType(_staged_moe_forward, module)
    return original


def workspace_shapes(rows: int) -> dict[str, tuple[int, ...]]:
    """Return graph-stable workspace shapes for flattened decode rows."""

    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    config = glm5_shard_config(_TP_SIZE)
    max_sorted = rows * _TOPK * _SORT_TILE
    return {
        "routes": (rows, _TOPK),
        "sorted": (max_sorted,),
        "intermediate": (max_sorted, config.inter),
        "output": (rows, config.hidden),
    }


class Glm52MoeWorkspace:
    """One workspace shared by all GLM MoE layers with the same row count."""

    def __init__(self, rows: int, device: torch.device) -> None:
        config = glm5_shard_config(_TP_SIZE)
        shapes = workspace_shapes(rows)
        self.rows = rows
        self.max_sorted = shapes["sorted"][0]
        self.router_scores = torch.empty(
            rows, config.n_experts, dtype=torch.float32, device=device
        )
        self.corrected_scores = torch.empty_like(self.router_scores)
        self.topk_keys = torch.empty(
            rows, _ROUTED_TOPK, dtype=torch.float32, device=device
        )
        self.topk_ids_i64 = torch.empty(
            rows, _ROUTED_TOPK, dtype=torch.int64, device=device
        )
        self.route_ids = torch.empty(
            shapes["routes"], dtype=torch.int32, device=device
        )
        self.route_weights = torch.empty(
            shapes["routes"], dtype=torch.float32, device=device
        )
        self.sorted_token_ids = torch.empty(
            shapes["sorted"], dtype=torch.int32, device=device
        )
        self.sorted_weights = torch.empty(
            shapes["sorted"], dtype=torch.float32, device=device
        )
        self.sorted_expert_ids = torch.empty(
            self.max_sorted // _SORT_TILE, dtype=torch.int32, device=device
        )
        self.num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
        self.intermediate = torch.empty(
            shapes["intermediate"], dtype=torch.bfloat16, device=device
        )
        self.partial = torch.empty(
            shapes["output"], dtype=torch.bfloat16, device=device
        )


class Glm52Tp4MoeStage:
    """Router, fused shared/routed MXFP4 experts, and optional TP reduction."""

    def __init__(
        self,
        weights: LayerWeights,
        gate,
        correction_bias: torch.Tensor,
        workspace: Glm52MoeWorkspace,
        *,
        reduce_results: bool,
    ) -> None:
        config = glm5_shard_config(_TP_SIZE)
        if weights.config != config or weights.physical_experts != _PHYSICAL_EXPERTS:
            raise ValueError("GLM-5.2 staged MoE requires TP4 with 257 physical experts")
        if correction_bias.shape != (config.n_experts,) or correction_bias.dtype is not torch.float32:
            raise ValueError("GLM-5.2 router correction bias must be contiguous FP32 [256]")
        if not correction_bias.is_contiguous():
            raise ValueError("GLM-5.2 router correction bias must be contiguous")
        self.config = config
        self.gate = gate
        self.bias = correction_bias
        self.workspace = workspace
        self.reduce_results = reduce_results
        self.w_ug, self.s_ug, self.w_dn, self.s_dn = prepare_mxfp4_expert_storage(
            weights
        )

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = self.workspace
        expected = (w.rows, self.config.hidden)
        if (
            hidden_states.shape != expected
            or hidden_states.dtype is not torch.bfloat16
            or not hidden_states.is_contiguous()
        ):
            raise ValueError(
                f"GLM-5.2 staged MoE input must be contiguous BF16 {list(expected)}"
            )

        logits = self.gate(hidden_states, otype=torch.float32)
        torch.sigmoid(logits, out=w.router_scores)
        torch.add(w.router_scores, self.bias, out=w.corrected_scores)
        torch.topk(
            w.corrected_scores,
            _ROUTED_TOPK,
            dim=-1,
            sorted=True,
            out=(w.topk_keys, w.topk_ids_i64),
        )
        w.route_ids[:, :_ROUTED_TOPK].copy_(w.topk_ids_i64)
        torch.gather(
            w.router_scores,
            1,
            w.topk_ids_i64,
            out=w.route_weights[:, :_ROUTED_TOPK],
        )
        routed = w.route_weights[:, :_ROUTED_TOPK]
        routed.div_(routed.sum(-1, keepdim=True)).mul_(self.config.route_scale)
        w.route_ids[:, _ROUTED_TOPK].fill_(self.config.n_experts)
        w.route_weights[:, _ROUTED_TOPK].fill_(1.0)

        moe_sorting_flydsl(
            w.route_ids,
            w.route_weights,
            w.sorted_token_ids,
            w.sorted_weights,
            w.sorted_expert_ids,
            w.num_valid_ids,
            w.partial,
            _PHYSICAL_EXPERTS,
            unit_size=_SORT_TILE,
            num_local_tokens=w.rows,
        )
        flydsl_a16w4_gemm1(
            a_bf16=hidden_states,
            w1_u8=self.w_ug,
            w1_scale_u8=self.s_ug,
            sorted_expert_ids=w.sorted_expert_ids,
            cumsum_tensor=w.num_valid_ids,
            m_indices=w.sorted_token_ids,
            inter_sorted_bf16=w.intermediate,
            n_tokens=w.rows,
            NE=_PHYSICAL_EXPERTS,
            D_HIDDEN=self.config.hidden,
            D_INTER=self.config.inter,
            topk=_TOPK,
            tile_m=_SORT_TILE,
            tile_n=128,
            tile_k=256,
            k_wave=2,
            b_nt=2,
            xcd_swizzle=0,
            gate_mode="separated",
            act="silu",
            w_dtype="fp4",
            w_layout="standard",
            stream=torch.cuda.current_stream(),
        )
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=w.intermediate,
            w2_u8=self.w_dn,
            w2_scale_u8=self.s_dn,
            sorted_expert_ids=w.sorted_expert_ids,
            cumsum_tensor=w.num_valid_ids,
            sorted_token_ids=w.sorted_token_ids,
            sorted_weights=w.sorted_weights,
            flat_out=w.partial,
            M_logical=w.rows,
            max_sorted=w.max_sorted,
            NE=_PHYSICAL_EXPERTS,
            D_HIDDEN=self.config.hidden,
            D_INTER=self.config.inter,
            topk=_TOPK,
            tile_m=_SORT_TILE,
            tile_n=128,
            tile_k=128,
            b_nt=2,
            xcd_swizzle=0,
            w_dtype="fp4",
            persist=False,
            epilog="atomic",
            stream=torch.cuda.current_stream(),
        )
        if self.reduce_results:
            return tensor_model_parallel_all_reduce(w.partial)
        return w.partial


__all__ = [
    "Glm52MoeWorkspace",
    "Glm52Tp4MoeStage",
    "glm52_staged_moe",
    "install_staged_moe_forward",
    "workspace_shapes",
]
