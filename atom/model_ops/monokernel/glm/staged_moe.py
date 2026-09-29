# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""TP4 staged MXFP4 MoE for GLM-5.2 Agentic decode."""

from __future__ import annotations

from types import MethodType

import torch
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.fused_moe import fused_moe
from aiter.ops.topk import biased_grouped_topk_hip

from atom.config import get_current_atom_config
from atom.model_ops.monokernel.config import glm5_shard_config
from atom.model_ops.monokernel.weights import (
    LayerWeights,
)
from atom.utils import mark_spliting_op

_TP_SIZE = 4
_PHYSICAL_EXPERTS = 257
_ROUTED_TOPK = 8
_TOPK = _ROUTED_TOPK + 1


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
    return {
        "routes": (rows, _TOPK),
        "output": (rows, config.hidden),
    }


class Glm52MoeWorkspace:
    """One workspace shared by all GLM MoE layers with the same row count."""

    def __init__(self, rows: int, device: torch.device) -> None:
        config = glm5_shard_config(_TP_SIZE)
        shapes = workspace_shapes(rows)
        self.rows = rows
        self.route_ids = torch.empty(
            shapes["routes"], dtype=torch.int32, device=device
        )
        self.route_weights = torch.empty(
            shapes["routes"], dtype=torch.float32, device=device
        )
        self.route_ids[:, _ROUTED_TOPK].fill_(_PHYSICAL_EXPERTS - 1)
        self.route_weights[:, _ROUTED_TOPK].fill_(1.0)
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
        experts,
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
        self.experts = experts
        self.reduce_results = reduce_results

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
        biased_grouped_topk_hip(
            logits,
            self.bias,
            w.route_weights[:, :_ROUTED_TOPK],
            w.route_ids[:, :_ROUTED_TOPK],
            1,
            1,
            True,
            self.config.route_scale,
        )

        method = self.experts.quant_method
        output = fused_moe(
            hidden_states,
            self.experts.w13_weight,
            self.experts.w2_weight,
            w.route_weights,
            w.route_ids,
            expert_mask=self.experts.expert_mask,
            activation=self.experts.activation,
            quant_type=method.quant_type,
            w1_scale=self.experts.w13_weight_scale,
            w2_scale=self.experts.w2_weight_scale,
            a1_scale=getattr(self.experts, "w13_input_scale", None),
            a2_scale=getattr(self.experts, "w2_input_scale", None),
            doweight_stage1=self.experts.apply_router_weight_on_input,
            bias1=self.experts.w13_bias,
            bias2=self.experts.w2_bias,
            swiglu_limit=getattr(self.experts, "swiglu_limit", 0.0),
            gate_mode="separated",
            output=w.partial,
        )
        if self.reduce_results:
            return tensor_model_parallel_all_reduce(output)
        return output


__all__ = [
    "Glm52MoeWorkspace",
    "Glm52Tp4MoeStage",
    "glm52_staged_moe",
    "install_staged_moe_forward",
    "workspace_shapes",
]
