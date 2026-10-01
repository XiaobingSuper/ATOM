# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-launch Kimi-K3 layer-0 KDA plus dense SiTU FFN."""

from __future__ import annotations

import torch

from atom.model_ops.monokernel.config import (
    KIMI_K3_CONFIG,
    ConvStateLayout,
)
from atom.model_ops.monokernel.k3.kda import KimiK3KdaAttention
from atom.model_ops.monokernel.k3.kernel import (
    build_kimi_k3_monokernel,
    monokernel_scratch_nbytes,
)
from atom.model_ops.monokernel.packing import pack_bf16
from atom.model_ops.monokernel.symmetric_allreduce import (
    SymmetricBf16Allreduce,
)
from atom.model_ops.monokernel.weights import LayerWeights


class KimiK3DenseMonoKernel:
    """Fuse layer-0 KDA, AttnRes, BF16 dense FFN, TP reduce and residual."""

    def __init__(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        layer_idx: int,
        rank: int,
        npes: int = 8,
        group=None,
        reduce_group=None,
        attention_symmetric_allreduce=None,
        moe_symmetric_allreduce=None,
        defer_collectives: bool = False,
        packed_artifacts: dict[str, object] | None = None,
        state_dtype: torch.dtype = torch.float16,
        agentic_batch_size: int = 0,
        **_ignored,
    ) -> None:
        if layer_idx != 0:
            raise ValueError("the dense Kimi full layer is layer 0 only")
        if (
            samples not in (8, 16, 32)
            or agentic_batch_size * 8 != samples
            or state_dtype is not torch.float16
        ):
            raise ValueError(
                "Kimi dense full layer requires DSpark q8 B1/B2/B4 "
                "with FP16 recurrent state"
            )
        if weights.config != KIMI_K3_CONFIG or rank != weights.rank:
            raise ValueError("Kimi dense full layer weight shard mismatch")
        self.W = weights
        self.t = weights.t
        self.config = weights.config
        self.S = samples
        self.rank = rank
        self.npes = npes
        self.layer_idx = layer_idx
        self.group = group
        self.block_write_idx = 0
        required = {
            "g_in",
            "g_post",
            "g_self_res",
            "w_self_res",
            "g_mlp_res",
            "w_mlp_res",
            "w_dense_ug",
            "w_dense_dn",
        }
        missing = sorted(required.difference(self.t))
        if missing:
            raise ValueError(
                "missing Kimi dense full weights: " + ", ".join(missing)
            )
        artifacts = packed_artifacts or {}
        self.attention = KimiK3KdaAttention(
            weights,
            samples,
            rank=rank,
            npes=npes,
            group=group,
            reduce_group=reduce_group,
            reduce_backend="symmetric",
            single_launch_attention=True,
            mtp=True,
            agentic_batch_size=agentic_batch_size,
            conv_state_layout=ConvStateLayout.TIME_MAJOR,
            symmetric_allreduce=attention_symmetric_allreduce,
            state_dtype=state_dtype,
            defer_collectives=defer_collectives,
            packed_artifacts=artifacts.get("attention"),
            monokernel_only=True,
        )
        device = self.t["w_dense_ug"].device
        self.pre_updated = torch.empty(
            samples, self.config.hidden, dtype=torch.bfloat16, device=device
        )
        self.pre_attn = torch.empty_like(self.pre_updated)
        self.updated_prefix = torch.empty_like(self.pre_updated)
        self.moe_input = torch.empty_like(self.pre_updated)
        self.output = torch.empty_like(self.pre_updated)
        self.attention_delta = torch.empty_like(self.pre_updated)
        # The post-AttnRes implementation publishes quantized MoE activation
        # as part of its fixed ABI. Dense BF16 ignores these two workspaces.
        self.quantized_moe_input = torch.empty(
            samples * self.config.hidden,
            dtype=torch.uint8,
            device=device,
        )
        self.quantized_moe_scale = torch.empty(
            samples,
            (self.config.hidden + 255) // 256,
            dtype=torch.uint8,
            device=device,
        )
        if artifacts:
            self.w_dense_ug = artifacts["w_dense_ug"]
            self.w_dense_dn = artifacts["w_dense_dn"]
        else:
            self.w_dense_ug = pack_bf16(self.t["w_dense_ug"])
            self.w_dense_dn = pack_bf16(self.t["w_dense_dn"])
        self.attention._pack_monokernel_projections()
        self.attention.attn_res_blocks = 0
        self.attention.block_write_idx = 0
        self.attention.fuse_attn_res = True
        self.attention.fuse_moe = True
        self.attention.moe_packed = {
            "w_shared_ug": self.w_dense_ug,
            "w_shared_dn": self.w_dense_dn,
        }
        self.attention.monokernel_scratch = torch.zeros(
            monokernel_scratch_nbytes(
                samples,
                fuse_attn_res=True,
                fuse_moe=True,
                mtp=True,
                dense_ffn=True,
            ),
            dtype=torch.uint8,
            device=device,
        )
        self.attention.monokernel_timeline = torch.empty(
            10, dtype=torch.int64, device=device
        )
        self.attention.monokernel_launch = build_kimi_k3_monokernel(
            samples,
            npes,
            attn_res_blocks=0,
            block_write_idx=0,
            fuse_moe=True,
            mtp=True,
            agentic_batch_size=agentic_batch_size,
            state_dtype=state_dtype,
            conv_state_layout=ConvStateLayout.TIME_MAJOR,
            dense_ffn=True,
        )
        self.step = self.attention.step
        self.symmetric_allreduce = moe_symmetric_allreduce
        if self.symmetric_allreduce is None and not defer_collectives:
            self.symmetric_allreduce = SymmetricBf16Allreduce(
                (self.output.numel(), self.output.numel()),
                rank=rank,
                npes=npes,
                group=group,
            )

    def initialize_collectives(
        self,
        attention=None,
        moe=None,
    ):
        attention = self.attention.initialize_symmetric_allreduce(attention)
        if self.symmetric_allreduce is None:
            self.symmetric_allreduce = (
                moe
                if moe is not None
                else SymmetricBf16Allreduce(
                    (self.output.numel(), self.output.numel()),
                    rank=self.rank,
                    npes=self.npes,
                    group=self.group,
                )
            )
        elif moe is not None and self.symmetric_allreduce is not moe:
            raise ValueError("dense FFN all-reduce arena mismatch")
        return attention, self.symmetric_allreduce

    def packed_artifacts(self) -> dict[str, object]:
        return {
            "attention": self.attention.packed_artifacts(),
            "w_dense_ug": self.w_dense_ug,
            "w_dense_dn": self.w_dense_dn,
        }

    def full_plan_workspace_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.pre_updated,
            self.pre_attn,
            self.updated_prefix,
            self.moe_input,
            self.output,
            self.attention_delta,
            self.quantized_moe_input,
            self.quantized_moe_scale,
            *self.attention.full_plan_workspace_tensors(),
        )

    def release_packed_sources(self) -> None:
        self.attention.release_packed_sources()
        keep = {
            "g_in",
            "g_post",
            "g_self_res",
            "w_self_res",
            "g_mlp_res",
            "w_mlp_res",
        }
        self.t = {name: value for name, value in self.t.items() if name in keep}
        self.W = None

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        state_indices: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        x_out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        epoch_layer: int = 0,
        advance: bool = True,
    ) -> torch.Tensor:
        if block_residual.shape[:1] != (self.S,) or block_residual.shape[1] < 1:
            raise ValueError("layer 0 requires block-residual slot 0")
        if self.symmetric_allreduce is None:
            raise ValueError("dense FFN collective is not initialized")
        target = self.output if x_out is None else x_out
        self.attention.forward(
            prefix_sum,
            state_indices,
            conv_state,
            recurrent_state,
            x_out=self.attention_delta,
            num_accepted_tokens=num_accepted_tokens,
            block_residual=block_residual,
            pre_updated=self.pre_updated,
            pre_output=self.pre_attn,
            updated_prefix=self.updated_prefix,
            moe_input=self.moe_input,
            quantized_moe_input=self.quantized_moe_input,
            quantized_moe_scale=self.quantized_moe_scale,
            monokernel_output=target,
            moe_symmetric=self.symmetric_allreduce.peer_buffer.local_address,
            moe_peers=self.symmetric_allreduce.peer_buffer.addresses,
            layer=epoch_layer,
            advance=advance,
        )
        return target


__all__ = ["KimiK3DenseMonoKernel"]
