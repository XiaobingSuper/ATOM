# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host wrapper for the single-launch Kimi-K3 decode MonoKernel."""

from __future__ import annotations

import torch

from atom.model_ops.monokernel.config import ConvStateLayout
from atom.model_ops.monokernel.k3.staged import _KimiK3KdaStagedPath
from atom.model_ops.monokernel.symmetric_allreduce import SymmetricBf16Allreduce
from atom.model_ops.monokernel.weights import LayerWeights


class KimiK3MonoKernel(_KimiK3KdaStagedPath):
    """Run KDA, AttnRes, latent-MoE, TP reductions, and residual update in one launch."""

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
        mtp: bool = False,
        agentic_batch_size: int = 0,
        conv_state_layout: ConvStateLayout = ConvStateLayout.CHANNEL_MAJOR,
        attention_symmetric_allreduce: SymmetricBf16Allreduce | None = None,
        moe_symmetric_allreduce: SymmetricBf16Allreduce | None = None,
        state_dtype: torch.dtype = torch.float32,
        defer_collectives: bool = False,
        packed_artifacts: dict[str, object] | None = None,
    ) -> None:
        if agentic_batch_size:
            if (
                not mtp
                or samples != agentic_batch_size * 8
                or state_dtype is not torch.float16
            ):
                raise ValueError(
                    "Kimi Agentic MonoKernel requires q=8 MTP rows "
                    "and FP16 recurrent state"
                )
        elif state_dtype is not torch.float32:
            raise ValueError("Kimi-K3 single-launch MonoKernel requires FP32 state")
        super().__init__(
            weights,
            samples,
            layer_idx=layer_idx,
            rank=rank,
            npes=npes,
            group=group,
            reduce_group=reduce_group,
            fuse_attn_res=True,
            fuse_router=True,
            fuse_shared_experts=True,
            reduce_backend="symmetric",
            mtp=mtp,
            agentic_batch_size=agentic_batch_size,
            conv_state_layout=conv_state_layout,
            attention_symmetric_allreduce=attention_symmetric_allreduce,
            moe_symmetric_allreduce=moe_symmetric_allreduce,
            state_dtype=state_dtype,
            defer_collectives=defer_collectives,
            packed_artifacts=packed_artifacts,
            monokernel_only=True,
        )
        self.attention.configure_monokernel(
            layer_idx,
            fuse_moe=True,
        )
        canonical_moe = getattr(self.attention, "moe_packed", None)
        if canonical_moe is not None:
            self.moe_packed = canonical_moe
            self.w_router = canonical_moe["w_r"]
            self.w_latent_down = canonical_moe["w_latent_down"]
            self.s_latent_down = canonical_moe["s_latent_down"]
            self.w_shared_ug = canonical_moe["w_shared_ug"]
            self.s_shared_ug = canonical_moe["s_shared_ug"]
            self.w_ug = canonical_moe["w_ug"]
            self.s_ug = canonical_moe["s_ug"]
            self.w_dn = canonical_moe["w_dn"]
            self.s_dn = canonical_moe["s_dn"]
            self.w_shared_dn = canonical_moe["w_shared_dn"]
            self.s_shared_dn = canonical_moe["s_shared_dn"]
            self.w_latent_up = canonical_moe["w_latent_up"]
            self.s_latent_up = canonical_moe["s_latent_up"]
            self.latent_projection.weight = self.w_latent_down
            self.latent_projection.scale = self.s_latent_down

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
        """Run one complete Kimi-K3 decode layer."""

        if (
            block_residual.ndim != 3
            or block_residual.shape[0] != self.S
            or block_residual.shape[2] != self.config.hidden
        ):
            raise ValueError(
                "block_residual must have shape "
                f"[{self.S}, blocks, {self.config.hidden}], got {tuple(block_residual.shape)}"
            )
        if block_residual.shape[1] <= self.block_write_idx:
            raise ValueError(
                f"block_residual needs index {self.block_write_idx}, " f"got {block_residual.shape[1]} blocks"
            )

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
            quantized_moe_input=self.latent_projection.activation,
            quantized_moe_scale=self.latent_projection.activation_scale,
            monokernel_output=target,
            moe_symmetric=self.symmetric_allreduce.peer_buffer.local_address,
            moe_peers=self.symmetric_allreduce.peer_buffer.addresses,
            layer=epoch_layer,
            advance=advance,
        )
        return target


__all__ = ["KimiK3MonoKernel"]
