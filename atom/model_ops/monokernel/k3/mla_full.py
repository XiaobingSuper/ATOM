# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the one-launch TP8 dense Kimi-K3 MLA layer."""

from __future__ import annotations

import torch

from atom.model_ops.monokernel.k3.abi import KimiMlaAgenticRuntime
from atom.model_ops.monokernel.k3.mla_full_kernel import (
    build_kimi_k3_mla_full_monokernel,
    mla_full_layout,
)
from atom.model_ops.monokernel.k3.staged import _KimiK3MlaPath
from atom.model_ops.monokernel.packing import pack_bf16
from atom.model_ops.monokernel.symmetric_allreduce import SymmetricBf16Allreduce
from atom.model_ops.monokernel.weights import LayerWeights


class _MlaAttentionArena:
    """Attention TP mailbox and epoch shared by every MLA layer bucket."""

    def __init__(
        self,
        rows: int,
        hidden: int,
        *,
        rank: int,
        npes: int,
        group,
        shared: SymmetricBf16Allreduce | None,
        defer: bool,
        device: torch.device,
    ) -> None:
        self.rows = rows
        self.hidden = hidden
        self.rank = rank
        self.npes = npes
        self.group = group
        self.step = torch.zeros(1, dtype=torch.int32, device=device)
        self.symmetric_allreduce = shared
        if self.symmetric_allreduce is None and not defer:
            self.symmetric_allreduce = SymmetricBf16Allreduce(
                (rows * hidden,),
                rank=rank,
                npes=npes,
                group=group,
            )

    def initialize_symmetric_allreduce(
        self,
        shared: SymmetricBf16Allreduce | None = None,
    ) -> SymmetricBf16Allreduce:
        if self.symmetric_allreduce is None:
            self.symmetric_allreduce = (
                shared
                if shared is not None
                else SymmetricBf16Allreduce(
                    (self.rows * self.hidden,),
                    rank=self.rank,
                    npes=self.npes,
                    group=self.group,
                )
            )
        elif shared is not None and self.symmetric_allreduce is not shared:
            raise ValueError("MLA attention all-reduce arena mismatch")
        return self.symmetric_allreduce

    def packed_artifacts(self) -> dict[str, torch.Tensor]:
        return {}

    def full_plan_workspace_tensors(self) -> tuple[torch.Tensor, ...]:
        return ()

    def release_packed_sources(self) -> None:
        return None

    def close(self) -> None:
        if self.symmetric_allreduce is not None:
            self.symmetric_allreduce.close()


class KimiK3MlaMonoKernel(_KimiK3MlaPath):
    """AttnRes, dense physical MLA and current latent-MoE in one launch."""

    def _build_attention(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        rank: int,
        npes: int,
        group,
        reduce_group,
        topk: int,
        reduce_backend: str,
        kv_cache_layout,
        mtp: bool,
        agentic_batch_size: int,
        conv_state_layout,
        attention_symmetric_allreduce,
        state_dtype: torch.dtype,
        defer_collectives: bool,
        packed_artifacts,
        monokernel_only: bool,
    ) -> _MlaAttentionArena:
        del (
            reduce_group,
            topk,
            reduce_backend,
            kv_cache_layout,
            mtp,
            agentic_batch_size,
            conv_state_layout,
            state_dtype,
            packed_artifacts,
            monokernel_only,
        )
        return _MlaAttentionArena(
            samples,
            weights.config.hidden,
            rank=rank,
            npes=npes,
            group=group,
            shared=attention_symmetric_allreduce,
            defer=defer_collectives,
            device=weights.t["w_qkv_a"].device,
        )

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
    ) -> None:
        if samples not in (8, 16, 32):
            raise ValueError("Kimi full MLA supports q8 B1/B2/B4 only")
        packed_artifacts = packed_artifacts or {}
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
            attention_symmetric_allreduce=attention_symmetric_allreduce,
            moe_symmetric_allreduce=moe_symmetric_allreduce,
            defer_collectives=defer_collectives,
            packed_artifacts=packed_artifacts,
            monokernel_only=True,
        )
        required = {
            "g_in",
            "g_post",
            "w_qkv_a",
            "g_q",
            "g_kv",
            "w_q_b",
            "w_uk",
            "w_uv",
            "w_o",
            "w_gate",
        }
        missing = sorted(required.difference(self.t))
        if missing:
            raise ValueError(
                "missing Kimi full-MLA weights: " + ", ".join(missing)
            )
        self.step = self.attention.step
        self.w_o_packed = packed_artifacts.get("w_o_packed")
        if self.w_o_packed is None:
            self.w_o_packed = pack_bf16(self.t["w_o"])
        self.moe_packed = {
            "w_r": self.w_router,
            "w_latent_down": self.w_latent_down,
            "s_latent_down": self.s_latent_down,
            "w_shared_ug": self.w_shared_ug,
            "s_shared_ug": self.s_shared_ug,
            "w_ug": self.w_ug,
            "s_ug": self.s_ug,
            "w_dn": self.w_dn,
            "s_dn": self.s_dn,
            "w_shared_dn": self.w_shared_dn,
            "s_shared_dn": self.s_shared_dn,
            "w_latent_up": self.w_latent_up,
            "s_latent_up": self.s_latent_up,
        }
        layout = mla_full_layout(samples)
        device = self.t["w_qkv_a"].device
        self.monokernel_scratch = torch.zeros(
            layout["_bytes"],
            dtype=torch.uint8,
            device=device,
        )
        self.monokernel_timeline = torch.empty(
            10,
            dtype=torch.int64,
            device=device,
        )
        self.monokernel_launch = build_kimi_k3_mla_full_monokernel(
            samples,
            npes=npes,
            attn_res_blocks=self.previous_valid_blocks,
            block_write_idx=(
                self.block_write_idx if self.is_block_write_layer else -1
            ),
            atom_expert_layout=True,
        )

    def packed_artifacts(self) -> dict[str, object]:
        return {
            **super().packed_artifacts(),
            "w_o_packed": self.w_o_packed,
        }

    def full_plan_workspace_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            *super().full_plan_workspace_tensors(),
            self.monokernel_timeline,
        )

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        runtime: KimiMlaAgenticRuntime,
        main_cache: torch.Tensor,
        main_scale: torch.Tensor,
        *,
        x_out: torch.Tensor | None = None,
        epoch_layer: int = 0,
        advance: bool = True,
    ) -> torch.Tensor:
        from atom.model_ops.monokernel.k3.mla_cache import (
            validate_fp8_mla_cache,
        )

        if runtime.shape.common.row_capacity != self.S:
            raise ValueError("MLA runtime row capacity does not match the bucket")
        validate_fp8_mla_cache(main_cache, main_scale)
        if self.attention.symmetric_allreduce is None:
            raise ValueError("MLA attention collective is not initialized")
        if self.symmetric_allreduce is None:
            raise ValueError("MLA MoE collective is not initialized")
        target = self.output if x_out is None else x_out
        packed = self.moe_packed
        block_stride = (
            block_residual.shape[1]
            if block_residual.ndim == 3
            else 1
        )
        dummy = prefix_sum
        self.monokernel_launch(
            prefix_sum.data_ptr(),
            self.attention_delta.data_ptr(),
            block_residual.data_ptr(),
            self.t["g_self_res"].data_ptr(),
            self.t["w_self_res"].data_ptr(),
            self.t["g_in"].data_ptr(),
            self.t["g_mlp_res"].data_ptr(),
            self.t["w_mlp_res"].data_ptr(),
            self.t["g_post"].data_ptr(),
            self.pre_updated.data_ptr(),
            self.pre_attn.data_ptr(),
            self.updated_prefix.data_ptr(),
            self.moe_input.data_ptr(),
            self.latent_projection.activation.data_ptr(),
            self.latent_projection.activation_scale.data_ptr(),
            block_stride,
            packed["w_r"].data_ptr(),
            self.t["bias"].data_ptr(),
            packed["w_latent_down"].data_ptr(),
            packed["s_latent_down"].data_ptr(),
            packed["w_shared_ug"].data_ptr(),
            packed["s_shared_ug"].data_ptr(),
            packed["w_ug"].data_ptr(),
            packed["s_ug"].data_ptr(),
            packed["w_dn"].data_ptr(),
            packed["s_dn"].data_ptr(),
            self.t["g_latent"].data_ptr(),
            packed["w_shared_dn"].data_ptr(),
            packed["s_shared_dn"].data_ptr(),
            packed["w_latent_up"].data_ptr(),
            packed["s_latent_up"].data_ptr(),
            self.symmetric_allreduce.peer_buffer.local_address,
            self.symmetric_allreduce.peer_buffer.addresses.data_ptr(),
            target.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            self.w_o_packed.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            dummy.data_ptr(),
            self.monokernel_scratch.data_ptr(),
            self.attention.symmetric_allreduce.peer_buffer.local_address,
            self.attention.symmetric_allreduce.peer_buffer.addresses.data_ptr(),
            self.step.data_ptr(),
            self.monokernel_timeline.data_ptr(),
            runtime.positions.data_ptr(),
            runtime.slot_mapping.data_ptr(),
            runtime.batch_ids.data_ptr(),
            runtime.context_lens.data_ptr(),
            runtime.block_tables.data_ptr(),
            runtime.block_tables.stride(0),
            runtime.block_size,
            runtime.block_ratio,
            main_cache.data_ptr(),
            main_scale.data_ptr(),
            self.t["w_qkv_a"].data_ptr(),
            self.t["g_q"].data_ptr(),
            self.t["g_kv"].data_ptr(),
            self.t["w_q_b"].data_ptr(),
            self.t["w_uk"].data_ptr(),
            self.t["w_uv"].data_ptr(),
            self.t["w_gate"].data_ptr(),
            self.rank,
            epoch_layer,
            int(advance),
            stream=torch.cuda.current_stream(),
        )
        return target

    def release_packed_sources(self) -> None:
        """Retain only tensors dereferenced by the fused MLA launch."""

        keep = {
            "bias",
            "g_in",
            "g_kv",
            "g_latent",
            "g_mlp_res",
            "g_post",
            "g_q",
            "g_self_res",
            "w_gate",
            "w_mlp_res",
            "w_q_b",
            "w_qkv_a",
            "w_self_res",
            "w_uk",
            "w_uv",
        }
        self.t = {
            name: value for name, value in self.t.items() if name in keep
        }
        self.W = None


__all__ = ["KimiK3MlaMonoKernel"]
