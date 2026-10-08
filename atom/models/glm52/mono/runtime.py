# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared peer memory and mailboxes for exact-width GLM layer launches."""

import torch

from atom.model_ops.monokernel.glm.layout import layout
from atom.models.glm52.mono.config import LEGACY_CORE_WIDTHS, TP
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes


class LayerRuntime:
    """Own one scratch arena and peer buffer shared by every layer and width."""

    def __init__(
        self,
        group,
        rank: int,
        *,
        topk: int,
        index_max_seq: int,
        debug: bool = False,
    ) -> None:
        from atom.model_ops.monokernel.config import glm5_tp_config

        cfg = glm5_tp_config(TP)
        layouts = [
            layout(
                rows,
                cfg.local_heads,
                TP,
                topk,
                with_indexer,
                index_max_seq,
                inter=cfg.inter,
                output_heads=cfg.local_heads,
                dcp_size=1,
                native_fp4_mfma=True,
            )
            for rows in LEGACY_CORE_WIDTHS
            for with_indexer in (False, True)
        ]
        device = torch.device("cuda", torch.cuda.current_device())
        self.scratch = torch.zeros(
            max(scratch["_bytes"] for scratch, _ in layouts),
            dtype=torch.uint8,
            device=device,
        )
        peer_bytes = max(symmetric["_bytes"] for _, symmetric in layouts)
        self.peers, self._finalizer = owned_peer_buffer(
            self,
            peer_bytes,
            group,
            rank,
            TP,
            device,
        )
        self.mailboxes = StepMailboxes(self.peers, self.scratch, debug=debug)
        self.step = torch.zeros(1, dtype=torch.int32, device=device)
        self.rank = rank
        self.topk = topk
        self.index_max_seq = index_max_seq
        self.heads = cfg.local_heads
        self.inter = cfg.inter

    def begin_step(self) -> None:
        self.mailboxes.begin_step()

    def finish_step(self) -> None:
        self.step.add_(1)

    def kernel_args(self) -> dict[str, int]:
        return {
            "scratch": self.scratch.data_ptr(),
            **self.peers.kernel_args(),
            "step": self.step.data_ptr(),
        }

    def selection_ptr(self, rows: int) -> int:
        scratch, _ = layout(
            rows,
            self.heads,
            TP,
            self.topk,
            True,
            self.index_max_seq,
            inter=self.inter,
            output_heads=self.heads,
            dcp_size=1,
            native_fp4_mfma=True,
        )
        return self.scratch.data_ptr() + scratch["indices"]

    def close(self) -> None:
        self._finalizer()
