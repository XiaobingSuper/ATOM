# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared-mono exact-width lifecycle for GLM layer launchers."""

from dataclasses import dataclass

import torch

from atom.models.glm52.mono.config import target_width
from atom.models.glm52.mono.kernel import ABI, LayerBuild, build_layer
from atom.models.glm52.mono.runtime import LayerRuntime
from atom.mono.runtime.widths import WidthBuilds


@dataclass(frozen=True)
class WidthLayers:
    full: object
    shared: object


class LayerKernels:
    """Compile and hold one full and one IndexShare launcher per exact width."""

    def __init__(
        self,
        group,
        rank: int,
        *,
        mtp_speculative_tokens: int,
        index_max_seq: int,
        timeline: bool = False,
    ) -> None:
        self.mtp_speculative_tokens = mtp_speculative_tokens
        self.index_max_seq = index_max_seq
        self.timeline = timeline
        self.runtime = LayerRuntime(
            group,
            rank,
            topk=2048,
            index_max_seq=index_max_seq,
        )
        self.widths = WidthBuilds(
            self._build_width,
            self._kernels,
            group,
            "GLM-5.2 batch mono",
        )

    def _key(self, rows: int, with_indexer: bool) -> LayerBuild:
        return LayerBuild(
            tokens=target_width(rows),
            with_indexer=with_indexer,
            mtp_speculative_tokens=self.mtp_speculative_tokens,
            index_max_seq=self.index_max_seq,
            timeline=self.timeline,
        )

    def _build_width(self, rows: int) -> WidthLayers:
        return WidthLayers(
            full=build_layer(self._key(rows, True)),
            shared=build_layer(self._key(rows, False)),
        )

    @staticmethod
    def _kernels(build: WidthLayers) -> list:
        return [(build.full, ABI), (build.shared, ABI)]

    def prepare(self, rows: int) -> bool:
        return self.widths.prepare(target_width(rows))

    def begin_step(self) -> None:
        self.runtime.begin_step()

    def finish_step(self) -> None:
        self.runtime.finish_step()

    def launch(
        self,
        rows: int,
        *,
        with_indexer: bool,
        arguments: dict,
    ) -> None:
        """Submit exactly one whole-layer launch for ``rows``."""

        build = self.widths[target_width(rows)]
        launcher = build.full if with_indexer else build.shared
        values = {**arguments, **self.runtime.kernel_args()}
        if not with_indexer:
            values.setdefault("indices", self.runtime.selection_ptr(rows))
        launcher(*ABI.pack(values), stream=torch.cuda.current_stream())

    def close(self) -> None:
        self.runtime.close()
