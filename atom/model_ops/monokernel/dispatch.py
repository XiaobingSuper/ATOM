# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Construction-time mode selection for native model MonoKernels."""

from __future__ import annotations

MODES = ("off", "auto", "mono", "staged")
SAMPLES = (4, 8)


class MonoUnsupported(Exception):
    """The loaded model or runtime state cannot use the native path."""


def tp_uniform_local_validation(
    error: Exception | None,
    *,
    group,
    world_size: int,
    context: str,
) -> None:
    """Make a local adapter validation result uniform across the TP group."""

    local = None if error is None else f"{type(error).__name__}: {error}"
    if world_size == 1:
        reports = [local]
    else:
        import torch.distributed as dist

        reports = [None] * world_size
        dist.all_gather_object(reports, local, group=group)
    failed = [(rank, report) for rank, report in enumerate(reports) if report is not None]
    if failed:
        detail = "; ".join(f"rank {rank}: {report}" for rank, report in failed)
        raise MonoUnsupported(f"{context}: {detail}")


def normalize_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in MODES:
        raise ValueError(f"MonoKernel mode must be one of {', '.join(MODES)}, got {mode!r}")
    return mode


def is_flat_atom_cache_page_size(page_size: int | None) -> bool:
    """Return whether MLA uses one physical cache row per token."""

    return page_size == 1


def select_backend(
    model: str,
    mode: str,
    *,
    samples: int,
    tp_size: int,
    kv_cache_dtype: str,
    native: bool = True,
    decode: bool = True,
    mtp: bool = False,
    dpa: bool = False,
    dcp: bool = False,
    plugin: bool = False,
    is_kda: bool = True,
    has_moe: bool = True,
    external_indexer: bool = True,
    cache_layout: str = "atom",
    segment: str = "layer",
) -> str | None:
    """Return a production backend name, or ``None`` for baseline fallback."""

    mode = normalize_mode(mode)
    if model == "glm52" and segment == "moe":
        if (
            mode == "staged"
            and native
            and decode
            and samples > 0
            and tp_size == 4
            and kv_cache_dtype == "fp8"
            and not dpa
            and not plugin
            and has_moe
            and external_indexer
            and cache_layout == "atom"
        ):
            return "staged_moe"
        return None
    if (
        mode == "off"
        or not native
        or not decode
        or samples not in SAMPLES
        or tp_size != 8
        or kv_cache_dtype not in ("bf16", "fp8")
        or mtp
        or dpa
        or plugin
    ):
        return None
    if model == "glm52":
        if (
            kv_cache_dtype == "bf16"
            and not dcp
            and has_moe
            and external_indexer
            and cache_layout == "atom"
            and mode == "mono"
        ):
            return "mono"
        return None
    if model == "kimi_k3" and is_kda and has_moe:
        if mode == "staged":
            return "staged"
        if mode == "mono":
            return "mono"
    return None
