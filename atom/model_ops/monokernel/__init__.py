# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared implementation details for model-specific MonoKernels.

Device sources track FlyDSL PR #1204 at 21a3d1ee; model adapters live in ATOM.
"""

from __future__ import annotations

from weakref import WeakSet

_closed_model_monokernels: WeakSet = WeakSet()


def _model_monokernel_closed(owned: object) -> bool:
    try:
        return owned in _closed_model_monokernels
    except TypeError:
        return bool(getattr(owned, "_atom_monokernel_closed", False))


def _mark_model_monokernel_closed(owned: object) -> None:
    try:
        _closed_model_monokernels.add(owned)
    except TypeError:
        setattr(owned, "_atom_monokernel_closed", True)


def _owned_model_monokernels(model):
    modules = model.modules() if callable(getattr(model, "modules", None)) else (model,)
    seen: set[int] = set()
    for module in modules:
        for name in ("_mono", "_glm52_mono"):
            owned = getattr(module, name, None)
            if owned is None or id(owned) in seen:
                continue
            seen.add(id(owned))
            yield owned


def model_monokernel_memory_reserve(model) -> int:
    """Return bytes that lazy native runners need beyond loaded model weights."""

    total = 0
    for owned in _owned_model_monokernels(model):
        reserve = getattr(owned, "memory_reserve_bytes", None)
        if callable(reserve):
            total += max(int(reserve()), 0)
    return total


def prepare_model_monokernels_for_capture(model) -> bool:
    """Prepare every enabled lazy native plan before CUDA graph capture."""

    prepared = True
    for owned in _owned_model_monokernels(model):
        prepare = getattr(owned, "prepare_for_capture", None)
        if callable(prepare):
            prepared = bool(prepare()) and prepared
    return prepared


def close_model_monokernels(model) -> None:
    """Close each model-owned native runner once before distributed teardown."""

    for owned in _owned_model_monokernels(model):
        if _model_monokernel_closed(owned):
            continue
        close = getattr(owned, "close", None)
        if callable(close):
            close()
        _mark_model_monokernel_closed(owned)


__all__ = [
    "close_model_monokernels",
    "model_monokernel_memory_reserve",
    "prepare_model_monokernels_for_capture",
]
