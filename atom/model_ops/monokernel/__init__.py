# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared implementation details for model-specific MonoKernels.

Device sources track FlyDSL PR #1204 at 21a3d1ee; model adapters live in ATOM.
"""

from __future__ import annotations

from weakref import WeakSet

_closed_model_monokernels: WeakSet = WeakSet()


def close_model_monokernels(model) -> None:
    """Close each model-owned native runner once before distributed teardown."""

    modules = model.modules() if callable(getattr(model, "modules", None)) else (model,)
    seen: set[int] = set()
    for module in modules:
        owned = getattr(module, "_mono", None)
        if owned is None or id(owned) in seen:
            continue
        seen.add(id(owned))
        try:
            if owned in _closed_model_monokernels:
                continue
            _closed_model_monokernels.add(owned)
        except TypeError:
            if getattr(owned, "_atom_monokernel_closed", False):
                continue
            setattr(owned, "_atom_monokernel_closed", True)
        close = getattr(owned, "close", None)
        if callable(close):
            close()


__all__ = ["close_model_monokernels"]
