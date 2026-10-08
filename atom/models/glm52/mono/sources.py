# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Sources carried by every GLM-5.2 mono JIT cache key."""

from atom.mono.plan.build_key import source_digest

SOURCES = source_digest(
    "mono",
    "models/glm52/mono",
    "model_ops/monokernel/glm",
)
