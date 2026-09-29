# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded host-side routing counters for native decode backends."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class MonoRouteStats:
    """Track native attempts without unbounded layer or request labels."""

    model: str
    attempts: int = 0
    hits: Counter[str] = field(default_factory=Counter)
    fallbacks: Counter[str] = field(default_factory=Counter)

    @staticmethod
    def _key(label: str, samples: int) -> str:
        return f"{label}:s{samples}"

    def record_attempt(self, samples: int) -> None:
        self.attempts += 1

    def record_hit(self, backend: str, samples: int) -> None:
        self.hits[self._key(backend, samples)] += 1

    def record_fallback(self, reason: str, samples: int) -> None:
        self.fallbacks[self._key(reason, samples)] += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "model": self.model,
            "attempts": self.attempts,
            "hits": dict(sorted(self.hits.items())),
            "fallbacks": dict(sorted(self.fallbacks.items())),
        }
