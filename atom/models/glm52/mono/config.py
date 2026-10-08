# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile-time deployment and row-width contracts for GLM-5.2 mono."""

TP = 4
DCP = 1
MTP_QUERY_LENGTHS = {4: 5, 5: 6}
TARGET_WIDTHS = (5, 6, 10, 12, 16, 24, 48)

# The retained whole-layer core has a bounded LDS arena. Wider builds need the
# K4 row-batched stages before they can be compiled or routed.
LEGACY_CORE_WIDTHS = TARGET_WIDTHS[:5]

# Product routing remains in atom.models.glm52_mono until model-level GPU
# evidence promotes an exact width here.
PRODUCTION_WIDTHS: tuple[int, ...] = ()


def deployment_refusal(
    *,
    tp: int,
    dcp: int,
    mtp_speculative_tokens: int,
    kv_cache_dtype: str,
) -> str | None:
    """Return why a deployment cannot build this path, or ``None``."""

    checks = (
        (tp == TP, f"TP {tp}"),
        (dcp == DCP, f"DCP {dcp}"),
        (
            mtp_speculative_tokens in MTP_QUERY_LENGTHS,
            f"MTP{mtp_speculative_tokens}",
        ),
        (kv_cache_dtype == "fp8", f"KV cache {kv_cache_dtype}"),
    )
    return next((why for ok, why in checks if not ok), None)


def query_length(mtp_speculative_tokens: int) -> int:
    try:
        return MTP_QUERY_LENGTHS[mtp_speculative_tokens]
    except KeyError as error:
        raise ValueError(
            f"MTP speculative tokens must be one of {tuple(MTP_QUERY_LENGTHS)}, "
            f"got {mtp_speculative_tokens}"
        ) from error


def target_width(rows: int) -> int:
    if rows not in TARGET_WIDTHS:
        raise ValueError(
            f"GLM-5.2 mono rows must be one of {TARGET_WIDTHS}, got {rows}"
        )
    return rows
