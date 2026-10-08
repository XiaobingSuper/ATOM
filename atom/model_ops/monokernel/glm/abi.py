# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The GLM whole-layer kernel's positional argument contract."""

from atom.mono.runtime.abi import KernelAbi

GLM5_ABI = KernelAbi(
    (
        "h_in",
        "x_out",
        "cur_pos",
        "positions",
        "slot_mapping",
        "sparse_kv_indptr",
        "kv_cache",
        "pe_cache",
        "indices",
        "rope_cos",
        "rope_sin",
        "g_in",
        "g_q",
        "g_kv",
        "g_post",
        "w_qkv_a",
        "s_qkv_a",
        "w_q_b",
        "s_q_b",
        "w_uk",
        "s_uk",
        "w_uv",
        "s_uv",
        "w_o",
        "s_o",
        "w_r",
        "bias",
        "w_ug",
        "s_ug",
        "w_dn",
        "s_dn",
        "scratch",
        "sym",
        "peers",
        "timeline_buf",
        "step",
        "rank",
        "layer",
    )
)
