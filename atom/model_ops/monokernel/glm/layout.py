# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compile-time storage layout and CTA schedule for the GLM-5 MonoKernel."""

from dataclasses import dataclass

from atom.model_ops.monokernel.config import (
    GLM5_CONFIG,
    HIDDEN,
    INTER,
    KV_LORA,
    LayerConfig,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    QKV_A_ROWS,
    TOP_K,
    V_DIM,
    as_layer_config,
)
from atom.model_ops.monokernel.glm.abi import GlmAgenticShape
from atom.model_ops.monokernel.layout import (
    BLOCKS,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
)

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
INDEX_TILE = 16
INDEX_KEYS_PER_TASK = 64
INDEX_RADIX_WORDS = 264


def index_selection_lds_regions(
    topk: int,
    *,
    histogram_offset: int | None = None,
) -> dict[str, slice]:
    """Return the disjoint LDS regions live during bounded index selection."""

    if not 0 < topk <= 2048:
        raise ValueError("index selection top-k must be in [1, 2048]")
    q_words = INDEX_Q_ROWS // 2
    regions = {
        "index_q": slice(0, q_words),
        "sort_keys": slice(q_words, q_words + topk),
        "sort_logical": slice(q_words + topk, q_words + 2 * topk),
        "index_weights": slice(
            q_words + 2 * topk,
            q_words + 2 * topk + INDEX_HEADS,
        ),
    }
    if histogram_offset is not None:
        regions["radix_histogram"] = slice(
            histogram_offset,
            histogram_offset + INDEX_RADIX_WORDS,
        )
    return regions


def sparse_cache_rows(
    sparse_kv_indices,
    *,
    sample: int,
    topk: int,
    cur_pos: int = 0,
    sparse_kv_indptr=None,
):
    """Resolve the cache rows consumed by one flat or paged attention row."""

    if sparse_kv_indptr is not None:
        start, end = sparse_kv_indptr[sample : sample + 2]
        return sparse_kv_indices[start:end]
    context = cur_pos + sample + 1
    count = min(context, topk)
    if context <= topk:
        return range(count)
    start = sample * topk
    return sparse_kv_indices[start : start + count]


def paged_row_contract(
    sparse_kv_indices,
    sparse_kv_indptr,
    sample: int,
    slot_mapping=None,
):
    """Reference the device contract for one ATOM-layout request row."""

    start, end = sparse_kv_indptr[sample : sample + 2]
    active = end > start
    rows = sparse_kv_indices[start:end] if active else ()
    owns_slot = slot_mapping is None or slot_mapping[sample] >= 0
    return {
        "active": active,
        "context": end - start,
        "index_base": start if active else 0,
        "safe_row": rows[0] if active else 0,
        "write_cache": active and owns_slot,
    }


N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
N_UG_PER_SLOT = INTER // UG_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
assert XQ_WAVES * 4 <= WAVES


def dn_tile(
    samples: int,
    expert_mxfp4: bool = False,
    model_config: LayerConfig = GLM5_CONFIG,
) -> int:
    """Return rows per expert-down/FFN-reduce task for the tuned schedule."""

    config = as_layer_config(model_config)
    return (
        32
        if samples == 1 or (samples > 4 and expert_mxfp4)
        else config.hidden // BLOCKS
    )


def sparse_keys_per_task(samples: int) -> int:
    """Use narrower sparse-attention tiles when batch eight fills the LDS arena."""

    return 32 if samples > 4 else 64


def fp8_scale_offset(
    row_group,
    k_chunk,
    *,
    k_size: int,
    block_m: int,
    block_k: int = 128,
):
    """Address one FP8 weight scale for a 16-row MFMA row group."""

    if block_m not in (64, 128) or block_k != 128:
        raise ValueError("GLM FP8 scales require block_m 64/128 and block_k 128")
    return (row_group * 16 // block_m) * (k_size // block_k) + k_chunk // 2


def ug_split(samples: int, model_config: LayerConfig = GLM5_CONFIG):
    """Return the balanced up/gate leftover split for batches two and four."""

    config = as_layer_config(model_config)
    n_ug_per_slot = config.inter // UG_TILE
    full_tiles, remainder = divmod(
        (samples * config.top_k + 1) * n_ug_per_slot,
        BLOCKS,
    )
    if samples not in (2, 4) or remainder == 0 or BLOCKS % remainder:
        return None
    segments = BLOCKS // remainder
    if (
        (config.hidden // 128) % segments
        or (config.hidden // 128) // segments > WAVES // 2
    ):
        return None
    return full_tiles, segments


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    heads: int,
    npes: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    model_config: LayerConfig = GLM5_CONFIG,
):
    """Return byte offsets for per-rank scratch and symmetric peer buffers."""

    config = as_layer_config(model_config)
    split_count = sparse_attention_topk // sparse_keys_per_task(samples)
    pair_bytes = 8
    items = [
        ("q_a", samples * config.q_lora * pair_bytes),
        ("q_an", samples * config.q_lora // 2 * pair_bytes),
        ("kv_a", samples * (config.kv_lora + config.pe_dim) * pair_bytes),
        ("kvnew", samples * config.kv_lora * pair_bytes),
        ("penew", samples * config.pe_dim * pair_bytes),
        ("q_nope", samples * heads * config.nope_dim * pair_bytes),
        ("q_pe", samples * heads * config.pe_dim * pair_bytes),
        ("q_lat", samples * heads * config.kv_lora * pair_bytes),
        (
            "sp_acc",
            samples * split_count * heads * config.kv_lora * pair_bytes,
        ),
        ("sp_m", samples * split_count * heads * pair_bytes),
        ("sp_l", samples * split_count * heads * pair_bytes),
        ("o", samples * heads * config.v_dim * pair_bytes),
        ("a", samples * config.hidden * pair_bytes),
        ("scores", samples * config.n_experts * pair_bytes),
        ("xq", samples * config.hidden // 4 * pair_bytes),
        ("xqs", samples * (config.hidden // 128) * pair_bytes),
        ("sel", samples * config.moe_slots * pair_bytes),
        ("prob", samples * config.moe_slots * pair_bytes),
        ("mid", samples * config.moe_slots * config.inter * pair_bytes),
        ("ugp", BLOCKS * samples * 2 * UG_TILE * pair_bytes),
        ("xqd", samples * config.hidden * 4),
    ]
    if with_indexer:
        items += [
            ("index_k", samples * INDEX_DIM * pair_bytes),
            ("index_k_new", samples * INDEX_DIM // 2 * pair_bytes),
            ("index_ready", samples * pair_bytes),
            ("index_w", samples * INDEX_HEADS * pair_bytes),
            ("index_q", samples * INDEX_Q_ROWS // 2 * pair_bytes),
            ("indices_ready", samples * pair_bytes),
        ]

    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * config.hidden * pair_bytes
    region = 2 * part
    symmetric = {
        "attn": 0,
        "ffn": region,
        "_part_stride": part,
        "_bytes": 2 * region,
    }
    return scratch, symmetric


def stage_tasks(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    model_config: LayerConfig = GLM5_CONFIG,
):
    """Return ``(stage name, task count)`` pairs in execution order."""

    config = as_layer_config(model_config)
    n_qkv_a = config.qkv_a_rows // QKV_A_TILE
    n_row_tiles = config.hidden // ROW_TILE
    n_router = config.n_experts // ROUTER_TILE
    head_groups = (heads + WAVES - 1) // WAVES
    split_ctas_per_tile = head_groups if samples == 1 else 1
    ug_rounds = (config.inter + BLOCKS - 1) // BLOCKS
    tasks = [
        ("qkv_a", n_qkv_a),
        ("q_norm", samples),
        ("cache", 1),
        ("q_b", heads * (config.nope_dim + config.pe_dim) // Q_B_TILE),
    ]
    if with_indexer:
        tasks += [("index_q", INDEX_Q_ROWS // INDEX_TILE)]
    tasks += [("uk", heads * config.kv_lora // UK_TILE)]
    if with_indexer:
        tasks += [
            # Selection recomputes score tiles during bounded radix passes.
            ("index_score", 0),
            ("index_select", samples),
        ]
    tasks += [
        (
            "split",
            samples
            * (sparse_attention_topk // sparse_keys_per_task(samples))
            * split_ctas_per_tile,
        ),
        ("uv", samples * (heads * config.v_dim // UV_TILE)),
        ("o", n_row_tiles),
        ("router", samples * n_router),
        (
            "ug",
            ug_rounds * BLOCKS
            if samples == 1
            else samples * ug_rounds * BLOCKS,
        ),
        (
            "down",
            config.hidden // dn_tile(samples, expert_mxfp4, config),
        ),
    ]
    return tasks


@dataclass(frozen=True)
class GlmAgenticWorkspaceLayout:
    """One row-tile arena reusable by every layer in an Agentic graph bucket."""

    config: LayerConfig
    row_capacity: int
    tile_rows: int
    tile_count: int
    physical_experts: int
    scratch: dict[str, int]
    symmetric: dict[str, int]


def agentic_workspace_layout(
    shape: GlmAgenticShape,
    *,
    npes: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
) -> GlmAgenticWorkspaceLayout:
    """Plan the shared tile arena without pretending the full launch is ready."""

    scratch, symmetric = layout(
        shape.common.tile_rows,
        shape.local_heads,
        npes,
        sparse_attention_topk,
        with_indexer,
        index_max_seq,
        shape.config,
    )
    return GlmAgenticWorkspaceLayout(
        config=shape.config,
        row_capacity=shape.common.row_capacity,
        tile_rows=shape.common.tile_rows,
        tile_count=shape.common.tiles,
        physical_experts=shape.physical_experts,
        scratch=scratch,
        symmetric=symmetric,
    )
