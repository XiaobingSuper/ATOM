# SPDX-License-Identifier: MIT

import runpy
import warnings
from pathlib import Path

import torch
from torch._inductor.utils import run_and_get_code

ROOT = Path(__file__).parents[1]
direct_register_custom_op = runpy.run_path(
    ROOT / "atom" / "utils" / "custom_register.py"
)["direct_register_custom_op"]
TEST_LIB = torch.library.Library("atom_test_sparse_indexer", "FRAGMENT")


def _write_index(
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_indptr: torch.Tensor,
    owned_counts: torch.Tensor,
) -> None:
    kv_cache.copy_(weights.flatten()[: kv_cache.numel()])
    sparse_indices.copy_(torch.arange(sparse_indices.numel(), dtype=torch.int32))
    sparse_indptr.add_(1)
    owned_counts.add_(2)


def _write_index_fake(*_args) -> None:
    pass


def _read_index(kv_cache: torch.Tensor, sparse_indices: torch.Tensor) -> torch.Tensor:
    return torch.cat((kv_cache.float(), sparse_indices.float()))


def _read_index_fake(
    kv_cache: torch.Tensor, sparse_indices: torch.Tensor
) -> torch.Tensor:
    return torch.empty(
        kv_cache.numel() + sparse_indices.numel(),
        device=kv_cache.device,
        dtype=torch.float32,
    )


direct_register_custom_op(
    op_name="write_index",
    op_func=_write_index,
    mutates_args=["kv_cache", "sparse_indices", "sparse_indptr", "owned_counts"],
    fake_impl=_write_index_fake,
    target_lib=TEST_LIB,
    dispatch_key="CPU",
)
direct_register_custom_op(
    op_name="read_index",
    op_func=_read_index,
    mutates_args=[],
    fake_impl=_read_index_fake,
    target_lib=TEST_LIB,
    dispatch_key="CPU",
)


def test_side_effect_op_stays_ordered_without_a_synthetic_output():
    op = torch.ops.atom_test_sparse_indexer.write_index.default
    assert len(op._schema.returns) == 0
    assert {
        argument.name
        for argument in op._schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    } == {
        "kv_cache",
        "sparse_indices",
        "sparse_indptr",
        "owned_counts",
    }

    def fn(weights, kv_cache, sparse_indices, sparse_indptr, owned_counts):
        torch.ops.atom_test_sparse_indexer.write_index(
            weights, kv_cache, sparse_indices, sparse_indptr, owned_counts
        )
        return torch.ops.atom_test_sparse_indexer.read_index(kv_cache, sparse_indices)

    weights = torch.arange(7 * 13, dtype=torch.bfloat16).reshape(7, 13)
    cache_storage = torch.zeros(5, dtype=torch.bfloat16)
    kv_cache = cache_storage[1:4]
    sparse_indices = torch.full((3,), -1, dtype=torch.int32)
    sparse_indptr = torch.zeros(2, dtype=torch.int32)
    owned_counts = torch.zeros(2, dtype=torch.int32)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*may not alias.*")
        output, code_parts = run_and_get_code(
            torch.compile(fn, fullgraph=True),
            weights,
            kv_cache,
            sparse_indices,
            sparse_indptr,
            owned_counts,
        )
    torch.testing.assert_close(output, torch.tensor([0, 1, 2, 0, 1, 2.0]))

    code = "\n".join(code_parts)
    writer = "torch.ops.atom_test_sparse_indexer.write_index.default("
    reader = "torch.ops.atom_test_sparse_indexer.read_index.default("
    assert writer in code and code.index(writer) < code.index(reader)
    writer_line = next(line for line in code.splitlines() if writer in line)
    assert "=" not in writer_line.split(writer, 1)[0]
    assert "async_compile.cpp" not in code
    assert "async_compile.triton" not in code
    assert "empty_strided_cpu((7, 13)" not in code
