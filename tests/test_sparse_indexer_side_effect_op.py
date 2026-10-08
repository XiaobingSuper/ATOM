# SPDX-License-Identifier: MIT

import ast
import runpy
import warnings
from pathlib import Path

import torch
from torch._inductor.utils import run_and_get_code
from torch._subclasses.fake_tensor import FakeTensorMode

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


def test_none_schema_and_fake_impl_declare_only_real_mutations():
    op = torch.ops.atom_test_sparse_indexer.write_index.default
    assert len(op._schema.returns) == 0
    mutated = {
        argument.name
        for argument in op._schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    }
    assert mutated == {
        "kv_cache",
        "sparse_indices",
        "sparse_indptr",
        "owned_counts",
    }

    with FakeTensorMode() as mode:
        args = [
            mode.from_tensor(torch.empty(7, 13, dtype=torch.bfloat16)),
            mode.from_tensor(torch.empty(3, dtype=torch.bfloat16)),
            mode.from_tensor(torch.empty(3, dtype=torch.int32)),
            mode.from_tensor(torch.empty(2, dtype=torch.int32)),
            mode.from_tensor(torch.empty(2, dtype=torch.int32)),
        ]
        assert op(*args) is None


def test_inductor_orders_no_output_writer_before_cache_and_sparse_reader():
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


def test_native_and_plugin_return_contracts_stay_separate():
    contracts = [
        ("atom/models/deepseek_v2.py", "sparse_attn_indexer", None),
        ("atom/models/deepseek_v2.py", "sparse_attn_indexer_fake", None),
        (
            "atom/plugin/vllm/attention/layer_sparse_mla.py",
            "sparse_attn_indexer_plugin_mode",
            "torch.Tensor",
        ),
        (
            "atom/plugin/sglang/attention_backend/sparse_mla_indexer.py",
            "sparse_attn_indexer_sglang_plugin_mode",
            "torch.Tensor",
        ),
        (
            "atom/plugin/rtpllm/attention_backend/rtp_sparse_mla_backend.py",
            "rtp_sparse_attn_indexer",
            "torch.Tensor",
        ),
    ]
    for relative_path, function_name, expected in contracts:
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        if expected is None:
            assert (
                isinstance(function.returns, ast.Constant)
                and function.returns.value is None
            )
            assert not any(
                isinstance(node, ast.Return) and node.value is not None
                for node in ast.walk(function)
            )
        else:
            assert ast.unparse(function.returns) == expected

    source = (ROOT / "atom/models/deepseek_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    registration = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and any(
            keyword.arg == "op_name"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "sparse_attn_indexer"
            for keyword in node.value.keywords
        )
    )
    mutations = next(
        keyword.value
        for keyword in registration.keywords
        if keyword.arg == "mutates_args"
    )
    assert [item.value for item in mutations.elts] == [
        "kv_cache",
        "sparse_kv_indices_buffer",
        "dcp_sparse_kv_indptr_buffer",
        "dcp_owned_counts_buffer",
    ]
