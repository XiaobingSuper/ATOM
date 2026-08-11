import pytest
import torch
from aiter import QuantType, dtypes

from atom.model_ops.kimi_k3.activations import (
    _situ_and_mul_torch,
    situ_and_mul,
)
from atom.model_ops.linear import _a8w8_preshuffle_k_padding


@pytest.mark.parametrize(("m", "d"), [(8, 768), (129, 768), (2, 3584)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm")
def test_situ_and_mul_matches_torch(m: int, d: int):
    torch.manual_seed(17 + d)
    x = torch.randn((m, 2 * d), device="cuda", dtype=torch.bfloat16)

    actual = situ_and_mul(x, 4.0, 25.0)
    expected = _situ_and_mul_torch(x, 4.0, 25.0)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("beta", "linear_beta"),
    [(2.5, 17.3), (6.0, 32.0)],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm")
def test_situ_and_mul_supports_generic_betas(beta: float, linear_beta: float):
    torch.manual_seed(71)
    x = torch.randn((4, 2 * 768), device="cuda", dtype=torch.bfloat16)

    actual = situ_and_mul(x, beta, linear_beta)
    expected = _situ_and_mul_torch(x, beta, linear_beta)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(("m", "d"), [(8, 768), (129, 768), (8, 3584)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm")
def test_situ_and_mul_ptpc_quant_output(m: int, d: int):
    torch.manual_seed(29 + d)
    x = torch.randn((m, 2 * d), device="cuda", dtype=torch.bfloat16)

    quantized, scale = situ_and_mul(
        x,
        4.0,
        25.0,
        quant_type=QuantType.per_Token,
        quant_dtype=dtypes.fp8,
    )
    expected = _situ_and_mul_torch(x, 4.0, 25.0).float()
    dequantized = quantized.float() * scale.float()

    assert quantized.dtype == dtypes.fp8
    assert quantized.shape == (m, d)
    assert scale.shape == (m, 1)
    torch.testing.assert_close(dequantized, expected, rtol=8e-2, atol=1.5e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm")
def test_situ_and_mul_ptpc_compiles_as_fullgraph():
    x = torch.randn((4, 2 * 768), device="cuda", dtype=torch.bfloat16)

    @torch.compile(backend="eager", fullgraph=True)
    def compiled(inp: torch.Tensor):
        return situ_and_mul(
            inp,
            4.0,
            25.0,
            quant_type=QuantType.per_Token,
            quant_dtype=dtypes.fp8,
        )

    quantized, scale = compiled(x)
    expected = _situ_and_mul_torch(x, 4.0, 25.0).float()
    torch.testing.assert_close(
        quantized.float() * scale.float(), expected, rtol=8e-2, atol=1.5e-1
    )


def test_gfx1250_ptpc_small_k_padding(monkeypatch):
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_gfx", lambda: "gfx1250")

    assert _a8w8_preshuffle_k_padding(128) == 256
    assert _a8w8_preshuffle_k_padding(256) == 0


def test_other_arch_does_not_pad_small_k(monkeypatch):
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_gfx", lambda: "gfx950")

    assert _a8w8_preshuffle_k_padding(128) == 0
