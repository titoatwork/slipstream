"""W4A16 group-wise INT4 quantization: format round-trip + GEMM parity. S9.

CPU, no weights. Proves the correctness core (MASTERPLAN §S9 / I3.1 analogue):
the packed 4-bit format round-trips, dequantization error is bounded by the group
scale, and ``w4a16_gemm`` matches a dense dequant-then-matmul. The Triton
dequant-fused kernel (GPU) must later match ``w4a16_gemm_ref`` to the same bar;
the ~4×-decode / ~0-prefill speedup is a benchmark, not a unit test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slipstream.kernels.quant_gemm import w4a16_gemm  # noqa: E402
from slipstream.kernels.quant_ref import (  # noqa: E402
    dequantize_w4a16,
    pack_int4,
    quantize_w4a16,
    unpack_int4,
    w4a16_gemm_ref,
)

GROUP_SIZES = [32, 64, 128]


def test_pack_unpack_roundtrip_exact() -> None:
    torch.manual_seed(0)
    q = torch.randint(0, 16, (7, 128), dtype=torch.int64)
    packed = pack_int4(q)
    assert packed.shape == (7, 128 // 8)
    assert packed.dtype == torch.int32
    back = unpack_int4(packed, 128)
    assert torch.equal(back, q)


def test_pack_covers_full_nibble_range() -> None:
    # The high nibble (shift 28) overflows signed int32; check the extremes survive.
    q = torch.tensor([[15, 15, 15, 15, 15, 15, 15, 15], [0, 1, 2, 3, 12, 13, 14, 15]])
    back = unpack_int4(pack_int4(q.to(torch.int64)), 8)
    assert torch.equal(back, q.to(torch.int64))


@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_dequant_error_bounded_by_scale(group_size: int) -> None:
    torch.manual_seed(1)
    out, in_features = 16, 256
    weight = torch.randn(out, in_features)
    qweight, scales, qzeros = quantize_w4a16(weight, group_size)

    assert qweight.shape == (out, in_features // 8)
    assert scales.shape == (out, in_features // group_size)
    assert qzeros.shape == (out, in_features // group_size)
    assert qzeros.min() >= 0 and qzeros.max() <= 15

    w_hat = dequantize_w4a16(qweight, scales, qzeros, group_size)
    n_groups = in_features // group_size
    scale_per_elem = (
        scales.reshape(out, n_groups, 1).expand(out, n_groups, group_size).reshape(out, in_features)
    )
    # Round-to-nearest with an affine zero-point: interior error <= scale/2, and
    # end-of-range clamping adds at most one more step, so <= scale everywhere.
    err = (weight - w_hat).abs()
    assert torch.all(err <= scale_per_elem * (1.0 + 1e-4))


def test_quantize_is_idempotent() -> None:
    # Re-quantizing an already-dequantized weight must reproduce the same nibbles.
    torch.manual_seed(2)
    weight = torch.randn(8, 128)
    qw1, s1, z1 = quantize_w4a16(weight, 32)
    w_hat = dequantize_w4a16(qw1, s1, z1, 32)
    qw2, s2, z2 = quantize_w4a16(w_hat, 32)
    assert torch.equal(unpack_int4(qw1, 128), unpack_int4(qw2, 128))


@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_gemm_matches_dense_dequant_matmul(group_size: int) -> None:
    torch.manual_seed(3)
    out, in_features, m = 24, 256, 5
    weight = torch.randn(out, in_features)
    x = torch.randn(m, in_features)
    qweight, scales, qzeros = quantize_w4a16(weight, group_size)

    expected = x @ dequantize_w4a16(qweight, scales, qzeros, group_size).t()
    torch.testing.assert_close(w4a16_gemm(x, qweight, scales, qzeros, group_size), expected)
    # The public entry and the oracle are the same computation.
    torch.testing.assert_close(
        w4a16_gemm(x, qweight, scales, qzeros, group_size),
        w4a16_gemm_ref(x, qweight, scales, qzeros, group_size),
    )


def test_gemm_preserves_batch_dims_and_dtype() -> None:
    torch.manual_seed(4)
    weight = torch.randn(12, 64)
    qweight, scales, qzeros = quantize_w4a16(weight, 32)
    x = torch.randn(2, 3, 64)
    y = w4a16_gemm(x, qweight, scales, qzeros, 32)
    assert y.shape == (2, 3, 12)
    assert y.dtype == x.dtype


def test_quantized_gemm_accuracy_vs_full_precision() -> None:
    # Accuracy smoke test (the unit-scale stand-in for the WikiText perplexity bar):
    # 4-bit group quant should perturb the GEMM output only slightly.
    torch.manual_seed(5)
    out, in_features, m = 128, 512, 16
    weight = torch.randn(out, in_features)
    x = torch.randn(m, in_features)
    qweight, scales, qzeros = quantize_w4a16(weight, 128)

    full = x @ weight.t()
    quant = w4a16_gemm(x, qweight, scales, qzeros, 128)
    rel = (quant - full).norm() / full.norm()
    assert rel < 0.1, f"relative error {rel:.4f} too high"


def test_constant_group_is_representable_exactly() -> None:
    # A group of equal values has zero range -> dequant reproduces it exactly.
    weight = torch.full((4, 64), 0.37)
    qweight, scales, qzeros = quantize_w4a16(weight, 32)
    w_hat = dequantize_w4a16(qweight, scales, qzeros, 32)
    torch.testing.assert_close(w_hat, weight, atol=1e-5, rtol=0)


def test_rejects_bad_arguments() -> None:
    weight = torch.randn(8, 100)
    with pytest.raises(ValueError):
        quantize_w4a16(weight, 32)  # 100 not divisible by 32
    with pytest.raises(ValueError):
        quantize_w4a16(torch.randn(8, 96), 12)  # group_size not multiple of 8
    with pytest.raises(ValueError):
        quantize_w4a16(torch.randn(64), 32)  # not 2-D
    with pytest.raises(ValueError):
        pack_int4(torch.randint(0, 16, (8, 100), dtype=torch.int64))  # in not divisible by 8
