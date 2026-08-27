"""CPU, no-weight checks of RMSNorm / RoPE / repeat_kv against the Phase 1 spec.

Imports resolve through tests.correctness._api so a slightly different A1
layout still works. Missing layer modules skip (A1 not landed yet).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.correctness._api import (  # noqa: E402
    make_rotary_embedding,
    rotary_cos_sin,
    run_apply_rotary,
    run_repeat_kv,
    run_rmsnorm,
    run_rotate_half,
    spec_apply_rotary,
    spec_repeat_kv,
    spec_rmsnorm,
    spec_rope_cos_sin,
    spec_rotate_half,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_spec_formula(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    hidden = 16
    x = torch.randn(3, 5, hidden, dtype=dtype)
    weight = torch.linspace(0.5, 1.5, hidden, dtype=dtype)
    eps = 1e-6
    ours = run_rmsnorm(x, weight, eps)
    ref = spec_rmsnorm(x, weight, eps)
    assert ours.shape == x.shape
    assert ours.dtype == x.dtype
    atol = 1e-6 if dtype == torch.float32 else 1e-3
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(ours.float(), ref.float(), atol=atol, rtol=rtol)


def test_rotate_half_matches_spec() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    ours = run_rotate_half(x)
    ref = spec_rotate_half(x)
    torch.testing.assert_close(ours, ref)
    # [a, b, c, d] -> [-c, -d, a, b]
    torch.testing.assert_close(ours[0], torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def _rope_inputs(
    dtype: torch.dtype, theta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    batch, n_heads, seq_len, head_dim = 2, 4, 8, 64
    q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    ref_cos, ref_sin = spec_rope_cos_sin(positions, head_dim, theta, dtype)
    return q, k, positions, ref_cos, ref_sin


@pytest.mark.parametrize(
    ("theta", "dtype"),
    [
        (1e6, torch.float32),  # Qwen2.5-0.5B
        (1e4, torch.float32),  # TinyLlama
        (1e6, torch.bfloat16),
    ],
)
def test_apply_rotary_matches_local_reference(theta: float, dtype: torch.dtype) -> None:
    """rotate_half + apply_rotary_pos_emb vs the 15-line HF-eager reference."""
    q, k, _positions, ref_cos, ref_sin = _rope_inputs(dtype, theta)
    q_ref, k_ref = spec_apply_rotary(q, k, ref_cos, ref_sin)
    q_ours, k_ours = run_apply_rotary(q, k, ref_cos, ref_sin)
    atol = 1e-5 if dtype == torch.float32 else 2e-2
    rtol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(q_ours.float(), q_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(k_ours.float(), k_ref.float(), atol=atol, rtol=rtol)


def test_rotary_embedding_cos_sin_matches_spec() -> None:
    """RotaryEmbedding default inv_freq / cos / sin vs the same reference."""
    dtype = torch.float32
    theta = 1e6
    q, _k, positions, ref_cos, ref_sin = _rope_inputs(dtype, theta)
    head_dim = q.shape[-1]
    seq_len = q.shape[2]
    batch = q.shape[0]
    rope = make_rotary_embedding(head_dim, theta, max_pos=seq_len + 8)
    cos, sin = rotary_cos_sin(rope, positions, like=q)
    cos_b, sin_b = _coerce_cis(cos, sin, batch=batch, seq_len=seq_len, head_dim=head_dim)
    torch.testing.assert_close(cos_b.float(), ref_cos.float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sin_b.float(), ref_sin.float(), atol=1e-5, rtol=1e-5)


def _coerce_cis(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    def _one(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4:
            # [B, 1, T, D] or [B, H, T, D] — take a heads-broadcast slice
            t = t[:, 0]
        if t.ndim == 2:
            t = t.unsqueeze(0).expand(batch, -1, -1)
        if t.shape[-1] != head_dim or t.shape[-2] != seq_len:
            t = t.reshape(t.shape[0], -1, t.shape[-1])
        return t.reshape(batch, seq_len, head_dim)

    return _one(cos), _one(sin)


def test_repeat_kv_qwen05_gqa_shape() -> None:
    """Qwen2.5-0.5B GQA: 2 KV heads × 7 → 14 Q heads."""
    torch.manual_seed(2)
    batch, n_kv, seq_len, head_dim = 3, 2, 5, 8
    x = torch.randn(batch, n_kv, seq_len, head_dim)
    ours = run_repeat_kv(x, n_rep=7)
    assert ours.shape == (batch, 14, seq_len, head_dim)
    ref = spec_repeat_kv(x, n_rep=7)
    torch.testing.assert_close(ours, ref)


def test_repeat_kv_identity_when_n_rep_is_one() -> None:
    x = torch.randn(2, 4, 3, 6)
    ours = run_repeat_kv(x, n_rep=1)
    assert ours.shape == x.shape
    torch.testing.assert_close(ours, x)
