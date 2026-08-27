"""Fused kernel numeric parity vs the Phase 1 eager ops."""

from __future__ import annotations

import torch
from slipstream.kernels.fused_rmsnorm import fused_rmsnorm
from slipstream.kernels.fused_rope import fused_rope
from slipstream.kernels.fused_swiglu import fused_swiglu
from slipstream.models.layers.rmsnorm import RMSNorm
from slipstream.models.layers.rope import apply_rotary_pos_emb


def test_fused_rmsnorm_matches_module() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    m = RMSNorm(16, 1e-6)
    ref = m(x)
    got = fused_rmsnorm(x, m.weight, eps=1e-6)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_fused_rmsnorm_with_residual() -> None:
    x = torch.randn(2, 4, 8)
    r = torch.randn(2, 4, 8)
    w = torch.ones(8)
    got = fused_rmsnorm(x, w, residual=r, eps=1e-6)
    ref = fused_rmsnorm(x + r, w, eps=1e-6)
    torch.testing.assert_close(got, ref)


def test_fused_swiglu() -> None:
    g = torch.randn(3, 5)
    u = torch.randn(3, 5)
    torch.testing.assert_close(fused_swiglu(g, u), torch.nn.functional.silu(g) * u)


def test_fused_rope_matches_apply() -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    cos = torch.randn(1, 8, 16)
    sin = torch.randn(1, 8, 16)
    a, b = fused_rope(q, k, cos, sin)
    c, d = apply_rotary_pos_emb(q, k, cos, sin)
    torch.testing.assert_close(a, c)
    torch.testing.assert_close(b, d)
