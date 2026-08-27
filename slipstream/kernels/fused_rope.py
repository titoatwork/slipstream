"""Fused RoPE on Q and K.

Default path is `apply_rotary_pos_emb`. Triton runs only when
`SLIPSTREAM_FUSED=1`, CUDA + triton are available, and launch succeeds.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from slipstream.models.layers.rope import apply_rotary_pos_emb

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError:
    triton = None
    tl = None

TRITON_AVAILABLE: bool = triton is not None
_rope_kernel: Any = None


def _want_fused() -> bool:
    return os.environ.get("SLIPSTREAM_FUSED", "0") not in {"0", "false", "False", ""}


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _rope_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_h,
        n_t,
        half,
        sx0,
        sx1,
        sx2,
        sx3,
        sc0,
        sc1,
        sc2,
        sc3,
        ss0,
        ss1,
        ss2,
        ss3,
        so0,
        so1,
        so2,
        so3,
        BLOCK_H: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        t = pid % n_t
        h = (pid // n_t) % n_h
        b = pid // (n_h * n_t)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < half
        x_base = b * sx0 + h * sx1 + t * sx2
        c_base = b * sc0 + h * sc1 + t * sc2
        s_base = b * ss0 + h * ss1 + t * ss2
        o_base = b * so0 + h * so1 + t * so2
        x1 = tl.load(x_ptr + x_base + offs * sx3, mask=mask, other=0.0)
        x2 = tl.load(x_ptr + x_base + (offs + half) * sx3, mask=mask, other=0.0)
        c1 = tl.load(cos_ptr + c_base + offs * sc3, mask=mask, other=0.0)
        c2 = tl.load(cos_ptr + c_base + (offs + half) * sc3, mask=mask, other=0.0)
        s1 = tl.load(sin_ptr + s_base + offs * ss3, mask=mask, other=0.0)
        s2 = tl.load(sin_ptr + s_base + (offs + half) * ss3, mask=mask, other=0.0)
        o1 = x1 * c1 + (-x2) * s1
        o2 = x2 * c2 + x1 * s2
        tl.store(out_ptr + o_base + offs * so3, o1, mask=mask)
        tl.store(out_ptr + o_base + (offs + half) * so3, o2, mask=mask)


def _torch_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


def _as_bhtd(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x.reshape(-1, 1, 1, dim)


def _triton_apply_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if not TRITON_AVAILABLE or _rope_kernel is None:
        raise RuntimeError("triton is not installed")
    dim = int(x.shape[-1])
    if dim % 2 != 0:
        raise RuntimeError(f"rope triton requires even head dim, got {dim}")
    if x.numel() == 0:
        return x
    x4 = _as_bhtd(x, dim)
    cos4 = _as_bhtd(cos.expand_as(x), dim)
    sin4 = _as_bhtd(sin.expand_as(x), dim)
    out4 = torch.empty_like(x4)
    _b, n_h, n_t, _d = x4.shape
    rows = n_h * n_t * _b
    half = dim // 2
    block_h = _next_power_of_two(half)
    _rope_kernel[(rows,)](
        x4,
        cos4,
        sin4,
        out4,
        n_h,
        n_t,
        half,
        x4.stride(0),
        x4.stride(1),
        x4.stride(2),
        x4.stride(3),
        cos4.stride(0),
        cos4.stride(1),
        cos4.stride(2),
        cos4.stride(3),
        sin4.stride(0),
        sin4.stride(1),
        sin4.stride(2),
        sin4.stride(3),
        out4.stride(0),
        out4.stride(1),
        out4.stride(2),
        out4.stride(3),
        BLOCK_H=block_h,
        num_warps=2,
    )
    return out4.reshape(x.shape)


def _triton_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.device.type != "cuda" or k.device.type != "cuda":
        raise RuntimeError("fused_rope triton requires CUDA tensors")
    if cos.device.type != "cuda" or sin.device.type != "cuda":
        raise RuntimeError("fused_rope cos/sin must be CUDA")
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    return _triton_apply_one(q, cos_u, sin_u), _triton_apply_one(k, cos_u, sin_u)


def fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same math as `apply_rotary_pos_emb`; kept as the fuse seam for Triton."""
    if _want_fused() and TRITON_AVAILABLE and q.device.type == "cuda":
        try:
            return _triton_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        except Exception:
            pass
    return _torch_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
