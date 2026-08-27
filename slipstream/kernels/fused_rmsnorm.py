"""Fused RMSNorm + optional residual add.

Default path is eager PyTorch (identical to `models.layers.rmsnorm.RMSNorm`).
Triton runs only when `SLIPSTREAM_FUSED=1`, CUDA + triton are available,
and compile/launch succeeds.
"""

from __future__ import annotations

import os
from typing import Any

import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError:
    triton = None
    tl = None

TRITON_AVAILABLE: bool = triton is not None
_rmsnorm_kernel: Any = None


def _want_fused() -> bool:
    return os.environ.get("SLIPSTREAM_FUSED", "0") not in {"0", "false", "False", ""}


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _rmsnorm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        w_ptr,
        r_ptr,
        y_ptr,
        n_rows,
        n_cols,
        eps,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        if row >= n_rows:
            return
        acc = tl.zeros((), dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            mask = offs < n_cols
            x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_RESIDUAL:
                x = x + tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(x * x)
        rstd = tl.rsqrt(acc / n_cols + eps)
        for start in range(0, n_cols, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            mask = offs < n_cols
            x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_RESIDUAL:
                x = x + tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            y = (x * rstd).to(y_ptr.dtype.element_ty) * w
            tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


def _torch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if residual is not None:
        x = x + residual
    x32 = x.float()
    x_hat = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x_hat.to(x.dtype)


def _triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not TRITON_AVAILABLE or _rmsnorm_kernel is None:
        raise RuntimeError("triton is not installed")
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise RuntimeError("fused_rmsnorm triton requires CUDA tensors")
    if residual is not None and residual.device.type != "cuda":
        raise RuntimeError("fused_rmsnorm residual must be CUDA")
    cols = int(x.shape[-1])
    if cols == 0 or x.numel() == 0:
        out = x if residual is None else x + residual
        return weight * out
    if weight.numel() != cols:
        raise RuntimeError(f"weight numel {weight.numel()} != hidden {cols}")
    x_c = x.contiguous()
    rows = x_c.numel() // cols
    x_rows = x_c.view(rows, cols)
    w = weight.reshape(cols).contiguous()
    if residual is not None:
        r_rows = residual.contiguous().reshape(rows, cols)
        has_residual = True
    else:
        r_rows = x_rows
        has_residual = False
    y = torch.empty_like(x_rows)
    block_n = min(4096, max(128, _next_power_of_two(cols)))
    _rmsnorm_kernel[(rows,)](
        x_rows,
        w,
        r_rows,
        y,
        rows,
        cols,
        float(eps),
        HAS_RESIDUAL=has_residual,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return y.view_as(x)


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """`rms(x) * w`, or `rms(x + residual) * w` when residual is set.

    Variance in fp32, matching `models.layers.rmsnorm.RMSNorm`.
    """
    if _want_fused() and TRITON_AVAILABLE and x.device.type == "cuda":
        try:
            return _triton_rmsnorm(x, weight, residual, eps)
        except Exception:
            pass
    return _torch_rmsnorm(x, weight, residual, eps)
