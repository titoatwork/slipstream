"""Fused SwiGLU: silu(gate) ⊙ up.

Default path is eager `F.silu(gate) * up`. Triton runs only when
`SLIPSTREAM_FUSED=1`, CUDA + triton are available, and launch succeeds.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError:
    triton = None
    tl = None

TRITON_AVAILABLE: bool = triton is not None
_swiglu_kernel: Any = None


def _want_fused() -> bool:
    return os.environ.get("SLIPSTREAM_FUSED", "0") not in {"0", "false", "False", ""}


if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _swiglu_kernel(  # type: ignore[no-untyped-def]
        g_ptr,
        u_ptr,
        o_ptr,
        n,
        BLOCK: tl.constexpr,
    ) -> None:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0)
        gf = g.to(tl.float32)
        out = (gf * tl.sigmoid(gf) * u.to(tl.float32)).to(o_ptr.dtype.element_ty)
        tl.store(o_ptr + offs, out, mask=mask)


def _torch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def _triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if not TRITON_AVAILABLE or _swiglu_kernel is None:
        raise RuntimeError("triton is not installed")
    if gate.device.type != "cuda" or up.device.type != "cuda":
        raise RuntimeError("fused_swiglu triton requires CUDA tensors")
    if gate.shape != up.shape or gate.dtype != up.dtype:
        raise RuntimeError("gate/up must share shape and dtype")
    if gate.numel() == 0:
        return gate * up
    g = gate.contiguous()
    u = up.contiguous()
    out = torch.empty_like(g)
    n = g.numel()
    block = 1024
    grid = (n + block - 1) // block
    _swiglu_kernel[(grid,)](g.view(-1), u.view(-1), out.view(-1), n, BLOCK=block, num_warps=4)
    return out


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """`silu(gate) * up` without a separate silu materialization when compiled."""
    if _want_fused() and TRITON_AVAILABLE and gate.device.type == "cuda":
        try:
            return _triton_swiglu(gate, up)
        except Exception:
            pass
    return _torch_swiglu(gate, up)
