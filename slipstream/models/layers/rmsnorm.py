"""RMSNorm matching HuggingFace eager (variance in fp32)."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square norm.

    Invariant: ``y = weight * rsqrt(mean(x_f32^2) + eps)`` then cast back
    to ``x.dtype``. Matches HF Llama/Qwen2 RMSNorm.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eager default (HF identity). fused_rmsnorm is opt-in via
        # SLIPSTREAM_FUSED=1 at the kernel seam, not wired here.
        # Variance in fp32; scale back to the incoming dtype before the weight.
        x32 = x.float()
        x_hat = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x_hat.to(x.dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
