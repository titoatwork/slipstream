"""SwiGLU MLP: ``down(silu(gate(x)) * up(x))``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from slipstream.core.config import ModelConfig


class MLP(nn.Module):
    """SwiGLU feed-forward. HF names: ``gate_proj``, ``up_proj``, ``down_proj``."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        if hidden is None or intermediate is None:
            raise ValueError("ModelConfig.hidden_size and intermediate_size are required")
        factory: dict[str, torch.device | str | torch.dtype] = {}
        if device is not None:
            factory["device"] = device
        if dtype is not None:
            factory["dtype"] = dtype
        bias = config.mlp_bias
        self.gate_proj = nn.Linear(hidden, intermediate, bias=bias, **factory)
        self.up_proj = nn.Linear(hidden, intermediate, bias=bias, **factory)
        self.down_proj = nn.Linear(intermediate, hidden, bias=bias, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eager silu-mul. fused_swiglu is opt-in (SLIPSTREAM_FUSED=1), not here.
        hidden: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return hidden
