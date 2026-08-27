"""Rotary position embeddings: default and Llama-3 scaling."""

from __future__ import annotations

import math

import torch
from torch import nn

from slipstream.core.config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims: ``cat(-x[..., D/2:], x[..., :D/2])``."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K.

    ``cos``/``sin`` are ``[B, T, D]`` (or already broadcastable). They are
    unsqueezed on ``unsqueeze_dim`` so they match ``[B, H, T, D]``.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def default_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """``inv_freq[i] = 1 / (theta ** (i / head_dim))`` for ``i = 0, 2, 4, ...``."""
    dims = torch.arange(0, head_dim, 2, dtype=torch.float32)
    return 1.0 / (theta ** (dims / head_dim))


def apply_llama3_scaling(inv_freq: torch.Tensor, rope_scaling: dict[str, object]) -> torch.Tensor:
    """Llama-3.1 wavelength-dependent inv-freq scaling (HF 5.15)."""
    factor = _as_float(rope_scaling["factor"], "factor")
    low_freq_factor = _as_float(rope_scaling["low_freq_factor"], "low_freq_factor")
    high_freq_factor = _as_float(rope_scaling["high_freq_factor"], "high_freq_factor")
    orig_max = _as_float(
        rope_scaling["original_max_position_embeddings"],
        "original_max_position_embeddings",
    )
    wavelen = 2.0 * math.pi / inv_freq
    low_w = orig_max / low_freq_factor
    high_w = orig_max / high_freq_factor
    inv = torch.where(wavelen > low_w, inv_freq / factor, inv_freq)
    smooth = (orig_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1.0 - smooth) * inv / factor + smooth * inv
    medium = ~(wavelen < high_w) & ~(wavelen > low_w)
    return torch.where(medium, smoothed, inv)


class RotaryEmbedding(nn.Module):
    """Default + llama3 RoPE.

    Cos/sin are computed in fp32 and cast to ``x.dtype``. ``inv_freq`` is a
    plain fp32 tensor (not a buffer) so ``Module.to(dtype=bf16)`` cannot
    quantize it.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.head_dim is None:
            raise ValueError("ModelConfig.head_dim is required for RoPE")
        theta = float(config.rope_theta if config.rope_theta is not None else 10000.0)
        inv_freq = default_inv_freq(config.head_dim, theta)
        if config.rope_type == "llama3":
            if not config.rope_scaling:
                raise ValueError("rope_type='llama3' requires rope_scaling")
            inv_freq = apply_llama3_scaling(inv_freq, config.rope_scaling)
        # Not a buffer: must stay fp32 across model.to(dtype=...).
        self.inv_freq = inv_freq
        self.head_dim = config.head_dim

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, inv_freq={tuple(self.inv_freq.shape)}"

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``[B, T, head_dim]`` in ``x.dtype``."""
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        # [B, D/2, 1] @ [B, 1, T] -> [B, D/2, T]
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].to(dtype=torch.float32)
        device_type = x.device.type if isinstance(x.device.type, str) else "cpu"
        if device_type == "mps":
            device_type = "cpu"
        # RoPE frequencies must be formed in fp32 (bf16 loses low-frequency bits).
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"rope_scaling[{name!r}] must be a number, got {value!r}")
    return float(value)
