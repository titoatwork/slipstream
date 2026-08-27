"""Transformer layer primitives (RMSNorm, RoPE, attention, MLP). Phase 1."""

from slipstream.models.layers.attention import Attention, eager_attention, repeat_kv
from slipstream.models.layers.mlp import MLP
from slipstream.models.layers.rmsnorm import RMSNorm
from slipstream.models.layers.rope import (
    RotaryEmbedding,
    apply_llama3_scaling,
    apply_rotary_pos_emb,
    default_inv_freq,
    rotate_half,
)

__all__ = [
    "Attention",
    "MLP",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_llama3_scaling",
    "apply_rotary_pos_emb",
    "default_inv_freq",
    "eager_attention",
    "repeat_kv",
    "rotate_half",
]
