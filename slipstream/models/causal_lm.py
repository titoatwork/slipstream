"""Shared Llama/Qwen causal LM. Weight names match HuggingFace."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from slipstream.core.config import ModelConfig
from slipstream.models.layers.attention import Attention
from slipstream.models.layers.mlp import MLP
from slipstream.models.layers.rmsnorm import RMSNorm
from slipstream.models.layers.rope import RotaryEmbedding

if TYPE_CHECKING:
    from slipstream.memory.contiguous_cache import NaiveKVCache
    from slipstream.memory.paged_cache import PagedForward

    CacheLike = NaiveKVCache | PagedForward
else:
    CacheLike = object


def _need(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"ModelConfig.{name} is required")
    return value


class DecoderLayer(nn.Module):
    """Pre-norm residual block: ``x + attn(rms(x)); x + mlp(rms(x))``."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden = _need(config.hidden_size, "hidden_size")
        eps = float(config.rms_norm_eps if config.rms_norm_eps is not None else 1e-6)
        self.self_attn = Attention(config, layer_idx, device=device, dtype=dtype)
        self.mlp = MLP(config, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(hidden, eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden, eps, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: CacheLike,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_ids, cos, sin, kv_cache
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class DecoderModel(nn.Module):
    """Token embeddings, decoder stack, final RMSNorm, RoPE."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden = _need(config.hidden_size, "hidden_size")
        vocab = _need(config.vocab_size, "vocab_size")
        n_layers = _need(config.num_layers, "num_layers")
        eps = float(config.rms_norm_eps if config.rms_norm_eps is not None else 1e-6)
        self.embed_tokens = nn.Embedding(vocab, hidden, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i, device=device, dtype=dtype) for i in range(n_layers)]
        )
        self.norm = RMSNorm(hidden, eps, device=device, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: CacheLike,
    ) -> torch.Tensor:
        hidden: torch.Tensor = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden, positions)
        for layer in self.layers:
            hidden = layer(hidden, positions, cos, sin, kv_cache)
        normed: torch.Tensor = self.norm(hidden)
        return normed


class CausalLM(nn.Module):
    """Tied/untied causal LM used by Llama and Qwen wrappers.

    ``forward`` returns logits ``[B, T, vocab]``. Every layer must call
    ``kv_cache.update``. Difference vs Llama/Qwen is only bias / RoPE
    fields on ``ModelConfig``.

    Phase 1 invariant: packed slot ``i`` is token position ``i`` (dense
    prefix from 0). RoPE uses ``positions``; the causal mask uses packed
    indices. Phase 2 paging must not pass global positions into the mask.

    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = _need(config.hidden_size, "hidden_size")
        vocab = _need(config.vocab_size, "vocab_size")
        self.model = DecoderModel(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(hidden, vocab, bias=False, device=device, dtype=dtype)
        self.vocab_size = vocab

    def tie_lm_head(self) -> None:
        """Point ``lm_head.weight`` at ``embed_tokens.weight`` (same Parameter)."""
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: CacheLike,
    ) -> torch.Tensor:
        hidden = self.model(input_ids, positions, kv_cache)
        logits: torch.Tensor = self.lm_head(hidden)
        return logits
