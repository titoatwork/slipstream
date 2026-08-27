"""Eager multi-head attention with GQA. Do not use SDPA (bf16 diverges)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch
from torch import nn

from slipstream.core.config import ModelConfig
from slipstream.models.layers.rope import apply_rotary_pos_emb

if TYPE_CHECKING:
    from slipstream.memory.contiguous_cache import NaiveKVCache
    from slipstream.memory.paged_cache import PagedForward

CacheLike = Union["NaiveKVCache", "PagedForward"]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads: ``[B, n_kv, T, D] -> [B, n_kv*n_rep, T, D]``.

    Uses ``expand`` then ``reshape`` (HF eager). ``n_rep == 1`` is a no-op.
    """
    batch, n_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv * n_rep, slen, head_dim)


def packed_causal_mask(
    query_len: int,
    key_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive mask ``[1, 1, Tq, Tk]`` for a dense prefix cache.

    Queries are the last ``Tq`` packed slots (standard incremental cache).
    RoPE ``position_ids`` must not enter this mask — they can diverge from
    packed indices once prefix cache / chunked prefill land (Phase 2–3).
    """
    if query_len > key_len:
        raise ValueError(f"query_len {query_len} > key_len {key_len}")
    key_pos = torch.arange(key_len, device=device)
    query_abs = torch.arange(key_len - query_len, key_len, device=device)
    allowed = key_pos.view(1, 1, 1, key_len) <= query_abs.view(1, 1, query_len, 1)
    return torch.zeros((), dtype=dtype, device=device).where(
        allowed, torch.tensor(float("-inf"), dtype=dtype, device=device)
    )


def causal_mask(
    query_positions: torch.Tensor,
    key_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Deprecated alias: Phase 1 generate uses packed indices, not RoPE ids."""
    return packed_causal_mask(query_positions.shape[-1], key_len, dtype, device)


def eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    query_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """HF-eager attention. Returns ``[B, Tq, Hq, D]``.

    Softmax is computed in fp32 then cast back to ``query.dtype``.
    The causal mask is packed-index based; ``query_positions`` is unused
    (kept so call sites stay stable) and reserved for RoPE only.
    """
    del query_positions
    n_rep = query.shape[1] // key.shape[1]
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)
    scores = torch.matmul(query, key.transpose(2, 3)) * scale
    scores = scores + packed_causal_mask(
        query.shape[-2], key.shape[-2], scores.dtype, scores.device
    )
    # fp32 softmax: bf16 softmax is not bit-stable vs HF eager.
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(weights, value)
    return out.transpose(1, 2).contiguous()


def _o_proj_bias(config: ModelConfig) -> bool:
    """Qwen q/k/v have bias; ``o_proj`` does not. Llama uses ``attention_bias`` for qkvo."""
    if config.model_type in {"qwen2", "qwen"}:
        return False
    return config.attention_bias


class Attention(nn.Module):
    """GQA attention. HF names: ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``."""

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
        n_q = _need(config.num_q_heads, "num_q_heads")
        n_kv = _need(config.num_kv_heads, "num_kv_heads")
        head_dim = _need(config.head_dim, "head_dim")
        if n_q % n_kv != 0:
            raise ValueError(f"num_q_heads ({n_q}) must be divisible by num_kv_heads ({n_kv})")

        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_q_heads = n_q
        self.num_kv_heads = n_kv
        self.num_key_value_groups = n_q // n_kv
        self.scaling = head_dim**-0.5

        factory: dict[str, torch.device | str | torch.dtype] = {}
        if device is not None:
            factory["device"] = device
        if dtype is not None:
            factory["dtype"] = dtype

        qkv_bias = config.attention_bias
        o_bias = _o_proj_bias(config)
        self.q_proj = nn.Linear(hidden, n_q * head_dim, bias=qkv_bias, **factory)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=qkv_bias, **factory)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=qkv_bias, **factory)
        self.o_proj = nn.Linear(n_q * head_dim, hidden, bias=o_bias, **factory)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: CacheLike,
    ) -> torch.Tensor:
        from slipstream.memory.paged_cache import PagedForward

        bsz, seqlen, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(bsz, seqlen, self.num_q_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        if isinstance(kv_cache, PagedForward):
            from slipstream.kernels.paged_attention import (
                paged_attention_decode,
                paged_attention_prefill,
            )
            from slipstream.kernels.reshape_and_cache import reshape_and_cache

            if kv_cache.write_kv and kv_cache.slot_mapping.numel() > 0:
                reshape_and_cache(
                    k, v, kv_cache.kv, kv_cache.slot_mapping, self.layer_idx, kv_cache.block_size
                )
            attn_fn = paged_attention_decode if seqlen == 1 else paged_attention_prefill
            attn = attn_fn(
                q,
                kv_cache.kv,
                kv_cache.block_tables,
                kv_cache.seq_lens,
                self.layer_idx,
                self.scaling,
                kv_cache.block_size,
            )
        else:
            k, v = kv_cache.update(self.layer_idx, k, v)
            attn = eager_attention(q, k, v, self.scaling, position_ids)
        projected: torch.Tensor = self.o_proj(
            attn.reshape(bsz, seqlen, self.num_q_heads * self.head_dim)
        )
        return projected


def _need(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"ModelConfig.{name} is required")
    return value
