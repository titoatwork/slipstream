"""Paged attention (decode + prefill).

Triton decode is opt-in (`SLIPSTREAM_TRITON=1`). Default is the gather+eager
ref so greedy stays token-identical to Phase 1 / HF. Triton is within
atol=1e-2 but flips some argmax decisions in bf16.
"""

from __future__ import annotations

import os

import torch

from slipstream.kernels.attention_ref import paged_attention_ref


def _triton_enabled() -> bool:
    return os.environ.get("SLIPSTREAM_TRITON", "0") not in {"0", "false", "False", ""}


def paged_attention_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    layer_idx: int,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """Gathers KV through the block table. `q` is `[B, n_q, 1, D]` typically."""
    if _triton_enabled() and q.device.type == "cuda" and kv_cache.device.type == "cuda":
        try:
            from slipstream.kernels.triton_paged import paged_attention_decode_triton

            return paged_attention_decode_triton(
                q, kv_cache, block_tables, seq_lens, layer_idx, scale, block_size
            )
        except Exception:
            pass
    return paged_attention_ref(q, kv_cache, block_tables, seq_lens, layer_idx, scale, block_size)


def paged_attention_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    layer_idx: int,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    return paged_attention_ref(q, kv_cache, block_tables, seq_lens, layer_idx, scale, block_size)
