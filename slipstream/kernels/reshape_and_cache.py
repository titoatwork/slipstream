"""Scatter K/V into the paged cache via slot_mapping."""

from __future__ import annotations

import torch

from slipstream.kernels.attention_ref import reshape_and_cache_ref


def reshape_and_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layer_idx: int,
    block_size: int,
) -> None:
    """Write new K/V. Triton on CUDA; ref on CPU or if the kernel fails."""
    if k.device.type == "cuda" and kv_cache.device.type == "cuda":
        try:
            from slipstream.kernels.triton_paged import reshape_and_cache_triton

            reshape_and_cache_triton(k, v, kv_cache, slot_mapping, layer_idx, block_size)
            return
        except Exception:
            pass
    reshape_and_cache_ref(k, v, kv_cache, slot_mapping, layer_idx, block_size)
