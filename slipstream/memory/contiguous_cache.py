"""Throwaway contiguous KV cache (Phase 1).

Phase-2 paging must be semantically identical to this cache: same
append / reset / overflow contract and the same returned prefix views
of shape ``[B, n_kv, seq_len, D]``. A paged implementation is a drop-in
replacement iff those semantics are preserved.

Layout (no paging)::

    k_cache, v_cache: [num_layers, max_batch, num_kv_heads, max_len, head_dim]
    invariant: seq_len <= max_len
"""

from __future__ import annotations

import torch


class NaiveKVCache:
    """Contiguous pre-allocated K/V buffers. Throwaway baseline for Phase 1.

    ``seq_len`` is shared across layers and is advanced **once per step**,
    by layer 0 only. Per-layer fill counts track how far each layer has
    written; ``seq_len`` equals layer 0's fill. Other layers must be
    updated after layer 0 in the same step with the same ``T_new`` (they
    write the slots layer 0 just reserved). The model calls ``update``
    for every layer each step with that shared ``T_new``.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_kv_heads < 1:
            raise ValueError(f"num_kv_heads must be >= 1, got {num_kv_heads}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_batch = max_batch
        self.max_len = max_len
        self.dtype = dtype
        self.device = device

        shape = (num_layers, max_batch, num_kv_heads, max_len, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

        self._seq_len = 0
        self._layer_fill = [0] * num_layers

    def reset(self) -> None:
        """Zero length; buffers are retained."""
        self._seq_len = 0
        for i in range(self.num_layers):
            self._layer_fill[i] = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``k, v`` of shape ``[B, n_kv, T_new, D]``.

        Returns full cached ``k, v`` views of shape ``[B, n_kv, seq_len, D]``
        (not clones). Layer 0 advances ``seq_len``; other layers must see
        the same ``seq_len``.
        """
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        if k.shape != v.shape:
            raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.ndim != 4:
            raise ValueError(f"k, v must be [B, n_kv, T_new, D], got {tuple(k.shape)}")

        batch, n_kv, t_new, dim = k.shape
        if batch > self.max_batch:
            raise ValueError(f"batch {batch} exceeds max_batch {self.max_batch}")
        if n_kv != self.num_kv_heads:
            raise ValueError(f"n_kv {n_kv} != num_kv_heads {self.num_kv_heads}")
        if dim != self.head_dim:
            raise ValueError(f"head_dim {dim} != {self.head_dim}")

        if layer_idx == 0:
            start = self._seq_len
            if start + t_new > self.max_len:
                raise ValueError(
                    f"KV cache overflow: seq_len={start} + T_new={t_new} > max_len={self.max_len}"
                )
            self._seq_len = start + t_new
        else:
            # Layer 0 reserved [seq_len - T_new, seq_len); this layer must fill that window.
            start = self._seq_len - t_new
            if start != self._layer_fill[layer_idx] or start < 0:
                raise ValueError(
                    f"layer {layer_idx} T_new={t_new} does not match layer 0 "
                    f"(fill={self._layer_fill[layer_idx]}, seq_len={self._seq_len}); "
                    f"update layer 0 first each step with the same T_new"
                )

        end = start + t_new
        self.k_cache[layer_idx, :batch, :, start:end, :] = k
        self.v_cache[layer_idx, :batch, :, start:end, :] = v
        self._layer_fill[layer_idx] = end

        return (
            self.k_cache[layer_idx, :batch, :, : self._seq_len, :],
            self.v_cache[layer_idx, :batch, :, : self._seq_len, :],
        )
