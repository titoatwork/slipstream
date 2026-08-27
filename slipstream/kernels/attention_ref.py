"""PyTorch reference for paged attention. Source of truth for kernel parity."""

from __future__ import annotations

import torch


def _flatten_kv(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`[B, n_kv, T, D]` or `[N, n_kv, D]` → `[N, n_kv, D]`."""
    if k.shape != v.shape:
        raise ValueError(f"k/v shape mismatch {tuple(k.shape)} vs {tuple(v.shape)}")
    if k.ndim == 4:
        # [B, n_kv, T, D] -> [B, T, n_kv, D] -> [N, n_kv, D]
        k = k.permute(0, 2, 1, 3).reshape(-1, k.shape[1], k.shape[3])
        v = v.permute(0, 2, 1, 3).reshape(-1, v.shape[1], v.shape[3])
    elif k.ndim != 3:
        raise ValueError(f"k must be [N,n_kv,D] or [B,n_kv,T,D], got {tuple(k.shape)}")
    return k.contiguous(), v.contiguous()


def reshape_and_cache_ref(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layer_idx: int,
    block_size: int,
) -> None:
    """Scatter new K/V into `kv_cache[layer, 0/1, block, offset]`."""
    k, v = _flatten_kv(k, v)
    n_tok = k.shape[0]
    if slot_mapping.numel() != n_tok:
        raise ValueError(f"slot_mapping {slot_mapping.numel()} != tokens {n_tok}")
    if n_tok == 0:
        return
    block = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offset = slot_mapping % block_size
    kv_cache[layer_idx, 0, block, offset] = k
    kv_cache[layer_idx, 1, block, offset] = v


def gather_kv_ref(
    kv_cache: torch.Tensor,
    layer_idx: int,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one sequence to contiguous `[1, n_kv, seq_len, D]`."""
    valid = int((block_table >= 0).sum().item()) if block_table.numel() else 0
    n_blocks = min((seq_len + block_size - 1) // block_size, valid)
    if n_blocks == 0:
        n_kv = kv_cache.shape[4]
        head_dim = kv_cache.shape[5]
        empty = kv_cache.new_zeros((1, n_kv, 0, head_dim))
        return empty, empty.clone()
    ids = block_table[:n_blocks].long()
    k_pages = kv_cache[layer_idx, 0, ids]  # [n_blocks, block_size, n_kv, D]
    v_pages = kv_cache[layer_idx, 1, ids]
    k = k_pages.reshape(n_blocks * block_size, k_pages.shape[2], k_pages.shape[3])[:seq_len]
    v = v_pages.reshape(n_blocks * block_size, v_pages.shape[2], v_pages.shape[3])[:seq_len]
    # [S, n_kv, D] -> [1, n_kv, S, D]
    k = k.permute(1, 0, 2).unsqueeze(0).contiguous()
    v = v.permute(1, 0, 2).unsqueeze(0).contiguous()
    return k, v


def paged_attention_ref(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    layer_idx: int,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """Paged attention via gather + eager. `q` is `[B, n_q, Tq, D]` (new tokens)."""
    from slipstream.models.layers.attention import eager_attention

    batch = q.shape[0]
    outs: list[torch.Tensor] = []
    for i in range(batch):
        sl = int(seq_lens[i].item())
        tq = q.shape[2]
        k, v = gather_kv_ref(kv_cache, layer_idx, block_tables[i], sl, block_size)
        outs.append(eager_attention(q[i : i + 1], k, v, scale))
        if outs[-1].shape[1] != tq:
            raise RuntimeError("eager_attention T mismatch")
    return torch.cat(outs, dim=0)


def copy_blocks_ref(kv_cache: torch.Tensor, src_to_dst: dict[int, int]) -> None:
    for src, dst in src_to_dst.items():
        kv_cache[:, :, dst].copy_(kv_cache[:, :, src])
