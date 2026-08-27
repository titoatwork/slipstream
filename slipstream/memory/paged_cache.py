"""Physical paged KV tensor and the per-forward view Attention consumes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slipstream.core.types import Sequence
from slipstream.memory.block_manager import BlockManagerImpl


def allocate_kv_cache(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Zeros `[L, 2, num_blocks, block_size, n_kv, D]`."""
    return torch.zeros(
        (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim),
        dtype=dtype,
        device=device,
    )


def slots_from_pairs(pairs: list[tuple[int, int]], block_size: int) -> list[int]:
    return [block_id * block_size + offset for block_id, offset in pairs]


@dataclass
class PagedForward:
    """What `Attention.forward` sees for a paged step.

    `slot_mapping` indexes the **new** tokens being written this step (`N = sum Tq`).
    `seq_lens` are lengths **after** those tokens have been reserved.
    Queries are the last `query_lens[b]` tokens of sequence `b`.
    """

    kv: torch.Tensor
    block_tables: torch.Tensor  # [B, max_blocks] int32, -1 padded
    seq_lens: torch.Tensor  # [B] int32
    slot_mapping: torch.Tensor  # [N] int64
    query_lens: torch.Tensor  # [B] int32
    block_size: int
    write_kv: bool = True

    @property
    def batch(self) -> int:
        return int(self.seq_lens.shape[0])


def pad_block_tables(
    tables: list[list[int]],
    max_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    batch = len(tables)
    out = torch.full((batch, max_blocks), -1, dtype=torch.int32, device=device)
    for i, table in enumerate(tables):
        n = min(len(table), max_blocks)
        if n:
            out[i, :n] = torch.tensor(table[:n], dtype=torch.int32, device=device)
    return out


def build_paged_forward(
    kv: torch.Tensor,
    manager: BlockManagerImpl,
    seqs: list[Sequence],
    new_slots: list[list[tuple[int, int]]],
    device: torch.device,
    seq_lens: list[int] | None = None,
) -> PagedForward:
    """`new_slots[i]` is the `append_slot` result for the new tokens of `seqs[i]`.

    `seq_lens` is KV-populated length. Default `num_tokens` is correct for
    decode; replay prefills must pass `num_computed + n_new` so we do not
    gather past the pages that exist yet.
    """
    if len(seqs) != len(new_slots):
        raise ValueError("seqs / new_slots length mismatch")
    max_blocks = max((len(s.block_table) for s in seqs), default=1)
    max_blocks = max(max_blocks, 1)
    tables = pad_block_tables([s.block_table for s in seqs], max_blocks, device)
    if seq_lens is None:
        lens = [sum(manager.blocks[bid].num_tokens for bid in s.block_table) for s in seqs]
    else:
        if len(seq_lens) != len(seqs):
            raise ValueError("seq_lens / seqs length mismatch")
        lens = seq_lens
    seq_lens_t = torch.tensor(lens, dtype=torch.int32, device=device)
    query_lens = torch.tensor([len(slots) for slots in new_slots], dtype=torch.int32, device=device)
    flat: list[int] = []
    for slots in new_slots:
        flat.extend(slots_from_pairs(slots, manager.block_size))
    mapping = torch.tensor(flat, dtype=torch.int64, device=device)
    return PagedForward(
        kv=kv,
        block_tables=tables,
        seq_lens=seq_lens_t,
        slot_mapping=mapping,
        query_lens=query_lens,
        block_size=manager.block_size,
    )


def estimate_num_gpu_blocks(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype_bytes: int,
    available_bytes: int,
    gpu_memory_utilization: float,
) -> int:
    block_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes * block_size
    if block_bytes <= 0:
        raise ValueError("invalid KV block size")
    budget = int(available_bytes * gpu_memory_utilization)
    return max(1, budget // block_bytes)
