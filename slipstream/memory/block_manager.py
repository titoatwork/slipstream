"""Paged KV block allocator (S2).

Invariants (asserted when SLIPSTREAM_DEBUG=1):
  I2.2  free-list ids have ref_count == 0; live ids are not on the free list
  I2.3  free(seq) returns exactly that seq's unique blocks (via refcount)
  I2.4  COW triggers iff the written tail had ref_count > 1
  I2.5  tail.num_tokens <= block_size
  Conservation: sum(ref_count) == sum(len(table) for live tables)
"""

from __future__ import annotations

import torch

from slipstream.core.debug import DEBUG, assert_debug
from slipstream.core.types import AllocStatus, PhysicalBlock, Sequence


class BlockManagerImpl:
    """Owns ALL KV block ids. Optional bound tensor for COW copies."""

    def __init__(
        self,
        num_gpu_blocks: int,
        block_size: int,
        num_cpu_blocks: int = 0,
        max_model_len: int = 4096,
    ) -> None:
        if num_gpu_blocks < 1:
            raise ValueError("num_gpu_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.num_cpu_blocks = num_cpu_blocks
        self.max_model_len = max_model_len
        self.blocks: list[PhysicalBlock] = [
            PhysicalBlock(block_id=i, ref_count=0, num_tokens=0) for i in range(num_gpu_blocks)
        ]
        self._free: list[int] = list(range(num_gpu_blocks))
        self._tables: dict[int, list[int]] = {}
        self._kv: torch.Tensor | None = None
        self._pin_count: dict[int, int] = {}
        self._swapped: dict[int, dict[int, int]] = {}  # seq_id -> gpu->cpu (old ids)
        self._swap_meta: dict[int, list[int]] = {}  # seq_id -> num_tokens per logical page
        self._swap: object | None = None
        self._prefix: object | None = None

    def bind_kv(self, kv_cache: torch.Tensor) -> None:
        """Bind the physical paged tensor so COW can copy pages."""
        if kv_cache.ndim != 6:
            raise ValueError(
                "kv_cache must be [num_layers, 2, num_blocks, block_size, n_kv, head_dim]"
            )
        if kv_cache.shape[2] != self.num_gpu_blocks:
            raise ValueError("kv_cache num_blocks does not match the manager")
        if kv_cache.shape[3] != self.block_size:
            raise ValueError("kv_cache block_size does not match the manager")
        self._kv = kv_cache

    def bind_swap(self, swap: object) -> None:
        self._swap = swap

    def bind_prefix(self, prefix: object) -> None:
        self._prefix = prefix

    def pin(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self.blocks[bid].ref_count += 1
            self._pin_count[bid] = self._pin_count.get(bid, 0) + 1
        self._check_invariants()

    def unpin(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            held = self._pin_count.get(bid, 0)
            if held <= 0:
                raise RuntimeError(f"unpin of unpinned block {bid}")
            self._pin_count[bid] = held - 1
            if self._pin_count[bid] == 0:
                del self._pin_count[bid]
            block = self.blocks[bid]
            block.ref_count -= 1
            if block.ref_count == 0:
                block.num_tokens = 0
                block.block_hash = None
                self._free.append(bid)
        self._check_invariants()

    def attach(self, seq: Sequence, block_ids: list[int]) -> None:
        """Give `seq` an existing table (prefix hit). Extra ref per page."""
        table = list(block_ids)
        for bid in table:
            self.blocks[bid].ref_count += 1
        self._tables[seq.seq_id] = table
        seq.block_table = list(table)
        self._check_invariants()

    def get_num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_for_tokens(self, n_tokens: int) -> int:
        if n_tokens <= 0:
            return 0
        return (n_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence) -> AllocStatus:
        # Peak is prompt + output cap, not current_len + max_tokens (that
        # double-counts already generated tokens and aborts recomputes).
        out_cap = (
            seq.oracle_output_len
            if seq.oracle_output_len is not None
            else seq.sampling_params.max_tokens
        )
        peak = min(self.max_model_len, seq.num_prompt_tokens + max(0, out_cap))
        if self.blocks_for_tokens(peak) > self.num_gpu_blocks:
            return AllocStatus.NEVER
        needed = self._blocks_still_needed(seq)
        if needed > self.get_num_free_blocks():
            prefix = self._prefix
            evict = getattr(prefix, "evict", None)
            if evict is not None:
                evict(needed - self.get_num_free_blocks())
            needed = self._blocks_still_needed(seq)
            if needed > self.get_num_free_blocks():
                return AllocStatus.LATER
        return AllocStatus.OK

    def allocate(self, seq: Sequence) -> None:
        status = self.can_allocate(seq)
        if status is not AllocStatus.OK:
            raise RuntimeError(f"cannot allocate seq {seq.seq_id}: {status}")
        needed = self._blocks_still_needed(seq)
        table = self._table(seq)
        for _ in range(needed):
            table.append(self._pop_free())
        seq.block_table = list(table)
        self._check_invariants()

    def append_slot(self, seq: Sequence) -> tuple[int, int]:
        table = self._table(seq)
        idx = self._first_writable(table)
        if idx is None:
            table.append(self._pop_free())
            idx = len(table) - 1
        tail_id = table[idx]
        tail = self.blocks[tail_id]
        if tail.ref_count > 1:
            tail_id = self._cow_at(table, idx)
            tail = self.blocks[tail_id]
        if tail.num_tokens >= self.block_size:
            raise RuntimeError("tail block full after allocation")
        offset = tail.num_tokens
        tail.num_tokens += 1
        seq.block_table = list(table)
        assert_debug(tail.num_tokens <= self.block_size, "I2.5 tail overflow")
        self._check_invariants()
        return tail_id, offset

    def fork(self, parent: Sequence, child: Sequence) -> None:
        parent_table = list(self._table(parent))
        for bid in parent_table:
            self.blocks[bid].ref_count += 1
        self._tables[child.seq_id] = parent_table
        child.block_table = list(parent_table)
        self._check_invariants()

    def free(self, seq: Sequence) -> None:
        table = self._tables.pop(seq.seq_id, list(seq.block_table))
        for bid in table:
            block = self.blocks[bid]
            if block.ref_count <= 0:
                raise RuntimeError(f"double-free of block {bid}")
            block.ref_count -= 1
            if block.ref_count == 0:
                block.num_tokens = 0
                block.block_hash = None
                self._free.append(bid)
        seq.block_table = []
        self._check_invariants()

    def swap_out(self, seq: Sequence) -> dict[int, int]:
        swap = self._swap
        if swap is None:
            raise NotImplementedError("no CpuSwapSpace bound")
        table = list(self._table(seq))
        if not table:
            self._swapped[seq.seq_id] = {}
            self._swap_meta[seq.seq_id] = []
            self.free(seq)
            return {}
        cpu_ids = swap.allocate(len(table))  # type: ignore[attr-defined]
        mapping: dict[int, int] = {}
        fills = [self.blocks[bid].num_tokens for bid in table]
        if self._kv is not None:
            cpu_data = swap.data  # type: ignore[attr-defined]
            for gpu_id, cpu_id in zip(table, cpu_ids, strict=True):
                cpu_data[:, :, cpu_id].copy_(self._kv[:, :, gpu_id].detach().to("cpu"))
                mapping[gpu_id] = cpu_id
        else:
            mapping = dict(zip(table, cpu_ids, strict=True))
        self._swapped[seq.seq_id] = mapping
        self._swap_meta[seq.seq_id] = fills
        # Drop GPU ownership; CPU copy is the source of truth.
        self.free(seq)
        return mapping

    def swap_in(self, seq: Sequence) -> dict[int, int]:
        swap = self._swap
        if swap is None:
            raise NotImplementedError("no CpuSwapSpace bound")
        mapping = self._swapped.pop(seq.seq_id, {})
        fills = self._swap_meta.pop(seq.seq_id, [])
        if not mapping:
            return {}
        gpu_ids = [self._pop_free() for _ in mapping]
        reverse: dict[int, int] = {}
        cpu_ids = list(mapping.values())
        if self._kv is not None:
            cpu_data = swap.data  # type: ignore[attr-defined]
            for gpu_id, cpu_id, fill in zip(gpu_ids, cpu_ids, fills, strict=True):
                self._kv[:, :, gpu_id].copy_(cpu_data[:, :, cpu_id].to(self._kv.device))
                self.blocks[gpu_id].num_tokens = fill
                reverse[cpu_id] = gpu_id
        else:
            reverse = dict(zip(cpu_ids, gpu_ids, strict=True))
            for gpu_id, fill in zip(gpu_ids, fills, strict=True):
                self.blocks[gpu_id].num_tokens = fill
        swap.free(cpu_ids)  # type: ignore[attr-defined]
        self._tables[seq.seq_id] = gpu_ids
        seq.block_table = list(gpu_ids)
        self._check_invariants()
        return reverse

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        prefix = self._prefix
        if prefix is None:
            return [], 0
        match = getattr(prefix, "match", None)
        if match is None:
            return [], 0
        result = match(token_ids)
        return result[0], result[1]

    def kv_utilization(self, live: list[Sequence]) -> float:
        """Useful tokens / allocated token-slots. 1 - this is waste."""
        slots = 0
        useful = 0
        for seq in live:
            table = seq.block_table
            if not table:
                continue
            slots += len(table) * self.block_size
            useful += seq.num_tokens
        if slots == 0:
            return 1.0
        return useful / slots

    def _blocks_still_needed(self, seq: Sequence) -> int:
        have = len(seq.block_table)
        want = self.blocks_for_tokens(max(seq.num_tokens, seq.num_prompt_tokens))
        return max(0, want - have)

    def _table(self, seq: Sequence) -> list[int]:
        table = self._tables.get(seq.seq_id)
        if table is None:
            table = list(seq.block_table)
            self._tables[seq.seq_id] = table
        return table

    def _pop_free(self) -> int:
        if not self._free:
            raise RuntimeError("out of KV blocks")
        bid = self._free.pop()
        block = self.blocks[bid]
        block.ref_count = 1
        block.num_tokens = 0
        block.block_hash = None
        return bid

    def _first_writable(self, table: list[int]) -> int | None:
        """Index of the first not-full block (so reserved empty pages fill in order)."""
        for i, bid in enumerate(table):
            if self.blocks[bid].num_tokens < self.block_size:
                return i
        return None

    def _cow_at(self, table: list[int], idx: int) -> int:
        src = table[idx]
        src_block = self.blocks[src]
        assert_debug(src_block.ref_count > 1, "I2.4 COW without share")
        dst = self._pop_free()
        dst_block = self.blocks[dst]
        dst_block.num_tokens = src_block.num_tokens
        dst_block.block_hash = None
        if self._kv is not None:
            self._kv[:, :, dst].copy_(self._kv[:, :, src])
        src_block.ref_count -= 1
        table[idx] = dst
        return dst

    def _check_invariants(self) -> None:
        if not DEBUG:
            return
        free_set = set(self._free)
        if len(free_set) != len(self._free):
            raise AssertionError("duplicate id on free list")
        for bid, block in enumerate(self.blocks):
            if block.ref_count < 0:
                raise AssertionError(f"negative ref_count on block {bid}")
            if block.ref_count == 0:
                if bid not in free_set:
                    raise AssertionError(f"I2.2 leaked block {bid}")
            elif bid in free_set:
                raise AssertionError(f"I2.2 live block {bid} on free list")
            if block.num_tokens > self.block_size:
                raise AssertionError("I2.5")
        live_slots = sum(len(t) for t in self._tables.values())
        pins = sum(self._pin_count.values())
        refs = sum(b.ref_count for b in self.blocks)
        if refs != live_slots + pins:
            raise AssertionError(
                f"refcount conservation: sum(ref)={refs} live_slots={live_slots} pins={pins}"
            )
