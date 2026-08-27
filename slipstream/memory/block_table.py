"""Per-sequence block table helpers. Implementation lands in Phase 2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockTableView:
    """Read-only view of a sequence's logical → physical block map."""

    block_ids: tuple[int, ...]
    block_size: int

    def physical_slot(self, logical_token: int) -> tuple[int, int]:
        """Return (physical_block_id, offset) for a logical token index."""
        raise NotImplementedError("Phase 2 — S2 BlockTableView.physical_slot")

    def slot_mapping(self, num_tokens: int) -> list[int]:
        """Flat slot indices: `block_id * block_size + offset` per token."""
        raise NotImplementedError("Phase 2 — S2 BlockTableView.slot_mapping")
