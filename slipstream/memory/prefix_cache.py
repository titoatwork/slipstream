"""Radix prefix cache. Only full KV blocks are shared."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slipstream.memory.block_manager import BlockManagerImpl


class _Node:
    __slots__ = ("children", "block_id", "last_used")

    def __init__(self) -> None:
        self.children: dict[int, _Node] = {}
        self.block_id: int | None = None
        self.last_used: float = 0.0


class RadixPrefixCache:
    """Token-sequence radix tree. Pins full blocks on the BlockManager."""

    def __init__(self, block_size: int = 16) -> None:
        self.block_size = block_size
        self.root = _Node()
        self._manager: BlockManagerImpl | None = None
        self.hits = 0
        self.queries = 0
        self.cached_tokens = 0
        self.queried_tokens = 0

    def bind(self, manager: BlockManagerImpl) -> None:
        self._manager = manager
        self.block_size = manager.block_size

    def match(self, token_ids: list[int]) -> tuple[list[int], int]:
        self.queries += 1
        self.queried_tokens += len(token_ids)
        node = self.root
        blocks: list[int] = []
        n_tokens = 0
        now = time.monotonic()
        for i, tok in enumerate(token_ids):
            child = node.children.get(tok)
            if child is None:
                break
            node = child
            node.last_used = now
            if (i + 1) % self.block_size == 0 and node.block_id is not None:
                blocks.append(node.block_id)
                n_tokens = i + 1
        if n_tokens:
            self.hits += 1
            self.cached_tokens += n_tokens
        return blocks, n_tokens

    def insert(self, token_ids: list[int], block_ids: list[int]) -> None:
        n_blocks = min(len(block_ids), len(token_ids) // self.block_size)
        if n_blocks <= 0:
            return
        manager = self._manager
        node = self.root
        now = time.monotonic()
        for i, tok in enumerate(token_ids[: n_blocks * self.block_size]):
            child = node.children.get(tok)
            if child is None:
                child = _Node()
                node.children[tok] = child
            node = child
            node.last_used = now
            if (i + 1) % self.block_size == 0:
                bid = block_ids[(i + 1) // self.block_size - 1]
                if node.block_id is None:
                    node.block_id = bid
                    if manager is not None:
                        manager.pin([bid])
                # If the tree already points at a different physical page for
                # the same token prefix, keep the first (stable identity).

    def evict(self, n_blocks: int) -> int:
        if n_blocks <= 0:
            return 0
        ranked = self._nodes_with_blocks()
        ranked.sort(key=lambda item: item[0].last_used)
        evicted = 0
        manager = self._manager
        for node, _parent, _tok in ranked:
            if evicted >= n_blocks:
                break
            if node.block_id is None:
                continue
            bid = node.block_id
            node.block_id = None
            if manager is not None:
                manager.unpin([bid])
            evicted += 1
        return evicted

    def hit_rate(self) -> float:
        if self.queried_tokens == 0:
            return 0.0
        return self.cached_tokens / self.queried_tokens

    def _nodes_with_blocks(self) -> list[tuple[_Node, _Node | None, int | None]]:
        out: list[tuple[_Node, _Node | None, int | None]] = []

        def walk(node: _Node, parent: _Node | None, tok: int | None) -> None:
            if node.block_id is not None:
                out.append((node, parent, tok))
            for t, child in node.children.items():
                walk(child, node, t)

        walk(self.root, None, None)
        return out
