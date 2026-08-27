"""CPU swap space for preempted KV blocks."""

from __future__ import annotations

import torch


class CpuSwapSpace:
    """Host-side block pool. Same page layout as the GPU cache, minus the layer-0 dim wait — full [L,2,N,...]."""

    def __init__(
        self,
        num_cpu_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        if num_cpu_blocks < 1:
            raise ValueError("num_cpu_blocks must be >= 1")
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        self.block_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype.itemsize * block_size
        self.data = torch.zeros(
            (num_layers, 2, num_cpu_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device="cpu",
        )
        self._free: list[int] = list(range(num_cpu_blocks))

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise RuntimeError("out of CPU swap blocks")
        out = [self._free.pop() for _ in range(n)]
        return out

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self._free.append(bid)
