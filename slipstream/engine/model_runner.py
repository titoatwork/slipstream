"""Batch assembly + forward. CUDA-graph capture lands in Phase 4."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from slipstream.core.config import EngineConfig
from slipstream.core.types import SchedulerOutput

if TYPE_CHECKING:
    from slipstream.memory.contiguous_cache import NaiveKVCache
    from slipstream.memory.paged_cache import PagedForward
    from slipstream.models.causal_lm import CausalLM

    CacheLike = NaiveKVCache | PagedForward
else:
    CacheLike = object


class ModelRunner:
    """Single-sequence eager forward. Continuous batching is Phase 2+."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        model: CausalLM | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if dtype is not None else torch.bfloat16

    def run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: CacheLike,
    ) -> torch.Tensor:
        """``input_ids``/``positions`` ``[B, T]`` → logits ``[B, T, vocab]``."""
        if self.model is None:
            raise RuntimeError("ModelRunner has no model")
        logits: torch.Tensor = self.model(input_ids, positions, kv_cache)
        return logits

    def prefill(self, token_ids: torch.Tensor, kv_cache: NaiveKVCache) -> torch.Tensor:
        """Forward ``token_ids`` ``[B, T]`` at packed positions starting at ``seq_len``.

        Same formula as ``decode``. A second prefill or an uncached suffix
        therefore continues RoPE instead of restarting at 0.
        """
        return self.decode(token_ids, kv_cache)

    def decode(self, token_ids: torch.Tensor, kv_cache: NaiveKVCache) -> torch.Tensor:
        """Forward ``token_ids`` ``[B, T]`` at positions starting at ``kv_cache.seq_len``."""
        batch, seqlen = token_ids.shape
        start = kv_cache.seq_len
        positions = (
            torch.arange(start, start + seqlen, device=token_ids.device)
            .unsqueeze(0)
            .expand(batch, seqlen)
        )
        return self.run(token_ids, positions, kv_cache)

    def execute(self, scheduled: SchedulerOutput) -> object:
        """Phase 1 generate path uses ``run``; kept so EngineCore imports stay valid."""
        if self.model is None:
            raise RuntimeError("ModelRunner.execute requires a loaded model")
        return scheduled
