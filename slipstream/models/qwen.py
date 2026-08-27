"""Qwen2/2.5 architecture. Phase 1 (A1)."""

from __future__ import annotations

import torch

from slipstream.core.config import ModelConfig
from slipstream.models.causal_lm import CausalLM


class QwenForCausalLM(CausalLM):
    """Qwen-family causal LM. Must not import transformers modeling code."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)
