"""Draft-model speculative decoding. Phase 5 (A4)."""

from __future__ import annotations


class SpeculativeConfig:
    draft_model_id: str = "meta-llama/Llama-3.2-1B"
    num_speculative_tokens: int = 4
