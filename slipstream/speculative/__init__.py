"""Draft-model speculative decoding. Phase 5 (A4)."""

from __future__ import annotations

from slipstream.speculative.decode import (
    ScoreFn,
    SpeculativeStats,
    speculative_generate,
)
from slipstream.speculative.rejection import VerifyResult, verify_tokens


class SpeculativeConfig:
    draft_model_id: str = "meta-llama/Llama-3.2-1B"
    num_speculative_tokens: int = 4


__all__ = [
    "ScoreFn",
    "SpeculativeConfig",
    "SpeculativeStats",
    "VerifyResult",
    "speculative_generate",
    "verify_tokens",
]
