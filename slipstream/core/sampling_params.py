"""Sampling parameters. Frozen after Phase 0 — amend §21 to change."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingParams:
    """Per-request sampling configuration.

    `n > 1` forks the sequence after prefill (block-table COW). Seeded
    sampling must be bitwise-reproducible for the same (params, tokens).
    """

    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = -1  # -1 disables
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop_token_ids: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()
    ignore_eos: bool = False
    seed: int | None = None
    n: int = 1

    extra: dict[str, object] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or >= 1")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if not (0.0 <= self.min_p <= 1.0):
            raise ValueError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")
        if self.n < 1:
            raise ValueError("n must be >= 1")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0 or (self.top_k == 1)
