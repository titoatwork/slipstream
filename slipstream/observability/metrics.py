"""Per-step metrics. Collection lands with the engine; dashboard in Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from slipstream.observability.goodput import (
    RequestTrace,
    meets_slo,
    percentile,
    summarize_goodput,
)

if TYPE_CHECKING:
    from slipstream.speculative.decode import SpeculativeStats

__all__ = [
    "MetricsSnapshot",
    "RequestTrace",
    "SpeculativeMetrics",
    "StepMetrics",
    "meets_slo",
    "percentile",
    "summarize_goodput",
]


@dataclass
class StepMetrics:
    step: int
    num_running: int
    num_waiting: int
    num_swapped: int
    batch_size: int
    num_batched_tokens: int
    kv_utilization: float
    prefill_tokens: int
    decode_tokens: int
    step_time_ms: float
    cache_hit_rate: float = 0.0
    preemptions: int = 0
    mfu: float | None = None
    achieved_bandwidth_gbps: float | None = None
    # Speculative decoding (S8): None when speculation is off this step.
    spec_acceptance_rate: float | None = None
    spec_mean_accepted_len: float | None = None


@dataclass
class SpeculativeMetrics:
    """Rolling acceptance accounting across speculative steps (S8).

    Folds per-generation ``SpeculativeStats`` (from ``SpeculativeRunner`` or the
    reference loop) so the dashboard and the Gate 5 ablation table can report
    acceptance rate and mean accepted length vs draft length. Off the hot path —
    the engine folds one ``SpeculativeStats`` in per finished generation, then
    stamps ``spec_*`` onto the step's ``StepMetrics``.
    """

    steps: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0

    def update(self, stats: SpeculativeStats) -> None:
        """Accumulate one generation's speculative counters."""
        self.steps += stats.steps
        self.proposed += stats.proposed
        self.accepted += stats.accepted
        self.emitted += stats.emitted

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target kept. 0 when nothing proposed."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def mean_accepted_len(self) -> float:
        """Mean tokens emitted per speculative step — the decode-speedup driver."""
        return self.emitted / self.steps if self.steps else 0.0


@dataclass
class MetricsSnapshot:
    """Point-in-time view consumed by /metrics and the dashboard."""

    ttft_ms: list[float] = field(default_factory=list)
    tpot_ms: list[float] = field(default_factory=list)
    itl_ms: list[float] = field(default_factory=list)
    throughput_tok_s: float = 0.0
    goodput_req_s: float = 0.0
    smooth_goodput_req_s: float = 0.0
    kv_utilization: float = 0.0
    last_step: StepMetrics | None = None
