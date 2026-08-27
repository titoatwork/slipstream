"""Per-step metrics. Collection lands with the engine; dashboard in Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field

from slipstream.observability.goodput import (
    RequestTrace,
    meets_slo,
    percentile,
    summarize_goodput,
)

__all__ = [
    "MetricsSnapshot",
    "RequestTrace",
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
