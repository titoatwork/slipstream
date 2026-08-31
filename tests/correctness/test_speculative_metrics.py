"""SpeculativeMetrics: folds SpeculativeStats for the Gate 5 acceptance table. S8."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slipstream.observability.metrics import SpeculativeMetrics, StepMetrics  # noqa: E402
from slipstream.speculative.decode import SpeculativeStats  # noqa: E402


def test_speculative_metrics_accumulates_across_generations() -> None:
    m = SpeculativeMetrics()
    assert m.acceptance_rate == 0.0  # empty is 0, not a ZeroDivisionError
    assert m.mean_accepted_len == 0.0

    m.update(SpeculativeStats(steps=2, proposed=8, accepted=6, emitted=10))
    m.update(SpeculativeStats(steps=1, proposed=4, accepted=1, emitted=2))

    assert m.steps == 3
    assert m.proposed == 12
    assert m.accepted == 7
    assert m.emitted == 12
    assert m.acceptance_rate == pytest.approx(7 / 12)
    assert m.mean_accepted_len == pytest.approx(12 / 3)


def test_step_metrics_spec_fields_default_off() -> None:
    # Non-speculative steps leave the spec fields None (nothing else changes).
    step = StepMetrics(
        step=1,
        num_running=1,
        num_waiting=0,
        num_swapped=0,
        batch_size=1,
        num_batched_tokens=1,
        kv_utilization=0.0,
        prefill_tokens=0,
        decode_tokens=1,
        step_time_ms=0.0,
    )
    assert step.spec_acceptance_rate is None
    assert step.spec_mean_accepted_len is None

    stats = SpeculativeStats(steps=4, proposed=16, accepted=12, emitted=20)
    stamped = StepMetrics(
        step=2,
        num_running=1,
        num_waiting=0,
        num_swapped=0,
        batch_size=1,
        num_batched_tokens=5,
        kv_utilization=0.0,
        prefill_tokens=0,
        decode_tokens=5,
        step_time_ms=1.0,
        spec_acceptance_rate=stats.acceptance_rate,
        spec_mean_accepted_len=stats.mean_accepted_len,
    )
    assert stamped.spec_acceptance_rate == pytest.approx(0.75)
    assert stamped.spec_mean_accepted_len == pytest.approx(5.0)
