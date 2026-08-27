"""Open-loop arrival release. CPU only — no model."""

from __future__ import annotations

from slipstream.core.types import Sequence
from slipstream.engine.llm_engine import take_ready
from slipstream.scheduler.policies import get_policy
from slipstream.scheduler.predictor.features import FeatureSet


def test_take_ready_keeps_future_arrivals() -> None:
    early = Sequence(seq_id=1, prompt_token_ids=[1], arrival_ts=1.0)
    late = Sequence(seq_id=2, prompt_token_ids=[1], arrival_ts=5.0)
    pending = [early, late]
    got = take_ready(pending, now=2.0)
    assert [s.seq_id for s in got] == [1]
    assert [s.seq_id for s in pending] == [2]


def test_take_ready_empty_when_none_due() -> None:
    pending = [Sequence(seq_id=1, prompt_token_ids=[1], arrival_ts=10.0)]
    assert take_ready(pending, now=0.0) == []
    assert len(pending) == 1


def test_closed_burst_clock_ignores_relative_offset() -> None:
    """inject=all must not treat arrival_ts=2 as a 2s delay or a negative TTFT."""
    t0_wall = 1_700_000_000.0
    t0_perf = 100.0
    arrival = 2.0  # test_generate_batch style
    abs_arr = t0_wall + arrival
    started_all = t0_perf  # inject=all
    started_open = t0_perf + (abs_arr - t0_wall)
    assert started_all == 100.0
    assert started_open == 102.0
    first_token_perf = 100.05
    assert first_token_perf - started_all > 0
    assert first_token_perf - started_open < 0


def test_get_policy_horizon_f0() -> None:
    h = get_policy("horizon", feature_set=FeatureSet.F0)
    assert h.name == "horizon"  # type: ignore[attr-defined]
    assert h.predictor.feature_set is FeatureSet.F0  # type: ignore[attr-defined]
