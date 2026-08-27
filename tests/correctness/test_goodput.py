"""Honest goodput helpers — aborted requests never count as SLO hits."""

from __future__ import annotations

from slipstream.observability.goodput import (
    RequestTrace,
    meets_slo,
    percentile,
    summarize_goodput,
)


def test_meets_slo_requires_ttft_and_tpot() -> None:
    ok = RequestTrace("a", 1, ttft_s=0.4, itls=[0.05, 0.05], output_len=3)
    assert ok.meets_slo(2.0, 0.2)
    assert meets_slo(ok, 2.0, 0.2)
    slow_ttft = RequestTrace("b", 2, ttft_s=3.0, itls=[0.05], output_len=2)
    assert not slow_ttft.meets_slo(2.0, 0.2)
    slow_tpot = RequestTrace("c", 3, ttft_s=0.1, itls=[0.5, 0.5], output_len=3)
    assert not slow_tpot.meets_slo(2.0, 0.2)
    # One stall fails strict (smooth) but can pass mean-TPOT (naive SLO).
    stalled = RequestTrace("e", 5, ttft_s=0.1, itls=[0.05, 0.05, 0.50], output_len=4)
    assert not stalled.meets_slo(2.0, 0.2, strict_itl=True)
    assert stalled.meets_slo(2.0, 0.2, strict_itl=False)


def test_aborted_never_counts() -> None:
    dropped = RequestTrace("d", 4, ttft_s=0.1, itls=[0.05], output_len=1, aborted=True)
    assert not dropped.meets_slo(2.0, 0.2)
    assert not meets_slo(dropped, 2.0, 0.2)
    summary = summarize_goodput([dropped], wall_s=1.0, slo_ttft_s=2.0, slo_tpot_s=0.2)
    assert summary["smooth_goodput"] == 0.0
    assert summary["n_slo"] == 0.0
    assert summary["n_aborted"] == 1.0
    assert summary["n_ok"] == 0.0
    assert summary["naive_goodput"] == 0.0


def test_summarize_empty() -> None:
    summary = summarize_goodput([], wall_s=2.0, slo_ttft_s=2.0, slo_tpot_s=0.2)
    assert summary["completed"] == 0.0
    assert summary["n_ok"] == 0.0
    assert summary["n_slo"] == 0.0
    assert summary["n_aborted"] == 0.0
    assert summary["smooth_goodput"] == 0.0
    assert summary["naive_goodput"] == 0.0
    assert summary["p50_ttft_s"] == 0.0
    assert summary["p99_ttft_s"] == 0.0
    assert summary["p99_itl_s"] == 0.0


def test_wait_and_prefill_percentiles() -> None:
    traces = [
        RequestTrace("a", 1, ttft_s=0.5, wait_s=0.4, prefill_s=0.1, itls=[0.05], output_len=4),
        RequestTrace("b", 2, ttft_s=0.2, wait_s=0.0, prefill_s=0.2, itls=[0.05], output_len=4),
    ]
    summary = summarize_goodput(traces, wall_s=1.0, slo_ttft_s=2.0, slo_tpot_s=0.2)
    assert summary["p50_wait_s"] >= 0.0
    assert summary["p99_wait_s"] >= summary["p50_wait_s"]
    assert summary["p50_prefill_s"] > 0.0


def test_p99_and_rates() -> None:
    traces = [
        RequestTrace("ok", 1, ttft_s=0.2, itls=[0.05], output_len=8),
        RequestTrace("ok2", 2, ttft_s=0.3, itls=[0.06], output_len=8),
        RequestTrace("miss", 3, ttft_s=5.0, itls=[0.05], output_len=8),
    ]
    summary = summarize_goodput(traces, wall_s=2.0, slo_ttft_s=2.0, slo_tpot_s=0.2)
    assert summary["completed"] == 3.0
    assert summary["n_ok"] == 3.0
    assert summary["n_slo"] == 2.0
    assert summary["smooth_goodput"] == 1.0
    assert summary["naive_goodput"] == 1.5
    assert summary["p99_ttft_s"] >= summary["p50_ttft_s"]
    assert summary["p99_ttft_s"] == 5.0
    assert summary["p50_ttft_s"] == 0.3


def test_p99_of_hundred_ttfts() -> None:
    traces = [
        RequestTrace(str(i), i, ttft_s=(i + 1) / 100.0, itls=[0.01], output_len=2)
        for i in range(100)
    ]
    summary = summarize_goodput(traces, wall_s=10.0, slo_ttft_s=2.0, slo_tpot_s=0.2)
    assert summary["p99_ttft_s"] == percentile([t.ttft_s or 0.0 for t in traces], 99)
    assert summary["p99_ttft_s"] == 0.99
    assert summary["p99_itl_s"] == 0.01
