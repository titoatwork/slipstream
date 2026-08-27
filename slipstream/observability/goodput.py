"""SLO goodput. Smooth goodput never counts aborted requests as hits."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestTrace:
    """Per-request latency for honest goodput (never drop-to-win)."""

    request_id: str
    seq_id: int
    ttft_s: float | None = None
    wait_s: float | None = None
    prefill_s: float | None = None
    admit_s: float | None = None
    itls: list[float] = field(default_factory=list)
    output_len: int = 0
    aborted: bool = False
    arrival_ts: float = 0.0
    finish_ts: float | None = None

    @property
    def mean_itl_s(self) -> float:
        return sum(self.itls) / len(self.itls) if self.itls else 0.0

    def fluidity(self, slo_tpot_s: float) -> float:
        if not self.itls:
            return 1.0
        return sum(1 for x in self.itls if x <= slo_tpot_s) / len(self.itls)

    def meets_slo(self, ttft_slo: float, tpot_slo: float, *, strict_itl: bool = True) -> bool:
        if self.aborted or self.ttft_s is None:
            return False
        if self.ttft_s > ttft_slo:
            return False
        if strict_itl:
            return self.fluidity(tpot_slo) >= 1.0
        return not (self.itls and self.mean_itl_s > tpot_slo)


def meets_slo(
    trace: RequestTrace, ttft_slo: float, tpot_slo: float, *, strict_itl: bool = True
) -> bool:
    return trace.meets_slo(ttft_slo, tpot_slo, strict_itl=strict_itl)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def summarize_goodput(
    traces: list[RequestTrace],
    wall_s: float,
    slo_ttft_s: float,
    slo_tpot_s: float,
) -> dict[str, float]:
    """Naive = completed/wall. Smooth = SLO hits/wall; aborts never count as hits."""
    n_ok = sum(1 for t in traces if not t.aborted and t.output_len > 0)
    n_slo = sum(1 for t in traces if t.meets_slo(slo_ttft_s, slo_tpot_s, strict_itl=True))
    n_slo_mean = sum(1 for t in traces if t.meets_slo(slo_ttft_s, slo_tpot_s, strict_itl=False))
    n_aborted = sum(1 for t in traces if t.aborted)
    ttfts = [t.ttft_s for t in traces if t.ttft_s is not None]
    waits = [t.wait_s for t in traces if t.wait_s is not None]
    prefills = [t.prefill_s for t in traces if t.prefill_s is not None]
    itls = [x for t in traces for x in t.itls]
    wall = max(wall_s, 1e-9)
    n_ok_f = float(n_ok)
    return {
        "n_ok": n_ok_f,
        "completed": n_ok_f,
        "n_slo": float(n_slo),
        "n_slo_mean_tpot": float(n_slo_mean),
        "n_aborted": float(n_aborted),
        "n_traces": float(len(traces)),
        "naive_goodput": n_ok / wall,
        "mean_slo_goodput": n_slo_mean / wall,
        "smooth_goodput": n_slo / wall,
        "p50_ttft_s": percentile(ttfts, 50),
        "p99_ttft_s": percentile(ttfts, 99),
        "p50_wait_s": percentile(waits, 50),
        "p99_wait_s": percentile(waits, 99),
        "p50_prefill_s": percentile(prefills, 50),
        "p99_prefill_s": percentile(prefills, 99),
        "p50_itl_s": percentile(itls, 50),
        "p99_itl_s": percentile(itls, 99),
        "mean_output": (sum(t.output_len for t in traces) / max(len(traces), 1) if traces else 0.0),
        "wall_s": wall_s,
    }
