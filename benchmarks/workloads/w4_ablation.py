"""W4 ablation: f0 vs f2, plus open-loop arrivals.

    python -m benchmarks.workloads.w4_ablation

Does not overwrite w4_goodput.json (the binding closed-burst cell).
"""

from __future__ import annotations

import json
from pathlib import Path

from slipstream.core.types import Request
from slipstream.scheduler.policies.horizon import HorizonPolicy
from slipstream.scheduler.predictor.features import FeatureSet

from benchmarks.workloads.w4_goodput import (
    MAX_NUM_SEQS,
    NUM_GPU_BLOCKS,
    T0_TPOT_S,
    T0_TTFT_S,
    _drop,
    _engine,
    _gains,
    _predictor_overhead,
    _requests,
    _warmup,
)

OPEN_LOOP_GAP_S = 0.20


def _open_loop_requests(n: int = 18) -> list[Request]:
    reqs = _requests(n, mode="bimodal")
    out: list[Request] = []
    for i, req in enumerate(reqs):
        out.append(
            Request(
                request_id=req.request_id,
                prompt=req.prompt,
                prompt_token_ids=req.prompt_token_ids,
                sampling_params=req.sampling_params,
                arrival_ts=i * OPEN_LOOP_GAP_S,
                slo_ttft_ms=req.slo_ttft_ms,
                slo_tpot_ms=req.slo_tpot_ms,
            )
        )
    return out


def _run_f0(reqs: list[Request]) -> dict[str, object]:
    engine = _engine("horizon")
    assert engine.scheduler is not None
    cfg = engine.config.scheduler
    engine.scheduler.policy = HorizonPolicy(
        feature_set=FeatureSet.F0,
        safety_factor=cfg.safety_factor,
        starvation_guard_ms=cfg.starvation_guard_ms,
    )
    # Reuse the timed path by calling generate_batch then summarizing like _run.
    import time

    from slipstream.observability.metrics import summarize_goodput

    t0 = time.perf_counter()
    outs = engine.generate_batch(reqs)
    wall = time.perf_counter() - t0
    traces = list(engine.last_request_traces)
    t0slo = summarize_goodput(traces, wall, T0_TTFT_S, T0_TPOT_S)
    pred = engine.scheduler.policy.predictor
    result: dict[str, object] = {
        "policy": "horizon-f0",
        "mean_decoded": float(sum(len(o) for o in outs) / max(len(outs), 1)),
        "predictor_mean": float(pred.mean_output),
        "preemptions": float(engine.scheduler.preemptions),
        "smooth_goodput": t0slo["smooth_goodput"],
        "naive_goodput": t0slo["naive_goodput"],
        "p99_ttft_s": t0slo["p99_ttft_s"],
        "p50_ttft_s": t0slo["p50_ttft_s"],
        "p99_itl_s": t0slo["p99_itl_s"],
        "wall_s": wall,
        "n_slo": t0slo["n_slo"],
        "completed": t0slo["completed"],
        "t0_slo": t0slo,
    }
    _drop(engine)
    return result


def _run_open(policy: str, reqs: list[Request]) -> dict[str, object]:
    engine = _engine(policy)
    import time

    from slipstream.observability.metrics import summarize_goodput

    t0 = time.perf_counter()
    outs = engine.generate_batch(reqs, inject="arrival")
    wall = time.perf_counter() - t0
    traces = list(engine.last_request_traces)
    t0slo = summarize_goodput(traces, wall, T0_TTFT_S, T0_TPOT_S)
    pred = getattr(engine.scheduler.policy, "predictor", None) if engine.scheduler else None
    pre = float(engine.scheduler.preemptions) if engine.scheduler else 0.0
    result: dict[str, object] = {
        "policy": policy,
        "inject": "arrival",
        "mean_decoded": float(sum(len(o) for o in outs) / max(len(outs), 1)),
        "predictor_mean": float(getattr(pred, "mean_output", 0.0) or 0.0),
        "preemptions": pre,
        "smooth_goodput": t0slo["smooth_goodput"],
        "naive_goodput": t0slo["naive_goodput"],
        "p99_ttft_s": t0slo["p99_ttft_s"],
        "p50_ttft_s": t0slo["p50_ttft_s"],
        "p99_itl_s": t0slo["p99_itl_s"],
        "wall_s": wall,
        "n_slo": t0slo["n_slo"],
        "completed": t0slo["completed"],
        "t0_slo": t0slo,
    }
    _drop(engine)
    return result


def main() -> None:
    _warmup()
    closed = _requests(18, mode="bimodal")
    f0 = _run_f0(closed)
    # Same closed list as w4_goodput so f0 is comparable to the stored f2/fcfs/oracle.
    open_reqs = _open_loop_requests(18)
    open_rows = [_run_open(p, open_reqs) for p in ("horizon", "oracle", "fcfs")]
    open_by = {str(r["policy"]): r for r in open_rows}
    summary = {
        "kv_blocks": NUM_GPU_BLOCKS,
        "max_num_seqs": MAX_NUM_SEQS,
        "open_loop_gap_s": OPEN_LOOP_GAP_S,
        "closed_horizon_f0": f0,
        "open_loop": open_rows,
        "open_loop_gains": _gains(open_by),
        "open_loop_kv_bound": float(open_by["fcfs"]["preemptions"]) > 0,
        "predictor_overhead": _predictor_overhead(),
        "notes": (
            "Closed f0 is the mean-only predictor on the same 18-req burst as "
            "w4_goodput.json. Open-loop inject=arrival, gap="
            f"{OPEN_LOOP_GAP_S}s. T0-smooth SLO {T0_TTFT_S}s/{T0_TPOT_S}s."
        ),
    }
    dest = Path("benchmarks/results/phase4")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "w4_ablation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
