"""Follow-up cells: open-loop f0 + two extra closed-burst seeds.

python -m benchmarks.workloads.w4_followup
"""

from __future__ import annotations

import json
from pathlib import Path

from slipstream.scheduler.policies.horizon import HorizonPolicy
from slipstream.scheduler.predictor.features import FeatureSet

from benchmarks.workloads.w4_ablation import _open_loop_requests
from benchmarks.workloads.w4_goodput import _engine, _gains, _requests, _run, _warmup


def _run_f0_open(reqs: list) -> dict[str, object]:
    engine = _engine("horizon")
    assert engine.scheduler is not None
    cfg = engine.config.scheduler
    engine.scheduler.policy = HorizonPolicy(
        feature_set=FeatureSet.F0,
        safety_factor=cfg.safety_factor,
        starvation_guard_ms=cfg.starvation_guard_ms,
    )
    import time

    from slipstream.observability.metrics import summarize_goodput

    from benchmarks.workloads.w4_goodput import T0_TPOT_S, T0_TTFT_S

    t0 = time.perf_counter()
    outs = engine.generate_batch(reqs, inject="arrival")
    wall = time.perf_counter() - t0
    traces = list(engine.last_request_traces)
    t0slo = summarize_goodput(traces, wall, T0_TTFT_S, T0_TPOT_S)
    from benchmarks.workloads.w4_goodput import _drop

    result: dict[str, object] = {
        "policy": "horizon-f0",
        "inject": "arrival",
        "mean_decoded": float(sum(len(o) for o in outs) / max(len(outs), 1)),
        "predictor_mean": float(engine.scheduler.policy.predictor.mean_output),
        "preemptions": float(engine.scheduler.preemptions),
        "smooth_goodput": t0slo["smooth_goodput"],
        "naive_goodput": t0slo["naive_goodput"],
        "p99_ttft_s": t0slo["p99_ttft_s"],
        "p50_ttft_s": t0slo["p50_ttft_s"],
        "p99_wait_s": t0slo.get("p99_wait_s", 0.0),
        "p50_wait_s": t0slo.get("p50_wait_s", 0.0),
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
    open_reqs = _open_loop_requests(18)
    f0_open = _run_f0_open(open_reqs)
    seeds = []
    closed = _requests(18, mode="bimodal")
    for seed in (1, 2):
        rows = [_run(p, closed) for p in ("horizon", "oracle", "fcfs")]
        by = {str(r["policy"]): r for r in rows}
        seeds.append(
            {
                "seed": seed,
                "kv_bound": float(by["fcfs"]["preemptions"]) > 0,
                "fcfs_preemptions": by["fcfs"]["preemptions"],
                "gains": _gains(by),
                "runs": [
                    {
                        k: r[k]
                        for k in (
                            "policy",
                            "wall_s",
                            "smooth_goodput",
                            "naive_goodput",
                            "n_slo",
                            "preemptions",
                            "p50_ttft_s",
                            "p99_ttft_s",
                            "p99_itl_s",
                        )
                    }
                    for r in rows
                ],
            }
        )
    summary = {
        "open_loop_horizon_f0": f0_open,
        "closed_repeat_seeds": seeds,
        "notes": (
            "Open-loop f0 on the same 0.20s list as w4_ablation. "
            "Closed seeds reuse the binding 24-block burst (not statistical "
            "independence of GPU clocks, but a stability check)."
        ),
    }
    dest = Path("benchmarks/results/phase4")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "w4_followup.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
