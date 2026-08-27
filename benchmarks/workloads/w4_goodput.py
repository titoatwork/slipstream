"""W4-like goodput: FCFS vs Horizon vs Oracle under a tight KV cap.

    python -m benchmarks.workloads.w4_goodput

Deterministic lengths (`ignore_eos=True`) so Oracle remaining is exact.
Warmup once so CUDA compile is not billed to the first policy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from slipstream.core.config import CacheConfig, EngineConfig, SchedulerConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request, Sequence
from slipstream.engine.llm_engine import LLMEngine
from slipstream.observability.metrics import RequestTrace, summarize_goodput
from slipstream.scheduler.predictor.features import FeatureSet
from slipstream.scheduler.predictor.length_model import LengthPredictor

from benchmarks.manifest import write_run_manifest

# Paper SLOs (A100/chat). Unattainable for an 18-req burst on T0 0.5B.
PAPER_TTFT_S = 2.0
PAPER_TPOT_S = 0.200
# T0-scaled SLOs: first-token and ITL that a 3050 can actually hit.
T0_TTFT_S = 30.0
T0_TPOT_S = 1.0

# KV must bind before max_num_seqs. 24 pages × 16 tok = 384 slots.
# 12 seqs × ~3–8 peak pages > 24, so FCFS grows into ensure_slot preemption.
NUM_GPU_BLOCKS = 24
MAX_NUM_SEQS = 12


def _requests(n: int = 18, *, mode: str = "bimodal") -> list[Request]:
    reqs: list[Request] = []
    for i in range(n):
        long = False if mode == "homogeneous" else i % 3 == 0
        prompt = ("def fibonacci(n): " if long else "Hi there. ") + f"id={i} "
        prompt = prompt + ("x " * (40 if long else 8))
        out = 48 if long else 12
        reqs.append(
            Request(
                request_id=f"w4-{mode}-{i}",
                prompt=prompt,
                prompt_token_ids=None,
                sampling_params=SamplingParams(
                    max_tokens=out,
                    temperature=0.0,
                    ignore_eos=True,
                    extra={"code_like": 1.0 if long else 0.0},
                ),
                arrival_ts=i * 0.01,
                slo_ttft_ms=T0_TTFT_S * 1000.0,
                slo_tpot_ms=T0_TPOT_S * 1000.0,
            )
        )
    return reqs


def _engine(policy: str) -> LLMEngine:
    return LLMEngine(
        EngineConfig.for_model(
            "Qwen/Qwen2.5-0.5B",
            cache=CacheConfig(
                num_gpu_blocks=NUM_GPU_BLOCKS,
                num_cpu_blocks=16,
                enable_prefix_caching=False,
            ),
            scheduler=SchedulerConfig(
                policy=policy,
                max_num_seqs=MAX_NUM_SEQS,
                max_num_batched_tokens=128,
                prefill_chunk_size=64,
                safety_factor=0.80,
                starvation_guard_ms=2_000.0,
            ),
        )
    )


def _drop(engine: LLMEngine) -> None:
    import gc

    import torch

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _warmup() -> None:
    engine = _engine("fcfs")
    engine.generate(
        Request(
            "warmup",
            "Hi there.",
            None,
            SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
            0.0,
        )
    )
    _drop(engine)


def _dump_traces(traces: list[RequestTrace]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for t in traces:
        out.append(
            {
                "request_id": t.request_id,
                "seq_id": t.seq_id,
                "ttft_s": t.ttft_s,
                "itls": list(t.itls),
                "output_len": t.output_len,
                "aborted": t.aborted,
            }
        )
    return out


def _run(policy: str, reqs: list[Request]) -> dict[str, object]:
    engine = _engine(policy)
    t0 = time.perf_counter()
    outs = engine.generate_batch(reqs)
    wall = time.perf_counter() - t0
    traces = list(engine.last_request_traces)
    paper = summarize_goodput(traces, wall, PAPER_TTFT_S, PAPER_TPOT_S)
    t0slo = summarize_goodput(traces, wall, T0_TTFT_S, T0_TPOT_S)
    tight = summarize_goodput(traces, wall, PAPER_TTFT_S, 0.020)
    pred = getattr(engine.scheduler.policy, "predictor", None) if engine.scheduler else None
    preemptions = float(engine.scheduler.preemptions) if engine.scheduler else 0.0
    result: dict[str, object] = {
        "policy": policy,
        "mean_decoded": float(sum(len(o) for o in outs) / max(len(outs), 1)),
        "predictor_mean": float(getattr(pred, "mean_output", 0.0) or 0.0),
        "preemptions": preemptions,
        "paper_slo": paper,
        "t0_slo": t0slo,
        "tight_slo": tight,
        "traces": _dump_traces(traces),
        # Headline aliases: T0-scaled smooth + completion rate.
        "smooth_goodput": t0slo["smooth_goodput"],
        "naive_goodput": t0slo["naive_goodput"],
        "p99_ttft_s": t0slo["p99_ttft_s"],
        "p50_ttft_s": t0slo["p50_ttft_s"],
        "p99_itl_s": t0slo["p99_itl_s"],
        "wall_s": wall,
        "n_slo": t0slo["n_slo"],
        "completed": t0slo["completed"],
    }
    _drop(engine)
    return result


def _predictor_overhead() -> dict[str, float]:
    pred = LengthPredictor(FeatureSet.F2)
    seq = Sequence(seq_id=1, prompt_token_ids=list(range(64)))
    for _ in range(32):
        pred.observe(seq, 24)
    t0 = time.perf_counter()
    n = 10_000
    for _ in range(n):
        pred.predict_remaining(seq)
    dt = time.perf_counter() - t0
    per_us = (dt / n) * 1e6
    return {
        "n": float(n),
        "total_s": dt,
        "per_predict_us": per_us,
        "vs_50ms_step_pct": (per_us / 1e3) / 50.0 * 100.0,
    }


def _gains(by: dict[str, dict[str, object]]) -> dict[str, float]:
    fcfs_g = float(by["fcfs"]["smooth_goodput"])
    hor_g = float(by["horizon"]["smooth_goodput"])
    ora_g = float(by["oracle"]["smooth_goodput"])
    fcfs_n = float(by["fcfs"]["naive_goodput"])
    hor_n = float(by["horizon"]["naive_goodput"])
    ora_n = float(by["oracle"]["naive_goodput"])
    return {
        "horizon_gain_vs_fcfs_smooth": (hor_g / fcfs_g - 1.0) if fcfs_g else 0.0,
        "horizon_gain_vs_fcfs_naive": (hor_n / fcfs_n - 1.0) if fcfs_n else 0.0,
        "gap_closed_to_oracle_smooth": (
            (hor_g - fcfs_g) / (ora_g - fcfs_g) if ora_g > fcfs_g else 0.0
        ),
        "gap_closed_to_oracle_naive": (
            (hor_n - fcfs_n) / (ora_n - fcfs_n) if ora_n > fcfs_n else 0.0
        ),
        "p99_ttft_ratio_vs_fcfs": (
            float(by["horizon"]["p99_ttft_s"]) / float(by["fcfs"]["p99_ttft_s"])
            if by["fcfs"]["p99_ttft_s"]
            else 0.0
        ),
        "wall_ratio_fcfs_over_horizon": (
            float(by["fcfs"]["wall_s"]) / float(by["horizon"]["wall_s"])
            if by["horizon"]["wall_s"]
            else 0.0
        ),
    }


def main() -> None:
    _warmup()
    reqs = _requests(18, mode="bimodal")
    # Horizon/oracle first after warmup; FCFS last so compile cannot inflate FCFS.
    rows = [_run(p, reqs) for p in ("horizon", "oracle", "fcfs")]
    by = {str(r["policy"]): r for r in rows}
    homo_reqs = _requests(12, mode="homogeneous")
    homo = [_run(p, homo_reqs) for p in ("horizon", "fcfs")]
    pred_cost = _predictor_overhead()
    fcfs_pre = float(by["fcfs"]["preemptions"])
    summary = {
        "warmup": True,
        "kv_blocks": NUM_GPU_BLOCKS,
        "max_num_seqs": MAX_NUM_SEQS,
        "kv_bound": fcfs_pre > 0,
        "fcfs_preemptions": fcfs_pre,
        "paper_slo": {"ttft_s": PAPER_TTFT_S, "tpot_s": PAPER_TPOT_S},
        "t0_slo": {"ttft_s": T0_TTFT_S, "tpot_s": T0_TPOT_S},
        "runs": rows,
        "gains": _gains(by),
        "homogeneous": homo,
        "predictor_overhead": pred_cost,
        "notes": (
            "Warmed CUDA/kernels before the timed policies. "
            "ignore_eos so oracle_output_len == max_tokens. "
            f"KV {NUM_GPU_BLOCKS} blocks, max_num_seqs={MAX_NUM_SEQS}. "
            "Valid length-prediction experiment iff kv_bound (FCFS preempts). "
            "Headline smooth goodput uses T0 SLOs (30s / 1.0s). "
            "paper_slo is the 2s/200ms over-capacity regime. "
            "Aborts never count as SLO hits."
        ),
    }
    dest = Path("benchmarks/results/phase4")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "w4_goodput.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_run_manifest(
        dest / "run_manifest.json",
        model="Qwen/Qwen2.5-0.5B",
        workload="phase4_w4_goodput",
        config={k: v for k, v in summary.items() if k not in {"runs", "homogeneous"}},
        notes="Warmed, tight KV, bimodal, ignore_eos. Same request list per policy.",
    )
    printable = json.loads(json.dumps(summary))
    for row in printable.get("runs", []) + printable.get("homogeneous", []):
        row.pop("traces", None)
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
