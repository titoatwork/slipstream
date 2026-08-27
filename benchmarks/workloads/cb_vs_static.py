"""Continuous batching vs sequential (static-one-at-a-time) on T0.

Usage:
    python -m benchmarks.workloads.cb_vs_static
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from slipstream.core.config import EngineConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request
from slipstream.engine.llm_engine import LLMEngine

from benchmarks.baselines.hf_generate import DEFAULT_PROMPTS
from benchmarks.manifest import write_run_manifest


def main() -> None:
    prompts = (DEFAULT_PROMPTS * 3)[:16]
    params = SamplingParams(max_tokens=32, temperature=0.0)
    engine = LLMEngine(EngineConfig.for_model("Qwen/Qwen2.5-0.5B"))

    # warmup
    engine.generate(
        Request("w", prompts[0], None, SamplingParams(max_tokens=4, temperature=0.0), 0.0)
    )

    t0 = time.perf_counter()
    seq_out = [
        engine.generate(Request(f"s{i}", p, None, params, float(i))) for i, p in enumerate(prompts)
    ]
    sequential_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch_out = engine.generate_batch(
        [Request(f"b{i}", p, None, params, float(i)) for i, p in enumerate(prompts)]
    )
    batch_s = time.perf_counter() - t0

    toks = sum(len(x) for x in seq_out)
    result = {
        "n_requests": len(prompts),
        "max_tokens": 32,
        "output_tokens": toks,
        "sequential_s": sequential_s,
        "continuous_batch_s": batch_s,
        "speedup": sequential_s / batch_s if batch_s else 0.0,
        "sequential_tok_s": toks / sequential_s if sequential_s else 0.0,
        "cb_tok_s": toks / batch_s if batch_s else 0.0,
        "token_match": batch_out == seq_out,
    }
    dest = Path("benchmarks/results/phase2")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cb_vs_static.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_manifest(
        dest / "run_manifest.json",
        model="Qwen/Qwen2.5-0.5B",
        workload="phase2_cb_vs_static",
        config=result,
        notes="Continuous batching vs sequential generate on T0.",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
