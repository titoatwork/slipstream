"""Mixed-load ITL: long prefill vs ongoing decodes, chunk on/off.

python -m benchmarks.workloads.mixed_itl
"""

from __future__ import annotations

import gc
import json
import statistics
from pathlib import Path

import torch
from slipstream.core.config import CacheConfig, EngineConfig, SchedulerConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request
from slipstream.engine.llm_engine import LLMEngine

from benchmarks.manifest import write_run_manifest


def _run(chunk: bool) -> list[float]:
    sched = SchedulerConfig(
        enable_chunked_prefill=chunk,
        prefill_chunk_size=64,
        max_num_batched_tokens=64 if chunk else 8192,
        max_num_seqs=8,
    )
    engine = LLMEngine(
        EngineConfig.for_model(
            "Qwen/Qwen2.5-0.5B",
            scheduler=sched,
            cache=CacheConfig(num_gpu_blocks=1024),
        )
    )
    engine.last_itl_s.clear()
    decodes = [
        Request(
            f"d{i}",
            "Once upon a time in a small village",
            None,
            SamplingParams(max_tokens=24, temperature=0.0),
            float(i),
        )
        for i in range(4)
    ]
    long_prompt = ("Explain PagedAttention in two sentences. " * 200).strip()
    burst = Request(
        "prefill", long_prompt, None, SamplingParams(max_tokens=8, temperature=0.0), 10.0
    )
    try:
        engine.generate_batch(decodes + [burst])
        return list(engine.last_itl_s)
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _p99(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(round(0.99 * (len(ys) - 1))))]


def main() -> None:
    off = _run(False)
    on = _run(True)
    result = {
        "itl_off_p99_s": _p99(off),
        "itl_on_p99_s": _p99(on),
        "itl_off_median_s": statistics.median(off) if off else 0.0,
        "itl_on_median_s": statistics.median(on) if on else 0.0,
        "n_off": len(off),
        "n_on": len(on),
        "p99_cut": (1.0 - _p99(on) / _p99(off)) if off and _p99(off) else 0.0,
    }
    dest = Path("benchmarks/results/phase3")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "mixed_itl.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_manifest(
        dest / "run_manifest.json",
        model="Qwen/Qwen2.5-0.5B",
        workload="phase3_mixed_itl",
        config=result,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
