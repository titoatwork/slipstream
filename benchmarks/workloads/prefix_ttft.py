"""Shared-prefix TTFT and hit rate (W3-like).

python -m benchmarks.workloads.prefix_ttft
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from slipstream.core.config import CacheConfig, EngineConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request
from slipstream.engine.llm_engine import LLMEngine

from benchmarks.manifest import write_run_manifest


def main() -> None:
    system = (
        "You are a helpful assistant. Follow the user's instructions carefully. " * 80
    ).strip()
    users = [f"Name three facts about the number {i}." for i in range(8)]
    params = SamplingParams(max_tokens=16, temperature=0.0)
    engine = LLMEngine(
        EngineConfig.for_model("Qwen/Qwen2.5-0.5B", cache=CacheConfig(num_gpu_blocks=1024))
    )

    cold: list[float] = []
    hot: list[float] = []

    def ttft(rid: str, prompt: str, t_arrive: float) -> float:
        req = Request(rid, prompt, None, params, t_arrive)
        t0 = time.perf_counter()
        gen = engine.stream(req)
        next(gen)
        first = time.perf_counter() - t0
        for _ in gen:
            pass
        return first

    for i, user in enumerate(users):
        cold.append(ttft(f"c{i}", system + " " + user, float(i)))
    for i, user in enumerate(users):
        hot.append(ttft(f"h{i}", system + " " + user, 100.0 + i))

    def med(xs: list[float]) -> float:
        ys = sorted(xs)
        return ys[len(ys) // 2]

    result = {
        "n": len(users),
        "cold_ttft_median_s": med(cold),
        "hot_ttft_median_s": med(hot),
        "ttft_cut": 1.0 - med(hot) / med(cold) if med(cold) else 0.0,
        "hit_rate": engine.prefix_cache.hit_rate() if engine.prefix_cache else 0.0,
        "cached_tokens": engine.prefix_cache.cached_tokens if engine.prefix_cache else 0,
        "queried_tokens": engine.prefix_cache.queried_tokens if engine.prefix_cache else 0,
    }
    dest = Path("benchmarks/results/phase3")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "prefix_ttft.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_manifest(
        dest / "run_manifest_prefix.json",
        model="Qwen/Qwen2.5-0.5B",
        workload="phase3_prefix_ttft",
        config=result,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
