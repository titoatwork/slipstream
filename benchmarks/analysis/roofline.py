"""Decode roofline / MFU sketch for T0 (and A100 constants for the report).

python -m benchmarks.analysis.roofline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from slipstream.core.config import EngineConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request
from slipstream.engine.llm_engine import LLMEngine

PEAKS = {
    "rtx3050_laptop": {"bw_gbs": 140.0, "tflops_fp16": 9.0},
    "a100_80gb": {"bw_gbs": 2039.0, "tflops_bf16": 312.0},
}


def main() -> None:
    engine = LLMEngine(EngineConfig.for_model("Qwen/Qwen2.5-0.5B"))
    cfg = engine.config.model
    assert cfg.num_layers and cfg.hidden_size
    # Weight bytes (bf16): roughly 2 * params. 0.5B → ~1 GiB
    param_bytes = sum(p.numel() * p.element_size() for p in engine.model.parameters())
    req = Request(
        "roof",
        "The capital of France is",
        None,
        SamplingParams(max_tokens=64, temperature=0.0),
        0.0,
    )
    engine.generate(req)  # warmup
    t0 = time.perf_counter()
    engine.generate(req)
    wall = time.perf_counter() - t0
    out_tokens = 64
    # Decode-dominated: each token reads the weight set once.
    bytes_moved = param_bytes * out_tokens
    achieved_gbs = (bytes_moved / wall) / 1e9
    # FLOPs ~ 2 * N_params * tokens
    flops = 2.0 * (param_bytes / 2.0) * out_tokens
    tflops = (flops / wall) / 1e12
    t0_peak = PEAKS["rtx3050_laptop"]
    result = {
        "model": "Qwen/Qwen2.5-0.5B",
        "param_bytes": param_bytes,
        "wall_s_64tok": wall,
        "achieved_gbs": achieved_gbs,
        "t0_peak_gbs": t0_peak["bw_gbs"],
        "frac_t0_bw": achieved_gbs / t0_peak["bw_gbs"],
        "tflops": tflops,
        "mfu_t0": tflops / t0_peak["tflops_fp16"],
        "a100_ridge_batch": 153,
        "note": "Decode AI ≈ batch_size. Single-seq decode is deep in the bandwidth roof.",
    }
    dest = Path("benchmarks/results/phase4")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "roofline.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
