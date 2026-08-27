"""HuggingFace Transformers floor baseline.

Allowed to import `transformers`. Never imported by `slipstream/`.

Usage (after `pip install -e '.[gpu,bench]'`):

    python -m benchmarks.baselines.hf_generate \\
        --model Qwen/Qwen2.5-0.5B \\
        --max-new-tokens 128 \\
        --out benchmarks/results/phase0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

# Default T0 prompt set — reused by the Phase 1 parity harness.
DEFAULT_PROMPTS: list[str] = [
    "The capital of France is",
    "In machine learning, a transformer is",
    "Write a one-sentence definition of throughput.",
    "2 + 2 =",
    "Once upon a time in a small village",
    "The purpose of a KV cache in autoregressive decoding is",
    "Explain PagedAttention in two sentences.",
    "def fibonacci(n):",
]


def run_baseline(
    model_id: str,
    prompts: list[str],
    max_new_tokens: int,
    warmup: int,
) -> dict[object, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()

    # warmup
    warm = tok(prompts[0], return_tensors="pt").to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model.generate(**warm, max_new_tokens=8, do_sample=False)

    if device == "cuda":
        torch.cuda.synchronize()

    per_prompt: list[dict[str, object]] = []
    for prompt in prompts:
        encoded = tok(prompt, return_tensors="pt").to(device)
        prompt_len = int(encoded["input_ids"].shape[-1])
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        gen = int(out.shape[-1]) - prompt_len
        per_prompt.append(
            {
                "prompt": prompt,
                "prompt_tokens": prompt_len,
                "output_tokens": gen,
                "wall_s": elapsed,
                "tok_s": gen / elapsed if elapsed > 0 else 0.0,
            }
        )

    tok_s = [float(p["tok_s"]) for p in per_prompt]  # type: ignore[arg-type]
    return {
        "model": model_id,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "max_new_tokens": max_new_tokens,
        "n_prompts": len(prompts),
        "tok_s_median": median(tok_s),
        "tok_s_min": min(tok_s),
        "tok_s_max": max(tok_s),
        "per_prompt": per_prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HF generate() T0 baseline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/phase0"))
    args = parser.parse_args()

    # Local import so `python -m` works without installing slipstream extras
    # just to parse --help, and so this file stays import-safe in unit tests.
    from benchmarks.manifest import write_run_manifest

    args.out.mkdir(parents=True, exist_ok=True)
    results = run_baseline(args.model, DEFAULT_PROMPTS, args.max_new_tokens, args.warmup)
    (args.out / "hf_qwen25_0.5b.json").write_text(json.dumps(results, indent=2) + "\n")
    write_run_manifest(
        args.out / "run_manifest.json",
        model=args.model,
        workload="phase0_hf_baseline",
        config={"max_new_tokens": args.max_new_tokens, "do_sample": False},
        notes="HuggingFace generate() floor baseline for Gate 0.",
    )
    print(json.dumps({k: v for k, v in results.items() if k != "per_prompt"}, indent=2))


if __name__ == "__main__":
    main()
