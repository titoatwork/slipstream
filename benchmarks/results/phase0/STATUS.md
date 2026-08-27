# Phase 0 results

| Artifact | Status |
|---|---|
| `run_manifest.json` | recorded 2026-08-13 |
| `hf_qwen25_0.5b.json` | recorded 2026-08-13 |

## HF `generate()` floor — Qwen2.5-0.5B / T0

Single-sequence greedy decode. This is the number later tables call **naive**.

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device | CUDA, RTX 3050 Laptop 6GB, SM 8.6 |
| Dtype | bfloat16 |
| `max_new_tokens` | 128 |
| Prompts | 8 (see `DEFAULT_PROMPTS`) |
| Median tok/s | **11.09** |
| Min / max tok/s | 7.12 / 12.21 |
| torch | 2.13.0+cu126 |
| transformers | 5.15.0 |
| Clocks | not locked; `exclusive_gpu` unset (laptop / WSL2) |

Reproduce:

```bash
python -m benchmarks.baselines.hf_generate \
    --model Qwen/Qwen2.5-0.5B \
    --max-new-tokens 128 \
    --out benchmarks/results/phase0
```
