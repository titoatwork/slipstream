# Phase 1 results (T0)

| Artifact | Status |
|---|---|
| `naive_t0.json` | recorded 2026-08-13 |
| `run_manifest.json` | recorded 2026-08-13 |

## Naive engine vs HF `generate()` — Qwen2.5-0.5B / RTX 3050 Laptop 6GB

Single-sequence greedy, 8 prompts, `max_new_tokens=128`, bf16. Same prompt set as Phase 0.

| System | Median tok/s | Min | Max |
|---|---|---|---|
| HF `generate()` (Phase 0) | 11.09 | 7.12 | 12.21 |
| Slipstream naive (contiguous KV, eager attn) | **15.15** | 10.28 | 18.34 |

This is the **naive** column for every later ablation table. Speed was not a Phase 1 goal; identity was.

Gate 1: 50 prompts × 128 tokens greedy, token-identical to HF eager.
