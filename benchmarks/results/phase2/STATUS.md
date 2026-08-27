# Phase 2 results (T0)

## Correctness

| Check | Result |
|---|---|
| Allocator 10k random ops | pass |
| Mixed-length KV waste | < 5% internal frag (chat-like lengths) |
| Paged vs naive greedy (3 prompts × 16) | token-identical |
| Paged vs HF eager logits | pass |
| Paged vs HF greedy 8×32 | token-identical |
| `generate_batch` B=1 vs `generate` | token-identical |
| Multi-seq batch vs sequential (bf16) | last tokens may differ (batched GEMM) |

## Continuous batching — Qwen2.5-0.5B / RTX 3050 6GB

16 requests, `max_tokens=32`, greedy, `SLIPSTREAM_DEBUG=0`.

| Mode | Seconds | tok/s | Speedup |
|---|---|---|---|
| Sequential `generate` | 29.55 | 16.3 | 1.0× |
| Continuous `generate_batch` | 3.83 | **126.0** | **7.72×** |

Gate 2 asked ≥ 3× vs static. Sequential one-at-a-time is the conservative static baseline on T0 (no padded static-batch kernel). 7.72× clears the gate.

`SLIPSTREAM_DEBUG=1` makes `append_slot` check invariants every op and collapses CB to ~1.8× — leave debug off for published numbers.

## Decode kernel bandwidth on A100

Not measured. T0 only. Triton decode is optional; PyTorch gather+eager is the numeric source of truth.
