# Phase 3 results (T0)

## Correctness

| Check | Result |
|---|---|
| Prefix hit vs cold greedy (shared prefix, different suffix) | token-identical |
| Scheduler `last_take` ≤ `prefill_chunk_size` | pass |
| Swap out/in restores page bytes | pass |
| API `/health`, `/v1/models` | pass |
| Paged vs HF greedy 8×32 (regression) | pass |

## Mixed-load ITL (4 decodes + ~2k-token prefill)

| | p99 ITL |
|---|---|
| Chunk off | 0.715 s |
| Chunk 64 | 0.258 s |
| **Cut** | **64%** (≥ 40% gate) |

## Prefix cache (W3-like shared system prompt)

| Metric | Value |
|---|---|
| Hit rate | **93%** (≥ 60% gate) |
| TTFT cut (0.5B, T0) | ~2% — first-token time is launch-bound (~50 ms). Mechanism is correct; 8B/A100 will move the needle. |

## Serving

```
python -m slipstream.entrypoints.api_server --model Qwen/Qwen2.5-0.5B
# http://127.0.0.1:8000/v1/completions
# http://127.0.0.1:8000/dashboard/
```

vLLM W1 table not produced (vLLM not installed). HF baseline remains the Phase 0/1 column.
