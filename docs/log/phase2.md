# Phase 2 log

## 2026-08-13

### Shipped

- `BlockManagerImpl`: pool, tables, refcount, COW, fill-first-incomplete-block, debug invariants.
- Paged KV tensor `[L, 2, blocks, block_size, n_kv, D]` + `PagedForward`.
- Gather+eager paged attention (numeric source of truth).
- Triton `reshape_and_cache` + `paged_attention_decode` in `kernels/triton_paged.py`.
  Default decode is the ref. `SLIPSTREAM_TRITON=1` enables Triton (bf16 argmax can flip).
- `LLMEngine.generate` uses paging when `CacheConfig.enable_paging=True` (default).
- `LLMEngine.generate_batch` + FCFS scheduler (decodes first, then prefills, then admit).
- Tests: 10k allocator fuzz, kernel vs ref, paged vs naive identity, CB B=1 identity.

### Numbers (T0, Qwen2.5-0.5B, `SLIPSTREAM_DEBUG=0`)

| Metric | Value |
|---|---|
| Paged vs naive greedy | identical |
| Paged vs HF 8×32 | identical |
| Allocator 10k ops | pass |
| CB vs sequential (16 req × 32 tok) | **7.72×** (126 vs 16 tok/s) |
| Triton default | off (identity) |

### Interface notes

- `append_slot` writes the first not-full block so `allocate()` reserved pages fill in order.
- `oracle_output_len` temporarily stores the per-request output cap in `generate_batch`.
- Multi-seq bf16 GEMM is not bit-identical to B=1. That is not a paging bug.

### Still T1

- Decode kernel ≥ 70% A100 HBM.
- Llama-3.1-8B identity.

### Next

Phase 3: chunked prefill (stall-free), radix prefix cache, API server, dashboard.
