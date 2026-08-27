# Phase 3 log

## 2026-08-14

### Shipped

- Stall-free chunks: `prefill_chunk_size` (default 256); `generate_batch` honors `scheduler.last_take`.
- Radix prefix cache: full-block pins, attach on admit, insert on finish, LRU evict when allocating under pressure.
- Exact full-prompt hits are *not* skipped (need last-token logits). Shared prefix + unique suffix is the hit path.
- CPU swap: `CpuSwapSpace` + `swap_out`/`swap_in`. FCFS swaps victims with ≥4 pages, else recomputes.
- FastAPI: `/v1/completions`, `/v1/chat/completions`, SSE, `/health`, `/metrics`, `/ws/metrics`.
- Dashboard: static block-table grid at `/dashboard/` (no npm).

### Numbers (T0, Qwen2.5-0.5B)

- ITL p99 cut **64%** under mixed load (chunk 64 vs full prefill).
- Prefix hit rate **93%** on a shared system prompt.
- TTFT cut ~2% on 0.5B (launch-bound). Hit path is token-identical.

### Next

Phase 4: CUDA graphs, fused kernels, EngineCore isolation, **Horizon**.
