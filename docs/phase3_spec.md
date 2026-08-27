# Phase 3 spec — Serving, prefix cache, stall-free prefill

Binding: MASTERPLAN §10 Phase 3, S4/S5/S7/S12.

## Gate 3 on T0

| Criterion | How we close it here |
|---|---|
| Chunked prefill cuts decode ITL p99 ≥ 40% under mixed load | Measure on Qwen 0.5B: 4 decodes + 1 long prefill, chunk on vs off |
| Prefix hit ≥ 60%, TTFT cut ≥ 50% on shared-prefix (W3-like) | Synthetic 90% shared system prompt |
| Prefix hit output **token-identical** to cold cache | Greedy test |
| Preempt/resume token-identical | Force SWAP and RECOMPUTE mid-generate |
| `openai` client works unmodified | Completions + chat, stream and not |
| Dashboard live, block table visible | Served at `/dashboard`, WS `/ws/metrics` |
| W1 vs HF/vLLM table | HF column we have; vLLM skip if not installed |

---

## File ownership

| Agent | Writes | Does not touch |
|---|---|---|
| **A4** | `scheduler/scheduler.py`, `engine/llm_engine.py` (chunk take + stream + metrics hooks), `scheduler/policies/fcfs.py` (swap vs recompute) | prefix_cache.py, dashboard, api_server |
| **A2** | `memory/prefix_cache.py`, `memory/swap.py`, `memory/block_manager.py` (attach/pin/swap_out/in/match_prefix) | engine generate loop, api, dashboard |
| **A5** | `entrypoints/api_server.py`, `entrypoints/openai_protocol.py`, `observability/*` | memory allocator internals |
| **A1** | `dashboard/**` (static HTML/JS is OK if it runs without npm) | slipstream/ except observability types |
| **A7** | `tests/correctness/test_prefix_cache.py`, `test_chunked_prefill.py`, `test_preempt.py`, `test_api_server.py`, `docs/log/phase3.md`, `benchmarks/results/phase3/` | production except via orchestrator |

---

## A4 — stall-free chunks

`SchedulerConfig.prefill_chunk_size: int = 256`

`schedule()` already does decodes first. **Honor `take` in `generate_batch`** — today it prefills the entire remainder and stalls the next decode.

```
self.last_take: dict[int, int]  # seq_id -> tokens this step
take = min(uncomputed, budget_left, prefill_chunk_size if enable_chunked_prefill else inf)
```

`generate_batch` uses `scheduler.last_take[seq_id]`, never `remaining`.

`LLMEngine.generate_stream(request) -> Iterator[int]` — same as paged generate, yield after each sampled token.

Record per-decode step wall time on a list `engine.last_itl_s` for the mixed-load bench.

---

## A2 — prefix cache + swap

### Pin API on BlockManager

```
pin(block_ids)      extra ref (cache holds the page)
unpin(block_ids)    drop extra ref; free if 0
attach(seq, ids)    seq.block_table = ids; ref++ each (live user)
```

Conservation: `sum(ref) == sum(len(tables)) + sum(pin_counts)`.

`match_prefix` delegates to `RadixPrefixCache` if bound.

### RadixPrefixCache

Token-walk radix. **Only full blocks** are cached (length multiple of `block_size`).

- `match(token_ids) -> (block_ids, n_tokens)` longest full-block prefix
- `insert(token_ids, block_ids)` pin new unique blocks; update LRU clocks
- `evict(n_blocks)` unpin oldest unused (pin-only, not in a live table)

Bind: `cache.bind(manager)`.

On admit (engine): match → attach → `num_cached_tokens = num_computed_tokens = n`.
On finish, **before** `free`: `insert(prompt_token_ids, block_table)` then free (pins keep prefix).

I5.1: hit vs miss greedy identity.
I5.2: shared blocks never written (COW if they would be).
I5.3: evicted blocks end at ref 0.

### Swap

`CpuSwapSpace` holds a CPU tensor `[L, 2, num_cpu_blocks, block_size, n_kv, D]`.

`swap_out(seq) -> {gpu: cpu}`: copy pages to CPU, unpin GPU from the table, stash map on the manager, `seq.status = SWAPPED`.
`swap_in(seq) -> {cpu: gpu}`: new GPU pages, copy back, restore table.

FCFS `preemption_mode`: SWAP if `len(block_table) >= 4` else RECOMPUTE.

Scheduler: `_try_admit` may SWAP the victim instead of only RECOMPUTE. Swapped seqs go to `self.swapped`. A later `schedule` swap-ins if free blocks allow (decode/prefill resume with `num_computed_tokens` unchanged).

---

## A5 — API

FastAPI app from `create_app(engine: LLMEngine | None = None)`.

| Route | Behavior |
|---|---|
| `GET /health` | `{status: ok}` |
| `GET /v1/models` | OpenAI list |
| `POST /v1/completions` | prompt str or token ids; `stream` → SSE |
| `POST /v1/chat/completions` | concat messages as `"role: content\n"`; same sampling |
| `GET /metrics` | Prometheus text + JSON twin |
| `WS /ws/metrics` | last `MetricsSnapshot` + block table |
| `GET /dashboard` | static files |

Sync `generate` in a thread pool. Stream via `generate_stream`. On disconnect, `scheduler.abort(seq_id)`.

CLI: `python -m slipstream.entrypoints.api_server --model Qwen/Qwen2.5-0.5B --host 127.0.0.1 --port 8000`

No auth (non-goal).

---

## A1 — dashboard

Single-page app under `dashboard/` that **runs without npm**. Centerpiece: canvas/SVG grid of physical blocks.

- Color by `ref_count` (0=empty, 1=owned, 2+=shared)
- Tooltip: block id, tokens, ref
- Side panel: TTFT/TPOT, queue, KV util, cache hit rate, preemptions
- Connect to `ws://.../ws/metrics`

`dashboard/README.md` says how to open it (server serves it).

---

## A7 — tests

- Prefix: two greedy requests, same 64-token prefix + different suffixes; outputs equal to two cold runs; `num_cached_tokens > 0` on the second.
- Chunk: `last_take` never exceeds `prefill_chunk_size`; mixed-load ITL p99 recorded.
- Preempt: force `preempt_recompute` and `swap_out/in` mid-run; tokens == uninterrupted.
- API: `TestClient` completions + stream smoke (no GPU required if we mock; GPU if engine present).

---

## Quality

Type hints, ruff/mypy, no forbidden imports. `SLIPSTREAM_DEBUG=1` in tests.
