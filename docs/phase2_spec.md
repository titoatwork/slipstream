# Phase 2 spec — Paged memory, kernels, continuous batching

Binding: MASTERPLAN §10 Phase 2, S2–S4. **Paging must be semantically invisible.**

## Gate 2 (what we close on T0 vs T1)

| Criterion | Where |
|---|---|
| Allocator fuzz 10k ops, zero invariant breaks | T0 CPU |
| Paged attention vs gather+eager `atol=1e-2` | T0 CUDA |
| Token-identical to Phase 1 naive / HF greedy | T0 (Qwen 8×32 default, 50×128 slow) |
| KV waste < 5% (vs >60% naive) | T0 measured |
| Continuous batching ≥ 3× static batching | T0 measured (same model, same prompts) |
| Decode kernel ≥ 70% HBM on **A100** | T1 — measure T0 bandwidth, do not fake A100 |

`slipstream/` still must not import `transformers` / `vllm` / `sglang`.

---

## Layout (frozen §8.4)

```
kv_cache: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
slot = physical_block_id * block_size + offset
block_tables[b, i] = physical id or -1
```

`block_size` default 16.

---

## File ownership

| Agent | Writes | Does not touch |
|---|---|---|
| **A2** | `slipstream/memory/block_manager.py`, `block_table.py`, `paged_cache.py`, `memory/__init__.py` | kernels, engine, models, tests |
| **A3** | `slipstream/kernels/reshape_and_cache.py`, `paged_attention.py`, `attention_ref.py` | memory allocator, tests |
| **A4** | `scheduler/scheduler.py`, `policies/fcfs.py`, `engine/llm_engine.py`, `engine/model_runner.py`, `engine/engine_core.py`, `models/layers/attention.py`, `models/causal_lm.py` | `kernels/*` internals (import only), `tests/` |
| **A7** | `tests/property/**`, `tests/kernels/**`, `tests/correctness/test_paged_identity.py`, `docs/log/phase2.md` | production except bugfix PRs via orchestrator |

---

## A2 — BlockManagerImpl

Track `PhysicalBlock` per id. Free list = stack of `ref_count==0` ids.

```
allocate(seq)     reserve ceil(prompt_len / block_size) empty blocks onto seq.block_table
append_slot(seq)  fill next slot; new block if tail full; COW if tail.ref_count > 1
                  return (block_id, offset)
fork(p, c)        c.block_table = copy(p.block_table); ref++ on each
free(seq)         ref-- ; if 0, clear and push free list; clear seq.block_table
can_allocate      NEVER if ceil(max_model_len/block_size) > num_gpu_blocks
                  LATER if free < blocks still needed for *current* seq.num_tokens
                  else OK
```

COW: allocate dst, `kv[:, :, dst] = kv[:, :, src]` if a tensor is bound via `bind_kv(tensor)`, decrement src.

`swap_*`: raise NotImplementedError (Phase 3).
`match_prefix`: return `([], 0)` until Phase 3.

Debug (`SLIPSTREAM_DEBUG=1`) after every mutating op:
- I2.2 every free-list id has `ref_count==0`; every `ref_count>0` is not on the free list
- I2.4 COW only when the written tail had `ref_count>1`
- I2.5 last block `num_tokens <= block_size`
- Conservation: `sum(ref_count) == sum(len(table) for live tables)`

`__init__(num_gpu_blocks, block_size, num_cpu_blocks=0, max_model_len=4096)`

---

## A3 — kernels

`attention_ref.py` (PyTorch, source of truth for numerics):
- `reshape_and_cache_ref(k, v, kv_cache, slot_mapping, layer_idx, block_size)`
  - `k,v`: `[N, n_kv, D]` or `[B, n_kv, T, D]`
  - `slot_mapping`: `[N]` int64
- `paged_attention_ref(q, kv_cache, block_tables, seq_lens, layer_idx, scale, block_size)`
  - `q`: `[B, n_q, Tq, D]` — the **new** tokens (last `Tq` of each seq)
  - gather pages → contiguous K/V `[B_i, n_kv, seq_len, D]`
  - `eager_attention` from `models.layers.attention` (fp32 softmax, packed causal)
- `copy_blocks_ref(kv_cache, src_to_dst: dict[int,int])`

`reshape_and_cache.py` / `paged_attention.py`:
- Public functions `reshape_and_cache` / `paged_attention_decode` / `paged_attention_prefill`
- Prefer Triton on CUDA; **must fall back** to the ref on CPU or if Triton fails
- Decode kernel: walk `block_tables`, online softmax, static-ish grid `(B, n_q_heads)`
- Numerics: vs ref `atol=1e-2` on randomized shapes (batch 1–16, seq 1–512, block 8/16/32)

Do not import transformers.

---

## A4 — engine + scheduler

`CacheConfig.enable_paging: bool = True` (amendment). `LLMEngine.generate` uses paged when True; naive `NaiveKVCache` remains for ablation (`enable_paging=False`).

`PagedForward` (in `paged_cache.py`) is what `Attention.forward` sees:
```
kv_cache tensor, block_tables [B, max_blocks], seq_lens [B],
slot_mapping [N], block_size, query_lens [B]
```

Single-seq generate (must stay token-identical):
1. `can_allocate` / `allocate`
2. `append_slot` once per new token **before** the forward
3. build `PagedForward`, run model
4. sample, loop decode

FCFS:
- `should_admit`: `ceil(prompt_len / block_size) <= num_free_blocks`
- `select_preemption_victim`: newest `arrival_ts` (then highest `seq_id`)
- `preemption_mode`: `RECOMPUTE` (swap is Phase 3)

Scheduler mechanism (iteration-level):
1. Running **decodes** first (1 token each) — stall-free *among running decodes*
2. Fill remaining token budget with running prefills (full remaining prompt, or budget)
3. Admit waiting via policy + `can_allocate` / `allocate`
4. If a new seq cannot allocate and running is non-empty, preempt RECOMPUTE (free blocks, put victim back on waiting with `num_computed_tokens=0`, clear table)
5. `num_batched_tokens <= token_budget`

`LLMEngine.generate_batch(requests) -> list[list[int]]` drives the scheduler until all finish.

Phase 2 may run prefills as their own step (not mixed T with decodes) if that keeps the kernel simple. Decodes must batch.

---

## A7 — tests

- Hypothesis: random allocate/append/fork/free/append after fork (COW); 10k ops; invariant checker after each
- Kernel sweep vs ref
- `test_paged_identity`: `enable_paging=True` vs `False` vs HF greedy, Qwen 8×32
- Waste: `(sum over seq of (blocks*block_size - tokens)) / (blocks*block_size) < 0.05` on a mixed-length batch
- CB vs static: document the measurement harness even if the 3× number is recorded later

---

## Quality

Type hints, ruff/mypy clean, no forbidden imports, no leftover prints. Comments only for non-obvious paging/COW constraints.
