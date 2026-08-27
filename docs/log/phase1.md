# Phase 1 log

Daily sync artifact (MASTERPLAN §11.4). Each agent appends: shipped / blocked / interface assumptions.

## 2026-08-13

### A7 Verification — harness landed

Shipped the Phase 1 correctness harness. No `slipstream/` edits. No benchmark numbers (GPU parity / naive_t0 not run).

| File | What it covers |
|---|---|
| `tests/correctness/conftest.py` | Snapshot path fixtures + `cuda_device`. Does **not** load 0.5B at collection. |
| `tests/correctness/_api.py` | Spec adapters (layer import scan, HF eager loader, `LLMEngine.generate` / `CausalLM.forward`) |
| `tests/correctness/test_layers_reference.py` | RMSNorm formula; `rotate_half` + default RoPE vs local HF-eager reference; `repeat_kv` `[B,2,T,D]→[B,14,T,D]` |
| `tests/correctness/test_cache_and_sampler.py` | `NaiveKVCache` two-step append / overflow `ValueError` / reset; Sampler greedy=argmax, same seed, device stay |
| `tests/correctness/test_parity_logits.py` | `@gpu @parity` last-token logits vs HF eager, `atol=rtol=1e-2` |
| `tests/correctness/test_parity_greedy.py` | `@gpu @parity` 8×32 Qwen default; `@slow` 50×128; TinyLlama 4×16 (skip if snapshot missing) |

HF reference (tests only):

```python
AutoModelForCausalLM.from_pretrained(
    snapshot, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
)
# logits: model(input_ids=..., use_cache=False)
# greedy: model.generate(input_ids=..., do_sample=False)
```

Both sides get the same `input_ids`. Tokenization prefers `slipstream.models.tokenizer.Tokenizer.encode(..., add_special_tokens=False)`; otherwise `AutoTokenizer.encode(..., add_special_tokens=False)`. HF is never given raw text, so it cannot inject a BOS we did not encode.

CPU smoke (allowed run; slow 50×128 **not** run):

```
.venv/bin/pytest tests/correctness/test_layers_reference.py \
    tests/correctness/test_cache_and_sampler.py -q
# 16 passed  (9 layer + 7 cache/sampler; CUDA device-stay included)
```

Collected 20 tests under `tests/correctness/` (the remaining 4 are GPU parity).

### Interface assumptions (call out if A1/A2/A6 change)

Adapters prefer the spec APIs and already match what landed today:

- `LLMEngine(EngineConfig.for_model(snapshot)).generate(Request(prompt_token_ids=..., sampling_params=SamplingParams(temperature=0, max_tokens=N)))` → output ids only.
- `CausalLM.forward(input_ids, positions, kv_cache)` → `[B, T, vocab]`. Used for last-token logits. Greedy falls back to this + argmax only if `generate` still raises `NotImplementedError`.
- Layers: `RMSNorm`, `rotate_half`, `apply_rotary_pos_emb`, `RotaryEmbedding(config)`, `repeat_kv(x, n_rep)`.
- `RotaryEmbedding.forward(x, position_ids)` → `(cos, sin)` `[B, T, D]`.
- `load_model(snapshot, *, device, dtype)` returns `CausalLM` (has `lm_head`; do not unwrap `.model`).
- Sampler: `sample(logits, params, *, generator=None, token_ids=None)`; `params.seed` rebuilds a `torch.Generator` each call.

### A1 / A2 / A6 — engine landed; Gate 1 verified

- A2: `NaiveKVCache` (layer 0 advances `seq_len`).
- A6: greedy / temp / top-k / top-p / min-p sampler; no host sync in `sample`.
- A1: Qwen2 + Llama CausalLM, safetensors loader, tokenizer, `LLMEngine.generate`.

**Verified on this machine (2026-08-13):**

| Check | Result |
|---|---|
| Last-token logits vs HF eager (Qwen 0.5B) | pass (`atol=1e-2`; A1 measured maxabs 0) |
| Greedy 8 prompts × 32 tokens (Qwen) | token-identical |
| Greedy 50 × 128 (Qwen, `@slow`) | token-identical (704 s) |
| Greedy TinyLlama 4 × 16 | token-identical |
| Naive T0 throughput (8 prompts × 128) | **15.15 tok/s** median vs HF **11.09 tok/s** |

Results: `benchmarks/results/phase1/naive_t0.json`.

### Orchestrator follow-up (review findings)

Fixed before calling Gate 1 done:
- Causal mask is packed-index based (RoPE positions no longer enter the mask).
- `ModelRunner.prefill` continues from `cache.seq_len` (same as decode).
- `capped_new_tokens` so `prompt + max_tokens > max_model_len` finishes `FINISHED_LENGTH` instead of overflowing the cache.

Still open: T1 Llama-3.1-8B not run (does not fit T0). EngineCore is a parallel path, not used by `generate()`.

### Blocked / next

- Phase 2: block manager + Triton paged attention. Paging must be token-identical to this naive engine.
