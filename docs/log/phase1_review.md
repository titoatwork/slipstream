# Phase 1 correctness review

Date: 2026-08-13
Scope: models/layers, causal_lm, loader, hf_config, tokenizer, engine, sampler, NaiveKVCache, correctness tests.
Method: spec + source review only. Production code not modified. GPU 8×32 / TinyLlama 4×16 / last-token logits (maxabs=0) taken as given; `50×128` not re-run.

## Verdict

The **single-sequence generate path that the parity harness actually calls** is internally consistent and matches HF eager on the checked gates:

- Prefill positions are `0..T-1`; decode positions continue at `cache.seq_len`.
- After every `update`, `seq_len == last_position + 1`.
- EOS is **included** then stop, same as `transformers` greedy for `B=1`.
- Qwen2.5-0.5B ties `lm_head` to `embed_tokens`; TinyLlama loads a real `lm_head`.
- Layers are constructed and iterated `0..N-1`, satisfying layer-0-advances-`seq_len`.
- T0 Qwen `generation_config.json` has a single `eos_token_id` (151643), not the Instruct `[151645, 151643]` list.

Remaining issues are **latent correctness** (public `forward` / `prefill` / long-context) and **Phase 2 semantic traps**. None of them are contradicted by last-token logits or short greedy runs.

## Findings

### 1. Causal mask uses key *indices* as if they were positions

- **severity:** bug
- **file:line:** `slipstream/models/layers/attention.py:29-47` (used at `65`), `slipstream/engine/model_runner.py:49-53`
- **description:** `causal_mask` builds `key_pos = arange(key_len)` and allows `key_idx <= query_positions`. That is correct **only** when packed slots are a dense prefix from position 0, so `slot i` is token position `i`. HF eager (this venv) uses the same index rule (`kv_idx <= q_idx` plus cache offsets) and applies `position_ids` only to RoPE. Slipstream feeds `position_ids` into the mask. If `seq_len != last_position + 1` — suffix prefill, positions `[10,11,12]` on an empty cache, sliding-window drop of a prefix, packed holes — the mask silently becomes almost all-true (queries with large positions attend to every packed key). Last-token logits do not catch a first-layer mask bug on earlier tokens only if later layers never see those corrupted keys; a non-zero start *does* change last-token logits from layer 1 on, but `generate()` never takes that path.
- **suggestion:** Mask on **packed key index vs query index** (or vs `cache_position`), not vs RoPE `position_ids`. Keep `position_ids` for RoPE only. For a general cache, compare `key_positions[j] <= query_positions[i]` with an explicit per-slot position tensor. Document the Phase 1 invariant (`positions[t] == cache_index[t]`, start at 0) next to `CausalLM.forward` so Phase 2 cannot “just pass global positions.”

### 2. `ModelRunner.prefill` always restarts positions at 0

- **severity:** bug
- **file:line:** `slipstream/engine/model_runner.py:49-53` vs `decode` at `55-64`
- **description:** `prefill` does `arange(0, T)` regardless of `kv_cache.seq_len`. `decode` correctly starts at `kv_cache.seq_len`. A second prefill, a prefix-cache suffix, or chunked prefill that calls `prefill` for the uncached tail will (a) RoPE-rotate the new tokens as if they were positions `0..T-1` and (b) apply finding 1’s mask against the already-packed prefix. `generate` / `EngineCore` only prefill once on an empty cache, so current tests pass.
- **suggestion:** Make prefill `start = kv_cache.seq_len` (i.e. the same formula as `decode`), or refuse `prefill` when `seq_len != 0`. Phase 2 prefix hits must go through the continuation path, never “prefill the suffix with 0..T-1.”

### 3. Cache sized to `min(max_model_len, prompt+max_tokens)` but the loop still asks for `max_tokens`

- **severity:** bug
- **file:line:** `slipstream/engine/llm_engine.py:57-66,83-86`; `slipstream/engine/engine_core.py:48-61`; `slipstream/memory/contiguous_cache.py:100-105`
- **description:** Cache capacity is `min(max_model_len, prompt + max_tokens)`. The decode loop still runs until `num_output_tokens >= max_tokens` or EOS. When `prompt + max_tokens > max_model_len` (legal: only `prompt >= max_model_len` is rejected), the last decode raises `ValueError: KV cache overflow` instead of `FINISHED_LENGTH`. Example: `max_model_len=4096`, prompt 4090, `max_tokens=16` → six tokens then crash. `generate` and `EngineCore` share this. Short CI prompts cannot hit it.
- **suggestion:** `effective = min(max_tokens, max_model_len - prompt_len)` and finish with `FINISHED_LENGTH` at that cap (or reject the request up front). Do not rely on `NaiveKVCache.update` as the length stop.

### 4. EOS include/exclude vs HF generate — **not a T0 mismatch**

- **severity:** suggestion (Instruct / API; not a T0 50×128 fail by itself)
- **file:line:** `slipstream/engine/llm_engine.py:123-130,138-144`; `slipstream/models/hf_config.py:114-125,236-245`; `tests/correctness/_api.py:718-735`
- **description:** Slipstream `append_token` then `is_stop`. HF `utils.py` `_sample` (this venv, ~2925–2937) `cat`s the argmax token then applies `EosTokenCriteria`; for `B=1` the loop exits without padding to `max_new_tokens`. Both **include** the EOS id. T0 snapshots: Qwen `config.json` + `generation_config.json` both `eos_token_id: 151643`; TinyLlama `2`. The harness passes the same single id into `model.generate(eos_token_id=...)`. So include/exclude will not, by itself, fail `50×128` on these checkpoints. Residual: `ModelConfig.eos_token_id` is one `int`; `_first_id` keeps only the first list element; `generation_config` is consulted **only if** `config.json` omitted eos. Qwen Instruct typically has `config.json: 151643` and `generation_config: [151645, 151643]`. HF generate would stop on `<|im_end|>`; Slipstream would continue. Frozen `ModelConfig` cannot store a list without a §21 amendment — callers must put extras in `SamplingParams.stop_token_ids`.
- **suggestion:** Union `config.json` and `generation_config.json` eos ids into `stop_token_ids` at engine init (keep `eos_token_id` as the first id). Add a parity case that forces EOS (or a fake stop id) so include-and-stop is tested; 8×32 can pass without ever emitting 151643.

### 5. Tied embeddings / load misses — **T0 load is correct**

- **severity:** suggestion
- **file:line:** `slipstream/models/loader.py:61-93`; `slipstream/models/causal_lm.py:116-121`
- **description:** Missing `lm_head.weight` + `tie_word_embeddings` → `tie_lm_head()` (same `Parameter`). Matches Qwen2.5-0.5B (`tie_word_embeddings: true`, no `lm_head` tensor). TinyLlama is untied and has `lm_head`. Shape mismatches raise; unknown keys are logged; skipped `inv_freq` / `rotary_emb` / `cos_cached` / `sin_cached` is right. Residual: if a checkpoint has **both** `lm_head.weight` and `tie_word_embeddings: true`, the two stay distinct copies (HF still ties). Fine while values match. Extra tensors that the module does not implement (`q_norm`, etc.) are ignored → silent wrong architecture on Qwen3+.
- **suggestion:** If `tie_word_embeddings`, always `tie_lm_head()` after load (and assert `lm_head` equals embed when the key was present). Fail (don’t warn-only) on unexpected non-RoPE keys.

### 6. `EngineCore` first step vs `generate()` — same tokens if ids match; dual path

- **severity:** suggestion
- **file:line:** `slipstream/engine/llm_engine.py:45-48,54-87,89-99`; `slipstream/engine/engine_core.py:28-63`
- **description:** `generate()` never calls `core.step()`. Both paths share `prefill` / `decode` / `append_sampled`. Given the same `prompt_token_ids` and an empty cache, step 1 is prefill+sample on both. Divergences:
  1. `generate(prompt=...)` runs `tokenize()` (optional BOS). `EngineCore` uses `seq.prompt_token_ids` as-is.
  2. `generate` rejects `len(prompt) >= max_model_len`. `EngineCore` will prefill a full-length prompt and then overflow on decode (finding 3).
  3. If `num_computed_tokens >= num_prompt_tokens` on first `step` (resume / prefix-cache bookkeeping) and `_cache is None`, the else branch does `decode(output_token_ids[-1])` on an empty cache. Empty outputs → `IndexError`. Non-empty outputs without a prompt prefill → token 0 RoPE’d at position 0.
- **suggestion:** `generate` should be `tokenize` + `add_request` + `while not finished: step()`. On cache create, if not `is_prefill`, either refuse or rebuild from `prompt + output[:-1]` before decoding the last id. Tests should drive `EngineCore.step` once, not only `LLMEngine.generate`.

### 7. Layer-0-must-go-first vs layer order

- **severity:** nit (Phase 1) / suggestion (Phase 2)
- **file:line:** `slipstream/memory/contiguous_cache.py:22-27,100-115`; `slipstream/models/causal_lm.py:76-78,90-91`
- **description:** `DecoderModel` builds and walks layers `0..N-1`. Layer 0 advances `_seq_len`; later layers require `start == _layer_fill[i] == seq_len - T_new`. Current stack matches the spec. A paged or pipelined implementation that updates layers out of order, in parallel, or after the full stack will hit the `ValueError` or (if the check is dropped) desynchronized fill. `seq_len` is one integer for the whole cache — it cannot represent a batch of sequences with different lengths.
- **suggestion:** Treat “layer 0 advances a global `seq_len`” as a Phase 1 artifact, not a Phase 2 contract. Per-sequence logical length (or `cache_position`) should live on `Sequence` / the block table. `update` should take `T_new` + the sequence’s logical start, not imply “whoever is layer 0.” Keep the debug check until the API changes.

### 8. BOS: Llama vs Qwen is right; string vs ids is a footgun

- **severity:** suggestion
- **file:line:** `slipstream/models/tokenizer.py:24-51,53-54`; `slipstream/engine/llm_engine.py:89-99`; `tests/correctness/_api.py:374-422,626-636`
- **description:** Qwen `tokenizer_config.json` has `add_bos_token: false` (and `bos_token: null`). TinyLlama has `tokenizer_class: LlamaTokenizer` and **no** `add_bos_token` key → Slipstream sets `add_bos_token=True`. That matches HF Llama defaults and TinyLlama’s `TemplateProcessing` BOS. Engine prepends `ModelConfig.bos_token_id` only on the **string** path, and only if `ids[0] != bos`. `Request(prompt_token_ids=...)` never adds BOS. Parity tests encode with `add_special_tokens=False` and pass those ids to both sides — **TinyLlama 4×16 never exercises BOS**. `generate(prompt="...")` vs `generate(prompt_token_ids=encode("..."))` therefore disagree on Llama. Qwen `bos_token_id` is 151643 (`<|endoftext|>`); if `add_bos_token` were ever true, every string prompt would start with EOS.
- **suggestion:** One policy: either the engine always owns BOS (and tests cover `Request(prompt=...)`), or token ids are sacred and `add_bos_token` is only documentation. Do not prepend 151643. Add a Llama string-path vs `AutoTokenizer(..., add_special_tokens=True)` case.

### 9. Paging will not be a drop-in if these semantics leak

- **severity:** suggestion
- **file:line:** `slipstream/memory/contiguous_cache.py:8-12,76-125`; `slipstream/engine/model_runner.py:55-64`; `slipstream/models/layers/attention.py:37-39`; `docs/architecture.md` KV layout (`num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim`)
- **description:** Phase 2 Gate 2 requires token-identity with Phase 1. Things that would make a paged engine *semantically* different if copied as-is:
  1. Mask = `key_index <= position_id` (finding 1). Paged kernels walk `block_tables`; RoPE needs logical positions. Those must stay separate.
  2. `decode` position = `kv_cache.seq_len` (token **count**, not last logical position). True only for a packed-from-0 prefix. A prefix-cache hit that omits the prefix from this buffer, or a sliding window that drops the head, desynchronizes RoPE.
  3. `prefill` ignores existing cache (finding 2). Prefix suffix-prefill will be wrong if it calls `prefill`.
  4. One `seq_len` / `max_batch=1` / writes `k_cache[layer, :batch]`. Continuous batching cannot share this object.
  5. `update` returns **views** of a dense `[B, n_kv, seq_len, D]` prefix (`contiguous_cache.py:122-125`). Paged layout cannot honor “views, not clones” without a gather into a staging tensor. Attention that later mutates or holds those views across a step will break.
  6. `reset()` zeros lengths only; tails stay dirty. Safe while callers only read `[:seq_len]`. A kernel that reads a full block must zero or track valid slots.
- **suggestion:** Freeze the *observable* contract (RoPE at logical positions, causal over the sequence’s tokens, same K/V values) and replace `NaiveKVCache.update` for Phase 2. Do not teach the paged kernel this mask or this global `seq_len`. Add a property test: prefix-hit vs cold prefill token-identical (`MASTERPLAN` I5.1) before trusting paging.

### 10. `apply_hf_config` keeps the 4096 default `max_model_len`

- **severity:** suggestion
- **file:line:** `slipstream/models/hf_config.py:156-159`; `slipstream/core/config.py:29,115-130`; `slipstream/engine/llm_engine.py:34`
- **description:** `EngineConfig.for_model` starts at `max_model_len=4096`. `apply_hf_config` intentionally does not copy the checkpoint value. Qwen2.5-0.5B is 32768 (silently capped). TinyLlama is 2048 (engine will allocate and RoPE past trained context). Parity lengths are tiny. Combined with finding 3, a long TinyLlama request can overflow or extrapolate.
- **suggestion:** `max_model_len = min(user_cap or +inf, checkpoint max_position_embeddings)` so the default is the model’s, and a user cap can only shrink it.

### 11. Sampler / stop-string footguns (not Gate 1)

- **severity:** nit
- **file:line:** `slipstream/engine/sampler.py:55-56,69-71`; `slipstream/engine/llm_engine.py:145-148`
- **description:** Greedy (`temperature==0` or `top_k==1`) is raw argmax; `repetition_penalty` / top-p / min-p are skipped. Spec allows this; HF greedy also ignores those filters. Seed rebuilds a `Generator` and `manual_seed`s **every** `sample()` call — I6.1 holds, but the stream is not sequential; non-greedy will not match HF’s advancing RNG. `stop_strings` decode with `skip_special_tokens=True`, so a stop string that is a special token never matches (eos id still works). `ModelRunner.execute` (`model_runner.py:66-70`) returns the `SchedulerOutput` unchanged — a future core that calls `execute` instead of `prefill`/`decode` would silently do nothing.
- **suggestion:** If greedy+penalty is required, apply penalty before argmax. Seed once per request and pass the same `generator` in. Decode stop strings with `skip_special_tokens=False`. Delete or implement `execute`.

## Focus checklist

| Focus | Result |
|---|---|
| 1. Mask vs positions when `seq_len != last_pos+1` | **Bug** (latent). `generate` keeps them equal. Public `forward` / suffix prefill / paging will not. |
| 2. EOS vs HF `generate` | **Aligned** on T0 Qwen/TinyLlama (include + stop, single id). Multi-eos Instruct is the 50×128-class trap on other checkpoints. |
| 3. Tied embeddings / missed weights | **OK** on T0. Tie only when `lm_head` absent; unexpected keys warned. |
| 4. `EngineCore` first step vs `generate` | **Same tokens** on clean `prompt_token_ids`. Dual path; BOS / resume / long-prompt differ. |
| 5. Layer-0-first vs iteration | **OK** in Phase 1 (`0..N-1`). Do not keep as a paged invariant. |
| 6. BOS Llama vs Qwen | **OK** flags. Tests never hit the string+BOS path. ids vs prompt footgun. |
| 7. Paging semantic drift | Several (mask, `seq_len` as position, prefill@0, views, global length). |

## Residual risks (no defect found in the generate path)

- `50×128` is not in `docs/log/phase1.md` as run; 8×32 + exact last-token logits make a long greedy mismatch unlikely unless a stop-id or overflow path is hit.
- No TinyLlama **logit** test (only 4×16 greedy). No full-sequence logit test (last token only).
- No `EngineCore.step` parity test. No `Request(prompt=)` BOS test.
- Llama-3.1-8B `rope_type=llama3` is implemented to the spec but untested here (no snapshot). `original_max_position_embeddings` only inside `rope_scaling` — a top-level-only config would `KeyError`.
- Sliding window / `use_mrope` / `q_norm` / `partial_rotary_factor` are ignored. Fine for T0 (`use_sliding_window: false`, `use_mrope: false`).
- `SamplingParams.n > 1` does not fork. Default `temperature=1.0` is not greedy (tests pass `0.0`).
- Architecture.md still says most of `slipstream/` is a stub; stale, not a runtime bug.
