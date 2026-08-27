# Phase 1 spec — Correct Engine

Binding: MASTERPLAN §10 Phase 1, S1, S6. Speed is forbidden as a goal.

**Gate 1**
- Greedy output token-identical to HuggingFace for ≥50 prompts × 128 tokens on Qwen2.5-0.5B (T0).
- Llama-family identity on TinyLlama-1.1B (T0 proxy). Llama-3.1-8B is the same code path; skip at runtime if weights/VRAM missing.
- Prefill logits vs HF eager: `atol=1e-2, rtol=1e-2` (bf16).
- Baseline throughput recorded (naive column).

`slipstream/` must not import `transformers`, `vllm`, or `sglang`. Tests and benchmarks may.

---

## File ownership (do not cross)

| Agent | May write | Must not touch |
|---|---|---|
| **A1 Runtime** | `slipstream/models/**`, `slipstream/engine/model_runner.py`, `slipstream/engine/llm_engine.py`, `slipstream/engine/engine_core.py` | sampler.py, memory/*, tests/* |
| **A2 Memory** | `slipstream/memory/contiguous_cache.py`, export from `slipstream/memory/__init__.py` | models/*, engine/*, tests/* |
| **A6 Sampler** | `slipstream/engine/sampler.py` | everything else |
| **A7 Tests** | `tests/correctness/**`, `tests/test_phase1_*.py`, `docs/log/phase1.md` (append) | slipstream/* except if a test-only helper is needed under `tests/` |

---

## Shared numerics (copy exactly — this is HF 5.15 eager)

### RMSNorm
```
x32 = x.float()
x_hat = x32 * rsqrt(mean(x32^2, dim=-1, keepdim=True) + eps)
return weight * x_hat.to(x.dtype)
```

### RoPE (default)
```
inv_freq[i] = 1 / (theta ** (2i / head_dim))   # i = 0,2,4,...  dtype float32
freqs = inv_freq @ position_ids                 # [B, D/2, 1] @ [B, 1, T] -> [B, D/2, T]
emb = cat(freqs, freqs, dim=-1).transpose → [B, T, D]
cos, sin = emb.cos(), emb.sin()                 # compute in fp32, cast to x.dtype
rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2], dim=-1)
q' = q * cos + rotate_half(q) * sin             # unsqueeze cos/sin on heads dim
```
`use_mrope` is false for our models. Qwen2.5-0.5B: `theta=1e6`, `head_dim=64`. TinyLlama: `theta=1e4`, `head_dim=64`.

### Llama-3 RoPE (`rope_type == "llama3"`)
After default `inv_freq`, with `factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings`:
```
wavelen = 2π / inv_freq
low_w = orig_max / low_freq_factor
high_w = orig_max / high_freq_factor
inv = where(wavelen > low_w, inv_freq / factor, inv_freq)
smooth = (orig_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
smoothed = (1 - smooth) * inv / factor + smooth * inv
medium = ~(wavelen < high_w) & ~(wavelen > low_w)
inv_freq = where(medium, smoothed, inv)
```

### Attention (eager, must match HF — do **not** use SDPA for the parity path)
```
scale = head_dim ** -0.5
k, v = repeat_kv(k, v, n_q_heads // n_kv_heads)   # expand, not reshape-alias
scores = (q @ k.T) * scale
scores = scores + causal_mask          # 0 allowed, -inf forbidden positions
weights = softmax(scores, dim=-1, dtype=fp32).to(q.dtype)
out = weights @ v                       # then merge heads, o_proj
```
Causal mask: `triu` on the *key* axis relative to query positions (prefill Tq=Tk; decode Tq=1, Tk=cache).

Qwen: `q/k/v` **bias=True**, `o_proj` bias=False.
Llama / TinyLlama: `attention_bias=False` (no qkvo bias). `mlp_bias=False`.

### MLP
`down(silu(gate(x)) * up(x))` — all bias-free for our models.

### Decoder layer
```
x = x + attn(rms(x))
x = x + mlp(rms(x))
```
Final `rms` then `lm_head`. Tied embeddings: `lm_head.weight is embed_tokens.weight` (Qwen2.5-0.5B has **no** `lm_head` tensor).

### Weight names (HF)
```
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.layers.{i}.self_attn.{q,k,v}_proj.bias     # Qwen only
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.{gate,up,down}_proj.weight
model.norm.weight
lm_head.weight                                   # absent if tied
```

---

## A2 — `NaiveKVCache`

`slipstream/memory/contiguous_cache.py`

```python
class NaiveKVCache:
    """Throwaway contiguous cache. Phase 2 paging must be semantically identical."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None: ...

    def reset(self) -> None:
        """Zero length; buffers may be retained."""

    @property
    def seq_len(self) -> int: ...

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append k,v of shape [B, n_kv, T_new, D].
        Return full cached k,v of shape [B, n_kv, seq_len, D] (views, not clones).
        Layer 0's update advances seq_len; other layers must see the same seq_len.
        """
```

Layout: `k_cache, v_cache` each `[L, B, n_kv, max_len, D]`. No paging. Invariant: `seq_len <= max_len`.

---

## A6 — `Sampler`

`slipstream/engine/sampler.py`

```python
class Sampler:
    def sample(
        self,
        logits: torch.Tensor,          # [B, vocab]
        params: SamplingParams,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:                 # [B] int64 on same device
```

Rules:
- `temperature == 0` or `top_k == 1` → argmax (greedy). **No softmax.**
- Else `logits / temperature`, then top-k (if `top_k > 0`), then top-p (if `top_p < 1`), then multinomial.
- `min_p` and `repetition_penalty` implemented (even if Gate 1 is greedy).
- Seeded: `params.seed` constructs a `torch.Generator` on the logits device. Same seed ⇒ same tokens (I6.1).
- Do not `.item()` / `.cpu()` inside `sample`. Finish/stop checks belong in the engine.

---

## A1 — model + engine

### Config load
`load_hf_config(path) -> ModelConfig` reads `config.json` only (stdlib json). Map:
- `num_hidden_layers → num_layers`
- `num_attention_heads → num_q_heads`
- `num_key_value_heads → num_kv_heads`
- `head_dim` or `hidden_size // num_attention_heads`
- `attention_bias` (default True for qwen2, False for llama)
- `rope_theta`, `rope_scaling` / `rope_parameters`
- `tie_word_embeddings`, `bos_token_id`, `eos_token_id`, `model_type`

Resolve a hub id to a local snapshot by walking `$HF_HOME/hub/models--org--name/refs/main` → `snapshots/<sha>`. Do not download. Do not import transformers.

### Tokenizer
`slipstream/models/tokenizer.py` — `tokenizers.Tokenizer.from_file(snapshot / "tokenizer.json")`. Encode without special tokens by default (`add_special_tokens=False`) so we control BOS. Decode skip_special_tokens=True for text.

### `CausalLM` (shared)
One module used by `LlamaForCausalLM` and `QwenForCausalLM`. Difference is only `attention_bias` / `mlp_bias` / rope type from config.

```python
def forward(
    self,
    input_ids: Tensor,            # [B, T]
    positions: Tensor,            # [B, T]
    kv_cache: NaiveKVCache,
) -> Tensor:                      # logits [B, T, vocab]
```

Every layer calls `kv_cache.update`. Prefill: T=prompt. Decode: T=1, positions continue.

### `LLMEngine.generate(request) -> list[int]`
1. Tokenize if needed.
2. `NaiveKVCache` sized to `min(max_model_len, prompt + max_tokens)`.
3. Prefill `input_ids`, take last-token logits, sample.
4. Decode loop until EOS / stop id / max_tokens.
5. Set `Sequence` status (`FINISHED_STOPPED` / `FINISHED_LENGTH`).
6. Return output token ids only (not the prompt).

`eos_token_id` from config. Qwen2.5-0.5B: 151643. TinyLlama: 2.

Device: CUDA if available else CPU. Dtype from `ModelConfig.dtype`.

`EngineCore.step` / `add_request` can stay thin wrappers around the same path for a single sequence. Do not implement continuous batching.

---

## A7 — tests

CPU-safe unit tests (no weights):
- RMSNorm vs the formula on random tensors.
- `rotate_half` / default RoPE vs a 10-line reference.
- GQA `repeat_kv` shape: `[B, 2, T, D] → [B, 14, T, D]` for Qwen 0.5B.
- Sampler: greedy is argmax; same seed ⇒ same sample; `sample` stays on device.
- Cache: append twice, `seq_len` and values correct; overflow raises.

GPU / weight tests (`@pytest.mark.gpu`, skip if no CUDA or no snapshot):
- `test_parity_logits`: one short prompt, last-token logits vs `transformers` **eager** (`attn_implementation="eager"`), `atol=1e-2, rtol=1e-2`.
- `test_parity_greedy_qwen`: token identity, start with 8 prompts × 32 tokens in default CI; `@pytest.mark.slow` for 50 × 128.
- `test_parity_greedy_tinyllama`: same, smaller N if VRAM tight.
- Force HF to eager. Do not compare against SDPA/FA2 — they diverge in bf16.

Prompts: reuse `benchmarks.baselines.hf_generate.DEFAULT_PROMPTS` plus enough extras to reach 50 for the slow test.

After a successful logit+small greedy run, append numbers to `docs/log/phase1.md` and write `benchmarks/results/phase1/naive_t0.json` + manifest (A7 may call `benchmarks.manifest`).

---

## Quality bar

- Full type hints on public APIs. Ruff + mypy clean on `slipstream/`.
- Module docstring states invariants.
- No `print` debugging left in.
- No unjustified extras (no CUDA graphs, no paging, no Triton).
- Conventional comments only for non-obvious numeric constraints (fp32 softmax, RoPE fp32).
