"""Adapters between the Phase 1 spec APIs and whatever A1 actually exported.

Expected APIs (phase1_spec.md):

  slipstream.models.layers.rmsnorm.RMSNorm
  slipstream.models.layers.rope.rotate_half / apply_rotary_pos_emb / RotaryEmbedding
  slipstream.models.layers.attention.repeat_kv
  slipstream.memory.contiguous_cache.NaiveKVCache
  slipstream.engine.sampler.Sampler
  slipstream.models.loader.load_model
  CausalLM.forward(input_ids, positions, kv_cache) -> logits [B, T, vocab]
  LLMEngine(EngineConfig.for_model(...)).generate(Request(...)) -> list[int]

HF is a test-only reference. Always load with attn_implementation="eager".
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

_LAYER_PREFERRED: dict[str, tuple[str, ...]] = {
    "RMSNorm": (
        "slipstream.models.layers.rmsnorm",
        "slipstream.models.layers.norm",
        "slipstream.models.layers",
    ),
    "rmsnorm": ("slipstream.models.layers.rmsnorm", "slipstream.models.layers"),
    "rotate_half": (
        "slipstream.models.layers.rope",
        "slipstream.models.layers.rotary",
        "slipstream.models.layers",
    ),
    "apply_rotary_pos_emb": (
        "slipstream.models.layers.rope",
        "slipstream.models.layers.rotary",
        "slipstream.models.layers",
    ),
    "RotaryEmbedding": (
        "slipstream.models.layers.rope",
        "slipstream.models.layers.rotary",
        "slipstream.models.layers",
    ),
    "repeat_kv": (
        "slipstream.models.layers.attention",
        "slipstream.models.layers",
    ),
}


def _layers_modules() -> list[str]:
    names: list[str] = ["slipstream.models.layers"]
    try:
        pkg = importlib.import_module("slipstream.models.layers")
    except ModuleNotFoundError:
        return names
    pkg_file = getattr(pkg, "__file__", None)
    if pkg_file is None:
        return names
    for py in sorted(Path(pkg_file).resolve().parent.glob("*.py")):
        if py.name == "__init__.py":
            continue
        names.append(f"slipstream.models.layers.{py.stem}")
    return names


def import_layers_symbol(*names: str) -> Any:
    """Return the first matching public name under models.layers; skip if absent."""
    modules: list[str] = []
    for name in names:
        modules.extend(_LAYER_PREFERRED.get(name, ()))
    modules.extend(_layers_modules())
    seen: set[str] = set()
    for mod_name in modules:
        if mod_name in seen:
            continue
        seen.add(mod_name)
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        for name in names:
            if hasattr(mod, name):
                return getattr(mod, name)
    pytest.skip(f"A1 has not exported {names} from slipstream.models.layers yet")


def _bind_known(fn: Callable[..., Any], mapping: dict[str, Any]) -> dict[str, Any]:
    target = fn.__init__ if inspect.isclass(fn) else fn
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return {}
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue
        if name in mapping:
            kwargs[name] = mapping[name]
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            break
    return kwargs


# ---------------------------------------------------------------------------
# Spec reference numerics (HF 5.15 eager / docs/phase1_spec.md)
# ---------------------------------------------------------------------------


def spec_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    x_hat = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x_hat.to(x.dtype)


def spec_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def spec_rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Default RoPE. ~15 lines matching HF 5.15 eager.

    inv_freq[i] = 1 / (theta ** (2i / head_dim)) for i = 0,1,...,D/2-1
    freqs = inv_freq @ positions   # [B, D/2, 1] @ [B, 1, T] -> [B, D/2, T]
    emb = cat(freqs, freqs).transpose → [B, T, D]
    """
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=position_ids.device) / head_dim
        )
    )
    inv = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    pos = position_ids[:, None, :].to(torch.float32)
    freqs = (inv @ pos).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def spec_apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, H, T, D]; cos/sin: [B, T, D] → unsqueeze heads dim
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    return (
        q * cos_u + spec_rotate_half(q) * sin_u,
        k * cos_u + spec_rotate_half(k) * sin_u,
    )


def spec_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, n_kv, slen, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, n_kv, n_rep, slen, head_dim)
    return expanded.reshape(batch, n_kv * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Layer callers
# ---------------------------------------------------------------------------


def run_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Expected: RMSNorm(hidden_size, eps=...).weight; forward(x)."""
    rms_cls = import_layers_symbol("RMSNorm", "RmsNorm", "rmsnorm")
    hidden = int(x.shape[-1])
    if inspect.isclass(rms_cls):
        kwargs = _bind_known(
            rms_cls,
            {
                "hidden_size": hidden,
                "dim": hidden,
                "normalized_shape": hidden,
                "eps": eps,
                "variance_epsilon": eps,
                "device": x.device,
                "dtype": x.dtype,
            },
        )
        module = rms_cls(**kwargs) if kwargs else rms_cls(hidden, eps=eps)
        if hasattr(module, "weight"):
            with torch.no_grad():
                module.weight.copy_(weight.to(dtype=module.weight.dtype))
        else:
            pytest.fail("RMSNorm instance has no .weight parameter")
        module.eval()
        with torch.no_grad():
            out = module(x)
        if not torch.is_tensor(out):
            pytest.fail(f"RMSNorm.forward returned {type(out)}")
        return out

    kwargs = _bind_known(rms_cls, {"x": x, "hidden_states": x, "weight": weight, "eps": eps})
    if kwargs:
        return rms_cls(**kwargs)
    return rms_cls(x, weight, eps)


def run_rotate_half(x: torch.Tensor) -> torch.Tensor:
    fn = import_layers_symbol("rotate_half")
    return fn(x)


def run_apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected: apply_rotary_pos_emb(q, k, cos, sin) -> (q', k')."""
    fn = import_layers_symbol("apply_rotary_pos_emb", "apply_rope")
    kwargs = _bind_known(
        fn,
        {
            "q": q,
            "k": k,
            "query": q,
            "key": k,
            "cos": cos,
            "sin": sin,
            "unsqueeze_dim": 1,
        },
    )
    try:
        out = fn(**kwargs) if kwargs else fn(q, k, cos, sin)
    except TypeError:
        # Single-tensor apply; run q and k separately.
        q_out = fn(q, cos, sin)
        k_out = fn(k, cos, sin)
        return q_out, k_out
    if isinstance(out, tuple) and len(out) >= 2:
        return out[0], out[1]
    if torch.is_tensor(out):
        return out, fn(k, cos, sin)
    pytest.fail(f"apply_rotary_pos_emb returned {type(out)}")


def _tiny_rope_config(head_dim: int, theta: float, max_pos: int) -> Any:
    from slipstream.core.config import ModelConfig

    return ModelConfig(
        model_id="phase1-rope-ref",
        hidden_size=head_dim,
        num_q_heads=1,
        num_kv_heads=1,
        head_dim=head_dim,
        rope_theta=theta,
        rope_type="default",
        max_model_len=max_pos,
    )


def make_rotary_embedding(head_dim: int, theta: float, max_pos: int) -> Any:
    """Expected: RotaryEmbedding(head_dim=..., theta=..., max_position_embeddings=...)."""
    cls = import_layers_symbol(
        "RotaryEmbedding",
        "LlamaRotaryEmbedding",
        "Qwen2RotaryEmbedding",
        "RotaryPositionalEmbedding",
    )
    mapping = {
        "head_dim": head_dim,
        "dim": head_dim,
        "rotary_dim": head_dim,
        "theta": theta,
        "base": theta,
        "rope_theta": theta,
        "max_position_embeddings": max_pos,
        "max_seq_len": max_pos,
        "max_position": max_pos,
        "max_seq_len_cached": max_pos,
        "rope_type": "default",
        "device": torch.device("cpu"),
        "config": _tiny_rope_config(head_dim, theta, max_pos),
    }
    kwargs = _bind_known(cls, mapping)
    if inspect.isclass(cls):
        return cls(**kwargs) if kwargs else cls(head_dim, theta)
    return cls(**kwargs) if kwargs else cls(head_dim, theta)


def rotary_cos_sin(
    rope: Any, position_ids: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    attempts: list[Callable[[], Any]] = [
        lambda: rope(like, position_ids),
        lambda: rope(position_ids),
        lambda: rope(x=like, position_ids=position_ids),
        lambda: rope(position_ids=position_ids),
        lambda: rope(like, positions=position_ids),
    ]
    last: Exception | None = None
    for call in attempts:
        try:
            out = call()
        except TypeError as exc:
            last = exc
            continue
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[1]
    pytest.fail(f"could not call RotaryEmbedding for cos/sin: {last}")


def run_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expected: repeat_kv(x, n_rep) -> [B, n_kv * n_rep, T, D].

    Spec attention comment also allows ``repeat_kv(k, v, n_rep)`` returning a pair.
    """
    fn = import_layers_symbol("repeat_kv")
    try:
        out = fn(hidden_states, n_rep)
    except TypeError:
        dummy_v = torch.zeros_like(hidden_states)
        out = fn(hidden_states, dummy_v, n_rep)
        if isinstance(out, tuple):
            return out[0]
        return out
    if isinstance(out, tuple):
        return out[0]
    return out


# ---------------------------------------------------------------------------
# Cache / sampler
# ---------------------------------------------------------------------------


def load_naive_kv_cache() -> type:
    try:
        from slipstream.memory.contiguous_cache import NaiveKVCache

        return NaiveKVCache
    except ImportError:
        mem = pytest.importorskip("slipstream.memory")
        if hasattr(mem, "NaiveKVCache"):
            return mem.NaiveKVCache
        pytest.skip("NaiveKVCache not exported from slipstream.memory")


def load_sampler() -> Any:
    mod = pytest.importorskip("slipstream.engine.sampler")
    if not hasattr(mod, "Sampler"):
        pytest.skip("slipstream.engine.sampler.Sampler missing")
    return mod.Sampler()


# ---------------------------------------------------------------------------
# Snapshots, tokenization, HF / Slipstream loaders
# ---------------------------------------------------------------------------


def read_hf_config(snapshot: Path) -> dict[str, Any]:
    return json.loads((snapshot / "config.json").read_text(encoding="utf-8"))


def tokenize_prompt(snapshot: Path, prompt: str) -> list[int]:
    """Slipstream tokenizer if present; else HF AutoTokenizer, no specials."""
    ids = _try_slipstream_encode(snapshot, prompt)
    if ids is not None:
        return ids
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), use_fast=True, local_files_only=True
    )
    encoded = tok.encode(prompt, add_special_tokens=False)
    return list(encoded)


def tokenize_prompts(snapshot: Path, prompts: Iterable[str]) -> list[list[int]]:
    return [tokenize_prompt(snapshot, p) for p in prompts]


def _try_slipstream_encode(snapshot: Path, prompt: str) -> list[int] | None:
    try:
        mod = importlib.import_module("slipstream.models.tokenizer")
    except ModuleNotFoundError:
        return None
    tok: Any = None
    if hasattr(mod, "Tokenizer"):
        cls = mod.Tokenizer
        for attempt in (
            lambda: cls(snapshot),
            lambda: cls(str(snapshot)),
            lambda: cls.from_file(snapshot / "tokenizer.json"),
        ):
            try:
                tok = attempt()
                break
            except (TypeError, FileNotFoundError, ValueError, AttributeError):
                continue
    elif hasattr(mod, "load_tokenizer"):
        try:
            tok = mod.load_tokenizer(snapshot)
        except TypeError:
            tok = mod.load_tokenizer(str(snapshot))
    if tok is None:
        return None
    try:
        ids = tok.encode(prompt, add_special_tokens=False)
    except TypeError:
        ids = tok.encode(prompt)
    if hasattr(ids, "ids"):
        ids = ids.ids
    return list(ids)


def load_hf_eager(snapshot: Path, device: torch.device) -> Any:
    """Reference model. Never SDPA / FlashAttention-2."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    impl = getattr(model.config, "_attn_implementation", None)
    if impl != "eager":
        pytest.fail(f"HF attn_implementation={impl!r}, expected 'eager'")
    return model


def load_slipstream_model(snapshot: Path, device: torch.device) -> Any:
    """Expected: load_model(snapshot) -> CausalLM nn.Module on `device`."""
    loader = importlib.import_module("slipstream.models.loader")
    if not hasattr(loader, "load_model"):
        pytest.fail("slipstream.models.loader.load_model is missing")
    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(loader.load_model)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        if "device" in sig.parameters:
            kwargs["device"] = device
        if "dtype" in sig.parameters:
            kwargs["dtype"] = torch.bfloat16
    try:
        model = loader.load_model(snapshot, **kwargs)
    except TypeError:
        from slipstream.core.config import ModelConfig

        model = loader.load_model(str(snapshot), ModelConfig(model_id=str(snapshot)), **kwargs)
    model = _unwrap_causal_lm(model)
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _unwrap_causal_lm(obj: Any) -> Any:
    # CausalLM has .model (the decoder). Do not unwrap that — we need logits.
    if hasattr(obj, "lm_head") and callable(getattr(obj, "forward", None)):
        return obj
    inner = getattr(obj, "model", None)
    if inner is not None and hasattr(inner, "lm_head"):
        return inner
    return obj


def _cfg_get(model: Any, *names: str) -> Any:
    cfg = getattr(model, "config", None)
    for name in names:
        if cfg is not None:
            value = getattr(cfg, name, None)
            if value is not None:
                return value
        value = getattr(model, name, None)
        if value is not None:
            return value
    return None


def infer_kv_meta(model: Any, snapshot: Path | None = None) -> tuple[int, int, int]:
    num_layers = _cfg_get(model, "num_layers", "num_hidden_layers")
    num_kv = _cfg_get(model, "num_kv_heads", "num_key_value_heads")
    head_dim = _cfg_get(model, "head_dim")
    hidden = _cfg_get(model, "hidden_size")
    n_q = _cfg_get(model, "num_q_heads", "num_attention_heads")
    if head_dim is None and hidden is not None and n_q is not None:
        head_dim = int(hidden) // int(n_q)
    if snapshot is not None and any(v is None for v in (num_layers, num_kv, head_dim)):
        raw = read_hf_config(snapshot)
        num_layers = num_layers or raw.get("num_hidden_layers")
        num_kv = num_kv or raw.get("num_key_value_heads")
        hidden = hidden or raw.get("hidden_size")
        n_q = n_q or raw.get("num_attention_heads")
        head_dim = head_dim or raw.get("head_dim")
        if head_dim is None and hidden is not None and n_q is not None:
            head_dim = int(hidden) // int(n_q)
    if num_layers is None or num_kv is None or head_dim is None:
        pytest.fail(
            f"cannot infer KV meta from model.config / snapshot "
            f"(layers={num_layers}, kv={num_kv}, head_dim={head_dim})"
        )
    return int(num_layers), int(num_kv), int(head_dim)


def _param_dtype(model: Any, default: torch.dtype) -> torch.dtype:
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return default
    return default


def make_naive_cache(
    model: Any,
    *,
    batch: int,
    max_len: int,
    device: torch.device,
    snapshot: Path | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    NaiveKVCache = load_naive_kv_cache()
    num_layers, num_kv, head_dim = infer_kv_meta(model, snapshot)
    if dtype is None:
        dtype = _param_dtype(model, torch.bfloat16)
    return NaiveKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        max_batch=batch,
        max_len=max_len,
        dtype=dtype,
        device=device,
    )


def as_logits(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple | list) and out and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(f"cannot extract logits from {type(out)}")


def last_token_logits(out: Any) -> torch.Tensor:
    logits = as_logits(out)
    if logits.ndim == 3:
        return logits[:, -1, :]
    if logits.ndim == 2:
        return logits
    raise ValueError(f"unexpected logits rank {logits.ndim} shape={tuple(logits.shape)}")


def call_causal_lm_forward(
    model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: Any,
) -> torch.Tensor:
    """Expected: CausalLM.forward(input_ids, positions, kv_cache) -> [B, T, vocab]."""
    attempts: list[Callable[[], Any]] = [
        lambda: model.forward(input_ids, positions, kv_cache),
        lambda: model(input_ids, positions, kv_cache),
        lambda: model.forward(input_ids=input_ids, positions=positions, kv_cache=kv_cache),
        lambda: model(input_ids=input_ids, positions=positions, kv_cache=kv_cache),
        lambda: model.forward(input_ids, positions),
        lambda: model(input_ids=input_ids),
    ]
    last: Exception | None = None
    for call in attempts:
        try:
            return as_logits(call())
        except NotImplementedError:
            raise
        except TypeError as exc:
            last = exc
            continue
    raise TypeError(f"could not call CausalLM.forward: {last}")


def slipstream_prefill_logits(
    snapshot: Path, input_ids: torch.Tensor, model: Any | None = None
) -> torch.Tensor:
    """Last-token logits from a Slipstream prefill forward."""
    device = input_ids.device
    if model is None:
        model = load_slipstream_model(snapshot, device)
    batch, seq_len = input_ids.shape
    positions = (
        torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    )
    cache = make_naive_cache(model, batch=batch, max_len=seq_len, device=device, snapshot=snapshot)
    with torch.inference_mode():
        logits = call_causal_lm_forward(model, input_ids, positions, cache)
    return last_token_logits(logits)


def hf_prefill_last_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )
    return last_token_logits(out)


def make_generate_request(token_ids: list[int], max_new_tokens: int, request_id: str) -> Any:
    from slipstream.core.sampling_params import SamplingParams
    from slipstream.core.types import Request

    return Request(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=list(token_ids),
        sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=0.0),
        arrival_ts=0.0,
    )


def make_llm_engine(snapshot: Path) -> Any:
    from slipstream.core.config import EngineConfig
    from slipstream.engine.llm_engine import LLMEngine

    # Expected: LLMEngine(EngineConfig.for_model(...)).generate(Request(...))
    return LLMEngine(EngineConfig.for_model(str(snapshot)))


def _tokens_from_generate(out: Any) -> list[int]:
    if isinstance(out, list) and (not out or isinstance(out[0], int)):
        return list(out)
    if hasattr(out, "output_token_ids"):
        return list(out.output_token_ids)
    if hasattr(out, "token_ids"):
        return list(out.token_ids)
    raise TypeError(f"LLMEngine.generate returned {type(out)}")


def slipstream_generate(
    snapshot: Path,
    token_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    *,
    request_id: str = "phase1-parity",
    model: Any | None = None,
    engine: Any | None = None,
) -> list[int]:
    """Prefer LLMEngine.generate(Request(...)). Fallback: CausalLM.forward + argmax."""
    if engine is None:
        engine = make_llm_engine(snapshot)
    request = make_generate_request(token_ids, max_new_tokens, request_id)
    try:
        return _tokens_from_generate(engine.generate(request))
    except NotImplementedError:
        return _greedy_via_forward(snapshot, token_ids, max_new_tokens, device, model=model)


def _greedy_via_forward(
    snapshot: Path,
    token_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    *,
    model: Any | None = None,
) -> list[int]:
    # Fallback when LLMEngine.generate is still a stub. Greedy is argmax (A6 spec).
    if model is None:
        model = load_slipstream_model(snapshot, device)
    eos = read_hf_config(snapshot).get("eos_token_id")
    ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    cache = make_naive_cache(
        model,
        batch=1,
        max_len=ids.shape[1] + max_new_tokens,
        device=device,
        snapshot=snapshot,
    )
    generated: list[int] = []
    positions = torch.arange(ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        logits = last_token_logits(call_causal_lm_forward(model, ids, positions, cache))
        next_id = torch.argmax(logits, dim=-1)
        tid = int(next_id[0].item())
        generated.append(tid)
        if eos is not None and tid == int(eos):
            return generated
        for step in range(1, max_new_tokens):
            step_ids = next_id.view(1, 1)
            step_pos = torch.tensor([[ids.shape[1] + step - 1]], device=device, dtype=torch.long)
            logits = last_token_logits(call_causal_lm_forward(model, step_ids, step_pos, cache))
            next_id = torch.argmax(logits, dim=-1)
            tid = int(next_id[0].item())
            generated.append(tid)
            if eos is not None and tid == int(eos):
                break
    return generated


def hf_generate_greedy(
    model: Any, token_ids: list[int], max_new_tokens: int, device: torch.device
) -> list[int]:
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    eos = getattr(model.config, "eos_token_id", None)
    pad = getattr(model.config, "pad_token_id", None)
    if pad is None:
        pad = eos
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad,
            eos_token_id=eos,
        )
    return out[0, input_ids.shape[1] :].tolist()


def expand_default_prompts(n: int) -> list[str]:
    from benchmarks.baselines.hf_generate import DEFAULT_PROMPTS

    extras = (
        " Continue the thought.",
        " Explain briefly.",
        " Give one example.",
        " In one sentence.",
        " List the key points.",
        " Why?",
    )
    prompts: list[str] = []
    i = 0
    while len(prompts) < n:
        stem = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
        cycle = i // len(DEFAULT_PROMPTS)
        if cycle == 0:
            prompts.append(stem)
        else:
            extra = extras[(cycle - 1) % len(extras)]
            tag = "" if cycle <= len(extras) else f" [{cycle}]"
            prompts.append(f"{stem}{extra}{tag}")
        i += 1
    return prompts
