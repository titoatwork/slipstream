"""Resolve a local HF snapshot and load ``config.json`` (stdlib json only)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from slipstream.core.config import ModelConfig

_QWEN_TYPES = frozenset({"qwen2", "qwen"})
_LLAMA_TYPES = frozenset({"llama"})


def resolve_model_path(model_id: str | Path, revision: str | None = None) -> Path:
    """Map a hub id or local dir to a snapshot that already contains ``config.json``.

    Walks ``$HF_HOME/hub/models--org--name`` (or ``$HUGGINGFACE_HUB_CACHE``).
    Does not download.
    """
    raw = Path(model_id)
    if raw.is_dir() and (raw / "config.json").is_file():
        return raw.resolve()

    hub = _hub_dir()
    cache_name = "models--" + str(model_id).replace("/", "--")
    repo = hub / cache_name
    if not repo.is_dir():
        raise FileNotFoundError(
            f"no local snapshot for {model_id!r} under {hub} "
            "(download the weights first; slipstream does not fetch)"
        )

    snapshots_root = repo / "snapshots"
    rev = revision or "main"
    ref = repo / "refs" / rev
    if ref.is_file():
        sha = ref.read_text(encoding="utf-8").strip()
        snap = snapshots_root / sha
        if (snap / "config.json").is_file():
            return snap.resolve()

    direct = snapshots_root / rev
    if (direct / "config.json").is_file():
        return direct.resolve()

    candidates = [
        p for p in snapshots_root.iterdir() if p.is_dir() and (p / "config.json").is_file()
    ]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if candidates:
        raise FileNotFoundError(
            f"ambiguous snapshots for {model_id!r} revision={rev!r}; "
            f"found {[c.name for c in candidates]}"
        )
    raise FileNotFoundError(f"no config.json under {repo}")


def load_hf_config(path: str | Path, model_id: str | None = None) -> ModelConfig:
    """Read ``config.json`` only. Architecture fields are required; no transformers."""
    snapshot = Path(path)
    if snapshot.is_file():
        cfg_path = snapshot
        snapshot = snapshot.parent
    else:
        cfg_path = snapshot / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json at {cfg_path}")

    data = _read_json(cfg_path)
    model_type = _as_opt_str(data.get("model_type"))
    hidden = _as_int(data["hidden_size"], "hidden_size")
    n_q = _as_int(data["num_attention_heads"], "num_attention_heads")
    n_layers = _as_int(data["num_hidden_layers"], "num_hidden_layers")
    vocab = _as_int(data["vocab_size"], "vocab_size")
    intermediate = _as_int(data["intermediate_size"], "intermediate_size")

    if "head_dim" in data and data["head_dim"] is not None:
        head_dim = _as_int(data["head_dim"], "head_dim")
    else:
        if hidden % n_q != 0:
            raise ValueError(f"hidden_size {hidden} not divisible by num_attention_heads {n_q}")
        head_dim = hidden // n_q

    n_kv_raw = data.get("num_key_value_heads")
    n_kv = _as_int(n_kv_raw, "num_key_value_heads") if n_kv_raw is not None else n_q

    attention_bias_default = model_type in _QWEN_TYPES
    if "attention_bias" in data and data["attention_bias"] is not None:
        attention_bias = _as_bool(data["attention_bias"], "attention_bias")
    else:
        attention_bias = attention_bias_default

    mlp_bias = (
        _as_bool(data["mlp_bias"], "mlp_bias")
        if "mlp_bias" in data and data["mlp_bias"] is not None
        else False
    )

    rope_scaling = _rope_dict(data)
    rope_type = _rope_type(rope_scaling)
    rope_theta = data.get("rope_theta")
    if rope_theta is None and rope_scaling is not None and "rope_theta" in rope_scaling:
        rope_theta = rope_scaling["rope_theta"]

    hidden_act = _as_opt_str(data.get("hidden_act")) or "silu"
    if hidden_act != "silu":
        raise ValueError(f"unsupported hidden_act: {hidden_act}")

    max_pos = data.get("max_position_embeddings")
    max_model_len = _as_int(max_pos, "max_position_embeddings") if max_pos is not None else 4096

    eos = _first_id(data.get("eos_token_id"))
    bos = _first_id(data.get("bos_token_id"))
    pad = _first_id(data.get("pad_token_id"))
    gen_path = snapshot / "generation_config.json"
    if gen_path.is_file():
        gen = _read_json(gen_path)
        if eos is None:
            eos = _first_id(gen.get("eos_token_id"))
        if bos is None:
            bos = _first_id(gen.get("bos_token_id"))
        if pad is None:
            pad = _first_id(gen.get("pad_token_id"))

    return ModelConfig(
        model_id=model_id if model_id is not None else str(snapshot),
        dtype=_map_dtype(data.get("torch_dtype")),
        max_model_len=max_model_len,
        num_layers=n_layers,
        hidden_size=hidden,
        num_q_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        vocab_size=vocab,
        intermediate_size=intermediate,
        rms_norm_eps=_as_float(data.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
        rope_theta=_as_float(rope_theta if rope_theta is not None else 10000.0, "rope_theta"),
        rope_scaling=rope_scaling,
        rope_type=rope_type,
        tie_word_embeddings=(
            _as_bool(data["tie_word_embeddings"], "tie_word_embeddings")
            if data.get("tie_word_embeddings") is not None
            else False
        ),
        attention_bias=attention_bias,
        mlp_bias=mlp_bias,
        model_type=model_type,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )


def apply_hf_config(dst: ModelConfig, src: ModelConfig) -> ModelConfig:
    """Copy architecture fields from ``src`` onto ``dst``.

    Keeps ``dst.model_id``, ``dst.revision``, ``dst.dtype``, and ``dst.max_model_len``.
    """
    dst.num_layers = src.num_layers
    dst.hidden_size = src.hidden_size
    dst.num_q_heads = src.num_q_heads
    dst.num_kv_heads = src.num_kv_heads
    dst.head_dim = src.head_dim
    dst.vocab_size = src.vocab_size
    dst.intermediate_size = src.intermediate_size
    dst.rms_norm_eps = src.rms_norm_eps
    dst.rope_theta = src.rope_theta
    dst.rope_scaling = src.rope_scaling
    dst.rope_type = src.rope_type
    dst.tie_word_embeddings = src.tie_word_embeddings
    dst.attention_bias = src.attention_bias
    dst.mlp_bias = src.mlp_bias
    dst.model_type = src.model_type
    dst.bos_token_id = src.bos_token_id
    dst.eos_token_id = src.eos_token_id
    dst.pad_token_id = src.pad_token_id
    return dst


def is_qwen(model_type: str | None) -> bool:
    return model_type in _QWEN_TYPES


def is_llama(model_type: str | None) -> bool:
    return model_type in _LLAMA_TYPES


def _hub_dir() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    return data


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise TypeError(f"{name} must be int, got {type(value).__name__}: {value!r}")
    return int(value)


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be float, got {type(value).__name__}: {value!r}")
    return float(value)


def _as_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {type(value).__name__}: {value!r}")
    return value


def _as_opt_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"expected str, got {type(value).__name__}")
    return value


def _first_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("token id must not be bool")
    if isinstance(value, int):
        return value
    if isinstance(value, list) and value:
        return _first_id(value[0])
    return None


def _map_dtype(value: object) -> str:
    if value is None:
        return "bfloat16"
    if not isinstance(value, str):
        return "bfloat16"
    mapping = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
        "half": "float16",
        "float32": "float32",
        "fp32": "float32",
    }
    return mapping.get(value, "bfloat16")


def _rope_dict(data: dict[str, object]) -> dict[str, object] | None:
    raw = data.get("rope_scaling")
    if raw is None:
        raw = data.get("rope_parameters")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("rope_scaling / rope_parameters must be an object")
    return {str(k): v for k, v in raw.items()}


def _rope_type(scaling: dict[str, object] | None) -> str:
    if scaling is None:
        return "default"
    raw = scaling.get("rope_type")
    if raw is None:
        raw = scaling.get("type")
    if raw is None or raw == "default":
        return "default"
    return str(raw)
