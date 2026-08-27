"""Safetensors weight loading. Must not import transformers modeling code."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open

from slipstream.core.config import ModelConfig
from slipstream.models.causal_lm import CausalLM
from slipstream.models.hf_config import (
    apply_hf_config,
    is_llama,
    is_qwen,
    load_hf_config,
    resolve_model_path,
)
from slipstream.models.llama import LlamaForCausalLM
from slipstream.models.qwen import QwenForCausalLM

logger = logging.getLogger(__name__)

_SKIP_KEY_PARTS = ("inv_freq", "rotary_emb", "cos_cached", "sin_cached")


def load_model(
    model_id: str | Path,
    config: ModelConfig | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> CausalLM:
    """Load a Llama or Qwen checkpoint into a Slipstream module.

    ``model_id`` is a local directory or a Hugging Face hub id whose weights
    have already been cached. Architecture is inferred from config.json.
    """
    revision = config.revision if config is not None else None
    snapshot = resolve_model_path(model_id, revision=revision)
    loaded = load_hf_config(snapshot, model_id=str(model_id))
    if config is None:
        config = loaded
    else:
        apply_hf_config(config, loaded)

    model: CausalLM
    if is_qwen(config.model_type):
        model = QwenForCausalLM(config, device=device, dtype=dtype)
    elif is_llama(config.model_type):
        model = LlamaForCausalLM(config, device=device, dtype=dtype)
    else:
        raise ValueError(f"unsupported model_type: {config.model_type!r}")

    load_weights(model, snapshot, config)
    return model


def load_weights(model: CausalLM, snapshot: str | Path, config: ModelConfig) -> None:
    """Copy safetensors tensors into modules by HF name.

    If ``lm_head.weight`` is absent and ``tie_word_embeddings``, assign
    ``lm_head.weight = embed_tokens.weight`` (same Parameter).
    """
    params = dict(model.named_parameters())
    used: set[str] = set()
    unexpected: list[str] = []

    for shard in _safetensor_files(Path(snapshot)):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            shard_keys = list(handle.keys())
            for key in shard_keys:
                if _skip_key(key):
                    continue
                if key not in params:
                    unexpected.append(key)
                    continue
                tensor = handle.get_tensor(key)
                dest = params[key]
                if tensor.shape != dest.shape:
                    raise RuntimeError(
                        f"shape mismatch for {key}: checkpoint {tuple(tensor.shape)} "
                        f"vs module {tuple(dest.shape)}"
                    )
                dest.data.copy_(tensor.to(device=dest.device, dtype=dest.dtype))
                used.add(key)

    missing = [name for name in params if name not in used]
    if "lm_head.weight" in missing and config.tie_word_embeddings:
        model.tie_lm_head()
        missing = [name for name in missing if name != "lm_head.weight"]

    if missing:
        raise RuntimeError(f"missing required weights: {sorted(missing)}")
    if unexpected:
        logger.warning("unexpected checkpoint keys ignored: %s", sorted(unexpected))

    for name, param in model.named_parameters():
        if not torch.isfinite(param.detach()).all():
            raise RuntimeError(f"non-finite parameter after load: {name}")


def _safetensor_files(snapshot: Path) -> list[Path]:
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise FileNotFoundError(f"empty weight_map in {index}")
        files = sorted({str(v) for v in weight_map.values()})
        paths = [snapshot / name for name in files]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"missing shard {path}")
        return paths

    single = snapshot / "model.safetensors"
    if single.is_file():
        return [single]

    shards = sorted(snapshot.glob("*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"no safetensors weights under {snapshot}")


def _skip_key(name: str) -> bool:
    return any(part in name for part in _SKIP_KEY_PARTS)
