"""HuggingFace ``tokenizer.json`` wrapper. Does not import transformers."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer as HfTokenizer  # type: ignore[import-untyped]


class Tokenizer:
    """Encode/decode via ``tokenizers.Tokenizer.from_file``.

    ``encode`` defaults to ``add_special_tokens=False`` so the engine
    controls BOS. ``decode`` skips special tokens by default.
    """

    def __init__(self, model_path: str | Path, *, model_type: str | None = None) -> None:
        snapshot = Path(model_path)
        tok_file = snapshot / "tokenizer.json" if snapshot.is_dir() else snapshot
        if not tok_file.is_file():
            raise FileNotFoundError(f"missing tokenizer.json at {tok_file}")
        self._tokenizer = HfTokenizer.from_file(str(tok_file))
        self.add_bos_token = False
        self.bos_token_id: int | None = None
        self.eos_token_id: int | None = None

        cfg_path = tok_file.parent / "tokenizer_config.json"
        tokenizer_class = ""
        if cfg_path.is_file():
            with cfg_path.open(encoding="utf-8") as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict):
                raise TypeError(f"{cfg_path} is not a JSON object")
            tokenizer_class = str(cfg.get("tokenizer_class") or "")
            if "add_bos_token" in cfg:
                self.add_bos_token = bool(cfg["add_bos_token"])
            elif "LlamaTokenizer" in tokenizer_class or model_type == "llama":
                self.add_bos_token = True
            bos = cfg.get("bos_token")
            eos = cfg.get("eos_token")
            if isinstance(bos, str):
                self.bos_token_id = self._tokenizer.token_to_id(bos)
            elif isinstance(bos, dict) and isinstance(bos.get("content"), str):
                self.bos_token_id = self._tokenizer.token_to_id(str(bos["content"]))
            if isinstance(eos, str):
                self.eos_token_id = self._tokenizer.token_to_id(eos)
            elif isinstance(eos, dict) and isinstance(eos.get("content"), str):
                self.eos_token_id = self._tokenizer.token_to_id(str(eos["content"]))
        elif model_type == "llama":
            self.add_bos_token = True

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return str(self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens))
