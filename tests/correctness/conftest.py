"""Phase 1 correctness fixtures.

Snapshots are path-only at collection time. 0.5B / 1.1B weights are loaded
inside individual tests (or inside helpers those tests call), never here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Parity tests must use the local hub snapshots. Do not download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

QWEN_SNAPSHOT = Path(
    "/home/titoisalive/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B/snapshots/"
    "060db6499f32faf8b98477b0a26969ef7d8b9987"
)
TINYLLAMA_SNAPSHOT = Path(
    "/home/titoisalive/.cache/huggingface/hub/"
    "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/"
    "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
)

PARITY_PROMPT = "The capital of France is"


def _require_snapshot(path: Path, label: str) -> Path:
    if not path.is_dir() or not (path / "config.json").is_file():
        pytest.skip(f"{label} snapshot missing: {path}")
    return path


@pytest.fixture
def qwen_snapshot() -> Path:
    return _require_snapshot(QWEN_SNAPSHOT, "Qwen2.5-0.5B")


@pytest.fixture
def tinyllama_snapshot() -> Path:
    return _require_snapshot(TINYLLAMA_SNAPSHOT, "TinyLlama-1.1B")


@pytest.fixture
def cuda_device():
    """CUDA device for @pytest.mark.gpu tests. Skips if CUDA is absent."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Phase 1 parity")
    yield torch.device("cuda")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
