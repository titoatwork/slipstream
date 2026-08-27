"""Paging must be semantically invisible vs the Phase 1 naive cache."""

from __future__ import annotations

import pytest
from slipstream.core.config import CacheConfig, EngineConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request

torch = pytest.importorskip("torch")


@pytest.mark.gpu
@pytest.mark.parity
def test_paged_matches_naive_greedy_qwen(qwen_snapshot, cuda_device) -> None:
    from slipstream.engine.llm_engine import LLMEngine

    prompts = [
        "The capital of France is",
        "2 + 2 =",
        "def fibonacci(n):",
    ]
    naive_cfg = EngineConfig.for_model(str(qwen_snapshot), cache=CacheConfig(enable_paging=False))
    paged_cfg = EngineConfig.for_model(str(qwen_snapshot), cache=CacheConfig(enable_paging=True))
    naive = LLMEngine(naive_cfg)
    ours_naive = []
    for i, p in enumerate(prompts):
        ours_naive.append(
            naive.generate(
                Request(f"n{i}", p, None, SamplingParams(max_tokens=16, temperature=0.0), 0.0)
            )
        )
    del naive
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    paged = LLMEngine(paged_cfg)
    for i, p in enumerate(prompts):
        got = paged.generate(
            Request(f"p{i}", p, None, SamplingParams(max_tokens=16, temperature=0.0), 0.0)
        )
        assert got == ours_naive[i], f"prompt {p!r} paged={got} naive={ours_naive[i]}"
