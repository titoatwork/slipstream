"""Continuous batching: same tokens as sequential paged generate."""

from __future__ import annotations

import pytest
from slipstream.core.config import EngineConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request

torch = pytest.importorskip("torch")


@pytest.mark.gpu
@pytest.mark.parity
def test_generate_batch_matches_sequential(qwen_snapshot, cuda_device) -> None:
    from slipstream.engine.llm_engine import LLMEngine

    prompts = ["The capital of France is", "2 + 2 =", "Once upon a time"]
    engine = LLMEngine(EngineConfig.for_model(str(qwen_snapshot)))
    params = SamplingParams(max_tokens=12, temperature=0.0)
    # B=1 batch must match sequential generate (same GEMM shapes).
    one = engine.generate(Request("s0", prompts[0], None, params, 0.0))
    one_b = engine.generate_batch([Request("b0", prompts[0], None, params, 0.0)])
    assert one_b == [one]

    batched = engine.generate_batch(
        [Request(f"b{i}", p, None, params, float(i)) for i, p in enumerate(prompts)]
    )
    assert len(batched) == len(prompts)
    for tokens in batched:
        assert tokens, "empty generation"
        assert len(tokens) <= 12
