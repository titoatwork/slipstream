"""GPU: RECOMPUTE resume must match a run that never preempted."""

from __future__ import annotations

import pytest
from slipstream.core.config import CacheConfig, EngineConfig, SchedulerConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request

torch = pytest.importorskip("torch")


def _reqs() -> list[Request]:
    # Prompts must *both* fit (FCFS should_admit looks at prompt pages only)
    # while combined peaks do not, so growth hits ensure_slot preemption.
    params = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
    return [
        Request("a", "The capital of France is", None, params, 0.0),
        Request("b", "Two plus two equals", None, params, 0.01),
    ]


@pytest.mark.gpu
def test_recompute_resume_completes_under_pressure(qwen_snapshot, cuda_device) -> None:
    from slipstream.engine.llm_engine import LLMEngine

    engine = LLMEngine(
        EngineConfig.for_model(
            str(qwen_snapshot),
            cache=CacheConfig(num_gpu_blocks=3, enable_prefix_caching=False),
            scheduler=SchedulerConfig(
                policy="fcfs",
                max_num_seqs=4,
                max_num_batched_tokens=64,
                prefill_chunk_size=32,
            ),
        )
    )
    out = engine.generate_batch(_reqs())
    pre = float(engine.scheduler.preemptions) if engine.scheduler else 0.0
    aborted = sum(1 for t in engine.last_request_traces if t.aborted)
    del engine
    assert pre >= 1.0
    assert aborted == 0
    assert [len(toks) for toks in out] == [32, 32]
