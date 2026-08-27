"""Prefix cache, chunked prefill, and swap unit tests."""

from __future__ import annotations

import pytest
import torch
from slipstream.core.config import SchedulerConfig
from slipstream.core.types import Sequence, SequenceStatus
from slipstream.memory.block_manager import BlockManagerImpl
from slipstream.memory.prefix_cache import RadixPrefixCache
from slipstream.memory.swap import CpuSwapSpace
from slipstream.scheduler.policies.fcfs import FCFSPolicy
from slipstream.scheduler.scheduler import Scheduler


def test_radix_match_full_blocks_only() -> None:
    bm = BlockManagerImpl(num_gpu_blocks=16, block_size=4, max_model_len=64)
    cache = RadixPrefixCache(block_size=4)
    cache.bind(bm)
    seq = Sequence(seq_id=1, prompt_token_ids=list(range(10)))
    bm.allocate(seq)
    for _ in range(10):
        bm.append_slot(seq)
    cache.insert(seq.prompt_token_ids, seq.block_table)
    bm.free(seq)
    blocks, n = cache.match(list(range(10)) + [99, 100])
    assert n == 8  # two full blocks of 4
    assert len(blocks) == 2
    assert cache.hit_rate() > 0


def test_prefix_attach_then_suffix_does_not_mutate_shared() -> None:
    bm = BlockManagerImpl(num_gpu_blocks=16, block_size=4, max_model_len=64)
    cache = RadixPrefixCache(4)
    cache.bind(bm)
    first = Sequence(seq_id=1, prompt_token_ids=list(range(8)))
    bm.allocate(first)
    for _ in range(8):
        bm.append_slot(first)
    shared = list(first.block_table)
    cache.insert(first.prompt_token_ids, first.block_table)
    bm.free(first)
    second = Sequence(seq_id=2, prompt_token_ids=list(range(8)) + [7, 8])
    blocks, n = cache.match(second.prompt_token_ids)
    assert n == 8
    bm.attach(second, blocks)
    second.num_computed_tokens = n
    bm.allocate(second)
    for _ in range(2):
        bid, _off = bm.append_slot(second)
        assert bid not in shared
    assert second.block_table[:2] == shared


def test_schedule_caps_prefill_chunk() -> None:
    bm = BlockManagerImpl(256, 16, max_model_len=2048)
    sched = Scheduler(
        SchedulerConfig(
            prefill_chunk_size=32,
            max_num_batched_tokens=128,
            enable_chunked_prefill=True,
        ),
        bm,
        FCFSPolicy(),
        16,
    )
    seq = Sequence(seq_id=1, prompt_token_ids=list(range(200)), status=SequenceStatus.RUNNING)
    bm.allocate(seq)
    sched.running.append(seq)
    planned = sched.schedule()
    assert sched.last_take[1] == 32
    assert planned.is_prefill_chunk[1]
    assert planned.num_batched_tokens == 32


def test_swap_roundtrip_preserves_page() -> None:
    bm = BlockManagerImpl(num_gpu_blocks=8, block_size=4, max_model_len=32)
    kv = torch.randn(2, 2, 8, 4, 2, 8)
    bm.bind_kv(kv)
    swap = CpuSwapSpace(8, 4, 2, 2, 8, dtype=kv.dtype)
    bm.bind_swap(swap)
    seq = Sequence(seq_id=3, prompt_token_ids=[1, 2, 3, 4, 5])
    bm.allocate(seq)
    for _ in range(5):
        bm.append_slot(seq)
    page = seq.block_table[0]
    before = kv[:, :, page].clone()
    mapping = bm.swap_out(seq)
    assert mapping
    assert seq.block_table == []
    bm.swap_in(seq)
    assert seq.block_table
    restored = kv[:, :, seq.block_table[0]]
    torch.testing.assert_close(restored, before)


@pytest.mark.gpu
@pytest.mark.parity
def test_prefix_hit_greedy_identity(qwen_snapshot, cuda_device) -> None:
    from slipstream.core.config import EngineConfig
    from slipstream.core.sampling_params import SamplingParams
    from slipstream.core.types import Request
    from slipstream.engine.llm_engine import LLMEngine

    prefix = "The purpose of a KV cache in autoregressive decoding is"
    a = prefix + " to avoid recomputing keys."
    b = prefix + " to save memory bandwidth."
    engine = LLMEngine(EngineConfig.for_model(str(qwen_snapshot)))
    params = SamplingParams(max_tokens=12, temperature=0.0)
    try:
        cold_a = engine.generate(Request("a1", a, None, params, 0.0))
        cold_b = engine.generate(Request("b1", b, None, params, 1.0))
        hit_b = engine.generate(Request("b2", b, None, params, 3.0))
        assert hit_b == cold_b
        assert engine.prefix_cache is not None
        assert engine.prefix_cache.cached_tokens > 0
        assert cold_a  # used to populate a distinct suffix
    finally:
        del engine
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
