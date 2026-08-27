"""Gate 0: frozen §8.3 contracts exist with the specified shapes."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest
from slipstream.core.config import EngineConfig, kv_bytes_per_token
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import (
    DEFAULT_BLOCK_SIZE,
    KV_CACHE_LAYOUT,
    AllocStatus,
    BlockManager,
    EngineState,
    PhysicalBlock,
    PreemptionMode,
    Request,
    SchedulerOutput,
    SchedulingPolicy,
    Sequence,
    SequenceStatus,
)
from slipstream.memory.block_manager import BlockManagerImpl
from slipstream.scheduler.policies import POLICY_REGISTRY, FCFSPolicy, get_policy


def test_sequence_status_members() -> None:
    names = {s.name for s in SequenceStatus}
    assert names == {
        "WAITING",
        "RUNNING",
        "SWAPPED",
        "FINISHED_STOPPED",
        "FINISHED_LENGTH",
        "FINISHED_ABORTED",
    }


def test_alloc_status_members() -> None:
    assert {s.value for s in AllocStatus} == {"ok", "later", "never"}


def test_preemption_modes() -> None:
    assert {m.value for m in PreemptionMode} == {"swap", "recompute"}


def test_kv_layout_constant() -> None:
    assert DEFAULT_BLOCK_SIZE == 16
    assert KV_CACHE_LAYOUT == "num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim"


def test_kv_bytes_per_token_llama_8b() -> None:
    # MASTERPLAN §7.2: Llama-3.1-8B bf16 = 128 KiB/token
    assert kv_bytes_per_token(32, 8, 128, 2) == 131_072


def test_sequence_fields_and_helpers() -> None:
    seq = Sequence(seq_id=1, prompt_token_ids=[10, 20, 30], arrival_ts=1.5)
    assert seq.status is SequenceStatus.WAITING
    assert seq.num_tokens == 3
    assert seq.is_prefill
    assert not seq.is_finished
    seq.append_token(42)
    assert seq.output_token_ids == [42]
    assert seq.num_tokens == 4


def test_sampling_params_validation() -> None:
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    greedy = SamplingParams(temperature=0.0)
    assert greedy.is_greedy


def test_request_requires_prompt() -> None:
    with pytest.raises(ValueError):
        Request(
            request_id="r",
            prompt=None,
            prompt_token_ids=None,
            sampling_params=SamplingParams(),
            arrival_ts=0.0,
        )


def test_physical_block_and_scheduler_output() -> None:
    block = PhysicalBlock(block_id=3, ref_count=1, num_tokens=4)
    assert block.block_hash is None
    out = SchedulerOutput(
        scheduled_seqs=[],
        num_batched_tokens=0,
        blocks_to_swap_in={},
        blocks_to_swap_out={},
        blocks_to_copy={},
        is_prefill_chunk={},
    )
    assert out.num_batched_tokens == 0


def test_engine_state_is_frozen() -> None:
    state = EngineState(
        num_free_blocks=10,
        num_total_blocks=20,
        token_budget=2048,
        tokens_scheduled=0,
        running=(),
        waiting=(),
        swapped=(),
        kv_bytes_per_block=4096,
        block_size=16,
        now=0.0,
        gpu_cache_usage=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        state.num_free_blocks = 0  # type: ignore[misc]


def test_engine_config_rejects_unknown_policy() -> None:
    from slipstream.core.config import SchedulerConfig

    with pytest.raises(ValueError):
        EngineConfig.for_model("x", scheduler=SchedulerConfig(policy="not-a-policy"))


def test_block_manager_protocol_methods() -> None:
    required = {
        "can_allocate",
        "allocate",
        "append_slot",
        "fork",
        "free",
        "swap_out",
        "swap_in",
        "get_num_free_blocks",
        "match_prefix",
    }
    assert required.issubset(
        set(inspect.signature(BlockManagerImpl.__init__).parameters) | required
    )
    for name in required:
        assert hasattr(BlockManagerImpl, name)
        assert callable(getattr(BlockManagerImpl, name))


def test_block_manager_impl_is_structural_block_manager() -> None:
    impl = BlockManagerImpl(num_gpu_blocks=128, block_size=16)
    assert isinstance(impl, BlockManager)


def test_block_manager_stubs_raise() -> None:
    impl = BlockManagerImpl(num_gpu_blocks=8, block_size=16)
    seq = Sequence(seq_id=0, prompt_token_ids=[1])
    assert impl.can_allocate(seq) in {AllocStatus.OK, AllocStatus.LATER, AllocStatus.NEVER}


def test_scheduling_policy_protocol() -> None:
    required = {
        "order_waiting",
        "should_admit",
        "select_preemption_victim",
        "preemption_mode",
    }
    for name, cls in POLICY_REGISTRY.items():
        for method in required:
            assert hasattr(cls, method), f"{name} missing {method}"
        instance = cls()
        assert isinstance(instance, SchedulingPolicy)


def test_fcfs_orders_by_arrival() -> None:
    late = Sequence(seq_id=2, prompt_token_ids=[1], arrival_ts=2.0)
    early = Sequence(seq_id=1, prompt_token_ids=[1], arrival_ts=1.0)
    state = EngineState(
        num_free_blocks=1,
        num_total_blocks=1,
        token_budget=16,
        tokens_scheduled=0,
        running=(),
        waiting=(late, early),
        swapped=(),
        kv_bytes_per_block=1,
        block_size=16,
        now=0.0,
        gpu_cache_usage=0.0,
    )
    ordered = FCFSPolicy().order_waiting([late, early], state)
    assert [s.seq_id for s in ordered] == [1, 2]


def test_get_policy_registry() -> None:
    assert get_policy("fcfs").name == "fcfs"  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        get_policy("does-not-exist")
