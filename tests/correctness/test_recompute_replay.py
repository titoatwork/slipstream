"""RECOMPUTE must restore KV for prompt + prior outputs (not prompt only)."""

from __future__ import annotations

from slipstream.core.config import SchedulerConfig
from slipstream.core.types import Sequence
from slipstream.memory.block_manager import BlockManagerImpl
from slipstream.scheduler.policies.horizon import HorizonPolicy
from slipstream.scheduler.replay import (
    kv_uncomputed,
    kv_written_target,
    needs_replay,
    replay_token_ids,
)
from slipstream.scheduler.scheduler import Scheduler


def test_replay_target_matches_incremental_invariant() -> None:
    prompt = list(range(8))
    seq = Sequence(seq_id=1, prompt_token_ids=prompt)
    assert kv_written_target(seq) == 8
    assert replay_token_ids(seq) == prompt
    assert needs_replay(seq)

    seq.num_computed_tokens = 8
    seq.output_token_ids = [99]
    # Last sample is the next decode query — not yet in KV.
    assert kv_written_target(seq) == 8
    assert not needs_replay(seq)
    assert replay_token_ids(seq) == prompt

    seq.output_token_ids = [99, 100, 101]
    seq.num_computed_tokens = 0  # recompute
    assert kv_written_target(seq) == 10
    assert replay_token_ids(seq) == prompt + [99, 100]
    assert kv_uncomputed(seq) == 10
    assert needs_replay(seq)


def test_schedule_replays_outputs_after_recompute() -> None:
    bm = BlockManagerImpl(num_gpu_blocks=8, block_size=16, max_model_len=256)
    sched = Scheduler(SchedulerConfig(policy="horizon", max_num_seqs=4), bm, HorizonPolicy())
    seq = Sequence(seq_id=1, prompt_token_ids=list(range(8)), output_token_ids=[9, 10, 11])
    bm.allocate(seq)
    seq.num_computed_tokens = 0
    sched.running.append(seq)
    out = sched.schedule()
    assert seq in out.scheduled_seqs
    assert out.is_prefill_chunk.get(seq.seq_id) is True
    assert sched.last_take[seq.seq_id] == kv_uncomputed(seq)


def test_recompute_does_not_duplicate_waiting() -> None:
    bm = BlockManagerImpl(num_gpu_blocks=4, block_size=16, max_model_len=256)
    sched = Scheduler(SchedulerConfig(policy="horizon", max_num_seqs=4), bm, HorizonPolicy())
    seq = Sequence(seq_id=1, prompt_token_ids=list(range(8)))
    bm.allocate(seq)
    sched.running.append(seq)
    sched.preempt_recompute(seq)
    sched.preempt_recompute(seq)
    assert sched.waiting.count(seq) == 1
    assert seq not in sched.running


def test_can_allocate_peak_does_not_double_count_outputs() -> None:
    from slipstream.core.sampling_params import SamplingParams
    from slipstream.core.types import AllocStatus

    bm = BlockManagerImpl(num_gpu_blocks=3, block_size=16, max_model_len=256)
    seq = Sequence(
        seq_id=1,
        prompt_token_ids=list(range(5)),
        output_token_ids=[0] * 14,
        sampling_params=SamplingParams(max_tokens=32),
    )
    seq.oracle_output_len = 32
    # prompt+32 = 37 tokens → 3 pages. Current+32 would be NEVER.
    assert bm.can_allocate(seq) is not AllocStatus.NEVER


def test_code_like_extra_overrides_bpe_ids() -> None:
    from slipstream.core.sampling_params import SamplingParams
    from slipstream.scheduler.predictor.features import FeatureSet, extract_features

    seq = Sequence(
        seq_id=1,
        prompt_token_ids=[1000, 2000, 3000],
        sampling_params=SamplingParams(extra={"code_like": 1.0}),
    )
    feats = extract_features(seq, FeatureSet.F1)
    assert feats["code_like"] == 1.0
