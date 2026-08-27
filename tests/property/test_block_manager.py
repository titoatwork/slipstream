"""Hypothesis fuzz for the paged allocator (Gate 2, I2.*)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from slipstream.core.types import AllocStatus, Sequence, SequenceStatus
from slipstream.memory.block_manager import BlockManagerImpl

pytestmark = pytest.mark.filterwarnings("ignore::hypothesis.errors.NonInteractiveExampleWarning")


def _seq(seq_id: int, n_prompt: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(max(n_prompt, 1))),
        status=SequenceStatus.WAITING,
    )


@settings(max_examples=80, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=4), min_size=20, max_size=80))
def test_random_ops_preserve_invariants(ops: list[int]) -> None:
    """10k-scale fuzz is the CI soak; this is the per-PR slice."""
    bm = BlockManagerImpl(num_gpu_blocks=32, block_size=8, max_model_len=64)
    live: dict[int, Sequence] = {}
    next_id = 0
    for raw in ops:
        op = raw % 5
        if op == 0 and len(live) < 8:
            seq = _seq(next_id, (raw % 17) + 1)
            next_id += 1
            if bm.can_allocate(seq) is AllocStatus.OK:
                bm.allocate(seq)
                live[seq.seq_id] = seq
        elif op == 1 and live:
            seq = next(iter(live.values()))
            try:
                bm.append_slot(seq)
                seq.output_token_ids.append(0)
            except RuntimeError:
                pass
        elif op == 2 and len(live) >= 1 and len(live) < 8:
            parent = next(iter(live.values()))
            child = _seq(next_id, 1)
            next_id += 1
            bm.fork(parent, child)
            live[child.seq_id] = child
        elif op == 3 and live:
            seq_id = next(iter(live))
            bm.free(live.pop(seq_id))
        elif op == 4 and live:
            seq = next(iter(live.values()))
            try:
                bm.append_slot(seq)
                seq.output_token_ids.append(1)
            except RuntimeError:
                pass
    for seq in list(live.values()):
        bm.free(seq)
    assert bm.get_num_free_blocks() == 32


def test_10k_random_ops() -> None:
    import random

    rng = random.Random(0)
    bm = BlockManagerImpl(num_gpu_blocks=64, block_size=8, max_model_len=128)
    live: dict[int, Sequence] = {}
    next_id = 0
    for _ in range(10_000):
        op = rng.randint(0, 4)
        if op == 0 and len(live) < 12:
            seq = _seq(next_id, rng.randint(1, 24))
            next_id += 1
            if bm.can_allocate(seq) is AllocStatus.OK:
                bm.allocate(seq)
                live[seq.seq_id] = seq
        elif op == 1 and live:
            seq = live[rng.choice(list(live))]
            try:
                bm.append_slot(seq)
                seq.output_token_ids.append(0)
            except RuntimeError:
                pass
        elif op == 2 and live and len(live) < 12:
            parent = live[rng.choice(list(live))]
            child = _seq(next_id, 1)
            next_id += 1
            bm.fork(parent, child)
            live[child.seq_id] = child
        elif op == 3 and live:
            seq_id = rng.choice(list(live))
            bm.free(live.pop(seq_id))
    for seq in list(live.values()):
        bm.free(seq)
    assert bm.get_num_free_blocks() == 64


def test_cow_on_shared_tail() -> None:
    bm = BlockManagerImpl(num_gpu_blocks=16, block_size=4, max_model_len=32)
    parent = _seq(1, 3)
    assert bm.can_allocate(parent) is AllocStatus.OK
    bm.allocate(parent)
    for _ in range(3):
        bm.append_slot(parent)
    child = _seq(2, 1)
    bm.fork(parent, child)
    assert parent.block_table == child.block_table
    old_tail = parent.block_table[-1]
    assert bm.blocks[old_tail].ref_count == 2
    new_id, _off = bm.append_slot(parent)
    assert new_id != old_tail
    assert bm.blocks[old_tail].ref_count == 1
    assert child.block_table[-1] == old_tail
    bm.free(parent)
    bm.free(child)
    assert bm.get_num_free_blocks() == 16


def test_waste_under_five_percent_mixed_lengths() -> None:
    bm = BlockManagerImpl(num_gpu_blocks=256, block_size=16, max_model_len=2048)
    # Chat-like lengths: last-block waste is ≤ 15 tokens; that is < 5% at these sizes.
    lengths = [200, 256, 300, 384, 512, 180, 220, 448]
    live: list[Sequence] = []
    for i, n in enumerate(lengths):
        seq = _seq(i, n)
        bm.allocate(seq)
        for _ in range(n):
            bm.append_slot(seq)
        live.append(seq)
    util = bm.kv_utilization(live)
    waste = 1.0 - util
    assert waste < 0.05, f"waste={waste:.3f} util={util:.3f}"
    naive_slots = len(lengths) * 2048
    naive_waste = 1.0 - (sum(lengths) / naive_slots)
    assert naive_waste > 0.60
    for seq in live:
        bm.free(seq)
