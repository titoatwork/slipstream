"""Horizon / oracle policy unit tests (CPU)."""

from __future__ import annotations

from slipstream.core.config import SchedulerConfig
from slipstream.core.types import EngineState, PreemptionMode, Sequence
from slipstream.memory.block_manager import BlockManagerImpl
from slipstream.scheduler.policies import get_policy
from slipstream.scheduler.policies.horizon import HorizonPolicy
from slipstream.scheduler.policies.oracle import OraclePolicy
from slipstream.scheduler.predictor.features import FeatureSet, extract_features
from slipstream.scheduler.predictor.length_model import LengthPredictor
from slipstream.scheduler.scheduler import Scheduler


def _seq(seq_id: int, prompt: int, out: int = 0, arrival: float = 0.0) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(max(prompt, 1))),
        output_token_ids=[0] * out,
        arrival_ts=arrival,
    )


def _state(
    running: list[Sequence],
    waiting: list[Sequence],
    free: int = 20,
    total: int = 32,
    now: float = 10.0,
) -> EngineState:
    return EngineState(
        num_free_blocks=free,
        num_total_blocks=total,
        token_budget=128,
        tokens_scheduled=0,
        running=tuple(running),
        waiting=tuple(waiting),
        swapped=(),
        kv_bytes_per_block=1,
        block_size=16,
        now=now,
        gpu_cache_usage=1.0 - free / total,
    )


def test_predictor_f0_learns_mean() -> None:
    p = LengthPredictor(FeatureSet.F0)
    seq = _seq(1, 8, 0)
    for n in (10, 10, 10, 10, 10):
        p.observe(seq, n)
    rem = p.predict_remaining(_seq(2, 8, 0))
    assert 5 <= rem <= 20


def test_f2_remaining_decreases_as_we_generate() -> None:
    p = LengthPredictor(FeatureSet.F2)
    early = p.predict_remaining(_seq(1, 40, 0))
    late = p.predict_remaining(_seq(1, 40, early))
    assert late <= early


def test_f2_longer_prompt_predicts_more_remaining() -> None:
    p = LengthPredictor(FeatureSet.F2)
    short = p.predict_remaining(_seq(1, 16, 0))
    long = p.predict_remaining(_seq(2, 80, 0))
    assert long > short


def test_horizon_srpt_orders_short_first() -> None:
    h = HorizonPolicy()
    short = _seq(1, 8, 0, arrival=1.0)
    long = _seq(2, 8, 0, arrival=0.0)
    short.predicted_remaining = 4
    long.predicted_remaining = 80
    # now-arrival must stay under the 5s guard or this degenerates to FIFO.
    state = _state([], [long, short], now=1.5)
    ordered = h.order_waiting([long, short], state)
    assert [s.seq_id for s in ordered] == [1, 2]


def test_horizon_starvation_guard_promotes_old() -> None:
    h = HorizonPolicy(starvation_guard_ms=100.0)
    old = _seq(1, 8, 0, arrival=0.0)
    young = _seq(2, 8, 0, arrival=9.9)
    old.predicted_remaining = 90
    young.predicted_remaining = 2
    state = _state([], [old, young], now=10.0)
    ordered = h.order_waiting([young, old], state)
    assert ordered[0].seq_id == 1


def test_horizon_does_not_starve_fresh_relative_arrivals() -> None:
    """Wall-clock `now` with recent arrivals must stay SRPT, not FIFO."""
    h = HorizonPolicy(starvation_guard_ms=5_000.0)
    short = _seq(1, 8, 0, arrival=1_700_000_000.05)
    long = _seq(2, 8, 0, arrival=1_700_000_000.00)
    short.predicted_remaining = 4
    long.predicted_remaining = 80
    state = _state([], [long, short], now=1_700_000_000.10)
    ordered = h.order_waiting([long, short], state)
    assert [s.seq_id for s in ordered] == [1, 2]


def test_horizon_admits_idle_request_that_fits() -> None:
    h = HorizonPolicy(safety_factor=0.5)
    seq = _seq(1, 32)
    seq.predicted_remaining = 16
    # Peak ~3 blocks, 50% of 8 is 4; idle path uses the raw pool.
    state = _state([], [seq], free=8, total=8)
    assert h.should_admit(seq, state) is True


def test_horizon_refuses_over_capacity() -> None:
    h = HorizonPolicy(safety_factor=0.5)
    seq = _seq(1, 200)
    seq.predicted_remaining = 200
    state = _state([], [seq], free=2, total=8)
    assert h.should_admit(seq, state) is False


def test_horizon_victim_is_longest_costly() -> None:
    h = HorizonPolicy()
    cheap = _seq(1, 8)
    costly = _seq(2, 8)
    cheap.predicted_remaining = 4
    costly.predicted_remaining = 80
    cheap.block_table = [0]
    costly.block_table = [1, 2, 3]
    cheap.slo_tpot_ms = 200.0
    costly.slo_tpot_ms = 200.0
    victim = h.select_preemption_victim([cheap, costly], _state([cheap, costly], []))
    assert victim.seq_id == 2


def test_preemption_mode_swap_threshold() -> None:
    h = HorizonPolicy()
    short = _seq(1, 8)
    short.block_table = [0, 1]
    long = _seq(2, 8)
    long.block_table = [0, 1, 2, 3]
    state = _state([short, long], [])
    assert h.preemption_mode(short, state) is PreemptionMode.RECOMPUTE
    assert h.preemption_mode(long, state) is PreemptionMode.SWAP


def test_oracle_uses_true_remaining() -> None:
    o = OraclePolicy()
    seq = _seq(1, 10, 5)
    seq.oracle_output_len = 20
    assert o.predictor.predict_remaining(seq) == 15


def test_get_policy_instantiates_horizon_and_oracle() -> None:
    h = get_policy("horizon", safety_factor=0.8, starvation_guard_ms=1_000.0)
    o = get_policy("oracle", safety_factor=0.8, starvation_guard_ms=1_000.0)
    assert h.name == "horizon"  # type: ignore[attr-defined]
    assert o.name == "oracle"  # type: ignore[attr-defined]
    assert isinstance(h, HorizonPolicy)
    assert isinstance(o, OraclePolicy)
    assert h.safety_factor == 0.8
    assert o.starvation_guard_ms == 1_000.0


def test_extract_features_f1_keys() -> None:
    feats = extract_features(_seq(1, 12, 3), FeatureSet.F1)
    assert "prompt_len" in feats and "entropy" in feats


def test_schedule_forwards_now_to_order_waiting() -> None:
    """Regression: snapshot default now=0 made every wall-clock arrival look fresh."""

    class _Spy(HorizonPolicy):
        def __init__(self) -> None:
            super().__init__(starvation_guard_ms=100.0)
            self.seen_now: float | None = None

        def order_waiting(self, waiting: list[Sequence], state: EngineState) -> list[Sequence]:
            self.seen_now = state.now
            return super().order_waiting(waiting, state)

    bm = BlockManagerImpl(num_gpu_blocks=8, block_size=16, max_model_len=256)
    spy = _Spy()
    sched = Scheduler(SchedulerConfig(policy="horizon"), bm, spy)
    seq = _seq(1, 8, arrival=1_700_000_000.0)
    sched.add_seq(seq)
    sched.schedule()
    assert spy.seen_now is not None
    assert spy.seen_now > 1_000_000_000.0


def test_ensure_slot_preempts_on_oom() -> None:
    bm = BlockManagerImpl(num_gpu_blocks=2, block_size=16, max_model_len=256)
    sched = Scheduler(SchedulerConfig(policy="horizon", max_num_seqs=4), bm, HorizonPolicy())
    keeper = _seq(1, 8)
    victim = _seq(2, 8)
    victim.predicted_remaining = 80
    keeper.predicted_remaining = 4
    bm.allocate(keeper)
    bm.allocate(victim)
    sched.running.extend([keeper, victim])
    for _ in range(16):
        bm.append_slot(keeper)
    assert bm.get_num_free_blocks() == 0
    pair = sched.ensure_slot(keeper)
    assert pair is not None
    assert victim not in sched.running
    assert sched.preemptions >= 1
