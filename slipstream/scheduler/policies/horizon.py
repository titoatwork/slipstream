"""Horizon — SLO-aware, length-predictive policy."""

from __future__ import annotations

from slipstream.core.types import EngineState, PreemptionMode, Sequence
from slipstream.scheduler.predictor.features import FeatureSet
from slipstream.scheduler.predictor.length_model import LengthPredictor


class HorizonPolicy:
    """Admit / order / preempt using predicted remaining length.

    Mechanism (queues, token budget) stays in Scheduler. This class is only
    policy. See MASTERPLAN §5.
    """

    name = "horizon"

    def __init__(
        self,
        predictor: LengthPredictor | None = None,
        *,
        safety_factor: float = 0.95,
        starvation_guard_ms: float = 5_000.0,
        feature_set: FeatureSet = FeatureSet.F2,
    ) -> None:
        self.predictor = predictor if predictor is not None else LengthPredictor(feature_set)
        self.safety_factor = safety_factor
        self.starvation_guard_ms = starvation_guard_ms

    def refresh(self, seqs: list[Sequence], state: EngineState) -> None:
        del state
        for seq in seqs:
            seq.predicted_remaining = self.predictor.predict_remaining(seq)

    def order_waiting(self, waiting: list[Sequence], state: EngineState) -> list[Sequence]:
        now = state.now

        def key(seq: Sequence) -> tuple[int, float, float]:
            wait_ms = max(0.0, (now - seq.arrival_ts) * 1000.0)
            if wait_ms >= self.starvation_guard_ms:
                return (0, seq.arrival_ts, seq.seq_id)
            rem = float(seq.predicted_remaining if seq.predicted_remaining is not None else 10**6)
            return (1, rem, seq.arrival_ts)

        return sorted(waiting, key=key)

    def should_admit(self, seq: Sequence, state: EngineState) -> bool:
        block_size = max(state.block_size, 1)
        used = max(0, state.num_total_blocks - state.num_free_blocks)
        extra = 0
        for running in state.running:
            rem = running.predicted_remaining or 1
            peak = (running.num_tokens + rem + block_size - 1) // block_size
            have = max(1, len(running.block_table))
            extra += max(0, peak - have)
        remaining = seq.predicted_remaining
        if remaining is None:
            remaining = self.predictor.predict_remaining(seq)
            seq.predicted_remaining = remaining
        peak = (seq.num_prompt_tokens + remaining + block_size - 1) // block_size
        # Prefix-attached pages are already in `used`; only count remaining growth.
        mine = max(0, peak - len(seq.block_table))
        if not state.running:
            # Safety factor must not deadlock an idle engine.
            return mine <= state.num_total_blocks
        hwm = used + extra + mine
        cap = int(state.num_total_blocks * self.safety_factor)
        return hwm <= max(cap, 1)

    def select_preemption_victim(self, running: list[Sequence], state: EngineState) -> Sequence:
        del state
        if not running:
            raise ValueError("no running sequence to preempt")

        def score(seq: Sequence) -> float:
            rem = float(seq.predicted_remaining if seq.predicted_remaining is not None else 1)
            growth = max(1, len(seq.block_table))
            slack = max(seq.slo_tpot_ms, 1.0)
            return (rem * growth) / slack

        return max(running, key=score)

    def preemption_mode(self, victim: Sequence, state: EngineState) -> PreemptionMode:
        del state
        if len(victim.block_table) >= 4:
            return PreemptionMode.SWAP
        return PreemptionMode.RECOMPUTE
