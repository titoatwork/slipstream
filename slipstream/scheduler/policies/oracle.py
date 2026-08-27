"""Clairvoyant upper bound. Uses `Sequence.oracle_output_len`."""

from __future__ import annotations

from slipstream.core.types import Sequence
from slipstream.scheduler.policies.horizon import HorizonPolicy
from slipstream.scheduler.predictor.length_model import LengthPredictor


class _OraclePredictor(LengthPredictor):
    def predict_remaining(self, seq: Sequence) -> int:
        if seq.oracle_output_len is None:
            return super().predict_remaining(seq)
        return max(1, int(seq.oracle_output_len) - seq.num_output_tokens)

    def observe(self, seq: Sequence, actual_output_len: int) -> None:
        del seq, actual_output_len


class OraclePolicy(HorizonPolicy):
    """Perfect remaining-length knowledge. Bounds Horizon's achievable gain."""

    name = "oracle"

    def __init__(
        self,
        predictor: LengthPredictor | None = None,
        *,
        safety_factor: float = 0.95,
        starvation_guard_ms: float = 5_000.0,
        feature_set: object = None,
    ) -> None:
        del predictor, feature_set
        super().__init__(
            predictor=_OraclePredictor(),
            safety_factor=safety_factor,
            starvation_guard_ms=starvation_guard_ms,
        )
