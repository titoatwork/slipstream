"""Horizon length predictor. Implementation lands in Phase 4 (A8)."""

from slipstream.scheduler.predictor.features import FeatureSet, extract_features
from slipstream.scheduler.predictor.length_model import LengthPredictor

__all__ = ["FeatureSet", "LengthPredictor", "extract_features"]
