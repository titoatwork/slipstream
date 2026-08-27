"""Online remaining-length predictor. Must cost < 0.5% of step time."""

from __future__ import annotations

from math import log1p

from slipstream.core.types import Sequence
from slipstream.scheduler.predictor.features import FeatureSet, extract_features


class LengthPredictor:
    """CPU-only. f0 = EMA mean; f1/f2 add prompt features and generated tokens."""

    def __init__(self, feature_set: FeatureSet = FeatureSet.F2) -> None:
        self.feature_set = feature_set
        self.mean_output = 32.0
        self._n = 0
        # prompt-length buckets: <32, <96, <256, else
        # Short chat prior is smaller than long/code; observe() still adapts.
        self._bucket_mean = [16.0, 48.0, 64.0, 96.0]
        self._bucket_n = [0, 0, 0, 0]

    def predict_remaining(self, seq: Sequence) -> int:
        generated = seq.num_output_tokens
        if self.feature_set is FeatureSet.F0:
            total = self.mean_output
        else:
            feats = extract_features(seq, self.feature_set)
            bucket = self._bucket(int(feats["prompt_len"]))
            base = self._bucket_mean[bucket]
            # Short prompts and high-entropy "chat" → shorter; code-like → longer.
            # Prompt length is the only tokenizer-agnostic size signal
            # (`code_like` on raw ids is weak under BPE).
            total = (
                0.55 * base
                + 0.20 * self.mean_output
                + 0.45 * feats["prompt_len"]
                + 4.0 * log1p(feats["prompt_len"])
                + 3.0 * feats["entropy"]
                + 40.0 * feats["code_like"]
            )
        remaining = total - generated
        if remaining < 1.0:
            remaining = 1.0
        if self.feature_set in {FeatureSet.F2, FeatureSet.F3} and generated > 0.85 * max(
            total, 1.0
        ):
            remaining = max(1.0, remaining * 0.5)
        return int(remaining)

    def observe(self, seq: Sequence, actual_output_len: int) -> None:
        actual = max(1, int(actual_output_len))
        self._n += 1
        alpha = 0.15 if self._n > 8 else 0.4
        self.mean_output = (1.0 - alpha) * self.mean_output + alpha * actual
        b = self._bucket(seq.num_prompt_tokens)
        self._bucket_n[b] += 1
        ba = 0.2 if self._bucket_n[b] > 4 else 0.5
        self._bucket_mean[b] = (1.0 - ba) * self._bucket_mean[b] + ba * actual

    @staticmethod
    def _bucket(prompt_len: int) -> int:
        if prompt_len < 32:
            return 0
        if prompt_len < 96:
            return 1
        if prompt_len < 256:
            return 2
        return 3
