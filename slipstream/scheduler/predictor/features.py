"""Length-predictor feature sets f0–f3 (MASTERPLAN §5.2)."""

from __future__ import annotations

from enum import Enum
from math import log

from slipstream.core.types import Sequence


class FeatureSet(Enum):
    F0 = "f0"
    F1 = "f1"
    F2 = "f2"
    F3 = "f3"


_CODEISH = frozenset({ord(c) for c in "{}[];()=<>/\\#" if ord(c) < 128})


def unigram_entropy(token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    counts: dict[int, int] = {}
    for t in token_ids:
        counts[t] = counts.get(t, 0) + 1
    n = float(len(token_ids))
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * log(p + 1e-12)
    return ent


def code_like(token_ids: list[int], extra: dict[str, object] | None = None) -> float:
    """Task-type heuristic.

    BPE ids are not ASCII code points, so the raw-id fallback is weak.
    Workloads may set `SamplingParams.extra['code_like']` in [0, 1].
    """
    if extra is not None and "code_like" in extra:
        return float(extra["code_like"])  # type: ignore[arg-type]
    if not token_ids:
        return 0.0
    hits = sum(1 for t in token_ids if t < 128 and t in _CODEISH)
    return hits / len(token_ids)


def extract_features(seq: Sequence, feature_set: FeatureSet) -> dict[str, float]:
    prompt = seq.prompt_token_ids
    feats = {
        "prompt_len": float(len(prompt)),
        "generated": float(seq.num_output_tokens),
        "entropy": unigram_entropy(prompt),
        "code_like": code_like(prompt, seq.sampling_params.extra),
    }
    if feature_set is FeatureSet.F0:
        return {"generated": feats["generated"]}
    return feats
