"""Replay helpers for RECOMPUTE preemption.

Frozen `Sequence.is_prefill` is `num_computed < num_prompt`. After a
recompute the prompt may already be done while earlier *output* tokens
are missing from KV. Incremental decode's invariant is: the last sampled
token is the next query and is not yet in the cache.
"""

from __future__ import annotations

from slipstream.core.types import Sequence


def kv_written_target(seq: Sequence) -> int:
    """How many logical tokens must be in KV before the next decode step."""
    n_out = seq.num_output_tokens
    if n_out == 0:
        return seq.num_prompt_tokens
    return seq.num_prompt_tokens + n_out - 1


def needs_replay(seq: Sequence) -> bool:
    return seq.num_computed_tokens < kv_written_target(seq)


def kv_uncomputed(seq: Sequence) -> int:
    return max(0, kv_written_target(seq) - seq.num_computed_tokens)


def replay_token_ids(seq: Sequence) -> list[int]:
    """Tokens to write into KV to catch up: prompt + outputs except the last."""
    if not seq.output_token_ids:
        return list(seq.prompt_token_ids)
    return list(seq.prompt_token_ids) + seq.output_token_ids[:-1]
