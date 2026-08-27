"""GPU step loop. Optional process split: slipstream.engine.isolated."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slipstream.core.config import EngineConfig
from slipstream.core.types import SchedulerOutput, Sequence, SequenceStatus
from slipstream.memory.contiguous_cache import NaiveKVCache

if TYPE_CHECKING:
    from slipstream.engine.llm_engine import LLMEngine


class EngineCore:
    """Owns the scheduler + model runner busy loop.

    Tokenization, detokenization, and HTTP must never run in this process.
    Phase 1: single-sequence prefill-or-decode steps. No continuous batching.
    """

    def __init__(self, config: EngineConfig, engine: LLMEngine | None = None) -> None:
        self.config = config
        self._engine = engine
        self._seq: Sequence | None = None
        self._cache: NaiveKVCache | None = None
        self._max_output: int | None = None

    def add_request(self, seq: Sequence) -> None:
        if not seq.prompt_token_ids:
            raise ValueError("Sequence needs prompt_token_ids")
        self._seq = seq
        self._cache = None
        seq.status = SequenceStatus.WAITING

    def step(self) -> SchedulerOutput:
        seq = self._seq
        if seq is None or seq.is_finished:
            return SchedulerOutput(
                scheduled_seqs=[],
                num_batched_tokens=0,
                blocks_to_swap_in={},
                blocks_to_swap_out={},
                blocks_to_copy={},
                is_prefill_chunk={},
            )

        engine = self._ensure_engine()
        if self._cache is None:
            max_model_len = engine.config.model.max_model_len
            max_new = max(
                0,
                min(seq.sampling_params.max_tokens, max_model_len - seq.num_prompt_tokens),
            )
            if max_new < 1:
                seq.status = SequenceStatus.FINISHED_LENGTH
                return SchedulerOutput(
                    scheduled_seqs=[],
                    num_batched_tokens=0,
                    blocks_to_swap_in={},
                    blocks_to_swap_out={},
                    blocks_to_copy={},
                    is_prefill_chunk={},
                )
            self._max_output = max_new
            self._cache = engine.make_cache(seq.num_prompt_tokens + max_new)
            seq.status = SequenceStatus.RUNNING

        if seq.is_prefill:
            logits = engine.prefill(seq.prompt_token_ids, self._cache)
            seq.num_computed_tokens = seq.num_prompt_tokens
            n_tokens = seq.num_prompt_tokens
        else:
            logits = engine.decode(seq.output_token_ids[-1], self._cache)
            seq.num_computed_tokens = seq.num_tokens
            n_tokens = 1

        engine.append_sampled(seq, logits, max_output=self._max_output)
        return SchedulerOutput(
            scheduled_seqs=[seq],
            num_batched_tokens=n_tokens,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            is_prefill_chunk={seq.seq_id: False},
        )

    def abort(self, seq_id: int) -> None:
        if self._seq is not None and self._seq.seq_id == seq_id:
            self._seq.status = SequenceStatus.FINISHED_ABORTED
            self._cache = None

    def _ensure_engine(self) -> LLMEngine:
        if self._engine is None:
            from slipstream.engine.llm_engine import LLMEngine

            self._engine = LLMEngine(self.config)
        return self._engine
