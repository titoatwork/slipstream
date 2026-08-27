"""FCFS reference policy."""

from __future__ import annotations

from slipstream.core.types import EngineState, PreemptionMode, Sequence


class FCFSPolicy:
    """First-come-first-served. The ablation baseline Horizon must beat."""

    name = "fcfs"

    def order_waiting(self, waiting: list[Sequence], state: EngineState) -> list[Sequence]:
        return sorted(waiting, key=lambda seq: (seq.arrival_ts, seq.seq_id))

    def should_admit(self, seq: Sequence, state: EngineState) -> bool:
        block_size = max(state.block_size, 1)
        tokens = max(seq.num_prompt_tokens, 1)
        needed = (tokens + block_size - 1) // block_size
        return needed <= state.num_free_blocks

    def select_preemption_victim(self, running: list[Sequence], state: EngineState) -> Sequence:
        if not running:
            raise ValueError("no running sequence to preempt")
        return max(running, key=lambda seq: (seq.arrival_ts, seq.seq_id))

    def preemption_mode(self, victim: Sequence, state: EngineState) -> PreemptionMode:
        # Long contexts are cheaper to swap; short ones recompute.
        if len(victim.block_table) >= 4:
            return PreemptionMode.SWAP
        return PreemptionMode.RECOMPUTE
