"""Iteration-level continuous batching (S4). FCFS mechanism; policy is swappable."""

from __future__ import annotations

from collections import deque

from slipstream.core.config import SchedulerConfig
from slipstream.core.debug import assert_debug
from slipstream.core.types import (
    AllocStatus,
    BlockManager,
    EngineState,
    PreemptionMode,
    SchedulerOutput,
    SchedulingPolicy,
    Sequence,
    SequenceStatus,
)
from slipstream.scheduler.replay import kv_uncomputed, needs_replay


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        block_manager: BlockManager,
        policy: SchedulingPolicy,
        block_size: int = 16,
        kv_bytes_per_block: int = 0,
    ) -> None:
        self.config = config
        self.block_manager = block_manager
        self.policy = policy
        self.block_size = block_size
        self.kv_bytes_per_block = kv_bytes_per_block
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.swapped: list[Sequence] = []
        self._seq_ids = 0
        self.last_take: dict[int, int] = {}
        self.preemptions = 0

    def _discard(self, bucket: list[Sequence] | deque[Sequence], seq: Sequence) -> None:
        if isinstance(bucket, deque):
            kept = [s for s in bucket if s is not seq]
            bucket.clear()
            bucket.extend(kept)
            return
        bucket[:] = [s for s in bucket if s is not seq]

    def _reclaim_stranded_allocs(self) -> None:
        """Waiting sequences must not hold GPU pages. If they do, they are running."""
        for seq in list(self.waiting):
            if not seq.block_table or seq.is_finished:
                continue
            self._discard(self.waiting, seq)
            if seq not in self.running:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)

    def add_seq(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def next_seq_id(self) -> int:
        self._seq_ids += 1
        return self._seq_ids

    def snapshot(self, tokens_scheduled: int = 0, now: float = 0.0) -> EngineState:
        total = getattr(self.block_manager, "num_gpu_blocks", 0)
        free = self.block_manager.get_num_free_blocks()
        return EngineState(
            num_free_blocks=free,
            num_total_blocks=int(total),
            token_budget=self.config.max_num_batched_tokens,
            tokens_scheduled=tokens_scheduled,
            running=tuple(self.running),
            waiting=tuple(self.waiting),
            swapped=tuple(self.swapped),
            kv_bytes_per_block=self.kv_bytes_per_block,
            block_size=self.block_size,
            now=now,
            gpu_cache_usage=0.0 if total == 0 else 1.0 - (free / total),
        )

    def schedule(self) -> SchedulerOutput:
        import time

        now = time.time()
        self._reclaim_stranded_allocs()
        refresh = getattr(self.policy, "refresh", None)
        if callable(refresh):
            refresh(list(self.running) + list(self.waiting), self.snapshot(now=now))

        budget = self.config.max_num_batched_tokens
        chunk = self.config.prefill_chunk_size if self.config.enable_chunked_prefill else 10**9
        scheduled: list[Sequence] = []
        is_prefill_chunk: dict[int, bool] = {}
        tokens = 0
        self.last_take = {}

        # Decodes first (1 token each) — stall-free among running decodes.
        for seq in list(self.running):
            if seq.is_finished or needs_replay(seq):
                continue
            if tokens + 1 > budget:
                break
            scheduled.append(seq)
            is_prefill_chunk[seq.seq_id] = False
            self.last_take[seq.seq_id] = 1
            tokens += 1

        # Then running prefills / recomputes, capped per chunk.
        for seq in list(self.running):
            if seq.is_finished or not needs_replay(seq):
                continue
            remaining = kv_uncomputed(seq)
            take = min(remaining, budget - tokens, chunk)
            if take <= 0:
                continue
            scheduled.append(seq)
            is_prefill_chunk[seq.seq_id] = take < seq.num_uncomputed_tokens
            self.last_take[seq.seq_id] = take
            tokens += take

        # Admit waiting (prefix already applied by the engine before add_seq).
        state = self.snapshot(tokens_scheduled=tokens, now=now)
        ordered = self.policy.order_waiting(list(self.waiting), state)
        for seq in ordered:
            if tokens >= budget:
                break
            if len(self.running) >= self.config.max_num_seqs:
                break
            state = self.snapshot(tokens_scheduled=tokens, now=now)
            if not self.policy.should_admit(seq, state):
                continue
            if not self._try_admit(seq):
                continue
            remaining = kv_uncomputed(seq)
            take = min(remaining, budget - tokens, chunk)
            if take <= 0:
                continue
            scheduled.append(seq)
            is_prefill_chunk[seq.seq_id] = take < remaining
            self.last_take[seq.seq_id] = take
            tokens += take

        self._maybe_swap_in(budget - tokens)

        assert_debug(tokens <= budget, "I4.1 token budget exceeded")
        return SchedulerOutput(
            scheduled_seqs=scheduled,
            num_batched_tokens=tokens,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            is_prefill_chunk=is_prefill_chunk,
        )

    def finish(self, seq: Sequence) -> None:
        predictor = getattr(self.policy, "predictor", None)
        observe = getattr(predictor, "observe", None)
        if callable(observe):
            observe(seq, seq.num_output_tokens)
        self._discard(self.running, seq)
        self._discard(self.waiting, seq)
        self.block_manager.free(seq)

    def preempt_recompute(self, seq: Sequence) -> None:
        self._discard(self.running, seq)
        self._discard(self.waiting, seq)
        self.block_manager.free(seq)
        seq.num_computed_tokens = 0
        seq.num_cached_tokens = 0
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
        self.preemptions += 1

    def abort(self, seq_id: int) -> None:
        for seq in list(self.running) + list(self.waiting):
            if seq.seq_id == seq_id:
                seq.status = SequenceStatus.FINISHED_ABORTED
                self.finish(seq)
                return

    def _try_admit(self, seq: Sequence) -> bool:
        import time

        attempts = max(1, len(self.running) + 1)
        for _ in range(attempts):
            status = self.block_manager.can_allocate(seq)
            if status is AllocStatus.OK:
                self.block_manager.allocate(seq)
                self._discard(self.waiting, seq)
                seq.status = SequenceStatus.RUNNING
                if seq not in self.running:
                    self.running.append(seq)
                return True
            if status is AllocStatus.NEVER:
                seq.status = SequenceStatus.FINISHED_ABORTED
                self._discard(self.waiting, seq)
                return False
            if not self.running:
                return False
            state = self.snapshot(now=time.time())
            victim = self.policy.select_preemption_victim(self.running, state)
            mode = self.policy.preemption_mode(victim, state)
            if mode is PreemptionMode.SWAP:
                try:
                    self.preempt_swap(victim)
                except (NotImplementedError, RuntimeError):
                    self.preempt_recompute(victim)
            else:
                self.preempt_recompute(victim)
        return False

    def preempt_swap(self, seq: Sequence) -> None:
        self._discard(self.running, seq)
        self._discard(self.waiting, seq)
        self.block_manager.swap_out(seq)
        seq.status = SequenceStatus.SWAPPED
        self.swapped.append(seq)
        self.preemptions += 1

    def ensure_slot(
        self, seq: Sequence, protected: set[int] | None = None
    ) -> tuple[int, int] | None:
        """Reserve the next token slot. On OOM, preempt a victim and retry.

        `protected` seq ids were already slotted this step and must not be
        freed (their pages are in the in-flight batch).
        Returns None if `seq` itself had to be preempted (cannot grow).
        """
        hold = set(protected or ())
        hold.add(seq.seq_id)
        attempts = max(1, len(self.running) + len(self.swapped) + 1)
        for _ in range(attempts):
            try:
                pair = self.block_manager.append_slot(seq)
            except RuntimeError as exc:
                if "out of KV blocks" not in str(exc):
                    raise
                if not self._preempt_for_growth(seq, hold):
                    return None
                continue
            if pair is None:
                if not self._preempt_for_growth(seq, hold):
                    return None
                continue
            return pair
        return None

    def _preempt_for_growth(self, keep: Sequence, protected: set[int]) -> bool:
        others = [s for s in self.running if s.seq_id != keep.seq_id and s.seq_id not in protected]
        if not others:
            # Keep KV and skip this token; self-preempt storms under a tight cap.
            return False
        state = self.snapshot()
        victim = self.policy.select_preemption_victim(others, state)
        mode = self.policy.preemption_mode(victim, state)
        if mode is PreemptionMode.SWAP:
            try:
                self.preempt_swap(victim)
            except (NotImplementedError, RuntimeError):
                self.preempt_recompute(victim)
        else:
            self.preempt_recompute(victim)
        return True

    def _maybe_swap_in(self, spare_tokens: int) -> None:
        del spare_tokens
        if not self.swapped:
            return
        if len(self.running) >= self.config.max_num_seqs:
            return
        victim = self.swapped[0]
        if self.block_manager.can_allocate(victim) is not AllocStatus.OK:
            return
        try:
            self.block_manager.swap_in(victim)
        except NotImplementedError:
            return
        self.swapped.pop(0)
        victim.status = SequenceStatus.RUNNING
        self.running.append(victim)
