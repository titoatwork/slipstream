"""Frozen interface contracts (§8.3).

Changing anything in this module requires:
  1. An amendment entry in MASTERPLAN.md §21
  2. Notification to every dependent agent
  3. A single integration commit

Agents implement against these types, never against each other's code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Protocol, runtime_checkable

from slipstream.core.sampling_params import SamplingParams

# ---------------------------------------------------------------------------
# KV cache layout (frozen)
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_SIZE: int = 16

# Physical tensor layout, documented so kernels and the allocator agree
# before either is implemented:
#   kv_cache: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
#                  K/V ^
KV_CACHE_LAYOUT: str = "num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SequenceStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    FINISHED_STOPPED = 3  # EOS or stop string
    FINISHED_LENGTH = 4  # hit max_tokens
    FINISHED_ABORTED = 5  # client cancel / engine abort


FINISHED_STATUSES: frozenset[SequenceStatus] = frozenset(
    {
        SequenceStatus.FINISHED_STOPPED,
        SequenceStatus.FINISHED_LENGTH,
        SequenceStatus.FINISHED_ABORTED,
    }
)


class AllocStatus(Enum):
    """Result of `BlockManager.can_allocate`."""

    OK = "ok"  # allocation succeeds now
    LATER = "later"  # would succeed after preemption / free
    NEVER = "never"  # request exceeds total GPU block capacity


class PreemptionMode(Enum):
    SWAP = "swap"  # copy blocks to CPU swap space
    RECOMPUTE = "recompute"  # drop blocks; replay prefill on resume


# ---------------------------------------------------------------------------
# Request (API → engine)
# ---------------------------------------------------------------------------


@dataclass
class Request:
    """Arrival-time request. Tokenization happens in an isolated process."""

    request_id: str
    prompt: str | None
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams
    arrival_ts: float
    slo_ttft_ms: float = 2000.0
    slo_tpot_ms: float = 100.0

    def __post_init__(self) -> None:
        if self.prompt is None and self.prompt_token_ids is None:
            raise ValueError("Request needs prompt or prompt_token_ids")


# ---------------------------------------------------------------------------
# Sequence (engine unit of work)
# ---------------------------------------------------------------------------


@dataclass
class Sequence:
    """One generation trajectory. The scheduler's unit of work.

    `block_table` maps logical block index → physical block id.
    `num_cached_tokens` is the prefix-cache hit length.
    `num_computed_tokens` is chunked-prefill progress (prompt + output).
    """

    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_ts: float = 0.0
    first_token_ts: float | None = None
    num_cached_tokens: int = 0
    num_computed_tokens: int = 0
    request_id: str = ""
    # --- research fields (Horizon) ---
    predicted_remaining: int | None = None
    slo_ttft_ms: float = 2000.0
    slo_tpot_ms: float = 100.0
    # oracle-only: filled by the workload generator, never by online policies
    oracle_output_len: int | None = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def num_uncomputed_tokens(self) -> int:
        return max(0, self.num_tokens - self.num_computed_tokens)

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0
    block_hash: int | None = None  # content hash for prefix cache
    num_tokens: int = 0  # filled slots in this block


@dataclass(frozen=True)
class EngineState:
    """Read-only snapshot handed to `SchedulingPolicy`.

    Policies must not mutate engine internals through this object. Sequence
    objects are shared references; mutating them from a policy is a contract
    violation.
    """

    num_free_blocks: int
    num_total_blocks: int
    token_budget: int
    tokens_scheduled: int
    running: tuple[Sequence, ...]
    waiting: tuple[Sequence, ...]
    swapped: tuple[Sequence, ...]
    kv_bytes_per_block: int
    block_size: int
    now: float
    gpu_cache_usage: float  # occupied / total, in [0, 1]


@dataclass
class SchedulerOutput:
    scheduled_seqs: list[Sequence]
    num_batched_tokens: int
    blocks_to_swap_in: dict[int, int]  # gpu_block -> cpu_block
    blocks_to_swap_out: dict[int, int]
    blocks_to_copy: dict[int, list[int]]  # COW: src -> [dst, ...]
    is_prefill_chunk: dict[int, bool]  # seq_id -> chunked this step


# ---------------------------------------------------------------------------
# Protocols — the parallelism seams
# ---------------------------------------------------------------------------


@runtime_checkable
class BlockManager(Protocol):
    """Owns ALL KV memory. Single source of truth for allocation.

    Implemented by `slipstream.memory.block_manager.BlockManagerImpl`.
    This Protocol is the contract; the impl may not exist yet.
    """

    def can_allocate(self, seq: Sequence) -> AllocStatus: ...

    def allocate(self, seq: Sequence) -> None: ...

    def append_slot(self, seq: Sequence) -> tuple[int, int] | None:
        """Reserve the next token slot. Returns (block_id, offset) or None.

        Triggers COW when the tail block has `ref_count > 1`.
        """
        ...

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Share parent's blocks with child (`ref_count++` on each)."""
        ...

    def free(self, seq: Sequence) -> None: ...

    def swap_out(self, seq: Sequence) -> dict[int, int]:
        """GPU block id → CPU block id."""
        ...

    def swap_in(self, seq: Sequence) -> dict[int, int]:
        """CPU block id → GPU block id."""
        ...

    def get_num_free_blocks(self) -> int: ...

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Longest prefix-cache hit. Returns (physical block ids, n_tokens)."""
        ...


@runtime_checkable
class SchedulingPolicy(Protocol):
    """Swappable scheduling policy. FCFS, Horizon, and Oracle implement this.

    This is the most important architectural seam in the project: the
    research contribution is a drop-in policy, not a fork of the scheduler.
    """

    def order_waiting(self, waiting: list[Sequence], state: EngineState) -> list[Sequence]:
        """Return waiting sequences in admission-priority order."""
        ...

    def should_admit(self, seq: Sequence, state: EngineState) -> bool:
        """Whether to move `seq` from WAITING to RUNNING this step."""
        ...

    def select_preemption_victim(self, running: list[Sequence], state: EngineState) -> Sequence:
        """Pick a running sequence to preempt under memory pressure."""
        ...

    def preemption_mode(self, victim: Sequence, state: EngineState) -> PreemptionMode:
        """SWAP vs RECOMPUTE for this victim."""
        ...


class PrefixCache(Protocol):
    """Radix-tree prefix cache (S5). Hash + longest-prefix match."""

    def insert(self, token_ids: list[int], block_ids: list[int]) -> None: ...

    def match(self, token_ids: list[int]) -> tuple[list[int], int]: ...

    def evict(self, n_blocks: int) -> int: ...


# Re-export Mapping for type checkers that want a read-only block map.
BlockTable = list[int]
SwapMap = dict[int, int]
CopyMap = dict[int, list[int]]
