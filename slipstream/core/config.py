"""Engine configuration. Frozen after Phase 0 — amend §21 to change fields."""

from __future__ import annotations

from dataclasses import dataclass, field

from slipstream.core.types import DEFAULT_BLOCK_SIZE


def kv_bytes_per_token(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Bytes of KV cache consumed by one token (all layers, K and V).

    `2 × n_layers × n_kv_heads × head_dim × dtype_bytes`

    Llama-3.1-8B bf16: 2 × 32 × 8 × 128 × 2 = 131072 = 128 KiB/token.
    """
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


@dataclass
class ModelConfig:
    model_id: str
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    max_model_len: int = 4096
    revision: str | None = None
    # Filled from the checkpoint / config.json at load time:
    num_layers: int | None = None
    hidden_size: int | None = None
    num_q_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    vocab_size: int | None = None
    intermediate_size: int | None = None
    rms_norm_eps: float | None = None
    rope_theta: float | None = None
    rope_scaling: dict[str, object] | None = None
    rope_type: str = "default"  # default | llama3
    tie_word_embeddings: bool | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    model_type: str | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")

    @property
    def dtype_bytes(self) -> int:
        return {"bfloat16": 2, "float16": 2, "float32": 4}[self.dtype]


@dataclass
class CacheConfig:
    block_size: int = DEFAULT_BLOCK_SIZE
    gpu_memory_utilization: float = 0.90
    swap_space_bytes: int = 4 * 1024**3
    enable_prefix_caching: bool = True
    enable_paging: bool = True
    num_gpu_blocks: int | None = None  # computed at engine init
    num_cpu_blocks: int | None = None

    def __post_init__(self) -> None:
        if self.block_size not in {8, 16, 32}:
            raise ValueError("block_size must be one of {8, 16, 32}")
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = True
    prefill_chunk_size: int = 256
    policy: str = "fcfs"  # fcfs | horizon | oracle
    max_wait_ms: float = 30_000.0
    safety_factor: float = 0.95  # Horizon high-water mark
    starvation_guard_ms: float = 5_000.0

    def __post_init__(self) -> None:
        if self.policy not in {"fcfs", "horizon", "oracle"}:
            raise ValueError(f"unknown scheduling policy: {self.policy}")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1")
        if not (0.0 < self.safety_factor <= 1.0):
            raise ValueError("safety_factor must be in (0, 1]")


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")


@dataclass
class EngineConfig:
    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    seed: int = 0

    @classmethod
    def for_model(
        cls,
        model_id: str,
        *,
        cache: CacheConfig | None = None,
        scheduler: SchedulerConfig | None = None,
        parallel: ParallelConfig | None = None,
        seed: int = 0,
    ) -> EngineConfig:
        return cls(
            model=ModelConfig(model_id=model_id),
            cache=cache if cache is not None else CacheConfig(),
            scheduler=scheduler if scheduler is not None else SchedulerConfig(),
            parallel=parallel if parallel is not None else ParallelConfig(),
            seed=seed,
        )
