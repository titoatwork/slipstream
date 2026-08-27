# Architecture (implementer view)

Source of truth: MASTERPLAN §8. This file is the working map. If it drifts from the master plan, the master plan wins and this file is updated in the same PR.

## Processes

```
HTTP/SSE  →  API server (FastAPI)
                 ↓ Request
          Tokenizer / detokenizer process   ← never on the GPU loop
                 ↓ token ids
          EngineCore process
              Scheduler  →  BlockManager  →  ModelRunner  →  Sampler
                 ↓ metrics
          Observability → dashboard
```

Phase 0–3 may run EngineCore in-process. Phase 4 measures CPU occupancy before/after isolation and keeps the split only if it pays.

## Request lifecycle

1. ARRIVE → `Request`
2. TOKENIZE → `prompt_token_ids`
3. PREFIX → `match_prefix` → cached blocks + uncached suffix
4. ADMIT → policy `should_admit` (Horizon: projected memory high-water)
5. ALLOCATE → `BlockManager.allocate`
6. PREFILL → possibly chunked; stall-free: decodes first, then prefill remainder
7. DECODE → one token / step; `append_slot` when the tail block fills
8. PREEMPT? → SWAP or RECOMPUTE
9. FINISH → EOS / max_tokens / stop / abort
10. FREE → refcount--; prefix blocks may stay in the radix tree
11. DETOKENIZE → SSE

## Frozen contracts

All of these live in `slipstream/core/types.py` (plus `sampling_params.py`, `config.py`). **Do not change them without a §21 amendment.**

| Type | Role |
|---|---|
| `Sequence` | Scheduler unit of work |
| `SequenceStatus` | WAITING / RUNNING / SWAPPED / FINISHED_* |
| `PhysicalBlock` | One page of the KV cache |
| `BlockManager` | Protocol: allocate / append / fork / free / swap / prefix |
| `AllocStatus` | OK / LATER / NEVER |
| `SchedulerOutput` | One step's plan, including swap/COW maps |
| `SchedulingPolicy` | Protocol: order / admit / victim / mode |
| `EngineState` | Frozen snapshot handed to policies |
| `PreemptionMode` | SWAP / RECOMPUTE |
| `SamplingParams` | Frozen per-request sampler config |
| `EngineConfig` | Model + cache + scheduler + parallel |

### KV layout

```
kv_cache[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
                         ^ K/V
block_size = 16 (ablate 8/16/32)
slot_mapping[i] = physical_block_id * block_size + offset
block_tables[batch, max_blocks_per_seq]  int32, -1 padded
```

The attention kernel never sees logical positions. It walks `block_tables`.

### Policy seam

```python
from slipstream.scheduler.policies import get_policy
policy = get_policy(config.scheduler.policy)  # "fcfs" | "horizon" | "oracle"
```

Horizon is a policy, not a fork of `Scheduler`. That is what makes the ablation a one-line config change.

## Ownership

See `CODEOWNERS`. Agents implement against these interfaces, never against each other's modules.

## What is here (Phase 2)

- Phase 1 naive engine still works (`enable_paging=False`).
- Default path is **paged KV** + FCFS continuous batching (`generate` / `generate_batch`).
- Attention: gather+eager ref (identity). Triton decode is `SLIPSTREAM_TRITON=1`.
- Paging is a config flip. Token-identical to naive / HF greedy on T0.

Phase 3 is in: chunked prefill, radix prefix cache, swap preemption, OpenAI HTTP+SSE, block-table dashboard.

Phase 4: CUDA graphs, fused kernels, EngineCore isolation, Horizon scheduler.
