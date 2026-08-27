# Phase 0 log

Daily sync artifact (MASTERPLAN §11.4). Each agent appends: shipped / blocked / interface assumptions.

## 2026-08-13

### Shipped

- Repo scaffold matching §12: package tree, `pyproject.toml`, CI, pre-commit, CODEOWNERS, MIT license.
- **§8.3 contracts frozen** in `slipstream/core/types.py` (+ `sampling_params.py`, `config.py`). Supporting types that §8.3 named but did not specify (`SamplingParams`, `AllocStatus`, `PreemptionMode`, `EngineState`, `EngineConfig`, `Request`) are specified and recorded in §21.
- Typed stubs with `NotImplementedError` for every subsystem. Policy registry: `fcfs` / `horizon` / `oracle`.
- Gate 0 tests: import surface, contract shapes, forbidden-import lint, manifest capture.
- Benchmark harness skeleton + HF baseline runner (`benchmarks/baselines/hf_generate.py`).
- Docs: `environment.md` (T0 captured), `architecture.md`, `related_work.md`, `benchmarking_protocol.md`.

### Later same day

- Installed torch 2.13.0+cu126, triton 3.7.1, transformers 5.15.0 into `.venv`.
- HF baseline on Qwen2.5-0.5B / T0 recorded: median **11.09 tok/s** greedy, 8 prompts × 128 tokens, bf16. See `benchmarks/results/phase0/`.
- Gate 0 software/contracts/tests/baseline: closed. T1 cluster access still unverified.

### Still open for a complete Gate 0

- T1 cluster access not verified.

### Interface assumptions (call out within 24h if wrong)

- `Sequence.request_id` and `Sequence.oracle_output_len` added so the API and the oracle policy have somewhere to hang data. Not in the §8.3 sketch; amended.
- `EngineState` is a frozen dataclass of shared `Sequence` references. Policies must not mutate sequences.
- `BlockManager` is a `Protocol`; the concrete stub is `BlockManagerImpl` so we do not shadow the protocol name.
- `SchedulingPolicy` lives in `slipstream.core.types` and is re-exported from `scheduler/policies/base.py`.
- `block_size ∈ {8, 16, 32}` enforced in `CacheConfig` (the ablation set).
- Engine code does not import torch at contract level. `. [gpu]` is an extra.
- Python 3.12.3 on T0 accepted against a §7.3 pin of 3.11.
