# Benchmarking protocol

Binding rules: MASTERPLAN §13 (hygiene) and §15 (methodology). This file is the operational checklist.

## Every run

1. Exclusive GPU. Shared nodes invalidate the number.
2. Warmup ≥ 10 iterations, discarded.
3. ≥ 3 independent runs. Report median and p99, never a single best.
4. Write `run_manifest.json` via `benchmarks.manifest.write_run_manifest`.
5. Baselines on the **same** machine, session, and weights.
6. Any README number traces to a committed manifest under `benchmarks/results/`.

## Manifest fields

`timestamp`, `hostname`, `platform`, `gpu[]`, `cuda`, `packages`, `git_sha`, `clock_locked`, `exclusive_gpu`, `slurm_job_id`, `model`, `workload`, `config`.

## Metrics

TTFT, TPOT/ITL, throughput (tok/s), goodput, **smooth goodput**, KV utilization, MFU, achieved HBM bandwidth, preemption rate, prefix-cache hit rate.

Never tune against naive goodput alone. Report aborted/dropped counts. Headline comparisons are latency–throughput Pareto curves, not single points.

## Workloads (frozen in Phase 2)

| ID | Shape | Why |
|---|---|---|
| W1 Chat | prompt/output ~256, 20% prefix | headline vs vLLM |
| W2 RAG | long prompt, short output, 60% prefix | prefill + cache |
| W3 Agent | 90% prefix | prefix-cache stress |
| W4 Heterogeneous | bimodal lengths | **Horizon / research** |
| W5 ShareGPT | real trace | external validity |

## Phase 0 harness

- `benchmarks/manifest.py` — environment capture (works without a GPU)
- `benchmarks/baselines/hf_generate.py` — HF `generate()` floor on Qwen2.5-0.5B
- Results directory: `benchmarks/results/phase0/`
