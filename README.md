# Slipstream

**A high-throughput LLM inference engine built from scratch.** The research lives at the *system* layer: a memory-aware, output-length-predictive scheduler operating under hard KV-cache constraints — not a single faster kernel, but the policy that decides which sequences occupy memory and run.

PagedAttention · continuous batching · chunked prefill · prefix caching · custom Triton kernels · speculative decoding · tensor parallelism · prefill/decode disaggregation.

---

> **Status — Phase 4, dev hardware only.** The numbers below are Qwen2.5-0.5B on a laptop RTX 3050 (the T0 dev tier), n≈3. Treat them as a sketch, not the A100 result table the targets further down are written against.
>
> Under memory pressure the scheduler does what it's meant to. On a KV-bound burst, FCFS over-admits and preempts 13 times per seed; Horizon refuses the overflow and preempts zero, which keeps its inter-token-latency tail flat (p99 ITL ~0.4s vs ~5s for FCFS — and the same holds when requests arrive over time rather than all at once). Measured by smooth goodput — the share of requests that stay under the latency cap the whole way through — Horizon beats FCFS on every seed, by 1.8× to 16×. The 16× was the first run and isn't representative; the median is ~6.6×. Two honest caveats: on raw throughput FCFS is still ahead, and Horizon misses the fairness target (p99 TTFT) on two of three seeds. The length prediction is earning its keep, though — a mean-only predictor holds 5 of 18 requests smooth where the full feature set holds all 18.
>
> Earlier phases, same hardware: chunked prefill cut p99 ITL by 64%; prefix-cache hit rate 93%.

## The thesis

Serving an LLM is a memory management problem disguised as a compute problem.

During decode, arithmetic intensity is approximately equal to batch size — which means on an A100 (ridge point ~153 FLOP/byte) the GPU is bandwidth-starved until roughly 150 sequences run concurrently. Naive KV cache allocation wastes 60–80% of the memory that determines that batch size. Fix the memory system and throughput follows; then the bottleneck becomes scheduling *policy*, which is where this project's research contribution lives.

## Research contribution

**Horizon** — an SLO-aware, output-length-predictive scheduler operating under hard KV-cache memory constraints. Production engines (vLLM, SGLang) schedule FCFS and are blind to how long a request will run. Horizon predicts remaining generation length online and schedules to maximize goodput rather than raw throughput.

Non-clairvoyant scheduling under KV constraints is an explicitly open problem in the 2026 literature.

## Documentation

📋 **[MASTERPLAN.md](./MASTERPLAN.md)** — the singular source of truth. Architecture, subsystem specs, phase gates, agent work division, benchmarking methodology, risk register.

## Targets

| Metric | Target |
|---|---|
| Throughput vs vLLM (A100, chat workload) | ≥ 85% |
| Throughput vs HuggingFace Transformers | ≥ 10× |
| KV cache memory waste | < 5% (vs > 60% naive) |
| Decode kernel achieved bandwidth (A100) | ≥ 70% |
| Horizon goodput gain over FCFS | ≥ 15% |

## Companion project

**[qgemm-mx](https://github.com/titoatwork/qgemm-mx)** is the deliberate counterpart to this work — the same thesis, one layer down. Both start from the fact that the decode path is bandwidth-bound. qgemm-mx attacks that wall at the **kernel** level, recovering block-scaled FP4 (MXFP4/NVFP4) throughput on GPUs without native FP4. Slipstream attacks it at the **system** level, where memory allocation and scheduling policy set the achievable batch size. A fast FP4 GEMM is exactly the kind of kernel that plugs into Slipstream's quantized decode path: kernel depth and system breadth, deliberately paired.

## AI assistance

This project was developed with AI assistance (Anthropic's Claude, via [Claude Code](https://claude.com/claude-code)). AI tooling contributed to the planning document, code generation, kernel drafts, and documentation. The research direction, architectural decisions, performance claims, and review of correctness are human-directed and human-owned; results are backed by the test and benchmark suites in this repo rather than by any assistant's assertion.

Commits containing substantial AI-generated content carry a `Co-Authored-By: Claude` trailer, following the [GitHub co-authorship convention](https://docs.github.com/articles/creating-a-commit-with-multiple-authors).

## License

MIT
