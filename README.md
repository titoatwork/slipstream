# Slipstream

**A high-throughput LLM inference engine built from scratch, with a novel memory-aware scheduler.**

PagedAttention · continuous batching · chunked prefill · prefix caching · custom Triton kernels · speculative decoding · tensor parallelism · prefill/decode disaggregation.

---

> **Status:** Phase 4 on T0, **binding KV**. FCFS preempts 13× every closed seed; Horizon 0. T0-smooth hits always favor Horizon; the ratio is **1.8–16×** (median ~6.6×) — do not headline the first-run 16×. Fairness p99 TTFT fails 2× on 2/3 seeds. Open-loop: FCFS 71 preempts / 0 smooth; Horizon-f2 16/18. f0≠f2 on the closed cell. Paper 2s/200ms is a T0 miss. Phases 0–3: ITL p99 **−64%**, prefix **93%**.

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

## License

MIT
