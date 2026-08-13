# SLIPSTREAM — Master Plan

**A high-throughput LLM inference engine built from scratch, with a novel memory-aware scheduler.**

---

| Field | Value |
|---|---|
| **Document status** | v1.0 — Authoritative |
| **Created** | 2026-08-13 |
| **Owner** | Ibtesham Ul Haque |
| **Classification** | Singular source of truth. All engineering decisions defer to this document. |
| **Timeline** | 16 weeks |
| **Target** | Minor project (built to major-project / publishable standard) |
| **Domain** | ML Systems · LLM Inference · GPU Kernel Engineering · Distributed Systems |

### How to use this document

1. **Every phase gate is binding.** Do not advance a phase until its exit criteria pass. Regressions block merges.
2. **Interface contracts (§8) are frozen after Phase 0.** Agents build against contracts, not against each other's code. Changing a contract requires an explicit amendment entry in §21.
3. **Numbers in this document are commitments, not aspirations.** If a target proves wrong, amend the document — do not silently miss it.
4. **When in doubt, correctness first.** A fast engine that produces different logits than the reference is worth zero.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Thesis](#2-the-thesis)
3. [State of the Art — 2026 Landscape](#3-state-of-the-art--2026-landscape)
4. [Competitive Positioning](#4-competitive-positioning)
5. [The Research Contribution](#5-the-research-contribution)
6. [Goals, Non-Goals, Success Criteria](#6-goals-non-goals-success-criteria)
7. [Hardware & Environment Plan](#7-hardware--environment-plan)
8. [System Architecture](#8-system-architecture)
9. [Subsystem Specifications](#9-subsystem-specifications)
10. [Phase Plan](#10-phase-plan)
11. [Agent Organization & Work Division](#11-agent-organization--work-division)
12. [Repository Structure](#12-repository-structure)
13. [Engineering Standards](#13-engineering-standards)
14. [Correctness Strategy](#14-correctness-strategy)
15. [Benchmarking Methodology](#15-benchmarking-methodology)
16. [Performance Model & Numeric Targets](#16-performance-model--numeric-targets)
17. [Risk Register](#17-risk-register)
18. [Academic Deliverables](#18-academic-deliverables)
19. [Career Positioning](#19-career-positioning)
20. [Reading List](#20-reading-list)
21. [Amendment Log](#21-amendment-log)
22. [Glossary](#22-glossary)

---

## 1. Executive Summary

Slipstream is a from-scratch LLM inference and serving engine implementing the full modern optimization stack — PagedAttention, continuous batching, chunked prefill, prefix caching, custom Triton kernels, speculative decoding, quantization, tensor parallelism, and prefill/decode disaggregation — benchmarked rigorously against HuggingFace Transformers, vLLM, and SGLang on A100/H100 hardware.

On top of the engine sits the project's original contribution: **SLO-aware, output-length-predictive scheduling under hard KV-cache memory constraints** — a documented open problem in the 2026 literature. Standard engines schedule FCFS and are blind to how long a request will run; Slipstream predicts remaining generation length and schedules to maximize *goodput* (SLO-satisfying requests/sec) rather than raw throughput.

**Why this project:** it is simultaneously (a) an infrastructure engineering artifact that maps directly to the highest-demand backend/ML-systems roles, (b) a legitimate systems-research contribution suitable for a workshop paper, and (c) a demo that visibly beats a production system on a metric that matters.

**Deliverables:**
- A working, documented, benchmarked inference engine (~6–8k LOC)
- A reproducible benchmark suite with published results and ablations
- A research paper (workshop-grade) on the scheduling contribution
- A live visual demo suitable for a panel
- Full academic report + a public technical blog series

---

## 2. The Thesis

> **Serving an LLM is a memory management problem disguised as a compute problem.**

Three facts drive every design decision in this document:

**Fact 1 — Decode is memory-bandwidth-bound, not compute-bound.**
During decode, one token per sequence is generated per forward pass. The engine reads the *entire model weight set* from HBM to compute a handful of FLOPs. Arithmetic intensity is approximately equal to the batch size. On an A100, the roofline ridge point is ~153 FLOP/byte — meaning you need a batch of roughly 150 concurrent sequences before the GPU stops being bandwidth-starved and starts being compute-limited.

**Corollary:** throughput is a function of *how many sequences you can hold in memory simultaneously*. Batching is not an optimization; it is the entire game.

**Fact 2 — Naive KV cache allocation wastes 60–80% of the memory that determines batch size.**
Allocating a contiguous `max_seq_len`-sized KV buffer per sequence means a request that generates 100 tokens under a 4096 limit wastes 97% of its allocation. This is textbook internal fragmentation, and it directly caps the batch size, which directly caps throughput. PagedAttention fixes this by treating the KV cache as virtual memory: fixed-size blocks, on-demand allocation, a per-sequence block table.

**Fact 3 — Once memory is efficiently managed, the bottleneck becomes *policy*.**
When you can hold 200 sequences but only have memory for 150, you must decide which to run, which to preempt, and which to admit. Production engines answer this with FCFS — a policy that is provably poor when request costs are heterogeneous and unknown. This is where Slipstream's contribution lives.

The project is therefore structured as: **build the memory system → build the kernels that exploit it → build the scheduler that the memory system enables → improve that scheduler beyond the state of the art.**

---

## 3. State of the Art — 2026 Landscape

*Synthesized from primary sources; see §20 for full reading list.*

### 3.1 The technique stack

| Technique | Origin | Status 2026 | Slipstream implements |
|---|---|---|---|
| **KV caching** | Baseline | Universal | ✅ Phase 1 |
| **PagedAttention** | vLLM (SOSP '23) | Universal default | ✅ Phase 2 |
| **Continuous batching** | Orca (OSDI '22) | Universal default | ✅ Phase 2 |
| **Chunked prefill / stall-free sched.** | Sarathi-Serve (OSDI '24) | Default in vLLM & SGLang | ✅ Phase 3 |
| **Prefix caching / RadixAttention** | SGLang | Default; 75–95% hit rates on agent workloads | ✅ Phase 3 |
| **CUDA graphs** | NVIDIA | Standard; 30–50% CPU overhead cut at small batch | ✅ Phase 4 |
| **FlashAttention-2 / -3** | Dao et al. | FA3 standard on Hopper; 75% peak FP8 FLOPS | ✅ Phase 2 (own Triton impl.) |
| **Speculative decoding (EAGLE-3)** | 2025 | Merged in vLLM/SGLang/TRT-LLM early 2026; 0.80–0.88 acceptance, 3–4× | ✅ Phase 5 (simplified) |
| **W4A16 quant (AWQ/GPTQ/Marlin)** | 2023–24 | Marlin standard; ~4× decode gain, zero prefill gain | ✅ Phase 5 |
| **FP8 (E4M3/E5M2)** | Hopper+ | Standard on H100 | ⚠️ Phase 5 stretch |
| **Tensor parallelism** | Megatron | Universal for >1 GPU | ✅ Phase 6 |
| **PD disaggregation** | DistServe, Splitwise | Production-adopted; up to 7.4× goodput | ✅ Phase 6 |
| **Multi-tier KV offload** | LMCache, Mooncake | Rapidly maturing; 3–10× latency cuts | ⚠️ Phase 6 stretch |
| **Length-predictive scheduling** | **Open problem** | **Active research, no production system** | ⭐ **Phase 4 — our contribution** |

### 3.2 Key quantitative findings from the literature

These numbers anchor our targets and must be reproduced or explained:

- **Continuous batching + PagedAttention** raises GPU utilization from 30–40% → 75–90%, yielding **2–4× more output tokens per GPU-hour**.
- **Sarathi-Serve chunked prefill**: 2.6× higher serving capacity vs vLLM on Mistral-7B / A100. Chunk sizes of 256–512 tokens satisfy tail-latency constraints; vLLM's current default online chunk budget is 8192.
- **RadixAttention**: significant TTFT reduction on workloads with >60% prefix overlap; 75–95% cache hit rates on multi-turn agent traffic.
- **Triton attention backend** (vLLM, Mar 2026): achieved **100.7% of FlashAttention-3 performance** for long decode on H100, and ~5.8× over prior impls on MI300 — in **~800 lines of Triton vs ~70,000 lines in FA3**. This is direct evidence that our Triton-first strategy is sound and not a toy compromise.
- **CUDA graphs**: kernel launch overhead is ~5–20 µs each; at small batch this is **30–60% of step time**. Reduce-overhead compilation drops overhead ratios from 144× to 11× for launch-heavy small models.
- **W4A16 INT4**: ~4× decode speedup at low concurrency (memory-bound regime hides dequant cost); **zero prefill speedup** and a net loss in the compute-bound regime. Marlin beats legacy AWQ kernels by 1.7–2.4× on Ada.
- **PD disaggregation**: Splitwise ~20% lower cost at 1.4× throughput; DistServe up to 7.4× higher goodput. Prefill draws 70–100% of peak GPU power, decode only 20–40%.
- **NVLink 4.0** offers 900 GB/s intra-node; using RDMA where NVLink was available wastes ~18× bandwidth — KV transfer path selection matters.

### 3.3 What is solved vs what is open

**Solved (engineering, not research):** paged memory, iteration-level batching, chunked prefill, prefix reuse, attention kernels, weight quantization, tensor parallelism. Implementing these well is *hard engineering* and is the bulk of our work — but it is not novel.

**Open (genuine research frontier, 2026):**
1. **Non-clairvoyant scheduling under KV constraints.** A request's memory footprint grows linearly with tokens generated, but output length is unknown at admission. Obtaining robust performance guarantees without prior knowledge of job sizes is explicitly framed in the 2026 literature as "a theoretically fundamental and practically important open problem."
2. **Goodput metric integrity.** Naive goodput definitions are gameable — systems can delay token delivery to smooth tail latency or drop requests about to miss SLOs. "Smooth goodput" proposals exist but are not standardized.
3. **Unifying PD aggregation and disaggregation** adaptively based on load regime.
4. **Multi-tier KV placement policy** across HBM/DRAM/NVMe under heterogeneous access patterns.
5. **Preemption policy**: swap-to-CPU vs recompute is decided by crude heuristics in production engines.

Slipstream attacks **(1)** directly and **(5)** as a secondary contribution, and takes **(2)** seriously in its measurement methodology.

---

## 4. Competitive Positioning

### 4.1 The landscape we are entering

| System | Scale | What it is | Our relationship to it |
|---|---|---|---|
| **vLLM** | ~100k LOC | Production reference. V1 engine isolates scheduler + EngineCore in separate processes for near-zero CPU overhead. Unified scheduler treats prompt/output tokens identically via a token budget dict. | **Primary benchmark baseline.** We target ≥85% of its throughput. |
| **SGLang** | ~80k LOC | RadixAttention pioneer, strong on structured/agentic workloads. | Secondary baseline for prefix-cache benchmarks. |
| **TensorRT-LLM** | Huge | NVIDIA's compiled-kernel stack. Fastest, least hackable. | Out of scope; cited in report. |
| **HuggingFace `generate()`** | — | Naive baseline. Static batching, contiguous cache. | **Floor baseline.** We target ≥10× on throughput. |
| **nano-vLLM** | ~1.2k LOC | Educational reimplementation of core vLLM ideas. Widely known. | **The bar we must clear.** See below. |

### 4.2 The nano-vLLM problem — and how we clear it

nano-vLLM exists, is popular, and implements PagedAttention + continuous batching in ~1,200 lines. **If Slipstream is "nano-vLLM again," the project has no defensible identity.** This is the single most important positioning risk and is addressed by four hard differentiators:

| # | Differentiator | nano-vLLM | Slipstream |
|---|---|---|---|
| **D1** | **Own Triton kernels** | Uses off-the-shelf attention | Writes paged attention, fused RMSNorm, RoPE, SwiGLU in Triton; microbenchmarked against PyTorch eager and FlashAttention with achieved-bandwidth analysis |
| **D2** | **Novel scheduler** | FCFS | Length-predictive, SLO-aware, memory-constrained scheduling with full ablation study — a research contribution |
| **D3** | **Rigorous measurement** | Basic throughput | Full goodput/SLO methodology, roofline + MFU analysis, latency-throughput Pareto frontiers, statistical rigor, gameability-resistant metrics |
| **D4** | **Distributed layer** | Single GPU | Tensor parallelism + prefill/decode disaggregation across cluster nodes |

**Positioning statement for the report and viva:**

> "nano-vLLM demonstrates *that* these ideas work. Slipstream investigates *how well* they work, *why* they work at the hardware level, and *where the remaining policy headroom is*. We reimplement the stack in order to have a research substrate we fully control, then use it to attack an open scheduling problem that production engines have not solved."

---

## 5. The Research Contribution

### 5.1 Problem statement

In an LLM serving system with a hard KV-cache memory budget `M`:
- Each admitted request `r` consumes memory that **grows linearly** with tokens generated: `mem(r, t) = ceil((prompt_len + t) / B) × block_bytes`
- The **total output length is unknown at admission time** (non-clairvoyant)
- Admitting too many requests → memory exhaustion → preemption cascades → thrashing
- Admitting too few → GPU underutilized → throughput collapse (see Fact 1)
- Production engines (vLLM V1, SGLang) use **FCFS with a greedy token budget**, which is blind to this tradeoff

**Research question:**
> Can a lightweight, online-learned predictor of remaining output length improve *goodput* (SLO-satisfying requests/sec) over FCFS scheduling under hard KV-cache memory constraints — and if so, by how much, under which workload regimes, and at what prediction-accuracy threshold does the benefit disappear?

### 5.2 Proposed method — "Horizon Scheduling"

**Component A — Length predictor.**
A cheap online model predicting remaining generation length for an in-flight request. Candidate feature sets (ablated):
- `f0` (baseline): global mean output length — no prediction
- `f1`: prompt length, prompt token entropy, task-type heuristics
- `f2`: `f1` + tokens generated so far + rolling EOS-logit probability from the sampler
- `f3`: `f2` + a small MLP head on the model's final hidden state (adds compute; must justify)

Predictor must cost **< 0.5% of step time** — this constraint is non-negotiable and is itself a result worth reporting.

**Component B — Memory-aware admission & preemption policy.**
Given predicted remaining lengths, compute per-request *projected peak memory* and *projected completion time*. Schedule to maximize expected goodput:
- **Admission**: admit request `r` iff projected memory high-water mark stays below `M × safety_factor`
- **Ordering**: approximate SRPT (Shortest Remaining Processing Time) over predicted remaining lengths, bounded by a starvation guard (max wait threshold) to preserve fairness
- **Preemption**: when memory pressure hits, evict the request with worst `(predicted_remaining × memory_growth_rate) / SLO_slack` — and choose swap-vs-recompute based on measured block count and PCIe bandwidth rather than a fixed heuristic

**Component C — Honest evaluation.**
- Report **smooth goodput**, not naive goodput, to avoid the known gameability failure mode
- Report the **fairness cost** of SRPT-like ordering (p99 wait time, starvation incidence) — an honest paper reports the tradeoff
- Include an **oracle upper bound** (perfect length knowledge) to bound achievable gains
- Include the **negative result regimes**: where prediction does not help

### 5.3 Success criteria for the research component

| Criterion | Threshold |
|---|---|
| Goodput improvement over FCFS on heterogeneous workloads | **≥ 15%** |
| Predictor overhead | **< 0.5%** of step time |
| Gap closed toward oracle | **≥ 40%** |
| Regimes characterized where method **fails** | ≥ 2 documented |
| Fairness degradation bounded | p99 wait increase **< 2×** vs FCFS |

**Fallback (if the primary hypothesis fails):** A rigorous negative result — "length prediction does not pay for itself below accuracy threshold X" — with the oracle bound and a characterization of *why*, is a legitimate and publishable outcome. **This project cannot fail scientifically**, only engineering-wise. That property is deliberate.

### 5.4 Secondary contribution

**Preemption policy study**: swap-to-CPU vs recompute is decided by fixed heuristics in production. We measure the true crossover as a function of (block count, PCIe/NVLink bandwidth, prompt length, current batch composition) and propose a measured decision rule. Smaller in scope, high confidence of a clean result.

---

## 6. Goals, Non-Goals, Success Criteria

### 6.1 Goals

**G1 — Engineering.** A working inference engine serving Llama-3.1-8B on A100 with paged KV, continuous batching, chunked prefill, prefix caching, custom Triton kernels, and an OpenAI-compatible API.

**G2 — Performance.** ≥85% of vLLM throughput and ≥10× HuggingFace on identical hardware and workload.

**G3 — Research.** A validated (or rigorously falsified) scheduling contribution with full ablations.

**G4 — Scale.** Tensor parallelism across ≥2 GPUs and a working PD-disaggregated deployment.

**G5 — Communication.** Academic report, workshop-grade paper, public repo with reproducible benchmarks, blog series, and a panel demo.

### 6.2 Non-Goals (explicitly out of scope — do not scope-creep)

| Not doing | Why |
|---|---|
| Training or fine-tuning models | Different problem entirely |
| Multi-modal (vision/audio) inference | Orthogonal complexity, no added signal |
| Pipeline parallelism | TP is sufficient to demonstrate the concept |
| Beating vLLM outright on throughput | Unrealistic; 85% parity + a novel scheduler is the honest, stronger claim |
| Custom CUDA C++ for all kernels | Triton is the industry direction and proven at 100.7% of FA3. Two hand-written CUDA kernels for the résumé line only. |
| Supporting >3 model architectures | Llama + Qwen families cover the demonstration |
| MoE models | Adds routing complexity without adding to the thesis |
| Production hardening (auth, rate limits, multi-tenancy) | Not a research or résumé signal |

### 6.3 Definition of Done

The project is complete when **all** of the following hold:

- [ ] All 6 phase gates passed
- [ ] `pytest` green, including numerical-parity suite against HuggingFace
- [ ] Benchmark suite reproducible from a single command on a clean cluster node
- [ ] Published results table: Slipstream vs vLLM vs SGLang vs HF, on ≥3 workload profiles
- [ ] Ablation study for every optimization (each must justify its existence with a number)
- [ ] Research contribution validated or rigorously falsified with oracle bound
- [ ] Report submitted; paper drafted; blog series published; demo rehearsed to ≤10 min

---

## 7. Hardware & Environment Plan

### 7.1 Tiered hardware strategy

| Tier | Hardware | Role | Model | Notes |
|---|---|---|---|---|
| **T0 — Dev** | RTX 3050 Laptop, 6GB, GA107, SM 8.6, ~139–144 GB/s | Daily iteration, correctness, kernel debugging | Qwen2.5-0.5B | Fast edit-run loop. bf16 supported. **All correctness work happens here.** |
| **T1 — Primary** | A100 (40/80GB), SM 8.0, 1555/2039 GB/s, 312 TFLOPS bf16 | Headline benchmarks, all published numbers | Llama-3.1-8B | **The canonical result platform.** |
| **T2 — Hopper** | H100, SM 9.0, 3350 GB/s, 989 TFLOPS bf16 | FP8 experiments, FA3 comparison, scaling study | Llama-3.1-8B | wgmma/TMA features; FP8 E4M3/E5M2. |
| **T3 — Legacy** | V100 32GB, SM 7.0, 900 GB/s, 125 TFLOPS fp16 | Portability + architecture-sensitivity study | Llama-3.1-8B fp16 | ⚠️ **No bf16.** Must use fp16. Triton support on Volta is weaker — treat any V100 result as a bonus, never a dependency. |
| **T4 — Multi-GPU** | 2–8× A100 (NVLink preferred) | TP, PD disaggregation | Llama-3.1-8B, optionally 70B | Record interconnect topology — NVLink vs PCIe changes conclusions. |

**Rule:** Every published number states GPU model, memory, driver, CUDA version, PyTorch version, Triton version, clock state, and whether the GPU was exclusive. Non-exclusive nodes invalidate results.

### 7.2 Models

| Model | Layers | Hidden | Q heads | KV heads | Head dim | KV bytes/token (bf16) | Use |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 896 | 14 | 2 | 64 | 12.3 KB | T0 dev, fast tests |
| Qwen2.5-1.5B | 28 | 1536 | 12 | 2 | 128 | 28.7 KB | Mid-tier validation |
| **Llama-3.1-8B** | 32 | 4096 | 32 | 8 | 128 | **128 KB** | **Primary benchmark model** |
| Llama-3.2-1B | 16 | 2048 | 32 | 8 | 64 | 32 KB | Draft model for spec decoding |

**KV memory formula (memorize this — it appears in the viva):**

```
kv_bytes_per_token = 2 (K and V) × n_layers × n_kv_heads × head_dim × dtype_bytes
```

Llama-3.1-8B: `2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KB/token`

On an A100-80GB with 16GB of bf16 weights and ~4GB activation/workspace overhead, ~60GB remains for KV → **~490,000 cached tokens** → ~240 concurrent sequences at 2048 tokens each. This number is the throughput ceiling and every memory optimization is measured against it.

### 7.3 Software environment

```
Python        3.11
PyTorch       2.6+ (pin exact version; record in every result)
Triton        3.x (ships with PyTorch)
CUDA          12.4+ (13.1 available on T0)
transformers  (reference implementation for parity testing only)
vLLM, SGLang  (baselines only — never imported by engine code)
FastAPI/uvicorn, pydantic
pytest, hypothesis
numpy, pandas, matplotlib
nsight-systems, nsight-compute (profiling)
```

**Hard rule:** `slipstream/` must never import `vllm`, `sglang`, or `transformers` modeling code. Reference implementations live only in `tests/` and `benchmarks/`. This is enforced by a CI lint check.

### 7.4 Cluster discipline

- Reserve exclusive GPU allocations for benchmark runs; shared nodes produce unusable numbers
- Lock clocks where permitted (`nvidia-smi -lgc`) and record whether locking was applied
- Log SLURM/scheduler job IDs alongside results for traceability
- Cache model weights on shared/local storage; never re-download in a benchmark run
- Every benchmark run writes a `run_manifest.json` with full environment capture

---

## 8. System Architecture

### 8.1 High-level component diagram

```
                    ┌─────────────────────────────────────┐
   HTTP/SSE ───────▶│  API Server (FastAPI, async)        │
                    │  OpenAI-compatible /v1/completions  │
                    └──────────────┬──────────────────────┘
                                   │ Request objects
                    ┌──────────────▼──────────────────────┐
                    │  Tokenizer / Detokenizer Process    │
                    │  (isolated — never blocks EngineCore)│
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────────────────▼───────────────────────────┐
        │              ENGINE CORE (own process)               │
        │  ┌────────────────────────────────────────────────┐  │
        │  │  SCHEDULER                                     │  │
        │  │  • waiting / running / swapped queues          │  │
        │  │  • token-budget batch formation                │  │
        │  │  • chunked prefill interleaving                │  │
        │  │  • preemption (swap ⇄ recompute)               │  │
        │  │  • ⭐ Horizon policy + length predictor        │  │
        │  └───────────────┬────────────────────────────────┘  │
        │                  │ SchedulerOutput                    │
        │  ┌───────────────▼────────────────────────────────┐  │
        │  │  BLOCK MANAGER (paged KV allocator)            │  │
        │  │  • free block pool  • per-seq block tables     │  │
        │  │  • refcounts + copy-on-write                   │  │
        │  │  • ⭐ radix prefix tree  • CPU swap space      │  │
        │  └───────────────┬────────────────────────────────┘  │
        │                  │ block_tables, slot_mapping         │
        │  ┌───────────────▼────────────────────────────────┐  │
        │  │  MODEL RUNNER                                  │  │
        │  │  • CUDA graph capture/replay (decode)          │  │
        │  │  • batch tensor assembly                       │  │
        │  │  ┌──────────────────────────────────────────┐  │  │
        │  │  │ TRANSFORMER (Llama/Qwen)                 │  │  │
        │  │  │  Triton: paged_attn · RMSNorm · RoPE ·   │  │  │
        │  │  │          SwiGLU · sampling               │  │  │
        │  │  │  Optional: TP shards + NCCL all-reduce   │  │  │
        │  │  └──────────────────────────────────────────┘  │  │
        │  └───────────────┬────────────────────────────────┘  │
        │  ┌───────────────▼────────────────────────────────┐  │
        │  │  SAMPLER  (greedy/temp/top-k/top-p, seeded)    │  │
        │  └────────────────────────────────────────────────┘  │
        └──────────────────────┬───────────────────────────────┘
                               │ metrics stream
                    ┌──────────▼──────────────────────────┐
                    │  OBSERVABILITY  →  Live Dashboard   │
                    │  block-table viz · TTFT/TPOT · MFU  │
                    └─────────────────────────────────────┘
```

### 8.2 Request lifecycle

```
1. ARRIVE     → Request(prompt, sampling_params, arrival_ts, slo)
2. TOKENIZE   → token_ids                        [separate process]
3. PREFIX     → radix tree lookup → cached_blocks, uncached_suffix
4. ADMIT      → Scheduler decides: WAITING → RUNNING
                (Horizon: check projected memory high-water mark)
5. ALLOCATE   → BlockManager assigns blocks for uncached tokens
                (shared prefix blocks get refcount++, COW on divergence)
6. PREFILL    → possibly chunked over N iterations, interleaved with decodes
7. DECODE     → one token/iteration; append block when last block fills
8. PREEMPT?   → on memory pressure: swap blocks to CPU, or free + recompute later
9. FINISH     → EOS token, max_tokens, or stop string
10. FREE      → refcount--, blocks return to pool; prefix blocks may persist in radix tree
11. DETOKENIZE→ stream tokens to client via SSE  [separate process]
```

### 8.3 Core data structures (frozen contracts)

```python
# ---- Request state ----
class SequenceStatus(Enum):
    WAITING, RUNNING, SWAPPED, FINISHED_STOPPED, FINISHED_LENGTH, FINISHED_ABORTED

@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    block_table: list[int]          # logical block idx -> physical block id
    status: SequenceStatus
    sampling_params: SamplingParams
    arrival_ts: float
    first_token_ts: float | None
    num_cached_tokens: int          # from prefix cache hit
    num_computed_tokens: int        # for chunked prefill progress
    # --- research fields ---
    predicted_remaining: int | None
    slo_ttft_ms: float
    slo_tpot_ms: float

# ---- Memory ----
class PhysicalBlock:
    block_id: int
    ref_count: int
    block_hash: int | None          # content hash for prefix cache
    num_tokens: int                 # filled slots in this block

class BlockManager:
    """Owns ALL KV memory. Single source of truth for allocation."""
    def can_allocate(self, seq) -> AllocStatus: ...      # OK | LATER | NEVER
    def allocate(self, seq) -> None: ...
    def append_slot(self, seq) -> tuple[int,int] | None: ...  # COW copy if needed
    def fork(self, parent, child) -> None: ...           # refcount++ for beam/parallel
    def free(self, seq) -> None: ...
    def swap_out(self, seq) -> dict[int,int]: ...        # gpu_block -> cpu_block
    def swap_in(self, seq) -> dict[int,int]: ...
    def get_num_free_blocks(self) -> int: ...
    def match_prefix(self, token_ids) -> tuple[list[int], int]: ...  # blocks, n_matched

# ---- Scheduling ----
@dataclass
class SchedulerOutput:
    scheduled_seqs: list[Sequence]
    num_batched_tokens: int
    blocks_to_swap_in: dict[int,int]
    blocks_to_swap_out: dict[int,int]
    blocks_to_copy: dict[int,list[int]]   # COW
    is_prefill_chunk: dict[int,bool]

class SchedulingPolicy(Protocol):
    """FROZEN INTERFACE. All policies (FCFS, Priority, Horizon) implement this.
       This is the seam that makes the research contribution swappable + ablatable."""
    def order_waiting(self, waiting: list[Sequence], state: EngineState) -> list[Sequence]: ...
    def should_admit(self, seq: Sequence, state: EngineState) -> bool: ...
    def select_preemption_victim(self, running: list[Sequence], state: EngineState) -> Sequence: ...
    def preemption_mode(self, victim: Sequence, state: EngineState) -> PreemptionMode: ...
```

**Design note:** `SchedulingPolicy` being a frozen Protocol from Phase 0 is the most important architectural decision in this document. It lets the Research agent develop Horizon in complete isolation from the Scheduler agent's mechanism work, and it makes every ablation a one-line config change rather than a code fork.

### 8.4 Memory layout of the KV cache

```
kv_cache: Tensor[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
                              ^K/V

block_size = 16 tokens (default; ablate 8/16/32)

slot_mapping[i] = physical_block_id * block_size + offset_in_block
    → flat index for token i, used by the kernel to scatter K/V writes

block_tables: Tensor[batch, max_blocks_per_seq]  (int32, -1 padded)
    → the "page table" the attention kernel walks
```

The attention kernel never sees logical positions. It receives `block_tables` and `seq_lens` and gathers KV through the indirection. **This indirection is the entire trick** — and the reason a naive contiguous kernel cannot be reused.

### 8.5 Process model

Following vLLM V1's proven design: **isolate the EngineCore loop in its own process.** Tokenization, detokenization, multimodal preprocessing, and HTTP serialization are CPU-heavy and must never block the GPU step loop. Communication via ZMQ or multiprocessing queues with msgpack-serialized batches.

Measured justification required in Phase 4: report step-loop CPU occupancy before and after isolation.

---

## 9. Subsystem Specifications

Each subsystem lists: **owner agent · dependencies · deliverable · invariants · tests · exit criteria.**

### S1 — Model Runtime
**Owner:** A1 · **Depends on:** none · **Phase:** 1

Weight loading (safetensors, sharded), Llama/Qwen architecture in PyTorch, RoPE (incl. Llama-3 scaling), GQA, RMSNorm, SwiGLU MLP, tied/untied embeddings, dtype handling (bf16 on Ampere+, fp16 on Volta).

**Invariants:**
- I1.1: Logits match HuggingFace within `atol=1e-2, rtol=1e-2` (bf16) for identical inputs
- I1.2: Greedy decoding produces **token-identical** output to HF for ≥50 prompts × 128 tokens
- I1.3: No dynamic shape recompilation in the decode path

**Tests:** `test_parity_logits`, `test_parity_greedy`, `test_rope_correctness`, `test_gqa_head_mapping`, weight-loading roundtrip.

**Exit:** Single-sequence generation matches HF token-for-token on all dev models.

---

### S2 — Paged KV Cache & Block Manager
**Owner:** A2 · **Depends on:** S1 (shapes) · **Phase:** 2

Physical block pool, per-sequence block tables, refcounting, copy-on-write, free-list management, CPU swap space, allocation status (`OK`/`LATER`/`NEVER`), fragmentation accounting.

**Invariants:**
- I2.1: `sum(ref_counts) == blocks_allocated` at every step boundary — **assert in debug builds**
- I2.2: No block is in the free list while `ref_count > 0`
- I2.3: Freeing a sequence returns exactly its blocks, never more
- I2.4: COW triggers iff `ref_count > 1` on write
- I2.5: Internal fragmentation ≤ `block_size - 1` tokens per sequence (structural guarantee)

**Tests:** Property-based (hypothesis) allocate/free/fork/swap sequences with an invariant checker after every op. Fuzz with 10k random op sequences. Leak test: run 10k requests, assert free-block count returns to initial.

**Exit:** Fuzz suite passes; measured KV waste < 5% vs > 60% for the naive allocator.

---

### S3 — Triton Kernels
**Owner:** A3 · **Depends on:** S2 (layout) · **Phase:** 2, 4

| Kernel | Priority | Notes |
|---|---|---|
| `paged_attention_decode` | **P0** | Gathers KV through block table. The hard one. |
| `paged_attention_prefill` | **P0** | FlashAttention-style tiling + causal mask + chunk support |
| `reshape_and_cache` | **P0** | Scatter K/V into paged cache via `slot_mapping` |
| `fused_rmsnorm` | P1 | + residual add fusion |
| `fused_rope` | P1 | Applied to Q,K in one pass |
| `fused_swiglu` | P1 | gate·silu ⊙ up, fused |
| `copy_blocks` / `swap_blocks` | P1 | COW + CPU swap |
| `sampling` (top-k/top-p) | P2 | Avoid CPU sync in the hot loop |

**Design guidance from the SOTA:** vLLM's Triton backend groups multiple query tokens into **Q blocks** to improve `tl.dot` utilization (critical under GQA), and uses **persistent kernels** with a fixed launch grid so CUDA graphs can be reused. Variable launch grids conflict with graph capture. Adopt both patterns.

**Invariants:**
- I3.1: Every kernel matches a naive PyTorch reference within `atol=1e-2` on randomized inputs
- I3.2: No kernel reads a block not present in the sequence's block table (bounds-check in debug)
- I3.3: Decode kernel launch grid is **static** (CUDA-graph compatible)

**Tests:** Per-kernel numerical parity vs reference across shape sweeps (batch 1–256, seqlen 1–8192, heads, block sizes 8/16/32). Autotune configs recorded per GPU arch.

**Exit:** All P0 kernels correct; `paged_attention_decode` ≥ 70% of achievable memory bandwidth on A100; microbenchmark table published.

---

### S4 — Scheduler (Mechanism)
**Owner:** A4 · **Depends on:** S2 · **Phase:** 2, 3

Iteration-level continuous batching, token-budget batch formation, chunked prefill interleaving (Sarathi-Serve stall-free schedule: **admit decodes first, then fill remaining budget with prefill chunks**), preemption (swap and recompute modes), FCFS reference policy.

**Invariants:**
- I4.1: `num_batched_tokens ≤ token_budget` every step
- I4.2: A running decode is **never** stalled by a newly admitted prefill (stall-free property — assert in tests)
- I4.3: No sequence starves beyond `max_wait_ms` under any policy
- I4.4: Preemption is always recoverable — a preempted sequence resumes with identical output

**Tests:** `test_stall_free` (inject a 8k-token prefill mid-decode, assert decode ITL variance stays bounded), `test_preempt_resume_identical` (force preemption, assert token-identical output), starvation test.

**Exit:** Continuous batching yields ≥3× throughput over static batching; chunked prefill cuts decode ITL p99 by ≥40% under mixed load.

---

### S5 — Prefix Cache (RadixAttention)
**Owner:** A2 · **Depends on:** S2 · **Phase:** 3

Radix tree keyed on token sequences, block-level content hashing, automatic longest-prefix match on admission, refcount-based sharing with COW, LRU eviction of unreferenced cached blocks.

**Invariants:**
- I5.1: Cache hit produces **token-identical** output to cache miss (critical — silent correctness bug otherwise)
- I5.2: Shared blocks never mutated in place (COW enforced)
- I5.3: Evicted blocks have `ref_count == 0`

**Tests:** `test_prefix_hit_identical_output` over 500 shared-prefix request pairs. Hit-rate measurement under synthetic agent workload.

**Exit:** ≥60% hit rate and ≥50% TTFT reduction on a shared-system-prompt workload.

---

### S6 — Sampler
**Owner:** A1 · **Depends on:** S1 · **Phase:** 1, 4

Greedy, temperature, top-k, top-p, repetition penalty, min-p, stop strings/tokens, seeded reproducibility, batched GPU-side sampling without host sync.

**Invariants:** I6.1: Same seed + same params ⇒ identical output. I6.2: No `.item()` / `.cpu()` in the decode hot path.

**Exit:** Deterministic under seed; sampler is <2% of step time.

---

### S7 — API Server
**Owner:** A5 · **Depends on:** S4 · **Phase:** 3

FastAPI, OpenAI-compatible `/v1/completions` and `/v1/chat/completions`, SSE token streaming, request cancellation/abort, health + `/metrics` endpoints, isolated tokenizer/detokenizer processes.

**Exit:** `openai` Python client works unmodified against Slipstream. Cancellation frees blocks within one step.

---

### S8 — Speculative Decoding
**Owner:** A3 + A4 · **Depends on:** S3, S4 · **Phase:** 5

Draft-model speculation (Llama-3.2-1B drafting for 8B), tree/linear candidate verification, rejection sampling that **provably preserves the target distribution**, acceptance-rate instrumentation, adaptive draft length.

*Scoped deliberately below EAGLE-3 (which needs a trained head). Draft-model speculation gives the same systems lessons at a fraction of the cost. EAGLE-3 is a documented stretch goal.*

**Invariants:** I8.1: Output distribution is statistically indistinguishable from non-speculative sampling (chi-squared test over 10k samples). **This is the classic silent-bug site.**

**Exit:** ≥1.5× decode speedup at low batch; acceptance rate reported vs draft length; the regime where speculation *hurts* (high batch) documented.

---

### S9 — Quantization
**Owner:** A3 · **Depends on:** S3 · **Phase:** 5

W4A16 (INT4 weights, FP16 activations) with group-wise scales, a Triton dequant-fused GEMM, AWQ-format weight loading, perplexity-based accuracy validation.

**Critical framing (must appear in the report):** W4A16 reduces *memory traffic*, not FLOPs. It gives ~4× decode gain in the memory-bound regime where dequant cost hides behind HBM latency — and **zero prefill gain**, where dequant lands directly on the critical path. Measuring and explaining this asymmetry is the deliverable, not just the speedup.

**Exit:** Decode speedup measured across batch sizes with the crossover point identified; perplexity delta < 0.5 on WikiText-2.

---

### S10 — Tensor Parallelism
**Owner:** A6 · **Depends on:** S1, S3 · **Phase:** 6

Column-parallel QKV + gate/up projections, row-parallel O + down projections, NCCL all-reduce after attention and MLP blocks, KV head sharding under GQA, sharded weight loading, rank-0 sampling.

**Invariants:** I10.1: TP=2 output is token-identical to TP=1 under greedy decoding. I10.2: KV heads divide evenly across ranks (assert at load).

**Exit:** TP=2 and TP=4 correct and benchmarked; scaling efficiency reported with communication overhead isolated; NVLink vs PCIe compared.

---

### S11 — PD Disaggregation
**Owner:** A6 · **Depends on:** S10 · **Phase:** 6

Separate prefill and decode worker pools, KV cache transfer over NVLink/RDMA with transport auto-selection, a router assigning requests to pools, goodput measurement vs the colocated baseline.

**Motivating data:** prefill is compute-bound and draws 70–100% of peak GPU power; decode is memory-bound at 20–40%. Colocation causes measurable interference. Choosing RDMA where NVLink was available wastes ~18× bandwidth — transport selection is a first-class concern.

**Exit:** Working 2-node disaggregated deployment; goodput vs colocated reported with KV transfer overhead isolated as its own measured term.

---

### S12 — Observability & Demo
**Owner:** A5 · **Depends on:** S4 · **Phase:** 3, 6

Per-step metrics (TTFT, TPOT, ITL, queue depth, batch size, KV utilization, preemption count, cache hit rate, MFU, achieved bandwidth), Prometheus `/metrics`, WebSocket stream, and a **React dashboard whose centerpiece is a live block-table visualization** showing physical blocks allocating, sharing (color-coded by refcount), and recycling in real time.

**Why this matters (do not deprioritize):** the block visualization is the single artifact that makes an invisible memory-management idea legible to an evaluator who does not know what a KV cache is. It is worth more marks than three additional optimizations. It is a **Phase 3 deliverable, not a Phase 6 nice-to-have.**

**Exit:** Dashboard runs live during a benchmark with <2% overhead; a non-expert can watch it and correctly describe what paging does.

---

### S13 — Benchmark Harness
**Owner:** A7 · **Depends on:** S7 · **Phase:** 2 onward

Workload generator (Poisson arrivals, configurable prompt/output length distributions, shared-prefix ratio), baseline runners (HF, vLLM, SGLang), metric collection, statistical analysis, plot generation, `run_manifest.json` environment capture, results database.

**Workload profiles (frozen in Phase 2):**

| Profile | Prompt len | Output len | Prefix share | Models |
|---|---|---|---|---|
| **W1 — Chat** | ~256 (lognormal) | ~256 (lognormal) | 20% | Interactive |
| **W2 — RAG** | ~4096 (heavy tail) | ~128 | 60% | Long-context |
| **W3 — Agent** | ~2048 | ~64 | 90% | Prefix-cache stress |
| **W4 — Heterogeneous** | bimodal 128/8192 | bimodal 32/1024 | 30% | **Scheduler stress — the research workload** |
| **W5 — Sharegpt trace** | real | real | real | External validity |

**Exit:** One command reproduces every published figure from scratch.

---

### S14 — Correctness Infrastructure
**Owner:** A7 · **Depends on:** all · **Phase:** 1 onward

Numerical parity harness, deterministic replay (seeded RNG + recorded scheduler decisions), invariant assertion framework toggled by `SLIPSTREAM_DEBUG`, chaos testing (random preemption injection), CI gates.

**Exit:** CI runs the full parity + fuzz suite on every PR; nightly long-run soak test with leak detection.

---

### S15 — Horizon Scheduler (Research)
**Owner:** A8 · **Depends on:** S4 (Protocol only) · **Phase:** 4, 5

Length predictor (feature sets `f0`–`f3`), memory-aware admission, SRPT-approximate ordering with a starvation guard, measured swap-vs-recompute decision rule, oracle scheduler for the upper bound, full ablation matrix, paper draft.

**Exit:** See §5.3 criteria.

---

## 10. Phase Plan

Sixteen weeks, six phases. **Every gate is binding.**

---

### PHASE 0 — Foundation & Contracts (Week 1)
*Agents: 4 · The most important week in the project.*

Nothing is built this week except the things that let everything else be built in parallel.

- Repo scaffolding, CI, lint, test harness skeleton, pre-commit hooks
- **Freeze all interface contracts from §8.3** — dataclasses, Protocols, type stubs with `NotImplementedError` bodies
- Environment provisioning on T0 + cluster access verified on T1
- Model weights downloaded and cached; HF reference baseline captured
- Benchmark harness skeleton + `run_manifest.json` capture
- Literature review completed and summarized into `docs/related_work.md`

**Gate 0:** `pytest` runs green on empty stubs. Every agent can import every interface they depend on. HF baseline numbers recorded for Qwen2.5-0.5B on T0. **No agent is blocked on another agent's implementation.**

---

### PHASE 1 — Correct Engine (Weeks 2–3)
*Agents: 4*

A slow, correct, single-sequence engine. Speed is explicitly forbidden as a goal this phase.

- Llama/Qwen architecture, weight loading, RoPE, GQA, RMSNorm, SwiGLU
- Naive contiguous KV cache
- Sampler (greedy + temperature + top-k/top-p)
- Numerical parity harness vs HuggingFace
- Baseline measurement infrastructure

**Gate 1:**
- Greedy output **token-identical** to HF for 50 prompts × 128 tokens on Qwen2.5-0.5B and Llama-3.1-8B
- Logit parity within `atol=1e-2`
- Baseline throughput recorded (this becomes the "naive" column in every future table)

---

### PHASE 2 — Paged Memory & Kernels (Weeks 4–7)
*Agents: 5 · The hardest stretch. Budget accordingly.*

- Block manager: pool, block tables, refcounting, COW, free list
- Triton `paged_attention_decode`, `paged_attention_prefill`, `reshape_and_cache`
- Continuous batching scheduler (FCFS mechanism)
- Property-based fuzz suite for the allocator
- Kernel microbenchmark suite
- Workload generator + vLLM baseline harness

**Gate 2:**
- Allocator fuzz suite passes 10k random op sequences with zero invariant violations
- Paged attention matches reference within `atol=1e-2` across the full shape sweep
- **Token-identical output to Phase 1** (paging must be semantically invisible)
- Continuous batching ≥ **3×** static batching throughput
- KV memory waste < **5%** (vs > 60% naive) — measured, not asserted
- Decode kernel ≥ **70%** of achievable memory bandwidth on A100

> ⚠️ **This is the phase where projects die.** Paged attention bugs are silent — output stays plausible while being wrong. Gate 2's token-identity requirement is the tripwire. Do not weaken it.

---

### PHASE 3 — Serving & Advanced Scheduling (Weeks 8–10)
*Agents: 5*

- Chunked prefill with Sarathi-Serve stall-free scheduling
- Prefix cache: radix tree, content hashing, COW sharing, LRU eviction
- FastAPI server, SSE streaming, cancellation, OpenAI compatibility
- **Live dashboard with block-table visualization**
- Preemption: swap-to-CPU and recompute modes

**Gate 3:**
- Chunked prefill cuts decode ITL p99 by ≥ **40%** under mixed load
- Prefix cache: ≥ **60%** hit rate, ≥ **50%** TTFT cut on W3; output token-identical to cold cache
- `openai` client works unmodified
- Dashboard live during benchmark, <2% overhead
- Preempt/resume produces identical output
- **First full comparison table published: Slipstream vs HF vs vLLM on W1**

---

### PHASE 4 — Optimization & The Research Contribution (Weeks 11–13)
*Agents: 6 · Peak parallelism.*

- CUDA graph capture/replay for decode; persistent-kernel refactor for graph compatibility
- Fused kernels: RMSNorm, RoPE, SwiGLU
- EngineCore process isolation
- **Horizon scheduler: predictor, admission policy, SRPT ordering, preemption rule**
- **Oracle scheduler (upper bound)**
- Roofline + MFU analysis
- Full ablation matrix

**Gate 4:**
- CUDA graphs cut decode step time ≥ **20%** at batch ≤ 32
- Fused kernels each show a measured win, or are **removed** (no unjustified complexity)
- Horizon achieves ≥ **15%** goodput gain over FCFS on W4, or a rigorous negative result with oracle bound
- Predictor overhead < **0.5%** of step time
- Ablation table complete: every optimization has a number attached
- Roofline analysis demonstrates the memory-bound decode regime empirically

---

### PHASE 5 — Speculation & Quantization (Week 14)
*Agents: 4*

- Draft-model speculative decoding with distribution-preserving verification
- W4A16 quantization with Triton dequant-fused GEMM
- FP8 experiments on H100 *(stretch)*

**Gate 5:**
- Speculative decoding: ≥ **1.5×** decode speedup at low batch; distribution test passes (chi-squared, 10k samples); the high-batch regime where it *hurts* is documented
- W4A16: decode speedup + prefill non-speedup both measured and explained; perplexity delta < 0.5

---

### PHASE 6 — Distributed & Delivery (Weeks 15–16)
*Agents: 5*

- Tensor parallelism (TP=2, TP=4)
- PD disaggregation across 2 nodes
- Final benchmark sweep on T1/T2/T3
- Report, paper, blog series, demo rehearsal

**Gate 6:**
- TP=2 token-identical to TP=1; scaling efficiency reported with comm overhead isolated
- PD disaggregation running; goodput vs colocated measured; KV transfer cost isolated
- **≥85% of vLLM throughput on W1/A100**
- **≥10× HuggingFace throughput**
- All Definition-of-Done items (§6.3) checked

---

### Phase summary

| Phase | Weeks | Agents | Theme | Risk |
|---|---|---|---|---|
| 0 | 1 | 4 | Contracts | Low |
| 1 | 2–3 | 4 | Correctness | Low |
| 2 | 4–7 | 5 | **Paged memory + kernels** | 🔴 **Critical** |
| 3 | 8–10 | 5 | Serving + scheduling | Medium |
| 4 | 11–13 | 6 | **Optimization + research** | 🟠 High |
| 5 | 14 | 4 | Speculation + quant | Medium |
| 6 | 15–16 | 5 | Distributed + delivery | Medium |

**Scope ladder (if behind schedule, cut in this order):**
`FP8 → PD disaggregation → tensor parallelism → quantization → speculative decoding → advanced kernels`

**Never cut:** paged memory, continuous batching, chunked prefill, prefix cache, Horizon scheduler, benchmark rigor, the dashboard. **Phases 0–4 alone constitute a complete, defensible, excellent project.**

---

## 11. Agent Organization & Work Division

### 11.1 Operating model

Work is executed by **specialist agents** running in parallel, coordinated through frozen interface contracts. The core principle:

> **Agents never wait on each other's implementations — only on each other's interfaces, which are frozen in Phase 0.**

This is what makes 5–6 way parallelism actually work rather than degenerate into merge conflicts and blocked handoffs.

### 11.2 Agent roster

| ID | Agent | Charter | Owns |
|---|---|---|---|
| **A1** | **Runtime** | Model architecture, weights, numerics, sampling | S1, S6 |
| **A2** | **Memory** | Block manager, paged KV, prefix cache, swap | S2, S5 |
| **A3** | **Kernel** | All Triton/CUDA kernels, quantization kernels | S3, S9 |
| **A4** | **Scheduler** | Batching mechanism, chunked prefill, preemption | S4, S8 |
| **A5** | **Serving** | API, streaming, observability, dashboard | S7, S12 |
| **A6** | **Distributed** | Tensor parallelism, PD disaggregation, NCCL | S10, S11 |
| **A7** | **Verification** | Tests, CI, benchmark harness, statistical analysis | S13, S14 |
| **A8** | **Research** | Horizon scheduler, predictor, ablations, paper | S15 |

### 11.3 Phase-by-phase deployment

**PHASE 0 — 4 agents**

| Agent | Task | Output |
|---|---|---|
| A7 | Repo scaffold, CI, pytest, pre-commit, manifest capture | `ci/`, `tests/conftest.py` |
| A1 | **Freeze all §8.3 contracts** as typed stubs | `slipstream/core/types.py` |
| A5 | Environment provisioning, cluster access, weight caching | `docs/environment.md` |
| A8 | Literature review, baseline HF measurement | `docs/related_work.md` |

*Sequencing note: A1's contract freeze is the critical path. It ships by day 3 or the whole schedule slips.*

**PHASE 1 — 4 agents**

| Agent | Task |
|---|---|
| A1 | Llama/Qwen architecture, weight loading, RoPE, GQA |
| A6 | Sampler (greedy, temp, top-k/p), stop criteria |
| A7 | HF parity harness, logit comparison, greedy-identity test |
| A2 | Naive contiguous KV cache (throwaway — the baseline we beat) |

**PHASE 2 — 5 agents** *(critical phase; A2 and A3 are the critical path)*

| Agent | Task |
|---|---|
| A2 | Block manager, block tables, refcount, COW, free list |
| A3 | `paged_attention_decode`, `paged_attention_prefill`, `reshape_and_cache` |
| A4 | Continuous batching scheduler, FCFS policy, batch assembly |
| A7 | Allocator fuzz suite, kernel parity sweep, microbenchmarks |
| A8 | Workload generator, vLLM/SGLang baseline harness |

*A2 → A3 dependency is resolved by A2 shipping the memory **layout spec** on day 1 of the phase, before the implementation. A3 codes against the layout, not the allocator.*

**PHASE 3 — 5 agents**

| Agent | Task |
|---|---|
| A4 | Chunked prefill, stall-free scheduling, preemption modes |
| A2 | Radix prefix tree, content hashing, COW sharing, LRU eviction |
| A5 | FastAPI server, SSE streaming, cancellation, OpenAI compat |
| A1 | **Dashboard + block-table visualization** |
| A7 | Prefix-cache identity tests, stall-free assertions, W1 comparison table |

**PHASE 4 — 6 agents** *(peak parallelism)*

| Agent | Task |
|---|---|
| A3 | CUDA graphs, persistent kernels, fused RMSNorm/RoPE/SwiGLU |
| A8 | **Horizon: predictor, admission, SRPT ordering, preemption rule** |
| A4 | Policy plumbing, oracle scheduler, config-driven policy swap |
| A7 | Ablation matrix, roofline + MFU analysis, statistical framework |
| A5 | EngineCore process isolation, metrics pipeline |
| A6 | Begin TP groundwork (sharded weight loading) |

**PHASE 5 — 4 agents**

| Agent | Task |
|---|---|
| A4 | Speculative decoding: draft loop, verification, adaptive draft length |
| A3 | W4A16 Triton dequant-fused GEMM, AWQ loading, FP8 stretch |
| A7 | Distribution-preservation test, perplexity validation |
| A8 | Paper draft, results consolidation |

**PHASE 6 — 5 agents**

| Agent | Task |
|---|---|
| A6 | Tensor parallelism, NCCL, PD disaggregation, KV transfer |
| A7 | Final benchmark sweep across T1/T2/T3, reproducibility audit |
| A8 | Paper finalization, ablation writeup |
| A5 | Demo polish, README, blog series |
| A1 | Academic report, architecture documentation |

### 11.4 Coordination protocol

**Interface changes.** Any change to a §8.3 contract requires: (1) an amendment entry in §21, (2) notification to every dependent agent, (3) a single integration commit. Contracts do not drift silently.

**Branching.** `main` is always green. Each agent works on `agent/<id>/<feature>`. PRs require CI green + one review. Integration branches per phase: `phase/<n>`.

**Ownership.** Every file has exactly one owning agent (declared in `CODEOWNERS`). Cross-boundary edits require the owner's review. This eliminates the merge-conflict failure mode that kills parallel agent work.

**Daily sync artifact.** Each agent appends to `docs/log/<phase>.md`: what shipped, what is blocked, what interface assumptions were made. Blocked items surface within 24 hours or the parallelism is fake.

**Definition of "done" for an agent task.** Code + tests + benchmark (if perf-relevant) + docstring + an entry in the phase log. Untested code is not done. Unbenchmarked optimizations are not done.

**Anti-patterns to actively police:**
- ❌ An agent stubbing another agent's module locally instead of using the frozen contract → divergence
- ❌ Optimizing before Gate 2 → premature and unmeasurable
- ❌ Adding an optimization without an ablation number → unjustified complexity, must be removed
- ❌ Weakening a gate to stay on schedule → cut scope instead (§10 ladder)

---

## 12. Repository Structure

```
slipstream/
├── MASTERPLAN.md                  ← this document (source of truth)
├── README.md                      ← public face: results table + GIF up top
├── CODEOWNERS                     ← agent file ownership
├── pyproject.toml
│
├── slipstream/
│   ├── core/
│   │   ├── types.py               ← FROZEN CONTRACTS (§8.3)
│   │   ├── sequence.py
│   │   ├── config.py
│   │   └── sampling_params.py
│   ├── memory/
│   │   ├── block_manager.py       ← A2
│   │   ├── block_table.py
│   │   ├── prefix_cache.py        ← radix tree
│   │   └── swap.py
│   ├── kernels/
│   │   ├── paged_attention.py     ← A3
│   │   ├── reshape_and_cache.py
│   │   ├── fused_rmsnorm.py
│   │   ├── fused_rope.py
│   │   ├── fused_swiglu.py
│   │   ├── quant_gemm.py
│   │   └── autotune_configs/      ← per-arch tuned configs
│   ├── scheduler/
│   │   ├── scheduler.py           ← A4 mechanism
│   │   ├── policies/
│   │   │   ├── base.py            ← SchedulingPolicy Protocol
│   │   │   ├── fcfs.py
│   │   │   ├── horizon.py         ← ⭐ A8 research
│   │   │   └── oracle.py          ← upper bound
│   │   └── predictor/
│   │       ├── features.py
│   │       └── length_model.py
│   ├── models/
│   │   ├── llama.py               ← A1
│   │   ├── qwen.py
│   │   ├── layers/
│   │   └── loader.py
│   ├── engine/
│   │   ├── engine_core.py         ← isolated process
│   │   ├── model_runner.py
│   │   ├── cuda_graph.py
│   │   └── llm_engine.py
│   ├── distributed/
│   │   ├── tensor_parallel.py     ← A6
│   │   ├── communication.py
│   │   └── disaggregated/
│   ├── speculative/
│   ├── entrypoints/
│   │   ├── api_server.py          ← A5
│   │   └── openai_protocol.py
│   └── observability/
│       ├── metrics.py
│       └── ws_stream.py
│
├── dashboard/                     ← React block-table visualizer
│
├── tests/
│   ├── correctness/               ← HF parity, greedy identity
│   ├── property/                  ← hypothesis fuzz (allocator)
│   ├── kernels/                   ← per-kernel parity sweeps
│   ├── integration/
│   └── chaos/                     ← random preemption injection
│
├── benchmarks/
│   ├── workloads/                 ← W1–W5 generators
│   ├── baselines/                 ← HF, vLLM, SGLang runners
│   ├── analysis/                  ← roofline, MFU, Pareto plots
│   └── results/                   ← versioned results + manifests
│
├── docs/
│   ├── related_work.md
│   ├── architecture.md
│   ├── environment.md
│   ├── benchmarking_protocol.md
│   ├── log/                       ← per-phase agent logs
│   └── paper/                     ← LaTeX source
│
└── ci/
```

---

## 13. Engineering Standards

**Language & style.** Python 3.11, full type hints on all public APIs, `ruff` + `mypy` in CI. Kernels in Triton; two hand-written CUDA C++ kernels in Phase 5 for the résumé line only.

**Commits.** Conventional commits (`feat:`, `fix:`, `perf:`, `test:`, `docs:`, `bench:`). Every `perf:` commit body **must** include before/after numbers with hardware stated. A `perf:` commit without numbers is rejected in review.

**Testing rules.**
- No PR merges with failing tests
- Every kernel ships with a parity test *in the same PR*
- Every allocator change re-runs the fuzz suite
- Every optimization ships with an ablation entry

**Benchmark hygiene (non-negotiable).**
- Warmup iterations discarded (≥10)
- ≥3 independent runs; report median and p99, never a single best
- Exclusive GPU allocation; record clock lock state
- Full environment captured in `run_manifest.json`
- Baselines run on **identical** hardware, same session, same model weights
- Any number in the README traces to a committed manifest

**Documentation.** Every module carries a docstring stating its invariants. Every kernel documents its memory access pattern and expected arithmetic intensity. `docs/architecture.md` stays current — it is a Gate item, not an afterthought.

---

## 14. Correctness Strategy

Correctness is the existential risk of this project. Paged attention, prefix caching, and speculative decoding all fail *silently* — output remains fluent while being wrong. The strategy is layered defense.

### Layer 1 — Numerical parity (continuous)
Every engine change re-runs greedy-identity against HuggingFace on a fixed prompt set. **Token divergence at any position fails CI.** This single test catches the majority of paged-attention bugs.

### Layer 2 — Semantic invariance (per feature)
Each optimization must be *semantically invisible*:

| Optimization | Invariance test |
|---|---|
| Paging | Output identical to contiguous cache |
| Prefix cache | Output identical to cold cache |
| Chunked prefill | Output identical to full prefill |
| Preemption | Output identical to uninterrupted run |
| CUDA graphs | Output identical to eager |
| Tensor parallelism | TP=N identical to TP=1 (greedy) |
| Speculative decoding | Output *distribution* statistically identical |

### Layer 3 — Property-based invariants
Hypothesis-driven fuzzing of the allocator with an invariant checker after every operation (§S2). 10k random op sequences per CI run.

### Layer 4 — Debug assertions
`SLIPSTREAM_DEBUG=1` enables: refcount conservation checks, block-table bounds checks in kernels, token-budget assertions, free-list integrity. Off in benchmarks, on in all tests.

### Layer 5 — Chaos testing
Random preemption injection, random swap-out, artificial memory pressure, request cancellation storms — all asserting output identity.

### Layer 6 — Soak testing
Nightly 10k-request run with leak detection: free-block count must return exactly to its initial value.

**The rule that prevents disaster:** *no optimization is merged before its invariance test exists.* Test first, optimize second. This is inverted from normal practice deliberately, because the failure mode here is silent.

---

## 15. Benchmarking Methodology

### 15.1 Metrics

| Metric | Definition | Why |
|---|---|---|
| **TTFT** | Arrival → first token, including queue wait | Interactive responsiveness |
| **TPOT / ITL** | Mean / inter-token latency during decode | Streaming smoothness |
| **Throughput** | Output tokens/sec, system-wide | Cost efficiency |
| **Goodput** | Requests/sec meeting *both* TTFT and TPOT SLOs | **The metric that matters** |
| **Smooth goodput** | Gameability-resistant goodput variant | Methodological integrity |
| **KV utilization** | Useful KV bytes / allocated KV bytes | Memory efficiency |
| **MFU** | Achieved FLOPs / peak FLOPs | Compute efficiency |
| **Achieved bandwidth** | Bytes moved / peak HBM bandwidth | **The real decode metric** |
| **Preemption rate** | Preemptions/sec | Scheduler health |
| **Cache hit rate** | Prefix-cached tokens / total prompt tokens | Prefix cache value |

### 15.2 Methodological commitments

**On goodput gameability.** The literature documents that naive goodput is gameable — a system can delay token delivery to smooth tail latency, or drop requests about to miss SLOs, and score better while serving users worse. Slipstream therefore: (a) reports smooth goodput alongside naive, (b) reports the full latency distribution, not just SLO attainment, (c) reports dropped/aborted request counts explicitly, and (d) never tunes against goodput alone.

**On the latency-throughput Pareto frontier.** Single-point comparisons are meaningless — any system can trade latency for throughput. Every headline comparison is reported as a **Pareto curve** sweeping request rate, not a single number.

**On statistical rigor.** ≥3 runs, median reported, p50/p95/p99 for latencies, confidence intervals on all headline claims, and outliers investigated rather than discarded.

**On honest baselines.** vLLM and SGLang are run with tuning effort **comparable to our own** — a deliberately misconfigured baseline is scientific misconduct and will be spotted in a viva. Baseline configurations are committed to the repo.

**On negative results.** Optimizations that do not help are reported, not buried. The regimes where speculative decoding hurts, where quantization loses, and where Horizon fails to beat FCFS are first-class results.

### 15.3 The canonical results table

Every headline comparison takes this form:

| System | Config | Throughput (tok/s) | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | Goodput (req/s) | KV util | MFU |
|---|---|---|---|---|---|---|---|
| HF `generate()` | static batch | — | — | — | — | — | — |
| Slipstream | naive (P1) | | | | | | |
| Slipstream | +paged+CB | | | | | | |
| Slipstream | +chunked+prefix | | | | | | |
| Slipstream | +graphs+fused | | | | | | |
| Slipstream | **+Horizon** | | | | | | |
| vLLM | tuned | | | | | | |
| SGLang | tuned | | | | | | |

*Stated for a fixed (model, hardware, workload, request rate). Repeated per W1–W5.*

---

## 16. Performance Model & Numeric Targets

### 16.1 Roofline analysis (the analytical spine of the report)

| GPU | Peak BW | Peak dense FLOPS | Ridge point (FLOP/byte) |
|---|---|---|---|
| A100-80GB SXM | 2039 GB/s | 312 TF bf16 | **~153** |
| H100 SXM | 3350 GB/s | 989 TF bf16 | **~295** |
| V100-32GB | 900 GB/s | 125 TF fp16 | **~139** |
| RTX 3050 Laptop | ~140 GB/s | ~9 TF fp16 | ~64 |

**The central analytical result to demonstrate empirically:**

During decode, per weight byte read, the engine performs ~2 FLOPs per sequence in the batch. Therefore:

```
arithmetic_intensity(decode) ≈ batch_size
```

Which means: **on an A100, decode remains memory-bandwidth-bound until batch size ≈ 153.** Below that, the GPU is starved — you are paying for compute you cannot use. This single fact explains why PagedAttention (which raises achievable batch size) is worth more than any kernel optimization, and it is the analytical justification for the entire project structure.

**Deliverable:** an empirical roofline plot showing measured decode operating points converging on the bandwidth roof, with the transition to compute-bound visible at high batch. This plot is the intellectual centerpiece of the report.

### 16.2 Committed targets

| # | Target | Threshold | Phase |
|---|---|---|---|
| T1 | Greedy token-identity vs HF | 100% | 1 |
| T2 | KV memory waste | < 5% | 2 |
| T3 | Continuous batching vs static | ≥ 3× | 2 |
| T4 | Decode kernel achieved bandwidth (A100) | ≥ 70% | 2 |
| T5 | Chunked prefill decode ITL p99 cut | ≥ 40% | 3 |
| T6 | Prefix cache hit rate (W3) | ≥ 60% | 3 |
| T7 | Prefix cache TTFT cut (W3) | ≥ 50% | 3 |
| T8 | CUDA graph decode step-time cut (batch ≤ 32) | ≥ 20% | 4 |
| T9 | **Horizon goodput gain over FCFS (W4)** | **≥ 15%** | 4 |
| T10 | Predictor overhead | < 0.5% step time | 4 |
| T11 | Speculative decoding speedup (low batch) | ≥ 1.5× | 5 |
| T12 | W4A16 perplexity delta (WikiText-2) | < 0.5 | 5 |
| T13 | TP=2 token-identity vs TP=1 | 100% | 6 |
| T14 | **Throughput vs vLLM (W1, A100)** | **≥ 85%** | 6 |
| T15 | **Throughput vs HF** | **≥ 10×** | 6 |

---

## 17. Risk Register

| # | Risk | P | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| **R1** | **Silent paged-attention bug** — output plausible but wrong | High | Critical | Gate 2 token-identity requirement; per-kernel parity sweeps; bounds-checked debug builds; test-before-optimize rule | A3, A7 |
| **R2** | **Phase 2 overruns** — kernels + allocator take longer than 4 weeks | High | High | Phase 2 has the largest budget already; A2 ships layout spec day 1 to unblock A3; scope ladder (§10) cuts Phase 5–6 first, never Phase 2 | A2, A3 |
| **R3** | **Horizon shows no gain** | Medium | Medium | Negative result is publishable **by design** (§5.3 fallback); oracle bound frames the result; secondary contribution (preemption policy) is independent and high-confidence | A8 |
| **R4** | **"This is just nano-vLLM"** critique | Medium | High | Four hard differentiators (§4.2); positioning statement rehearsed for viva; kernels + research + rigor + distributed are all absent from nano-vLLM | Owner |
| **R5** | **Faculty cannot evaluate the work** | High | High | Lead with cost/throughput framing, not mechanism; block-table visualization as Phase 3 deliverable; demo script rehearsed for a non-expert audience | A5, Owner |
| **R6** | **Cluster contention / queue waits** | Medium | Medium | T0 laptop for all correctness work; batch benchmark jobs; reserve exclusive slots early; never block dev on cluster availability | A7 |
| **R7** | **Baseline comparison challenged as unfair** | Medium | High | Commit baseline configs to repo; tune baselines with equal effort; document tuning process; report vLLM's own published numbers as cross-check | A7 |
| **R8** | **V100 Triton incompatibility** | Medium | Low | V100 results declared a bonus from the start; no deliverable depends on T3; fp16-only path tested separately | A6 |
| **R9** | **Agent interface drift** | Medium | High | Contracts frozen Phase 0; `CODEOWNERS`; amendment log (§21); daily sync artifact surfaces assumptions within 24h | All |
| **R10** | **Scope creep into non-goals** | High | Medium | §6.2 non-goals list is binding; every new feature needs an ablation number to justify existing | Owner |
| **R11** | **Speculative decoding distribution bug** | Medium | High | Chi-squared distribution test as a merge gate; verification math reviewed against the reference derivation before implementation | A4, A7 |
| **R12** | **Benchmark irreproducibility** | Medium | High | `run_manifest.json` on every run; single-command reproduction as a Gate 6 item; results versioned in-repo | A7 |

---

## 18. Academic Deliverables

### 18.1 Report structure

1. Introduction — the economics of LLM serving; why memory is the bottleneck
2. Background — transformer inference, prefill vs decode, the KV cache
3. Related work — vLLM, Orca, Sarathi-Serve, SGLang, DistServe, nano-vLLM (positioned honestly)
4. System design — architecture, paged memory, scheduling, kernels
5. Implementation — Triton kernels, block manager, engine core
6. **The Horizon scheduler** — problem, method, oracle bound
7. Evaluation — methodology, results, ablations, roofline analysis
8. Discussion — negative results, limitations, threats to validity
9. Future work — multi-tier KV, EAGLE-3, adaptive PD
10. Conclusion

### 18.2 Paper

Target: an MLSys/EuroSys/HotOS-adjacent workshop, or arXiv preprint. Scope: the Horizon contribution plus the engine as the experimental substrate. A workshop paper on a minor project is an outsized differentiator and is achievable if Phase 4 lands.

### 18.3 The demo (≤10 minutes, rehearsed)

1. **Open with cost, not mechanism.** "Serving an LLM is dominated by GPU cost. Standard implementations waste 60–80% of the memory that determines how many users you can serve. We rebuilt that memory system." *(R5 mitigation — this framing is mandatory.)*
2. Split screen: HuggingFace vs Slipstream, same GPU, same 64 concurrent requests. Live throughput counters diverge.
3. **The block-table visualization.** Blocks allocating, sharing (color-coded by refcount), recycling. This is the moment the idea becomes visible.
4. Turn on prefix caching live with a shared system prompt — watch TTFT collapse.
5. Inject a heterogeneous burst; switch FCFS → Horizon live; watch goodput rise and the SLO-violation counter drop.
6. Close on the results table vs vLLM.

### 18.4 Viva preparation

Be ready to answer, from first principles:
- Why is decode memory-bound? Derive the arithmetic intensity.
- Why block size 16? What breaks at 4 and at 256?
- Why does chunked prefill help latency but cost throughput?
- Swap or recompute on preemption — and what determines the crossover?
- Why does INT4 help decode but not prefill?
- How does speculative decoding preserve the output distribution?
- Where does Horizon fail, and why?
- What is your weakest result, and what would you do with another month?

*The last two matter most. Confident ownership of limitations is what separates a strong viva from a defensive one.*

---

## 19. Career Positioning

**Résumé entry:**

> **Slipstream — LLM Inference Engine** · *Python, Triton, CUDA, PyTorch*
> Built a high-throughput LLM inference engine from scratch implementing PagedAttention, continuous batching, chunked prefill, and RadixAttention-style prefix caching. Wrote custom Triton kernels for paged attention and fused transformer ops reaching ≥70% of peak HBM bandwidth on A100. Achieved 85% of vLLM's throughput and 10× HuggingFace Transformers, cutting KV-cache memory waste from >60% to <5%. Designed a novel output-length-predictive scheduler improving SLO goodput 15% over FCFS under memory constraints; extended to tensor-parallel and prefill/decode-disaggregated multi-GPU deployment.

**Target companies:** Anthropic, OpenAI, Together AI, Fireworks, Baseten, Modal, Groq, NVIDIA, Databricks, Perplexity, plus every infrastructure team at scale.

**Content strategy — publish as you build, not after:**
1. "Why LLM decode is memory-bound" — the roofline post
2. "Writing paged attention in Triton" — the kernel post
3. "Continuous batching from scratch" — the scheduler post
4. "Your scheduler doesn't know how long requests will run" — the research post
5. "Benchmarking inference engines honestly" — the methodology post

The methodology post is the sleeper. Rigor is rarer than capability, and it is the trait that gets senior engineers to pay attention.

---

## 20. Reading List

**Tier 1 — required before Phase 2**
- *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et al., SOSP 2023) — the vLLM paper
- *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — continuous batching
- *Taming Throughput-Latency Tradeoff with Sarathi-Serve* (Agrawal et al., OSDI 2024) — chunked prefill
- *FlashAttention-2* (Dao, 2023)
- vLLM V1 architecture blog (2025) — EngineCore, unified scheduler
- vLLM Triton Attention Backend Deep Dive (Mar 2026) — Q blocks, persistent kernels, autotuning

**Tier 2 — before Phase 3–4**
- *SGLang: Efficient Execution of Structured LM Programs* — RadixAttention
- *Online Scheduling for LLM Inference with KV Cache Constraints* (Jaillet et al.)
- *Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference* (2026)
- *On Evaluating Performance of LLM Inference Serving Systems* — the gameability paper
- *Taming the Titans: A Survey of Efficient LLM Inference Serving*

**Tier 3 — before Phase 5–6**
- *DistServe* — PD disaggregation for goodput
- *Splitwise* — phase splitting on heterogeneous hardware
- *FlashAttention-3* — Hopper: TMA, wgmma, FP8
- *EAGLE-3* — speculative decoding SOTA
- *LMCache* — multi-tier KV offloading
- Megatron-LM — tensor parallelism

**Code to read (not copy):** nano-vLLM (orientation), vLLM V1 `scheduler.py` + `block_manager.py`, SGLang radix cache, vLLM Triton attention backend.

---

## 21. Amendment Log

| Date | Section | Change | Rationale |
|---|---|---|---|
| 2026-08-13 | — | Initial version | — |

*All contract changes and target revisions are recorded here. A target missed without an amendment entry is a failure, not a revision.*

---

## 22. Glossary

| Term | Definition |
|---|---|
| **Arithmetic intensity** | FLOPs performed per byte moved from memory. Determines whether a kernel is compute- or bandwidth-bound. |
| **Block table** | Per-sequence map from logical block index → physical KV block ID. The "page table" of PagedAttention. |
| **Chunked prefill** | Splitting a long prompt's prefill across iterations so it can be interleaved with ongoing decodes. |
| **Continuous batching** | Iteration-level scheduling: sequences join and leave the batch every forward pass. |
| **COW** | Copy-on-write. A shared KV block is duplicated when one sharer writes to it. |
| **Decode** | Autoregressive phase; one token per sequence per pass. Memory-bandwidth-bound. |
| **Goodput** | Requests/sec meeting their SLOs. Throughput filtered by quality. |
| **GQA** | Grouped-query attention. Multiple Q heads share one KV head, shrinking the KV cache. |
| **ITL** | Inter-token latency. Gap between consecutive streamed tokens. |
| **MFU** | Model FLOPs Utilization. Achieved FLOPs / peak FLOPs. |
| **Non-clairvoyant** | Scheduling without knowing job sizes in advance. |
| **PagedAttention** | KV cache managed as fixed-size blocks with an indirection table, à la OS virtual memory. |
| **PD disaggregation** | Running prefill and decode on separate GPU pools. |
| **Prefill** | Processing the input prompt. Compute-bound, parallel over all prompt tokens. |
| **Ridge point** | Arithmetic intensity where a roofline transitions from bandwidth- to compute-bound. |
| **Slot mapping** | Flat index (`block_id × block_size + offset`) telling a kernel where to write each token's KV. |
| **SRPT** | Shortest Remaining Processing Time. Latency-optimal ordering; unfair without a starvation guard. |
| **TPOT** | Time Per Output Token. |
| **TTFT** | Time To First Token, including queue wait. |
| **W4A16** | 4-bit weights, 16-bit activations. Cuts memory traffic, not FLOPs. |

---

*End of master plan. Amendments go in §21.*
