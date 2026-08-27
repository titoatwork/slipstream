# Related work

Phase 0 literature review (A8). This is the substrate for the report §3 and the Horizon paper. It is a reading record, not a claim of novelty.

Positioning one-liner (MASTERPLAN §4.2): *nano-vLLM demonstrates that these ideas work. Slipstream investigates how well they work, why they work at the hardware level, and where the remaining policy headroom is.*

---

## 1. Memory system

**PagedAttention — Kwon et al., SOSP 2023 (vLLM).**
Treats the KV cache as virtual memory: fixed-size blocks, a per-sequence block table, on-demand allocation. Removes the 60–80% internal fragmentation of `max_seq_len`-sized contiguous buffers. This is the single highest-leverage engineering idea in the stack. We reimplement it (S2) because Horizon needs a memory allocator we fully control.

**Implication for us:** block size 16 is the default; 8/16/32 is an ablation, not a research question. Refcounting + COW are load-bearing for prefix sharing and `n>1` samples.

## 2. Batching and prefill/decode interference

**Orca — Yu et al., OSDI 2022.**
Iteration-level (continuous) batching: sequences join and leave the batch every forward pass. The difference versus static batching is the throughput gap we must reproduce (≥3×, Gate 2).

**Sarathi-Serve — Agrawal et al., OSDI 2024.**
Chunked prefill + stall-free schedule: admit *decodes first*, fill the remaining token budget with prefill chunks. Reported 2.6× serving capacity vs contemporaneous vLLM on Mistral-7B / A100. Chunk sizes 256–512 tokens hold tail latency; vLLM V1's online chunk budget is much larger (8192).

**Implication for us:** I4.2 (a running decode is never stalled by a newly admitted prefill) is a testable invariant, not a slogan. The 40% ITL p99 cut (T5) is the Gate 3 number.

## 3. Prefix reuse

**SGLang / RadixAttention.**
A radix tree over token sequences with block-level sharing. 75–95% hit rates on multi-turn / agent traffic; large TTFT cuts when prefixes overlap. Our S5 is this idea with COW + LRU of unreferenced blocks. Gate 3: ≥60% hit rate and ≥50% TTFT cut on W3, token-identical to a cold cache.

## 4. Attention kernels

**FlashAttention-2 (Dao, 2023) and FlashAttention-3.**
IO-aware tiled attention. FA3 is the Hopper baseline (TMA, wgmma, FP8); 75% of peak FP8 FLOPS is the published headline.

**vLLM Triton attention backend (Ringlein / IBM Research, vLLM blog 2026-03-04).**
~800 lines of Triton, one source across NVIDIA / AMD / Intel. **100.7% of FA3** on long decode on H100; ~5.8× over the prior impl on MI300. Two patterns we adopt: (1) group query tokens into **Q blocks** so `tl.dot` is utilized under GQA; (2) **persistent kernels** with a fixed launch grid so CUDA graphs can be reused. Variable launch grids conflict with graph capture.

**Implication for us:** a Triton-first strategy is not a toy compromise. Our decode kernel target is ≥70% of achievable HBM bandwidth on A100 (T4), not "beat FA3."

## 5. Serving architecture

**vLLM V1 (blog 2025-01-27; Ubicloud "life of a request", 2025).**
EngineCore in its own process; tokenizer / HTTP never on the GPU loop; ZMQ between API and core. Unified scheduler treats prompt and output tokens identically via a `{request_id: num_tokens}` budget, which is what makes chunked prefill, prefix cache, and speculation compose.

We copy the *process split* and the *token-budget* idea. We do **not** copy the FCFS policy. The policy is the research surface.

**nano-vLLM (~1.2k LOC).**
Educational reimplementation of PagedAttention + continuous batching. The identity trap. Our four differentiators (own Triton kernels, Horizon, measurement rigor, distributed layer) exist specifically so this project is not "nano-vLLM again."

## 6. The scheduling open problem

This is the research section.

**The structural fact.** A request's KV footprint grows linearly with tokens generated. Output length is unknown at admission. Admit too many → preemption cascades. Admit too few → the GPU sits below the roofline ridge (~153 concurrent sequences on A100) and throughput collapses.

**Jaillet, Jiang, Mellou, Molinaro, Podimata, Zhou — *Online Scheduling for LLM Inference with KV Cache Constraints* (arXiv:2502.07115, v5 Jan 2026).**
First clean theoretical model of online batching under a hard KV budget. Two results we must cite accurately:

- No deterministic online algorithm has a constant competitive ratio for *arbitrary* arrivals.
- Under structured arrival assumptions their MC-SF algorithm is O(1)-competitive. They also analyze the case of *inaccurate* output-length predictions (appendix B.3) — directly relevant to Horizon's predictor-quality threshold.

**Feng, Yang, Zhang — *Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference* (arXiv:2601.22996, Jan 2026).**
The sentence the master plan quotes is here, verbatim:

> obtaining robust performance guarantees without any prior knowledge of job sizes remains a theoretically fundamental and practically important open problem.

They give the first constant-competitive *non-clairvoyant* policy in the offline batch setting: Geometric Slicing (GSA), competitive ratio ≤ 61.92 (32 in the large-memory regime), via periodic restart to bound memory exposure plus a staggered pipeline. The clairvoyant counterpart (GBA) improves the previous approximation bound from >9000 down to 10.67 / 6.75.

**What this means for Horizon.** GSA is a worst-case algorithm that *restarts* jobs. That is the right object for a competitive-analysis paper and the wrong object for a serving engine — restarts destroy TTFT and waste prefill. Horizon is the complementary *systems* attack: a cheap online predictor + goodput-maximizing admission/preemption, evaluated empirically against FCFS and an oracle, with the accuracy threshold at which prediction stops paying for itself as a first-class result. We do not claim a better competitive ratio. We claim a measured goodput delta (or a rigorous negative) on W4.

Production engines (vLLM V1, SGLang) remain FCFS + greedy token budget. That is the baseline, not a straw man.

## 7. Measurement integrity

**Agrawal, Kedia, Agarwal, Mohan, Kwatra, Kundu, Ramjee, Tumanov — *On Evaluating Performance of LLM Inference Serving Systems* (arXiv:2507.09019, Jul 2025).**
Documents evaluation anti-patterns. Naive goodput is gameable: delay tokens to smooth a tail, or drop requests about to miss an SLO, and the number goes up while users do worse.

**Our commitments (MASTERPLAN §15.2):** report smooth goodput alongside naive; report the full latency distribution; report aborted/dropped counts; never tune against goodput alone; Pareto curves, not single points; ≥3 runs, median + p99.

## 8. Later phases (read before the phase, not now)

| Phase | Papers |
|---|---|
| 5 | EAGLE-3 (speculative decoding SOTA); AWQ / GPTQ / Marlin (W4A16). Draft-model speculation is in-scope; EAGLE-3 is stretch. |
| 6 | DistServe, Splitwise (PD disaggregation, up to 7.4× goodput); Megatron-LM (TP); LMCache / Mooncake (multi-tier KV, stretch). Prefill draws 70–100% of peak power, decode 20–40% — the interference argument for disaggregation. NVLink 4.0 is 900 GB/s; picking RDMA where NVLink exists wastes ~18×. |

## 9. Code to read (not copy)

- nano-vLLM — orientation, one evening
- vLLM V1 `scheduler.py` + block manager — mechanism, not policy
- SGLang radix cache
- vLLM Triton attention backend (`vllm/v1/attention/backends/triton_attn.py`)

Hard rule still holds: `slipstream/` never imports these packages. Ideas move by reading; bytes do not.

## 10. Open questions we will actually answer

1. Under W4 (bimodal lengths, hard KV cap), does a <0.5%-of-step predictor improve *smooth* goodput ≥15% over FCFS?
2. What prediction accuracy is the break-even against FCFS? (Jaillet's inaccurate-prediction analysis is the theoretical cousin.)
3. Where does the method fail? At least two regimes, documented.
4. What is the fairness cost (p99 wait, starvation) of SRPT-like ordering?
5. What is the measured swap-vs-recompute crossover as a function of block count and interconnect — not a fixed heuristic?

If (1) is no, (2)–(5) are still a paper. That is deliberate.
