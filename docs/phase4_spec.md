# Phase 4 spec — Horizon, graphs, fused kernels

Binding: MASTERPLAN §5, §10 Phase 4. Negative results are valid.

## Gate 4 on T0

| Criterion | T0 plan |
|---|---|
| Horizon goodput ≥ 15% over FCFS on W4-like | Tight KV, bimodal short/long; report oracle bound |
| Predictor < 0.5% of step time | Microbench 10k predicts |
| Gap closed toward oracle ≥ 40% | `(H-F)/(O-F)` if O>F else N/A |
| ≥2 failure regimes | Homogeneous lengths; over-capacity SLO |
| Fairness p99 wait < 2× FCFS | Record wait times |
| CUDA graphs ≥ 20% decode step cut, batch≤32 | Graph a static decode if possible; else document |
| Fused kernels measured win or removed | RMSNorm+residual, SwiGLU, RoPE |
| Roofline / MFU | Decode bytes vs A100/T0 peaks |

---

## File ownership

| Agent | Writes |
|---|---|
| **A8** | `scheduler/predictor/*`, `policies/horizon.py`, `policies/oracle.py`; may add `policy.refresh()` hook in `scheduler.py` |
| **A3** | `kernels/fused_*.py`, `engine/cuda_graph.py`; optional vectorized attention for graphs |
| **A5** | `engine/engine_core.py` isolation helper `engine/isolated.py` (optional process) |
| **A7** | `benchmarks/workloads/w4_goodput.py`, `benchmarks/analysis/roofline.py`, `tests/correctness/test_horizon.py`, `docs/log/phase4.md` |

A8 must not break FCFS or paging identity.

---

## Predictor (f0–f2; f3 stretch)

- **f0**: EMA of observed output length; remaining = max(1, mean − generated)
- **f1**: `a + b·log1p(prompt_len) + c·unigram_entropy + d·code_like`
- **f2**: f1, then subtract tokens already generated; if generated ≫ predicted, decay remaining

`observe()` updates EMA and a 4-bucket prompt-length mean. No GPU. No `.item()` in a hot GPU path — predictor is CPU and called once per schedule, not per layer.

`seq.predicted_remaining` is the only field the policy reads.

---

## Horizon policy

- **refresh(seqs, state)**: write `predicted_remaining` for every seq
- **order_waiting**: SRPT on remaining; if `(now−arrival)*1000 ≥ starvation_guard_ms`, FIFO among those (starvation guard)
- **should_admit**: projected block high-water (running remaining + this request's peak) ≤ `num_total_blocks * safety_factor`
- **select_preemption_victim**: max `(remaining · blocks) / max(slo_tpot_ms, 1)`
- **preemption_mode**: SWAP if `len(block_table)≥4` else RECOMPUTE

Oracle: same rules with `remaining = oracle_output_len − num_output` (must be set by the harness).

---

## Engine hook

`Scheduler.schedule()` calls `policy.refresh(...)` if present, then existing FCFS mechanism.

On finish: `predictor.observe(seq, num_output)` if the policy has a predictor.

---

## Graphs & fused

- `CudaGraphPool`: static buffers keyed by batch size; capture/replay `fn()` of GPU ops only
- Fused kernels: correct vs current RMSNorm / silu-mul / RoPE; use Triton on CUDA with torch fallback
- If a fuse does not beat eager on T0, keep the eager path as default and record the number

---

## W4 harness (scaled for T0)

Not 8192-token prompts (won't fit). Bimodal, **`ignore_eos=True`** so Oracle remaining is exact:

- Short: chat-like prompt, output 12
- Long: code-like prompt, output 48
- `num_gpu_blocks=24`, `max_num_seqs=12` so FCFS over-admits prompt pages and grows into preemption (`preemptions>0` is the validity check)
- SLO: TTFT 2s, TPOT 200ms
- Same request list for FCFS / Horizon / Oracle
- Failure regimes: homogeneous lengths; over-tight TPOT=20ms on the same traces

### Metric definitions

| Name | Definition |
|---|---|
| Naive goodput | `#{completed} / wall` |
| Mean-SLO goodput | finished ∧ TTFT≤2s ∧ mean ITL≤200ms / wall |
| **Smooth goodput** | finished ∧ not aborted ∧ TTFT≤2s ∧ **every** ITL≤200ms / wall |
| Gap closed | `(H−F)/(O−F)` if `O>F` else N/A |
| Fairness | Horizon p99 TTFT / FCFS p99 TTFT (gate: < 2×) |

Aborts stay in the offered set and never count as SLO hits (Agrawal et al. 2025).

CUDA graphs: capture a **static** GEMM+RMSNorm decode-like fn only. Paged attention walks a variable block table and cannot be one Python-loop graph (vLLM uses persistent kernels / fixed grids).
