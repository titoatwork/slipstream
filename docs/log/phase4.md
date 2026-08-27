# Phase 4 log

## 2026-08-14

### Shipped (code)

- Horizon f0–f2 predictor (`scheduler/predictor/`). CPU-only. `seq.predicted_remaining` is the only field the policy reads.
- `HorizonPolicy`: projected-HWM admission, SRPT order, starvation guard, victim = `(rem · blocks) / slo_tpot`, SWAP if ≥4 pages.
- `OraclePolicy`: remaining = `oracle_output_len − num_output`.
- `get_policy` forwards `safety_factor` / `starvation_guard_ms` from `SchedulerConfig`.
- Scheduler `refresh` / `observe` hooks; `ensure_slot` preempts on grow-OOM (FCFS can over-admit prompt pages then blow up on decode).
- RECOMPUTE replays prompt+prior outputs; `can_allocate` peak is prompt+cap; no same-step victim; no self-preempt storm.
- Arrival clock: relative W4 offsets are converted to wall time so the 5s guard does not turn Horizon into FIFO.
- Idle-engine fallback: safety factor cannot deadlock a lone request that fits the pool.
- Per-request `RequestTrace` + smooth goodput (fluidity = 1.0; aborts never count as hits).
- `engine/isolated.py`: in-process default; `SLIPSTREAM_ISOLATE=1` spawns a child. T0 cannot hold two model copies.
- Fused kernel *seams* + `CudaGraphPool`. Default forward stays eager.

### Research (do not over-claim)

- Jaillet et al. arXiv:2502.07115v5: no deterministic online algorithm has a constant CR on arbitrary arrivals. MC-SF is O(1) only under structured arrivals + overestimate predictions + `M ≥ 2 max (s+ô)`. Horizon does **not** claim a better competitive ratio.
- Jaillet App. B.3: overestimate factor `α` degrades CR polynomially (`α³`). Underestimates overflow `M`.
- Feng et al. arXiv:2601.22996: GSA is non-clairvoyant via *restart*. Wrong object for a serving engine (destroys TTFT). Horizon is the systems counterpart.
- llm-d (Mar 2026) predicts **TTFT/TPOT for replica routing**, not remaining decode length. Not evidence that f0–f2 will win on W4.
- Li et al. “Beyond Prediction” (2026): SRPT with *perfect* lengths can lose on p99 vs a tail-aware policy. Fairness p99 < 2× FCFS is load-bearing.
- vLLM CUDA graphs: variable block tables / launch grids cannot be one captured graph. Graph static GEMM+norm; leave paged attention eager until a persistent padded kernel exists.
- T0 prior: 0.5B on a 6 GB 3050 is weight-bound unless `num_gpu_blocks` is artificially tight. A null Horizon gain on T0 is expected and does not falsify T1.

### Metrics (binding)

| Name | Definition |
|---|---|
| Naive goodput | completed / wall (aborts excluded from numerator, stay in `R`) |
| Mean-SLO goodput | finished ∧ TTFT≤2s ∧ mean ITL≤200ms / wall |
| **Smooth goodput** | finished ∧ not aborted ∧ TTFT≤2s ∧ **every** ITL≤200ms / wall |
| Gap closed | `(H−F)/(O−F)` if `O>F` else N/A |
| Fairness | Horizon p99 TTFT / FCFS p99 TTFT |

Never tune on naive goodput. Oracle remaining is exact because W4 sets `ignore_eos=True`.

### Gate 4 — slack KV (invalid for admission)

36 blocks / 6 seqs, `preemptions=0`. Discarded as a Horizon *policy* result. SRPT still cut p50 TTFT 15.0s → 6.9s.

### Gate 4 — tight KV (binding). This is the cell.

24 blocks / 12 seqs / 16 CPU pages. Warmed. FCFS **13 preemptions**, Horizon/Oracle **0**.

| | FCFS | Horizon | Oracle |
|---|---|---|---|
| Wall | 23.9s | 26.8s | 23.2s |
| Naive goodput | **0.752** | 0.672 (−11%) | 0.777 |
| T0-smooth hits (30s / 1.0s) | 1/18 | **18/18** | 18/18 |
| T0-smooth goodput | 0.042 | **0.672 (16×)** | 0.777 |
| Gap closed (smooth) | — | **86%** | — |
| p99 ITL | **5.01s** | 0.36s | 0.48s |
| p99 TTFT vs FCFS | — | 1.34× | — |
| Paper SLO 2s/200ms | 0 | 1 | 0 |

FCFS over-admits prompt pages, then grow-OOM preempts; ITL tail explodes. Horizon’s HWM refuses that set. Naive wall still likes the larger FCFS batch. Paper SLO remains T0-unattainable (Horizon ITL p99 358ms).

Homogeneous shorts: Horizon *loses* wall (5.2 vs 3.7s). No SRPT signal.

Predictor: **9.7 µs**, 0.019% of a 50 ms step.

### Preemption path (required for the tight cell)

- Replay prompt + outputs[:-1] after RECOMPUTE (do not drop outputs).
- `can_allocate` peak = prompt + output cap (do not double-count generated tokens).
- `seq_lens` = filled KV slots, not `num_tokens`.
- Do not preempt a seq already slotted this step.
- Do not self-preempt when no unprotected victim exists (that was a 1111-preempt storm).
- Waiting seqs must not hold GPU pages; identity-safe queue discard.

GPU: 2-seq / 3-block pressure completes 32+32 with 1 preemption, 0 aborts.

### Ablation + open-loop (2026-08-15)

`inject="arrival"` is opt-in. Default `generate_batch` stays closed-burst (tests unchanged). TTFT clock starts at arrival.

**f0 vs f2 (same closed 24-block burst):** f0 5/18 T0-smooth, wall 47s, p99 ITL 1.36s. f2 18/18, 26.8s, 0.36s. Both 0 preemptions. Features, not just HWM.

**Open-loop 0.20s gap:** FCFS 71 preemptions, 0/18 smooth, p99 ITL 6.0s. Horizon-f2 0 preempt, 16/18 smooth, p99 ITL 0.51s, 77% of oracle gap. FCFS still wins p50 TTFT (4.2s vs 7.9s). Open-loop f0: 13/18, wall 34s, p99 ITL 1.10s.

**Closed repeats (2 extra seeds):** FCFS preempts 13 every time; Horizon 0. Smooth hits Horizon 12 and 15 vs FCFS 1 and 5. Ratio 6.6× and 1.81× (first seed was 16×). p99 TTFT ratio **2.69× / 2.68×** — fairness gate fails on the repeats. Do not headline 16×.

Clock fix: `inject="all"` TTFT starts at enqueue, not at `arrival` offset (avoids negative TTFT on `arrival_ts=i`). Traces now split `wait_s` / `prefill_s`.

### Next

- Isolation CPU occupancy on a host that can hold two copies.
- T1: Llama-3.1-8B identity, A100 70% HBM, paper 2s/200ms, ≥3 seeds on a cooled GPU.
