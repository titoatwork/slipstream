# Phase 4 results (T0)

Hardware: RTX 3050 6GB Laptop, Qwen2.5-0.5B, `SLIPSTREAM_DEBUG=0`, warmed, n=1.

Two W4 shapes. Only the **tight** run is a valid test of Horizon’s memory policy.

## Tight KV (binding) — the real Gate 4 cell

`num_gpu_blocks=24`, `max_num_seqs=12`, `num_cpu_blocks=16`, ignore_eos, 18 bimodal (12/48).

**Validity:** FCFS `preemptions=13`, Horizon `0`, Oracle `0`. KV bound before the seq cap.

| | FCFS | Horizon | Oracle |
|---|---|---|---|
| Wall | 23.9s | 26.8s | **23.2s** |
| Naive goodput (completed/wall) | **0.752** | 0.672 (−11%) | 0.777 |
| T0-smooth hits (30s / 1.0s, *every* ITL ≤ 1s) | **1/18** | **18/18** | **18/18** |
| T0-smooth goodput | 0.042 | **0.672 (16.1×)** | 0.777 |
| Gap closed to Oracle (smooth) | — | **86%** | — |
| p50 TTFT | 2.03s | 2.26s | **0.88s** |
| p99 TTFT | 17.8s | 23.8s (**1.34×**) | 17.7s |
| p50 ITL | 0.20s | **0.080s** | 0.149s |
| p99 ITL | **5.01s** | 0.36s | 0.48s |
| Preemptions | **13** | 0 | 0 |
| Paper SLO hits (2s / 200ms) | 0 | 1 | 0 |

Mechanism, not magic: FCFS over-admits prompt pages, then `ensure_slot` preempts on decode growth. Those recomputes stall ITL (p99 **5s**). Horizon’s HWM refuses the overflow set, never preempts, every request stays under the 1.0s T0 ITL cap.

Naive wall still slightly favors FCFS (larger batch, more GPU work per step). **Do not tune on naive goodput.** Paper 2s/200ms remains almost all-zero on this 0.5B burst (T0 ITL p99 Horizon is 358ms).

### Predictor ablation (same closed burst, f0 vs f2)

f0 = global mean only (no prompt features). Same 24-block cap.

| | FCFS | Horizon f0 | Horizon f2 | Oracle |
|---|---|---|---|---|
| T0-smooth hits | 1/18 | **5/18** | **18/18** | 18/18 |
| T0-smooth goodput | 0.042 | 0.106 | **0.672** | 0.777 |
| Wall | 23.9s | 47.0s | 26.8s | 23.2s |
| p99 ITL | 5.01s | 1.36s | 0.36s | 0.48s |
| Preemptions | 13 | 0 | 0 | 0 |

HWM alone (f0) already stops preemption, but without a length signal it under-fills and only 5/18 stay smooth. **f2 is not just “admit fewer.”**

### Open-loop (inject=arrival, 0.20s gap)

`generate_batch(..., inject="arrival")`. TTFT includes queue wait from `arrival_ts`. Still KV-bound (FCFS **71** preemptions).

| | FCFS | Horizon f2 | Oracle |
|---|---|---|---|
| T0-smooth hits | **0/18** | **16/18** | 18/18 |
| T0-smooth goodput | 0 | **0.321** | 0.419 |
| Gap closed | — | **77%** | — |
| Wall | 44.4s | 49.8s | 42.9s |
| p50 TTFT | **4.21s** | 7.87s | 8.20s |
| p99 TTFT | 33.3s | 39.6s (1.19×) | 27.4s |
| p99 ITL | **5.96s** | 0.51s | 0.96s |
| Preemptions | **71** | 0 | 0 |

FCFS still has the better *median* TTFT (it admits immediately). Horizon still wins the ITL tail and smooth goodput. Two Horizon misses are TTFT > 30s (longs waiting), not ITL.

### Open-loop f0 (same 0.20s list)

| | Horizon f0 | Horizon f2 | Oracle | FCFS |
|---|---|---|---|---|
| T0-smooth hits | 13/18 | **16/18** | 18/18 | 0/18 |
| T0-smooth goodput | **0.383** | 0.321 | 0.419 | 0 |
| Wall | **34.0s** | 49.8s | 42.9s | 44.4s |
| p99 ITL | 1.10s | **0.51s** | 0.96s | 5.96s |
| p99 wait | 23.5s | — | — | — |
| Preemptions | 0 | 0 | 0 | 71 |

f2 still has more SLO hits and a cleaner ITL tail. f0’s higher goodput here is a shorter wall (rate), not a better serving tail. Closed-burst f0 (5/18) remains the cleaner “features matter” cell.

### Closed-burst repeats (seeds 1–2 + original)

Same 24-block list. FCFS preempts **13 / 13 / 13**. Horizon **0 / 0 / 0**. The mechanism replicates. The **16×** does not.

| Seed | H hits | F hits | H/F smooth | H wall | F wall | p99 TTFT H/F |
|---|---|---|---|---|---|---|
| 0 (first) | 18/18 | 1/18 | **16.1×** | 26.8s | 23.9s | 1.34× |
| 1 | 12/18 | 1/18 | 6.6× | 61.4s | 33.8s | **2.69×** |
| 2 | 15/18 | 5/18 | 1.81× | 53.4s | 32.3s | **2.68×** |

Robust on T0: Horizon always more T0-smooth hits; FCFS always has the ITL-tail problem (p99 ITL 5–7s vs Horizon 0.36–0.87s). **Not robust:** gain magnitude, Horizon wall, fairness p99 TTFT (fails 2× on 2/3 seeds). Do not headline 16×. Median smooth ratio is about **6.6×**. Fairness gate is a miss on the median seed.

Likely T0 variance: laptop clocks/thermals after a long session, plus SRPT wait on longs (Horizon p99 TTFT 48–54s on repeats). n=3 is still a sketch, not a paper table.

### Failure regimes (same day)

| Regime | Result |
|---|---|
| Homogeneous shorts (12×12 tok) | FCFS preempts **0** (peaks fit). Horizon wall 5.2s vs FCFS 3.7s — no SRPT signal, HWM under-fills. |
| Paper SLO 2s/200ms | Horizon 1/18, others 0. Over-capacity on T0. |
| Slack KV (36 blocks / 6 seqs) | `preemptions=0` for everyone. Discarded as a Horizon *policy* result (see below). |
| Horizon f0 (mean only) | 5/18 smooth — HWM without features is not enough. |

### Other gates

| Criterion | Tight-KV T0 |
|---|---|
| Horizon ≥15% goodput | T0-smooth hits always higher. Ratio **1.8–16×** across 3 closed seeds (median ~6.6×). Paper-SLO miss. Naive wall: Horizon slower on 3/3 later cells. |
| Gap to Oracle ≥40% | First seed 86%; repeats 36% and 55%. Fragile. |
| Predictor <0.5% step | **9.7 µs, 0.019%** of 50ms. |
| Fairness p99 TTFT <2× | **Fail on 2/3 seeds** (2.68–2.69×). First seed 1.34×. |
| ≥2 failure regimes | Homogeneous; paper SLO; f0 without features. |

## Slack KV (invalid for admission) — kept for the record

`36` blocks / `max_num_seqs=6`, `preemptions=0`. `max_num_seqs` bound first. Horizon ≈ FCFS on wall; SRPT cut p50 TTFT 15.0s → 6.9s. Not a test of HWM admission.

Cold-start `w4_goodput_cold.json` billed CUDA compile to FCFS. Discarded.

## CUDA graphs / fused / roofline

Unchanged from the earlier T0 micros: static GEMM+norm **6–12×**; Triton RMSNorm **3–4×** opt-in; single-seq **2.3%** of 140 GB/s (launch-bound 0.5B).

## Isolation

In-process default. Two model copies OOM on 7.4 GiB RAM.

## Files

- `w4_goodput_tight.json` / `w4_goodput_summary.json` — binding-KV headlines
- `w4_goodput.json` — same + traces
- `w4_ablation.json` — f0 closed + open-loop f2/oracle/fcfs
- `w4_followup.json` — open-loop f0 + two closed-burst repeats
- `w4_goodput_cold.json` — invalid
- `fused_micro.json`, `graph_micro.json`, `roofline.json`
