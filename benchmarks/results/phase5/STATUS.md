# Phase 5 results — Speculative decoding (S8)

Reproduce: `python -m benchmarks.workloads.spec_decode`
(env: `SPEC_TARGET`, `SPEC_DRAFT`, `SPEC_MAX_TOKENS`, `SPEC_CHI2_N`).

## Correctness — Qwen2.5-0.5B / RTX 3050 6GB (bf16, T0)

Self-speculation (draft == target); greedy `max_tokens=48`; chi² over 3000 samples.

| Check | Result |
|---|---|
| KV-cached spec vs plain greedy, K∈{1,2,4,8} | **token-identical** |
| Distribution test I8.1 (χ², 44 cells, dof=43) | χ² = 34.9, χ²/dof = 0.81 — **pass** (crit ≈ 59.3) |
| Acceptance rate (bf16 self-spec) | 0.93–0.96 |

I8.1 is the classic silent-bug site; token-identity + the χ² fit confirm the
KV-cache rollback path preserves the target distribution on real weights.
Acceptance is <1.0 even for self-speculation because the draft's single-token
incremental forwards and the target's batched block forward are different bf16
numerical paths — output stays exact; the verifier corrects the divergences.

## Decode speedup — NOT the Gate 5 number (self-speculation, no cheaper draft)

| K | acceptance | mean accepted len | spec tok/s | speedup vs plain |
|---|---|---|---|---|
| 1 | 0.96 | 1.96 | 7.7 | 0.57× |
| 2 | 0.94 | 2.88 | 8.2 | 0.61× |
| 4 | 0.93 | 4.45 | 8.3 | **0.61×** |
| 8 | 0.96 | 8.17 | 7.9 | 0.58× |

Plain greedy baseline: 13.5 tok/s. Self-speculation is **slower** (~0.6×) — it
does strictly more work per token (K+1 draft forwards + 1 batched target forward
vs 1 target forward) with no cheaper proposer to pay for it. This is the expected
"speculation without a cheaper draft does not help" result, not a Gate 5 loss.

## Gate 5 headline (≥1.5×) — pending T1/A100

Needs a genuinely cheaper draft with a matching vocabulary — MASTERPLAN §S8:
**Llama-3.2-1B drafting for Llama-3.1-8B** on A100. The harness is ready:

```
SPEC_TARGET=meta-llama/Llama-3.1-8B SPEC_DRAFT=meta-llama/Llama-3.2-1B \
    SPEC_MAX_TOKENS=128 SPEC_CHI2_N=10000 python -m benchmarks.workloads.spec_decode
```

The high-batch regime where speculation *hurts* (verify cost ∝ batch×K dominates
the acceptance-driven token savings) needs **batched** speculative verification,
which `SpeculativeRunner` (single-sequence) does not yet implement — not measured
here, no fabricated number.
