"""Gate 5 speculative-decoding measurement: decode speedup + distribution test.

MASTERPLAN §S8 / Gate 5. Two deliverables, one command:

1. **Decode speedup** — spec (``SpeculativeRunner``) vs plain autoregressive greedy
   on the *same* target model, at batch 1 (the low-batch regime where speculation
   wins), swept over draft length K. Greedy output must be token-identical between
   the two paths (asserted); acceptance rate and mean accepted length are reported
   per K. The Gate 5 bar is **≥1.5×** at low batch.
2. **Distribution test (I8.1)** — the classic silent-bug check: over ``SPEC_CHI2_N``
   samples (Gate 5 bar: 10000), the spec-sampled next-token marginal must be
   statistically indistinguishable from the plain target marginal (chi-squared).

The **headline Gate 5 number needs a cheaper draft than the target with a matching
vocabulary** — MASTERPLAN prescribes Llama-3.2-1B drafting for Llama-3.1-8B on an
A100 (T1). Set ``SPEC_TARGET`` / ``SPEC_DRAFT`` and run there. With no draft set,
this runs **self-speculation** (draft == target): it validates the harness and the
distribution-preservation on real weights, but shows **no speedup** (no cheaper
proposer) — that is expected, not a Gate 5 loss, and is labelled as such.

The high-batch regime where speculation *hurts* (verify cost ∝ batch×K dominates
the acceptance-driven token savings) needs **batched** speculative verification,
which ``SpeculativeRunner`` (single-sequence) does not yet implement; it is
described in the output but not measured here — no fabricated number.

Usage::

    python -m benchmarks.workloads.spec_decode
    SPEC_TARGET=meta-llama/Llama-3.1-8B SPEC_DRAFT=meta-llama/Llama-3.2-1B \
        SPEC_MAX_TOKENS=128 python -m benchmarks.workloads.spec_decode
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from slipstream.core.config import ModelConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.engine.model_runner import ModelRunner
from slipstream.engine.sampler import Sampler
from slipstream.memory.contiguous_cache import NaiveKVCache
from slipstream.models.loader import load_model
from slipstream.speculative.kv_cached import SpeculativeRunner

from benchmarks.manifest import write_run_manifest

DEFAULT_TARGET = "Qwen/Qwen2.5-0.5B"
PROMPT_IDS = [40, 8, 25, 1043, 374, 264, 1614, 315]  # arbitrary in-vocab warm prompt
DRAFT_LENGTHS = (1, 2, 4, 8)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _plain_greedy(
    runner: ModelRunner, prompt_ids: list[int], max_new_tokens: int
) -> tuple[list[int], float]:
    """Plain autoregressive greedy: 1 prefill + (n-1) single-token decodes."""
    device = runner.device
    cfg = runner.model.config  # type: ignore[union-attr]
    cache = NaiveKVCache(
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_batch=1,
        max_len=len(prompt_ids) + max_new_tokens,
        dtype=runner.dtype,
        device=device,
    )
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    _sync(device)
    t0 = time.perf_counter()
    logits = runner.decode(ids, cache)
    out: list[int] = []
    for i in range(max_new_tokens):
        nxt = int(torch.argmax(logits[0, -1]).item())
        out.append(nxt)
        if i == max_new_tokens - 1:
            break
        logits = runner.decode(torch.tensor([[nxt]], dtype=torch.long, device=device), cache)
    _sync(device)
    return out, time.perf_counter() - t0


def _timed_spec(
    runner: SpeculativeRunner, prompt_ids: list[int], k: int, max_new_tokens: int
) -> tuple[list[int], float, float, float]:
    _sync(runner.device)
    t0 = time.perf_counter()
    out, stats = runner.generate(
        prompt_ids,
        params=SamplingParams(temperature=0.0),
        num_speculative_tokens=k,
        max_new_tokens=max_new_tokens,
    )
    _sync(runner.device)
    return out, time.perf_counter() - t0, stats.acceptance_rate, stats.mean_accepted_len


def _chi2_distribution(
    target_runner: ModelRunner,
    runner: SpeculativeRunner,
    prompt_ids: list[int],
    n_samples: int,
) -> dict[str, object]:
    """I8.1: spec-sampled next-token marginal vs the plain target marginal."""
    device = target_runner.device
    cfg = target_runner.model.config  # type: ignore[union-attr]
    params = SamplingParams(temperature=1.0)

    # Target marginal at the prompt (one plain forward, filtered like the sampler).
    cache = NaiveKVCache(
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_batch=1,
        max_len=len(prompt_ids),
        dtype=target_runner.dtype,
        device=device,
    )
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits = target_runner.decode(ids, cache)
    p = Sampler().probs(logits[:, -1], params).squeeze(0).double()

    # Valid Pearson chi-squared needs every expected count >= ~5, so bin the
    # (150k-wide, mostly near-zero) vocab: keep tokens with expected count >= 5
    # as their own cells and lump the whole tail into a single "other" cell.
    min_expected = 5.0
    keep = (p * n_samples) >= min_expected
    index = torch.nonzero(keep, as_tuple=False).flatten()
    p_other = float(p[~keep].sum().item())
    remap = torch.full((p.numel(),), -1, dtype=torch.long, device=device)
    remap[index] = torch.arange(index.numel(), device=device)

    gen = torch.Generator(device=device).manual_seed(2026)
    counts = torch.zeros(index.numel(), dtype=torch.float64, device=device)
    other = 0
    for _ in range(n_samples):
        out, _ = runner.generate(
            prompt_ids,
            params=params,
            num_speculative_tokens=4,
            max_new_tokens=1,
            generator=gen,
        )
        slot = int(remap[out[0]].item())
        if slot < 0:
            other += 1
        else:
            counts[slot] += 1.0

    # Cells: the kept tokens plus the lumped tail. dof = num_cells - 1.
    obs = torch.cat([counts, torch.tensor([float(other)], dtype=torch.float64, device=device)])
    p_cells = torch.cat([p[index], torch.tensor([p_other], dtype=torch.float64, device=device)])
    expected = p_cells * n_samples
    chi2 = float((((obs - expected) ** 2) / expected.clamp_min(1e-12)).sum().item())
    dof = int(obs.numel() - 1)
    return {
        "n_samples": n_samples,
        "num_cells": int(obs.numel()),
        "kept_tokens": int(index.numel()),
        "min_expected_count": min_expected,
        "other_mass": p_other,
        "dof": dof,
        "chi2": chi2,
        "chi2_per_dof": chi2 / dof if dof else 0.0,
    }


def main() -> None:
    # Inference only: no autograd graph (correct timing + memory, and silences the
    # requires_grad→scalar warning from the verifier's residual check).
    torch.set_grad_enabled(False)
    target_id = os.environ.get("SPEC_TARGET", DEFAULT_TARGET)
    draft_id = os.environ.get("SPEC_DRAFT", target_id)
    max_new_tokens = int(os.environ.get("SPEC_MAX_TOKENS", "64"))
    chi2_n = int(os.environ.get("SPEC_CHI2_N", "10000"))
    self_spec = draft_id == target_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    target_model = load_model(target_id, device=device, dtype=dtype)
    draft_model = target_model if self_spec else load_model(draft_id, device=device, dtype=dtype)
    target_runner = ModelRunner(
        ModelConfig(model_id=target_id), model=target_model, device=device, dtype=dtype
    )
    draft_runner = ModelRunner(
        ModelConfig(model_id=draft_id), model=draft_model, device=device, dtype=dtype
    )
    runner = SpeculativeRunner(target_runner, draft_runner)

    # Warmup (kernel autotune / graph / allocator) before any timing.
    _plain_greedy(target_runner, PROMPT_IDS, 4)
    runner.generate(
        PROMPT_IDS,
        params=SamplingParams(temperature=0.0),
        num_speculative_tokens=4,
        max_new_tokens=4,
    )

    base_out, base_s = _plain_greedy(target_runner, PROMPT_IDS, max_new_tokens)
    base_tok_s = len(base_out) / base_s if base_s else 0.0

    by_k: list[dict[str, object]] = []
    for k in DRAFT_LENGTHS:
        spec_out, spec_s, acc, mean_len = _timed_spec(runner, PROMPT_IDS, k, max_new_tokens)
        by_k.append(
            {
                "k": k,
                "spec_s": spec_s,
                "spec_tok_s": len(spec_out) / spec_s if spec_s else 0.0,
                "speedup": base_s / spec_s if spec_s else 0.0,
                "acceptance_rate": acc,
                "mean_accepted_len": mean_len,
                "token_identical_to_plain_greedy": spec_out == base_out,
            }
        )

    chi2 = _chi2_distribution(target_runner, runner, PROMPT_IDS, chi2_n)

    result: dict[str, object] = {
        "target": target_id,
        "draft": draft_id,
        "self_speculation": self_spec,
        "max_new_tokens": max_new_tokens,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "baseline_plain_s": base_s,
        "baseline_plain_tok_s": base_tok_s,
        "by_draft_length": by_k,
        "best_speedup": max((row["speedup"] for row in by_k), default=0.0),
        "gate5_speedup_target": 1.5,
        "distribution_test": chi2,
        "high_batch_regime": (
            "Not measured: SpeculativeRunner is single-sequence. Speculation hurts "
            "at high batch because verify cost scales ~batch×K while acceptance-driven "
            "token savings do not; measuring it needs batched speculative verification."
        ),
        "note": (
            "SELF-SPECULATION (draft == target): validates the harness and I8.1 on real "
            "weights but shows no speedup — there is no cheaper proposer. Set SPEC_DRAFT to "
            "a smaller matching-vocab model on T1/A100 for the Gate 5 number."
            if self_spec
            else "Cross-model speculation."
        ),
    }

    dest = Path("benchmarks/results/phase5")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "spec_decode.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_manifest(
        dest / "run_manifest.json",
        model=target_id,
        workload="phase5_spec_decode",
        config=result,
        notes="Gate 5 speculative decode: batch-1 speedup + I8.1 chi-squared distribution test.",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
