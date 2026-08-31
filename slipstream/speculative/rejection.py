"""Distribution-preserving verification for speculative decoding. Phase 5 (S8).

This is the correctness heart of speculative decoding — the classic silent-bug
site (MASTERPLAN §S8, I8.1). Given the target model's distribution at each drafted
position and the draft's proposal distribution, ``verify_tokens`` decides how many
drafted tokens to keep and samples a replacement/bonus token such that **every
emitted token is distributed exactly as if it had been sampled from the target
model** — no draft bias leaks into the output.

Algorithm (Leviathan et al. 2023; Chen et al. 2023). For draft token ``x_i`` at
position ``i`` with target mass ``p_i`` and proposal mass ``q_i``:

* accept ``x_i`` with probability ``min(1, p_i(x_i) / q_i(x_i))``;
* on the first rejection, emit one token drawn from the residual
  ``norm(relu(p_i - q_i))`` and stop;
* if all ``K`` drafts are accepted, emit one bonus token drawn from the target
  distribution at position ``K``.

The emitted-token marginal telescopes to ``p`` exactly (see ``test_speculative``):
``min(q, p) + relu(p - q) == p`` elementwise. Greedy targets fall out as the
one-hot special case — a one-hot ``p`` accepts iff the draft matched the target
argmax and otherwise resamples that argmax — so no separate greedy path is needed.

Note: this reference verifier syncs to host (``.item``) to find the first
rejection. That is fine for correctness and the T0 acceptance study; a
sync-free batched variant is a Phase-4-style perf follow-up, not a correctness
concern.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# q(x) can underflow to 0 for a token the draft nonetheless sampled; clamp the
# denominator rather than divide by zero. p/q is capped at 1 regardless, so a
# tiny clamp only ever maps to "accept", which is the correct limit.
_TINY = 1e-30
# relu(p - q) sums to 0 only when p == q (where rejection is unreachable);
# guard the normalisation and fall back to the target distribution.
_RESID_EPS = 1e-12


@dataclass
class VerifyResult:
    """Outcome of verifying one drafted block against the target distribution.

    ``output_token_ids`` holds between 1 and ``num_drafted + 1`` accepted/emitted
    tokens (int64, on the input device). ``num_accepted`` is how many drafts were
    kept before the first rejection (``== num_drafted`` when the whole block was
    accepted and a bonus token was appended).
    """

    output_token_ids: torch.Tensor
    num_accepted: int
    num_drafted: int


def verify_tokens(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> VerifyResult:
    """Verify a drafted block, preserving the target distribution.

    Args:
        target_probs: Target distributions, shape ``[K + 1, vocab]``. Row ``i``
            (``i < K``) verifies draft token ``i``; row ``K`` supplies the bonus
            token when every draft is accepted. Rows must be normalised
            (e.g. ``Sampler.probs`` output; one-hot for greedy).
        draft_probs: Draft proposal distributions, shape ``[K, vocab]`` —
            the mass each draft token was actually sampled from.
        draft_token_ids: The ``K`` proposed tokens, shape ``[K]``, int64.
        generator: Optional RNG for reproducible accept/resample draws.

    Returns:
        VerifyResult with the emitted tokens and acceptance count.
    """
    if target_probs.ndim != 2 or draft_probs.ndim != 2:
        raise ValueError("target_probs and draft_probs must be 2-D [positions, vocab]")
    num_drafted = int(draft_token_ids.shape[0])
    if draft_probs.shape[0] != num_drafted:
        raise ValueError(f"draft_probs has {draft_probs.shape[0]} rows, expected {num_drafted}")
    if target_probs.shape[0] != num_drafted + 1:
        raise ValueError(
            f"target_probs must have K+1={num_drafted + 1} rows, got {target_probs.shape[0]}"
        )
    if target_probs.shape[1] != draft_probs.shape[1]:
        raise ValueError("target and draft vocab sizes differ")

    device = target_probs.device
    draft_token_ids = draft_token_ids.to(device=device, dtype=torch.long)

    if num_drafted > 0:
        pos = torch.arange(num_drafted, device=device)
        p_x = target_probs[pos, draft_token_ids]  # [K]
        q_x = draft_probs[pos, draft_token_ids]  # [K]
        ratio = (p_x / q_x.clamp_min(_TINY)).clamp_(max=1.0)
        u = torch.rand(num_drafted, device=device, generator=generator)
        accepted = u < ratio  # [K]
    else:
        accepted = torch.ones(0, dtype=torch.bool, device=device)

    if bool(accepted.all()):
        # Whole block accepted → append one bonus token from the target.
        bonus = _sample_from(target_probs[num_drafted], generator)
        emitted = torch.cat([draft_token_ids, bonus.view(1)])
        return VerifyResult(emitted, num_drafted, num_drafted)

    # First rejection: argmax over the boolean-negation gives the first False.
    j = int((~accepted).to(torch.uint8).argmax().item())
    residual = torch.clamp(target_probs[j] - draft_probs[j], min=0.0)
    total = residual.sum()
    if float(total) <= _RESID_EPS:
        residual = target_probs[j]
        total = residual.sum()
    residual = residual / total
    resample = _sample_from(residual, generator)
    emitted = torch.cat([draft_token_ids[:j], resample.view(1)])
    return VerifyResult(emitted, j, num_drafted)


def _sample_from(dist: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Draw one int64 token from a 1-D normalised distribution."""
    return torch.multinomial(dist, num_samples=1, generator=generator).to(torch.long)
