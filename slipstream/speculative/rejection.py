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


@dataclass
class BatchedVerifyResult:
    """Outcome of verifying ``B`` drafted blocks at once, **without host syncs**.

    The single-block :func:`verify_tokens` calls ``.item()`` to locate the first
    rejection and returns a variable-length list — fine for the acceptance study,
    but it serialises the batch and forces a device→host round-trip per block. This
    result keeps everything on-device as fixed-shape tensors so a batched runner can
    advance every sequence's committed length on-GPU:

    * ``num_accepted`` ``[B]`` — drafts kept before the first rejection per row
      (``== num_drafted`` when the whole block was accepted).
    * ``last_token`` ``[B]`` — the one trailing token each row emits: the bonus
      token (fully accepted) or the residual resample (first rejection).

    Row ``b`` therefore emits ``num_accepted[b] + 1`` tokens: the drafts
    ``draft_token_ids[b, :num_accepted[b]]`` followed by ``last_token[b]``. The
    caller slices per row (the only place a length is needed on host).
    """

    num_accepted: torch.Tensor
    last_token: torch.Tensor
    num_drafted: int


def batched_verify_tokens(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> BatchedVerifyResult:
    """Verify ``B`` drafted blocks in one vectorised, sync-free pass.

    Row-for-row equivalent of :func:`verify_tokens` (same accept rule, same
    residual/bonus construction), computed for the whole batch at once with no
    ``.item()`` / host branch. Every block must share the same block length ``K``.

    Args:
        target_probs: Target distributions ``[B, K + 1, vocab]`` (rows ``0..K-1``
            verify the drafts; row ``K`` supplies the bonus). Normalised.
        draft_probs: Draft proposal distributions ``[B, K, vocab]``.
        draft_token_ids: Proposed tokens ``[B, K]``, int64.
        generator: Optional RNG for reproducible accept/resample draws.

    Returns:
        A :class:`BatchedVerifyResult` with per-row acceptance counts and the
        trailing emitted token.
    """
    if target_probs.ndim != 3 or draft_probs.ndim != 3:
        raise ValueError("target_probs and draft_probs must be 3-D [B, positions, vocab]")
    if draft_token_ids.ndim != 2:
        raise ValueError("draft_token_ids must be 2-D [B, K]")
    batch, num_drafted = draft_token_ids.shape
    if tuple(draft_probs.shape[:2]) != (batch, num_drafted):
        raise ValueError(
            f"draft_probs shape {tuple(draft_probs.shape[:2])} != (B, K)=({batch}, {num_drafted})"
        )
    if tuple(target_probs.shape[:2]) != (batch, num_drafted + 1):
        raise ValueError(
            f"target_probs shape {tuple(target_probs.shape[:2])} != "
            f"(B, K+1)=({batch}, {num_drafted + 1})"
        )
    if target_probs.shape[2] != draft_probs.shape[2]:
        raise ValueError("target and draft vocab sizes differ")

    device = target_probs.device
    vocab = target_probs.shape[2]
    draft_token_ids = draft_token_ids.to(device=device, dtype=torch.long)

    if num_drafted > 0:
        idx = draft_token_ids.unsqueeze(-1)  # [B, K, 1]
        p_x = target_probs[:, :num_drafted].gather(-1, idx).squeeze(-1)  # [B, K]
        q_x = draft_probs.gather(-1, idx).squeeze(-1)  # [B, K]
        ratio = (p_x / q_x.clamp_min(_TINY)).clamp_(max=1.0)  # [B, K]
        u = torch.rand(batch, num_drafted, device=device, generator=generator)
        accept = u < ratio  # [B, K]
        all_accepted = accept.all(dim=1)  # [B]
        # First rejection per row: argmax finds the first True in ~accept (0 if
        # none — masked out below by `all_accepted`), so num_accepted = K there.
        first_reject = (~accept).to(torch.uint8).argmax(dim=1)  # [B]
        num_accepted = torch.where(
            all_accepted, torch.full_like(first_reject, num_drafted), first_reject
        ).to(torch.long)
    else:
        all_accepted = torch.ones(batch, dtype=torch.bool, device=device)
        num_accepted = torch.zeros(batch, dtype=torch.long, device=device)

    # Build each row's trailing distribution from target row j = num_accepted:
    # j == K (fully accepted) → the bonus distribution target[K]; j < K (first
    # rejection) → the residual relu(target[j] - draft[j]).
    j = num_accepted
    tgt_j = target_probs.gather(1, j.view(batch, 1, 1).expand(batch, 1, vocab)).squeeze(1)  # [B, V]
    if num_drafted > 0:
        j_draft = j.clamp(max=num_drafted - 1)  # draft has only K rows; masked when j == K
        draft_j = draft_probs.gather(1, j_draft.view(batch, 1, 1).expand(batch, 1, vocab)).squeeze(
            1
        )
        subtract = torch.where(all_accepted.view(batch, 1), torch.zeros_like(draft_j), draft_j)
        resid = torch.clamp(tgt_j - subtract, min=0.0)  # [B, V]
    else:
        resid = tgt_j
    total = resid.sum(dim=1, keepdim=True)
    # relu(p - q) sums to 0 only when p == q (rejection unreachable) — fall back
    # to the target distribution, matching the single-block verifier.
    resid = torch.where(total <= _RESID_EPS, tgt_j, resid)
    resid = resid / resid.sum(dim=1, keepdim=True)
    last_token = (
        torch.multinomial(resid, num_samples=1, generator=generator).squeeze(1).to(torch.long)
    )  # [B]

    return BatchedVerifyResult(
        num_accepted=num_accepted, last_token=last_token, num_drafted=num_drafted
    )
