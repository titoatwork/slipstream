"""Batched, sync-free speculative verification. CPU. Phase 5 (S8).

``batched_verify_tokens`` verifies ``B`` drafted blocks at once with no host sync,
and must be a row-for-row equivalent of the single-block ``verify_tokens``: same
accept rule, same first-rejection index, same residual/bonus construction. Two
bars pin that:

* **Deterministic identity** — forced-accept / forced-reject blocks with one-hot
  finals make the outcome independent of the RNG, so the batched result must equal
  both the per-row reference and the hand-computed expectation, exactly.
* **Distribution preservation (I8.1)** — the batched emitted-token marginal matches
  the *target* under a chi-squared test, and a broken "always accept" batched
  verifier is rejected by that same test (teeth).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slipstream.core.sampling_params import SamplingParams  # noqa: E402
from slipstream.engine.sampler import Sampler  # noqa: E402
from slipstream.speculative.rejection import (  # noqa: E402
    batched_verify_tokens,
    verify_tokens,
)

# chi-squared critical value, alpha = 0.001, df = vocab - 1 = 7.
_CHI2_CRIT_7 = 24.322


def _dist(logits: torch.Tensor) -> torch.Tensor:
    return Sampler().probs(logits.unsqueeze(0), SamplingParams()).squeeze(0)


def _onehot(vocab: int, token: int) -> torch.Tensor:
    row = torch.zeros(vocab)
    row[token] = 1.0
    return row


def _forced_block(
    vocab: int, k: int, reject_at: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build one block whose outcome is RNG-independent.

    Positions before ``reject_at`` accept (target == draft == one-hot on the drafted
    token, ratio 1). At ``reject_at`` the draft proposes a token with zero target
    mass (ratio 0 → reject) and the target is one-hot elsewhere, so the residual is
    that one-hot → a deterministic resample. ``reject_at=None`` accepts the whole
    block and the bonus row ``target[k]`` is one-hot → a deterministic bonus.

    Returns ``(target [k+1, V], draft [k, V], draft_tokens [k], expected_last)``.
    """
    target = torch.zeros(k + 1, vocab)
    draft = torch.zeros(k, vocab)
    tokens = torch.zeros(k, dtype=torch.long)
    n_accept = k if reject_at is None else reject_at

    for i in range(n_accept):
        a = (i * 2 + 1) % vocab
        target[i] = _onehot(vocab, a)
        draft[i] = _onehot(vocab, a)
        tokens[i] = a

    if reject_at is None:
        bonus = (k + 3) % vocab
        target[k] = _onehot(vocab, bonus)
        expected_last = bonus
    else:
        j = reject_at
        x = 0  # drafted (to be rejected) token
        a = 1 if vocab > 1 else 0  # target's mass, distinct from x
        draft[j] = _onehot(vocab, x)
        target[j] = _onehot(vocab, a)
        tokens[j] = x
        expected_last = a
        # Positions after the rejection are never reached; fill with valid rows.
        for i in range(j + 1, k):
            target[i] = _onehot(vocab, 0)
            draft[i] = _onehot(vocab, 0)
            tokens[i] = 0

    return target, draft, tokens, expected_last


def test_batched_matches_reference_on_forced_blocks() -> None:
    # A batch spanning every first-rejection position (0..k-1) plus full accept.
    vocab, k = 9, 4
    reject_positions: list[int | None] = [None, 0, 1, 2, 3, None, 2]
    targets, drafts, tokens, expected_last, expected_na = [], [], [], [], []
    for r in reject_positions:
        t, d, tok, last = _forced_block(vocab, k, r)
        targets.append(t)
        drafts.append(d)
        tokens.append(tok)
        expected_last.append(last)
        expected_na.append(k if r is None else r)

    target_probs = torch.stack(targets)
    draft_probs = torch.stack(drafts)
    draft_tokens = torch.stack(tokens)

    res = batched_verify_tokens(target_probs, draft_probs, draft_tokens)

    assert res.num_drafted == k
    assert res.num_accepted.tolist() == expected_na
    assert res.last_token.tolist() == expected_last

    # Row-for-row equality with the single-block reference (also RNG-independent here).
    for b in range(len(reject_positions)):
        ref = verify_tokens(target_probs[b], draft_probs[b], draft_tokens[b])
        emitted_ref = ref.output_token_ids.tolist()
        na = int(res.num_accepted[b].item())
        emitted_batched = draft_tokens[b, :na].tolist() + [int(res.last_token[b].item())]
        assert emitted_batched == emitted_ref
        assert na == ref.num_accepted


def test_batched_full_accept_appends_target_bonus() -> None:
    # p == q on every position ⇒ whole block accepted ⇒ num_accepted == k, and the
    # bonus is drawn from target[k].
    vocab, k, batch = 16, 4, 5
    probs = _dist(torch.randn(vocab))
    target = probs.view(1, 1, vocab).expand(batch, k + 1, vocab).contiguous()
    draft = probs.view(1, 1, vocab).expand(batch, k, vocab).contiguous()
    gen = torch.Generator().manual_seed(0)
    draft_tokens = torch.multinomial(probs.expand(batch, vocab), k, replacement=True, generator=gen)

    res = batched_verify_tokens(target, draft, draft_tokens, generator=gen)
    assert torch.equal(res.num_accepted, torch.full((batch,), k, dtype=torch.long))
    assert res.last_token.shape == (batch,)


def _batched_first_emitted_counts(
    target: torch.Tensor,
    draft: torch.Tensor,
    *,
    n_trials: int,
    batch: int,
    seed: int,
    broken: bool = False,
) -> torch.Tensor:
    """Marginal of the first emitted token, K=1, over ``n_trials`` batched calls.

    Each call verifies ``batch`` independent draws at once. For K=1 the first
    emitted token is the draft token when accepted, else the residual resample. A
    ``broken`` run keeps the draft token unconditionally → the draft marginal.
    """
    vocab = target.shape[-1]
    gen = torch.Generator().manual_seed(seed)
    counts = torch.zeros(vocab, dtype=torch.float64)
    tgt = target.view(1, 1, vocab).expand(batch, 2, vocab).contiguous()
    drf = draft.view(1, 1, vocab).expand(batch, 1, vocab).contiguous()
    for _ in range(n_trials):
        draft_tok = torch.multinomial(draft.expand(batch, vocab), 1, generator=gen)  # [B, 1]
        if broken:
            first = draft_tok.squeeze(1)
        else:
            res = batched_verify_tokens(tgt, drf, draft_tok, generator=gen)
            first = torch.where(res.num_accepted == 1, draft_tok.squeeze(1), res.last_token)
        counts += torch.bincount(first, minlength=vocab).double()
    return counts


def test_batched_first_emitted_matches_target_distribution() -> None:
    target = _dist(torch.tensor([3.0, 1.0, 0.5, 0.0, -1.0, -1.0, 0.2, 0.7]))
    draft = _dist(torch.tensor([-1.0, 0.0, 0.5, 2.0, 1.5, 0.3, -0.5, 0.1]))
    n_trials, batch = 40, 1000
    counts = _batched_first_emitted_counts(target, draft, n_trials=n_trials, batch=batch, seed=2026)
    n = n_trials * batch
    expected = target.double() * n
    stat = float((((counts - expected) ** 2) / expected).sum())
    assert stat < _CHI2_CRIT_7, f"chi2={stat:.2f} — batched emitted marginal drifted from target"


def test_batched_distribution_test_has_teeth() -> None:
    target = _dist(torch.tensor([3.0, 1.0, 0.5, 0.0, -1.0, -1.0, 0.2, 0.7]))
    draft = _dist(torch.tensor([-1.0, 0.0, 0.5, 2.0, 1.5, 0.3, -0.5, 0.1]))
    n_trials, batch = 40, 1000
    counts = _batched_first_emitted_counts(
        target, draft, n_trials=n_trials, batch=batch, seed=2026, broken=True
    )
    n = n_trials * batch
    expected = target.double() * n
    stat = float((((counts - expected) ** 2) / expected).sum())
    assert stat > _CHI2_CRIT_7, "broken batched verifier slipped past the chi-squared test"


def test_batched_rejects_malformed_shapes() -> None:
    vocab, k, batch = 8, 3, 4
    good_target = torch.rand(batch, k + 1, vocab)
    good_draft = torch.rand(batch, k, vocab)
    good_tokens = torch.zeros(batch, k, dtype=torch.long)

    with pytest.raises(ValueError):
        batched_verify_tokens(good_target[:, :k], good_draft, good_tokens)  # target needs K+1 rows
    with pytest.raises(ValueError):
        batched_verify_tokens(good_target, good_draft, good_tokens[0])  # tokens must be 2-D
    with pytest.raises(ValueError):
        batched_verify_tokens(good_target[0], good_draft, good_tokens)  # target must be 3-D
    with pytest.raises(ValueError):
        batched_verify_tokens(
            good_target, torch.rand(batch, k, vocab + 1), good_tokens
        )  # vocab mismatch
