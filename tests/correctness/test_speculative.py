"""Speculative-decoding verification: distribution preservation (I8.1). CPU.

The invariant that makes speculative decoding correct rather than merely fast:
every emitted token is distributed exactly as if drawn from the target model,
never the draft. These tests pin that with a chi-squared goodness-of-fit test
over many trials, and — critically — show the test has teeth by confirming a
deliberately broken "always accept the draft" verifier *fails* it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slipstream.core.sampling_params import SamplingParams  # noqa: E402
from slipstream.engine.sampler import Sampler  # noqa: E402
from slipstream.speculative.rejection import verify_tokens  # noqa: E402

# chi-squared critical values, alpha = 0.001 (very lenient — the RNG is seeded so
# a correct verifier is deterministic and lands far below; a wrong one is far
# above). df = vocab - 1.
_CHI2_CRIT = {7: 24.322, 15: 37.697}


def _chi2(counts: torch.Tensor, expected_probs: torch.Tensor, n: int) -> float:
    expected = expected_probs * n
    return float((((counts - expected) ** 2) / expected).sum())


def _first_emitted_counts(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    *,
    n_trials: int,
    seed: int,
    broken: bool = False,
) -> torch.Tensor:
    """Marginal of the first emitted token over ``n_trials`` speculative steps.

    K = 1: draft proposes one token from ``draft_probs``; the target verifies it
    against ``target_probs`` (row 0 = verify position, row 1 = bonus). If
    ``broken`` we skip verification and always keep the draft token — the marginal
    should then be the *draft* distribution and fail the target chi-squared.
    """
    vocab = target_probs.shape[-1]
    gen = torch.Generator().manual_seed(seed)
    counts = torch.zeros(vocab, dtype=torch.float64)
    p_block = target_probs.expand(2, vocab).contiguous()  # [K+1=2, vocab]
    q_block = draft_probs.view(1, vocab)  # [K=1, vocab]
    for _ in range(n_trials):
        draft_tok = torch.multinomial(draft_probs, 1, generator=gen)
        if broken:
            first = int(draft_tok.item())
        else:
            res = verify_tokens(p_block, q_block, draft_tok, generator=gen)
            first = int(res.output_token_ids[0].item())
        counts[first] += 1
    return counts


def _dist(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    return Sampler().probs(logits.unsqueeze(0), params).squeeze(0)


def test_first_emitted_token_matches_target_distribution() -> None:
    # Target peaked, draft deliberately different (skewed the other way).
    target = _dist(torch.tensor([3.0, 1.0, 0.5, 0.0, -1.0, -1.0, 0.2, 0.7]), SamplingParams())
    draft = _dist(torch.tensor([-1.0, 0.0, 0.5, 2.0, 1.5, 0.3, -0.5, 0.1]), SamplingParams())

    n = 40_000
    counts = _first_emitted_counts(target, draft, n_trials=n, seed=2026)
    stat = _chi2(counts, target.double(), n)
    assert stat < _CHI2_CRIT[7], f"chi2={stat:.2f} — emitted marginal drifted from target"


def test_distribution_test_has_teeth_broken_verifier_fails() -> None:
    # Same setup, but the broken verifier keeps the draft token: its marginal is
    # the draft distribution, which must be *rejected* against the target.
    target = _dist(torch.tensor([3.0, 1.0, 0.5, 0.0, -1.0, -1.0, 0.2, 0.7]), SamplingParams())
    draft = _dist(torch.tensor([-1.0, 0.0, 0.5, 2.0, 1.5, 0.3, -0.5, 0.1]), SamplingParams())

    n = 40_000
    counts = _first_emitted_counts(target, draft, n_trials=n, seed=2026, broken=True)
    stat = _chi2(counts, target.double(), n)
    assert stat > _CHI2_CRIT[7], "broken verifier slipped past the chi-squared test"


def test_identical_target_and_draft_accepts_whole_block() -> None:
    # p == q ⇒ acceptance ratio 1 everywhere ⇒ every draft kept + a bonus token.
    probs = _dist(torch.randn(16), SamplingParams(seed=1))
    k = 4
    target = probs.expand(k + 1, 16).contiguous()
    draft = probs.expand(k, 16).contiguous()
    gen = torch.Generator().manual_seed(0)
    draft_tokens = torch.multinomial(probs, k, replacement=True, generator=gen)
    res = verify_tokens(target, draft, draft_tokens, generator=gen)
    assert res.num_accepted == k
    assert res.output_token_ids.shape[0] == k + 1
    torch.testing.assert_close(res.output_token_ids[:k], draft_tokens.to(torch.long))


def test_greedy_target_emits_target_argmax_sequence() -> None:
    # Greedy target = one-hot distributions. A draft token is kept iff it equals
    # the target's argmax at that position; otherwise the argmax is resampled.
    # Either way the emitted token is the target argmax — deterministic.
    vocab = 10
    torch.manual_seed(3)
    target_logits = torch.randn(5, vocab)  # K+1 = 5 positions
    greedy = SamplingParams(temperature=0.0)
    target = torch.stack([_dist(target_logits[i], greedy) for i in range(5)])
    argmaxes = target_logits.argmax(dim=-1)

    # Draft proposes a mix of correct and wrong tokens.
    draft_tokens = argmaxes[:4].clone()
    draft_tokens[2] = (argmaxes[2] + 1) % vocab  # force a mismatch at position 2
    draft = torch.zeros(4, vocab)
    draft.scatter_(-1, draft_tokens.view(-1, 1), 1.0)

    res = verify_tokens(target, draft, draft_tokens)
    # Positions 0,1 match → accepted; position 2 mismatches → reject + resample
    # the target argmax, then stop. Emitted = argmax(0), argmax(1), argmax(2).
    assert res.num_accepted == 2
    expected = argmaxes[:3].to(torch.long)
    torch.testing.assert_close(res.output_token_ids, expected)


def test_acceptance_rate_tracks_distribution_overlap() -> None:
    # Closer draft ⇒ higher expected acceptance. Averaged over trials.
    vocab = 12
    target = _dist(torch.tensor([2.0, 1.5, 1.0, 0.5] + [0.0] * 8), SamplingParams())
    close = target.clone()
    far = _dist(torch.tensor([0.0] * 8 + [1.0, 1.5, 2.0, 0.5]), SamplingParams())

    def mean_accept(draft: torch.Tensor, seed: int) -> float:
        gen = torch.Generator().manual_seed(seed)
        k = 4
        tgt = target.expand(k + 1, vocab).contiguous()
        drf = draft.expand(k, vocab).contiguous()
        total = 0
        trials = 3000
        for _ in range(trials):
            toks = torch.multinomial(draft, k, replacement=True, generator=gen)
            total += verify_tokens(tgt, drf, toks, generator=gen).num_accepted
        return total / trials

    assert mean_accept(close, 7) > mean_accept(far, 7)


def test_rejects_malformed_shapes() -> None:
    probs = _dist(torch.randn(8), SamplingParams(seed=1))
    with pytest.raises(ValueError):
        # target must have K+1 rows, not K.
        verify_tokens(
            probs.expand(3, 8).contiguous(),
            probs.expand(3, 8).contiguous(),
            torch.tensor([0, 1, 2]),
        )
    with pytest.raises(ValueError):
        verify_tokens(
            probs.view(1, 8), probs.view(1, 8), torch.tensor([0])
        )  # K=1 needs 2 target rows
