"""Reference speculative-decode loop: end-to-end correctness (I8.1). CPU.

Gold standard: greedy speculative decoding is token-identical to plain greedy
decoding from the target, for *any* draft — the draft changes speed, never the
tokens. Also checks acceptance accounting, EOS / length stopping, and that a
one-step stochastic run keeps the target marginal end-to-end.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slipstream.core.sampling_params import SamplingParams  # noqa: E402
from slipstream.speculative.decode import ScoreFn, speculative_generate  # noqa: E402


def _toy_model(vocab: int, seed: int) -> ScoreFn:
    """Deterministic next-token logits as a function of the last token.

    A fixed [vocab, vocab] table indexed by the last prefix token. Deterministic
    in the prefix, so greedy decoding has a single well-defined trajectory.
    """
    gen = torch.Generator().manual_seed(seed)
    table = torch.randn(vocab, vocab, generator=gen)

    def score(prefix: torch.Tensor) -> torch.Tensor:
        return table[int(prefix[-1].item())]

    return score


def _greedy_reference(prompt: list[int], target: ScoreFn, n: int) -> list[int]:
    prefix = list(prompt)
    out: list[int] = []
    for _ in range(n):
        logits = target(torch.tensor(prefix, dtype=torch.long))
        tok = int(torch.argmax(logits).item())
        out.append(tok)
        prefix.append(tok)
    return out


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_greedy_speculative_is_token_identical_to_target_greedy(k: int) -> None:
    vocab = 32
    target = _toy_model(vocab, seed=1)
    # A *different* draft model — wrong proposals must not corrupt the output.
    draft = _toy_model(vocab, seed=99)
    prompt = [3, 7, 1]
    greedy = SamplingParams(temperature=0.0)

    out, stats = speculative_generate(
        prompt,
        target=target,
        draft=draft,
        params=greedy,
        num_speculative_tokens=k,
        max_new_tokens=40,
    )
    ref = _greedy_reference(prompt, target, 40)
    assert out == ref
    assert stats.emitted == 40
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_perfect_draft_accepts_everything() -> None:
    # Draft == target ⇒ greedy proposals always match ⇒ full acceptance every
    # step, and each step emits k accepted + 1 bonus.
    vocab = 24
    model = _toy_model(vocab, seed=5)
    out, stats = speculative_generate(
        [2, 4],
        target=model,
        draft=model,
        params=SamplingParams(temperature=0.0),
        num_speculative_tokens=4,
        max_new_tokens=40,
    )
    assert out == _greedy_reference([2, 4], model, 40)
    assert stats.accepted == stats.proposed  # nothing rejected
    assert stats.acceptance_rate == 1.0
    assert stats.mean_accepted_len > 4.0  # k accepted + bonus per step


def test_respects_max_new_tokens_and_eos() -> None:
    vocab = 16
    target = _toy_model(vocab, seed=2)
    draft = _toy_model(vocab, seed=8)
    greedy = SamplingParams(temperature=0.0)

    # Exact length even though blocks emit up to k+1 (bonus cannot overshoot).
    for n in (1, 5, 7, 13):
        out, _ = speculative_generate(
            [1],
            target=target,
            draft=draft,
            params=greedy,
            num_speculative_tokens=4,
            max_new_tokens=n,
        )
        assert len(out) == n

    # EOS stops generation inclusively at the first occurrence.
    ref = _greedy_reference([1], target, 40)
    eos = ref[6]
    out, _ = speculative_generate(
        [1],
        target=target,
        draft=draft,
        params=greedy,
        num_speculative_tokens=4,
        max_new_tokens=40,
        eos_token_id=eos,
    )
    assert out[-1] == eos
    assert eos not in out[:-1]  # first occurrence only


def test_one_step_sampling_keeps_target_marginal() -> None:
    # End-to-end distribution check through the whole loop: max_new_tokens=1, so
    # the single emitted token must be distributed as the target at the prompt.
    vocab = 8
    target = _toy_model(vocab, seed=4)
    draft = _toy_model(vocab, seed=41)  # deliberately mismatched proposer
    params = SamplingParams(temperature=1.0)
    prompt = [5]

    # Target marginal at the prompt (row indexed by last token).
    from slipstream.engine.sampler import Sampler

    p = Sampler().probs(target(torch.tensor(prompt)).unsqueeze(0), params).squeeze(0).double()

    n = 40_000
    gen = torch.Generator().manual_seed(2026)
    counts = torch.zeros(vocab, dtype=torch.float64)
    for _ in range(n):
        out, _ = speculative_generate(
            prompt,
            target=target,
            draft=draft,
            params=params,
            num_speculative_tokens=3,
            max_new_tokens=1,
            generator=gen,
        )
        counts[out[0]] += 1

    chi2 = float((((counts - p * n) ** 2) / (p * n)).sum())
    assert chi2 < 24.322, f"chi2={chi2:.2f} — end-to-end emitted marginal drifted"


def test_rejects_bad_arguments() -> None:
    model = _toy_model(8, seed=1)
    greedy = SamplingParams(temperature=0.0)
    with pytest.raises(ValueError):
        speculative_generate(
            [1],
            target=model,
            draft=model,
            params=greedy,
            num_speculative_tokens=0,
            max_new_tokens=4,
        )
    with pytest.raises(ValueError):
        speculative_generate(
            [1],
            target=model,
            draft=model,
            params=greedy,
            num_speculative_tokens=4,
            max_new_tokens=0,
        )
