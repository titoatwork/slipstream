"""KV-cache-backed speculative runner: token-identity with the reference. S8.

The reference loop (``speculative_generate``, ``decode.py``) recomputes the target
on every growing prefix; ``SpeculativeRunner`` keeps live KV caches and rolls back
rejected positions. The contract is that the cache is invisible: greedy output is
**token-identical** to the reference (and to plain greedy target decode) for any
draft and any K.

The primary test runs on a tiny random-weight ``CausalLM`` on CPU — no downloaded
weights, so it runs in CI — while exercising the real attention / RoPE / cache
path the rollback logic depends on. A second test asserts the same identity on
Qwen2.5-0.5B when its snapshot is present (skipped in CPU-only CI).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slipstream.core.config import ModelConfig  # noqa: E402
from slipstream.core.sampling_params import SamplingParams  # noqa: E402
from slipstream.engine.model_runner import ModelRunner  # noqa: E402
from slipstream.memory.contiguous_cache import NaiveKVCache  # noqa: E402
from slipstream.models.qwen import QwenForCausalLM  # noqa: E402
from slipstream.speculative.decode import ScoreFn, speculative_generate  # noqa: E402
from slipstream.speculative.kv_cached import SpeculativeRunner  # noqa: E402

CPU = torch.device("cpu")


def _tiny_model(seed: int) -> QwenForCausalLM:
    """A small random-weight Qwen-shaped CausalLM in fp32 on CPU (GQA, 2 layers)."""
    cfg = ModelConfig(
        model_id=f"tiny-spec-{seed}",
        dtype="float32",
        max_model_len=512,
        num_layers=2,
        hidden_size=64,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        vocab_size=48,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_type="default",
        model_type="qwen2",
        tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    model = QwenForCausalLM(cfg, device=CPU, dtype=torch.float32)
    model.eval()
    return model


def _recompute_scorefn(model: QwenForCausalLM) -> ScoreFn:
    """Reference ScoreFn: fresh cache, full recompute over the prefix (pure)."""
    cfg = model.config

    def score(prefix: torch.Tensor) -> torch.Tensor:
        ids = prefix.view(1, -1).to(CPU)
        length = ids.shape[1]
        cache = NaiveKVCache(
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            max_batch=1,
            max_len=length,
            dtype=torch.float32,
            device=CPU,
        )
        positions = torch.arange(length, device=CPU).view(1, -1)
        logits = model(ids, positions, cache)
        return logits[0, -1]

    return score


def _greedy_recompute(prompt: list[int], model: QwenForCausalLM, n: int) -> list[int]:
    score = _recompute_scorefn(model)
    prefix = list(prompt)
    out: list[int] = []
    for _ in range(n):
        tok = int(torch.argmax(score(torch.tensor(prefix, dtype=torch.long))).item())
        out.append(tok)
        prefix.append(tok)
    return out


def _runner(target: QwenForCausalLM, draft: QwenForCausalLM) -> SpeculativeRunner:
    return SpeculativeRunner(
        ModelRunner(ModelConfig(model_id="t"), model=target, device=CPU, dtype=torch.float32),
        ModelRunner(ModelConfig(model_id="d"), model=draft, device=CPU, dtype=torch.float32),
    )


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_kv_cached_greedy_matches_reference_and_target(k: int) -> None:
    target = _tiny_model(seed=1)
    draft = _tiny_model(seed=99)  # a *different* draft — must not corrupt output
    prompt = [3, 7, 1, 5]
    greedy = SamplingParams(temperature=0.0)
    n = 32

    ref, _ = speculative_generate(
        prompt,
        target=_recompute_scorefn(target),
        draft=_recompute_scorefn(draft),
        params=greedy,
        num_speculative_tokens=k,
        max_new_tokens=n,
    )
    out, stats = _runner(target, draft).generate(
        prompt,
        params=greedy,
        num_speculative_tokens=k,
        max_new_tokens=n,
    )

    assert out == ref  # cache path is token-identical to the recompute reference
    assert out == _greedy_recompute(prompt, target, n)  # and to plain target greedy
    assert stats.emitted == n
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_kv_cached_perfect_draft_accepts_everything() -> None:
    # draft is the target ⇒ every greedy proposal matches ⇒ full acceptance,
    # k accepted + 1 bonus per step. Exercises the all-accepted rollback branch.
    model = _tiny_model(seed=5)
    out, stats = _runner(model, model).generate(
        [2, 4],
        params=SamplingParams(temperature=0.0),
        num_speculative_tokens=4,
        max_new_tokens=30,
    )
    assert out == _greedy_recompute([2, 4], model, 30)
    assert stats.accepted == stats.proposed
    assert stats.acceptance_rate == 1.0
    assert stats.mean_accepted_len > 4.0


def test_kv_cached_respects_length_and_eos() -> None:
    target = _tiny_model(seed=2)
    draft = _tiny_model(seed=8)
    greedy = SamplingParams(temperature=0.0)
    runner = _runner(target, draft)

    # Exact length even though a block can emit k+1 (bonus cannot overshoot).
    for n in (1, 5, 7, 13):
        out, _ = runner.generate([1], params=greedy, num_speculative_tokens=4, max_new_tokens=n)
        assert len(out) == n

    # EOS stops inclusively at first occurrence.
    ref = _greedy_recompute([1], target, 40)
    eos = ref[6]
    out, _ = runner.generate(
        [1],
        params=greedy,
        num_speculative_tokens=4,
        max_new_tokens=40,
        eos_token_id=eos,
    )
    assert out[-1] == eos
    assert eos not in out[:-1]


def test_kv_cached_single_token_prompt() -> None:
    # len(prompt) == 1 skips the prime prefill; the cache starts empty.
    target = _tiny_model(seed=3)
    draft = _tiny_model(seed=17)
    greedy = SamplingParams(temperature=0.0)
    out, _ = _runner(target, draft).generate(
        [9], params=greedy, num_speculative_tokens=4, max_new_tokens=20
    )
    assert out == _greedy_recompute([9], target, 20)


def test_kv_cached_rejects_bad_arguments() -> None:
    model = _tiny_model(seed=1)
    runner = _runner(model, model)
    greedy = SamplingParams(temperature=0.0)
    with pytest.raises(ValueError):
        runner.generate([1], params=greedy, num_speculative_tokens=0, max_new_tokens=4)
    with pytest.raises(ValueError):
        runner.generate([1], params=greedy, num_speculative_tokens=4, max_new_tokens=0)
    with pytest.raises(ValueError):
        runner.generate([], params=greedy, num_speculative_tokens=4, max_new_tokens=4)


# ---------------------------------------------------------------------------
# Real-model fidelity: Qwen2.5-0.5B drafting for itself. Skipped without weights.
# ---------------------------------------------------------------------------


def test_kv_cached_matches_reference_on_qwen(qwen_snapshot) -> None:
    from tests.correctness._api import load_slipstream_model

    model = load_slipstream_model(qwen_snapshot, CPU)
    cfg = model.config
    if any(v is None for v in (cfg.num_layers, cfg.num_kv_heads, cfg.head_dim)):
        pytest.skip("model config missing KV dims")

    # Same model drafts and verifies (self-speculation): a correctness harness,
    # not a speedup one — most greedy proposals are accepted, driving the
    # all-accepted rollback path on real weights over many steps.
    def score(prefix: torch.Tensor) -> torch.Tensor:
        ids = prefix.view(1, -1).to(CPU)
        length = ids.shape[1]
        cache = NaiveKVCache(
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            max_batch=1,
            max_len=length,
            dtype=next(model.parameters()).dtype,
            device=CPU,
        )
        positions = torch.arange(length, device=CPU).view(1, -1)
        return model(ids, positions, cache)[0, -1]

    prompt = [40, 8, 25, 1043, 374]  # arbitrary in-vocab ids
    greedy = SamplingParams(temperature=0.0)
    n = 24

    ref, _ = speculative_generate(
        prompt,
        target=score,
        draft=score,
        params=greedy,
        num_speculative_tokens=4,
        max_new_tokens=n,
    )
    runner = SpeculativeRunner(
        ModelRunner(cfg, model=model, device=CPU, dtype=next(model.parameters()).dtype),
        ModelRunner(cfg, model=model, device=CPU, dtype=next(model.parameters()).dtype),
    )
    out, stats = runner.generate(prompt, params=greedy, num_speculative_tokens=4, max_new_tokens=n)
    assert out == ref  # the contract: cache path == recompute reference, exactly
    # Self-speculation: the draft *is* the target, but the draft's single-token
    # incremental forwards and the target's batched block forward are different
    # numerical paths in bf16, so a handful of greedy argmaxes diverge and are
    # corrected by the verifier — acceptance is high but need not be exactly 1.0.
    assert stats.acceptance_rate > 0.5
