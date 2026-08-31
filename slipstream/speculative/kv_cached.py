"""KV-cache-backed speculative decoding over ModelRunner. Phase 5 (S8).

The perf path behind the reference loop in ``decode.py``. Where the reference
recomputes the target on every growing prefix (its ``ScoreFn`` is pure), this
runner keeps a live ``NaiveKVCache`` for both models and pays one incremental
forward per new token:

* **draft** proposes ``K`` tokens, one single-token forward each, appending its
  KV as it goes;
* **target** scores the whole ``K``-token block in **one batched forward** over
  ``[last, d0, …, d_{K-1}]`` — ``eager_attention``'s packed causal mask makes the
  block internally causal, so row ``r`` is the next-token distribution after
  ``last + d[:r]``, i.e. exactly the ``K+1`` rows the verifier needs;
* rejected (and unused speculative) positions are **rolled back** on both caches
  via ``NaiveKVCache.truncate`` so the next step starts from the committed prefix.

Verification is the shared, distribution-preserving ``verify_tokens`` and the
sampling distributions come from the shared ``Sampler.probs``, so this path emits
from exactly the target distribution the reference does. Its contract is
**token-for-token identity with the reference loop** (``test_speculative_kv_cached``
checks this on a real ``CausalLM``): the cache is an implementation detail, never
a change in what is emitted.

Cache invariant across the main loop: before a step over committed prefix length
``L``, both caches hold KV for positions ``[0, L-2]`` and ``last`` is the token at
position ``L-1`` (uncached — it is re-fed as the first query of the target block).
After emitting ``a`` accepted + 1 bonus/resample token, both caches are truncated
to ``L + a`` and ``last`` becomes the trailing emitted token, restoring the
invariant for length ``L + a + 1``.
"""

from __future__ import annotations

import torch

from slipstream.core.sampling_params import SamplingParams
from slipstream.engine.model_runner import ModelRunner
from slipstream.engine.sampler import Sampler
from slipstream.memory.contiguous_cache import NaiveKVCache
from slipstream.speculative.decode import SpeculativeStats
from slipstream.speculative.rejection import verify_tokens


class SpeculativeRunner:
    """Draft-then-verify decoding with live KV caches over two ``ModelRunner``s.

    Both runners must wrap a ``CausalLM`` whose ``config`` carries
    ``num_layers`` / ``num_kv_heads`` / ``head_dim`` (set at load) and share a
    tokenizer vocabulary. Single sequence (batch 1); continuous batching of
    speculative requests is a later concern.
    """

    def __init__(self, target: ModelRunner, draft: ModelRunner) -> None:
        if target.model is None or draft.model is None:
            raise ValueError("target and draft runners must each have a loaded model")
        if target.device != draft.device:
            raise ValueError(f"target/draft device mismatch: {target.device} vs {draft.device}")
        self.target = target
        self.draft = draft
        self.device = target.device

    def _make_cache(self, runner: ModelRunner, max_len: int) -> NaiveKVCache:
        cfg = runner.model.config  # type: ignore[union-attr]
        return NaiveKVCache(
            num_layers=_need(cfg.num_layers, "num_layers"),
            num_kv_heads=_need(cfg.num_kv_heads, "num_kv_heads"),
            head_dim=_need(cfg.head_dim, "head_dim"),
            max_batch=1,
            max_len=max_len,
            dtype=runner.dtype,
            device=runner.device,
        )

    def generate(
        self,
        prompt_ids: list[int] | torch.Tensor,
        *,
        params: SamplingParams,
        num_speculative_tokens: int,
        max_new_tokens: int,
        draft_params: SamplingParams | None = None,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], SpeculativeStats]:
        """Generate up to ``max_new_tokens`` tokens; mirrors ``speculative_generate``.

        Returns ``(output_token_ids, stats)``. The output is token-identical to the
        reference loop for the same models and params (greedy exactly; sampling in
        distribution). The bonus token cannot overshoot ``max_new_tokens`` — the
        return is truncated — and ``eos_token_id`` stops generation inclusively.
        """
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        if isinstance(prompt_ids, torch.Tensor):
            prompt = prompt_ids.to(dtype=torch.long).flatten().tolist()
        else:
            prompt = list(prompt_ids)
        if not prompt:
            raise ValueError("prompt_ids must be non-empty")

        draft_params = draft_params if draft_params is not None else params
        sampler = Sampler()
        stats = SpeculativeStats()
        output: list[int] = []
        device = self.device

        # Peak cache fill in a step is L + K (target block write); size for it.
        max_len = len(prompt) + max_new_tokens + num_speculative_tokens + 1
        target_cache = self._make_cache(self.target, max_len)
        draft_cache = self._make_cache(self.draft, max_len)

        # Prime: cache the prompt except its last token; `last` is re-fed each step.
        last = prompt[-1]
        if len(prompt) > 1:
            prime = torch.tensor([prompt[:-1]], dtype=torch.long, device=device)
            self.target.decode(prime, target_cache)
            self.draft.decode(prime, draft_cache)
        committed_len = len(prompt)

        while len(output) < max_new_tokens:
            length = committed_len
            k = min(num_speculative_tokens, max_new_tokens - len(output))

            # --- Draft proposes k tokens; caches `last`, d0, …, d_{k-1}. ---
            draft_tokens: list[int] = []
            draft_prob_rows: list[torch.Tensor] = []
            cur = last
            for _ in range(k):
                logits = self.draft.decode(_tok(cur, device), draft_cache)
                q = sampler.probs(logits[:, -1], draft_params).squeeze(0)
                tok = int(torch.multinomial(q, 1, generator=generator).item())
                draft_prob_rows.append(q)
                draft_tokens.append(tok)
                cur = tok
            # One extra forward so the last proposed token is cached too; this
            # makes the rollback rule uniform (truncate both caches to L + a
            # regardless of whether the block was fully accepted). Distribution
            # discarded — the bonus, if any, comes from the target.
            self.draft.decode(_tok(cur, device), draft_cache)

            # --- Target scores the whole block in one batched forward. ---
            block = torch.tensor([[last, *draft_tokens]], dtype=torch.long, device=device)
            logits = self.target.decode(block, target_cache)  # [1, k+1, vocab]
            target_probs = torch.stack(
                [sampler.probs(logits[:, r], params).squeeze(0) for r in range(k + 1)]
            )
            draft_probs = torch.stack(draft_prob_rows)
            draft_tok_t = torch.tensor(draft_tokens, dtype=torch.long, device=device)

            res = verify_tokens(target_probs, draft_probs, draft_tok_t, generator=generator)
            emitted = res.output_token_ids.tolist()
            accepted = res.num_accepted

            # --- Roll back speculative KV; re-establish the loop invariant. ---
            target_cache.truncate(length + accepted)
            draft_cache.truncate(length + accepted)
            last = emitted[-1]
            committed_len = length + accepted + 1

            stats.steps += 1
            stats.proposed += k
            stats.accepted += accepted
            stats.emitted += len(emitted)

            for tok in emitted:
                output.append(tok)
                if (eos_token_id is not None and tok == eos_token_id) or len(
                    output
                ) >= max_new_tokens:
                    return output[:max_new_tokens], stats

        return output[:max_new_tokens], stats


def _tok(token_id: int, device: torch.device) -> torch.Tensor:
    """Single-token ``[1, 1]`` batch for a one-step decode forward."""
    return torch.tensor([[token_id]], dtype=torch.long, device=device)


def _need(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"ModelConfig.{name} is required to size the KV cache")
    return value
