"""Reference speculative-decode loop: propose K, verify, emit. Phase 5 (S8).

Ties the draft proposer to the distribution-preserving verifier
(``rejection.verify_tokens``) into a full generation loop. It is written against
a minimal, model-agnostic contract — a ``ScoreFn`` mapping a token prefix to
next-token logits — so it can be exercised on CPU with toy models and no weights.

Because attention is causal, next-token logits for a prefix depend only on that
prefix, so evaluating the target on the growing prefixes ``prompt``,
``prompt+[d0]``, …, ``prompt+[d0..d_{K-1}]`` yields exactly the K+1 distributions
a single batched target forward would read off. This loop is therefore the
*correctness oracle*: the production engine path (one batched target forward over
the drafted block, KV-cache rollback of rejected positions) must reproduce its
output token-for-token. It is not the perf path — it recomputes rather than
caching — and is deliberately kept as the reference, mirroring
``kernels/attention_ref.py``.

Gold-standard property (``test_speculative_decode``): with greedy sampling, the
emitted sequence is **token-identical to plain greedy decoding from the target**,
for any draft — a correct draft only changes *how fast* tokens are produced, never
*which* tokens. Under sampling, each emitted token keeps the target marginal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from slipstream.core.sampling_params import SamplingParams
from slipstream.engine.sampler import Sampler
from slipstream.speculative.rejection import verify_tokens

# A model as far as speculative decoding is concerned: prefix ids [L] (int64,
# 1-D) -> next-token logits [vocab] (1-D). Pure in the prefix; any KV caching is
# an implementation detail invisible here.
ScoreFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class SpeculativeStats:
    """Per-generation acceptance accounting for the ablation table."""

    steps: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target kept. 0 when nothing proposed."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def mean_accepted_len(self) -> float:
        """Mean tokens emitted per speculative step — the decode-speedup driver."""
        return self.emitted / self.steps if self.steps else 0.0


def speculative_generate(
    prompt_ids: list[int] | torch.Tensor,
    *,
    target: ScoreFn,
    draft: ScoreFn,
    params: SamplingParams,
    num_speculative_tokens: int,
    max_new_tokens: int,
    draft_params: SamplingParams | None = None,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[list[int], SpeculativeStats]:
    """Generate up to ``max_new_tokens`` tokens by draft-then-verify speculation.

    Args:
        prompt_ids: Prompt token ids.
        target: The model whose distribution the output must match.
        draft: The cheap proposer. Its quality affects speed, never correctness.
        params: Target sampling params (greedy or stochastic).
        num_speculative_tokens: Draft block length ``K`` per step (>= 1).
        max_new_tokens: Hard cap on emitted tokens (the bonus token cannot
            overshoot it — the return is truncated).
        draft_params: Sampling params for the draft; defaults to ``params``.
        eos_token_id: If emitted, generation stops after it (inclusive).
        generator: RNG for reproducible proposal/verification draws.
        device: Compute device; defaults to the prompt tensor's or CPU.

    Returns:
        ``(output_token_ids, stats)``.
    """
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    if isinstance(prompt_ids, torch.Tensor):
        device = device or prompt_ids.device
        prefix = prompt_ids.to(device=device, dtype=torch.long).flatten().tolist()
    else:
        device = device or torch.device("cpu")
        prefix = list(prompt_ids)

    draft_params = draft_params if draft_params is not None else params
    sampler = Sampler()
    stats = SpeculativeStats()
    output: list[int] = []

    def _row(logits: torch.Tensor, p: SamplingParams) -> torch.Tensor:
        # ScoreFn may return [vocab] or [1, vocab]; normalise to a [vocab] prob row.
        flat = logits.reshape(-1, logits.shape[-1])[-1]
        return sampler.probs(flat.unsqueeze(0), p).squeeze(0)

    while len(output) < max_new_tokens:
        k = min(num_speculative_tokens, max_new_tokens - len(output))

        # --- Draft proposes k tokens autoregressively. ---
        draft_prob_rows: list[torch.Tensor] = []
        draft_tokens: list[int] = []
        work = list(prefix)
        for _ in range(k):
            ids = torch.tensor(work, dtype=torch.long, device=device)
            q = _row(draft(ids), draft_params)
            tok = int(torch.multinomial(q, 1, generator=generator).item())
            draft_prob_rows.append(q)
            draft_tokens.append(tok)
            work.append(tok)

        # --- Target scores the k+1 positions (prefix, prefix+d0, ..., +d_{k-1}). ---
        target_prob_rows: list[torch.Tensor] = []
        for i in range(k + 1):
            ids = torch.tensor(prefix + draft_tokens[:i], dtype=torch.long, device=device)
            target_prob_rows.append(_row(target(ids), params))

        draft_probs = torch.stack(draft_prob_rows)  # [k, vocab]
        target_probs = torch.stack(target_prob_rows)  # [k+1, vocab]
        draft_tok_t = torch.tensor(draft_tokens, dtype=torch.long, device=device)

        res = verify_tokens(target_probs, draft_probs, draft_tok_t, generator=generator)
        emitted = res.output_token_ids.tolist()

        stats.steps += 1
        stats.proposed += k
        stats.accepted += res.num_accepted
        stats.emitted += len(emitted)

        # --- Commit emitted tokens, honouring max_new_tokens and EOS. ---
        for tok in emitted:
            output.append(tok)
            prefix.append(tok)
            if (eos_token_id is not None and tok == eos_token_id) or len(output) >= max_new_tokens:
                return output[:max_new_tokens], stats

    return output[:max_new_tokens], stats
