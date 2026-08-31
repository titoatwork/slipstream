"""GPU-side sampler. Phase 1 (greedy + filters) / Phase 4 (no host sync)."""

from __future__ import annotations

import torch

from slipstream.core.sampling_params import SamplingParams


class Sampler:
    """Greedy / temperature / top-k / top-p / min-p / repetition penalty.

    Invariants:
      I6.1  same seed + same params ⇒ identical output
      I6.2  no ``.item()`` / ``.cpu()`` in the decode hot path (Phase 4)

    Non-greedy filter order: repetition → temperature → top_k → top_p →
    min_p → softmax → multinomial. ``top_p`` / ``min_p`` / the final
    softmax run in float32; masks are written back onto the working logits.

    ``repetition_penalty`` needs previous token ids, which the frozen
    signature does not carry. Pass them as the optional kw-only
    ``token_ids`` (``[B, T]`` int64, same device as ``logits``). If
    ``repetition_penalty != 1.0`` and ``token_ids is None``, the penalty
    is a no-op — the engine will apply it in a later phase once it
    threads history through this call. Greedy (``params.is_greedy``) is
    always argmax of the incoming logits; filters are skipped.

    Seeded sampling: if ``generator`` is omitted and ``params.seed`` is
    set, a ``torch.Generator`` is built on ``logits.device`` (CUDA
    generator when logits are on CUDA, CPU generator when they are on
    CPU) and ``manual_seed``-ed. ``torch.multinomial`` requires that
    generator and ``probs`` share a device.
    """

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        *,
        generator: torch.Generator | None = None,
        token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample one token per batch row.

        Args:
            logits: Unnormalized scores, shape ``[B, vocab]``.
            params: Per-request sampling configuration.
            generator: Optional RNG. Wins over ``params.seed`` when given.
            token_ids: Optional previous ids ``[B, T]`` for repetition penalty.

        Returns:
            Token ids, shape ``[B]``, dtype int64, same device as ``logits``.
        """
        if params.is_greedy:
            return torch.argmax(logits, dim=-1)

        if generator is None and params.seed is not None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(params.seed)

        # `probs` applies the same filter chain and float32 softmax the sampler
        # has always used; sampling from it is behaviourally unchanged.
        probs = self.probs(logits, params, token_ids=token_ids)
        sampled = torch.multinomial(probs, num_samples=1, generator=generator)
        return sampled.squeeze(-1)

    def probs(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        *,
        token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Post-filter sampling distribution — the mass ``sample`` draws from.

        Returns ``[B, vocab]`` float32 rows that sum to 1, applying the exact
        filter order ``sample`` uses (repetition → temperature → top_k → top_p
        → min_p → softmax). Greedy params (``params.is_greedy``) collapse to a
        one-hot at ``argmax``, i.e. the point mass greedy decoding samples.

        Speculative verification needs the target's and draft's *distributions*,
        not one sample from each; routing both through this method guarantees
        they are filtered identically, which the rejection-sampling correctness
        proof (I8.1) depends on.
        """
        if params.is_greedy:
            out = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device)
            out.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
            return out

        if params.repetition_penalty != 1.0 and token_ids is not None:
            logits = _apply_repetition_penalty(logits, token_ids, params.repetition_penalty)

        logits = logits / params.temperature
        if params.top_k > 0:
            logits = _apply_top_k(logits, params.top_k)
        if params.top_p < 1.0:
            logits = _apply_top_p(logits, params.top_p)
        if params.min_p > 0.0:
            logits = _apply_min_p(logits, params.min_p)

        # float32 softmax so multinomial sees stable, renormalized mass.
        return torch.softmax(logits.float(), dim=-1)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """HF-style: score < 0 → score * penalty, else score / penalty."""
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(-1)
    out = logits.clone()
    gathered = out.gather(-1, token_ids)
    adjusted = torch.where(gathered < 0, gathered * penalty, gathered / penalty)
    return out.scatter(-1, token_ids, adjusted)


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    vocab = logits.size(-1)
    if top_k >= vocab:
        return logits
    kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus: keep the smallest descending-softmax prefix with cumsum >= top_p."""
    # Softmax/cumsum in fp32; scatter the mask back onto the working logits.
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Token i is dropped once the mass *before* it already meets top_p.
    # Index 0 has preceding mass 0, so at least one token always survives.
    remove_sorted = (cumsum - sorted_probs) >= top_p
    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def _apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    p_max = probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < min_p * p_max, float("-inf"))
