"""Greedy token identity vs HuggingFace generate(do_sample=False).

Both sides receive the same input_ids. HF is never given raw text, so it
cannot inject a BOS we did not encode. Tokenization is Slipstream if
present, otherwise AutoTokenizer.encode(..., add_special_tokens=False).

Expected Slipstream API:
  LLMEngine(EngineConfig.for_model(snapshot)).generate(Request(
      prompt=None,
      prompt_token_ids=ids,
      sampling_params=SamplingParams(temperature=0.0, max_tokens=N),
      ...
  )) -> list[int]   # output tokens only
Fallback if generate is still NotImplementedError: CausalLM.forward + argmax.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.correctness._api import (  # noqa: E402
    expand_default_prompts,
    hf_generate_greedy,
    load_hf_eager,
    load_slipstream_model,
    make_generate_request,
    make_llm_engine,
    slipstream_generate,
    tokenize_prompts,
)


def _first_mismatch(ours: list[int], hf: list[int]) -> str:
    limit = min(len(ours), len(hf))
    for i in range(limit):
        if ours[i] != hf[i]:
            return f"index {i}: ours={ours[i]} hf={hf[i]}"
    if len(ours) != len(hf):
        return f"length ours={len(ours)} hf={len(hf)}"
    return "none"


def _assert_greedy_identity(
    snapshot: Path,
    device: torch.device,
    *,
    n_prompts: int,
    max_new_tokens: int,
    label: str,
) -> None:
    prompts = expand_default_prompts(n_prompts)
    all_ids = tokenize_prompts(snapshot, prompts)

    # Slipstream first (fail-fast on stubs) so HF and SS are never coresident.
    engine = make_llm_engine(snapshot)
    ss_model = None
    probe = make_generate_request(all_ids[0], 1, f"{label}-probe")
    try:
        engine.generate(probe)
    except NotImplementedError:
        engine = None
        ss_model = load_slipstream_model(snapshot, device)

    ours_outs: list[list[int]] = []
    try:
        for i, ids in enumerate(all_ids):
            ours_outs.append(
                slipstream_generate(
                    snapshot,
                    ids,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    request_id=f"{label}-{i}",
                    engine=engine,
                    model=ss_model,
                )
            )
    finally:
        del engine, ss_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hf_model = load_hf_eager(snapshot, device)
    hf_outs: list[list[int]] = []
    try:
        for ids in all_ids:
            hf_outs.append(hf_generate_greedy(hf_model, ids, max_new_tokens, device))
    finally:
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    failures: list[str] = []
    for i, (prompt, ids, ours, hf_tokens) in enumerate(
        zip(prompts, all_ids, ours_outs, hf_outs, strict=True)
    ):
        if ours != hf_tokens:
            failures.append(
                f"[{i}] {prompt!r}\n"
                f"  input_ids={ids}\n"
                f"  first mismatch: {_first_mismatch(ours, hf_tokens)}\n"
                f"  ours ({len(ours)}): {ours}\n"
                f"  hf   ({len(hf_tokens)}): {hf_tokens}"
            )
    assert not failures, (
        f"{label}: {len(failures)}/{n_prompts} prompts diverged from HF greedy "
        f"(max_new_tokens={max_new_tokens}):\n" + "\n".join(failures)
    )


@pytest.mark.gpu
@pytest.mark.parity
def test_parity_greedy_qwen_8x32(qwen_snapshot, cuda_device) -> None:
    """Default CI: 8 prompts × 32 new tokens on Qwen2.5-0.5B."""
    _assert_greedy_identity(
        qwen_snapshot,
        cuda_device,
        n_prompts=8,
        max_new_tokens=32,
        label="qwen-8x32",
    )


@pytest.mark.gpu
@pytest.mark.parity
@pytest.mark.slow
def test_parity_greedy_qwen_50x128(qwen_snapshot, cuda_device) -> None:
    """Gate 1: 50 prompts × 128 tokens. Built by cycling DEFAULT_PROMPTS."""
    _assert_greedy_identity(
        qwen_snapshot,
        cuda_device,
        n_prompts=50,
        max_new_tokens=128,
        label="qwen-50x128",
    )


@pytest.mark.gpu
@pytest.mark.parity
def test_parity_greedy_tinyllama_4x16(tinyllama_snapshot, cuda_device) -> None:
    """Llama-family identity on TinyLlama-1.1B (T0 proxy). Smaller N for 6 GB."""
    _assert_greedy_identity(
        tinyllama_snapshot,
        cuda_device,
        n_prompts=4,
        max_new_tokens=16,
        label="tinyllama-4x16",
    )
