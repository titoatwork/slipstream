"""Last-token prefill logits vs HuggingFace eager (bf16, CUDA)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.correctness._api import (  # noqa: E402
    hf_prefill_last_logits,
    load_hf_eager,
    load_slipstream_model,
    slipstream_prefill_logits,
    tokenize_prompt,
)
from tests.correctness.conftest import PARITY_PROMPT  # noqa: E402


@pytest.mark.gpu
@pytest.mark.parity
def test_parity_logits_qwen_last_token(qwen_snapshot, cuda_device) -> None:
    """I1.1: last-token logits match HF eager within atol=rtol=1e-2."""
    token_ids = tokenize_prompt(qwen_snapshot, PARITY_PROMPT)
    input_ids = torch.tensor([token_ids], device=cuda_device, dtype=torch.long)

    # Load Slipstream first so a stub impl fails before we pull HF into VRAM.
    ss_model = load_slipstream_model(qwen_snapshot, cuda_device)
    with torch.inference_mode():
        ours = slipstream_prefill_logits(qwen_snapshot, input_ids, model=ss_model).float()

    hf_model = load_hf_eager(qwen_snapshot, cuda_device)
    with torch.inference_mode():
        hf = hf_prefill_last_logits(hf_model, input_ids).float()

    assert ours.shape == hf.shape, f"ours {tuple(ours.shape)} vs hf {tuple(hf.shape)}"
    max_abs = (ours - hf).abs().max().item()
    assert torch.allclose(ours, hf, atol=1e-2, rtol=1e-2), (
        f"last-token logits max abs diff={max_abs:.6g} (atol=1e-2, rtol=1e-2); "
        f"prompt={PARITY_PROMPT!r} n_tokens={len(token_ids)}"
    )
