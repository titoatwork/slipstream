"""NaiveKVCache + Sampler unit tests. No weights. CPU (CUDA device-stay extra)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slipstream.core.sampling_params import SamplingParams  # noqa: E402
from tests.correctness._api import load_naive_kv_cache, load_sampler  # noqa: E402


def _cpu() -> torch.device:
    return torch.device("cpu")


def _devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def _make_cache(
    *,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 4,
    max_batch: int = 1,
    max_len: int = 8,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
):
    NaiveKVCache = load_naive_kv_cache()
    return NaiveKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_batch=max_batch,
        max_len=max_len,
        dtype=dtype,
        device=device if device is not None else _cpu(),
    )


def test_naive_kv_cache_two_step_append_shapes_and_values() -> None:
    cache = _make_cache(max_len=8)
    assert cache.seq_len == 0

    k0_l0 = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    v0_l0 = k0_l0 + 100
    k0_l1 = k0_l0 + 200
    v0_l1 = k0_l0 + 300

    full_k, full_v = cache.update(0, k0_l0, v0_l0)
    assert cache.seq_len == 3
    assert full_k.shape == (1, 2, 3, 4)
    assert full_v.shape == (1, 2, 3, 4)
    torch.testing.assert_close(full_k, k0_l0)
    torch.testing.assert_close(full_v, v0_l0)
    # Spec: returned tensors are views into the cache buffers, not clones.
    assert full_k.untyped_storage().data_ptr() == cache.k_cache.untyped_storage().data_ptr()

    full_k1, full_v1 = cache.update(1, k0_l1, v0_l1)
    assert cache.seq_len == 3  # only layer 0 advances seq_len
    torch.testing.assert_close(full_k1, k0_l1)
    torch.testing.assert_close(full_v1, v0_l1)

    k1_l0 = torch.arange(1000, 1000 + 1 * 2 * 2 * 4, dtype=torch.float32).reshape(1, 2, 2, 4)
    v1_l0 = k1_l0 + 1
    k1_l1 = k1_l0 + 2
    v1_l1 = k1_l0 + 3

    full_k, full_v = cache.update(0, k1_l0, v1_l0)
    assert cache.seq_len == 5
    assert full_k.shape == (1, 2, 5, 4)
    torch.testing.assert_close(full_k[..., :3, :], k0_l0)
    torch.testing.assert_close(full_k[..., 3:, :], k1_l0)
    torch.testing.assert_close(full_v[..., :3, :], v0_l0)
    torch.testing.assert_close(full_v[..., 3:, :], v1_l0)

    full_k1, full_v1 = cache.update(1, k1_l1, v1_l1)
    assert cache.seq_len == 5
    torch.testing.assert_close(full_k1[..., :3, :], k0_l1)
    torch.testing.assert_close(full_k1[..., 3:, :], k1_l1)
    torch.testing.assert_close(full_v1[..., :3, :], v0_l1)
    torch.testing.assert_close(full_v1[..., 3:, :], v1_l1)


def test_naive_kv_cache_overflow_raises_value_error() -> None:
    cache = _make_cache(max_len=4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    cache.update(0, k, v)
    cache.update(1, k, v)
    assert cache.seq_len == 3

    with pytest.raises(ValueError):
        cache.update(0, torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))
    assert cache.seq_len == 3


def test_naive_kv_cache_reset_zeros_length() -> None:
    cache = _make_cache(max_len=8)
    k = torch.randn(1, 2, 4, 4)
    v = torch.randn(1, 2, 4, 4)
    cache.update(0, k, v)
    cache.update(1, k, v)
    assert cache.seq_len == 4
    cache.reset()
    assert cache.seq_len == 0
    full_k, _ = cache.update(0, k[..., :2, :], v[..., :2, :])
    assert cache.seq_len == 2
    assert full_k.shape == (1, 2, 2, 4)


def test_naive_kv_cache_truncate_rolls_back_and_overwrites() -> None:
    # Rollback of rejected speculative positions (S8): truncate then re-append
    # must land on the truncated slots and read back the new values.
    cache = _make_cache(max_len=8)
    k0 = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    v0 = k0 + 100
    cache.update(0, k0, v0)
    cache.update(1, k0 + 200, v0 + 200)
    assert cache.seq_len == 4

    cache.truncate(2)
    assert cache.seq_len == 2

    # Next append starts exactly at the truncation point (layer 0 then layer 1).
    k_new = torch.full((1, 2, 3, 4), 7.0)
    full_k, _ = cache.update(0, k_new, k_new + 1)
    assert cache.seq_len == 5
    assert full_k.shape == (1, 2, 5, 4)
    torch.testing.assert_close(full_k[..., :2, :], k0[..., :2, :])  # kept prefix
    torch.testing.assert_close(full_k[..., 2:, :], k_new)  # overwrote past the cut
    cache.update(1, k_new + 2, k_new + 3)  # layer-1 consistency check must pass
    assert cache.seq_len == 5


def test_naive_kv_cache_truncate_rejects_out_of_range() -> None:
    cache = _make_cache(max_len=8)
    k = torch.randn(1, 2, 3, 4)
    cache.update(0, k, k + 1)
    cache.update(1, k, k + 1)
    assert cache.seq_len == 3
    with pytest.raises(ValueError):
        cache.truncate(4)  # cannot extend
    with pytest.raises(ValueError):
        cache.truncate(-1)
    cache.truncate(3)  # no-op is allowed
    assert cache.seq_len == 3


def test_sampler_greedy_is_argmax() -> None:
    sampler = load_sampler()
    logits = torch.tensor(
        [[0.1, 5.0, -1.0, 4.9], [2.0, 2.0, 2.1, -3.0]],
        dtype=torch.float32,
    )
    out = sampler.sample(logits, SamplingParams(temperature=0.0))
    assert out.dtype == torch.int64
    assert out.shape == (2,)
    torch.testing.assert_close(out, torch.argmax(logits, dim=-1))

    # top_k == 1 is also greedy (SamplingParams.is_greedy).
    out_k = sampler.sample(logits, SamplingParams(temperature=1.0, top_k=1))
    torch.testing.assert_close(out_k, torch.argmax(logits, dim=-1))


def test_sampler_same_seed_identical_tokens() -> None:
    sampler = load_sampler()
    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    params = SamplingParams(temperature=0.8, seed=2026)
    a = sampler.sample(logits, params)
    b = sampler.sample(logits, params)
    assert torch.equal(a, b)
    assert a.dtype == torch.int64
    assert a.shape == (4,)


@pytest.mark.parametrize("device", _devices())
def test_sampler_result_stays_on_logits_device(device: torch.device) -> None:
    sampler = load_sampler()
    logits = torch.randn(3, 17, device=device)
    greedy = sampler.sample(logits, SamplingParams(temperature=0.0))
    assert greedy.device.type == device.type
    assert greedy.device == logits.device
    sampled = sampler.sample(logits, SamplingParams(temperature=0.9, seed=7))
    assert sampled.device == logits.device
    assert sampled.dtype == torch.int64


def test_capped_new_tokens_prevents_context_overflow() -> None:
    from slipstream.engine.llm_engine import capped_new_tokens

    assert capped_new_tokens(4090, 16, 4096) == 6
    assert capped_new_tokens(100, 16, 4096) == 16
    assert capped_new_tokens(4096, 16, 4096) == 0
    assert capped_new_tokens(8, 0, 4096) == 0


def test_packed_causal_mask_prefill_and_decode() -> None:
    from slipstream.models.layers.attention import packed_causal_mask

    prefill = packed_causal_mask(4, 4, torch.float32, torch.device("cpu"))
    assert prefill.shape == (1, 1, 4, 4)
    # lower triangle allowed (0), strict upper -inf
    assert torch.isneginf(prefill[0, 0, 0, 1])
    assert prefill[0, 0, 3, 0] == 0
    decode = packed_causal_mask(1, 5, torch.float32, torch.device("cpu"))
    assert decode.shape == (1, 1, 1, 5)
    assert not torch.isneginf(decode).any()
