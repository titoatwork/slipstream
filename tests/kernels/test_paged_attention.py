"""Paged attention / reshape_and_cache vs the gather+eager reference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slipstream.kernels.attention_ref import (  # noqa: E402
    paged_attention_ref,
    reshape_and_cache_ref,
)
from slipstream.kernels.paged_attention import paged_attention_decode  # noqa: E402
from slipstream.kernels.reshape_and_cache import reshape_and_cache  # noqa: E402
from slipstream.memory.block_manager import BlockManagerImpl  # noqa: E402
from slipstream.memory.paged_cache import allocate_kv_cache  # noqa: E402


@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("seq_len", [1, 17, 64])
def test_reshape_and_gather_roundtrip(block_size: int, seq_len: int) -> None:
    torch.manual_seed(0)
    n_kv, d, layers = 2, 8, 1
    n_blocks = (seq_len + block_size - 1) // block_size + 2
    kv = allocate_kv_cache(
        layers, n_blocks, block_size, n_kv, d, dtype=torch.float32, device=torch.device("cpu")
    )
    bm = BlockManagerImpl(n_blocks, block_size, max_model_len=seq_len)
    from slipstream.core.types import Sequence

    seq = Sequence(seq_id=0, prompt_token_ids=list(range(seq_len)))
    bm.allocate(seq)
    slots = [bm.append_slot(seq) for _ in range(seq_len)]
    mapping = torch.tensor([b * block_size + o for b, o in slots], dtype=torch.long)
    k = torch.randn(seq_len, n_kv, d)
    v = torch.randn(seq_len, n_kv, d)
    reshape_and_cache(k, v, kv, mapping, 0, block_size)
    table = torch.tensor(seq.block_table + [-1] * 4, dtype=torch.int32)
    from slipstream.kernels.attention_ref import gather_kv_ref

    kg, vg = gather_kv_ref(kv, 0, table, seq_len, block_size)
    torch.testing.assert_close(kg[0].permute(1, 0, 2), k, atol=0, rtol=0)
    torch.testing.assert_close(vg[0].permute(1, 0, 2), v, atol=0, rtol=0)


@pytest.mark.parametrize("tq", [1, 8])
def test_paged_attention_matches_ref(tq: int) -> None:
    torch.manual_seed(1)
    block_size, n_q, n_kv, d = 16, 4, 2, 8
    seq_len = 32
    n_blocks = 4
    device = torch.device("cpu")
    kv = allocate_kv_cache(1, n_blocks, block_size, n_kv, d, dtype=torch.float32, device=device)
    from slipstream.core.types import Sequence

    bm = BlockManagerImpl(n_blocks, block_size, max_model_len=64)
    seq = Sequence(seq_id=0, prompt_token_ids=list(range(seq_len)))
    bm.allocate(seq)
    slots = [bm.append_slot(seq) for _ in range(seq_len)]
    mapping = torch.tensor([b * block_size + o for b, o in slots], dtype=torch.long)
    k = torch.randn(1, n_kv, seq_len, d)
    v = torch.randn(1, n_kv, seq_len, d)
    reshape_and_cache_ref(k, v, kv, mapping, 0, block_size)
    q = torch.randn(1, n_q, tq, d)
    tables = torch.tensor([seq.block_table], dtype=torch.int32)
    lens = torch.tensor([seq_len], dtype=torch.int32)
    scale = d**-0.5
    ref = paged_attention_ref(q, kv, tables, lens, 0, scale, block_size)
    got = paged_attention_decode(q, kv, tables, lens, 0, scale, block_size)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
