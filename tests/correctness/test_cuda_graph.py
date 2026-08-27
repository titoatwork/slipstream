"""CudaGraphPool: CPU stores fn; GPU capture/replay matches eager."""

from __future__ import annotations

import pytest
import torch
from slipstream.engine.cuda_graph import CudaGraphPool


def test_cpu_pool_stores_fn_and_replay_calls_it() -> None:
    pool = CudaGraphPool(enabled=False)
    calls: list[int] = []

    def fn() -> int:
        calls.append(1)
        return 42

    pool.capture(1, fn)
    assert calls == []
    assert pool.has(1)
    assert not pool.is_graph(1)
    assert pool.replay(1) == 42
    assert calls == [1]


def test_cpu_copy_in_before_replay() -> None:
    pool = CudaGraphPool(enabled=False)
    buf = torch.zeros(3)
    out = torch.zeros(3)

    def fn() -> torch.Tensor:
        out.copy_(buf * 2)
        return out

    pool.capture(3, fn, static_inputs={"buf": buf})
    pool.replay(3, copy_in={"buf": torch.tensor([1.0, 2.0, 3.0])})
    torch.testing.assert_close(out, torch.tensor([2.0, 4.0, 6.0]))


def test_gpu_capture_add_mul_matches_eager() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for graph capture")
    pool = CudaGraphPool()
    assert pool.enabled
    x = torch.randn(8, device="cuda")
    y = torch.randn(8, device="cuda")
    out = torch.empty_like(x)

    def fn() -> torch.Tensor:
        out.copy_(x.add(y).mul(2))
        return out

    pool.capture(8, fn, static_inputs={"x": x, "y": y})
    assert pool.has(8)
    x2 = torch.randn(8, device="cuda")
    y2 = torch.randn(8, device="cuda")
    eager = (x2 + y2) * 2
    pool.replay(8, copy_in={"x": x2, "y": y2})
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager)
    if pool.is_graph(8):
        assert pool.replay(8) is out
