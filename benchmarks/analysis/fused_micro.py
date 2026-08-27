"""Microbench fused RMSNorm/SwiGLU/RoPE and a static CUDA-graph decode stub.

python -m benchmarks.analysis.fused_micro
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from slipstream.engine.cuda_graph import CudaGraphPool
from slipstream.kernels.fused_rmsnorm import _torch_rmsnorm, _triton_rmsnorm, fused_rmsnorm
from slipstream.kernels.fused_rope import _torch_rope, _triton_rope, fused_rope
from slipstream.kernels.fused_swiglu import _torch_swiglu, _triton_swiglu, fused_swiglu
from slipstream.models.layers.rmsnorm import RMSNorm
from slipstream.models.layers.rope import apply_rotary_pos_emb

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "benchmarks" / "results" / "phase4"


def _time_cuda(fn: Callable[[], object], warmup: int = 20, iters: int = 80) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _skip_payload(reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    fused = {"skipped": True, "reason": reason, "cuda": False}
    graph = {"skipped": True, "reason": reason, "cuda": False}
    return fused, graph


def _match(a: torch.Tensor, b: torch.Tensor) -> bool:
    # fp16/bf16: same band as tests/correctness/test_layers_reference.py
    if a.dtype in (torch.float16, torch.bfloat16):
        atol = rtol = 2e-2
    else:
        atol = rtol = 1e-5
    return bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))


def _bench_fused(device: torch.device) -> dict[str, Any]:
    dtype = torch.float16
    rows: list[dict[str, Any]] = []

    def record(
        name: str,
        eager_fn: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
        fused_fn: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        eager_out = eager_fn()
        try:
            fused_out = fused_fn()
            triton_ok = True
            err = None
        except Exception as exc:
            fused_out = None
            triton_ok = False
            err = type(exc).__name__ + ": " + str(exc)
        if isinstance(eager_out, tuple):
            match = (
                triton_ok
                and isinstance(fused_out, tuple)
                and _match(eager_out[0], fused_out[0])
                and _match(eager_out[1], fused_out[1])
            )
        else:
            match = (
                triton_ok and isinstance(fused_out, torch.Tensor) and _match(eager_out, fused_out)
            )
        eager_ms = _time_cuda(eager_fn)
        fused_ms: float | None
        if triton_ok:
            fused_ms = _time_cuda(fused_fn)
            speedup = eager_ms / fused_ms if fused_ms else None
        else:
            fused_ms = None
            speedup = None
        rows.append(
            {
                "op": name,
                "eager_ms": eager_ms,
                "fused_ms": fused_ms,
                "speedup": speedup,
                "triton_ok": triton_ok,
                "numeric_match": match,
                "error": err,
            }
        )

    # RMSNorm: decode-like and a short prefill tile (Qwen2.5-0.5B hidden=896).
    x = torch.randn(8, 1, 896, device=device, dtype=dtype)
    w = torch.ones(896, device=device, dtype=dtype)
    r = torch.randn_like(x)
    ref_mod = RMSNorm(896, 1e-6, device=device, dtype=dtype)
    with torch.no_grad():
        ref_mod.weight.copy_(w)
    record(
        "rmsnorm_decode_b8",
        lambda: _torch_rmsnorm(x, w),
        lambda: _triton_rmsnorm(x, w),
    )
    record(
        "rmsnorm_residual_b8",
        lambda: _torch_rmsnorm(x, w, residual=r),
        lambda: _triton_rmsnorm(x, w, residual=r),
    )
    x_pre = torch.randn(4, 128, 896, device=device, dtype=dtype)
    w_pre = torch.ones(896, device=device, dtype=dtype)
    record(
        "rmsnorm_prefill_4x128",
        lambda: _torch_rmsnorm(x_pre, w_pre),
        lambda: _triton_rmsnorm(x_pre, w_pre),
    )
    # Public API with flag off must stay the eager module path.
    env_prev = os.environ.get("SLIPSTREAM_FUSED")
    os.environ["SLIPSTREAM_FUSED"] = "0"
    try:
        pub = fused_rmsnorm(x, ref_mod.weight)
        rows.append(
            {
                "op": "rmsnorm_public_eager_matches_module",
                "match": _match(pub, ref_mod(x)),
            }
        )
    finally:
        if env_prev is None:
            os.environ.pop("SLIPSTREAM_FUSED", None)
        else:
            os.environ["SLIPSTREAM_FUSED"] = env_prev

    g = torch.randn(8, 1, 4864, device=device, dtype=dtype)
    u = torch.randn_like(g)
    record("swiglu_decode_b8", lambda: _torch_swiglu(g, u), lambda: _triton_swiglu(g, u))
    g2 = torch.randn(4, 128, 2048, device=device, dtype=dtype)
    u2 = torch.randn_like(g2)
    record("swiglu_prefill_4x128", lambda: _torch_swiglu(g2, u2), lambda: _triton_swiglu(g2, u2))
    os.environ["SLIPSTREAM_FUSED"] = "0"
    try:
        rows.append(
            {
                "op": "swiglu_public_eager_matches_silu",
                "match": _match(fused_swiglu(g, u), F.silu(g) * u),
            }
        )
    finally:
        if env_prev is None:
            os.environ.pop("SLIPSTREAM_FUSED", None)
        else:
            os.environ["SLIPSTREAM_FUSED"] = env_prev

    q = torch.randn(8, 14, 1, 64, device=device, dtype=dtype)
    k = torch.randn(8, 2, 1, 64, device=device, dtype=dtype)
    cos = torch.randn(8, 1, 64, device=device, dtype=dtype)
    sin = torch.randn(8, 1, 64, device=device, dtype=dtype)
    record(
        "rope_decode_b8",
        lambda: _torch_rope(q, k, cos, sin)[0],
        lambda: _triton_rope(q, k, cos, sin)[0],
    )
    q2 = torch.randn(2, 14, 128, 64, device=device, dtype=dtype)
    k2 = torch.randn(2, 2, 128, 64, device=device, dtype=dtype)
    cos2 = torch.randn(2, 128, 64, device=device, dtype=dtype)
    sin2 = torch.randn(2, 128, 64, device=device, dtype=dtype)
    record(
        "rope_prefill_2x128",
        lambda: _torch_rope(q2, k2, cos2, sin2)[0],
        lambda: _triton_rope(q2, k2, cos2, sin2)[0],
    )
    os.environ["SLIPSTREAM_FUSED"] = "0"
    try:
        fq, fk = fused_rope(q, k, cos, sin)
        rq, rk = apply_rotary_pos_emb(q, k, cos, sin)
        rows.append(
            {
                "op": "rope_public_eager_matches_apply",
                "match": _match(fq, rq) and _match(fk, rk),
            }
        )
    finally:
        if env_prev is None:
            os.environ.pop("SLIPSTREAM_FUSED", None)
        else:
            os.environ["SLIPSTREAM_FUSED"] = env_prev

    timed = [r for r in rows if "speedup" in r and r.get("fused_ms") is not None]
    wins = [r for r in timed if r["speedup"] is not None and r["speedup"] > 1.0]
    return {
        "skipped": False,
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "triton_fused_beats_eager": bool(wins) and len(wins) == len(timed),
        "note": (
            "Triton is opt-in (SLIPSTREAM_FUSED=1). Default public API stays eager. "
            "RMSNorm/RoPE typically beat eager; SwiGLU decode may lose to ATen."
        ),
        "ops_triton_faster": [r["op"] for r in wins],
        "ops_eager_faster": [
            r["op"] for r in timed if r["speedup"] is not None and r["speedup"] <= 1.0
        ],
        "rows": rows,
    }


def _bench_graph(device: torch.device) -> dict[str, Any]:
    # Static decode stub: RMSNorm + Linear + Linear. No paged attention.
    dtype = torch.float16
    batch, hidden = 8, 896
    x = torch.randn(batch, 1, hidden, device=device, dtype=dtype)
    w = torch.ones(hidden, device=device, dtype=dtype)
    w1 = torch.randn(hidden, hidden, device=device, dtype=dtype)
    w2 = torch.randn(hidden, hidden, device=device, dtype=dtype)
    out = torch.empty_like(x)

    def eager() -> torch.Tensor:
        y = _torch_rmsnorm(x, w)
        y = F.linear(y, w1)
        return F.linear(y, w2)

    def graph_fn() -> torch.Tensor:
        y = _torch_rmsnorm(x, w)
        y = F.linear(y, w1)
        y = F.linear(y, w2)
        out.copy_(y)
        return out

    pool = CudaGraphPool()
    pool.capture(batch, graph_fn, warmup=5, static_inputs={"x": x})
    captured = pool.is_graph(batch)
    x2 = torch.randn_like(x)
    x.copy_(x2)
    eager_ref = eager()
    x.copy_(x2)
    replayed = pool.replay(batch)
    torch.cuda.synchronize()
    match = _match(out, eager_ref)
    eager_ms = _time_cuda(eager)
    if captured:
        graph_ms = _time_cuda(lambda: pool.replay(batch))
        copy_in_ms = _time_cuda(lambda: pool.replay(batch, copy_in={"x": x2}))
        speedup = eager_ms / graph_ms if graph_ms else None
    else:
        graph_ms = None
        copy_in_ms = None
        speedup = None
    return {
        "skipped": False,
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "fn": "rmsnorm + linear + linear (static decode, no paged attention)",
        "batch": batch,
        "hidden": hidden,
        "dtype": str(dtype),
        "captured": captured,
        "numeric_match": match,
        "eager_ms": eager_ms,
        "graph_ms": graph_ms,
        "graph_copy_in_ms": copy_in_ms,
        "speedup": speedup,
        "replay_returns_static_out": replayed is out,
    }


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        fused, graph = _skip_payload("CUDA not available")
    else:
        device = torch.device("cuda")
        fused = _bench_fused(device)
        graph = _bench_graph(device)
    (_OUT / "fused_micro.json").write_text(json.dumps(fused, indent=2) + "\n")
    (_OUT / "graph_micro.json").write_text(json.dumps(graph, indent=2) + "\n")
    print(json.dumps({"fused_micro": fused, "graph_micro": graph}, indent=2))


if __name__ == "__main__":
    main()
