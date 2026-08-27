"""Decode CUDA-graph capture / replay for static (batch, T=1) shapes.

Do not capture Python-loop paged attention: block tables and seq lens
change every step, and the gather loop is host-side / not graph-safe.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch


class CudaGraphPool:
    """Static-shape graph pool keyed by `batch_size`.

    The callable must touch **only GPU tensors** (no Python host sync).
    First `capture` warms up, then records. `replay` copies into the
    captured static buffers if `copy_in` is provided.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        cuda = torch.cuda.is_available()
        self.enabled: bool = cuda if enabled is None else bool(enabled) and cuda
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._fns: dict[int, Callable[[], object]] = {}
        self._static_inputs: dict[int, dict[str, torch.Tensor]] = {}
        self._static_out: dict[int, object] = {}

    def capture(
        self,
        batch_size: int,
        fn: Callable[[], object],
        warmup: int = 3,
        static_inputs: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Record a static GPU-only `fn` for `batch_size`.

        `static_inputs` are the captured addresses `replay(..., copy_in=)` writes.
        CPU / `enabled=False` stores `fn` and does not record a graph.
        """
        self._fns[batch_size] = fn
        if static_inputs is not None:
            self._static_inputs[batch_size] = dict(static_inputs)
        if not self.enabled:
            return
        last: object = None
        try:
            for _ in range(max(warmup, 0)):
                last = fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                last = fn()
        except Exception:
            return
        self._graphs[batch_size] = graph
        self._static_out[batch_size] = last

    def replay(
        self,
        batch_size: int,
        copy_in: Mapping[str, torch.Tensor] | None = None,
    ) -> object:
        """Copy `copy_in` into static buffers, then replay the graph or call `fn`."""
        self._copy_in(batch_size, copy_in)
        graph = self._graphs.get(batch_size)
        if graph is not None:
            graph.replay()
            return self._static_out.get(batch_size)
        fn = self._fns.get(batch_size)
        if fn is None:
            raise KeyError(f"no captured graph for batch_size={batch_size}")
        return fn()

    def _copy_in(self, batch_size: int, copy_in: Mapping[str, torch.Tensor] | None) -> None:
        if not copy_in:
            return
        static = self._static_inputs.get(batch_size)
        if static is None:
            raise KeyError(f"copy_in given but no static_inputs for batch_size={batch_size}")
        for name, src in copy_in.items():
            dst = static.get(name)
            if dst is None:
                raise KeyError(f"unknown static input {name!r} for batch_size={batch_size}")
            dst.copy_(src)

    def has(self, batch_size: int) -> bool:
        return batch_size in self._graphs or batch_size in self._fns

    def is_graph(self, batch_size: int) -> bool:
        """True only if a CUDA graph was actually recorded."""
        return batch_size in self._graphs
