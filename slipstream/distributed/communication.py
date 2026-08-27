"""NCCL / NVLink / RDMA transport selection. Phase 6 (A6)."""

from __future__ import annotations


def select_kv_transport() -> str:
    """Return 'nvlink' | 'rdma' | 'pcie' based on measured topology."""
    raise NotImplementedError("Phase 6 — S11 select_kv_transport")
