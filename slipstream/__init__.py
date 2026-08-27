"""Slipstream — high-throughput LLM inference with a memory-aware scheduler.

Public imports stay shallow. Importing this package must not pull torch,
Triton, or any baseline library (`vllm`, `sglang`, `transformers`).
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__"]
