"""Debug-assertion toggle. On in tests, off in published benchmarks."""

from __future__ import annotations

import os

DEBUG: bool = os.environ.get("SLIPSTREAM_DEBUG", "0") not in {"0", "false", "False", ""}


def assert_debug(condition: bool, message: str) -> None:
    """Invariant check that is a no-op unless `SLIPSTREAM_DEBUG` is set."""
    if DEBUG and not condition:
        raise AssertionError(message)
