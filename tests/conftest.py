"""Shared fixtures. SLIPSTREAM_DEBUG is on for every test."""

from __future__ import annotations

import os

import pytest

# Invariant checks must fire in the test suite (MASTERPLAN §14 Layer 4).
os.environ.setdefault("SLIPSTREAM_DEBUG", "1")


@pytest.fixture
def tiny_prompt_ids() -> list[int]:
    return [1, 2, 3, 4, 5]
