"""slipstream/ must never import vllm, sglang, or transformers modeling code."""

from __future__ import annotations

from pathlib import Path

from ci.check_forbidden_imports import find_violations

REPO = Path(__file__).resolve().parents[1]


def test_no_forbidden_imports_in_engine() -> None:
    violations = find_violations(REPO / "slipstream")
    assert violations == [], "forbidden imports:\n" + "\n".join(
        f"  {path}:{lineno}: {line}" for path, lineno, line in violations
    )
