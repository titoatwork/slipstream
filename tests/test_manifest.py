"""run_manifest.json capture works without a GPU."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.manifest import capture_run_manifest, write_run_manifest


def test_capture_run_manifest_shape() -> None:
    manifest = capture_run_manifest(model="dummy", workload="unit")
    assert "timestamp" in manifest
    assert "platform" in manifest
    assert "gpu" in manifest
    assert manifest["model"] == "dummy"
    assert manifest["workload"] == "unit"
    assert "python_version" in manifest["platform"]


def test_write_run_manifest(tmp_path: Path) -> None:
    dest = write_run_manifest(tmp_path / "run_manifest.json", notes="gate0")
    data = json.loads(dest.read_text())
    assert data["notes"] == "gate0"
