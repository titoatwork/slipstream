"""Environment capture for every published number (MASTERPLAN §7.4, §15)."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _safe(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip()


def _git_sha(repo: Path) -> str | None:
    return _safe(["git", "-C", str(repo), "rev-parse", "HEAD"])


def _nvidia_query() -> list[dict[str, str]]:
    raw = _safe(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap,clocks.gr,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
    )
    if raw is None:
        return []
    gpus: list[dict[str, str]] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_mib": parts[2],
                "driver": parts[3],
                "compute_cap": parts[4],
                "clock_graphics": parts[5] if len(parts) > 5 else "",
                "clock_memory": parts[6] if len(parts) > 6 else "",
            }
        )
    return gpus


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(name)
    except Exception:
        return None


def capture_run_manifest(**overrides: Any) -> dict[str, Any]:
    """Full environment snapshot. Every benchmark writes this next to results."""
    repo = Path(__file__).resolve().parents[1]
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_version": platform.python_version(),
        },
        "gpu": _nvidia_query(),
        "cuda": {
            "nvcc": _safe(["nvcc", "--version"]),
        },
        "packages": {
            "torch": _pkg_version("torch"),
            "triton": _pkg_version("triton"),
            "transformers": _pkg_version("transformers"),
            "numpy": _pkg_version("numpy"),
            "slipstream": _pkg_version("slipstream"),
        },
        "git_sha": _git_sha(repo),
        "clock_locked": None,
        "exclusive_gpu": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": None,
        "workload": None,
        "config": None,
        "notes": None,
    }
    manifest.update(overrides)
    return manifest


def write_run_manifest(path: str | Path, **overrides: Any) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(capture_run_manifest(**overrides), indent=2) + "\n")
    return dest
