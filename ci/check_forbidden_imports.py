"""Fail if slipstream/ imports vllm, sglang, or transformers.

Reference implementations live only in tests/ and benchmarks/.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN = frozenset({"vllm", "sglang", "transformers"})


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _is_forbidden(mod: str | None) -> bool:
    return mod is not None and _root_module(mod) in FORBIDDEN


def find_violations(root: Path) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            hits.append((str(path), exc.lineno or 0, f"syntax error: {exc.msg}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name):
                        hits.append((str(path), node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom) and _is_forbidden(node.module):
                hits.append((str(path), node.lineno, f"from {node.module} import ..."))
    return hits


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    root = Path(args[0]) if args else Path(__file__).resolve().parents[1] / "slipstream"
    violations = find_violations(root)
    if not violations:
        print(f"ok: no forbidden imports under {root}")
        return 0
    print("FORBIDDEN IMPORTS (vllm / sglang / transformers) in slipstream/:", file=sys.stderr)
    for path, lineno, line in violations:
        print(f"  {path}:{lineno}: {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
