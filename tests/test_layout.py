"""The core holds no provider.

The providers live in their own repositories (``aiollm_openai``,
``aiollm_anthropic``) and stand on the extension API. The core's own
suite runs without them: only ``tests/docs`` (the examples of the site)
and ``tests/downstream`` (the providers against the extension API)
import a provider, and they run in the ``downstream`` job, where the
providers are installed.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WITH_PROVIDERS = (ROOT / "tests" / "docs", ROOT / "tests" / "downstream")
"""The folders that run with the providers installed."""


def imported(path: Path) -> Iterator[str]:
    """Yield every module a file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and not (node.level)
        ):
            yield node.module


def test_there_is_no_provider_in_the_core() -> None:
    for folder in (ROOT / "src" / "aiollm", ROOT / "tests"):
        assert list((folder / "providers").rglob("*.py")) == []


def test_the_core_suite_takes_no_provider() -> None:
    files = [
        path
        for path in (ROOT / "tests").rglob("*.py")
        if not any(path.is_relative_to(folder) for folder in WITH_PROVIDERS)
    ]

    found = [
        f"{path.relative_to(ROOT).as_posix()}: {module}"
        for path in files
        for module in imported(path)
        if module.startswith(("aiollm_", "aiollm.providers"))
    ]

    assert found == []
