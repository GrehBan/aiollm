"""The repository is laid out as the three repositories it will become.

Each provider's tests are ``tests/providers/<name>/`` and its scripts
``scripts/<name>/``: they import nothing of another provider, and the
core's tests import no provider's tests and, but for the examples of the
docs, no provider at all. So each folder moves out with its history
alone.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ("anthropic", "openai")


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


def test_the_providers_are_the_folders() -> None:
    folders = {
        path.name
        for path in (ROOT / "tests" / "providers").iterdir()
        if (path / "__init__.py").exists()
    }

    assert folders == set(PROVIDERS)
    scripts = {
        path.name
        for path in (ROOT / "scripts").iterdir()
        if path.is_dir() and not path.name.startswith(("_", "."))
    }
    assert scripts == set(PROVIDERS)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_provider_takes_nothing_of_another(provider: str) -> None:
    own = (f"tests.providers.{provider}", f"aiollm.providers.{provider}")
    files = [
        *(ROOT / "tests" / "providers" / provider).rglob("*.py"),
        *(ROOT / "scripts" / provider).rglob("*.py"),
    ]

    found = [
        f"{path.relative_to(ROOT).as_posix()}: {module}"
        for path in files
        for module in imported(path)
        if module.startswith(("tests", "aiollm.providers"))
        and not module.startswith(own)
    ]

    assert found == []


def test_the_core_takes_no_provider() -> None:
    files = [
        path
        for path in (ROOT / "tests").rglob("*.py")
        if not path.is_relative_to(ROOT / "tests" / "providers")
    ]

    found = [
        f"{path.relative_to(ROOT).as_posix()}: {module}"
        for path in files
        for module in imported(path)
        if module.startswith("tests.providers")
        or (
            module.startswith("aiollm.providers")
            and not path.is_relative_to(ROOT / "tests" / "docs")
        )
    ]

    assert found == []
