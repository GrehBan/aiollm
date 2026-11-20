"""The extension API: what a provider package may take from the core.

``extension_api.json`` lists, by module, every name a provider package
imports from ``aiollm``. A provider that takes a name outside it fails
here, so a new name is added to the snapshot in the same change and
shows in its diff; the core removes or changes a listed name only with a
**Breaking** changelog fragment, since the provider packages stand on
them. Provider packages do not import each other.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

import aiollm

ROOT = Path(aiollm.__file__).parent / "providers"

SNAPSHOT: dict[str, list[str]] = json.loads(
    Path(__file__).with_name("extension_api.json").read_text(encoding="utf-8")
)


def outside_names(
    package: str, source: str, allowed: dict[str, list[str]]
) -> list[str]:
    """The imports of a provider module that the extension API lacks.

    Args:
        package: The provider package the module belongs to, such as
            ``aiollm.providers.openai``.
        source: The module's source.
        allowed: The extension API, by module.

    Returns:
        Each wrong import as ``line: module.name``.
    """
    wrong = []

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            wrong += [
                f"{node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name.startswith("aiollm")
            ]
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        module = node.module or ""
        if not module.startswith("aiollm.") or module.startswith(package):
            continue
        wrong += [
            f"{node.lineno}: {module}.{alias.name}"
            for alias in node.names
            if module.startswith("aiollm.providers.")
            or alias.name not in allowed.get(module, [])
        ]

    return wrong


def provider_modules() -> list[tuple[str, Path]]:
    """Every module of every provider package, with its package."""
    return [
        (f"aiollm.providers.{package.name}", path)
        for package in sorted(ROOT.iterdir())
        if (package / "__init__.py").exists()
        for path in sorted(package.rglob("*.py"))
    ]


def test_the_snapshot_is_sorted() -> None:
    assert list(SNAPSHOT) == sorted(SNAPSHOT)
    assert all(names == sorted(set(names)) for names in SNAPSHOT.values())


@pytest.mark.parametrize("module", SNAPSHOT)
def test_every_listed_name_exists(module: str) -> None:
    imported = importlib.import_module(module)

    assert [n for n in SNAPSHOT[module] if not hasattr(imported, n)] == []


def test_providers_take_only_the_extension_api() -> None:
    wrong = [
        f"{path.relative_to(ROOT).as_posix()}:{found}"
        for package, path in provider_modules()
        for found in outside_names(
            package, path.read_text(encoding="utf-8"), SNAPSHOT
        )
    ]

    assert wrong == []


def test_the_snapshot_lists_only_what_providers_take() -> None:
    # A name nobody takes is not part of the contract yet.
    taken = {
        (node.module, alias.name)
        for _, path in provider_modules()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    listed = {
        (module, name) for module, names in SNAPSHOT.items() for name in names
    }

    assert listed - taken == set()


CASES = {
    "listed": ("from aiollm.enums import Provider", []),
    "own-package": ("from aiollm.providers.openai.api import PROVIDER", []),
    "relative": ("from .api import PROVIDER", []),
    "not-listed": (
        "from aiollm.enums import BodyFormat, Effort",
        ["1: aiollm.enums.Effort"],
    ),
    "unknown-module": (
        "from aiollm.methods.fields import names_missing",
        ["1: aiollm.methods.fields.names_missing"],
    ),
    "another-provider": (
        "from aiollm.providers.anthropic.api import PROVIDER",
        ["1: aiollm.providers.anthropic.api.PROVIDER"],
    ),
    "plain-import": ("import aiollm.enums", ["1: import aiollm.enums"]),
}
"""Sources the check is tried on, and what it must report."""


@pytest.mark.parametrize("case", CASES)
def test_the_check_tells_what_is_allowed(case: str) -> None:
    source, found = CASES[case]
    allowed = {"aiollm.enums": ["BodyFormat", "Provider"]}

    assert outside_names("aiollm.providers.openai", source, allowed) == found
