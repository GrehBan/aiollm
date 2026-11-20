"""The providers against the extension API of this core.

Runs in the ``downstream`` job, where ``aiollm_openai`` and
``aiollm_anthropic`` are installed from their ``master`` on top of this
core: each takes only names of ``aiollm/testing/extension_api.json``, and
the snapshot lists no name that none of them (nor the echo provider)
takes, since a name nobody takes is not part of the contract yet.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from aiollm.testing.extension import EXTENSION_API, check_extension_api

PROVIDERS = ("aiollm_anthropic", "aiollm_openai", "aiollm.testing.echo")
"""Every provider package the core knows of."""


@pytest.mark.parametrize("package", PROVIDERS)
def test_providers_take_only_the_extension_api(package: str) -> None:
    assert check_extension_api(package) == []


def test_the_snapshot_lists_only_what_providers_take() -> None:
    taken = {
        (node.module, alias.name)
        for package in PROVIDERS
        for path in sorted(
            Path(importlib.import_module(package).__file__ or "").parent.rglob(
                "*.py"
            )
        )
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    listed = {
        (module, name)
        for module, names in EXTENSION_API.items()
        for name in names
    }

    assert listed - taken == set()
