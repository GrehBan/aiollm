"""Block P: the names re-exported by the packages."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGES = [
    "aiollm",
    "aiollm.adapters",
    "aiollm.client",
    "aiollm.http",
    "aiollm.http.exceptions",
    "aiollm.methods",
    "aiollm.resources",
    "aiollm.response",
    "aiollm.response.parse",
    "aiollm.serialization",
    "aiollm.session",
    "aiollm.session.body",
    "aiollm.session.retry",
    "aiollm.testing",
    "aiollm.testing.contract",
    "aiollm.testing.echo",
    "aiollm.types",
    "aiollm.types.requests",
    "aiollm.types.responses",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_all_names_exist_and_are_sorted(name: str) -> None:
    package = importlib.import_module(name)
    exported: list[str] = package.__all__

    assert exported == sorted(exported)
    assert all(hasattr(package, item) for item in exported)


SNAPSHOT = Path(__file__).with_name("public_api.json")
"""The public names of every package of the core, as the last release
left them; each provider keeps its own beside its tests."""


def test_the_public_api_matches_its_snapshot() -> None:
    # A public name added, renamed or removed must be a decision: update
    # public_api.json in the same change, with a changelog fragment.
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = {
        name: list(importlib.import_module(name).__all__) for name in PACKAGES
    }

    assert current == snapshot


def test_root_reexports_the_error_tree() -> None:
    import aiollm  # noqa: PLC0415 - checked against its subpackage
    from aiollm.http import exceptions  # noqa: PLC0415

    errors = set(exceptions.__all__) - {
        "BaseErrorResolver",
        "HTTP_ERRORS",
        "IErrorResolver",
    }

    assert errors <= set(aiollm.__all__)


@pytest.mark.parametrize("package", ["aiollm", "aiollm.testing"])
def test_importing_a_package_imports_no_transport_and_no_pytest(
    package: str,
) -> None:
    code = (
        f"import sys, {package}; "
        "print(sorted({'aiohttp', 'pytest'} & set(sys.modules)))"
    )

    result = subprocess.run(  # noqa: S603 - fixed command
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "[]"
