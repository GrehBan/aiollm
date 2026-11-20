"""The extension API: what a provider package may take from the core.

``aiollm/testing/extension_api.json`` lists, by module, every name a
provider package imports from ``aiollm``. A provider that takes a name
outside it fails its own ``check_extension_api`` test, so a new name is
added to the snapshot in the same change and shows in its diff; the core
removes or changes a listed name only with a **Breaking** changelog
fragment, since the provider packages stand on them. Provider packages
do not import each other. The echo provider of the test kit is held to
it here; the providers of their own repositories, and the rule that the
snapshot lists nothing they do not take, in ``tests/downstream``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from aiollm.testing.extension import (
    EXTENSION_API,
    check_extension_api,
    outside_names,
)


def test_the_snapshot_is_sorted() -> None:
    assert list(EXTENSION_API) == sorted(EXTENSION_API)
    assert all(
        list(names) == sorted(set(names)) for names in EXTENSION_API.values()
    )


@pytest.mark.parametrize("module", EXTENSION_API)
def test_every_listed_name_exists(module: str) -> None:
    imported = importlib.import_module(module)

    assert [n for n in EXTENSION_API[module] if not hasattr(imported, n)] == []


def test_the_echo_provider_takes_only_the_extension_api() -> None:
    assert check_extension_api("aiollm.testing.echo") == []


CASES = {
    "listed": ("from aiollm.enums import Provider", []),
    "own-package": ("from aiollm_openai.api import PROVIDER", []),
    "own-package-itself": ("from aiollm_openai import PROVIDER", []),
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
        "from aiollm_anthropic.api import PROVIDER",
        ["1: aiollm_anthropic.api.PROVIDER"],
    ),
    "a-provider-of-the-core": (
        "from aiollm.providers.anthropic.api import PROVIDER",
        ["1: aiollm.providers.anthropic.api.PROVIDER"],
    ),
    "plain-import": ("import aiollm.enums", ["1: import aiollm.enums"]),
    "plain-import-of-another": (
        "import aiollm_anthropic",
        ["1: import aiollm_anthropic"],
    ),
    "plain-import-of-itself": ("import aiollm_openai.api", []),
    "other-libraries": ("import json\nfrom pydantic import MISSING", []),
}
"""Sources the check is tried on, and what it must report."""


@pytest.mark.parametrize("case", CASES)
def test_the_check_tells_what_is_allowed(case: str) -> None:
    source, found = CASES[case]
    allowed = {"aiollm.enums": ["BodyFormat", "Provider"]}

    assert outside_names("aiollm_openai", source, allowed) == found


def test_a_package_is_checked_file_by_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "aiollm_example"
    (package / "types").mkdir(parents=True)
    (package / "__init__.py").write_text("from aiollm.enums import Effort\n")
    (package / "types" / "__init__.py").write_text("")
    (package / "types" / "base.py").write_text(
        "from aiollm.enums import Provider\nimport aiollm_other\n"
    )
    monkeypatch.syspath_prepend(tmp_path)

    assert check_extension_api("aiollm_example") == [
        "__init__.py:1: aiollm.enums.Effort",
        "types/base.py:2: import aiollm_other",
    ]


def test_a_package_without_files_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "aiollm_namespace").mkdir()
    monkeypatch.syspath_prepend(tmp_path)

    with pytest.raises(ValueError, match="no source files"):
        check_extension_api("aiollm_namespace")
