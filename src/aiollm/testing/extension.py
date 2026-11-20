"""The extension API: what a provider package may take from the core.

``extension_api.json``, next to this module, lists by module every name
a provider package imports from ``aiollm``. A provider's tests call
``check_extension_api`` with its package: an import of any other name
of the core, of another provider or a plain ``import aiollm...`` is
reported, so a provider learns at once when it leans on something the
core does not promise to keep. The core removes or changes a listed
name only with a **Breaking** changelog fragment.

Example:
    >>> def test_the_package_takes_only_the_extension_api() -> None:
    ...     assert check_extension_api("aiollm_example") == []
"""

from __future__ import annotations

import ast
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

EXTENSION_API: Final[Mapping[str, Sequence[str]]] = json.loads(
    Path(__file__).with_name("extension_api.json").read_text(encoding="utf-8")
)
"""Every name of the extension API, by the module it is imported from."""

PROVIDER_PREFIXES: Final = ("aiollm_",)
"""Module prefixes of provider packages, which never import each other."""


def outside_names(
    package: str,
    source: str,
    allowed: Mapping[str, Sequence[str]] = EXTENSION_API,
) -> list[str]:
    """Find the imports of a provider module that the extension API lacks.

    Args:
        package: The provider package the module belongs to, such as
            ``aiollm_openai``; its own modules may be imported.
        source: The module's source.
        allowed: The extension API, by module.

    Returns:
        Each wrong import as ``line: module.name``, or ``line: import
        module`` for a plain import of the core.
    """
    wrong: list[str] = []

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            wrong += [
                f"{node.lineno}: import {alias.name}"
                for alias in node.names
                if _is_core(alias.name) or _is_other(alias.name, package)
            ]
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        module = node.module or ""
        if _is_own(module, package):
            continue
        if _is_other(module, package):
            wrong += [f"{node.lineno}: {module}.{a.name}" for a in node.names]
        elif _is_core(module):
            wrong += [
                f"{node.lineno}: {module}.{alias.name}"
                for alias in node.names
                if alias.name not in allowed.get(module, ())
            ]

    return wrong


def check_extension_api(
    package: str,
    allowed: Mapping[str, Sequence[str]] = EXTENSION_API,
) -> list[str]:
    """Check every module of a provider package against the extension API.

    Args:
        package: The importable name of the provider package, such as
            ``aiollm_openai``.
        allowed: The extension API, by module.

    Returns:
        Each wrong import as ``path:line: module.name``, the path
        relative to the package; empty when the package takes only the
        extension API.

    Raises:
        ValueError: The package has no source files of its own, such as
            a namespace package.
    """
    imported = importlib.import_module(package)
    if imported.__file__ is None:
        message = f"{package} has no source files of its own"
        raise ValueError(message)
    root = Path(imported.__file__).parent
    return [
        f"{path.relative_to(root).as_posix()}:{found}"
        for path in sorted(root.rglob("*.py"))
        for found in outside_names(
            package, path.read_text(encoding="utf-8"), allowed
        )
    ]


def _is_own(module: str, package: str) -> bool:
    """Tell whether a module is the package's own.

    Args:
        module: A dotted module name.
        package: The provider package.

    Returns:
        ``True`` for the package and its modules.
    """
    return module == package or module.startswith(f"{package}.")


def _is_core(module: str) -> bool:
    """Tell whether a module belongs to the core.

    Args:
        module: A dotted module name.

    Returns:
        ``True`` for ``aiollm`` and its modules.
    """
    return module == "aiollm" or module.startswith("aiollm.")


def _is_other(module: str, package: str) -> bool:
    """Tell whether a module belongs to another provider package.

    Args:
        module: A dotted module name.
        package: The provider package.

    Returns:
        ``True`` for a provider package that is not ``package``.
    """
    return not _is_own(module, package) and module.startswith(
        PROVIDER_PREFIXES
    )
