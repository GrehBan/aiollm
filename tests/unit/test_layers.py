"""Packages import only from the layers below them (CLAUDE.md).

The order is core → data and HTTP primitives → methods → response →
session → client → adapters → resources → testing. The exceptions are
the ones ``CLAUDE.md`` names: ``methods/base.py`` imports the default
parsers, and protocols that refer to each other import under
``TYPE_CHECKING``. Transport modules are never imported from an
``__init__.py``. The core imports no provider package (they live in
their own repositories), and its infrastructure (everything but the
shared generation and files layer and the test kit, which runs the
shared scenarios) does not import that shared layer, so the two could
be split apart.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import aiollm

ROOT = Path(aiollm.__file__).parent

LAYERS = {
    "__meta__": 0,
    "enums": 0,
    "exceptions": 0,
    "types": 1,
    "serialization": 1,
    "http": 1,
    "methods": 2,
    "response": 3,
    "session": 4,
    "client": 5,
    "adapters": 6,
    "resources": 7,
    "testing": 8,
}
"""The layer of each top-level package or module of ``aiollm``."""

ALLOWED = {("methods/base.py", "response.parse")}
"""Upward imports ``CLAUDE.md`` allows: the module and the target."""

TRANSPORTS = {"aiohttp"}
"""Names of transport modules (``<layer>/<transport>.py``)."""

SHARED = ("types.requests", "types.responses", "adapters", "resources")
"""The shared generation and files layer, as dotted targets."""


@dataclass(frozen=True)
class Import:
    """One ``from aiollm... import`` of a module."""

    module: str
    target: str
    line: int
    deferred: bool


def imports(module: str, source: str) -> list[Import]:
    """Find the ``aiollm`` imports of a module's source."""
    tree = ast.parse(source)
    deferred: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(
            node.test
        ):
            deferred.update(
                sub.lineno
                for sub in ast.walk(node)
                if isinstance(sub, ast.ImportFrom)
            )

    return [
        Import(
            module=module,
            target=node.module.removeprefix("aiollm."),
            line=node.lineno,
            deferred=node.lineno in deferred,
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("aiollm.")
    ]


def layer(name: str) -> int:
    """The layer of a module path or a dotted target."""
    return LAYERS[name.replace("/", ".").split(".")[0]]


def violations(found: list[Import]) -> list[str]:
    """Upward imports that no rule allows."""
    wrong = []

    for item in found:
        if layer(item.target) <= layer(item.module):
            continue
        if any(
            item.module == module and item.target.startswith(target)
            for module, target in ALLOWED
        ):
            continue
        if (
            item.deferred
            and item.module.endswith("protocols.py")
            and item.target.endswith("protocols")
        ):
            continue
        wrong.append(f"{item.module}:{item.line} imports {item.target}")

    return wrong


def transport_imports(found: list[Import]) -> list[str]:
    """Transport modules imported from an ``__init__.py``."""
    return [
        f"{item.module}:{item.line} imports {item.target}"
        for item in found
        if item.module.endswith("__init__.py")
        and item.target.split(".")[-1] in TRANSPORTS
    ]


def in_package(target: str, packages: tuple[str, ...]) -> bool:
    """Whether a module path or a dotted target is in one of the packages."""
    dotted = target.removesuffix(".py").replace("/", ".")
    return any(
        dotted == package or dotted.startswith(f"{package}.")
        for package in packages
    )


def shared_layer_imports(found: list[Import]) -> list[str]:
    """Imports of the shared layer by the infrastructure."""
    return [
        f"{item.module}:{item.line} imports {item.target}"
        for item in found
        if not in_package(item.module, (*SHARED, "testing"))
        and in_package(item.target, SHARED)
    ]


def provider_imports(module: str, source: str) -> list[str]:
    """Imports of a provider package by a module of the core."""
    found = []

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or ""]
        else:
            continue
        found += [
            f"{module}:{node.lineno} imports {name}"
            for name in names
            if name.startswith(("aiollm.providers", "aiollm_"))
        ]

    return found


def library_imports() -> list[Import]:
    """The ``aiollm`` imports of every module of the library."""
    found = []

    for path in sorted(ROOT.rglob("*.py")):
        module = path.relative_to(ROOT).as_posix()
        if module == "__init__.py":
            continue
        found += imports(module, path.read_text(encoding="utf-8"))

    return found


def test_every_package_has_a_layer() -> None:
    names = {
        path.stem if path.is_file() else path.name
        for path in ROOT.iterdir()
        if path.name != "__init__.py"
        and (path.suffix == ".py" or (path / "__init__.py").exists())
    }

    assert names == set(LAYERS)


def test_no_module_imports_from_a_higher_layer() -> None:
    assert violations(library_imports()) == []


def test_no_init_imports_a_transport() -> None:
    assert transport_imports(library_imports()) == []


def test_the_infrastructure_does_not_import_the_shared_layer() -> None:
    assert shared_layer_imports(library_imports()) == []


def test_the_core_imports_no_provider() -> None:
    found = [
        problem
        for path in sorted(ROOT.rglob("*.py"))
        for problem in provider_imports(
            path.relative_to(ROOT).as_posix(),
            path.read_text(encoding="utf-8"),
        )
    ]

    assert found == []


CASES = {
    "upward": (
        "client/base.py",
        ["from aiollm.resources.binding import X"],
        ["client/base.py:1 imports resources.binding"],
    ),
    "upward-outside-the-allowed-module": (
        "methods/fields.py",
        ["from aiollm.response.parse.sse import X"],
        ["methods/fields.py:1 imports response.parse.sse"],
    ),
    "deferred-but-no-protocol": (
        "session/base.py",
        ["if TYPE_CHECKING:", "    from aiollm.client.base import X"],
        ["session/base.py:2 imports client.base"],
    ),
    "allowed": (
        "methods/base.py",
        ["from aiollm.response.parse.sse import X"],
        [],
    ),
    "protocols-under-type-checking": (
        "methods/protocols.py",
        [
            "if TYPE_CHECKING:",
            "    from aiollm.response.parse.protocols import X",
        ],
        [],
    ),
    "downward": (
        "resources/base.py",
        ["from aiollm.session.base import X"],
        [],
    ),
}
"""Sources the check is tried on, as lines, and what it must report."""


@pytest.mark.parametrize("case", CASES)
def test_the_check_tells_what_is_allowed(case: str) -> None:
    # The check itself is tested, so a passing run means no violation.
    module, lines, found = CASES[case]

    assert violations(imports(module, "\n".join(lines))) == found


SHARED_CASES = {
    "infrastructure-imports-shared": (
        "client/base.py",
        "from aiollm.resources.generation import X",
        ["client/base.py:1 imports resources.generation"],
    ),
    "types-imports-requests": (
        "types/files.py",
        "from aiollm.types.requests.content import X",
        ["types/files.py:1 imports types.requests.content"],
    ),
    "shared-imports-shared": (
        "resources/files.py",
        "from aiollm.types.requests.upload import X",
        [],
    ),
    "shared-imports-infrastructure": (
        "types/requests/content.py",
        "from aiollm.types.files import X",
        [],
    ),
    "test-kit-imports-shared": (
        "testing/contract/files.py",
        "from aiollm.types.requests import X",
        [],
    ),
}
"""Sources the shared-layer check is tried on, and what it must report."""


@pytest.mark.parametrize("case", SHARED_CASES)
def test_the_check_keeps_the_shared_layer_out(case: str) -> None:
    module, source, found = SHARED_CASES[case]

    assert shared_layer_imports(imports(module, source)) == found


@pytest.mark.parametrize(
    ("source", "found"),
    [
        (
            "from aiollm.providers.openai import X",
            ["m.py:1 imports aiollm.providers.openai"],
        ),
        ("import aiollm_openai", ["m.py:1 imports aiollm_openai"]),
        (
            "from aiollm_anthropic.api import X",
            ["m.py:1 imports aiollm_anthropic.api"],
        ),
        ("from aiollm.enums import Provider", []),
        ("from .providers import X", []),
    ],
)
def test_the_check_finds_a_provider_import(
    source: str, found: list[str]
) -> None:
    assert provider_imports("m.py", source) == found


def test_the_check_finds_a_transport_in_an_init() -> None:
    source = "from aiollm.session.aiohttp import X"
    found = imports("session/__init__.py", source)

    assert transport_imports(found) == [
        "session/__init__.py:1 imports session.aiohttp"
    ]
