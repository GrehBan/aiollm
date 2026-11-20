"""Regenerate the provider types and request parameters with aiollm-codegen.

Run from the aiollm checkout:

    uv run python scripts/generate_provider_types.py [--codegen DIR]

Both inputs are pinned, so the same commit always gives the same files
(the CI job ``codegen`` regenerates them and fails on any difference).
The generator is the ``codegen`` dependency group, a commit of
aiollm-codegen named in ``pyproject.toml`` and ``uv.lock``; ``--codegen
DIR`` runs a local checkout instead, to try a change of the generator.
OpenAI's types come from its official OpenAPI file at the commit of
``OPENAI_OPENAPI_URL``, downloaded and checked against
``OPENAI_OPENAPI_SHA256`` (the file of the contract snapshot,
``tests/providers/contracts/manifest.json``, a test checks), or from the
file ``OPENAI_OPENAPI`` names, which must be the same; restricted to
``OPENAI_OPERATIONS``. Anthropic's come from the contract snapshot in
``tests/providers/contracts/anthropic/messages``, derived from the
pinned ``anthropic`` SDK, and, for the Files API, from
``tests/providers/contracts/sources/anthropic-files.openapi.json``, a
transcription of Anthropic's API reference. To follow a new
OpenAI file, refresh the contract snapshot from it and change the URL
and the hash here together.

For each provider the script replaces ``types/<family>.py`` and
``methods/params/`` with the generated modules, keeps the hand-written
``types/options.py`` and the method classes in ``methods/``, and writes
``types/__init__.py`` from the generated one with the options added.
Nothing else is touched.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
PROVIDERS: Final = ROOT / "src" / "aiollm" / "providers"
CONTRACTS: Final = ROOT / "tests" / "providers" / "contracts"
ANTHROPIC_VERSION: Final = "2026-10-21"

OPENAI_OPENAPI_URL: Final = (
    "https://raw.githubusercontent.com/openai/openai-openapi/"
    "76a4810f6cec796923bb6f4cc874237571a6305b/openapi.yaml"
)
"""OpenAI's OpenAPI file, at the commit of 2026-11-20."""

OPENAI_OPENAPI_SHA256: Final = (
    "e5666b742f332bc67a43f76f2988f21e81cdc48636c3d25f79f15c7d16ab37b8"
)
"""SHA-256 of that file."""

OPENAI_OPERATIONS: Final = (
    "createResponse",
    "createFile",
    "listFiles",
    "retrieveFile",
    "deleteFile",
    "downloadFile",
)
"""The operations of OpenAI's file the library implements."""

ANTHROPIC_FILES: Final = CONTRACTS / "sources" / "anthropic-files.openapi.json"
"""Anthropic's Files API, transcribed from its API reference."""

HAND_WRITTEN_TYPES = {"options.py"}
"""Modules of ``types/`` the script never replaces."""


def codegen(codegen_dir: Path | None, *args: str, output: Path) -> None:
    """Run aiollm-codegen: the pinned one, or a local checkout."""
    if codegen_dir is None:
        command = ["uv", "run", "--group", "codegen", "aiollm-codegen"]
        cwd = ROOT
    else:
        command = ["uv", "run", "aiollm-codegen"]
        cwd = codegen_dir
    command += [*args, "-o", str(output)]
    subprocess.run(command, cwd=cwd, check=True)  # noqa: S603


def openai_openapi(directory: Path) -> Path:
    """The pinned OpenAPI file of OpenAI: ``OPENAI_OPENAPI``, or a download.

    Raises:
        SystemExit: The file is not the pinned one.
    """
    given = os.environ.get("OPENAI_OPENAPI")
    if given:
        path = Path(given).resolve()
    else:
        path = directory / "openai.openapi.yaml"
        with urllib.request.urlopen(OPENAI_OPENAPI_URL) as response:  # noqa: S310
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OPENAI_OPENAPI_SHA256:
        message = (
            f"{path} has SHA-256 {digest}, not the pinned "
            f"{OPENAI_OPENAPI_SHA256}; to follow a new file, refresh the "
            "contract snapshot from it and change OPENAI_OPENAPI_URL and "
            "OPENAI_OPENAPI_SHA256 in this script"
        )
        raise SystemExit(message)
    return path


def exported(module: Path) -> tuple[list[str], list[str]]:
    """Read the import lines and ``__all__`` of a generated ``__init__``."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    lines = module.read_text(encoding="utf-8").splitlines()
    imports: list[str] = []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imports.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            names = list(ast.literal_eval(node.value))
    return imports, names


def options_names(provider: Path) -> list[str]:
    """Public names of the hand-written ``types/options.py``."""
    tree = ast.parse((provider / "types" / "options.py").read_text("utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]


def install(name: str, generated: Path) -> None:
    """Replace one provider's generated modules."""
    provider = PROVIDERS / name
    types_dir = provider / "types"
    params_dir = provider / "methods" / "params"

    for module in types_dir.glob("*.py"):
        if module.name not in HAND_WRITTEN_TYPES:
            module.unlink()
    shutil.rmtree(params_dir, ignore_errors=True)
    params_dir.mkdir(parents=True)

    for module in (generated / "types").glob("*.py"):
        if module.name != "__init__.py":
            shutil.copy2(module, types_dir / module.name)
    for module in (generated / "methods" / "params").glob("*.py"):
        shutil.copy2(module, params_dir / module.name)

    imports, names = exported(generated / "types" / "__init__.py")
    options = options_names(provider)
    title = "OpenAI" if name == "openai" else "Anthropic"
    body = [
        f'"""{title}\'s own types and the options of a generation call.',
        "",
        "Every module but ``options`` is written by aiollm-codegen through",
        "``scripts/generate_provider_types.py``; do not edit them by hand.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        *imports,
        f"from aiollm.providers.{name}.types.options import (",
        *[f"    {option}," for option in sorted(options)],
        ")",
        "",
        "__all__ = [",
        *[f'    "{item}",' for item in sorted({*names, *options})],
        "]",
        "",
    ]
    (types_dir / "__init__.py").write_text(
        "\n".join(body), encoding="utf-8", newline="\n"
    )


def main() -> None:
    """Generate both providers and install them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--codegen",
        type=Path,
        help="run this checkout of aiollm-codegen, not the pinned one",
    )
    args = parser.parse_args()
    codegen_dir: Path | None = args.codegen and args.codegen.resolve()

    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        openapi = openai_openapi(out)
        codegen(
            codegen_dir,
            str(openapi),
            "--package",
            *(
                argument
                for operation in OPENAI_OPERATIONS
                for argument in ("--operation", operation)
            ),
            output=out / "openai",
        )
        codegen(
            codegen_dir,
            str(CONTRACTS / "anthropic" / "messages"),
            "--package",
            "--contract",
            "/messages",
            "--merge",
            str(ANTHROPIC_FILES),
            "--header",
            f"anthropic-version={ANTHROPIC_VERSION}",
            output=out / "anthropic",
        )
        install("openai", out / "openai")
        install("anthropic", out / "anthropic")

    inits = [
        str(PROVIDERS / name / "types" / "__init__.py")
        for name in ("openai", "anthropic")
    ]
    ruff = ["uv", "run", "ruff"]
    for command in (
        [*ruff, "check", "--select", "I", "--fix", "-q", *inits],
        [*ruff, "format", "-q", "src/aiollm/providers"],
    ):
        subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603
    print("provider types regenerated", file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    main()
