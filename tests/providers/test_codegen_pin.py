"""The generator's inputs are the files of the contract snapshot."""

from __future__ import annotations

import json
import runpy
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def test_the_pinned_openapi_file_is_the_snapshot_one() -> None:
    # The types and the wire contracts come from one OpenAI file, so a
    # new file changes the pin and the snapshot together.
    script: dict[str, Any] = runpy.run_path(
        str(ROOT / "scripts" / "generate_provider_types.py")
    )
    manifest = json.loads(
        (ROOT / "tests/providers/contracts/manifest.json").read_text("utf-8")
    )
    snapshot = manifest["endpoints"]["openai/responses"]["openapi"]

    assert script["OPENAI_OPENAPI_SHA256"] == snapshot["sha256"]
    assert "/openai/openai-openapi/" in script["OPENAI_OPENAPI_URL"]


def test_the_pinned_sdk_is_the_snapshot_one() -> None:
    # Anthropic's schemas are derived from its SDK, so the snapshot, the
    # pin of the upstream group and the API version the types are made
    # for change together.
    script: dict[str, Any] = runpy.run_path(
        str(ROOT / "scripts" / "generate_provider_types.py")
    )
    manifest = json.loads(
        (ROOT / "tests/providers/contracts/manifest.json").read_text("utf-8")
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    sdk = manifest["endpoints"]["anthropic/messages"]["sdk"]

    pin = f"{sdk['package']}=={sdk['version']}"
    assert pin in project["dependency-groups"]["upstream"]
    assert script["ANTHROPIC_VERSION"] == sdk["api_version"]
