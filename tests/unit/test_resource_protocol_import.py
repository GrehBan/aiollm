"""Import checks for resource protocol runtime definitions."""

from __future__ import annotations

import subprocess
import sys


def test_resource_protocols_import_without_runtime_name_error() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import aiollm.resources.protocols"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
