"""Shared fixtures: the ``aiollm.testing`` plugin, set up for this suite.

The doubles are imported from ``aiollm.testing`` and the local server
from ``aiollm.testing.aiohttp``; see ``docs/testing-plan.md`` (stage 0)
for what each stage uses.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["aiollm.testing.plugin", "pytester"]


@pytest.fixture
def provider_key_envs() -> tuple[str, ...]:
    """The keys the library's provider clients read, cleared in every test."""
    return ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
