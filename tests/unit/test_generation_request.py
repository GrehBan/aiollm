"""The shared request refuses tool settings that no provider takes."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aiollm.enums import ToolChoice
from aiollm.testing.contract.generation import TOOLS
from aiollm.types.requests import ForcedTool, GenerationRequest, Tool


@pytest.mark.parametrize(
    "changes",
    [
        {"tool_choice": ToolChoice.AUTO},
        {"parallel_tool_calls": False},
        {"tools": [TOOLS[0], TOOLS[0]]},
        {"tools": TOOLS, "tool_choice": ForcedTool(name="other")},
        {
            "tools": TOOLS,
            "tool_choice": ToolChoice.NONE,
            "parallel_tool_calls": True,
        },
        {"tools": []},
    ],
    ids=[
        "choice-without-tools",
        "parallel-without-tools",
        "duplicate-names",
        "forced-unknown-tool",
        "parallel-with-none",
        "no-tools",
    ],
)
def test_tool_settings_no_provider_takes_are_refused(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(model="model-1", input="Hi", **changes)


@pytest.mark.parametrize(
    "name", ["", "has space", "a" * 129, "dots.not.allowed"]
)
def test_tool_names_both_providers_take(name: str) -> None:
    with pytest.raises(ValidationError, match="name"):
        Tool(name=name, parameters={"type": "object"})
    Tool(name="a" * 128, parameters={"type": "object"})


def test_a_single_tool_becomes_a_list() -> None:
    tool = Tool(name="lookup", parameters={"type": "object"})

    request = GenerationRequest(model="model-1", input="Hi", tools=tool)

    assert request.tools == [tool]
