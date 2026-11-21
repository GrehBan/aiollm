"""Payloads of the echo API that the contract cases of echo use."""

from __future__ import annotations

import json
from typing import Any

from aiollm.testing.contract.generation import (
    CALLS,
    CITATIONS,
    SEARCH_ID,
    SEARCH_QUERIES,
)

RESPONSE: dict[str, Any] = {
    "id": "gen_1",
    "model": "echo-1",
    "output": [{"type": "text", "text": "Hi there."}],
    "stop": "end",
    "usage": {"input_tokens": 12, "output_tokens": 4},
}
"""A finished response."""

STREAM: list[dict[str, Any]] = [
    {"type": "start", "id": "gen_2", "model": "echo-1"},
    {"type": "text_delta", "delta": "Hello"},
    {"type": "text_delta", "delta": "!"},
    {
        "type": "done",
        "stop": "end",
        "usage": {"input_tokens": 12, "output_tokens": 2},
    },
]
"""A finished stream."""

CALL_ITEMS: list[dict[str, Any]] = [
    {
        "type": "tool_call",
        "id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }
    for call_id, name, arguments in CALLS
]
"""The output items of ``CALLS``."""

CALLED: dict[str, Any] = RESPONSE | {
    "output": [{"type": "text", "text": "Checking."}, *CALL_ITEMS],
    "stop": "tool_calls",
}
"""A response with text and ``CALLS``."""


def called_stream() -> list[dict[str, Any]]:
    """Build a stream of ``CALLS``, the arguments in two pieces.

    Returns:
        The events.
    """
    events: list[dict[str, Any]] = [STREAM[0]]
    for call_id, name, arguments in CALLS:
        text = json.dumps(arguments)
        events += [
            {"type": "tool_call_start", "id": call_id, "name": name},
            {"type": "tool_call_delta", "id": call_id, "delta": text[:3]},
            {"type": "tool_call_delta", "id": call_id, "delta": text[3:]},
            {"type": "tool_call_done", "id": call_id},
        ]
    events.append(STREAM[-1] | {"stop": "tool_calls"})
    return events


RESULTS: list[dict[str, Any]] = [
    {"role": "tool", "call_id": call_id, "output": f"result of {name}"}
    for call_id, name, _ in CALLS
]
"""The results of ``CALLS``, as the next request sends them."""


SEARCH_ITEM: dict[str, Any] = {
    "type": "server_tool",
    "id": SEARCH_ID,
    "kind": "web_search",
    "status": "completed",
    "queries": SEARCH_QUERIES,
    "result": {"pages": ["https://example.com/paris"]},
}
"""A finished web search."""

CITATION_ITEM: dict[str, Any] = {
    "type": "citation",
    "kind": "url",
    "url": CITATIONS[0][1],
    "title": CITATIONS[0][2],
}
"""The source of the text after the search."""

SEARCHED: dict[str, Any] = RESPONSE | {
    "output": [
        SEARCH_ITEM,
        {"type": "text", "text": "It is sunny."},
        CITATION_ITEM,
    ],
}
"""A response with a search, its text and the source."""

SEARCHED_STREAM: list[dict[str, Any]] = [
    STREAM[0],
    {
        "type": "server_tool_start",
        "call": SEARCH_ITEM | {"status": "in_progress", "result": None},
    },
    {"type": "server_tool_done", "call": SEARCH_ITEM},
    {"type": "text_delta", "delta": "It is "},
    {"type": "text_delta", "delta": "sunny."},
    {"type": "citation", "citation": CITATION_ITEM},
    STREAM[-1],
]
"""A stream of the same."""

FAILED: dict[str, Any] = RESPONSE | {
    "output": [
        SEARCH_ITEM
        | {"status": "failed", "error": "unavailable", "result": None},
        {"type": "text", "text": "I could not search."},
    ],
}
"""A response whose search failed."""

PAUSED_ITEMS: list[dict[str, Any]] = [
    SEARCH_ITEM | {"status": "in_progress", "result": None}
]
"""The output of a response that paused in the middle of a search."""

PAUSED: dict[str, Any] = RESPONSE | {"output": PAUSED_ITEMS, "stop": "paused"}
"""A response that paused."""
