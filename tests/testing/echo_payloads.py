"""Payloads of the echo API that the contract cases of echo use."""

from __future__ import annotations

import json
from typing import Any

from aiollm.testing.contract.generation import CALLS

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
