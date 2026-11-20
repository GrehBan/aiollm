"""What the echo provider does besides the shared scenarios."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import TypeAdapter

from aiollm.enums import IncompleteReason
from aiollm.http.exceptions.response import APIResponseError
from aiollm.testing.echo import EchoGenerationAdapter
from aiollm.testing.echo.generation import to_stop
from aiollm.testing.echo.types import EchoEvent
from aiollm.types.requests import GenerationRequest
from aiollm.types.responses import CompletedEvent, OtherEvent

EVENTS: TypeAdapter[EchoEvent] = TypeAdapter(EchoEvent)
REQUEST = GenerationRequest(model="echo-1", input="Hi")
START = {"type": "start", "id": "gen_1", "model": "echo-1"}
DONE = {
    "type": "done",
    "stop": "end",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


async def converted(payloads: list[dict[str, Any]]) -> list[Any]:
    """Run echo events through the adapter."""

    async def items() -> AsyncIterator[EchoEvent]:
        for payload in payloads:
            yield EVENTS.validate_python(payload)

    adapter = EchoGenerationAdapter()
    return [event async for event in adapter.to_events(items(), REQUEST)]


async def test_an_unknown_event_is_another_event() -> None:
    received = await converted([START, {"type": "ping", "n": 1}, DONE])

    assert isinstance(received[1], OtherEvent)
    assert received[1].raw.model_extra == {"n": 1}
    assert isinstance(received[-1], CompletedEvent)
    assert received[-1].text is None
    assert received[-1].native is None


async def test_arguments_of_an_unknown_call_are_refused() -> None:
    delta = {"type": "tool_call_delta", "id": "call_9", "delta": "{}"}

    with pytest.raises(APIResponseError, match="unknown call call_9"):
        await converted([START, delta])


async def test_a_stream_must_start() -> None:
    with pytest.raises(APIResponseError, match="without a start"):
        await converted([DONE])


@pytest.mark.parametrize(
    ("stop", "reason"),
    [
        ("end", None),
        ("tool_calls", None),
        ("max_tokens", IncompleteReason.MAX_TOKENS),
        ("refusal", IncompleteReason.REFUSAL),
        ("content_filter", IncompleteReason.OTHER),
    ],
)
def test_stops(stop: str, reason: IncompleteReason | None) -> None:
    result = to_stop(stop, "no")

    assert result.incomplete_reason is reason
    assert result.refusal == ("no" if stop == "refusal" else None)
