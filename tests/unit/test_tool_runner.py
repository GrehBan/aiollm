"""Tests for running callable tools on buffered and streamed generations."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, cast

import pytest
from pydantic import MISSING, ValidationError

from aiollm.enums import IncompleteReason, Provider
from aiollm.resources.generation import GenerationResource
from aiollm.resources.tool_runner import Callback, ToolRunner
from aiollm.types.requests.generation import (
    GenerationRequest,
    Tool,
    ToolCall,
    ToolResult,
    tool,
)
from aiollm.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    TextDeltaEvent,
)
from aiollm.types.responses.generation import GenerationResponse


class EventStream:
    """An async iterator over a fixed sequence of generation events."""

    def __init__(
        self,
        events: Sequence[GenerationEvent[Any]],
        error: BaseException | None = None,
    ) -> None:
        self._events = iter(events)
        self._error = error

    def __aiter__(self) -> EventStream:
        return self

    async def __anext__(self) -> GenerationEvent[Any]:
        try:
            return next(self._events)
        except StopIteration:
            if self._error is not None:
                error = self._error
                self._error = None
                raise error from None
            raise StopAsyncIteration from None


class FakeGenerationResource:
    """Records calls and yields a scripted event stream."""

    def __init__(
        self,
        response: GenerationResponse[Any],
        events: Sequence[GenerationEvent[Any]] = (),
        *,
        stream_error: BaseException | None = None,
        stream_exit_error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.events = events
        self.stream_error = stream_error
        self.stream_exit_error = stream_exit_error
        self.create_calls: list[tuple[GenerationRequest, dict[str, Any]]] = []
        self.stream_calls: list[tuple[GenerationRequest, dict[str, Any]]] = []
        self.stream_exits = 0

    async def create(
        self,
        request: GenerationRequest,
        **options: Any,
    ) -> GenerationResponse[Any]:
        self.create_calls.append((request, options))
        return self.response

    def stream(
        self,
        request: GenerationRequest,
        **options: Any,
    ) -> AbstractAsyncContextManager[EventStream]:
        self.stream_calls.append((request, options))

        @asynccontextmanager
        async def open_stream() -> AsyncGenerator[EventStream]:
            try:
                yield EventStream(self.events, self.stream_error)
            finally:
                self.stream_exits += 1
                if self.stream_exit_error is not None:
                    raise self.stream_exit_error

        return open_stream()


def make_runner(
    resource: FakeGenerationResource,
) -> ToolRunner[Any, Any, Any, Any]:
    return ToolRunner(cast(GenerationResource[Any, Any, Any, Any], resource))


def response_with_tools(
    calls: list[ToolCall],
    *,
    incomplete_reason: IncompleteReason | None = None,
) -> GenerationResponse[Any]:
    return GenerationResponse(
        id="response-1",
        provider=Provider.GENERIC,
        model="model-1",
        incomplete_reason=incomplete_reason,
        tool_calls=calls,
        raw={"id": "response-1"},
    )


async def test_create_runs_sync_and_async_callbacks_and_keeps_response() -> (
    None
):
    executed: list[str] = []

    @tool
    def add(left: int, right: int) -> dict[str, int]:
        executed.append("add")
        return {"sum": left + right}

    @tool
    async def greet(name: str) -> str:
        executed.append("greet")
        return f"Hello, {name}"

    tool_calls = [
        ToolCall(id="call-1", name="add", arguments='{"left": 2, "right": 3}'),
        ToolCall(id="call-2", name="greet", arguments='{"name": "Ada"}'),
    ]
    response = response_with_tools(tool_calls)
    resource = FakeGenerationResource(response)
    runner = make_runner(resource)
    request = GenerationRequest(
        model="model-1", input="Hi", tools=[add.tool, greet.tool]
    )
    headers = {"X-Trace": "trace-1"}
    provider_options = {"store": False}

    iterator = await runner.create(
        request,
        headers=headers,
        timeout=5.0,
        provider_options=provider_options,
    )
    results = [item async for item in iterator]

    assert results == [
        ToolResult(call_id="call-1", output='{"sum": 5}'),
        ToolResult(call_id="call-2", output="Hello, Ada"),
    ]
    assert iterator.original is response
    assert executed == ["add", "greet"]
    assert resource.create_calls[0] == (
        request,
        {
            "headers": headers,
            "timeout": 5.0,
            "retry_policy": None,
            "provider_options": provider_options,
        },
    )


@pytest.mark.parametrize(
    ("call_name", "arguments", "error"),
    [
        ("not_registered", "{}", ValueError),
        ("lookup", '{"city": 42}', ValidationError),
    ],
    ids=["unknown-tool", "invalid-arguments"],
)
async def test_create_resolves_all_callbacks_before_executing(
    call_name: str,
    arguments: str,
    error: type[Exception],
) -> None:
    executed: list[str] = []

    @tool
    def lookup(city: str) -> str:
        executed.append(city)
        return city

    response = response_with_tools(
        [ToolCall(id="call-1", name=call_name, arguments=arguments)]
    )
    resource = FakeGenerationResource(response)
    request = GenerationRequest(
        model="model-1", input="Hi", tools=[lookup.tool]
    )

    with pytest.raises(error):
        await make_runner(resource).create(request)

    assert executed == []
    assert len(resource.create_calls) == 1


async def test_create_skips_tool_calls_on_an_incomplete_response() -> None:
    executed: list[str] = []

    @tool
    def lookup(city: str) -> str:
        executed.append(city)
        return city

    response = response_with_tools(
        [ToolCall(id="call-1", name="lookup", arguments='{"city": "Paris"}')],
        incomplete_reason=IncompleteReason.REFUSAL,
    )
    request = GenerationRequest(
        model="model-1", input="Hi", tools=[lookup.tool]
    )

    iterator = await make_runner(FakeGenerationResource(response)).create(
        request
    )
    results = [item async for item in iterator]

    assert results == []
    assert executed == []


async def test_stream_forwards_events_then_runs_completed_tool_calls() -> None:
    executed: list[str] = []

    @tool(description="Echo a value", strict=True)
    def echo(value: str) -> str:
        executed.append(value)
        return value

    call = ToolCall(id="call-1", name="echo", arguments='{"value": "done"}')
    events: list[GenerationEvent[Any]] = [
        TextDeltaEvent(text="Working", raw={}),
        CompletedEvent(
            provider=Provider.GENERIC,
            model="model-1",
            tool_calls=[call],
            raw={},
        ),
    ]
    response = response_with_tools([call])
    resource = FakeGenerationResource(response, events)
    runner = make_runner(resource)
    request = GenerationRequest(model="model-1", input="Hi", tools=[echo.tool])

    async with runner.stream(
        request,
        headers={"X-Trace": "trace-2"},
        timeout=2.0,
    ) as stream:
        received = [item async for item in stream]

    assert [item.type for item in received] == [
        "text_delta",
        "completed",
        "tool_result",
    ]
    assert received[-1] == ToolResult(call_id="call-1", output="done")
    assert executed == ["done"]
    assert resource.stream_calls[0][0] is request
    assert resource.stream_calls[0][1] == {
        "headers": {"X-Trace": "trace-2"},
        "timeout": 2.0,
        "retry_policy": None,
        "provider_options": MISSING,
    }
    assert resource.stream_exits == 1


async def test_stream_does_not_run_tools_for_incomplete_completion() -> None:
    executed: list[str] = []

    @tool
    def lookup(city: str) -> str:
        executed.append(city)
        return city

    call = ToolCall(id="call-1", name="lookup", arguments='{"city": "Paris"}')
    events: list[GenerationEvent[Any]] = [
        CompletedEvent(
            provider=Provider.GENERIC,
            model="model-1",
            incomplete_reason=IncompleteReason.REFUSAL,
            tool_calls=[call],
            raw={},
        ),
    ]
    resource = FakeGenerationResource(response_with_tools([call]), events)
    request = GenerationRequest(
        model="model-1", input="Hi", tools=[lookup.tool]
    )

    async with make_runner(resource).stream(request) as stream:
        received = [item async for item in stream]

    assert len(received) == 1
    assert isinstance(received[0], CompletedEvent)
    assert executed == []
    assert resource.stream_exits == 1


async def test_stream_propagates_body_errors_and_closes_source() -> None:
    @tool
    def lookup(city: str) -> str:
        return city

    response = response_with_tools([])
    resource = FakeGenerationResource(
        response,
        [TextDeltaEvent(text="Working", raw={})],
    )
    request = GenerationRequest(
        model="model-1", input="Hi", tools=[lookup.tool]
    )
    error = RuntimeError("consumer failed")

    with pytest.raises(RuntimeError) as caught:
        async with make_runner(resource).stream(request) as stream:
            await anext(stream)
            raise error

    assert caught.value is error
    assert resource.stream_exits == 1


async def test_tool_runner_requires_at_least_one_tool_before_sending() -> None:
    resource = FakeGenerationResource(response_with_tools([]))
    request = GenerationRequest(model="model-1", input="Hi")

    with pytest.raises(ValueError, match="at least one tool"):
        await make_runner(resource).create(request)

    assert resource.create_calls == []


def test_tool_from_model_cannot_be_executed_without_a_callback() -> None:
    call = ToolCall(id="call-1", name="lookup", arguments='{"city": "Paris"}')
    definition = Tool(
        name="lookup",
        parameters={"type": "object", "properties": {}},
    )

    with pytest.raises(TypeError, match="no associated callback"):
        Callback.bind(definition, call)


def test_tool_decorator_rejects_definition_without_a_callback() -> None:
    definition = Tool(
        name="lookup",
        parameters={"type": "object", "properties": {}},
    )

    with pytest.raises(TypeError, match="no associated callable"):
        tool(definition)


def test_tool_decorator_wraps_an_existing_callable_tool() -> None:
    @tool
    def lookup(city: str) -> str:
        return city

    wrapped = tool(lookup.tool)

    assert wrapped.tool is lookup.tool
    assert wrapped("Paris") == "Paris"


def test_tool_call_rejects_a_tool_without_an_argument_validator() -> None:
    definition = Tool(
        name="lookup",
        parameters={"type": "object", "properties": {}},
    )
    call = ToolCall(id="call-1", name="lookup", arguments="{}")

    with pytest.raises(TypeError, match="no associated argument validator"):
        call.parse_arguments(definition)


def test_tool_call_rejects_an_unsupported_argument_model() -> None:
    call = ToolCall(id="call-1", name="lookup", arguments="{}")

    with pytest.raises(TypeError, match="Expected a Pydantic model"):
        call.parse_arguments(cast(Any, object()))


@pytest.mark.parametrize(
    "tools",
    [object(), [object()]],
    ids=["unsupported-value", "unsupported-sequence-item"],
)
def test_generation_request_rejects_unsupported_tools(tools: Any) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(model="model-1", input="Hi", tools=tools)


def test_generation_request_preserves_missing_tools() -> None:
    request = GenerationRequest(model="model-1", input="Hi", tools=MISSING)

    assert request.tools is MISSING


def test_callback_bind_rejects_a_mismatched_tool_name() -> None:
    @tool
    def lookup(city: str) -> str:
        return city

    call = ToolCall(id="call-1", name="other", arguments='{"city": "Paris"}')

    with pytest.raises(ValueError, match="Tool name mismatch"):
        Callback.bind(lookup.tool, call)


async def test_stream_propagates_source_errors_and_closes_the_stream() -> None:
    error = RuntimeError("source failed")

    @tool
    def lookup() -> None:
        return None

    resource = FakeGenerationResource(
        response_with_tools([]),
        stream_error=error,
    )
    request = GenerationRequest(
        model="model-1", input="Hi", tools=[lookup.tool]
    )
    runner = make_runner(resource)

    async with runner.stream(request) as stream:
        with pytest.raises(RuntimeError) as caught:
            await anext(stream)
        assert caught.value is error
        await stream.aclose()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    assert resource.stream_exits == 1


async def test_stream_propagates_stream_exit_errors() -> None:
    error = RuntimeError("stream close failed")
    event = TextDeltaEvent(text="Working", raw={})

    @tool
    def lookup() -> None:
        return None

    resource = FakeGenerationResource(
        response_with_tools([]),
        [event],
        stream_exit_error=error,
    )
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[lookup.tool],
    )

    async with make_runner(resource).stream(request) as stream:
        assert await anext(stream) is event
        with pytest.raises(RuntimeError) as caught:
            await anext(stream)

    assert caught.value is error
    assert resource.stream_exits == 1


async def test_stream_propagates_tool_callback_errors() -> None:
    error = RuntimeError("callback failed")

    @tool
    def lookup(city: str) -> str:
        raise error

    call = ToolCall(id="call-1", name="lookup", arguments='{"city": "Paris"}')
    events: list[GenerationEvent[Any]] = [
        CompletedEvent(
            provider=Provider.GENERIC,
            model="model-1",
            tool_calls=[call],
            raw={},
        ),
    ]
    resource = FakeGenerationResource(response_with_tools([call]), events)
    request = GenerationRequest(
        model="model-1", input="Hi", tools=[lookup.tool]
    )

    async with make_runner(resource).stream(request) as stream:
        assert isinstance(await anext(stream), CompletedEvent)
        with pytest.raises(RuntimeError) as caught:
            await anext(stream)

    assert caught.value is error
    assert resource.stream_exits == 1
