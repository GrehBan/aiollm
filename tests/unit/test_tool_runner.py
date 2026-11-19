"""Tests for running callable tools on buffered and streamed generations."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import threading
from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, cast

import pytest
from aiopulse import CallbackCancelledError, CallbackExceptionGroup
from pydantic import MISSING, ValidationError

from aiollm.enums import IncompleteReason, Provider
from aiollm.resources.generation import GenerationResource
from aiollm.resources.tool_runner import (
    Callback,
    ToolRunner,
    ToolRunnerIterator,
)
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
        with pytest.raises(CallbackExceptionGroup) as caught:
            await anext(stream)

    assert caught.value.exceptions == (error,)
    assert [failure.index for failure in caught.value.failures] == [0]
    assert resource.stream_exits == 1


async def test_stream_callback_failure_cancels_running_callbacks() -> None:
    waiting_started = asyncio.Event()
    failing_started = asyncio.Event()
    waiting_cancelled = asyncio.Event()
    error = RuntimeError("callback failed")

    @tool
    async def wait_forever() -> str:
        waiting_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            waiting_cancelled.set()
        return "unreachable"

    @tool
    async def fail() -> str:
        failing_started.set()
        raise error

    calls = [
        ToolCall(id="call-1", name="wait_forever", arguments="{}"),
        ToolCall(id="call-2", name="fail", arguments="{}"),
    ]
    completed = CompletedEvent(
        provider=Provider.GENERIC,
        model="model-1",
        tool_calls=calls,
        raw={},
    )
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[wait_forever.tool, fail.tool],
    )
    resource = FakeGenerationResource(response_with_tools(calls), [completed])
    runner = make_runner(resource)

    async with runner.stream(request, concurrency=2) as stream:
        assert await anext(stream) is completed
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(failing_started.wait(), timeout=1)
        with pytest.raises(CallbackExceptionGroup) as caught:
            await pending

    assert caught.value.exceptions == (error,)
    assert waiting_started.is_set()
    assert waiting_cancelled.is_set()
    assert resource.stream_exits == 1


@pytest.mark.parametrize(
    ("concurrency", "buffer_size", "message"),
    [
        (0, 0, "concurrency must be greater than 0"),
        (1, -1, "buffer_size must not be negative"),
    ],
)
async def test_invalid_limits_fail_before_request(
    concurrency: int,
    buffer_size: int,
    message: str,
) -> None:
    @tool
    def lookup() -> str:
        return "value"

    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[lookup.tool],
    )
    resource = FakeGenerationResource(response_with_tools([]))
    runner = make_runner(resource)

    with pytest.raises(ValueError, match=message):
        await runner.create(
            request,
            concurrency=concurrency,
            buffer_size=buffer_size,
        )

    with pytest.raises(ValueError, match=message):
        async with runner.stream(
            request,
            concurrency=concurrency,
            buffer_size=buffer_size,
        ):
            pytest.fail("invalid limits opened a stream")

    assert resource.create_calls == []
    assert resource.stream_calls == []


def iterate(
    calls: dict[str, Callback],
    concurrency: int = 1,
    *,
    buffer_size: int = 0,
    in_order: bool = True,
) -> ToolRunnerIterator[Any]:
    return ToolRunnerIterator(
        calls,
        response_with_tools([]),
        concurrency,
        buffer_size=buffer_size,
        in_order=in_order,
    )


async def test_concurrency_sets_the_scheduling_window() -> None:
    active = 0
    peak_active = 0
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def callback() -> str:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        if active == 2:
            two_started.set()
        try:
            await release.wait()
        finally:
            active -= 1
        return "done"

    calls = {str(index): Callback(callback, (), {}) for index in range(3)}
    runner = iterate(calls, concurrency=2)
    first = asyncio.create_task(anext(runner))

    try:
        await asyncio.wait_for(two_started.wait(), timeout=0.1)
    finally:
        release.set()

    results = [await first, await anext(runner), await anext(runner)]

    assert peak_active == 2
    assert [result.call_id for result in results] == ["0", "1", "2"]


async def test_callbacks_run_one_at_a_time_by_default() -> None:
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> str:
        first_started.set()
        await release_first.wait()
        return "first"

    async def second() -> str:
        second_started.set()
        return "second"

    runner = iterate(
        {
            "first": Callback(first, (), {}),
            "second": Callback(second, (), {}),
        }
    )
    first_result = asyncio.create_task(anext(runner))
    await asyncio.wait_for(first_started.wait(), timeout=1)

    assert not second_started.is_set()

    release_first.set()
    assert (await first_result).call_id == "first"
    assert (await anext(runner)).call_id == "second"
    assert second_started.is_set()


async def test_results_come_from_async_iteration_and_empty_calls() -> None:
    empty_results = [result async for result in iterate({})]

    async def lookup() -> str:
        return "value"

    results = [
        result
        async for result in iterate({"call-1": Callback(lookup, (), {})})
    ]

    assert empty_results == []
    assert results == [ToolResult(call_id="call-1", output="value")]


async def test_a_closed_iterator_cannot_be_entered() -> None:
    runner = iterate({})
    await runner.aclose()
    await runner.aclose()

    with pytest.raises(RuntimeError, match="CallbackExecutor is closed"):
        async with runner:
            pytest.fail("a closed iterator was entered")

    with pytest.raises(StopAsyncIteration):
        await anext(runner)


async def test_leaving_the_block_early_cancels_unfinished_callbacks() -> None:
    cancelled: list[str] = []

    async def quick() -> str:
        return "quick"

    async def slow() -> str:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append("slow")
        return "unreachable"

    calls = {
        "call-1": Callback(quick, (), {}),
        "call-2": Callback(slow, (), {}),
    }

    async with iterate(calls, concurrency=2) as runner:
        async for result in runner:
            assert result.call_id == "call-1"
            break

    assert cancelled == ["slow"]
    with pytest.raises(StopAsyncIteration):
        await anext(runner)


async def test_every_failure_observed_is_in_the_group() -> None:
    error = RuntimeError("callback failed")
    cleanup_error = OSError("cleanup failed")
    waiting = asyncio.Event()

    async def fail() -> str:
        await waiting.wait()
        raise error

    async def fail_on_cancel() -> str:
        waiting.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise cleanup_error from None
        return "unreachable"

    calls = {
        "call-1": Callback(fail, (), {}),
        "call-2": Callback(fail_on_cancel, (), {}),
    }

    with pytest.raises(CallbackExceptionGroup) as caught:
        async for _ in iterate(calls, concurrency=2):
            pytest.fail("a failed callback produced a result")

    assert caught.value.exceptions == (error, cleanup_error)
    assert [failure.index for failure in caught.value.failures] == [0, 1]


async def test_a_failure_lets_cancelled_callbacks_finish_cleanup() -> None:
    error = RuntimeError("callback failed")
    waiting = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup: list[str] = []

    async def fail() -> str:
        await waiting.wait()
        raise error

    async def clean_up_slowly() -> str:
        waiting.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            try:
                await release_cleanup.wait()
            except asyncio.CancelledError:
                cleanup.append("interrupted")
                raise
            cleanup.append("finished")
        return "unreachable"

    calls = {
        "call-1": Callback(clean_up_slowly, (), {}),
        "call-2": Callback(fail, (), {}),
    }
    runner = iterate(calls, concurrency=2)
    consumer = asyncio.create_task(anext(runner))
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    for _ in range(5):
        await asyncio.sleep(0)

    assert not consumer.done()
    release_cleanup.set()
    with pytest.raises(CallbackExceptionGroup) as caught:
        await consumer

    assert caught.value.exceptions == (error,)
    assert cleanup == ["finished"]


async def test_create_runs_callbacks_concurrently_and_keeps_call_ids() -> None:
    first_started = asyncio.Event()
    second_finished = asyncio.Event()
    release_first = asyncio.Event()
    second_task: asyncio.Task[Any] | None = None

    @tool
    async def first() -> str:
        first_started.set()
        await release_first.wait()
        return "first-result"

    @tool
    async def second() -> str:
        nonlocal second_task
        second_task = asyncio.current_task()
        second_finished.set()
        return "second-result"

    calls = [
        ToolCall(id="call-1", name="first", arguments="{}"),
        ToolCall(id="call-2", name="second", arguments="{}"),
    ]
    response = response_with_tools(calls)
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[first.tool, second.tool],
    )
    iterator = await make_runner(FakeGenerationResource(response)).create(
        request,
        concurrency=2,
    )
    first_result = asyncio.create_task(anext(iterator))

    try:
        await asyncio.wait_for(second_finished.wait(), timeout=1)
        assert second_task is not None
        await second_task
        await asyncio.sleep(0)
        assert first_started.is_set()
        assert not first_result.done()
    finally:
        release_first.set()

    results = [await first_result, await anext(iterator)]

    assert results == [
        ToolResult(call_id="call-1", output="first-result"),
        ToolResult(call_id="call-2", output="second-result"),
    ]


async def test_stream_runs_callbacks_concurrently_and_keeps_call_ids() -> None:
    second_finished = asyncio.Event()
    release_first = asyncio.Event()
    second_task: asyncio.Task[Any] | None = None

    @tool
    async def first() -> str:
        await release_first.wait()
        return "first-result"

    @tool
    async def second() -> str:
        nonlocal second_task
        second_task = asyncio.current_task()
        second_finished.set()
        return "second-result"

    calls = [
        ToolCall(id="call-1", name="first", arguments="{}"),
        ToolCall(id="call-2", name="second", arguments="{}"),
    ]
    completed = CompletedEvent(
        provider=Provider.GENERIC,
        model="model-1",
        tool_calls=calls,
        raw={},
    )
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[first.tool, second.tool],
    )
    resource = FakeGenerationResource(response_with_tools(calls), [completed])
    runner = make_runner(resource)

    async with runner.stream(request, concurrency=2) as stream:
        assert await anext(stream) is completed
        first_result = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(second_finished.wait(), timeout=1)
            assert second_task is not None
            await second_task
            await asyncio.sleep(0)
            assert not first_result.done()
        finally:
            release_first.set()

        results = [await first_result, await anext(stream)]

    assert results == [
        ToolResult(call_id="call-1", output="first-result"),
        ToolResult(call_id="call-2", output="second-result"),
    ]


async def test_create_callback_failure_cancels_running_callbacks() -> None:
    waiting_started = asyncio.Event()
    failing_started = asyncio.Event()
    waiting_cancelled = asyncio.Event()
    error = RuntimeError("callback failed")

    @tool
    async def wait_forever() -> str:
        waiting_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            waiting_cancelled.set()
        return "unreachable"

    @tool
    async def fail() -> str:
        failing_started.set()
        raise error

    calls = [
        ToolCall(id="call-1", name="wait_forever", arguments="{}"),
        ToolCall(id="call-2", name="fail", arguments="{}"),
    ]
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[wait_forever.tool, fail.tool],
    )
    iterator = await make_runner(
        FakeGenerationResource(response_with_tools(calls))
    ).create(request, concurrency=2)
    pending = asyncio.create_task(anext(iterator))

    try:
        await asyncio.wait_for(failing_started.wait(), timeout=1)
        with pytest.raises(CallbackExceptionGroup) as caught:
            await pending
    finally:
        await iterator.aclose()

    assert caught.value.exceptions == (error,)
    assert [failure.index for failure in caught.value.failures] == [1]
    assert waiting_started.is_set()
    assert waiting_cancelled.is_set()


async def test_a_callback_cancelling_itself_is_a_failure() -> None:
    waiting_started = asyncio.Event()
    waiting_cancelled = asyncio.Event()
    cancellation_started = asyncio.Event()

    @tool
    async def wait_forever() -> str:
        waiting_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            waiting_cancelled.set()
        return "unreachable"

    @tool
    async def cancel() -> str:
        cancellation_started.set()
        raise asyncio.CancelledError

    calls = [
        ToolCall(id="call-1", name="wait_forever", arguments="{}"),
        ToolCall(id="call-2", name="cancel", arguments="{}"),
    ]
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[wait_forever.tool, cancel.tool],
    )
    iterator = await make_runner(
        FakeGenerationResource(response_with_tools(calls))
    ).create(request, concurrency=2)
    pending = asyncio.create_task(anext(iterator))

    try:
        await asyncio.wait_for(cancellation_started.wait(), timeout=1)
        with pytest.raises(CallbackExceptionGroup) as caught:
            await pending
    finally:
        await iterator.aclose()

    [failure] = caught.value.exceptions
    assert isinstance(failure, CallbackCancelledError)
    assert isinstance(failure.__cause__, asyncio.CancelledError)
    assert not pending.cancelled()
    assert waiting_started.is_set()
    assert waiting_cancelled.is_set()


async def test_create_aclose_waits_for_callback_cancellation() -> None:
    two_started = asyncio.Event()
    callback_tasks: list[asyncio.Task[Any]] = []
    active = 0

    @tool
    async def wait_forever(index: int) -> str:
        nonlocal active
        task = asyncio.current_task()
        assert task is not None
        callback_tasks.append(task)
        active += 1
        if active == 2:
            two_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
        return str(index)

    calls = [
        ToolCall(
            id=f"call-{index}",
            name="wait_forever",
            arguments=f'{{"index": {index}}}',
        )
        for index in range(3)
    ]
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[wait_forever.tool],
    )
    iterator = await make_runner(
        FakeGenerationResource(response_with_tools(calls))
    ).create(request, concurrency=2)
    pending = asyncio.create_task(anext(iterator))
    await asyncio.wait_for(two_started.wait(), timeout=1)

    await iterator.aclose()
    with pytest.raises(StopAsyncIteration):
        await pending

    assert active == 0
    assert len(callback_tasks) == 2
    assert all(task is not None and task.done() for task in callback_tasks)


async def test_buffered_close_before_iteration_prevents_callbacks() -> None:
    executed: list[str] = []

    @tool
    def lookup() -> str:
        executed.append("lookup")
        return "value"

    call = ToolCall(id="call-1", name="lookup", arguments="{}")
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[lookup.tool],
    )
    iterator = await make_runner(
        FakeGenerationResource(response_with_tools([call]))
    ).create(request)

    await iterator.aclose()
    results = [result async for result in iterator]

    assert results == []
    assert executed == []


async def test_stream_context_exit_waits_for_callback_cancellation() -> None:
    two_started = asyncio.Event()
    callback_tasks: list[asyncio.Task[Any]] = []
    active = 0

    @tool
    async def wait_forever(index: int) -> str:
        nonlocal active
        task = asyncio.current_task()
        assert task is not None
        callback_tasks.append(task)
        active += 1
        if active == 2:
            two_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
        return str(index)

    calls = [
        ToolCall(
            id=f"call-{index}",
            name="wait_forever",
            arguments=f'{{"index": {index}}}',
        )
        for index in range(3)
    ]
    completed = CompletedEvent(
        provider=Provider.GENERIC,
        model="model-1",
        tool_calls=calls,
        raw={},
    )
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[wait_forever.tool],
    )
    resource = FakeGenerationResource(response_with_tools(calls), [completed])
    runner = make_runner(resource)

    async with runner.stream(request, concurrency=2) as stream:
        assert await anext(stream) is completed
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(two_started.wait(), timeout=1)

    with pytest.raises(StopAsyncIteration):
        await pending

    assert active == 0
    assert len(callback_tasks) == 2
    assert all(task is not None and task.done() for task in callback_tasks)
    assert resource.stream_exits == 1


async def test_cancelling_the_consumer_cancels_running_callbacks() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_forever() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return "unreachable"

    runner = iterate({"call-1": Callback(wait_forever, (), {})})
    pending = asyncio.create_task(anext(runner))
    await asyncio.wait_for(started.wait(), timeout=1)

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert cancelled.is_set()
    with pytest.raises(StopAsyncIteration):
        await anext(runner)


async def test_sync_callbacks_run_in_worker_threads() -> None:
    loop_thread = threading.get_ident()
    release = threading.Event()
    threads: list[int] = []

    def wait_for_release() -> str:
        threads.append(threading.get_ident())
        assert release.wait(timeout=1)
        return "released"

    async def set_release() -> str:
        await asyncio.sleep(0)
        release.set()
        return "set"

    calls = {
        "call-1": Callback(wait_for_release, (), {}),
        "call-2": Callback(set_release, (), {}),
    }
    results = [result async for result in iterate(calls, concurrency=2)]

    assert [result.output for result in results] == ["released", "set"]
    assert threads and threads[0] != loop_thread


async def test_sync_callbacks_overlap_up_to_the_concurrency() -> None:
    barrier = threading.Barrier(2, timeout=1)

    def meet(name: str) -> str:
        barrier.wait()
        return name

    calls = {
        name: Callback(meet, (name,), {}) for name in ("call-1", "call-2")
    }
    results = [result async for result in iterate(calls, concurrency=2)]

    assert [result.output for result in results] == ["call-1", "call-2"]


class AsyncCallable:
    """An object whose ``__call__`` is a coroutine function."""

    def __init__(self) -> None:
        self.thread: int | None = None

    async def __call__(self) -> str:
        self.thread = threading.get_ident()
        return "called"


async def test_async_callables_run_in_the_event_loop() -> None:
    callable_object = AsyncCallable()

    async def coroutine() -> str:
        return "awaited"

    def returns_coroutine() -> Coroutine[Any, Any, str]:
        return coroutine()

    calls = {
        "call-1": Callback(callable_object, (), {}),
        "call-2": Callback(returns_coroutine, (), {}),
    }
    results = [result async for result in iterate(calls)]

    assert callable_object.thread == threading.get_ident()
    assert [result.output for result in results] == ["called", "awaited"]


async def abandon(
    late: Callable[[], object],
    settled: Callable[[list[dict[str, Any]]], bool] = bool,
) -> list[dict[str, Any]]:
    """Cancel the consumer of a sync callback, then let it end with ``late``.

    Waits until ``settled`` holds for what the loop's exception handler
    received, and returns that.
    """
    loop = asyncio.get_running_loop()
    reported: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _, context: reported.append(context))
    started = asyncio.Event()
    release = threading.Event()

    def end_late() -> object:
        loop.call_soon_threadsafe(started.set)
        release.wait()
        return late()

    runner = iterate({"call-1": Callback(end_late, (), {})})
    consumer = asyncio.create_task(anext(runner))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert reported == []
    finally:
        release.set()
        await runner.aclose()

    try:
        for _ in range(500):
            if settled(reported):
                break
            await asyncio.sleep(0.01)
    finally:
        loop.set_exception_handler(None)
    return reported


async def test_a_cancelled_sync_callback_reports_its_late_failure() -> None:
    late_error = OSError("late failure")

    def fail() -> object:
        raise late_error

    reported = await abandon(fail)

    assert [context["exception"] for context in reported] == [late_error]


async def test_a_late_future_cancellation_is_reported_as_it_was() -> None:
    late_error = concurrent.futures.CancelledError()

    def fail() -> object:
        raise late_error

    reported = await abandon(fail)

    assert [context["exception"] for context in reported] == [late_error]


async def test_a_late_coroutine_result_is_closed_unawaited() -> None:
    async def never_awaited() -> str:
        return "unreachable"

    coroutines: list[Coroutine[Any, Any, str]] = []

    def make() -> object:
        coroutine = never_awaited()
        coroutines.append(coroutine)
        return coroutine

    def closed(_: list[dict[str, Any]]) -> bool:
        return bool(coroutines) and (
            inspect.getcoroutinestate(coroutines[0]) == inspect.CORO_CLOSED
        )

    reported = await abandon(make, closed)

    assert reported == []
    assert inspect.getcoroutinestate(coroutines[0]) == inspect.CORO_CLOSED


async def test_a_cancelled_future_read_by_a_callback_is_its_failure() -> None:
    def read_cancelled() -> str:
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        future.cancel()
        return future.result()

    runner = ToolRunnerIterator(
        {"call-1": Callback(read_cancelled, (), {})},
        response_with_tools([]),
        on_error=lambda error: type(error).__module__,
    )
    results = [result.output async for result in runner]

    assert results == ["concurrent.futures._base"]


@pytest.mark.parametrize("buffer_size", [0, 1])
async def test_buffer_size_frees_the_slot_of_an_unread_result(
    buffer_size: int,
) -> None:
    release_first = asyncio.Event()
    third_started = asyncio.Event()

    async def first() -> str:
        await release_first.wait()
        return "first"

    async def second() -> str:
        return "second"

    async def third() -> str:
        third_started.set()
        return "third"

    runner = iterate(
        {
            "call-1": Callback(first, (), {}),
            "call-2": Callback(second, (), {}),
            "call-3": Callback(third, (), {}),
        },
        concurrency=2,
        buffer_size=buffer_size,
    )
    async with runner:
        waiting = asyncio.create_task(anext(runner))
        for _ in range(5):
            await asyncio.sleep(0)
        assert not waiting.done()
        assert third_started.is_set() is (buffer_size > 0)
        release_first.set()
        results = [(await waiting).output]
        results += [result.output async for result in runner]

    assert results == ["first", "second", "third"]


def slow_and_quick_tools() -> tuple[list[Tool], list[ToolCall]]:
    release_slow = asyncio.Event()

    @tool
    async def slow() -> str:
        await release_slow.wait()
        return "slow"

    @tool
    async def quick() -> str:
        release_slow.set()
        return "quick"

    calls = [
        ToolCall(id="call-1", name="slow", arguments="{}"),
        ToolCall(id="call-2", name="quick", arguments="{}"),
    ]
    return [slow.tool, quick.tool], calls


async def test_create_yields_results_as_they_finish() -> None:
    tools, calls = slow_and_quick_tools()
    request = GenerationRequest(model="model-1", input="Hi", tools=tools)
    runner = make_runner(FakeGenerationResource(response_with_tools(calls)))

    iterator = await runner.create(request, concurrency=2, in_order=False)
    async with iterator:
        results = [result.call_id async for result in iterator]

    assert results == ["call-2", "call-1"]


async def test_stream_yields_results_as_they_finish() -> None:
    tools, calls = slow_and_quick_tools()
    completed = CompletedEvent(
        provider=Provider.GENERIC,
        model="model-1",
        tool_calls=calls,
        raw={},
    )
    request = GenerationRequest(model="model-1", input="Hi", tools=tools)
    resource = FakeGenerationResource(response_with_tools(calls), [completed])

    async with make_runner(resource).stream(
        request,
        concurrency=2,
        in_order=False,
    ) as events:
        results = [
            event.call_id
            async for event in events
            if isinstance(event, ToolResult)
        ]

    assert results == ["call-2", "call-1"]


def describe(error: Exception) -> str:
    return f"Error: {error}"


async def test_on_error_turns_a_failure_into_the_output() -> None:
    @tool
    def fail() -> str:
        raise ValueError("no such city")

    @tool
    async def lookup() -> dict[str, int]:
        return {"celsius": 15}

    calls = [
        ToolCall(id="call-1", name="fail", arguments="{}"),
        ToolCall(id="call-2", name="lookup", arguments="{}"),
    ]
    request = GenerationRequest(
        model="model-1",
        input="Hi",
        tools=[fail.tool, lookup.tool],
    )
    runner = make_runner(FakeGenerationResource(response_with_tools(calls)))

    iterator = await runner.create(request, on_error=describe)
    results = [result async for result in iterator]

    assert results == [
        ToolResult(call_id="call-1", output="Error: no such city"),
        ToolResult(call_id="call-2", output='{"celsius": 15}'),
    ]


async def test_on_error_gets_serialization_failures_and_may_return_json() -> (
    None
):
    seen: list[Exception] = []

    def record(error: Exception) -> dict[str, str]:
        seen.append(error)
        return {"error": type(error).__name__}

    async def unserializable() -> object:
        return object()

    runner = ToolRunnerIterator(
        {"call-1": Callback(unserializable, (), {})},
        response_with_tools([]),
        on_error=record,
    )
    results = [result.output async for result in runner]

    assert results == ['{"error": "TypeError"}']
    assert [type(error) for error in seen] == [TypeError]


async def test_a_failing_on_error_is_the_failure() -> None:
    original = RuntimeError("callback failed")
    handler_error = LookupError("handler failed")

    async def fail() -> str:
        raise original

    def broken_handler(error: Exception) -> str:
        raise handler_error

    runner = ToolRunnerIterator(
        {"call-1": Callback(fail, (), {})},
        response_with_tools([]),
        on_error=broken_handler,
    )

    with pytest.raises(CallbackExceptionGroup) as caught:
        async for _ in runner:
            pytest.fail("a failed handler produced a result")

    assert caught.value.exceptions == (handler_error,)
    assert handler_error.__context__ is original


async def test_on_error_does_not_receive_an_unrequested_cancellation() -> None:
    seen: list[Exception] = []

    def record(error: Exception) -> str:
        seen.append(error)
        return "recorded"

    async def cancel_itself() -> str:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)
        return "unreachable"

    runner = ToolRunnerIterator(
        {"call-1": Callback(cancel_itself, (), {})},
        response_with_tools([]),
        on_error=record,
    )

    with pytest.raises(CallbackExceptionGroup) as caught:
        async for _ in runner:
            pytest.fail("a cancelled callback produced a result")

    assert isinstance(caught.value.exceptions[0], CallbackCancelledError)
    assert seen == []


async def test_stream_passes_on_error_to_its_callbacks() -> None:
    @tool
    def fail() -> str:
        raise ValueError("no such city")

    calls = [ToolCall(id="call-1", name="fail", arguments="{}")]
    completed = CompletedEvent(
        provider=Provider.GENERIC,
        model="model-1",
        tool_calls=calls,
        raw={},
    )
    request = GenerationRequest(model="model-1", input="Hi", tools=[fail.tool])
    resource = FakeGenerationResource(response_with_tools(calls), [completed])

    async with make_runner(resource).stream(
        request, on_error=describe
    ) as events:
        results = [
            event async for event in events if isinstance(event, ToolResult)
        ]

    assert results == [
        ToolResult(call_id="call-1", output="Error: no such city")
    ]
