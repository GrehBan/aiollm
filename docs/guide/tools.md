# Tool calls

A tool call is a request from the model for your application to run a
function. The library sends tool definitions and returns calls; it does not
run functions unless you use `generation.tool_runner`. Either way, your
application decides how to handle failures, how many rounds to allow, and
whether a call is safe to run.

```python title="tools.py"
import json
from typing import Any

from aiollm import (
    GenerationResponse,
    IGenerationClient,
    InputItem,
    Tool,
    ToolResult,
    Turn,
    tool,
)

@tool
def get_weather(city: str) -> dict[str, str | int]:
    """Current weather in a city."""
    return {"city": city, "celsius": 15}

async def ask_with_tools(
    client: IGenerationClient, question: str
) -> GenerationResponse[Any]:
    conversation: list[InputItem] = [Turn.user(question)]
    while True:
        response = await client.generation.create(
            model="gpt-6-astra",
            input=conversation,
            tools=get_weather,
        )
        if response.incomplete_reason is not None or not response.tool_calls:
            return response
        conversation.append(response.to_turn())
        for call in response.tool_calls:
            if call.name != "get_weather":
                raise ValueError(f"Unknown tool: {call.name}")
            args, kwargs = call.parse_arguments(get_weather)
            result = get_weather(*args, **kwargs)
            output = result if isinstance(result, str) else json.dumps(result)
            conversation.append(ToolResult(call_id=call.id, output=output))
```

## Define a callable tool

`@tool` wraps a function without changing how you call it in Python. Its
cached `Tool` definition is generated from the function name, parameter
annotations and docstring. Pass the callable itself as `tools=get_weather` or
include it in a sequence such as `tools=[get_weather, get_time]`; there is no
need to write `get_weather.tool` or construct a `GenerationRequest`.

The same keyword fields are accepted by `generation.create()` and
`generation.stream()`. They build a `GenerationRequest` internally. Pass a
ready `GenerationRequest` only when you already have one or want to reuse it;
do not pass both a request and request fields.

An undecorated callable is accepted too, but its schema is generated when the
request is built. Decorate reusable functions once to cache their schema.
`Tool.from_callable()` returns a callback-bearing `Tool` definition;
`Tool.from_model()` and `Tool(...)` define a schema without a callback.
`Tool.from_model()` is useful when you want to validate arguments as a
Pydantic model and dispatch the call yourself.

On the wire, a tool definition has a `name`, an optional `description`, a
`parameters` JSON Schema object, and an optional `strict` flag. Names may use
letters, digits, `_` and `-`, are limited to 128 characters, and must be
unique within the request. The schema is sent as written; each provider may
support only part of JSON Schema.

`Tool.from_callable(function, description=..., strict=...)` uses the
function's name, parameter annotations and docstring; `description=`
overrides the docstring. `@tool(description=..., strict=...)` provides the
same options and exposes the cached definition as `.tool`. Neither executes
the function while defining it.

## Manual buffered workflow

The first example sends a request with a callable tool. When the model returns
calls, `response.to_turn()` adds its turn to the conversation and each
`ToolResult` answers one call. The next `create()` continues that conversation.
The `while` loop is yours to bound; a production agent should set a maximum
number of rounds and decide what to do if a tool fails.

`ToolCall.parse_arguments()` without an argument parses JSON into an object
but does not validate it against the tool schema. Pass the callable or a
Pydantic model to validate: for a callable it returns `(args, kwargs)`, and
for a model it returns the validated model. `strict=True` asks the provider to
enforce the schema too; local validation is still useful when strict mode is
off.

`ToolResult.output` is text. Serialize non-string results as JSON. To report a
tool failure, put an error description in the output; there is no separate
error flag. The adapters check that every call has exactly one result, in
order, before sending the next request.

`response.to_turn()` preserves the provider's native output alongside the
shared calls and text. This lets the same provider continue with its own
reasoning data; another provider receives only the shared conversation.
If `incomplete_reason` is set, do not run the calls: the last call may be
truncated and `to_turn()` refuses to continue that response.

## Choose calls

`tool_choice` accepts `ToolChoice.AUTO` (the default), `ToolChoice.NONE`,
`ToolChoice.REQUIRED`, or `ForcedTool(name=...)` to require one named tool.
`parallel_tool_calls=False` limits the response to one call; both providers
allow several by default. These options require `tools`; a forced name not in
the request and `parallel_tool_calls` with `ToolChoice.NONE` are rejected
before sending.

With `strict=True`, the generated schema must have
`additionalProperties: false` and list every property in `required`;
otherwise the providers return 400. Without `strict=True`, neither provider
requests strict schema adherence.

## Automatic callback runner

The runner is available on a concrete provider resource such as
`OpenAIClient.generation.tool_runner`. It does not belong to the generic
`IGenerationResource` protocol, which guarantees only `create()` and
`stream()`. The runner invokes registered callbacks and packages their
results, but it does not send the next model request; append the response turn
and results, then call generation again as in this example.

`tool_runner.create()` waits for the full response and returns an async
iterator. The callback execution starts when the iterator is consumed, not
when `create()` returns. It binds and validates every call before running the
first callback; an unknown tool, invalid arguments or missing callback
therefore fails without running any callback. Set `concurrency` on
`tool_runner.create()` or `tool_runner.stream()` to limit how many callbacks
run at once; it defaults to `1`, preserving sequential execution. Results
are yielded in tool-call order even when callbacks finish in a different
order, and each result keeps its matching `call_id`; pass `in_order=False` to
get each result as soon as its callback finishes. A finished result that has
not been read yet keeps its slot, so with tool-call order a slow first
callback can hold back the next ones: `buffer_size` (default `0`) is how many
such results may wait beyond `concurrency` while further callbacks start.

The runner executes callbacks with
[aiopulse](https://github.com/GrehBan/aiopulse)'s `CallbackExecutor`.
When a callback or the JSON serialization of its output fails, the runner
cancels unfinished callbacks, waits for them, and raises a
`CallbackExceptionGroup` (`from aiollm import CallbackExceptionGroup`) from
the iteration, even for a single failure. It holds every failure observed
before the callbacks stopped, so `except* ValueError:` handles them by type,
and its `failures` give each one's `index`, the position of its tool call. A
callback that ends with a cancellation the runner did not request, for example
by awaiting a cancelled task, is one of those failures, as a
`CallbackCancelledError` whose `__cause__` is the original `CancelledError`;
only a cancellation of the consumer itself propagates as `CancelledError`,
after unfinished callbacks have been cancelled and awaited. Callbacks already
started may have performed side effects before cancellation.

To let the model read a failure instead, pass `on_error`: it is called with
the `Exception` of a callback, or of serializing its output, and what it
returns becomes that call's `ToolResult.output` (a string as is, anything else
as JSON), while the other callbacks go on. For example,
`on_error=lambda error: f"Error: {error}"` sends the exception's message; keep
in mind that the message goes to the provider. An exception raised by
`on_error` itself is the call's failure, and a cancellation is never passed to
it.

If buffered results are only partially consumed, use the iterator as an async
context manager (`async with calls:`) or call `await calls.aclose()` to cancel
and wait for unfinished callbacks; closing raises the failures recorded so far,
unless the block is already raising an exception, which then takes precedence.
An iteration still waiting in another task ends with `StopAsyncIteration`. The
streaming context closes its callbacks the same way when it exits.
Async callbacks (`async def`, or an object whose `__call__` is one) run in the
event loop. Any other callback runs in a worker thread (`asyncio.to_thread`),
so a blocking one does not stop the loop, and with `concurrency` above `1`
synchronous callbacks run at the same time: they must then be thread-safe, and
an object bound to the thread that made it (an `sqlite3` connection, for
example) must be used from an `async def` callback instead. A thread cannot be
stopped: a cancelled synchronous callback runs to its end after the runner has
closed, and a failure it then ends with is reported through the event loop's
exception handler.

The response is available as `.original`. An incomplete response has no
results and does not invoke callbacks; inspect `incomplete_reason` before
trying to continue it. Strings returned by callbacks become `ToolResult.output`
as-is; other values are encoded with `json.dumps()` and must be JSON
serializable.

```python title="tools_runner.py"
from typing import Any

from aiollm import GenerationResponse, InputItem, Turn, tool
from aiollm.providers.openai import OpenAIClient


@tool
def get_weather(city: str) -> dict[str, str | int]:
    """Get the current weather in a city."""
    return {"city": city, "celsius": 15}


async def ask_with_runner(
    client: OpenAIClient, question: str
) -> GenerationResponse[Any]:
    conversation: list[InputItem] = [Turn.user(question)]
    while True:
        calls = await client.generation.tool_runner.create(
            model="gpt-6-astra",
            input=conversation,
            tools=get_weather,
            concurrency=4,
        )
        response = calls.original
        if response.incomplete_reason is not None or not response.tool_calls:
            return response
        conversation.append(response.to_turn())
        async with calls:
            conversation.extend([result async for result in calls])
```

`tool_runner.stream()` is an async context manager. It yields the generation's
events, then yields `ToolResult`s only after a successful `CompletedEvent`.
Consume it to exhaustion to receive the results. If the stream fails or ends
early, callbacks are not run; an incomplete `CompletedEvent` is yielded but
produces no results. As with `create()`, the caller decides whether to continue
the conversation. Only append `completed.to_turn()` when the completion is
not incomplete; a truncated tool turn cannot be continued.

This example runs one streamed round and returns its completion and results.
The caller can append `completed.to_turn()` followed by the results before
the next generation call.

```python title="tools_runner_stream.py"
from typing import Any

from aiollm import CompletedEvent, InputItem, ToolResult, tool
from aiollm.providers.openai import OpenAIClient


@tool
def get_weather(city: str) -> dict[str, str | int]:
    """Get the current weather in a city."""
    return {"city": city, "celsius": 15}


@tool
def get_time() -> str:
    """Get the current time."""
    return "3 PM"


async def stream_round(
    client: OpenAIClient, conversation: list[InputItem]
) -> tuple[CompletedEvent[Any], list[ToolResult]]:
    completed: CompletedEvent[Any] | None = None
    results: list[ToolResult] = []
    async with client.generation.tool_runner.stream(
        model="gpt-6-astra",
        input=conversation,
        tools=[get_weather, get_time],
        concurrency=2,
    ) as events:
        async for event in events:
            if isinstance(event, CompletedEvent):
                completed = event
            elif isinstance(event, ToolResult):
                results.append(event)
    if completed is None:
        raise RuntimeError("The stream ended without a completed event")
    return completed, results
```

A tool can take its arguments from a Pydantic model: `Tool.from_model(name, Model, ...)` sends the model's JSON Schema as `parameters`, and `parse_arguments(Model)` gives the arguments back as the model. For `strict=True`, the model forbids extra fields and has no defaults, since the providers want `additionalProperties: false` and every property required:

```python title="tools_model.py"
from pydantic import BaseModel, ConfigDict, ValidationError


class City(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str


CITY_WEATHER = Tool.from_model(
    "get_weather",
    City,
    description="Current weather in a city.",
    strict=True,
)


def answer(call: ToolCall) -> ToolResult:
    try:
        arguments = call.parse_arguments(City)
    except ValidationError as error:
        output = json.dumps({"error": str(error)})
    else:
        output = json.dumps(get_weather(arguments.city))
    return ToolResult(call_id=call.id, output=output)
```

A streamed call can start a tool before the response ends. Start that way only the tools that are safe to run for a generation that then fails, and run the others once the stream has ended with its `CompletedEvent`:

`generation.stream()` emits `ToolCallStartEvent`, argument
`ToolCallDeltaEvent`s and `ToolCallDoneEvent` as the model writes a call.
`ToolCallDoneEvent` means that call's arguments are complete, not that the
generation succeeded: the stream can still fail before `CompletedEvent`.
`events.final()` returns the terminal event after iteration and preserves the
provider output for `completed.to_turn()`.

The example starts only a read-only callback early. Cancelling its asyncio task
does not stop work already running in `asyncio.to_thread`, so work started
early must remain safe even if the generation later fails.

```python title="tools_stream.py"
import asyncio

from aiollm import ToolCall, ToolCallDoneEvent

READ_ONLY = {"get_weather"}


def run(call: ToolCall) -> str:
    if call.name == "get_weather":
        args, kwargs = call.parse_arguments(get_weather)
        result = get_weather(*args, **kwargs)
        return result if isinstance(result, str) else json.dumps(result)
    return json.dumps({"error": f"unknown tool {call.name}"})


async def stream_round(
    client: IGenerationClient, conversation: list[InputItem]
) -> list[InputItem]:
    early: dict[str, asyncio.Task[str]] = {}
    try:
        async with client.generation.stream(
            model="gpt-6-astra", input=conversation, tools=get_weather
        ) as events:
            async for event in events:
                if (
                    isinstance(event, ToolCallDoneEvent)
                    and event.call.name in READ_ONLY
                ):
                    early[event.call.id] = asyncio.create_task(
                        asyncio.to_thread(run, event.call)
                    )
            completed = await events.final()
    except BaseException:
        for task in early.values():
            task.cancel()
        raise
    # The generation succeeded: now the tools with side effects run.
    results: list[InputItem] = [completed.to_turn()]
    for call in completed.tool_calls:
        output = await early[call.id] if call.id in early else run(call)
        results.append(ToolResult(call_id=call.id, output=output))
    return [*conversation, *results]
```

## Provider-specific tools

The shared callable workflow covers function tools defined by your
application. Provider-hosted tools and their result formats are not shared:
for example, OpenAI web search or file search and Anthropic server tools must
be sent through the providers' own `CreateResponse` or `CreateMessage`
methods. `tool_runner` executes callbacks attached to shared `Tool`
definitions; it does not execute provider-hosted tools.

What each field becomes for each provider is listed in [Shared field mapping](../unification.md).
