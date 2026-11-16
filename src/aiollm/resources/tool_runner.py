"""Utilities and runners for executing tool callbacks on responses."""

from __future__ import annotations

import inspect
import json
from collections.abc import (
    AsyncGenerator,
    Callable,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, Unpack, overload

from pydantic import MISSING

from aiollm.resources.protocols import make_request
from aiollm.session.protocols import CallOptions, ISession, Timeout
from aiollm.types.requests.generation import (
    GenerationRequest,
    NoProviderOptions,
    ToolResult,
)
from aiollm.types.responses.events import CompletedEvent, GenerationEvent

if TYPE_CHECKING:
    from aiollm.enums import Effort, ToolChoice
    from aiollm.http.protocols import HeadersType
    from aiollm.resources.generation import GenerationResource
    from aiollm.resources.stream import GenerationStream
    from aiollm.session.retry.policy import RetryPolicy
    from aiollm.types.requests.generation import (
        ForcedTool,
        InputItem,
        JSONSchema,
        Tool,
        ToolCall,
        ToolType,
    )
    from aiollm.types.responses.generation import GenerationResponse


type ToolRunnerEvent[RawItemT] = GenerationEvent[RawItemT] | ToolResult


def grab_tools(request: GenerationRequest) -> dict[str, Tool]:
    """Extract a name-to-tool mapping from a generation request.

    Args:
        request: The generation request containing tool definitions.

    Returns:
        A dictionary mapping tool names to their ``Tool`` definitions.

    Raises:
        ValueError: If the request does not define any tools.
    """
    if request.tools is MISSING or not request.tools:
        raise ValueError("ToolRunner requires at least one tool")
    return {tool.name: tool for tool in request.tools}


def _serialize_tool_output(result: Any) -> str:
    """Serialize a tool callback result to a JSON string."""
    if isinstance(result, str):
        return result
    return json.dumps(result)


@dataclass(slots=True, frozen=True)
class Callback:
    """A bound tool callback ready for execution."""

    callback: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    @classmethod
    def bind(cls, tool: Tool, tool_call: ToolCall) -> Self:
        """Bind a tool definition and parsed arguments into a Callback.

        Args:
            tool: The tool definition holding the Python callable.
            tool_call: The model's tool call containing raw arguments.

        Returns:
            An initialized ``Callback`` instance.

        Raises:
            ValueError: If the tool name does not match the tool call name.
            TypeError: If the tool has no associated callback.
        """
        if tool.name != tool_call.name:
            raise ValueError(
                f"Tool name mismatch: {tool.name!r} != {tool_call.name!r}"
            )

        callback = tool._tool_callback  # pyright: ignore[reportPrivateUsage]
        if callback is None:
            raise TypeError(f"Tool {tool.name!r} has no associated callback")

        args, kwargs = tool_call.parse_arguments(tool)
        return cls(callback=callback, args=args, kwargs=kwargs)

    async def __call__(self) -> Any:
        """Invoke the bound callback, awaiting it if awaitable."""
        result = self.callback(*self.args, **self.kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def bind_callbacks(
    tool_calls: Sequence[ToolCall],
    tools: Mapping[str, Tool],
) -> dict[str, Callback]:
    """Resolve and bind all tool calls up front before execution.

    Args:
        tool_calls: Sequence of tool calls emitted by the model.
        tools: Available tools keyed by tool name.

    Returns:
        A dictionary mapping tool call IDs to ``Callback`` instances.

    Raises:
        ValueError: If a tool call references an unknown tool name.
    """
    callbacks: dict[str, Callback] = {}
    for call in tool_calls:
        definition = tools.get(call.name)
        if definition is None:
            raise ValueError(f"Unknown tool in response: {call.name!r}")
        callbacks[call.id] = Callback.bind(definition, call)
    return callbacks


class ToolRunnerIterator[RawT]:
    """Single-use async iterator executing bound tool callbacks."""

    def __init__(
        self,
        calls: Mapping[str, Callback],
        original: GenerationResponse[RawT],
    ) -> None:
        """Initialize the iterator with bound callbacks and response."""
        self._calls = iter(calls.items())
        self.original = original

    def __aiter__(self) -> Self:
        """Return self as a single-use async iterator."""
        return self

    async def __anext__(self) -> ToolResult:
        """Execute the next bound callback and return its ``ToolResult``."""
        try:
            call_id, call = next(self._calls)
        except StopIteration:
            raise StopAsyncIteration from None

        result = await call()
        return ToolResult(
            call_id=call_id,
            output=_serialize_tool_output(result),
        )


class ToolStreamRunnerIterator[RawItemT]:
    """Async iterator yielding stream events followed by tool results."""

    def __init__(
        self,
        original: AbstractAsyncContextManager[GenerationStream[RawItemT]],
        tools: Mapping[str, Tool],
    ) -> None:
        """Initialize the stream iterator with a stream and tool map."""
        self.original = original
        self.tools = tools

        self._stream: GenerationStream[RawItemT] | None = None
        self._completed: CompletedEvent[Any] | None = None
        self._tool_iterator: AsyncGenerator[ToolResult] | None = None

        self._original_finished = False
        self._original_entered = False
        self._closed = False
        self._finalized = False

    def __aiter__(self) -> Self:
        """Return self as the async iterator."""
        return self

    async def _enter_original(self) -> GenerationStream[RawItemT]:
        if self._stream is None:
            self._stream = await self.original.__aenter__()
            self._original_entered = True
        return self._stream

    async def _exit_original(self, exc: BaseException | None = None) -> None:
        if not self._original_entered:
            return

        self._original_entered = False
        if exc is None:
            await self.original.__aexit__(None, None, None)
        else:
            await self.original.__aexit__(type(exc), exc, exc.__traceback__)

    async def _iterate_tools(
        self,
        callbacks: Mapping[str, Callback],
    ) -> AsyncGenerator[ToolResult]:
        for call_id, call in callbacks.items():
            result = await call()
            yield ToolResult(
                call_id=call_id,
                output=_serialize_tool_output(result),
            )

    async def _next_stream_event(self) -> GenerationEvent[RawItemT] | None:
        try:
            stream = await self._enter_original()
            event = await anext(stream)
        except StopAsyncIteration:
            self._original_finished = True
            try:
                await self._exit_original()
            except BaseException:
                self._closed = True
                raise

            completed = self._completed
            if completed is None or completed.incomplete_reason is not None:
                self._closed = True
                raise

            callbacks = bind_callbacks(completed.tool_calls, self.tools)
            self._tool_iterator = self._iterate_tools(callbacks)
            return None
        except BaseException as exc:
            self._closed = True
            await self._exit_original(exc)
            raise

        if isinstance(event, CompletedEvent):
            self._completed = event
        return event

    async def __anext__(self) -> ToolRunnerEvent[RawItemT]:
        """Yield upstream events followed by executed ``ToolResult`` items."""
        if self._closed:
            raise StopAsyncIteration

        if not self._original_finished:
            try:
                event = await self._next_stream_event()
            except BaseException:
                self._closed = True
                raise

            if event is not None:
                return event

        if self._tool_iterator is None:
            self._closed = True
            raise StopAsyncIteration

        try:
            return await anext(self._tool_iterator)
        except BaseException:
            self._closed = True
            raise

    async def aclose(self, exc: BaseException | None = None) -> None:
        """Close the tool iterator and exit the underlying stream context."""
        if self._finalized:
            return

        self._closed = True
        self._finalized = True
        try:
            if self._tool_iterator is not None:
                await self._tool_iterator.aclose()
        finally:
            await self._exit_original(exc)


class ToolRunner[
    RawT,
    RawItemT,
    OptionsT: Mapping[str, object] = NoProviderOptions,
    SessionT: ISession = ISession,
]:
    """Wrapper around ``GenerationResource`` executing tool calls."""

    def __init__(
        self,
        resource: GenerationResource[RawT, RawItemT, OptionsT, SessionT],
    ) -> None:
        """Initialize the runner with a generation resource."""
        self.resource = resource

    @overload
    async def create(
        self,
        /,
        *,
        model: str,
        input: str | Sequence[InputItem],
        instructions: str | MISSING = MISSING,
        max_output_tokens: int | MISSING = MISSING,
        end_user_id: str | MISSING = MISSING,
        effort: Effort | MISSING = MISSING,
        output_schema: JSONSchema | MISSING = MISSING,
        tools: ToolType | Sequence[ToolType] | MISSING = MISSING,
        tool_choice: ToolChoice | ForcedTool | MISSING = MISSING,
        parallel_tool_calls: bool | MISSING = MISSING,
        provider_options: OptionsT | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> ToolRunnerIterator[RawT]: ...

    @overload
    async def create(
        self,
        request: GenerationRequest,
        /,
        *,
        provider_options: OptionsT | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> ToolRunnerIterator[RawT]: ...

    async def create(
        self,
        request: GenerationRequest | None = None,
        /,
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
        provider_options: OptionsT | MISSING = MISSING,
        **fields: Any,
    ) -> ToolRunnerIterator[RawT]:
        """Generate a response and return an iterator executing tool calls.

        Args:
            request: What to generate, as a ready request.
            headers: Headers for this request only.
            timeout: Seconds for this request instead of the session's
                limit; ``None`` to keep it.
            retry_policy: Retry policy of this request; ``None`` for the
                session's.
            provider_options: The provider's own fields of this request,
                such as ``{"store": False}`` for OpenAI; a key left out
                is not sent.
            **fields: What to generate, as the fields of a
                ``GenerationRequest``.

        Returns:
            A ``ToolRunnerIterator`` yielding ``ToolResult`` objects for
            each tool call in the response, with the raw response
            available via ``.original``.

        Raises:
            TypeError: Both a request and fields are given, or a tool lacks
                an associated callback.
            ValidationError: The fields do not make a valid request.
            RequestValidationError: A key of ``provider_options`` is not
                one of the provider's options, or a value is invalid;
                raised before anything is sent.
            ValueError: No tools were provided in the request, or the
                response references an unknown tool.
        """
        request = make_request(request, fields)
        tools = grab_tools(request)

        response = await self.resource.create(
            request,
            headers=headers,
            timeout=timeout,
            retry_policy=retry_policy,
            provider_options=provider_options,
        )

        tool_calls = (
            ()
            if response.incomplete_reason is not None
            else response.tool_calls
        )
        callbacks = bind_callbacks(tool_calls, tools)
        return ToolRunnerIterator[RawT](callbacks, response)

    @overload
    def stream(
        self,
        /,
        *,
        model: str,
        input: str | Sequence[InputItem],
        instructions: str | MISSING = MISSING,
        max_output_tokens: int | MISSING = MISSING,
        end_user_id: str | MISSING = MISSING,
        effort: Effort | MISSING = MISSING,
        output_schema: JSONSchema | MISSING = MISSING,
        tools: ToolType | Sequence[ToolType] | MISSING = MISSING,
        tool_choice: ToolChoice | ForcedTool | MISSING = MISSING,
        parallel_tool_calls: bool | MISSING = MISSING,
        provider_options: OptionsT | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[ToolStreamRunnerIterator[RawItemT]]: ...

    @overload
    def stream(
        self,
        request: GenerationRequest,
        /,
        *,
        provider_options: OptionsT | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[ToolStreamRunnerIterator[RawItemT]]: ...

    @asynccontextmanager
    async def stream(
        self,
        request: GenerationRequest | None = None,
        /,
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
        provider_options: OptionsT | MISSING = MISSING,
        **fields: Any,
    ) -> AsyncGenerator[ToolStreamRunnerIterator[RawItemT]]:
        """Open a stream yielding generation events followed by tool results.

        Use with ``async with``.

        Args:
            request: What to generate, as a ready request.
            headers: Headers for this request only.
            timeout: Seconds for this request instead of the session's
                limit; ``None`` to keep it.
            retry_policy: Retry policy of this request; ``None`` for the
                session's.
            provider_options: The provider's own fields of this request,
                such as ``{"store": False}`` for OpenAI; a key left out
                is not sent.
            **fields: What to generate, as the fields of a
                ``GenerationRequest``.

        Returns:
            An async context manager yielding a
            ``ToolStreamRunnerIterator`` that forwards upstream
            ``GenerationEvent`` items and then yields ``ToolResult``
            objects for each tool call once the stream completes.

        Raises:
            TypeError: Both a request and fields are given, or a tool lacks
                an associated callback.
            ValidationError: The fields do not make a valid request.
            RequestValidationError: A key of ``provider_options`` is not
                one of the provider's options, or a value is invalid;
                raised before anything is sent.
            ValueError: No tools were provided in the request, or a tool
                call references an unknown tool.
        """
        request = make_request(request, fields)
        tools = grab_tools(request)
        response = self.resource.stream(
            request,
            headers=headers,
            timeout=timeout,
            retry_policy=retry_policy,
            provider_options=provider_options,
        )
        iterator = ToolStreamRunnerIterator[RawItemT](response, tools)
        try:
            yield iterator
        except BaseException as exc:
            await iterator.aclose(exc)
            raise
        else:
            await iterator.aclose()
