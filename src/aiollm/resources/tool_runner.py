"""Utilities and runners for executing tool callbacks on responses."""

from __future__ import annotations

import inspect
import json
from asyncio import FIRST_COMPLETED, CancelledError, Task, TaskGroup, wait
from collections import deque
from collections.abc import (
    AsyncGenerator,
    Callable,
    Mapping,
    Sequence,
)
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
)
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


@dataclass(frozen=True, slots=True)
class _CallbackOutcome:
    result: ToolResult | None = None
    error: Exception | None = None


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


class CallbackRunner:
    """Run callbacks concurrently and yield results in input order.

    At most ``concurrency`` callbacks are scheduled at once. Results are
    yielded in the original mapping order, regardless of completion order.
    When a callback failure is observed, the remaining tasks are cancelled
    and the exception is propagated. If multiple failures are observed
    together, the first one in input order is propagated. Cancellation from
    either the consumer or a callback cancels all pending tasks before it
    propagates.

    A runner is single-use. It closes itself once the mapping is exhausted
    and every result has been yielded, or when an error or cancellation
    occurs.

    Example:
        >>> runner = CallbackRunner(calls, concurrency=4)
        >>> async for result in runner:
        ...     handle(result)
    """

    def __init__(
        self,
        calls: Mapping[str, Callback],
        concurrency: int = 1,
    ) -> None:
        """Initialize the runner.

        Args:
            calls: Mapping of call ID to async callback. Iterated lazily,
                in insertion order.
            concurrency: Maximum number of callbacks executing at the
                same time. Defaults to ``1``.

        Raises:
            ValueError: If ``concurrency`` is less than 1.
        """
        _check_concurrency(concurrency)

        self._calls = iter(calls.items())

        self._concurrency = concurrency
        self._task_group = TaskGroup()
        self._entered = False
        self._closed = False
        self._exhausted = False
        self._pending: deque[Task[_CallbackOutcome]] = deque()

    def __aiter__(self) -> Self:
        """Return the runner itself as the async iterator."""
        return self

    async def __anext__(self) -> ToolResult:
        """Return the next result in input order.

        Lazily enters the task group, fills the bounded task window, and
        waits for callback completion while preserving input order.

        Returns:
            The ``ToolResult`` of the next callback.

        Raises:
            StopAsyncIteration: When all results have been yielded or the
                runner is already closed.
            CancelledError: If the consumer or a callback is cancelled; the
                runner is closed before re-raising.
            Exception: Any exception raised by a callback; the runner is
                closed before re-raising.
        """
        if self._closed:
            raise StopAsyncIteration

        await self.aenter()
        self._schedule()

        if not self._pending:
            await self.aclose()
            raise StopAsyncIteration from None

        try:
            task = self._pending[0]
            while True:
                pending = tuple(self._pending)
                done, _ = await wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                outcomes = [
                    (completed, completed.result())
                    for completed in pending
                    if completed in done
                ]
                failure = next(
                    (
                        outcome.error
                        for _, outcome in outcomes
                        if outcome.error is not None
                    ),
                    None,
                )
                if failure is not None:
                    await self.aclose()
                    raise failure

                if task in done:
                    outcome = next(
                        outcome
                        for completed, outcome in outcomes
                        if completed is task
                    )
                    break
        except CancelledError:
            await self.aclose()
            raise

        self._pending.popleft()
        self._schedule()

        if not self._pending and self._exhausted:
            await self.aclose()

        assert outcome.result is not None
        return outcome.result

    def _schedule(self) -> None:
        """Fill the task window up to the configured concurrency.

        Pulls callbacks from the iterator and creates tasks in the task
        group. Sets ``_exhausted`` once the iterator runs out.
        """
        while len(self._pending) < self._concurrency:
            try:
                call_id, callback = next(self._calls)
            except StopIteration:
                self._exhausted = True
                return

            task = self._task_group.create_task(
                self.run_callback(call_id, callback)
            )
            self._pending.append(task)

    async def run_callback(
        self,
        call_id: str,
        callback: Callback,
    ) -> _CallbackOutcome:
        """Execute a single callback and package its result or failure.

        Non-cancellation exceptions are captured rather than raised, so a
        failing callback does not tear down the task group. The consumer
        detects and re-raises the first failure from ``__anext__``.

        Args:
            call_id: Identifier attached to the resulting ``ToolResult``.
            callback: Async callable to execute.

        Returns:
            An outcome holding either the serialized ``ToolResult`` or the
            captured exception.
        """
        try:
            output = await callback()
            result = ToolResult(
                call_id=call_id,
                output=_serialize_tool_output(output),
            )
        except Exception as exc:
            return _CallbackOutcome(error=exc)
        return _CallbackOutcome(result=result)

    async def aenter(self) -> None:
        """Enter the underlying task group. Idempotent.

        Called automatically on first iteration; call it manually only if
        you need the group started earlier.
        """
        if self._closed:
            raise RuntimeError("CallbackRunner is closed")
        if self._entered:
            return
        await self._task_group.__aenter__()
        self._entered = True

    async def aclose(self) -> None:
        """Cancel pending tasks and close the runner. Idempotent.

        Cancels every pending task, then exits the task group, which waits
        for the cancelled tasks to finish. Safe to call multiple times;
        subsequent iteration raises ``StopAsyncIteration``.
        """
        if self._closed:
            return

        self._closed = True

        while self._pending:
            task = self._pending.popleft()
            task.cancel()

        if self._entered:
            await self._task_group.__aexit__(None, None, None)

        self._pending.clear()


def _check_concurrency(concurrency: int) -> None:
    if concurrency < 1:
        raise ValueError("concurrency must be greater than 0")


class ToolRunnerIterator[RawT]:
    """Single-use async iterator executing tool callbacks for response.

    Wraps a :class:`CallbackRunner` and exposes its results through the
    async iterator protocol. The original generation response is available
    through ``original``.

    Results are yielded in the order of the input mapping, regardless of
    callback completion order. The first callback failure is propagated.
    """

    def __init__(
        self,
        calls: Mapping[str, Callback],
        original: GenerationResponse[RawT],
        concurrency: int,
    ) -> None:
        """Initialize the iterator.

        Args:
            calls: Mapping of tool call IDs to their bound callbacks.
                Callbacks are executed in the mapping's iteration order.
            original: The generation response containing the tool calls.
                Preserved and exposed through the ``original`` attribute.
            concurrency: Maximum number of callbacks executing concurrently.

        Raises:
            ValueError: If ``concurrency`` is less than ``1``.
        """
        _check_concurrency(concurrency)
        self.original = original
        self._calls = calls
        self._concurrency = concurrency
        self._runner: CallbackRunner | None = None
        self._closed = False

    def __aiter__(self) -> Self:
        """Return this iterator, initializing the callback runner if needed.

        The runner is created lazily and is not replaced on subsequent
        calls, ensuring that iteration remains single-use.

        Returns:
            This iterator instance.
        """
        if self._runner is None and not self._closed:
            self._runner = CallbackRunner(
                self._calls,
                concurrency=self._concurrency,
            )
        return self

    async def __anext__(self) -> ToolResult:
        """Execute callbacks and return the next tool result.

        Results are yielded in the input mapping's order, regardless of
        callback completion order.

        Returns:
            The next tool result.

        Raises:
            StopAsyncIteration: When all callbacks have been executed and
                their results have been yielded, or the iterator is closed.
            Exception: An exception raised by a callback, propagated after
                the runner has been closed.
            CancelledError: If the current task is cancelled. The runner
                is closed before cancellation is propagated.
        """
        if self._closed:
            raise StopAsyncIteration

        if self._runner is None:
            self.__aiter__()

        assert self._runner is not None
        return await anext(self._runner)

    async def aclose(self) -> None:
        """Close the iterator and cancel any pending callbacks.

        Waits for cancelled callback tasks to finish and releases resources
        held by the underlying runner.

        Safe to call before iteration starts or multiple times.
        After closing, further iteration raises ``StopAsyncIteration``.
        """
        self._closed = True
        if self._runner is not None:
            await self._runner.aclose()


class ToolStreamRunnerIterator[RawItemT]:
    """Async iterator yielding generation events followed by tool results.

    Wraps a generation stream and forwards its events to the consumer.
    After the stream completes successfully, tool callbacks are executed
    through a :class:`CallbackRunner`, and their results are yielded in
    the original tool call order.

    The underlying generation stream is opened lazily during iteration.
    """

    def __init__(
        self,
        original: AbstractAsyncContextManager[GenerationStream[RawItemT]],
        tools: Mapping[str, Tool],
        concurrency: int,
    ) -> None:
        """Initialize the stream iterator.

        Args:
            original: Async context manager providing the generation stream.
                The stream is entered lazily when iteration begins.
            tools: Available tool definitions indexed by their names.
                Used to resolve and bind tool calls from the completed
                generation response.
            concurrency: Maximum number of callbacks executing concurrently.

        Raises:
            ValueError: If ``concurrency`` is less than ``1``.
        """
        _check_concurrency(concurrency)
        self.original = original
        self.tools = tools
        self._concurrency = concurrency
        self._runner: CallbackRunner | None = None

        self._stream: GenerationStream[RawItemT] | None = None
        self._completed: CompletedEvent[Any] | None = None

        self._original_finished = False
        self._original_entered = False
        self._closed = False
        self._finalized = False

    def __aiter__(self) -> Self:
        """Return this iterator.

        Returns:
            This iterator instance.
        """
        return self

    async def _enter_original(self) -> GenerationStream[RawItemT]:
        """Enter the original context manager and return its generation stream.

        Enters the context manager only once and caches the resulting stream.

        Returns:
            The active generation stream.
        """
        if self._stream is None:
            self._stream = await self.original.__aenter__()
            self._original_entered = True
        return self._stream

    async def _exit_original(
        self,
        exc: BaseException | None = None,
    ) -> None:
        """Exit the original generation stream context manager.

        Does nothing if the context manager has not been entered or has
        already been exited.

        Args:
            exc: Exception that caused the stream to close, if any.
                Passed to the original context manager's ``__aexit__``.
        """
        if not self._original_entered:
            return

        self._original_entered = False
        if exc is None:
            await self.original.__aexit__(None, None, None)
        else:
            await self.original.__aexit__(type(exc), exc, exc.__traceback__)

    async def _next_stream_event(
        self,
    ) -> GenerationEvent[RawItemT] | None:
        """Fetch the next generation event or initialize callback execution.

        When the generation stream is exhausted, exits its context manager
        and initializes the callback runner using tool calls from the
        completed response.

        Returns:
            The next generation event, or ``None`` when the stream has
            finished and callback execution has been initialized.

        Raises:
            StopAsyncIteration: If the stream finishes without a completed
                response or the response is incomplete.
            BaseException: If reading from the stream, exiting its context,
                or binding tool callbacks fails.
        """
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
            self._runner = CallbackRunner(
                callbacks,
                concurrency=self._concurrency,
            )
            return None
        except BaseException as exc:
            self._closed = True
            await self._exit_original(exc)
            raise

        if isinstance(event, CompletedEvent):
            self._completed = event

        return event

    async def __anext__(self) -> ToolRunnerEvent[RawItemT]:
        """Yield generation events followed by tool callback results.

        Generation events are forwarded unchanged. Once the generation
        stream completes successfully, callback results are yielded in
        the order of the corresponding tool calls.

        Returns:
            The next generation event or tool result.

        Raises:
            StopAsyncIteration: When the generation stream and all tool
                callbacks have been exhausted, or the iterator is closed.
            Exception: An exception raised while reading the generation
                stream, binding tool calls, or executing callbacks.
            CancelledError: If the current task is cancelled.
        """
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

        assert self._runner is not None
        try:
            return await anext(self._runner)
        except StopAsyncIteration:
            self._closed = True
            raise
        except BaseException:
            self._closed = True
            raise

    async def aclose(self, exc: BaseException | None = None) -> None:
        """Close callback execution and exit the generation stream context.

        Cancels pending callback tasks and waits for their termination.
        Also exits the original stream context manager if it is still active.

        Args:
            exc: Exception that caused the iterator to close, if any.
                Forwarded to the original context manager's ``__aexit__``.

        Safe to call multiple times.
        """
        if self._finalized:
            return

        self._closed = True
        self._finalized = True
        try:
            if self._runner is not None:
                await self._runner.aclose()
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
        """Initialize the runner.

        Args:
            resource: The generation resource used to create responses.
        """
        self.resource = resource

    @overload
    async def create(
        self,
        /,
        *,
        concurrency: int = 1,
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
        concurrency: int = 1,
        provider_options: OptionsT | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> ToolRunnerIterator[RawT]: ...

    async def create(
        self,
        request: GenerationRequest | None = None,
        /,
        *,
        concurrency: int = 1,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
        provider_options: OptionsT | MISSING = MISSING,
        **fields: Any,
    ) -> ToolRunnerIterator[RawT]:
        """Generate a response and return an iterator executing tool calls.

        Args:
            request: What to generate, as a ready request.
            concurrency: Maximum number of callbacks executing concurrently.
                Defaults to ``1``, which executes them sequentially.
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
            ValueError: ``concurrency`` is less than ``1``, no tools were
                provided in the request, or the response references an
                unknown tool.
            Exception: A callback or serialization failure, after unfinished
                callbacks have been cancelled.
            CancelledError: The consumer or a callback was cancelled, after
                unfinished callbacks have been cancelled.
            ValidationError: The fields do not make a valid request.
            RequestValidationError: A key of ``provider_options`` is not
                one of the provider's options, or a value is invalid;
                raised before anything is sent.
        """
        _check_concurrency(concurrency)
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
        return ToolRunnerIterator[RawT](
            callbacks,
            response,
            concurrency=concurrency,
        )

    @overload
    def stream(
        self,
        /,
        *,
        concurrency: int = 1,
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
        concurrency: int = 1,
        provider_options: OptionsT | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[ToolStreamRunnerIterator[RawItemT]]: ...

    @asynccontextmanager
    async def stream(
        self,
        request: GenerationRequest | None = None,
        /,
        *,
        concurrency: int = 1,
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
            concurrency: Maximum number of callbacks executing concurrently.
                Defaults to ``1``, which executes them sequentially.
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
            ValueError: ``concurrency`` is less than ``1``, no tools were
                provided in the request, or a tool call references an
                unknown tool.
            Exception: A callback or serialization failure, after unfinished
                callbacks have been cancelled.
            CancelledError: The consumer or a callback was cancelled, after
                unfinished callbacks have been cancelled.
            ValidationError: The fields do not make a valid request.
            RequestValidationError: A key of ``provider_options`` is not
                one of the provider's options, or a value is invalid;
                raised before anything is sent.
        """
        _check_concurrency(concurrency)
        request = make_request(request, fields)
        tools = grab_tools(request)
        response = self.resource.stream(
            request,
            headers=headers,
            timeout=timeout,
            retry_policy=retry_policy,
            provider_options=provider_options,
        )
        iterator = ToolStreamRunnerIterator[RawItemT](
            response,
            tools,
            concurrency=concurrency,
        )
        try:
            yield iterator
        except BaseException as exc:
            await iterator.aclose(exc)
            raise
        else:
            await iterator.aclose()
