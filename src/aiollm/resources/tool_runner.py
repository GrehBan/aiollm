"""Utilities and runners for executing tool callbacks on responses."""

from __future__ import annotations

import concurrent.futures
import inspect
import json
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

from aiopulse import CallbackExecutor, ExecutionPolicy, invoke
from pydantic import MISSING

from aiollm.resources.protocols import make_request
from aiollm.session.protocols import CallOptions, ISession, Timeout
from aiollm.types.files import ThreadWork
from aiollm.types.requests.generation import (
    GenerationRequest,
    NoProviderOptions,
    ToolResult,
)
from aiollm.types.responses.events import CompletedEvent, GenerationEvent

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from aiopulse.callback import CallbackInvocation

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
        """Invoke the bound callback and return what it produced.

        An async callback runs in the event loop. Any other runs in a
        worker thread, so a blocking callback does not stop the loop; a
        thread cannot be stopped, so a cancelled one runs to its end and
        a failure it then ends with goes to the loop's exception handler
        (``ThreadWork``). An awaitable it returns is awaited in the loop.

        Returns:
            The callback's result.
        """
        if _is_async(self.callback):
            result = self.callback(*self.args, **self.kwargs)
        else:
            result = await ThreadWork(dispose=_dispose).run(
                _call_in_thread, self.callback, self.args, self.kwargs
            )
            if isinstance(result, _Raised):
                raise result.error
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass(slots=True, frozen=True)
class _Raised:
    """An exception a callback raised that must cross the thread as is.

    Attributes:
        error: The exception.
    """

    error: Exception


def _call_in_thread(
    callback: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Call a synchronous callback in its worker thread.

    A ``concurrent.futures.CancelledError`` (a callback that read a
    cancelled future) is returned in a ``_Raised``: asyncio would turn it
    into ``asyncio.CancelledError`` on its way out of the thread, and the
    callback's failure would then look like a cancellation.

    Args:
        callback: The callback.
        args: Its positional arguments.
        kwargs: Its keyword arguments.

    Returns:
        What the callback returned, or the ``_Raised`` error.
    """
    try:
        return callback(*args, **kwargs)
    except concurrent.futures.CancelledError as error:
        return _Raised(error)


def _dispose(result: object) -> None:
    """Release what a synchronous callback returned after a cancel.

    Args:
        result: What the callback returned.

    Raises:
        Exception: The error a ``_Raised`` holds, to be reported.
    """
    if isinstance(result, _Raised):
        raise result.error
    if inspect.iscoroutine(result):
        result.close()


def _is_async(callback: Callable[..., Any]) -> bool:
    """Tell whether a callback is an async function or async callable.

    Args:
        callback: A tool's callback.

    Returns:
        ``True`` for an ``async def`` function, a partial of one, or an
        object whose ``__call__`` is one.
    """
    return inspect.iscoroutinefunction(
        callback
    ) or inspect.iscoroutinefunction(type(callback).__call__)


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


type ToolErrorHandler = Callable[[Exception], object]
"""Turns a failed callback's exception into the output the model reads."""


@invoke
async def _run_tool(
    call_id: str,
    callback: Callback,
    on_error: ToolErrorHandler | None,
) -> ToolResult:
    """Run one bound callback and wrap its output as the call's result.

    Args:
        call_id: The tool call the result answers.
        callback: The bound callback.
        on_error: Makes the output of a callback, or of serializing its
            output, that failed with an ``Exception``; ``None`` lets the
            failure through.

    Returns:
        The callback's output, serialized, under ``call_id``.
    """
    try:
        output = _serialize_tool_output(await callback())
    except Exception as exc:
        if on_error is None:
            raise
        output = _serialize_tool_output(on_error(exc))
    return ToolResult(call_id=call_id, output=output)


def _invocations(
    calls: Mapping[str, Callback],
    on_error: ToolErrorHandler | None,
) -> Iterator[CallbackInvocation[..., ToolResult]]:
    """Turn bound callbacks into invocations for the executor, in order.

    Args:
        calls: Bound callbacks keyed by tool call ID.
        on_error: Makes the output of a failed callback, if given.

    Yields:
        One invocation per call, producing that call's ``ToolResult``.
    """
    for call_id, callback in calls.items():
        yield _run_tool(call_id, callback, on_error)


@dataclass(slots=True, frozen=True, kw_only=True)
class _RunOptions:
    """How the callbacks of one response are run.

    Attributes:
        concurrency: Maximum number of callbacks executing at once.
        buffer_size: Finished results kept beyond ``concurrency`` before
            the next callback waits for the consumer.
        in_order: Yield results in tool-call order; ``False`` yields
            each as soon as its callback finishes.
        on_error: Makes the output of a failed callback, so the model
            reads the failure; ``None`` raises it.
    """

    concurrency: int = 1
    buffer_size: int = 0
    in_order: bool = True
    on_error: ToolErrorHandler | None = None

    def __post_init__(self) -> None:
        """Refuse limits the executor would refuse, before any request.

        Raises:
            ValueError: ``concurrency`` is less than ``1`` or
                ``buffer_size`` is negative.
        """
        if self.concurrency < 1:
            raise ValueError("concurrency must be greater than 0")
        if self.buffer_size < 0:
            raise ValueError("buffer_size must not be negative")


def _executor(
    calls: Mapping[str, Callback],
    run_options: _RunOptions,
) -> CallbackExecutor[ToolResult]:
    """Make the executor that runs the callbacks of one response.

    The first failure cancels the callbacks still running.

    Args:
        calls: Bound callbacks keyed by tool call ID.
        run_options: How the callbacks are run.

    Returns:
        The executor, not entered yet.
    """
    return CallbackExecutor(
        _invocations(calls, run_options.on_error),
        concurrency=run_options.concurrency,
        buffer_size=run_options.buffer_size,
        execution_policy=(
            ExecutionPolicy.SEQUENTIAL
            if run_options.in_order
            else ExecutionPolicy.FIRST_COMPLETED
        ),
    )


class ToolRunnerIterator[RawT]:
    """Single-use async iterator executing tool callbacks for response.

    Runs the callbacks through an ``aiopulse.CallbackExecutor`` and yields
    their results in tool-call order, regardless of completion order,
    unless ``in_order`` is ``False``. The generation response is
    available through ``original``. Use it with ``async with`` to cancel
    and await unfinished callbacks when leaving early.

    Callback failures are raised together in one
    ``CallbackExceptionGroup``; a cancellation the runner did not request
    is one of them, as a ``CallbackCancelledError``.
    """

    def __init__(
        self,
        calls: Mapping[str, Callback],
        original: GenerationResponse[RawT],
        concurrency: int = 1,
        *,
        buffer_size: int = 0,
        in_order: bool = True,
        on_error: ToolErrorHandler | None = None,
    ) -> None:
        """Initialize the iterator.

        Args:
            calls: Mapping of tool call IDs to their bound callbacks.
                Callbacks are executed in the mapping's iteration order.
            original: The generation response containing the tool calls.
                Preserved and exposed through the ``original`` attribute.
            concurrency: Maximum number of callbacks executing concurrently.
            buffer_size: Finished results kept beyond ``concurrency``
                before the next callback waits for the consumer.
            in_order: Yield results in tool-call order; ``False`` yields
                each as soon as its callback finishes.
            on_error: Makes the output of a callback that failed with an
                ``Exception``, so the model reads the failure; ``None``
                raises it.

        Raises:
            ValueError: If ``concurrency`` is less than ``1`` or
                ``buffer_size`` is negative.
        """
        run_options = _RunOptions(
            concurrency=concurrency,
            buffer_size=buffer_size,
            in_order=in_order,
            on_error=on_error,
        )
        self.original = original
        self._executor = _executor(calls, run_options)

    def __aiter__(self) -> Self:
        """Return this iterator.

        Returns:
            This iterator instance.
        """
        return self

    async def __anext__(self) -> ToolResult:
        """Execute callbacks and return the next tool result.

        Returns:
            The next tool result.

        Raises:
            StopAsyncIteration: When every result has been yielded, or the
                iterator is closed.
            CallbackExceptionGroup: The callback failures, after the
                unfinished callbacks have been cancelled.
            CancelledError: If the current task is cancelled; the
                unfinished callbacks are cancelled first.
        """
        return await anext(self._executor)

    async def __aenter__(self) -> Self:
        """Start the callbacks' task group in the current task.

        Returns:
            This iterator instance.
        """
        await self._executor.aenter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Cancel and await unfinished callbacks when leaving the block.

        Args:
            exc_type: Type of the exception raised in the block, if any.
            exc: The exception raised in the block, if any; it takes
                precedence over callback failures.
            tb: Its traceback.

        Raises:
            CallbackExceptionGroup: Callback failures recorded so far,
                when the block raised nothing.
        """
        await self.aclose(exc)

    async def aclose(self, exc: BaseException | None = None) -> None:
        """Close the iterator and cancel any pending callbacks.

        Waits for cancelled callback tasks to finish. Safe to call before
        iteration starts or multiple times; after closing, iteration
        raises ``StopAsyncIteration``.

        Args:
            exc: An exception the caller is already raising; recorded
                callback failures are then dropped in its favour.

        Raises:
            CallbackExceptionGroup: Callback failures recorded so far, when
                ``exc`` is ``None``.
        """
        await self._executor.aclose(exc)


class ToolStreamRunnerIterator[RawItemT]:
    """Async iterator yielding generation events followed by tool results.

    Wraps a generation stream and forwards its events to the consumer.
    After the stream completes successfully, tool callbacks are executed
    through an ``aiopulse.CallbackExecutor``, and their results are yielded
    in the original tool call order, unless ``in_order`` is ``False``.

    The underlying generation stream is opened lazily during iteration.
    """

    def __init__(
        self,
        original: AbstractAsyncContextManager[GenerationStream[RawItemT]],
        tools: Mapping[str, Tool],
        concurrency: int = 1,
        *,
        buffer_size: int = 0,
        in_order: bool = True,
        on_error: ToolErrorHandler | None = None,
    ) -> None:
        """Initialize the stream iterator.

        Args:
            original: Async context manager providing the generation stream.
                The stream is entered lazily when iteration begins.
            tools: Available tool definitions indexed by their names.
                Used to resolve and bind tool calls from the completed
                generation response.
            concurrency: Maximum number of callbacks executing concurrently.
            buffer_size: Finished results kept beyond ``concurrency``
                before the next callback waits for the consumer.
            in_order: Yield results in tool-call order; ``False`` yields
                each as soon as its callback finishes.
            on_error: Makes the output of a callback that failed with an
                ``Exception``, so the model reads the failure; ``None``
                raises it.

        Raises:
            ValueError: If ``concurrency`` is less than ``1`` or
                ``buffer_size`` is negative.
        """
        self._run_options = _RunOptions(
            concurrency=concurrency,
            buffer_size=buffer_size,
            in_order=in_order,
            on_error=on_error,
        )
        self.original = original
        self.tools = tools
        self._runner: CallbackExecutor[ToolResult] | None = None

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
            self._runner = _executor(callbacks, self._run_options)
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
        the order of the corresponding tool calls, or as each callback
        finishes when ``in_order`` is ``False``.

        Returns:
            The next generation event or tool result.

        Raises:
            StopAsyncIteration: When the generation stream and all tool
                callbacks have been exhausted, or the iterator is closed.
            Exception: An exception raised while reading the generation
                stream or binding tool calls.
            CallbackExceptionGroup: The callback failures, after the
                unfinished callbacks have been cancelled.
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

        Safe to call multiple times.

        Args:
            exc: Exception that caused the iterator to close, if any.
                Forwarded to the original context manager's ``__aexit__``;
                recorded callback failures are dropped in its favour.

        Raises:
            CallbackExceptionGroup: Callback failures recorded so far, when
                ``exc`` is ``None``.
        """
        if self._finalized:
            return

        self._closed = True
        self._finalized = True
        try:
            if self._runner is not None:
                await self._runner.aclose(exc)
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
        buffer_size: int = 0,
        in_order: bool = True,
        on_error: ToolErrorHandler | None = None,
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
        buffer_size: int = 0,
        in_order: bool = True,
        on_error: ToolErrorHandler | None = None,
        provider_options: OptionsT | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> ToolRunnerIterator[RawT]: ...

    async def create(
        self,
        request: GenerationRequest | None = None,
        /,
        *,
        concurrency: int = 1,
        buffer_size: int = 0,
        in_order: bool = True,
        on_error: ToolErrorHandler | None = None,
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
            buffer_size: Finished results kept beyond ``concurrency``
                before the next callback waits for the consumer; with
                ``0``, a finished result that has not been read yet still
                takes one of the ``concurrency`` slots.
            in_order: Yield results in tool-call order (the default);
                ``False`` yields each as soon as its callback finishes.
            on_error: Called with the exception of a callback, or of
                serializing its output, that failed with an
                ``Exception``; what it returns is that call's output
                (a string as is, anything else as JSON), so the model
                reads the failure and the other callbacks go on.
                ``None`` (the default) raises the failure instead.
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
            available via ``.original``; use it with ``async with`` to
            cancel unfinished callbacks when leaving early.

        Raises:
            TypeError: Both a request and fields are given, or a tool lacks
                an associated callback.
            ValueError: ``concurrency`` is less than ``1``,
                ``buffer_size`` is negative, no tools were
                provided in the request, or the response references an
                unknown tool.
            CallbackExceptionGroup: Raised while iterating: the failures
                of callbacks or of serializing their output (without
                ``on_error``, or raised by it), after the unfinished
                callbacks have been cancelled; a cancellation
                inside a callback that the runner did not request is a
                ``CallbackCancelledError`` among them.
            CancelledError: The consumer was cancelled, after unfinished
                callbacks have been cancelled.
            ValidationError: The fields do not make a valid request.
            RequestValidationError: A key of ``provider_options`` is not
                one of the provider's options, or a value is invalid;
                raised before anything is sent.
        """
        run_options = _RunOptions(
            concurrency=concurrency,
            buffer_size=buffer_size,
            in_order=in_order,
            on_error=on_error,
        )
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
            concurrency=run_options.concurrency,
            buffer_size=run_options.buffer_size,
            in_order=run_options.in_order,
            on_error=run_options.on_error,
        )

    @overload
    def stream(
        self,
        /,
        *,
        concurrency: int = 1,
        buffer_size: int = 0,
        in_order: bool = True,
        on_error: ToolErrorHandler | None = None,
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
        buffer_size: int = 0,
        in_order: bool = True,
        on_error: ToolErrorHandler | None = None,
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
        buffer_size: int = 0,
        in_order: bool = True,
        on_error: ToolErrorHandler | None = None,
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
            buffer_size: Finished results kept beyond ``concurrency``
                before the next callback waits for the consumer; with
                ``0``, a finished result that has not been read yet still
                takes one of the ``concurrency`` slots.
            in_order: Yield results in tool-call order (the default);
                ``False`` yields each as soon as its callback finishes.
            on_error: Called with the exception of a callback, or of
                serializing its output, that failed with an
                ``Exception``; what it returns is that call's output
                (a string as is, anything else as JSON), so the model
                reads the failure and the other callbacks go on.
                ``None`` (the default) raises the failure instead.
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
            ValueError: ``concurrency`` is less than ``1``,
                ``buffer_size`` is negative, no tools were
                provided in the request, or a tool call references an
                unknown tool.
            CallbackExceptionGroup: Raised while iterating: the failures
                of callbacks or of serializing their output (without
                ``on_error``, or raised by it), after the unfinished
                callbacks have been cancelled; a cancellation
                inside a callback that the runner did not request is a
                ``CallbackCancelledError`` among them.
            CancelledError: The consumer was cancelled, after unfinished
                callbacks have been cancelled.
            ValidationError: The fields do not make a valid request.
            RequestValidationError: A key of ``provider_options`` is not
                one of the provider's options, or a value is invalid;
                raised before anything is sent.
        """
        run_options = _RunOptions(
            concurrency=concurrency,
            buffer_size=buffer_size,
            in_order=in_order,
            on_error=on_error,
        )
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
            concurrency=run_options.concurrency,
            buffer_size=run_options.buffer_size,
            in_order=run_options.in_order,
            on_error=run_options.on_error,
        )
        try:
            yield iterator
        except BaseException as exc:
            await iterator.aclose(exc)
            raise
        else:
            await iterator.aclose()
