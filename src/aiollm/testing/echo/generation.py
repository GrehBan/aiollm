"""Adapter of shared generation to the echo API."""

from __future__ import annotations

import base64
import copy
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Self

from pydantic import MISSING

from aiollm.adapters.base import BaseCompositeAdapter
from aiollm.adapters.stop import Stop
from aiollm.adapters.validation import (
    check_builtin_tools,
    check_conversation,
    check_files,
    checked_options,
    merge_options,
    request_errors,
)
from aiollm.enums import (
    CitationKind,
    IncompleteReason,
    Role,
    ServerToolKind,
    ServerToolPhase,
    ServerToolStatus,
)
from aiollm.http.exceptions.response import APIResponseError
from aiollm.testing.echo.api import PROVIDER
from aiollm.testing.echo.methods import Generate, StreamGenerate
from aiollm.testing.echo.types import (
    CitationEvent,
    CitationItem,
    DoneEvent,
    EchoEvent,
    EchoGenerationOptions,
    EchoResponse,
    EchoUsage,
    FileEvent,
    FileItem,
    RefusalItem,
    ServerToolDoneEvent,
    ServerToolItem,
    ServerToolStartEvent,
    StartEvent,
    TextItem,
    ToolCallItem,
)
from aiollm.testing.echo.types import (
    TextDeltaEvent as EchoTextDelta,
)
from aiollm.testing.echo.types import (
    ToolCallDeltaEvent as EchoToolCallDelta,
)
from aiollm.testing.echo.types import (
    ToolCallDoneEvent as EchoToolCallDone,
)
from aiollm.testing.echo.types import (
    ToolCallStartEvent as EchoToolCallStart,
)
from aiollm.types.requests.builtin import (
    BuiltinTool,
    CodeExecution,
    ProviderTool,
    WebSearch,
)
from aiollm.types.requests.content import (
    ContentPart,
    DocumentPart,
    ImagePart,
    ProviderFile,
)
from aiollm.types.requests.generation import (
    ForcedTool,
    GenerationRequest,
    InputItem,
    NativeOutput,
    ToolCall,
    ToolCallTurn,
    ToolResult,
    Turn,
)
from aiollm.types.responses.builtin import (
    Citation,
    ProducedFile,
    ServerToolCall,
)
from aiollm.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    ServerToolCallEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallDoneEvent,
    ToolCallStartEvent,
)
from aiollm.types.responses.generation import (
    GenerationResponse,
    TokenUsage,
)

FINISHED: Final = frozenset({"end", "tool_calls"})
"""Stops of a generation that reached its end."""

INCOMPLETE_REASONS: Final[Mapping[str, IncompleteReason]] = {
    "max_tokens": IncompleteReason.MAX_TOKENS,
    "refusal": IncompleteReason.REFUSAL,
    "paused": IncompleteReason.PAUSED,
}
"""Shared reason of each early stop that has one; any other is ``OTHER``."""


def to_input(source: str | Sequence[InputItem]) -> list[dict[str, Any]]:
    """Convert the shared ``input`` into the echo ``input``.

    A text becomes one ``user`` message. A turn, or a turn of calls, with
    echo's own ``native`` output goes back as that output; any other
    turn of calls becomes an ``assistant`` message with its text and
    ``tool_calls``. A result is a ``tool`` message.

    Args:
        source: The text or the conversation of the shared request.

    Returns:
        The messages.
    """
    if isinstance(source, str):
        return [{"role": Role.USER.value, "content": source}]
    messages: list[dict[str, Any]] = []
    for item in source:
        match item:
            case Turn(native=NativeOutput(provider=provider) as native) if (
                provider == PROVIDER
            ):
                messages.append(
                    {"role": item.role.value, "output": native.items}
                )
            case Turn():
                messages.append(
                    {"role": item.role.value, "content": _content(item)}
                )
            case ToolResult():
                messages.append(
                    {
                        "role": "tool",
                        "call_id": item.call_id,
                        "output": item.output,
                    }
                )
            case _:
                messages.append(_calls(item))
    return messages


def _content(turn: Turn) -> str | list[dict[str, Any]]:
    """Convert a turn's content.

    Args:
        turn: The turn.

    Returns:
        Its text, or one item per part.
    """
    if isinstance(turn.content, str):
        return turn.content
    return [_part(part) for part in turn.content]


def _part(part: ContentPart) -> dict[str, Any]:
    """Convert a part of a turn.

    An image or a document has ``data`` (base64) and ``media_type``,
    ``url`` or ``file_id``; a document its ``filename`` when given.

    Args:
        part: The part.

    Returns:
        The part's item.
    """
    match part:
        case ImagePart() | DocumentPart():
            item: dict[str, Any] = {"type": part.type}
            if isinstance(part.data, bytes):
                item["data"] = base64.b64encode(part.data).decode()
                item["media_type"] = part.media_type
            elif isinstance(part.file, ProviderFile):
                item["file_id"] = part.file.id
            else:
                item["url"] = part.url
            if isinstance(part, DocumentPart) and part.filename is not MISSING:
                item["filename"] = part.filename
            return item
        case _:
            return {"type": "text", "text": part.text}


def _calls(turn: ToolCallTurn) -> dict[str, Any]:
    """Convert a turn of calls.

    Args:
        turn: The turn.

    Returns:
        An ``assistant`` message: echo's own ``output`` when the turn
        has it, else its text and calls.
    """
    native = turn.native
    if native is not MISSING and native.provider == PROVIDER:
        return {"role": Role.ASSISTANT.value, "output": native.items}
    message: dict[str, Any] = {"role": Role.ASSISTANT.value}
    if turn.text is not MISSING:
        message["content"] = turn.text
    message["tool_calls"] = [
        {"id": call.id, "name": call.name, "arguments": call.arguments}
        for call in turn.tool_calls
    ]
    return message


def to_usage(usage: EchoUsage) -> TokenUsage:
    """Convert the echo token counts.

    Args:
        usage: The counts.

    Returns:
        The shared counts, with their total.
    """
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
    )


def to_stop(stop: str, refusal: str | None) -> Stop:
    """Tell how a generation stopped.

    Args:
        stop: The echo stop.
        refusal: What the model said about a refusal, if anything.

    Returns:
        Nothing set if it reached its end; else the shared and the echo
        reason, and the refusal's text for a refusal.
    """
    if stop in FINISHED:
        return Stop()
    return Stop(
        INCOMPLETE_REASONS.get(stop, IncompleteReason.OTHER),
        stop,
        refusal if stop == "refusal" else None,
    )


def to_server_call(item: ServerToolItem) -> ServerToolCall:
    """Convert a built-in tool call.

    Args:
        item: The echo item.

    Returns:
        The shared call; a kind or status the library lacks is ``OTHER``
        or ``FAILED``.
    """
    kinds = {kind.value: kind for kind in ServerToolKind}
    statuses = {status.value: status for status in ServerToolStatus}
    return ServerToolCall(
        id=item.id,
        kind=kinds.get(item.kind, ServerToolKind.OTHER),
        status=statuses.get(item.status, ServerToolStatus.FAILED),
        queries=item.queries,
        code=item.code,
        error=item.error,
        raw=[item.model_dump(mode="json", exclude_unset=True)],
    )


def to_citation(item: CitationItem) -> Citation:
    """Convert a source.

    Args:
        item: The echo item.

    Returns:
        The shared citation.
    """
    return Citation(
        kind=CitationKind(item.kind),
        url=item.url,
        title=item.title,
        file_id=item.file_id,
        container_id=item.container_id,
        filename=item.filename,
        cited_text=item.cited_text,
        start_index=item.start_index,
        end_index=item.end_index,
    )


def to_file(item: FileItem) -> ProducedFile:
    """Convert a file a built-in tool made.

    Args:
        item: The echo item.

    Returns:
        The shared file.
    """
    return ProducedFile(
        provider=PROVIDER,
        id=item.id,
        filename=item.filename,
        container_id=item.container_id,
        source_call_id=item.call_id,
    )


def _tool_definition(tool: BuiltinTool) -> dict[str, Any]:
    """Convert a built-in tool of the request.

    Args:
        tool: The shared tool, or one of echo's own.

    Returns:
        Its object for ``tools``: a ``ProviderTool``'s definition as it
        is, a code execution with the ids of its files as ``file_ids``,
        another shared tool by its fields.
    """
    match tool:
        case ProviderTool():
            return dict(tool.definition)
        case CodeExecution(files=list() as files):
            return {"type": tool.type, "file_ids": [f.id for f in files]}
        case _:
            return tool.model_dump(mode="json")


def _native(items: list[dict[str, Any]]) -> NativeOutput | None:
    """Keep the output as the next request takes it back.

    Args:
        items: The output items.

    Returns:
        The items as echo's native output; ``None`` without any.
    """
    return NativeOutput(provider=PROVIDER, items=items) if items else None


class EchoGenerationAdapter(
    BaseCompositeAdapter[
        GenerationRequest,
        EchoResponse,
        GenerationResponse[EchoResponse],
        EchoEvent,
        GenerationEvent[EchoEvent],
    ],
):
    """Generates text with ``POST /echo``.

    Every shared field is sent under its own name: the effort as its
    value, the tools as ``name``, ``description``, ``parameters`` and
    ``strict``, the tool choice as its value or ``{"name": ...}``. A
    request the conversation rules refuse raises
    ``RequestValidationError`` before anything is sent.

    Example:
        >>> generation = GenerationResource(session, EchoGenerationAdapter())
    """

    provider = PROVIDER

    def __init__(self) -> None:
        """Start without provider options."""
        self.provider_options: dict[str, Any] = {}

    def with_options(self, options: EchoGenerationOptions, /) -> Self:
        """Return a copy that sends the options with every method.

        Args:
            options: Fields of ``Generate`` the shared request does not
                hold, such as ``{"temperature": 0.2}``.

        Returns:
            A copy of the adapter; this one is left as it is.

        Raises:
            RequestValidationError: A key is not one of the options.
        """
        adapter = copy.copy(self)
        adapter.provider_options = checked_options(
            options, EchoGenerationOptions, self.provider
        )
        return adapter

    def to_method(self, request: GenerationRequest) -> Generate:
        """Build the method that generates at once.

        Args:
            request: The shared request.

        Returns:
            ``Generate`` for the request.

        Raises:
            RequestValidationError: The echo API cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(self.provider):
            return Generate(**fields)

    def to_response(
        self,
        raw: EchoResponse,
        request: GenerationRequest,
    ) -> GenerationResponse[EchoResponse]:
        """Convert a finished generation.

        Args:
            raw: The echo response.
            request: The request it answers.

        Returns:
            Its text, calls, counts and stop, the output as ``native``.
        """
        texts = [
            item.text for item in raw.output if isinstance(item, TextItem)
        ]
        refusals = [
            item.text for item in raw.output if isinstance(item, RefusalItem)
        ]
        stop = to_stop(raw.stop, refusals[0] if refusals else None)
        citations = [
            to_citation(item)
            for item in raw.output
            if isinstance(item, CitationItem)
        ]
        server_calls = [
            to_server_call(item)
            for item in raw.output
            if isinstance(item, ServerToolItem)
        ]
        files = [
            to_file(item) for item in raw.output if isinstance(item, FileItem)
        ]
        return GenerationResponse[EchoResponse](
            id=raw.id,
            provider=self.provider,
            model=raw.model,
            text="".join(texts) if texts else None,
            usage=to_usage(raw.usage),
            incomplete_reason=stop.incomplete_reason,
            raw_incomplete_reason=stop.raw_incomplete_reason,
            refusal=stop.refusal,
            tool_calls=[
                ToolCall(id=item.id, name=item.name, arguments=item.arguments)
                for item in raw.output
                if isinstance(item, ToolCallItem)
            ],
            server_tool_calls=server_calls,
            citations=citations,
            files=files,
            native=_native(
                [
                    item.model_dump(mode="json", exclude_unset=True)
                    for item in raw.output
                ]
            ),
            raw=raw,
        )

    def to_stream_method(self, request: GenerationRequest) -> StreamGenerate:
        """Build the method that streams.

        Args:
            request: The shared request.

        Returns:
            ``StreamGenerate`` for the request.

        Raises:
            RequestValidationError: The echo API cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(self.provider):
            return StreamGenerate(**fields)

    async def to_events(
        self,
        items: AsyncIterator[EchoEvent],
        request: GenerationRequest,
    ) -> AsyncGenerator[GenerationEvent[EchoEvent]]:
        """Convert the events of a stream.

        Text deltas become ``TextDeltaEvent``; the events of a call
        ``ToolCallStartEvent``, ``ToolCallDeltaEvent`` and
        ``ToolCallDoneEvent``; those of a built-in tool call
        ``ServerToolCallEvent``; ``done`` becomes ``CompletedEvent``
        with what the stream told, the built-in calls, citations and
        files among it; every other event is an ``OtherEvent``.

        Args:
            items: The events the echo API sent.
            request: The request the stream answers.

        Yields:
            One shared event per echo event.

        Raises:
            APIResponseError: ``done`` came without ``start``, or a call
                event names a call no ``tool_call_start`` opened.
        """
        state = _StreamState()
        async for item in items:
            match item:
                case StartEvent():
                    state.started = item
                case EchoTextDelta():
                    state.text.append(item.delta)
                    yield TextDeltaEvent[EchoEvent](text=item.delta, raw=item)
                    continue
                case EchoToolCallStart():
                    state.calls[item.id] = (item.name, [])
                    yield ToolCallStartEvent[EchoEvent](
                        call_id=item.id, name=item.name, raw=item
                    )
                    continue
                case EchoToolCallDelta():
                    state.pieces(item.id).append(item.delta)
                    yield ToolCallDeltaEvent[EchoEvent](
                        call_id=item.id, arguments=item.delta, raw=item
                    )
                    continue
                case EchoToolCallDone():
                    yield ToolCallDoneEvent[EchoEvent](
                        call=state.call(item.id), raw=item
                    )
                    continue
                case ServerToolStartEvent() | ServerToolDoneEvent():
                    yield state.server_event(item)
                    continue
                case CitationEvent() | FileEvent():
                    state.keep(item)
                case DoneEvent():
                    yield state.completed(item)
                    continue
                case _:
                    pass
            yield OtherEvent[EchoEvent](raw=item)

    def _fields(self, request: GenerationRequest) -> dict[str, Any]:
        """Take the method fields of a request.

        Args:
            request: The shared request.

        Returns:
            The fields both methods take, the provider options of
            ``with_options`` included.

        Raises:
            RequestValidationError: The conversation is out of order, or
                a part names another provider's file.
        """
        check_conversation(request, self.provider)
        check_files(request, self.provider)
        check_builtin_tools(request, self.provider, (WebSearch, CodeExecution))
        tools = request.tools
        definitions = [
            *(
                []
                if tools is MISSING
                else [t.model_dump(mode="json") for t in tools]
            ),
            *map(_tool_definition, request.builtin_tools),
        ]
        fields: dict[str, Any] = {
            "model": request.model,
            "input": to_input(request.input),
            "instructions": request.instructions,
            "max_output_tokens": request.max_output_tokens,
            "end_user_id": request.end_user_id,
            "effort": (
                MISSING if request.effort is MISSING else request.effort.value
            ),
            "output_schema": request.output_schema,
            "tools": definitions or MISSING,
            "tool_choice": _tool_choice(request),
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        return merge_options(fields, self.provider_options, self.provider)


def _tool_choice(request: GenerationRequest) -> object:
    """Build the echo ``tool_choice``.

    Args:
        request: The shared request.

    Returns:
        The mode's value, ``{"name": ...}`` for a forced tool, or
        ``MISSING`` when the request sets none.
    """
    choice = request.tool_choice
    if isinstance(choice, ForcedTool):
        return {"name": choice.name}
    return choice if choice is MISSING else choice.value


@dataclass
class _StreamState:
    """What a stream has told so far.

    Attributes:
        started: Its ``start`` event, once it came.
        text: The text deltas.
        calls: The name and argument pieces of each call, by id, in the
            order they started.
        server_calls: The built-in tool calls that are done.
        citations: The sources of the text.
        files: The files the built-in tools made.
    """

    started: StartEvent | None = None
    text: list[str] = field(default_factory=list[str])
    calls: dict[str, tuple[str, list[str]]] = field(
        default_factory=dict[str, tuple[str, list[str]]]
    )
    server_calls: list[ServerToolItem] = field(
        default_factory=list[ServerToolItem]
    )
    citations: list[CitationItem] = field(default_factory=list[CitationItem])
    files: list[FileItem] = field(default_factory=list[FileItem])

    def pieces(self, call_id: str) -> list[str]:
        """Find the argument pieces of a call.

        Args:
            call_id: The call's id.

        Returns:
            Its pieces so far.

        Raises:
            APIResponseError: No ``tool_call_start`` opened the call.
        """
        if call_id not in self.calls:
            message = f"Echo stream sent arguments of unknown call {call_id}"
            raise APIResponseError(message)
        return self.calls[call_id][1]

    def call(self, call_id: str) -> ToolCall:
        """Build a call from what the stream told of it.

        Args:
            call_id: The call's id.

        Returns:
            The call, its pieces joined.

        Raises:
            APIResponseError: No ``tool_call_start`` opened the call.
        """
        arguments = "".join(self.pieces(call_id))
        return ToolCall(
            id=call_id, name=self.calls[call_id][0], arguments=arguments
        )

    def keep(self, item: CitationEvent | FileEvent) -> None:
        """Keep a source or a file for the end of the stream.

        Args:
            item: The echo event.
        """
        if isinstance(item, CitationEvent):
            self.citations.append(item.citation)
        else:
            self.files.append(item.file)

    def server_event(
        self, item: ServerToolStartEvent | ServerToolDoneEvent
    ) -> ServerToolCallEvent[EchoEvent]:
        """Convert the event of a built-in tool call.

        Args:
            item: The echo event; a ``server_tool_done`` is kept for the
                end of the stream.

        Returns:
            The shared event of that phase.
        """
        done = isinstance(item, ServerToolDoneEvent)
        if done:
            self.server_calls.append(item.call)
        return ServerToolCallEvent[EchoEvent](
            phase=ServerToolPhase.DONE if done else ServerToolPhase.STARTED,
            call=to_server_call(item.call),
            raw=item,
        )

    def completed(self, done: DoneEvent) -> CompletedEvent[EchoEvent]:
        """Build the event of a complete stream.

        Args:
            done: The stream's ``done`` event.

        Returns:
            The completed event, its output put together as ``native``,
            its built-in calls, citations and files in their own fields.

        Raises:
            APIResponseError: ``done`` came without ``start``.
        """
        if self.started is None:
            message = "Echo stream finished without a start event"
            raise APIResponseError(message)
        stop = to_stop(done.stop, done.refusal)
        text = "".join(self.text) if self.text else None
        calls = [self.call(call_id) for call_id in self.calls]
        items: list[dict[str, Any]] = [
            call.model_dump(mode="json", exclude_unset=True)
            for call in self.server_calls
        ]
        if text is not None:
            items.append({"type": "text", "text": text})
        items += [
            citation.model_dump(mode="json", exclude_unset=True)
            for citation in self.citations
        ]
        items += [
            file.model_dump(mode="json", exclude_unset=True)
            for file in self.files
        ]
        items += [
            {
                "type": "tool_call",
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in calls
        ]
        return CompletedEvent[EchoEvent](
            id=self.started.id,
            provider=PROVIDER,
            model=self.started.model,
            text=text,
            usage=to_usage(done.usage),
            incomplete_reason=stop.incomplete_reason,
            raw_incomplete_reason=stop.raw_incomplete_reason,
            refusal=stop.refusal,
            tool_calls=calls,
            server_tool_calls=[to_server_call(c) for c in self.server_calls],
            citations=[to_citation(c) for c in self.citations],
            files=[to_file(f) for f in self.files],
            native=_native(items),
            raw=done,
        )
