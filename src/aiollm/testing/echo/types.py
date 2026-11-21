"""Types of the echo API: its response, stream events, files and options."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import Discriminator, Field, Tag

from aiollm.types.base import BaseAPITypeExtra
from aiollm.types.unknown import UNKNOWN, BaseUnknown, by_type


class TextItem(BaseAPITypeExtra):
    """Text the model wrote.

    Attributes:
        type: Always ``text``.
        text: The text.
    """

    type: Literal["text"] = "text"
    text: str


class RefusalItem(BaseAPITypeExtra):
    """The model's refusal to answer.

    Attributes:
        type: Always ``refusal``.
        text: What the model says about it.
    """

    type: Literal["refusal"] = "refusal"
    text: str


class ToolCallItem(BaseAPITypeExtra):
    """A call of one of the request's tools.

    Attributes:
        type: Always ``tool_call``.
        id: The call's id, which its result names.
        name: The tool.
        arguments: The arguments, as JSON text.
    """

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: str


class ServerToolItem(BaseAPITypeExtra):
    """A call of a built-in tool, which the echo API ran itself.

    Attributes:
        type: Always ``server_tool``.
        id: The call's id.
        kind: What it does: ``web_search``, ``code_execution`` or
            another kind of ``ServerToolKind``.
        status: ``in_progress``, ``completed``, ``incomplete`` or
            ``failed``.
        queries: What a search looked for.
        code: The code a code execution ran.
        error: What went wrong, for a failed call.
        result: The call's result, in the echo API's own words.
    """

    type: Literal["server_tool"] = "server_tool"
    id: str
    kind: str
    status: str = "completed"
    queries: list[str] = Field(default_factory=list[str])
    code: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class CitationItem(BaseAPITypeExtra):
    """A source the text rests on.

    Attributes:
        type: Always ``citation``.
        kind: ``url``, ``file`` or ``container_file``.
        url: The page, for ``url``.
        title: The title of the page or file.
        file_id: The file's id, for a file.
        container_id: The container's id, for ``container_file``.
        filename: The file's name.
        cited_text: The cited words of the source.
        start_index: Where the cited part of the text starts.
        end_index: Where it ends.
    """

    type: Literal["citation"] = "citation"
    kind: str
    url: str | None = None
    title: str | None = None
    file_id: str | None = None
    container_id: str | None = None
    filename: str | None = None
    cited_text: str | None = None
    start_index: int | None = None
    end_index: int | None = None


OUTPUT_TAGS = {
    "text": "text",
    "refusal": "refusal",
    "tool_call": "tool_call",
    "server_tool": "server_tool",
    "citation": "citation",
}
"""Tag of each output ``type`` the echo types model."""

type OutputItem = Annotated[
    Annotated[TextItem, Tag("text")]
    | Annotated[RefusalItem, Tag("refusal")]
    | Annotated[ToolCallItem, Tag("tool_call")]
    | Annotated[ServerToolItem, Tag("server_tool")]
    | Annotated[CitationItem, Tag("citation")]
    | Annotated[BaseUnknown, Tag(UNKNOWN)],
    Discriminator(by_type(OUTPUT_TAGS)),
]
"""An item of a response's ``output``; an unknown ``type`` is kept."""


class EchoUsage(BaseAPITypeExtra):
    """Token counts of a generation.

    Attributes:
        input_tokens: Tokens of the request.
        output_tokens: Tokens generated.
    """

    input_tokens: int
    output_tokens: int


class EchoResponse(BaseAPITypeExtra):
    """A finished generation, the answer of ``POST /echo``.

    Attributes:
        id: The generation's id.
        model: The model that generated it.
        output: What the model wrote and called, in order.
        stop: Why it stopped: ``end`` or ``tool_calls`` when it reached
            its end, ``max_tokens``, ``refusal``, ``paused`` or another
            reason when it stopped early.
        usage: Its token counts.
    """

    id: str
    model: str
    output: list[OutputItem]
    stop: str
    usage: EchoUsage


class StartEvent(BaseAPITypeExtra):
    """The first event of a stream.

    Attributes:
        type: Always ``start``.
        id: The generation's id.
        model: The model that generates it.
    """

    type: Literal["start"] = "start"
    id: str
    model: str


class TextDeltaEvent(BaseAPITypeExtra):
    """A piece of the text.

    Attributes:
        type: Always ``text_delta``.
        delta: The piece.
    """

    type: Literal["text_delta"] = "text_delta"
    delta: str


class ToolCallStartEvent(BaseAPITypeExtra):
    """The start of a tool call.

    Attributes:
        type: Always ``tool_call_start``.
        id: The call's id.
        name: The tool.
    """

    type: Literal["tool_call_start"] = "tool_call_start"
    id: str
    name: str


class ToolCallDeltaEvent(BaseAPITypeExtra):
    """A piece of a call's arguments.

    Attributes:
        type: Always ``tool_call_delta``.
        id: The call's id.
        delta: The piece of the JSON text.
    """

    type: Literal["tool_call_delta"] = "tool_call_delta"
    id: str
    delta: str


class ToolCallDoneEvent(BaseAPITypeExtra):
    """The end of a tool call: its arguments are complete.

    Attributes:
        type: Always ``tool_call_done``.
        id: The call's id.
    """

    type: Literal["tool_call_done"] = "tool_call_done"
    id: str


class ServerToolStartEvent(BaseAPITypeExtra):
    """A built-in tool call begins.

    Attributes:
        type: Always ``server_tool_start``.
        call: The call as it is known at the start.
    """

    type: Literal["server_tool_start"] = "server_tool_start"
    call: ServerToolItem


class ServerToolDoneEvent(BaseAPITypeExtra):
    """A built-in tool call has its result, or failed.

    Attributes:
        type: Always ``server_tool_done``.
        call: The whole call.
    """

    type: Literal["server_tool_done"] = "server_tool_done"
    call: ServerToolItem


class CitationEvent(BaseAPITypeExtra):
    """A source of the text.

    Attributes:
        type: Always ``citation``.
        citation: The source.
    """

    type: Literal["citation"] = "citation"
    citation: CitationItem


class DoneEvent(BaseAPITypeExtra):
    """The last event of a complete stream.

    Attributes:
        type: Always ``done``.
        stop: Why the generation stopped, as ``EchoResponse.stop``.
        usage: Its token counts.
        refusal: What the model says about a refusal, if it refused.
    """

    type: Literal["done"] = "done"
    stop: str
    usage: EchoUsage
    refusal: str | None = None


EVENT_TAGS = {
    "start": "start",
    "text_delta": "text_delta",
    "tool_call_start": "tool_call_start",
    "tool_call_delta": "tool_call_delta",
    "tool_call_done": "tool_call_done",
    "server_tool_start": "server_tool_start",
    "server_tool_done": "server_tool_done",
    "citation": "citation",
    "done": "done",
}
"""Tag of each event ``type`` the echo types model."""

type EchoEvent = Annotated[
    Annotated[StartEvent, Tag("start")]
    | Annotated[TextDeltaEvent, Tag("text_delta")]
    | Annotated[ToolCallStartEvent, Tag("tool_call_start")]
    | Annotated[ToolCallDeltaEvent, Tag("tool_call_delta")]
    | Annotated[ToolCallDoneEvent, Tag("tool_call_done")]
    | Annotated[ServerToolStartEvent, Tag("server_tool_start")]
    | Annotated[ServerToolDoneEvent, Tag("server_tool_done")]
    | Annotated[CitationEvent, Tag("citation")]
    | Annotated[DoneEvent, Tag("done")]
    | Annotated[BaseUnknown, Tag(UNKNOWN)],
    Discriminator(by_type(EVENT_TAGS)),
]
"""An event of a stream; an unknown ``type`` is kept."""


class EchoFile(BaseAPITypeExtra):
    """A stored file.

    Attributes:
        id: The file's id.
        filename: Its name.
        bytes: Its size in bytes.
        created_at: When it was stored, in seconds since the epoch.
        expires_at: When it expires, in seconds since the epoch;
            ``None`` if it is kept.
    """

    id: str
    filename: str
    bytes: int
    created_at: int
    expires_at: int | None = None


class EchoFilePage(BaseAPITypeExtra):
    """A page of the stored files, newest first.

    Attributes:
        data: The page's files.
        next: Where the next page starts; ``None`` after the last one.
    """

    data: list[EchoFile]
    next: str | None = None


class EchoDeleted(BaseAPITypeExtra):
    """The answer to a deletion.

    Attributes:
        id: The deleted file's id.
        deleted: Always ``True``.
    """

    id: str
    deleted: bool


class EchoGenerationOptions(TypedDict, total=False):
    """Fields of ``POST /echo`` that the shared request does not hold.

    Given to ``client.generation.create()`` or ``stream()`` as
    ``provider_options``.

    Attributes:
        temperature: Sampling temperature, 0 to 2.
        stop: Texts that end the generation where they appear.
    """

    temperature: Annotated[float, Field(ge=0, le=2)]
    stop: list[str]


class EchoFileOptions(TypedDict, total=False):
    """Fields of ``POST /files`` that the shared upload does not hold.

    Given to ``client.files.upload()`` as ``provider_options``.

    Attributes:
        purpose: What the file is for.
    """

    purpose: str
