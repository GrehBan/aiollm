"""Identifiers shared by all layers: providers, formats and shared values."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler


class Provider(str):
    """API provider an error or integration belongs to.

    An open value, not an enum: every provider package declares its own,
    ``PROVIDER: Final = Provider("openai")`` in its ``api.py``, and the
    core knows only ``GENERIC``. Two providers are the same when their
    values are equal, so compare them with ``==``, never with ``is``.
    A ``Provider`` is a ``str``: it formats, hashes and compares as
    its value, and a pydantic field of this type takes a non-empty
    string.

    Attributes:
        GENERIC: Not tied to a provider, e.g. the base error resolver.

    Example:
        >>> ECHO = Provider("echo")
        >>> ECHO == "echo", ECHO.value
        (True, 'echo')
        >>> ECHO == Provider.GENERIC
        False
    """

    __slots__ = ()

    GENERIC: ClassVar[Provider]

    def __new__(cls, value: str) -> Self:
        """Create a provider from its value.

        Args:
            value: The provider's name, such as ``"openai"``.

        Returns:
            The provider.

        Raises:
            TypeError: If the value is not a ``str``.
            ValueError: If the value is empty.
        """
        given = cast("object", value)  # not checked at run time

        if not isinstance(given, str):
            message = f"A provider is a str, not {type(given).__name__}"
            raise TypeError(message)

        if not value:
            message = "A provider needs a name"
            raise ValueError(message)

        return super().__new__(cls, value)

    @property
    def value(self) -> str:
        """The provider's name as a plain ``str``.

        Returns:
            The value the provider was created from.
        """
        return str.__str__(self)

    def __repr__(self) -> str:
        """Show the provider as the call that creates it.

        Returns:
            ``Provider('<value>')``.
        """
        return f"{type(self).__name__}({self.value!r})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Let Pydantic validate and dump fields of this type.

        Args:
            source_type: The annotated type.
            handler: Pydantic's schema handler.

        Returns:
            A schema that takes a non-empty string, makes it a
            ``Provider`` and dumps it to JSON as a plain string.
        """
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(min_length=1),
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, when_used="json"
            ),
        )


Provider.GENERIC = Provider("generic")


@unique
class BodyFormat(StrEnum):
    """Encoding of the request body.

    Attributes:
        JSON: ``application/json``; files and binary data are rejected.
        FORM: ``application/x-www-form-urlencoded``; files are rejected.
        MULTIPART: ``multipart/form-data``; the only format for files.
        NONE: The request has no body.
    """

    JSON = "json"
    FORM = "form"
    MULTIPART = "multipart"
    NONE = "none"


@unique
class ResponseFormat(StrEnum):
    """How the response body is parsed.

    Attributes:
        JSON: Validated into the method's response model.
        TEXT: Decoded to ``str`` using the response charset.
        BYTES: Returned as raw ``bytes``.
        NONE: The body must be empty; ``None`` is returned.
        SSE: Server-sent events (``text/event-stream``); streams only.
        NDJSON: Newline-delimited JSON, one value per line; streams only.
    """

    JSON = "json"
    TEXT = "text"
    BYTES = "bytes"
    NONE = "none"
    SSE = "sse"
    NDJSON = "ndjson"


@unique
class Role(StrEnum):
    """Author of a turn of a conversation.

    Attributes:
        USER: The end user.
        ASSISTANT: The model.
    """

    USER = "user"
    ASSISTANT = "assistant"


@unique
class IncompleteReason(StrEnum):
    """Why a generation stopped before its end, in terms every provider has.

    The provider's own reason is kept apart, as
    ``raw_incomplete_reason``.

    Attributes:
        MAX_TOKENS: The output reached the token limit of the request.
        REFUSAL: The model refused to answer.
        PAUSED: The provider paused a long turn of its built-in tools
            (Anthropic's ``pause_turn``); send ``to_turn()`` back to
            continue it.
        OTHER: Any other reason, such as a content filter or a full
            context window.
    """

    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    PAUSED = "paused"
    OTHER = "other"


@unique
class Effort(StrEnum):
    """How much work the model puts into a response, for every provider.

    The levels both providers share: OpenAI's ``reasoning.effort`` and
    Anthropic's ``output_config.effort``. More effort trades speed and
    cost for quality. A model that does not take a level answers 400.

    Attributes:
        LOW: The least work, the fastest and cheapest.
        MEDIUM: A balance of quality, speed and cost.
        HIGH: More work for harder tasks.
        XHIGH: More still, for the hardest tasks.
        MAX: The most work the model can put in.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@unique
class ToolChoice(StrEnum):
    """Whether the model calls a tool, for every provider.

    The modes both providers share; to make it call one tool, give a
    ``ForcedTool`` instead.

    Attributes:
        AUTO: The model decides; the providers' default.
        NONE: The model calls no tool and answers in text.
        REQUIRED: The model calls at least one tool (OpenAI's
            ``required``, Anthropic's ``any``).
    """

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@unique
class ServerToolKind(StrEnum):
    """What a built-in tool call does, for every provider.

    A built-in (server) tool runs at the provider, not in your code.

    Attributes:
        WEB_SEARCH: A search of the web.
        FILE_SEARCH: A search of the provider's stored files.
        CODE_EXECUTION: Code run in the provider's sandbox.
        WEB_FETCH: A fetch of one web page.
        IMAGE_GENERATION: An image made by the model.
        MCP: A call of a tool of a remote MCP server.
        OTHER: Any other built-in tool; ``ServerToolCall.raw`` has it.
    """

    WEB_SEARCH = "web_search"
    FILE_SEARCH = "file_search"
    CODE_EXECUTION = "code_execution"
    WEB_FETCH = "web_fetch"
    IMAGE_GENERATION = "image_generation"
    MCP = "mcp"
    OTHER = "other"


@unique
class ServerToolStatus(StrEnum):
    """How far a built-in tool call got.

    A tool that failed is a call with ``FAILED``, not an exception: the
    response that holds it is valid.

    Attributes:
        IN_PROGRESS: The call has started and has no result yet.
        COMPLETED: The call finished and has its result.
        INCOMPLETE: The call stopped early, such as at a limit.
        FAILED: The call finished with an error of the tool.
    """

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@unique
class ServerToolPhase(StrEnum):
    """Which moment of a built-in tool call a stream event tells.

    Attributes:
        STARTED: The call has begun.
        DONE: The call has its result, or has failed.
    """

    STARTED = "started"
    DONE = "done"


@unique
class CitationKind(StrEnum):
    """What a citation points to.

    Attributes:
        URL: A page of the web.
        FILE: A file the provider stores.
        CONTAINER_FILE: A file a code execution made in its container.
    """

    URL = "url"
    FILE = "file"
    CONTAINER_FILE = "container_file"


@unique
class GenerationEventType(StrEnum):
    """``type`` of a shared generation event.

    Attributes:
        TEXT_DELTA: A piece of generated text, ``TextDeltaEvent``.
        REASONING_DELTA: A piece of a reasoning summary,
            ``ReasoningDeltaEvent``.
        TOOL_CALL_START: A tool call begins, ``ToolCallStartEvent``.
        TOOL_CALL_DELTA: A piece of a call's arguments,
            ``ToolCallDeltaEvent``.
        TOOL_CALL_DONE: A tool call is complete, ``ToolCallDoneEvent``.
        SERVER_TOOL_CALL: A built-in tool call starts or ends,
            ``ServerToolCallEvent``.
        COMPLETED: The end of the generation, ``CompletedEvent``.
        OTHER: A provider event without a shared counterpart,
            ``OtherEvent``.
    """

    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_DONE = "tool_call_done"
    SERVER_TOOL_CALL = "server_tool_call"
    COMPLETED = "completed"
    OTHER = "other"
