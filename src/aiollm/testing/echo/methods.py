"""Methods of the echo API: generation and files."""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal

from pydantic import MISSING, Field

from aiollm.enums import BodyFormat, ResponseFormat
from aiollm.methods.base import BaseAPIMethod, BaseStreamMethod
from aiollm.testing.echo.types import (
    EchoDeleted,
    EchoEvent,
    EchoFile,
    EchoFilePage,
    EchoResponse,
)
from aiollm.types.base import BaseAPIType
from aiollm.types.files import InputFile
from aiollm.types.matchers import EventTypes

TERMINAL_EVENTS: Final = EventTypes("done")
"""The event that ends a complete stream."""

MIN_EXPIRY: Final = 3600
"""The shortest ``expires_in_seconds`` an upload takes: an hour."""

MAX_EXPIRY: Final = 2_592_000
"""The longest ``expires_in_seconds`` an upload takes: thirty days."""


class GenerateParams(BaseAPIType):
    """Fields of ``POST /echo``, buffered or streamed.

    Every field but ``model`` and ``input`` is left out until it is
    given.

    Attributes:
        model: Model identifier.
        input: The conversation: messages, turns of calls and results.
        instructions: A system prompt.
        max_output_tokens: Upper bound on generated tokens.
        end_user_id: An opaque id of the end user.
        effort: How much work the model puts in, such as ``high``.
        output_schema: A JSON Schema the text must follow.
        tools: The tools the model may call.
        tool_choice: ``auto``, ``none``, ``required`` or
            ``{"name": ...}`` of one tool.
        parallel_tool_calls: Whether several calls may come at once.
        temperature: Sampling temperature, 0 to 2.
        stop: Texts that end the generation where they appear.
    """

    model: Annotated[str, Field(min_length=1)]
    input: Annotated[list[dict[str, Any]], Field(min_length=1)]
    instructions: str | MISSING = MISSING
    max_output_tokens: Annotated[int, Field(gt=0)] | MISSING = MISSING
    end_user_id: Annotated[str, Field(max_length=64)] | MISSING = MISSING
    effort: str | MISSING = MISSING
    output_schema: dict[str, Any] | MISSING = MISSING
    tools: list[dict[str, Any]] | MISSING = MISSING
    tool_choice: str | dict[str, Any] | MISSING = MISSING
    parallel_tool_calls: bool | MISSING = MISSING
    temperature: Annotated[float, Field(ge=0, le=2)] | MISSING = MISSING
    stop: list[str] | MISSING = MISSING


class Generate(GenerateParams, BaseAPIMethod[EchoResponse]):
    """Generate a response and wait for the whole of it.

    A transient failure is retried.

    Example:
        >>> response = await client.send_request(
        ...     Generate(model="echo-1", input=[{"role": "user", ...}])
        ... )
    """

    __request_path__ = "/echo"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __retryable__ = True


class StreamGenerate(GenerateParams, BaseStreamMethod[EchoEvent]):
    """Generate a response and stream its events.

    The stream is ``start``, text deltas and tool calls (each a
    ``tool_call_start``, ``tool_call_delta`` pieces and a
    ``tool_call_done``), and ``done``; an ``error`` event is raised as
    an ``APIStatusError``. Only opening the stream is retried, and a
    body that ends before ``done`` raises ``APIStreamInterruptedError``.

    Attributes:
        stream: Always ``True``: it asks for server-sent events.

    Example:
        >>> method = StreamGenerate(model="echo-1", input=[...])
        >>> async with client.stream(method) as events:
        ...     async for event in events:
        ...         ...
    """

    stream: Literal[True] = True

    __request_path__ = "/echo"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __retryable__ = True
    __response_format__ = ResponseFormat.SSE
    __stream_terminal__ = TERMINAL_EVENTS


class UploadFile(BaseAPIMethod[EchoFile]):
    """Store a file; never retried, since it is not idempotent.

    Attributes:
        file: The file.
        purpose: What the file is for.
        expires_in_seconds: Seconds until the file expires, an hour to
            thirty days; kept without one.

    Example:
        >>> stored = await client.send_request(UploadFile(file=path))
    """

    file: InputFile
    purpose: str | MISSING = MISSING
    expires_in_seconds: (
        Annotated[int, Field(ge=MIN_EXPIRY, le=MAX_EXPIRY)] | MISSING
    ) = MISSING

    __request_path__ = "/files"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART
    __retryable__ = False


class ListFiles(BaseAPIMethod[EchoFilePage]):
    """Read one page of the stored files, newest first.

    Attributes:
        limit: Files in the page.
        after: ``next`` of the page before; the first page without it.

    Example:
        >>> page = await client.send_request(ListFiles(limit=100))
    """

    limit: Annotated[int, Field(ge=1, le=1000)]
    after: str | MISSING = MISSING

    __request_path__ = "/files"
    __request_method__ = "GET"
    __request_query__ = frozenset({"limit", "after"})
    __retryable__ = True


class RetrieveFile(BaseAPIMethod[EchoFile]):
    """Read a file's metadata.

    Attributes:
        file_id: The file's id.

    Example:
        >>> stored = await client.send_request(RetrieveFile(file_id="f1"))
    """

    file_id: str

    __request_path__ = "/files/{file_id}"
    __request_method__ = "GET"
    __retryable__ = True


class DeleteFile(BaseAPIMethod[EchoDeleted]):
    """Delete a file.

    Attributes:
        file_id: The file's id.

    Example:
        >>> await client.send_request(DeleteFile(file_id="f1"))
    """

    file_id: str

    __request_path__ = "/files/{file_id}"
    __request_method__ = "DELETE"
    __retryable__ = True


class DownloadFile(BaseStreamMethod[bytes]):
    """Stream a file's content.

    Attributes:
        file_id: The file's id.

    Example:
        >>> async with client.stream(DownloadFile(file_id="f1")) as content:
        ...     await content.save(path)
    """

    file_id: str

    __request_path__ = "/files/{file_id}/content"
    __request_method__ = "GET"
    __response_format__ = ResponseFormat.BYTES
    __retryable__ = True
