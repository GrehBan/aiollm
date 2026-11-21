"""Built-in tool calls and citations of a generation."""

from __future__ import annotations

from pydantic import Field

from aiollm.enums import CitationKind, ServerToolKind, ServerToolStatus
from aiollm.types.base import BaseAPIType
from aiollm.types.requests.schema import JSONValue


class ServerToolCall(BaseAPIType):
    """A call of a built-in tool, which the provider ran itself.

    Unlike a ``ToolCall``, it needs no result from you: the provider
    made the call and holds its result, which stays in ``raw``. A call
    that failed is a call with ``status`` ``FAILED``, not an error of
    the response. The calls are what the provider bills, so count them
    to know a request's cost.

    Attributes:
        id: The call's id at the provider.
        kind: What the call does.
        status: How far it got.
        queries: What a search looked for; empty for other calls.
        code: The code a code execution ran, if the provider tells it.
        error: What went wrong when ``status`` is ``FAILED``, if the
            provider tells it.
        raw: The provider's items of the call, in order: OpenAI's one
            output item, Anthropic's call block and its result block.
            The results (found pages, the output of code) are here.

    Example:
        >>> for call in response.server_tool_calls:
        ...     print(call.kind.value, call.status.value, call.queries)
    """

    id: str
    kind: ServerToolKind
    status: ServerToolStatus
    queries: list[str] = Field(default_factory=list[str])
    code: str | None = None
    error: str | None = None
    raw: list[dict[str, JSONValue]] = Field(
        default_factory=list[dict[str, JSONValue]]
    )


class Citation(BaseAPIType):
    """A source a part of the text rests on.

    What both providers give: where it points and which words of the
    text it backs. Each provider's own fields (Anthropic's
    ``encrypted_index``) stay in ``GenerationResponse.raw`` and
    ``native``.

    Attributes:
        kind: What it points to.
        url: The page, for ``URL``.
        title: The title of the page or file, if known.
        file_id: The file's id at the provider, for ``FILE`` and
            ``CONTAINER_FILE``.
        container_id: The container's id, for ``CONTAINER_FILE``.
        filename: The file's name, for a file.
        cited_text: The words of the source that were cited, if the
            provider tells them.
        start_index: Where the cited part of the response ``text``
            starts, if the provider tells it.
        end_index: Where it ends, if the provider tells it.

    Example:
        >>> [c.url for c in response.citations if c.url]
    """

    kind: CitationKind
    url: str | None = None
    title: str | None = None
    file_id: str | None = None
    container_id: str | None = None
    filename: str | None = None
    cited_text: str | None = None
    start_index: int | None = None
    end_index: int | None = None
