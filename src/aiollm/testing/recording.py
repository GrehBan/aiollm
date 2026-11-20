"""What the local test server receives and what it answers."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multidict import CIMultiDictProxy, MultiDict


@dataclass
class RecordedPart:
    """One multipart part as the server received it.

    Attributes:
        name: The part's ``name`` in ``Content-Disposition``.
        filename: The part's ``filename``, if it has one.
        content_type: The part's ``Content-Type``, if it has one.
        data: The part's body.
    """

    name: str | None
    filename: str | None
    content_type: str | None
    data: bytes


@dataclass
class RecordedRequest:
    """A request as the server received it.

    Attributes:
        method: The HTTP method.
        raw_path: The path exactly as sent, still percent-encoded and
            without the query string.
        query_string: The query string as sent.
        query: The query parameters, decoded.
        headers: The request headers.
        body: The body; empty for a multipart body, which is split into
            ``parts``.
        parts: The parts of a multipart body.
    """

    method: str
    raw_path: str
    query_string: str
    query: MultiDict[str]
    headers: CIMultiDictProxy[str]
    body: bytes
    parts: list[RecordedPart]


type Chunk = bytes | asyncio.Event
"""A streamed chunk, or an event the server waits on before going on."""


@dataclass
class Reply:
    """What the server answers to one request.

    Attributes:
        status: HTTP status code.
        body: Whole body, sent with ``Content-Length``.
        json: Body serialized as JSON; overrides ``body``.
        headers: Extra response headers.
        content_type: ``Content-Type``, charset included.
        chunks: Streamed body (chunked encoding); an ``asyncio.Event``
            item makes the server wait for it. May be infinite.
        delay: Seconds to wait before answering.
        chunk_delay: Seconds to wait before each chunk.
        content_length: Declared ``Content-Length`` for ``chunks``; with
            ``abort`` it makes the body end before it is complete.
        abort: Drop the connection after the chunks are sent.
    """

    status: int = 200
    body: bytes = b""
    json: Any = None
    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    content_type: str | None = None
    chunks: Iterable[Chunk] | None = None
    delay: float = 0.0
    chunk_delay: float = 0.0
    content_length: int | None = None
    abort: bool = False
