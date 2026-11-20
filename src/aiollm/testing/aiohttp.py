"""The local test server, an aiohttp application that follows a script.

Transport-specific, like every ``aiohttp.py`` module of the library, so
``aiollm.testing`` imports without it.
"""

from __future__ import annotations

import asyncio
import email.policy
from collections import deque
from email.message import EmailMessage
from email.parser import BytesParser

from aiohttp import web
from multidict import MultiDict

from aiollm.http.headers import CONTENT_TYPE
from aiollm.testing.recording import RecordedPart, RecordedRequest, Reply


def _parse_multipart(content_type: str, body: bytes) -> list[RecordedPart]:
    """Split a multipart body into its parts.

    Args:
        content_type: ``Content-Type`` of the request, with the boundary.
        body: The whole request body.

    Returns:
        The parts, in order.
    """
    message = BytesParser(policy=email.policy.HTTP).parsebytes(
        f"{CONTENT_TYPE}: {content_type}\r\n\r\n".encode() + body
    )
    assert isinstance(message, EmailMessage)
    assert message.is_multipart()

    parts: list[RecordedPart] = []

    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True)
        assert isinstance(name, str | None)
        assert isinstance(data, bytes)
        parts.append(
            RecordedPart(
                name=name,
                filename=part.get_filename(),
                content_type=part.get(CONTENT_TYPE),
                data=data,
            )
        )

    return parts


class Recorder:
    """aiohttp app that records requests and answers from a script.

    Replies are taken from ``replies`` in order; once the script is
    exhausted, ``default`` answers, if a test set one. Without it, the
    request is unscripted: it gets a 500 and the ``recorder`` fixture
    fails the test, so a retry or a second request nobody planned for
    cannot pass silently. Every path and HTTP method is routed here.

    Attributes:
        requests: Every request received, in order.
        replies: The replies not given yet.
        default: The reply once ``replies`` is empty, if any.
        unscripted: The requests no reply was scripted for.
        app: The aiohttp application to serve.
    """

    def __init__(self) -> None:
        """Start with an empty script and no default reply."""
        self.requests: list[RecordedRequest] = []
        self.replies: deque[Reply] = deque()
        self.default: Reply | None = None
        self.unscripted: list[RecordedRequest] = []
        self.app = web.Application()
        self.app.router.add_route("*", "/{tail:.*}", self._handle)

    def reply(self, *replies: Reply) -> None:
        """Queue replies for the next requests.

        Args:
            *replies: The replies, in the order of the requests.
        """
        self.replies.extend(replies)

    @property
    def last(self) -> RecordedRequest:
        """The most recent request.

        Raises:
            AssertionError: If no request was received.
        """
        assert self.requests, "no request was received"
        return self.requests[-1]

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        """Record a request and answer it from the script.

        Args:
            request: The request.

        Returns:
            The scripted response, or a 500 for an unscripted request.
        """
        recorded = await self._record(request)
        self.requests.append(recorded)
        reply = self.replies.popleft() if self.replies else self.default

        if reply is None:
            self.unscripted.append(recorded)
            return web.json_response(
                {"error": {"message": "unscripted request"}}, status=500
            )

        if reply.delay:
            await asyncio.sleep(reply.delay)

        if reply.chunks is None:
            return self._buffered(reply)

        return await self._streamed(request, reply)

    @staticmethod
    async def _record(request: web.Request) -> RecordedRequest:
        """Read a request as it came.

        Args:
            request: The request.

        Returns:
            Its method, path, query, headers and body or parts.
        """
        parts: list[RecordedPart] = []
        body = b""

        if request.content_type.startswith("multipart/"):
            # Parsed from the whole body: aiohttp's streaming multipart
            # reader calls the deprecated ``unread_data`` for parts without
            # a length, which fails the test under ``-W error`` whenever a
            # boundary falls inside a received chunk.
            parts = _parse_multipart(
                request.headers[CONTENT_TYPE],
                await request.read(),
            )
        else:
            body = await request.read()

        return RecordedRequest(
            method=request.method,
            raw_path=request.raw_path.partition("?")[0],
            query_string=request.query_string,
            query=MultiDict(request.query),
            headers=request.headers,
            body=body,
            parts=parts,
        )

    @staticmethod
    def _buffered(reply: Reply) -> web.Response:
        """Build a response with the whole body.

        Args:
            reply: The scripted reply.

        Returns:
            The response.
        """
        if reply.json is not None:
            return web.json_response(
                reply.json,
                status=reply.status,
                headers=reply.headers,
            )

        response = web.Response(
            status=reply.status,
            body=reply.body,
            headers=reply.headers,
        )

        if reply.content_type is not None:
            response.headers[CONTENT_TYPE] = reply.content_type

        return response

    @staticmethod
    async def _streamed(
        request: web.Request,
        reply: Reply,
    ) -> web.StreamResponse:
        """Stream a body chunk by chunk.

        Args:
            request: The request answered.
            reply: The scripted reply, with ``chunks``.

        Returns:
            The response, written; its connection is dropped for
            ``abort``.
        """
        assert reply.chunks is not None
        response = web.StreamResponse(
            status=reply.status,
            headers=reply.headers,
        )

        if reply.content_type is not None:
            response.headers[CONTENT_TYPE] = reply.content_type

        if reply.content_length is not None:
            response.content_length = reply.content_length
        else:
            response.enable_chunked_encoding()

        await response.prepare(request)

        for chunk in reply.chunks:
            if isinstance(chunk, asyncio.Event):
                await chunk.wait()
                continue

            if reply.chunk_delay:
                await asyncio.sleep(reply.chunk_delay)

            await response.write(chunk)

        if reply.abort:
            assert request.transport is not None
            request.transport.close()
            return response

        await response.write_eof()
        return response
