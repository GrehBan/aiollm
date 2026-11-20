"""Test doubles that need no server: an API, a transport and methods.

``FakeTransport`` replays queued responses through the real
``BaseSession``, so retries, error mapping and parsing run as in
production while nothing is sent.
"""

from __future__ import annotations

import datetime
from collections import deque
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from aiollm.http.auth import BearerAuth
from aiollm.http.exceptions.resolver import BaseErrorResolver
from aiollm.session.base import BaseSession

if TYPE_CHECKING:
    from aiollm.http.exceptions.protocols import IErrorResolver
    from aiollm.http.protocols import IAPI, HeadersType, IAuth, Query
    from aiollm.methods.base import BaseMethod
    from aiollm.methods.protocols import IAPIMethod, IStreamMethod
    from aiollm.response.buffered import BufferedResponse
    from aiollm.session.protocols import Timeout

TEST_TOKEN: Final = "test-token"
"""The token the test sessions authenticate with."""

NOW: Final = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC)
"""The current time while the ``frozen_now`` fixture is active."""


@dataclass
class LocalAPI:
    """``IAPI`` that joins a base URL, a version and a method path.

    Attributes:
        base_url: The server's URL, such as the local test server's.
        version: The path segment before every method path.
    """

    base_url: URL
    version: str = "v1"

    def build_url(
        self,
        method: str,
        query: Query | None = None,
    ) -> str:
        """Build the URL of a request.

        Args:
            method: The method's path, already percent-encoded.
            query: The query parameters, if any.

        Returns:
            The full URL.
        """
        # `method` is already percent-encoded: do not encode it again.
        url = self.base_url.joinpath(
            self.version,
            method.lstrip("/"),
            encoded=True,
        )
        return str(url.with_query(query or {}))


class FakeStreamResponse:
    """``IStreamResponse`` that yields a fixed list of chunks.

    Args:
        chunks: The body, chunk by chunk.
        status: HTTP status code.
        headers: Response headers.
        charset: Charset from the ``Content-Type`` header.
        fail_at: Index of the chunk at which ``error`` is raised instead
            of yielding it; ``None`` never fails.
        error: The exception raised at ``fail_at``.

    Attributes:
        chunks: The body, chunk by chunk.
        fail_at: Index of the chunk that fails, if any.
        error: The exception raised there.
        aclose_calls: How many times ``aclose()`` was called.
        read_calls: How many times ``read()`` was called.

    Raises:
        ValueError: If only one of ``fail_at`` and ``error`` is given.
    """

    def __init__(
        self,
        chunks: Iterable[bytes] = (),
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        charset: str | None = None,
        fail_at: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Keep the body and the failure to replay."""
        if (fail_at is None) != (error is None):
            message = "fail_at and error go together"
            raise ValueError(message)

        self.chunks = list(chunks)
        self._status = status
        self._headers = CIMultiDictProxy(CIMultiDict(headers or {}))
        self._charset = charset
        self.fail_at = fail_at
        self.error = error
        self.aclose_calls = 0
        self.read_calls = 0

    @property
    def status(self) -> int:
        """HTTP status code."""
        return self._status

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers, case-insensitive."""
        return self._headers

    @property
    def charset(self) -> str | None:
        """Charset from the ``Content-Type`` header."""
        return self._charset

    @property
    def closed(self) -> bool:
        """Whether ``aclose()`` has been called at least once."""
        return self.aclose_calls > 0

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        """Yield the chunks, raising ``error`` at ``fail_at``.

        Yields:
            Each chunk of the body.

        Raises:
            BaseException: ``error``, at the chunk ``fail_at``.
        """
        for index, chunk in enumerate(self.chunks):
            if index == self.fail_at and self.error is not None:
                raise self.error
            yield chunk

    async def read(self) -> bytes:
        """Read the whole body.

        Returns:
            Every chunk, joined.
        """
        self.read_calls += 1
        return b"".join(self.chunks)

    async def aclose(self) -> None:
        """Count a close; closing again is allowed."""
        self.aclose_calls += 1


type Outcome[R] = R | BaseException
"""A queued transport result: returned as is, or raised if an exception."""


class FakeTransport(BaseSession):
    """``BaseSession`` whose transport replays queued outcomes.

    ``_execute`` pops from ``responses`` and ``_open_stream`` from
    ``stream_responses``; an exception in the queue is raised instead of
    returned, and an empty queue raises ``AssertionError``. Every call is
    recorded, so tests can assert exact attempt counts and which stream
    responses were closed, and the headers and timeout of each attempt
    go to ``call_options``.

    Args:
        responses: Outcomes of ``_execute``, in order.
        stream_responses: Outcomes of ``_open_stream``, in order.
        api: URL builder; never used by the fake transport itself.
        auth: Authorization header source; a bearer ``TEST_TOKEN`` if
            not given.
        error_resolver: Turns error responses into exceptions; the base
            resolver if not given.
        **kwargs: Forwarded to ``BaseSession`` (headers, JSON codecs,
            retry policy).

    Attributes:
        responses: Outcomes ``_execute`` has not replayed yet.
        stream_responses: Outcomes ``_open_stream`` has not replayed yet.
        execute_calls: Every method ``_execute`` was called with.
        stream_calls: Every method ``_open_stream`` was called with.
        opened: Every stream response ``_open_stream`` returned.
        call_options: The headers and timeout of each attempt.
        close_calls: How many times ``close()`` was called.
    """

    def __init__(
        self,
        responses: Iterable[Outcome[BufferedResponse]] = (),
        stream_responses: Iterable[Outcome[FakeStreamResponse]] = (),
        *,
        api: IAPI | None = None,
        auth: IAuth | None = None,
        error_resolver: IErrorResolver | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the session and queue the outcomes."""
        super().__init__(
            api or LocalAPI(URL("http://fake.invalid")),
            auth or BearerAuth(TEST_TOKEN),
            error_resolver or BaseErrorResolver(),
            **kwargs,
        )
        self.responses: deque[Outcome[BufferedResponse]] = deque(responses)
        self.stream_responses: deque[Outcome[FakeStreamResponse]] = deque(
            stream_responses
        )
        self.execute_calls: list[IAPIMethod[Any]] = []
        self.stream_calls: list[IStreamMethod[Any]] = []
        self.opened: list[FakeStreamResponse] = []
        self.call_options: list[tuple[HeadersType | None, Timeout | None]] = []
        self.close_calls = 0

    @property
    def attempts(self) -> int:
        """Number of ``_execute`` and ``_open_stream`` calls."""
        return len(self.execute_calls) + len(self.stream_calls)

    @property
    def closed_responses(self) -> int:
        """Number of opened stream responses that were closed."""
        return sum(response.closed for response in self.opened)

    async def close(self) -> None:
        """Count a close; nothing is open."""
        self.close_calls += 1

    async def _execute(
        self,
        method: IAPIMethod[Any],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> BufferedResponse:
        """Record the attempt and replay the next buffered outcome.

        Args:
            method: The method sent.
            headers: The call's headers.
            timeout: The call's timeout.

        Returns:
            The next queued response.
        """
        self.execute_calls.append(method)
        self.call_options.append((headers, timeout))
        return self._next(self.responses, "_execute")

    async def _open_stream(
        self,
        method: IStreamMethod[Any],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> FakeStreamResponse:
        """Record the attempt and replay the next stream outcome.

        Args:
            method: The method sent.
            headers: The call's headers.
            timeout: The call's timeout.

        Returns:
            The next queued stream response.
        """
        self.stream_calls.append(method)
        self.call_options.append((headers, timeout))
        response = self._next(self.stream_responses, "_open_stream")
        self.opened.append(response)
        return response

    @staticmethod
    def _next[R](queue: deque[Outcome[R]], name: str) -> R:
        """Take the next outcome of a queue.

        Args:
            queue: The queue.
            name: The transport method, for the error message.

        Returns:
            The outcome, if it is not an exception.

        Raises:
            AssertionError: If the queue is empty.
            BaseException: The outcome, if it is one.
        """
        if not queue:
            message = f"FakeTransport.{name}: queue is empty"
            raise AssertionError(message)

        outcome = queue.popleft()

        if isinstance(outcome, BaseException):
            raise outcome

        return outcome


def make_method[M: BaseMethod[Any]](
    base: type[M],
    *,
    path: str | None = "/test",
    fields: Mapping[str, Any] | None = None,
    name: str = "Method",
    http_method: str = "POST",
    **class_vars: Any,
) -> type[M]:
    """Declare a method class with ``type(...)``.

    Meant for parametrized class-definition checks; wrap the call in
    ``pytest.raises`` when the definition must fail. Type checkers only
    see ``base``, so create instances with dynamic fields through
    ``model_validate({...})``.

    Args:
        base: Base class, e.g. ``BaseAPIMethod[str]``.
        path: ``__request_path__``; ``None`` inherits it from ``base``.
        fields: Field annotations; a ``(type, default)`` tuple also sets
            the default.
        name: Class name, shown in error messages.
        http_method: ``__request_method__``.
        **class_vars: Other class attributes, e.g.
            ``__request_query__={"q"}``.

    Returns:
        The new method class.

    Example:
        >>> GetItem = make_method(
        ...     BaseAPIMethod[str],
        ...     path="/items/{item_id}",
        ...     fields={"item_id": str},
        ...     http_method="GET",
        ... )
        >>> GetItem.model_validate({"item_id": "a/b"}).request_path
        '/items/a%2Fb'
    """
    annotations: dict[str, Any] = {}
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": name,
        "__request_method__": http_method,
        **class_vars,
    }

    if path is not None:
        namespace["__request_path__"] = path

    for field_name, spec in (fields or {}).items():
        if isinstance(spec, tuple):
            annotations[field_name], namespace[field_name] = spec
        else:
            annotations[field_name] = spec

    namespace["__annotations__"] = annotations
    return cast("type[M]", type(name, (base,), namespace))
