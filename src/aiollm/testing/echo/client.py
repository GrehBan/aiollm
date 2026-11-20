"""Client of the echo API."""

from __future__ import annotations

from typing import Final, Unpack

from aiollm.client.base import BaseProviderClient
from aiollm.client.binding import bind_resource
from aiollm.http.auth import BearerAuth
from aiollm.http.headers import AUTHORIZATION
from aiollm.resources.files import FilesResource
from aiollm.resources.generation import GenerationResource
from aiollm.session.aiohttp import AiohttpSession, AiohttpSessionOptions
from aiollm.session.factory import SessionFactory
from aiollm.session.protocols import ISession
from aiollm.testing.echo.api import ECHO_API, PROVIDER
from aiollm.testing.echo.files import EchoFilesAdapter
from aiollm.testing.echo.generation import EchoGenerationAdapter
from aiollm.testing.echo.resolver import EchoErrorResolver
from aiollm.testing.echo.types import (
    EchoEvent,
    EchoFile,
    EchoFileOptions,
    EchoFilePage,
    EchoGenerationOptions,
    EchoResponse,
)

session_factory: Final = SessionFactory(
    AiohttpSession,
    auth_factory=BearerAuth,
    api=ECHO_API,
    error_resolver=EchoErrorResolver(),
)
"""Builds the ``AiohttpSession`` of an ``EchoClient`` from a token."""

ECHO_API_KEY_ENV: Final = "ECHO_API_KEY"
"""Environment variable with the key, read without a token or session."""


class EchoClient(BaseProviderClient):
    """Client of the echo API, a provider made for tests.

    The echo API exists only on a local server that a test scripts
    (``aiollm.testing.aiohttp.Recorder``): it is the smallest provider
    that every shared scenario runs on, built from the extension API
    alone, and an example of a provider package.

    Attributes:
        generation: Shared text generation over ``POST /echo``.
        files: Shared file storage over ``/files``.

    Example:
        >>> monkeypatch.setattr(session_factory, "api", api)
        >>> async with EchoClient(TEST_TOKEN) as client:
        ...     response = await client.generation.create(
        ...         model="echo-1", input="Hi"
        ...     )
    """

    provider = PROVIDER
    session_factory = session_factory
    api_key_env = ECHO_API_KEY_ENV
    auth_headers = frozenset({AUTHORIZATION})

    generation = bind_resource(
        GenerationResource[EchoResponse, EchoEvent, EchoGenerationOptions],
        EchoGenerationAdapter,
    )

    files = bind_resource(
        FilesResource[EchoFile, EchoFilePage, EchoFileOptions],
        EchoFilesAdapter,
    )

    def __init__(
        self,
        token: str | None = None,
        *,
        session: ISession | None = None,
        **session_kwargs: Unpack[AiohttpSessionOptions],
    ) -> None:
        """Create the client from a token or a ready session.

        A provider client declares the arguments of its session, so a
        misspelled one is a type error.

        Args:
            token: API key; an ``AiohttpSession`` with the echo API's
                URL, error resolver and ``BearerAuth`` is created from
                it. Without a token and a session, ``ECHO_API_KEY`` is
                used.
            session: A ready session, for custom settings; it is checked
                with ``warn_about_session``.
            **session_kwargs: Further arguments of the
                ``AiohttpSession`` built from the token, such as
                ``timeout`` or ``max_body_size``.

        Raises:
            ValueError: If both ``token`` and ``session`` are given, or
                neither is and ``ECHO_API_KEY`` is unset or empty, or
                ``session_kwargs`` are given with a ``session``.
        """
        super().__init__(token, session=session, **session_kwargs)
