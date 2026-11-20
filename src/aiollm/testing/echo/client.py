"""Client of the echo API."""

from __future__ import annotations

from typing import Final

from aiollm.client.base import BaseProviderClient
from aiollm.client.binding import bind_resource
from aiollm.http.auth import BearerAuth
from aiollm.http.headers import AUTHORIZATION
from aiollm.resources.files import FilesResource
from aiollm.resources.generation import GenerationResource
from aiollm.session.aiohttp import AiohttpSession
from aiollm.session.factory import SessionFactory
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
