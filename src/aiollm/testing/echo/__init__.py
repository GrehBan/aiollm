"""The echo provider: a provider package made for tests.

Its API exists only on the local server a test scripts, so it runs every
shared scenario of ``aiollm.testing.contract`` without a real provider,
and it takes nothing from the core outside the extension API: it is the
smallest example of a provider package. Its client binds the aiohttp
transport, as a provider package's does.
"""

from __future__ import annotations

from aiollm.testing.echo.api import ECHO_API, PROVIDER, EchoAPI
from aiollm.testing.echo.client import EchoClient, session_factory
from aiollm.testing.echo.files import EchoFilesAdapter
from aiollm.testing.echo.generation import EchoGenerationAdapter
from aiollm.testing.echo.resolver import EchoErrorResolver

__all__ = [
    "ECHO_API",
    "EchoAPI",
    "EchoClient",
    "EchoErrorResolver",
    "EchoFilesAdapter",
    "EchoGenerationAdapter",
    "PROVIDER",
    "session_factory",
]
