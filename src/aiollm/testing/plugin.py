"""pytest fixtures for testing code built on aiollm.

Load the plugin from a ``conftest.py`` with
``pytest_plugins = ["aiollm.testing.plugin"]``; it is not registered as
an entry point, since its autouse fixtures change how the library
behaves in every test. It needs the ``testing`` extra (pytest,
pytest-asyncio and pytest-aiohttp).

Autouse: ``no_retry_wait`` (retries do not wait), ``no_provider_keys``
(the variables ``provider_key_envs`` names are cleared) and
``no_close_delay`` (``AiohttpSession.close()`` does not pause). On
request: ``frozen_now``, ``fake_transport``, and the local server
``recorder``, ``server``, ``api`` and ``session``; ``recorder`` fails
the test at teardown if a request came that no reply was scripted for.
"""

from __future__ import annotations

import datetime
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, cast
from unittest import mock

import pytest
import pytest_asyncio

from aiollm.http import headers as http_headers
from aiollm.http.auth import BearerAuth
from aiollm.http.exceptions.resolver import BaseErrorResolver
from aiollm.session import aiohttp as session_aiohttp
from aiollm.session.aiohttp import AiohttpSession
from aiollm.session.retry.runner import calculate_wait
from aiollm.testing.aiohttp import Recorder
from aiollm.testing.doubles import NOW, TEST_TOKEN, FakeTransport, LocalAPI

if TYPE_CHECKING:
    from aiohttp.test_utils import TestServer
    from pytest_aiohttp import AiohttpServer
    from tenacity import RetryCallState

    from aiollm.session.retry.policy import RetryPolicy


@pytest.fixture(autouse=True)
def no_retry_wait() -> Iterator[mock.MagicMock]:
    """Make retries instant.

    Replaces ``calculate_wait`` where the retry loop looks it up, so it
    waits 0 seconds and the policy's ``backoff`` is not called. Tenacity
    reuses one mutable state object, so the mock's ``call_args`` only
    show the final state; the error each wait was computed for is kept
    in ``mock.outcomes`` instead. A test that needs the real wait sets
    the mock's ``side_effect`` to ``calculate_wait`` imported from
    ``aiollm.session.retry.runner`` at the top of its module, before the
    patch.

    Yields:
        The mock, with ``outcomes``.
    """
    outcomes: list[BaseException | None] = []

    def wait(state: RetryCallState, policy: RetryPolicy) -> float:
        """Record the failed attempt's error and wait 0 seconds.

        Args:
            state: Tenacity's state of the call.
            policy: The call's retry policy, unused.

        Returns:
            ``0.0``.
        """
        outcomes.append(
            None if state.outcome is None else state.outcome.exception()
        )
        return 0.0

    # The spec is the function itself, taken when the plugin loads, so a
    # suite run inside another (pytester) can patch it again.
    with mock.patch(
        "aiollm.session.retry.runner.calculate_wait",
        autospec=calculate_wait,
        side_effect=wait,
    ) as patched:
        patched.outcomes = outcomes
        yield cast("mock.MagicMock", patched)


@pytest.fixture
def provider_key_envs() -> tuple[str, ...]:
    """Environment variables ``no_provider_keys`` clears.

    Override it in a ``conftest.py`` with the variables a provider
    client reads its key from, such as ``("EXAMPLE_API_KEY",)``.

    Returns:
        No variable.
    """
    return ()


@pytest.fixture(autouse=True)
def no_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
    provider_key_envs: tuple[str, ...],
) -> None:
    """Hide API keys of the environment, so clients see only test tokens.

    Args:
        monkeypatch: pytest's monkeypatch.
        provider_key_envs: The variables to clear.
    """
    for name in provider_key_envs:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_close_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the 0.25 s pause of ``AiohttpSession.close()``.

    Only the delay constant is replaced: the module and its ``asyncio``
    stay real, so everything else in it behaves as in production.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    monkeypatch.setattr(session_aiohttp, "_CLOSE_DELAY", 0)


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> datetime.datetime:
    """Fix the current time at ``NOW`` for the library's clock reads.

    Replaces ``http.headers._now``, the only place the library reads the
    clock (``Retry-After`` dates), and nothing else.

    Args:
        monkeypatch: pytest's monkeypatch.

    Returns:
        ``NOW``.
    """
    monkeypatch.setattr(http_headers, "_now", lambda: NOW)
    return NOW


@pytest.fixture
def fake_transport() -> FakeTransport:
    """A ``FakeTransport`` with empty queues; fill them in the test.

    Returns:
        The transport.
    """
    return FakeTransport()


@pytest.fixture
def recorder() -> Iterator[Recorder]:
    """A fresh request recorder; ``api`` serves it.

    Yields:
        The recorder.

    Raises:
        AssertionError: At teardown, if a request came that no reply was
            scripted for.
    """
    recorder = Recorder()
    yield recorder
    unscripted = [
        f"{request.method} {request.raw_path}"
        for request in recorder.unscripted
    ]
    assert not unscripted, f"unscripted requests: {unscripted}"


@pytest_asyncio.fixture
async def server(
    aiohttp_server: AiohttpServer,
    recorder: Recorder,
) -> TestServer:
    """The recorder running on a local port.

    Args:
        aiohttp_server: pytest-aiohttp's server factory.
        recorder: The recorder to serve.

    Returns:
        The running server.
    """
    return await aiohttp_server(recorder.app)


@pytest.fixture
def api(server: TestServer) -> LocalAPI:
    """``IAPI`` pointing at the local server.

    Args:
        server: The running server.

    Returns:
        The API.
    """
    return LocalAPI(server.make_url("/"))


@pytest_asyncio.fixture
async def session(api: LocalAPI) -> AsyncIterator[AiohttpSession]:
    """``AiohttpSession`` against the local server, closed on teardown.

    Args:
        api: The local server's API.

    Yields:
        The session.
    """
    session = AiohttpSession(
        api,
        BearerAuth(TEST_TOKEN),
        BaseErrorResolver(),
    )

    try:
        yield session
    finally:
        await session.close()
