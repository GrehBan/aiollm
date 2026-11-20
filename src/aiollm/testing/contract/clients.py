"""Scenarios every provider client passes.

A provider's tests subclass ``ClientContract`` with a ``ClientCase`` of
its client: how it is built from a token, a key in the environment or a
given session, and the warnings about a session that does not fit it.

Example:
    >>> from aiollm.testing.contract import ClientCase, ClientContract
    >>> class TestExampleClient(ClientContract):
    ...     case = ClientCase(client=ExampleClient, provider=PROVIDER, ...)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final

import pytest

import aiollm
from aiollm.client.base import BaseClient, BaseProviderClient
from aiollm.enums import Provider
from aiollm.exceptions import SessionMismatchWarning
from aiollm.http.auth import APIKeyAuth, BearerAuth
from aiollm.http.exceptions.resolver import BaseErrorResolver
from aiollm.http.protocols import IAPI, IAuth
from aiollm.session.aiohttp import AiohttpSession
from aiollm.testing.doubles import TEST_TOKEN, FakeTransport

LIBRARY: Final = Path(aiollm.__file__).resolve().parent
"""The library's directory, which a warning skips."""


class AnotherErrorResolver(BaseErrorResolver):
    """The resolver of a provider no client is of.

    The default ``ClientCase.foreign_resolver``: a session with it does
    not fit any client.
    """

    provider = Provider("another")


@dataclass(frozen=True)
class ClientCase:
    """One provider client in the client scenarios.

    Attributes:
        client: The client class.
        provider: Its provider.
        api: The API its sessions are built for.
        url: A path of the API and the URL it builds.
        resolver: The provider's error resolver class.
        env: The environment variable with the key.
        auths: Authentications the provider takes.
        foreign_resolver: Another provider's resolver, warned about.
    """

    client: type[BaseProviderClient]
    provider: Provider
    api: IAPI
    url: tuple[str, str]
    resolver: type[BaseErrorResolver]
    env: str
    auths: tuple[IAuth, ...] = field(default=(BearerAuth("k"),))
    foreign_resolver: type[BaseErrorResolver] = AnotherErrorResolver


class ClientContract:
    """The client scenarios, for a subclass to run on its provider.

    A subclass whose name starts with ``Test`` sets ``case``; pytest
    collects its scenarios.

    Attributes:
        case: The client the scenarios run on.
    """

    case: ClassVar[ClientCase]

    def test_client_from_a_token_outside_a_coroutine(self) -> None:
        """Client from a token outside a coroutine."""
        case = self.case
        session = case.client(TEST_TOKEN).session

        assert isinstance(session, AiohttpSession)
        assert session.api is case.api
        assert session.auth == BearerAuth(TEST_TOKEN)
        assert type(session.error_resolver) is case.resolver
        assert case.client.provider == case.provider

    def test_api_url(self) -> None:
        """Api url."""
        case = self.case
        path, url = case.url

        assert case.api.build_url(path) == url

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"token": "t", "session": FakeTransport()}],
        ids=["neither", "both"],
    )
    def test_client_needs_a_token_or_a_session(
        self,
        kwargs: Any,
    ) -> None:
        """Client needs a token or a session."""
        case = self.case
        with pytest.raises(ValueError, match="token or a session"):
            case.client(**kwargs)

    def test_client_reads_the_key_from_the_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Client reads the key from the environment."""
        case = self.case
        monkeypatch.setenv(case.env, "sk-env")

        session = case.client().session

        assert isinstance(session, AiohttpSession)
        assert session.auth.encode() == "Bearer sk-env"

    def test_explicit_token_wins_over_the_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit token wins over the environment."""
        case = self.case
        monkeypatch.setenv(case.env, "sk-env")

        session = case.client("sk-explicit").session

        assert isinstance(session, AiohttpSession)
        assert session.auth.encode() == "Bearer sk-explicit"

    def test_session_ignores_the_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Session ignores the environment."""
        case = self.case
        monkeypatch.setenv(case.env, "sk-env")
        session = FakeTransport(error_resolver=case.resolver())

        assert case.client(session=session).session is session

    @pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
    def test_missing_key_names_the_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str | None,
    ) -> None:
        """Missing key names the variable."""
        case = self.case
        # An empty variable counts as unset.
        monkeypatch.delenv(case.env, raising=False)
        if value is not None:
            monkeypatch.setenv(case.env, value)

        with pytest.raises(ValueError, match=f"or set {case.env}"):
            case.client()

    def test_a_fitting_session_gets_no_warning(self) -> None:
        """A fitting session gets no warning."""
        case = self.case
        for auth in case.auths:
            session = FakeTransport(auth=auth, error_resolver=case.resolver())

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                client = case.client(session=session)

            assert client.session is session

    @pytest.mark.parametrize(
        "resolver",
        ["generic", "foreign"],
    )
    def test_a_resolver_of_no_or_another_provider_is_warned_about(
        self,
        resolver: str,
    ) -> None:
        """A resolver of no or another provider is warned about."""
        case = self.case
        given = (
            BaseErrorResolver()
            if resolver == "generic"
            else case.foreign_resolver()
        )
        session = FakeTransport(error_resolver=given)

        with pytest.warns(
            SessionMismatchWarning,
            match=f"is using {type(given).__name__}. Consider using "
            f"{case.resolver.__name__}",
        ) as info:
            case.client(session=session)

        # The warning points past the library, at the code that built
        # the client.
        assert not Path(info[0].filename).resolve().is_relative_to(LIBRARY)

    def test_another_auth_header_is_warned_about(self) -> None:
        """Another auth header is warned about."""
        case = self.case
        session = FakeTransport(
            auth=APIKeyAuth("k", header_name="X-Key"),
            error_resolver=case.resolver(),
        )

        with pytest.warns(
            SessionMismatchWarning,
            match=(
                f"sends the key in X-Key, but {case.provider.value} takes it"
            ),
        ):
            case.client(session=session)

    def test_other_sessions_are_not_checked(self) -> None:
        """Other sessions are not checked."""
        case = self.case
        # Not a BaseSession, but it has the ISession methods.
        session = BaseClient(FakeTransport(error_resolver=BaseErrorResolver()))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            client = case.client(session=session)

        assert client.session is session

    def test_session_arguments_are_refused_with_a_session(
        self,
    ) -> None:
        """Session arguments are refused with a session."""
        case = self.case
        session = FakeTransport(error_resolver=case.resolver())

        with pytest.raises(ValueError, match="pass them to the given session"):
            case.client(session=session, max_body_size=1)
