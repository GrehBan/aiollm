"""``SessionFactory``: a provider's session from a token."""

from __future__ import annotations

from typing import Any

from aiollm.http.auth import BearerAuth
from aiollm.session.factory import SessionFactory
from aiollm.testing import FakeTransport
from aiollm.testing.echo import ECHO_API, EchoErrorResolver


def test_factory_turns_a_token_into_auth_and_forwards_arguments() -> None:
    calls: list[tuple[Any, ...]] = []

    def build(
        api: Any,
        auth: BearerAuth,
        resolver: Any,
        *,
        limit: int = 1,
    ) -> FakeTransport:
        calls.append((api, auth, resolver, limit))
        return FakeTransport()

    resolver = EchoErrorResolver()
    factory = SessionFactory(
        build,
        auth_factory=BearerAuth,
        api=ECHO_API,
        error_resolver=resolver,
    )

    factory("token", limit=5)
    factory(BearerAuth("ready"))

    assert calls == [
        (ECHO_API, BearerAuth("token"), resolver, 5),
        (ECHO_API, BearerAuth("ready"), resolver, 1),
    ]
