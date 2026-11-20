"""URL and provider of the echo API."""

from __future__ import annotations

from typing import Final

from aiollm.enums import Provider
from aiollm.http.api import BaseAPI

PROVIDER: Final = Provider("echo")
"""The provider of this package; its client, resolver and adapters use it."""


class EchoAPI(BaseAPI):
    """Builds echo request URLs."""


ECHO_API: Final[EchoAPI] = EchoAPI(
    base_url="https://echo.invalid/",
    version="v1",
)
"""The echo API, version 1.

``.invalid`` never resolves, so a client that was not pointed at a
local server fails at once instead of reaching anything.
"""
