"""Error mapping of the echo API."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Final

from aiollm.http.exceptions.resolver import BaseErrorResolver
from aiollm.http.exceptions.status import APIStatusError, QuotaExceededError
from aiollm.testing.echo.api import PROVIDER

QUOTA_CODE: Final = "insufficient_quota"
"""``error.code`` of an exhausted quota, over HTTP or in a stream."""

ERROR_STATUSES: Final[Mapping[str, int]] = {
    "invalid_request": HTTPStatus.BAD_REQUEST,
    QUOTA_CODE: HTTPStatus.TOO_MANY_REQUESTS,
    "overloaded": HTTPStatus.SERVICE_UNAVAILABLE,
}
"""HTTP status of each error code an ``error`` event of a stream carries."""


class EchoErrorResolver(BaseErrorResolver):
    """Error resolver of the echo API.

    Errors come in the common ``{"error": {"code", "message"}}`` shape,
    over HTTP and as the ``error`` event of a stream. An exhausted quota
    (``insufficient_quota``) is a ``QuotaExceededError`` and is never
    retried; everything else is classified by its status.
    """

    provider = PROVIDER
    error_statuses = ERROR_STATUSES

    def classify_status(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIStatusError], bool]:
        """Pick the error class and transience of the echo statuses.

        Args:
            status: HTTP status the error is classified by.
            code: Provider error code, if any.
            error_type: Provider error type, if any.
            message: Error message, if any.
            retry_after: Seconds from ``Retry-After``, if any.

        Returns:
            ``QuotaExceededError`` (never transient) for an exhausted
            quota, else the default classification.
        """
        if code == QUOTA_CODE:
            return QuotaExceededError, False
        return super().classify_status(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )
