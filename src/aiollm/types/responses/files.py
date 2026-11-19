"""Shared metadata of a file stored by a provider."""

from __future__ import annotations

from pydantic import AwareDatetime

from aiollm.types.requests.content import ProviderFile
from aiollm.types.responses.base import BaseResponse


class StoredFile[RawT](BaseResponse):
    """A file stored by a provider, with the provider's own metadata.

    ``RawT`` is the provider's file type (OpenAI's ``OpenAIFile``,
    Anthropic's ``FileMetadata``); like ``GenerationResponse``, it has
    no bound, so ``raw`` is kept as it is given. What only one provider
    has, such as OpenAI's ``purpose`` or Anthropic's ``mime_type`` and
    ``downloadable``, stays in ``raw``.

    Attributes:
        ref: The file as a part or a ``client.files`` call names it.
        filename: The name the provider keeps.
        size: Its size in bytes; ``None`` when the provider does not
            know it yet (OpenAI may send ``null``).
        created_at: When it was stored, in UTC.
        expires_at: When the provider deletes it, in UTC; ``None`` if
            it is kept until it is deleted.
        raw: The provider's own metadata.

    Example:
        >>> stored = await client.files.upload(Path("report.pdf"))
        >>> DocumentPart(file=stored.ref)
    """

    ref: ProviderFile
    filename: str
    size: int | None
    created_at: AwareDatetime
    expires_at: AwareDatetime | None
    raw: RawT
