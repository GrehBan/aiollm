"""Adapter of shared files to the echo API."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final, Self

from pydantic import MISSING

from aiollm.adapters.base import BaseFilesAdapter
from aiollm.adapters.validation import (
    checked_options,
    merge_options,
    request_errors,
)
from aiollm.testing.echo.api import PROVIDER
from aiollm.testing.echo.methods import (
    DeleteFile,
    DownloadFile,
    ListFiles,
    RetrieveFile,
    UploadFile,
)
from aiollm.testing.echo.types import EchoFile, EchoFileOptions, EchoFilePage
from aiollm.types.requests.content import ProviderFile
from aiollm.types.requests.upload import FileUpload
from aiollm.types.responses.files import StoredFile

DEFAULT_PAGE_SIZE: Final = 100
"""Files asked for in one page of ``client.files.list()``."""


class EchoFilesAdapter(BaseFilesAdapter[EchoFile, EchoFilePage]):
    """Stores files with the echo ``/files``.

    ``expires_in_seconds`` is sent as it is, and ``provider_options``
    may add a ``purpose``. The list goes newest first, ``page_size``
    files a request, by the page's ``next``.

    Attributes:
        page_size: Files asked for in one page of the list.

    Example:
        >>> files = FilesResource(session, EchoFilesAdapter())
    """

    provider = PROVIDER

    def __init__(self, *, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        """Store the page size.

        Args:
            page_size: Files asked for in one page of the list.
        """
        self.page_size = page_size
        self.provider_options: dict[str, Any] = {}

    def with_options(self, options: EchoFileOptions, /) -> Self:
        """Return a copy that sends the options with every upload.

        Args:
            options: Fields of ``UploadFile`` the shared upload does not
                hold, such as ``{"purpose": "vision"}``.

        Returns:
            A copy of the adapter; this one is left as it is.

        Raises:
            RequestValidationError: A key is not one of the options.
        """
        adapter = copy.copy(self)
        adapter.provider_options = checked_options(
            options, EchoFileOptions, self.provider
        )
        return adapter

    def to_upload_method(self, request: FileUpload) -> UploadFile:
        """Build the method that stores a file.

        Args:
            request: The shared upload.

        Returns:
            ``UploadFile`` with the file, its expiry and the options.

        Raises:
            RequestValidationError: The echo API cannot take the upload,
                such as an expiry outside an hour to thirty days.
        """
        fields = merge_options(
            {
                "file": request.file,
                "expires_in_seconds": request.expires_in_seconds,
            },
            self.provider_options,
            self.provider,
        )
        with request_errors(self.provider):
            return UploadFile(**fields)

    def to_retrieve_method(self, file_id: str) -> RetrieveFile:
        """Build the method that reads a file's metadata.

        Args:
            file_id: The file's id.

        Returns:
            ``RetrieveFile``.
        """
        return RetrieveFile(file_id=file_id)

    def to_list_method(self, cursor: str | None) -> ListFiles:
        """Build the method that reads one page of the stored files.

        Args:
            cursor: ``next`` of the page before, or ``None`` for the
                first page.

        Returns:
            ``ListFiles`` of ``page_size`` files.
        """
        return ListFiles(
            limit=self.page_size,
            after=MISSING if cursor is None else cursor,
        )

    def to_page(
        self, raw: EchoFilePage
    ) -> tuple[Sequence[EchoFile], str | None]:
        """Take the files of a page and where the next one starts.

        Args:
            raw: The page.

        Returns:
            Its files and its ``next``.
        """
        return raw.data, raw.next

    def to_delete_method(self, file_id: str) -> DeleteFile:
        """Build the method that deletes a file.

        Args:
            file_id: The file's id.

        Returns:
            ``DeleteFile``.
        """
        return DeleteFile(file_id=file_id)

    def to_download_method(self, file_id: str) -> DownloadFile:
        """Build the method that streams a file's content.

        Args:
            file_id: The file's id.

        Returns:
            ``DownloadFile``.
        """
        return DownloadFile(file_id=file_id)

    def to_stored_file(self, raw: EchoFile) -> StoredFile[EchoFile]:
        """Convert the echo file into the shared metadata.

        Args:
            raw: The echo file.

        Returns:
            Its id, name, ``bytes`` as ``size`` and its times in UTC.
        """
        return StoredFile[EchoFile](
            ref=ProviderFile(provider=self.provider, id=raw.id),
            filename=raw.filename,
            size=raw.bytes,
            created_at=datetime.fromtimestamp(raw.created_at, UTC),
            expires_at=None
            if raw.expires_at is None
            else datetime.fromtimestamp(raw.expires_at, UTC),
            raw=raw,
        )
