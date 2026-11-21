"""The shared resource of files stored with a provider."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Unpack

from pydantic import MISSING

from aiollm.adapters.validation import container_of
from aiollm.exceptions import RequestValidationError
from aiollm.resources.base import BaseResource
from aiollm.resources.protocols import IFilesAdapter
from aiollm.session.protocols import CallOptions, ISession
from aiollm.types.requests.generation import NoProviderOptions
from aiollm.types.requests.upload import FileUpload

if TYPE_CHECKING:
    from os import PathLike

    from aiollm.response.stream import Stream
    from aiollm.types.files import BinaryStream, InputFile
    from aiollm.types.requests.content import ProviderFile
    from aiollm.types.responses.files import StoredFile


class FilesResource[
    RawFileT,
    RawPageT,
    OptionsT: Mapping[str, object] = NoProviderOptions,
    SessionT: ISession = ISession,
](
    BaseResource[SessionT],
):
    """Stores files with any provider through that provider's adapter.

    The class is the same for every provider: a provider's client builds
    it with its own adapter. A stored file is named by a
    ``ProviderFile``, which ``StoredFile.ref`` holds and a part takes as
    ``file=``; one of another provider raises ``RequestValidationError``
    before anything is sent.

    The library deletes nothing by itself: a file is kept until it is
    deleted or expires (``expires_in_seconds``). Every method takes the
    options of the call (``headers``, ``timeout``, ``retry_policy``) as
    ``send_request`` does; the provider's own upload fields, such as
    OpenAI's ``purpose``, go in ``provider_options`` of ``upload``.

    Example:
        >>> stored = await client.files.upload(
        ...     Path("report.pdf"), expires_in_seconds=3600
        ... )
        >>> response = await client.generation.create(
        ...     model="gpt-5",
        ...     input=[Turn.user([DocumentPart(file=stored.ref)])],
        ... )
        >>> await client.files.delete(stored.ref)
    """

    def __init__(
        self,
        session: SessionT,
        adapter: IFilesAdapter[RawFileT, RawPageT, OptionsT],
    ) -> None:
        """Keep the session and the adapter.

        Args:
            session: The session that sends the requests.
            adapter: Builds the provider's methods and converts what
                they return.
        """
        super().__init__(session)
        self._adapter = adapter

    async def upload(
        self,
        file: InputFile | bytes | PathLike[str] | BinaryStream,
        /,
        *,
        expires_in_seconds: int | MISSING = MISSING,
        provider_options: OptionsT | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> StoredFile[RawFileT]:
        """Store a file with the provider.

        Never retried: ``POST /files`` is not idempotent, and a retry
        after a lost answer would store the file twice. A timeout or a
        broken connection can therefore leave a file stored whose id was
        never received; ``expires_in_seconds`` bounds how long it stays.

        Args:
            file: The content: an ``InputFile``, ``bytes``, a
                ``pathlib.Path`` or a binary stream; its ``filename`` is
                the name the provider keeps.
            expires_in_seconds: Seconds after which the provider deletes
                the file; left out, it is kept until it is deleted.
            provider_options: The provider's own upload fields, such as
                ``{"purpose": "vision"}`` for OpenAI.
            **options: Options of the call (``headers``, ``timeout``,
                ``retry_policy``), passed on to the session.

        Returns:
            The stored file.

        Raises:
            ValidationError: ``file`` is not a file, or
                ``expires_in_seconds`` is not positive.
            RequestValidationError: The provider cannot take the upload,
                such as an expiry outside its range, or a key of
                ``provider_options`` is not one of its options; raised
                before anything is sent.
        """
        adapter = self._adapter
        if provider_options is not MISSING:
            adapter = adapter.with_options(provider_options)
        # Validated from a dict: the field converts bytes, paths and
        # streams into an InputFile, which its type alone does not say.
        request = FileUpload.model_validate(
            {"file": file, "expires_in_seconds": expires_in_seconds}
        )
        raw = await self.session.send_request(
            adapter.to_upload_method(request), **options
        )
        return adapter.to_stored_file(raw)

    async def retrieve(
        self,
        file: ProviderFile,
        /,
        **options: Unpack[CallOptions],
    ) -> StoredFile[RawFileT]:
        """Read a stored file's metadata.

        Args:
            file: The file.
            **options: Options of the call, passed on to the session.

        Returns:
            The stored file.

        Raises:
            RequestValidationError: ``file`` is another provider's, or a
                file of a container (``ProducedFile.container_id``),
                which only ``download`` reaches.
            NotFoundError: The provider has no such file.
        """
        method = self._adapter.to_retrieve_method(self._stored_id(file))
        raw = await self.session.send_request(method, **options)
        return self._adapter.to_stored_file(raw)

    async def list(
        self,
        **options: Unpack[CallOptions],
    ) -> AsyncGenerator[StoredFile[RawFileT]]:
        """Go through every stored file, a page at a time.

        Each page is one request; the next is sent when the last file
        of a page has been taken, so stopping early sends no more.

        Args:
            **options: Options of each page's call, passed on to the
                session.

        Yields:
            The files, in the provider's order (OpenAI and Anthropic:
            the newest first).
        """
        cursor: str | None = None
        while True:
            method = self._adapter.to_list_method(cursor)
            page = await self.session.send_request(method, **options)
            files, cursor = self._adapter.to_page(page)
            for raw in files:
                yield self._adapter.to_stored_file(raw)
            if cursor is None:
                return

    async def delete(
        self,
        file: ProviderFile,
        /,
        **options: Unpack[CallOptions],
    ) -> None:
        """Delete a stored file.

        A deletion retried after its answer was lost raises
        ``NotFoundError``, since the file is already gone.

        Args:
            file: The file.
            **options: Options of the call, passed on to the session.

        Raises:
            RequestValidationError: ``file`` is another provider's, or a
                file of a container, which the container's expiry
                deletes.
            NotFoundError: The provider has no such file.
        """
        method = self._adapter.to_delete_method(self._stored_id(file))
        await self.session.send_request(method, **options)

    def download(
        self,
        file: ProviderFile,
        /,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[Stream[bytes]]:
        """Stream a stored file's content; use with ``async with``.

        Each provider lets only some files be downloaded: OpenAI refuses
        a ``user_data`` file, Anthropic one it did not make itself (code
        execution, skills); both answer ``BadRequestError``. A file a
        built-in tool made (``GenerationResponse.files``) is downloaded
        from wherever it lives, its container included.

        Args:
            file: The file, such as a ``StoredFile.ref`` or a
                ``ProducedFile``.
            **options: Options of opening the stream, passed on to the
                session.

        Returns:
            A context manager that yields the content in chunks
            (``save(path)`` writes it to a file without holding it in
            memory) and closes it on exit.

        Raises:
            RequestValidationError: ``file`` is another provider's, or a
                file of a container where the provider has none.
        """
        self._check_provider(file)
        method = self._adapter.to_download_method(file)
        return self.session.stream(method, **options)

    def _stored_id(self, file: ProviderFile) -> str:
        """Take the id of a stored file of this resource's provider.

        Args:
            file: The file.

        Returns:
            Its id.

        Raises:
            RequestValidationError: The file is another provider's, or
                a file of a container.
        """
        self._check_provider(file)
        if container_of(file) is not None:
            message = (
                f"file {file.id} is in a container, which only a "
                "download reaches"
            )
            raise RequestValidationError(
                message, provider=self._adapter.provider, field="file"
            )
        return file.id

    def _check_provider(self, file: ProviderFile) -> None:
        """Check that a file is this resource's provider's.

        Args:
            file: The file.

        Raises:
            RequestValidationError: The file is another provider's.
        """
        provider = self._adapter.provider
        if file.provider != provider:
            message = (
                f"the file is stored by {file.provider.value}, "
                f"not by {provider.value}"
            )
            raise RequestValidationError(
                message, provider=provider, field="file"
            )
