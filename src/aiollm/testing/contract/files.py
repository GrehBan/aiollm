"""Scenarios every provider's ``client.files`` passes.

A provider's tests subclass ``FilesContract`` with a ``FilesCase`` of
its Files API; every scenario then runs against the provider's client on
the local server (``aiollm.testing.plugin`` fixtures), with the
provider's documented results. Only the wire format differs per
provider: the shared results and errors must not. The client needs
``generation`` too, since a stored file goes into a turn.

Example:
    >>> from aiollm.testing.contract import FilesCase, FilesContract
    >>> class TestExampleFiles(FilesContract):
    ...     case = FilesCase(provider=PROVIDER, client=ExampleClient, ...)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Final

import pytest
from pydantic import ValidationError

from aiollm.enums import Provider
from aiollm.exceptions import RequestValidationError
from aiollm.http.exceptions.status import (
    NotFoundError,
    ServiceUnavailableError,
)
from aiollm.testing.doubles import TEST_TOKEN
from aiollm.testing.recording import Reply
from aiollm.types.files import BytesFile
from aiollm.types.requests import (
    DocumentPart,
    ImagePart,
    ProviderFile,
    TextPart,
    Turn,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aiollm.resources import IFilesResource
    from aiollm.session.factory import SessionFactory
    from aiollm.testing.aiohttp import Recorder
    from aiollm.testing.doubles import LocalAPI

PDF: Final = b"%PDF-1.4\n%%EOF\n"
"""Bytes the upload sends; the server never looks into them."""

FOREIGN: Final = ProviderFile(provider=Provider("another"), id="file-1")
"""A file of a provider no client is of."""


@dataclass(frozen=True)
class FilesCase:
    """One provider in the files scenarios.

    Attributes:
        provider: The provider.
        client: Builds the client from a token.
        factory: The session factory of the client, sent to the local
            server by replacing its ``api``.
        raw_type: The provider's file type, ``StoredFile.raw``.
        file: A documented file.
        size: Its size in bytes.
        created_at: When it was stored, in UTC.
        expires_at: When it expires, in UTC.
        deleted: A documented deletion.
        page: A page of the list with ``file`` and a next page.
        cursor: The query key and value that ask for the next page.
        last_page: The body of a page with no next one, its first file
            another than ``file``.
        upload_parts: The form fields of an upload of ``report.pdf``
            that expires in an hour, in order, besides ``file``.
        bare_upload_parts: The names of the form fields of an upload
            with neither expiry nor options, in order.
        options: Valid upload options, with the fields they change;
            ``None`` if the provider's uploads take none.
        message_path: Path of the generation endpoint.
        message: A finished generation response.
        input_field: The body field of the conversation.
        stored_parts: What a document and an image of ``file`` become
            in a user turn.
    """

    provider: Provider
    client: Callable[..., Any]
    factory: SessionFactory[Any, Any, ...]
    raw_type: type
    file: dict[str, Any]
    size: int
    created_at: datetime
    expires_at: datetime
    deleted: dict[str, Any]
    page: dict[str, Any]
    cursor: tuple[str, str]
    last_page: dict[str, Any]
    upload_parts: dict[str, bytes]
    bare_upload_parts: list[str]
    options: tuple[dict[str, Any], dict[str, bytes]] | None
    message_path: str
    message: dict[str, Any]
    input_field: str
    stored_parts: list[dict[str, Any]]


class FilesContract:
    """The files scenarios, for a subclass to run on its provider.

    A subclass whose name starts with ``Test`` sets ``case``; pytest
    collects its scenarios. It needs the ``aiollm.testing.plugin``
    fixtures (``recorder``, ``api``).

    Attributes:
        case: The provider the scenarios run on.
    """

    case: ClassVar[FilesCase]

    @pytest.fixture
    def client(self, api: LocalAPI, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Build the provider's client, sent to the local server.

        Args:
            api: The local server's API.
            monkeypatch: pytest's monkeypatch.

        Returns:
            The client, built from ``TEST_TOKEN``.
        """
        monkeypatch.setattr(self.case.factory, "api", api)
        return self.case.client(TEST_TOKEN)

    def ref(self) -> ProviderFile:
        """Refer to the documented file.

        Returns:
            The file of the case's provider.
        """
        return ProviderFile(
            provider=self.case.provider, id=self.case.file["id"]
        )

    async def test_an_upload_returns_the_shared_metadata(
        self, client: Any, recorder: Recorder
    ) -> None:
        """An upload returns the shared metadata."""
        case = self.case
        recorder.reply(Reply(json=case.file))

        async with client:
            stored = await client.files.upload(
                BytesFile(PDF, "report.pdf"), expires_in_seconds=3600
            )

        assert stored.ref == self.ref()
        assert stored.filename == case.file["filename"]
        assert stored.size == case.size
        assert stored.created_at == case.created_at
        assert stored.expires_at == case.expires_at
        assert stored.created_at.tzinfo is UTC
        assert isinstance(stored.raw, case.raw_type)
        assert (recorder.last.method, recorder.last.raw_path) == (
            "POST",
            "/v1/files",
        )
        parts = {part.name: part for part in recorder.last.parts}
        assert parts["file"].filename == "report.pdf"
        assert parts["file"].data == PDF
        assert {
            name: part.data for name, part in parts.items() if name != "file"
        } == case.upload_parts

    async def test_an_upload_without_expiry_is_kept(
        self, client: Any, recorder: Recorder
    ) -> None:
        """An upload without expiry is kept."""
        case = self.case
        recorder.reply(Reply(json=case.file | {"expires_at": None}))

        async with client:
            stored = await client.files.upload(PDF)

        assert stored.expires_at is None
        names = [part.name for part in recorder.last.parts]
        # No expiry is sent; a default of the provider may be.
        assert names == case.bare_upload_parts

    async def test_upload_options_reach_the_form(
        self, client: Any, recorder: Recorder
    ) -> None:
        """Upload options reach the form."""
        case = self.case
        if case.options is None:
            async with client:
                with pytest.raises(RequestValidationError) as caught:
                    await client.files.upload(
                        PDF, provider_options={"purpose": "vision"}
                    )
            assert caught.value.field == "purpose"
            assert recorder.requests == []
            return

        options, sent = case.options
        recorder.reply(Reply(json=case.file))

        async with client:
            await client.files.upload(
                BytesFile(PDF, "report.pdf"),
                expires_in_seconds=3600,
                provider_options=options,
            )

        parts = {part.name: part.data for part in recorder.last.parts}
        del parts["file"]
        assert parts == sent

    async def test_empty_upload_options_change_nothing(
        self, client: Any, recorder: Recorder
    ) -> None:
        """Empty upload options change nothing."""
        case = self.case
        recorder.reply(Reply(json=case.file))

        async with client:
            await client.files.upload(
                BytesFile(PDF, "report.pdf"),
                expires_in_seconds=3600,
                provider_options={},
            )

        parts = {part.name: part.data for part in recorder.last.parts}
        del parts["file"]
        assert parts == case.upload_parts

    @pytest.mark.parametrize("seconds", [60, 10**9])
    async def test_an_expiry_out_of_range_is_refused_before_sending(
        self, client: Any, recorder: Recorder, seconds: int
    ) -> None:
        """An expiry out of range is refused before sending."""
        async with client:
            with pytest.raises(RequestValidationError) as caught:
                await client.files.upload(PDF, expires_in_seconds=seconds)

        assert caught.value.field == "expires_in_seconds"
        assert recorder.requests == []

    async def test_an_upload_that_is_not_a_file_is_refused(
        self, client: Any, recorder: Recorder
    ) -> None:
        """An upload that is not a file is refused."""
        async with client:
            with pytest.raises(ValidationError):
                await client.files.upload("report.pdf")
            with pytest.raises(ValidationError):
                await client.files.upload(PDF, expires_in_seconds=0)

        assert recorder.requests == []

    async def test_an_upload_is_not_retried(
        self, client: Any, recorder: Recorder
    ) -> None:
        """An upload is not retried."""
        recorder.reply(Reply(status=503, json={"error": {"message": "Later"}}))

        async with client:
            with pytest.raises(ServiceUnavailableError):
                await client.files.upload(PDF)

        assert len(recorder.requests) == 1

    async def test_retrieve_reads_the_metadata(
        self, client: Any, recorder: Recorder
    ) -> None:
        """Retrieve reads the metadata."""
        case = self.case
        recorder.reply(Reply(json=case.file))

        async with client:
            stored = await client.files.retrieve(self.ref(), timeout=5)

        assert stored.ref == self.ref()
        assert (recorder.last.method, recorder.last.raw_path) == (
            "GET",
            f"/v1/files/{case.file['id']}",
        )

    async def test_list_goes_through_every_page(
        self, client: Any, recorder: Recorder
    ) -> None:
        """List goes through every page."""
        case = self.case
        recorder.reply(Reply(json=case.page), Reply(json=case.last_page))

        async with client:
            files = [stored async for stored in client.files.list()]

        assert [stored.ref.id for stored in files] == [
            case.file["id"],
            case.last_page["data"][0]["id"],
        ]
        first, second = recorder.requests
        key, value = case.cursor
        assert first.query.get("limit") == second.query.get("limit") == "100"
        assert key not in first.query
        assert second.query.get(key) == value

    async def test_list_stops_asking_when_the_caller_stops(
        self, client: Any, recorder: Recorder
    ) -> None:
        """List stops asking when the caller stops."""
        case = self.case
        recorder.reply(Reply(json=case.page))

        async with client:
            files = client.files.list()
            await anext(files)
            await files.aclose()

        assert len(recorder.requests) == 1

    async def test_delete_returns_nothing(
        self, client: Any, recorder: Recorder
    ) -> None:
        """Delete returns nothing."""
        case = self.case
        recorder.reply(Reply(json=case.deleted))

        async with client:
            assert await client.files.delete(self.ref()) is None

        assert (recorder.last.method, recorder.last.raw_path) == (
            "DELETE",
            f"/v1/files/{case.file['id']}",
        )

    async def test_a_missing_file_raises_not_found(
        self, client: Any, recorder: Recorder
    ) -> None:
        """A missing file raises not found."""
        recorder.reply(
            Reply(status=404, json={"error": {"message": "No such file"}})
        )

        async with client:
            with pytest.raises(NotFoundError):
                await client.files.retrieve(self.ref())

    async def test_download_streams_the_content(
        self, client: Any, recorder: Recorder, tmp_path: Path
    ) -> None:
        """Download streams the content."""
        case = self.case
        recorder.reply(
            Reply(chunks=[PDF[:4], PDF[4:]], content_type="application/pdf")
        )
        target = tmp_path / "copy.pdf"

        async with client, client.files.download(self.ref()) as content:
            written = await content.save(target)

        assert (written, target.read_bytes()) == (len(PDF), PDF)
        assert recorder.last.raw_path == f"/v1/files/{case.file['id']}/content"

    @pytest.mark.parametrize("call", ["retrieve", "delete", "download"])
    async def test_another_providers_file_is_refused_before_sending(
        self, client: Any, recorder: Recorder, call: str
    ) -> None:
        """Another providers file is refused before sending."""
        case = self.case
        files: IFilesResource[Any] = client.files

        async with client:
            with pytest.raises(RequestValidationError) as caught:
                if call == "download":
                    files.download(FOREIGN)
                else:
                    await getattr(files, call)(FOREIGN)

        assert (caught.value.provider, caught.value.field) == (
            case.provider,
            "file",
        )
        assert recorder.requests == []

    async def test_a_stored_file_is_sent_by_its_id(
        self, client: Any, recorder: Recorder
    ) -> None:
        """A stored file is sent by its id."""
        case = self.case
        recorder.reply(Reply(json=case.message))
        turn = Turn.user(
            [
                TextPart(text="What is in these?"),
                DocumentPart(file=self.ref()),
                ImagePart(file=self.ref()),
            ]
        )

        async with client:
            await client.generation.create(model="m", input=[turn])

        body = json.loads(recorder.last.body)
        assert recorder.last.raw_path == case.message_path
        assert body[case.input_field][0]["content"][1:] == case.stored_parts

    async def test_another_providers_file_in_a_part_is_refused(
        self, client: Any, recorder: Recorder
    ) -> None:
        """Another providers file in a part is refused."""
        case = self.case
        turn = Turn.user([DocumentPart(file=FOREIGN)])

        async with client:
            with pytest.raises(RequestValidationError) as caught:
                await client.generation.create(model="m", input=[turn])

        assert (caught.value.provider, caught.value.field) == (
            case.provider,
            "input",
        )
        assert recorder.requests == []
