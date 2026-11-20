"""Shared files behave the same for every provider.

Each scenario runs against ``OpenAIClient.files`` and
``AnthropicClient.files`` on the local server, with the documented
results of each provider's Files API; only the wire format differs per
provider, the shared results and errors must not.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from aiollm.enums import Provider
from aiollm.exceptions import RequestValidationError
from aiollm.http.exceptions.status import (
    NotFoundError,
    ServiceUnavailableError,
)
from aiollm.providers.anthropic import AnthropicClient
from aiollm.providers.anthropic.api import PROVIDER as ANTHROPIC_PROVIDER
from aiollm.providers.anthropic.client import (
    session_factory as anthropic_factory,
)
from aiollm.providers.anthropic.types import FileMetadata
from aiollm.providers.openai import OpenAIClient
from aiollm.providers.openai.api import PROVIDER as OPENAI_PROVIDER
from aiollm.providers.openai.client import session_factory as openai_factory
from aiollm.providers.openai.types import OpenAIFile
from aiollm.resources import IFilesResource
from aiollm.session.factory import SessionFactory
from aiollm.types.files import BytesFile
from aiollm.types.requests import (
    DocumentPart,
    ImagePart,
    ProviderFile,
    TextPart,
    Turn,
)
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply
from tests.providers import test_anthropic_messages as anthropic
from tests.providers import test_openai_responses as openai
from tests.providers.test_files_methods import documented

PDF = b"%PDF-1.4\n%%EOF\n"
"""Bytes the upload sends; the server never looks into them."""


@dataclass(frozen=True)
class Case:
    """One provider in the scenarios.

    Attributes:
        provider: The provider.
        client: Builds the client from a token.
        factory: The session factory to send to the local server.
        raw_type: The provider's file type, ``StoredFile.raw``.
        file: A documented file.
        size: Its size in bytes.
        created_at: When it was stored, in UTC.
        expires_at: When it expires, in UTC.
        deleted: A documented deletion.
        page: A page of the list with ``file`` and a next page.
        cursor: The query key and value that ask for the next page.
        last_page: The body of a page with no next one.
        upload_parts: The form fields of an upload of ``report.pdf``
            that expires in an hour, in order, besides ``file``.
        options: Valid upload options, with the fields they change.
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
    options: tuple[dict[str, Any], dict[str, bytes]] | None
    message_path: str
    message: dict[str, Any]
    input_field: str
    stored_parts: list[dict[str, Any]]


OPENAI_FILE = documented("openai/files", "createFile")
ANTHROPIC_FILE = documented("anthropic/files", "createFile")

OPENAI = Case(
    provider=OPENAI_PROVIDER,
    client=OpenAIClient,
    factory=openai_factory,
    raw_type=OpenAIFile,
    file=OPENAI_FILE,
    size=120000,
    created_at=datetime(2023, 2, 28, 18, 56, 42, tzinfo=UTC),
    expires_at=datetime(2023, 3, 30, 18, 56, 42, tzinfo=UTC),
    deleted=documented("openai/files", "deleteFile"),
    page={
        "object": "list",
        "data": [OPENAI_FILE],
        "first_id": OPENAI_FILE["id"],
        "last_id": OPENAI_FILE["id"],
        "has_more": True,
    },
    cursor=("after", OPENAI_FILE["id"]),
    last_page={
        "object": "list",
        "data": [OPENAI_FILE | {"id": "file-older"}],
        "first_id": "file-older",
        "last_id": "file-older",
        "has_more": False,
    },
    upload_parts={
        "purpose": b"user_data",
        "expires_after[anchor]": b"created_at",
        "expires_after[seconds]": b"3600",
    },
    options=(
        {"purpose": "vision"},
        {
            "purpose": b"vision",
            "expires_after[anchor]": b"created_at",
            "expires_after[seconds]": b"3600",
        },
    ),
    message_path="/v1/responses",
    message=openai.RESPONSE,
    input_field="input",
    stored_parts=[
        {"type": "input_file", "file_id": OPENAI_FILE["id"]},
        {
            "type": "input_image",
            "file_id": OPENAI_FILE["id"],
            "detail": "auto",
        },
    ],
)

ANTHROPIC = Case(
    provider=ANTHROPIC_PROVIDER,
    client=AnthropicClient,
    factory=anthropic_factory,
    raw_type=FileMetadata,
    file=ANTHROPIC_FILE,
    size=102400,
    created_at=datetime(2025, 4, 15, 18, 37, 24, 100435, tzinfo=UTC),
    expires_at=datetime(2025, 5, 15, 18, 37, 24, 100435, tzinfo=UTC),
    deleted=documented("anthropic/files", "deleteFile"),
    page={"data": [ANTHROPIC_FILE], "next_page": "page_2"},
    cursor=("page", "page_2"),
    last_page={
        "data": [ANTHROPIC_FILE | {"id": "file_older"}],
        "next_page": None,
    },
    upload_parts={"expires_in_seconds": b"3600"},
    options=None,
    message_path="/v1/messages",
    message=anthropic.MESSAGE,
    input_field="messages",
    stored_parts=[
        {
            "type": "document",
            "source": {"type": "file", "file_id": ANTHROPIC_FILE["id"]},
        },
        {
            "type": "image",
            "source": {"type": "file", "file_id": ANTHROPIC_FILE["id"]},
        },
    ],
)


@pytest.fixture(params=[OPENAI, ANTHROPIC], ids=["openai", "anthropic"])
def case(request: pytest.FixtureRequest) -> Case:
    """Each provider in turn."""
    return request.param


@pytest.fixture
def client(
    case: Case,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """The provider's client, sent to the local server."""
    monkeypatch.setattr(case.factory, "api", api)
    return case.client(TEST_TOKEN)


def ref(case: Case) -> ProviderFile:
    """The documented file of the case's provider."""
    return ProviderFile(provider=case.provider, id=case.file["id"])


def foreign(case: Case) -> ProviderFile:
    """A file of the other provider."""
    other = (
        ANTHROPIC_PROVIDER
        if case.provider == OPENAI_PROVIDER
        else OPENAI_PROVIDER
    )
    return ProviderFile(provider=other, id="file-of-another")


async def test_an_upload_returns_the_shared_metadata(
    case: Case, client: Any, recorder: Recorder
) -> None:
    recorder.reply(Reply(json=case.file))

    async with client:
        stored = await client.files.upload(
            BytesFile(PDF, "report.pdf"), expires_in_seconds=3600
        )

    assert stored.ref == ref(case)
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
    case: Case, client: Any, recorder: Recorder
) -> None:
    recorder.reply(Reply(json=case.file | {"expires_at": None}))

    async with client:
        stored = await client.files.upload(PDF)

    assert stored.expires_at is None
    names = [part.name for part in recorder.last.parts]
    # OpenAI's purpose is the only other field: no expiry is sent.
    assert names == ["file", "purpose"] if case.options else ["file"]


async def test_upload_options_reach_the_form(
    case: Case, client: Any, recorder: Recorder
) -> None:
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
    case: Case, client: Any, recorder: Recorder
) -> None:
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
    client: Any, recorder: Recorder, seconds: int
) -> None:
    async with client:
        with pytest.raises(RequestValidationError) as caught:
            await client.files.upload(PDF, expires_in_seconds=seconds)

    assert caught.value.field == "expires_in_seconds"
    assert recorder.requests == []


async def test_an_upload_that_is_not_a_file_is_refused(
    client: Any, recorder: Recorder
) -> None:
    async with client:
        with pytest.raises(ValidationError):
            await client.files.upload("report.pdf")
        with pytest.raises(ValidationError):
            await client.files.upload(PDF, expires_in_seconds=0)

    assert recorder.requests == []


async def test_an_upload_is_not_retried(
    client: Any, recorder: Recorder
) -> None:
    recorder.reply(Reply(status=503, json={"error": {"message": "Later"}}))

    async with client:
        with pytest.raises(ServiceUnavailableError):
            await client.files.upload(PDF)

    assert len(recorder.requests) == 1


async def test_retrieve_reads_the_metadata(
    case: Case, client: Any, recorder: Recorder
) -> None:
    recorder.reply(Reply(json=case.file))

    async with client:
        stored = await client.files.retrieve(ref(case), timeout=5)

    assert stored.ref == ref(case)
    assert (recorder.last.method, recorder.last.raw_path) == (
        "GET",
        f"/v1/files/{case.file['id']}",
    )


async def test_list_goes_through_every_page(
    case: Case, client: Any, recorder: Recorder
) -> None:
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
    case: Case, client: Any, recorder: Recorder
) -> None:
    recorder.reply(Reply(json=case.page))

    async with client:
        files = client.files.list()
        async for _ in files:
            break
        await files.aclose()

    assert len(recorder.requests) == 1


async def test_delete_returns_nothing(
    case: Case, client: Any, recorder: Recorder
) -> None:
    recorder.reply(Reply(json=case.deleted))

    async with client:
        assert await client.files.delete(ref(case)) is None

    assert (recorder.last.method, recorder.last.raw_path) == (
        "DELETE",
        f"/v1/files/{case.file['id']}",
    )


async def test_a_missing_file_raises_not_found(
    case: Case, client: Any, recorder: Recorder
) -> None:
    recorder.reply(
        Reply(status=404, json={"error": {"message": "No such file"}})
    )

    async with client:
        with pytest.raises(NotFoundError):
            await client.files.retrieve(ref(case))


async def test_download_streams_the_content(
    case: Case, client: Any, recorder: Recorder, tmp_path: Path
) -> None:
    recorder.reply(
        Reply(chunks=[PDF[:4], PDF[4:]], content_type="application/pdf")
    )
    target = tmp_path / "copy.pdf"

    async with client, client.files.download(ref(case)) as content:
        written = await content.save(target)

    assert (written, target.read_bytes()) == (len(PDF), PDF)
    assert recorder.last.raw_path == f"/v1/files/{case.file['id']}/content"


@pytest.mark.parametrize("call", ["retrieve", "delete", "download"])
async def test_another_providers_file_is_refused_before_sending(
    case: Case, client: Any, recorder: Recorder, call: str
) -> None:
    files: IFilesResource[Any] = client.files

    async with client:
        with pytest.raises(RequestValidationError) as caught:
            if call == "download":
                files.download(foreign(case))
            else:
                await getattr(files, call)(foreign(case))

    assert (caught.value.provider, caught.value.field) == (
        case.provider,
        "file",
    )
    assert recorder.requests == []


async def test_a_stored_file_is_sent_by_its_id(
    case: Case, client: Any, recorder: Recorder
) -> None:
    recorder.reply(Reply(json=case.message))
    turn = Turn.user(
        [
            TextPart(text="What is in these?"),
            DocumentPart(file=ref(case)),
            ImagePart(file=ref(case)),
        ]
    )

    async with client:
        await client.generation.create(model="m", input=[turn])

    body = json.loads(recorder.last.body)
    assert recorder.last.raw_path == case.message_path
    assert body[case.input_field][0]["content"][1:] == case.stored_parts


async def test_another_providers_file_in_a_part_is_refused(
    case: Case, client: Any, recorder: Recorder
) -> None:
    turn = Turn.user([DocumentPart(file=foreign(case))])

    async with client:
        with pytest.raises(RequestValidationError) as caught:
            await client.generation.create(model="m", input=[turn])

    assert (caught.value.provider, caught.value.field) == (
        case.provider,
        "input",
    )
    assert recorder.requests == []
