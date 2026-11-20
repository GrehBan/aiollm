"""Shared files behave the same for every provider.

The scenarios of ``aiollm.testing.contract.files`` run against
``OpenAIClient.files`` and ``AnthropicClient.files`` on the local
server, with the documented results of each provider's Files API; only
the wire format differs per provider, the shared results and errors
must not.
"""

from __future__ import annotations

from datetime import UTC, datetime

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
from aiollm.testing.contract.files import FilesCase, FilesContract
from tests.providers import test_anthropic_messages as anthropic
from tests.providers import test_openai_responses as openai
from tests.providers.test_files_methods import documented

OPENAI_FILE = documented("openai/files", "createFile")
ANTHROPIC_FILE = documented("anthropic/files", "createFile")

OPENAI = FilesCase(
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
    bare_upload_parts=["file", "purpose"],
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

ANTHROPIC = FilesCase(
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
    bare_upload_parts=["file"],
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


class TestOpenAIFiles(FilesContract):
    case = OPENAI


class TestAnthropicFiles(FilesContract):
    case = ANTHROPIC
