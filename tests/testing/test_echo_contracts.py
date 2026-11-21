"""The shared scenarios of ``aiollm.testing.contract`` on the echo provider.

Echo is built from the extension API alone, so these runs show that the
core, without any real provider, passes every scenario.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aiollm.testing.contract.clients import ClientCase, ClientContract
from aiollm.testing.contract.files import FilesCase, FilesContract
from aiollm.testing.contract.generation import (
    SCHEMA,
    BuiltinCase,
    GenerationCase,
    GenerationContract,
)
from aiollm.testing.echo import (
    ECHO_API,
    PROVIDER,
    EchoClient,
    EchoErrorResolver,
    session_factory,
)
from aiollm.testing.echo.methods import StreamGenerate
from aiollm.testing.echo.types import EchoFile
from aiollm.types.responses import TokenUsage
from tests.testing import echo_payloads as echo

ECHO_GENERATION = GenerationCase(
    provider=PROVIDER,
    client=EchoClient,
    factory=session_factory,
    path="/v1/echo",
    response=echo.RESPONSE,
    text="Hi there.",
    usage=TokenUsage(input_tokens=12, output_tokens=4, total_tokens=16),
    truncated=echo.RESPONSE | {"stop": "max_tokens"},
    refused=echo.RESPONSE
    | {"output": [{"type": "refusal", "text": "I can't."}], "stop": "refusal"},
    refusal="I can't.",
    stream=echo.STREAM,
    deltas=["Hello", "!"],
    stream_usage=TokenUsage(input_tokens=12, output_tokens=2, total_tokens=14),
    overloaded={
        "type": "error",
        "error": {"code": "overloaded", "message": "Overloaded"},
    },
    quota={
        "type": "error",
        "error": {"code": "insufficient_quota", "message": "No quota"},
    },
    bad_request={"error": {"code": "invalid_request", "message": "Bad"}},
    raw_stream=lambda: StreamGenerate(
        model="echo-1", input=[{"role": "user", "content": "Hi"}]
    ),
    options={"temperature": 0.2, "stop": ["END"]},
    other_options={"temperature": 0.7},
    bad_option={"temperature": 5},
    effort={"effort": "high"},
    output_schema={
        "effort": "high",
        "output_schema": {"type": "object", "required": ["n"]},
    },
    tools={
        "tools": [
            {
                "name": "get_weather",
                "description": "Current weather",
                "parameters": SCHEMA,
                "strict": True,
            },
            {
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    },
    forced={"tool_choice": {"name": "get_time"}},
    only_parallel={"parallel_tool_calls": True},
    called=echo.CALLED,
    called_stream=echo.called_stream(),
    cut_off={"stop": "max_tokens"},
    input_field="input",
    continued=[
        {"role": "assistant", "output": echo.CALLED["output"]},
        *echo.RESULTS,
    ],
    continued_shared=[
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
                for item in echo.CALL_ITEMS
            ],
        },
        *echo.RESULTS,
    ],
    continued_streamed=[
        {"role": "assistant", "output": echo.CALL_ITEMS},
        *echo.RESULTS,
    ],
    parts={
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in these?"},
            {"type": "image", "data": "iVBORw0K", "media_type": "image/png"},
            {"type": "image", "url": "https://example.com/cat.jpg"},
            {
                "type": "document",
                "data": "JVBERi0xLjc=",
                "media_type": "application/pdf",
                "filename": "report.pdf",
            },
            {
                "type": "document",
                "data": "JVBERi0xLjc=",
                "media_type": "application/pdf",
            },
            {"type": "document", "url": "https://example.com/report.pdf"},
        ],
    },
    builtin=BuiltinCase(
        tools={
            "tools": [
                {
                    "type": "web_search",
                    "allowed_domains": ["example.com"],
                    "user_location": {"country": "FR", "city": "Paris"},
                },
                {"type": "code_execution"},
            ]
        },
        mixed={
            "tools": [
                {
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {}},
                },
                {"type": "web_search"},
            ]
        },
        searched=echo.SEARCHED,
        searched_stream=echo.SEARCHED_STREAM,
        failed=echo.FAILED,
        paused=echo.PAUSED,
        continued=[{"role": "assistant", "output": echo.PAUSED_ITEMS}],
    ),
    after_results=[
        {"role": "tool", "call_id": "call_1", "output": "noon"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in these?"},
                {
                    "type": "image",
                    "data": "iVBORw0K",
                    "media_type": "image/png",
                },
            ],
        },
    ],
)


class TestEchoGeneration(GenerationContract):
    case = ECHO_GENERATION


ECHO_FILE: dict[str, Any] = {
    "id": "file_1",
    "filename": "report.pdf",
    "bytes": 2048,
    "created_at": 1_800_000_000,
    "expires_at": 1_800_003_600,
}
"""A stored file."""

ECHO_FILES = FilesCase(
    provider=PROVIDER,
    client=EchoClient,
    factory=session_factory,
    raw_type=EchoFile,
    file=ECHO_FILE,
    size=2048,
    created_at=datetime(2027, 1, 15, 8, 0, tzinfo=UTC),
    expires_at=datetime(2027, 1, 15, 9, 0, tzinfo=UTC),
    deleted={"id": "file_1", "deleted": True},
    page={"data": [ECHO_FILE], "next": "file_1"},
    cursor=("after", "file_1"),
    last_page={"data": [ECHO_FILE | {"id": "file_0"}], "next": None},
    upload_parts={"expires_in_seconds": b"3600"},
    bare_upload_parts=["file"],
    options=(
        {"purpose": "vision"},
        {"purpose": b"vision", "expires_in_seconds": b"3600"},
    ),
    message_path="/v1/echo",
    message=echo.RESPONSE,
    input_field="input",
    stored_parts=[
        {"type": "document", "file_id": "file_1"},
        {"type": "image", "file_id": "file_1"},
    ],
)


class TestEchoFiles(FilesContract):
    case = ECHO_FILES


class TestEchoClient(ClientContract):
    case = ClientCase(
        client=EchoClient,
        provider=PROVIDER,
        api=ECHO_API,
        url=("/echo", "https://echo.invalid/v1/echo"),
        resolver=EchoErrorResolver,
        env="ECHO_API_KEY",
    )
