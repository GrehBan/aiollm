"""Block L: the documentation examples run against a local server.

Every ```` ```python ```` block of ``README.md`` and of the pages in
``docs/`` has a ``title="name.py"``; the tests find blocks by it, so a
block can move between pages, and a block without a title, or one no
test runs, fails ``test_every_example_is_tested``.
"""

import ast
import json
import mimetypes
import re
import sys
import types
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pytest

from aiollm import (
    APIConnectionError,
    APIRedirectError,
    APIStreamInterruptedError,
    BadRequestError,
    BaseErrorResolver,
    IncompleteReason,
    QuotaExceededError,
    RateLimitError,
    ResponseFormat,
    ServiceUnavailableError,
    ToolCall,
    Turn,
)
from aiollm_anthropic import AnthropicClient
from aiollm_anthropic.client import (
    session_factory as anthropic_session_factory,
)
from aiollm_openai import OpenAIClient
from aiollm_openai.client import session_factory
from aiollm.session.aiohttp import AiohttpSession
from aiollm.types.base import BaseAPIType
from aiollm.types.requests import GenerationRequest
from aiollm.testing import LocalAPI, Reply
from aiollm.testing.aiohttp import Recorder
from aiollm.testing.contract.generation import events as sse
from aiollm.testing.echo import EchoClient
from aiollm.testing.echo import session_factory as echo_session_factory
from tests.testing import echo_payloads
from tests.docs.payloads import OPENAI_OVERLOADED, openai_called_stream
from tests.docs.payloads import OPENAI_RESPONSE as OPENAI_EXAMPLE

ROOT = Path(__file__).parents[2]
HISTORICAL: Final = frozenset({"testing-plan.md", "sse-ndjson-plan.md"})
"""Records of old plans, not documentation: their code is not run."""

FENCE: Final = re.compile(
    r"^```python(?P<info>[^\n]*)\n(?P<code>.*?)^```", re.S | re.M
)
TITLE: Final = re.compile(r'title="([^"]+)"')


def documentation() -> list[Path]:
    """Return the README and every page of the site."""
    pages = sorted((ROOT / "docs").rglob("*.md"))
    return [ROOT / "README.md"] + [
        page for page in pages if page.name not in HISTORICAL
    ]


def collect() -> tuple[dict[str, tuple[Path, str]], list[str]]:
    """Find the python blocks by title; list what is wrong."""
    blocks: dict[str, tuple[Path, str]] = {}
    problems: list[str] = []

    for path in documentation():
        where = path.relative_to(ROOT).as_posix()

        for match in FENCE.finditer(path.read_text("utf-8")):
            title = TITLE.search(match["info"])

            if title is None:
                problems.append(f"{where}: a python block without a title")
                continue

            name = title.group(1)

            if name in blocks:
                problems.append(f"{where}: a second block {name!r}")

            blocks[name] = (path, match["code"])

    return blocks, problems


BLOCKS, PROBLEMS = collect()

QUICK_START: Final = "quick_start.py"
INHERITANCE: Final = "inheritance.py"
DATES: Final = "dates.py"
OPTIONAL_FIELDS: Final = "optional_fields.py"
QUERY: Final = "query.py"
ALIASES: Final = "aliases.py"
PATH_PARAMS: Final = "path_params.py"
UPLOADS: Final = "uploads.py"
STREAMING: Final = "streaming.py"
EVENTS: Final = "events.py"
RESTART: Final = "restart.py"
HANDLING: Final = "handling.py"
RESOLVER: Final = "resolver.py"
AUTH: Final = "auth.py"
CALL_OPTIONS: Final = "call_options.py"
RETRY_POLICY: Final = "retry_policy.py"
CLIENT: Final = "client.py"
def openai_message(text: str, status: str = "completed") -> dict[str, Any]:
    """An OpenAI output message with one text part."""
    return {
        "type": "message",
        "id": "msg_1",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def openai_response(**changes: Any) -> dict[str, Any]:
    """A whole OpenAI response: the documented example, changed."""
    return OPENAI_EXAMPLE | {"id": "resp_1", "model": "gpt-6-astra"} | changes


OPENAI_RESPONSES: Final = "openai_responses.py"
ANTHROPIC_MESSAGES: Final = "anthropic_messages.py"
SHARED_GENERATION: Final = "shared_generation.py"
GENERATION_PARTS: Final = "generation_parts.py"
PROVIDER_OPTIONS: Final = "provider_options.py"
TOOLS: Final = "tools.py"
TOOLS_RUNNER: Final = "tools_runner.py"
TOOLS_RUNNER_STREAM: Final = "tools_runner_stream.py"
TOOLS_STREAM: Final = "tools_stream.py"
TOOLS_MODEL: Final = "tools_model.py"
RESOURCES: Final = "resources.py"
OPENAI_FILES: Final = "openai_files.py"
ANTHROPIC_FILES: Final = "anthropic_files.py"
SHARED_FILES: Final = "shared_files.py"
BUILTIN_TOOLS: Final = "builtin_tools.py"
BUILTIN_TOOLS_PROVIDER: Final = "builtin_tools_provider.py"
BUILTIN_TOOLS_STREAM: Final = "builtin_tools_stream.py"
BUILTIN_TOOLS_CONTINUE: Final = "builtin_tools_continue.py"
BUILTIN_TOOLS_FILES: Final = "builtin_tools_files.py"

TESTED: Final = (
    QUICK_START,
    INHERITANCE,
    DATES,
    OPTIONAL_FIELDS,
    QUERY,
    ALIASES,
    PATH_PARAMS,
    UPLOADS,
    STREAMING,
    EVENTS,
    RESTART,
    HANDLING,
    RESOLVER,
    AUTH,
    CALL_OPTIONS,
    RETRY_POLICY,
    CLIENT,
    OPENAI_RESPONSES,
    ANTHROPIC_MESSAGES,
    SHARED_GENERATION,
    GENERATION_PARTS,
    PROVIDER_OPTIONS,
    TOOLS,
    TOOLS_RUNNER,
    TOOLS_RUNNER_STREAM,
    TOOLS_STREAM,
    TOOLS_MODEL,
    RESOURCES,
    OPENAI_FILES,
    ANTHROPIC_FILES,
    SHARED_FILES,
    BUILTIN_TOOLS,
    BUILTIN_TOOLS_PROVIDER,
    BUILTIN_TOOLS_STREAM,
    BUILTIN_TOOLS_CONTINUE,
    BUILTIN_TOOLS_FILES,
)
"""The title of every example a test below runs."""

PNG = b"\x89PNG\r\n\x1a\n"


class ModelList(BaseAPIType):
    data: list[str] = []


class Model(BaseAPIType):
    id: str


class Transcription(BaseAPIType):
    text: str


async def chunks() -> AsyncIterator[bytes]:
    yield b"RIFF"
    yield b"WAVE"


class Examples:
    """Runs example blocks in one module, like a reader pasting them."""

    def __init__(self, module: types.ModuleType) -> None:
        self.module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)

    async def run(self, *titles: str) -> list[Any]:
        """Run blocks in order; return values of top-level expressions."""
        values: list[Any] = []

        for title in titles:
            path, code = BLOCKS[title]

            for statement in ast.parse(code).body:
                values.append(await self._run(statement, path))

        return values

    async def _run(self, statement: ast.stmt, path: Path) -> Any:
        if is_asyncio_run(statement):
            # The caller awaits ``main()`` itself, inside the test's loop.
            return None

        is_expression = isinstance(statement, ast.Expr)
        tree: ast.Module | ast.Expression = (
            ast.Expression(statement.value)
            if isinstance(statement, ast.Expr)
            else ast.Module([statement], type_ignores=[])
        )
        code = compile(
            tree,
            str(path),
            "eval" if is_expression else "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        result = eval(code, self.module.__dict__)

        if isinstance(result, types.CoroutineType):
            result = await result

        return result


def is_asyncio_run(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value.func) == "asyncio.run"
    )


@pytest.fixture
def examples(
    session: AiohttpSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Examples]:
    # Pydantic needs the module of the classes in sys.modules.
    module = types.ModuleType("readme_examples")
    module.__dict__.update(
        ModelList=ModelList,
        Model=Model,
        Transcription=Transcription,
        data=PNG,
        chunks=chunks,
        session=session,
    )
    sys.modules[module.__name__] = module
    monkeypatch.chdir(tmp_path)

    try:
        yield Examples(module)
    finally:
        del sys.modules[module.__name__]


def test_every_example_is_tested() -> None:
    # A new ```python block needs a title and a test here.
    assert PROBLEMS == []
    assert sorted(BLOCKS) == sorted(TESTED)


def test_quick_start_ends_with_asyncio_run() -> None:
    last = ast.parse(BLOCKS[QUICK_START][1]).body[-1]

    assert ast.unparse(last) == "asyncio.run(main())"


async def test_quick_start(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await examples.run(QUICK_START)
    examples.API.base_url = api.base_url
    recorder.reply(Reply(json={"id": "c-1", "text": "Hello back!"}))

    await examples.main()

    request = recorder.last
    assert request.method == "POST"
    assert request.raw_path == "/v1/completions"
    assert request.headers["Authorization"] == "Bearer sk-..."
    assert request.body == (
        b'{"model":"example-1",'
        b'"messages":[{"role":"user","content":"Hello!"}]}'
    )
    assert capsys.readouterr().out == "Hello back!\n"


async def test_inherited_attributes(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, INHERITANCE)
    recorder.reply(Reply(json={"id": "s-1", "text": "Short."}))

    result = await session.send_request(examples.Summarize(text="Long."))

    assert result.text == "Short."
    assert recorder.last.method == "POST"
    assert recorder.last.raw_path == "/v1/summaries"
    assert recorder.last.body == b'{"text":"Long."}'


async def test_dates_and_their_formatter(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, DATES)
    recorder.reply(Reply(json={"id": "job-1"}))
    at = examples.datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    result = await session.send_request(
        examples.CreateJob(
            run_at=at,
            timeout=examples.datetime.timedelta(seconds=30),
            expires_at=at,
        )
    )

    assert result == Model(id="job-1")
    assert recorder.last.body == (
        b'{"run_at":"1767323045","timeout":"PT30S","expires_at":1767323045}'
    )


async def test_optional_fields(examples: Examples) -> None:
    *_, absent, given = await examples.run(QUICK_START, OPTIONAL_FIELDS)

    assert absent == {"query": "cats"}
    assert given == {"query": "cats", "cursor": None}
    nested = examples.Search(query="q", filters=[{"field": "f"}])
    assert nested.get_body_map() == {
        "query": "q",
        "filters": [{"field": "f"}],
    }


async def test_query_parameters(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, QUERY)
    recorder.reply(Reply(json={"data": ["example-1"]}))

    result = await session.send_request(examples.ListModels(limit=5))

    assert result == ModelList(data=["example-1"])
    assert recorder.last.raw_path == "/v1/models"
    assert recorder.last.query_string == "limit=5"


async def test_aliased_fields(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, ALIASES)
    recorder.reply(Reply(json={"data": []}))

    await session.send_request(examples.ListFiles(page_size=10))

    assert recorder.last.query_string == "pageSize=10"


async def test_path_parameters(examples: Examples) -> None:
    *_, request_path = await examples.run(QUICK_START, PATH_PARAMS)

    assert request_path == "/models/example-1"


async def test_uploads(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
    tmp_path: Path,
) -> None:
    (tmp_path / "voice.mp3").write_bytes(b"ID3audio")
    *_, from_path, from_bytes, from_stream = await examples.run(
        QUICK_START,
        UPLOADS,
    )
    recorder.reply(Reply(json={"text": "a"}), Reply(json={"text": "b"}))

    for method in (from_path, from_bytes):
        await session.send_request(method)

    parts = [request.parts[0] for request in recorder.requests]
    assert [
        (part.filename, part.content_type, part.data) for part in parts
    ] == [
        ("voice.mp3", "audio/mpeg", b"ID3audio"),
        ("image.png", "image/png", PNG),
    ]
    assert from_stream.file.filename == "s.wav"
    assert (
        from_stream.file.content_type == mimetypes.guess_file_type("s.wav")[0]
    )


async def test_stream_file_upload_is_not_retried(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    *_, from_stream = await examples.run(QUICK_START, UPLOADS)
    recorder.reply(Reply(status=503), Reply(json={"text": "late"}))

    with pytest.raises(ServiceUnavailableError):
        await session.send_request(from_stream)

    assert len(recorder.requests) == 1
    assert recorder.last.parts[0].data == b"RIFFWAVE"


async def test_streaming(
    examples: Examples,
    recorder: Recorder,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder.reply(
        Reply(chunks=[b"binary ", b"content"]),
        Reply(
            chunks=[b"Hi ", b"there ", "😀".encode()[:2], "😀".encode()[2:]],
            content_type="text/plain; charset=utf-8",
        ),
    )

    await examples.run(QUICK_START, STREAMING)

    download, generate = recorder.requests
    assert download.raw_path == "/v1/files/content"
    assert download.query_string == "file_id=f-1"
    assert (tmp_path / "result.bin").read_bytes() == b"binary content"
    assert generate.body == b'{"prompt":"Hi"}'
    assert capsys.readouterr().out == "Hi there 😀"


async def test_events(
    examples: Examples,
    recorder: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The streaming block runs first: it imports BaseStreamMethod.
    recorder.reply(
        Reply(chunks=[b"file"]),
        Reply(chunks=[b"Text. "], content_type="text/plain"),
        Reply(
            chunks=[
                b'event: text_delta\ndata: {"type": "text_delta", ',
                b'"text": "Hello"}\n\nevent: text_delta\ndata: ',
                b'{"type": "text_delta", "text": " there"}\n\n',
                b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
            ],
            content_type="text/event-stream",
        )
    )

    await examples.run(QUICK_START, STREAMING, EVENTS)

    assert recorder.last.raw_path == "/v1/messages"
    assert recorder.last.body == b'{"prompt":"Hi","stream":true}'
    assert capsys.readouterr().out == "Text. Hello there"
    assert examples.RawEvents.__response_format__ is ResponseFormat.SSE


def stream_replies() -> list[Reply]:
    """Replies to the requests the streaming block itself sends."""
    return [
        Reply(chunks=[b"file"]),
        Reply(chunks=[b"Text. "], content_type="text/plain"),
    ]


async def test_restart_after_a_broken_stream(
    examples: Examples,
    recorder: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder.reply(
        *stream_replies(),
        Reply(chunks=[b"Hel"], content_type="text/plain", abort=True),
        Reply(chunks=[b"Hello", b"!"], content_type="text/plain"),
    )
    await examples.run(QUICK_START, STREAMING, RESTART)
    capsys.readouterr()

    text = await examples.generate_text("Hi")

    assert text == "Hello!"
    assert len(recorder.requests) == 4


async def test_restart_gives_up_after_the_last_attempt(
    examples: Examples,
    recorder: Recorder,
) -> None:
    recorder.reply(
        *stream_replies(),
        Reply(chunks=[b"Hel"], content_type="text/plain", abort=True),
    )
    await examples.run(QUICK_START, STREAMING, RESTART)

    with pytest.raises(APIStreamInterruptedError) as info:
        await examples.generate_text("Hi", attempts=1)

    assert isinstance(info.value.__cause__, APIConnectionError)

    with pytest.raises(ValueError, match="at least 1"):
        await examples.generate_text("Hi", attempts=0)


async def test_error_handling(
    examples: Examples,
    recorder: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await examples.run(QUICK_START, HANDLING)
    recorder.reply(
        Reply(status=429, headers={"X-Request-Id": "req-1"}),
        Reply(status=400, json={"error": {"code": "bad_model"}}),
        Reply(json={"id": "c-1", "text": "Fine."}),
    )

    assert await examples.ask("Hi") is None
    with pytest.raises(BadRequestError):
        await examples.ask("Hi")
    assert await examples.ask("Hi") == "Fine."

    assert capsys.readouterr().out == (
        "Rate limited, request req-1\n"
        "API error 400, code bad_model\n"
    )
    assert len(recorder.requests) == 3


async def test_error_handling_of_transport_errors(
    examples: Examples,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await examples.run(QUICK_START, HANDLING)

    async def fail(method: Any) -> Any:
        raise APIRedirectError("moved")

    monkeypatch.setattr(examples.session, "send_request", fail)

    with pytest.raises(APIRedirectError):
        await examples.ask("Hi")

    assert capsys.readouterr().out == "Network error, transient: False\n"


async def test_error_resolver(examples: Examples) -> None:
    await examples.run(QUICK_START, RESOLVER)
    resolver = examples.ExampleErrorResolver()

    quota = resolver.resolve(
        status=429,
        body={"error": {"code": "insufficient_quota"}},
        headers={},
    )
    rate = resolver.resolve(
        status=429,
        body={"error": {"code": "rate_limit_exceeded"}},
        headers={},
    )

    assert type(quota) is QuotaExceededError
    assert quota.transient is False
    assert quota.provider == "example"
    assert type(rate) is RateLimitError


async def test_rotating_auth(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
) -> None:
    await examples.run(QUICK_START, QUERY, AUTH)
    auth = examples.RotatingAuth("sk-old")
    recorder.reply(Reply(json={"data": []}), Reply(json={"data": []}))

    async with AiohttpSession(api, auth, BaseErrorResolver()) as session:
        await session.send_request(examples.ListModels())
        auth.token = "sk-new"
        await session.send_request(examples.ListModels())

    assert [
        request.headers["Authorization"] for request in recorder.requests
    ] == ["Bearer sk-old", "Bearer sk-new"]


async def test_call_options(
    examples: Examples,
    recorder: Recorder,
) -> None:
    await examples.run(QUICK_START, CALL_OPTIONS)
    recorder.reply(Reply(json={"id": "c-1", "text": "Hi."}))

    result = await examples.complete("Hello!", "key-1")

    assert result == examples.Completion(id="c-1", text="Hi.")
    assert recorder.last.raw_path == "/v1/completions"
    assert recorder.last.headers["x-beta-features"] == "long-context"
    assert recorder.last.headers["Idempotency-Key"] == "key-1"


async def test_retry_policy(
    examples: Examples,
    recorder: Recorder,
) -> None:
    await examples.run(QUICK_START, RETRY_POLICY)
    recorder.reply(
        *[Reply(status=503)] * 5,
        Reply(json={"id": "c-1", "text": "Done."}),
    )

    result = await examples.complete_patiently("Hello!")

    assert result == examples.Completion(id="c-1", text="Done.")
    assert len(recorder.requests) == 6


async def test_retry_policy_leaves_a_rate_limit_to_the_caller(
    examples: Examples,
    recorder: Recorder,
) -> None:
    await examples.run(QUICK_START, RETRY_POLICY)
    recorder.reply(Reply(status=429, headers={"Retry-After": "1"}))

    with pytest.raises(RateLimitError) as caught:
        await examples.complete_patiently("Hello!")

    assert caught.value.transient is True
    assert len(recorder.requests) == 1

async def test_openai_client(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, QUERY, CLIENT)
    monkeypatch.setattr(session_factory, "api", api)
    recorder.reply(Reply(json={"data": ["gpt-x"]}))

    assert await examples.list_models() == ModelList(data=["gpt-x"])
    assert recorder.last.raw_path == "/v1/models"
    assert recorder.last.headers["Authorization"] == "Bearer sk-..."


async def test_openai_responses(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, CLIENT, OPENAI_RESPONSES)
    delta = {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 0,
        "logprobs": [],
    }
    response = openai_response(output=[openai_message("Hello!")])
    completed = {
        "type": "response.completed",
        "sequence_number": 2,
        "response": response | {"status": "completed"},
    }
    recorder.reply(
        Reply(json=response),
        Reply(
            chunks=[
                f"data: {json.dumps(data)}\n\n".encode()
                for data in [
                    delta | {"delta": "He"},
                    delta | {"delta": "y"},
                    completed,
                ]
            ],
            content_type="text/event-stream",
        ),
    )

    assert await examples.ask(session) == "Hello!"
    assert await examples.ask_streamed(session) == "Hey"
    assert recorder.requests[-2].raw_path == "/v1/responses"
    assert json.loads(recorder.requests[-2].body) == {
        "model": "gpt-6-astra",
        "input": "Hi",
        "instructions": "Be brief",
    }


async def test_anthropic_messages(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, ANTHROPIC_MESSAGES)
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    delta = {"type": "content_block_delta", "index": 0}
    recorder.reply(
        Reply(json=message),
        Reply(
            chunks=[
                f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"
                .encode()
                for data in [
                    delta | {"delta": {"type": "text_delta", "text": "He"}},
                    delta | {"delta": {"type": "signature_delta", "signature": "s"}},
                    delta | {"delta": {"type": "text_delta", "text": "y"}},
                ]
            ]
            + [b'event: message_stop\ndata: {"type": "message_stop"}\n\n'],
            content_type="text/event-stream",
        ),
    )

    assert await examples.ask_claude(session) == "Hello!"
    assert await examples.ask_claude_streamed(session) == "Hey"
    first, second = recorder.requests
    assert first.raw_path == "/v1/messages"
    assert first.headers["anthropic-version"] == "2026-10-21"
    assert json.loads(first.body) == {
        "model": "claude-opus-5-5",
        "max_tokens": 1024,
        "system": "Be brief",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert json.loads(second.body)["stream"] is True


async def test_shared_generation(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, CLIENT, SHARED_GENERATION)
    monkeypatch.setattr(session_factory, "api", api)
    delta = {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 0,
        "logprobs": [],
    }
    completed = {
        "type": "response.completed",
        "sequence_number": 2,
        "response": openai_response(id="resp_2", output=[]),
    }
    usage = OPENAI_EXAMPLE["usage"] | {
        "input_tokens": 5,
        "output_tokens": 64,
        "total_tokens": 69,
    }
    recorder.reply(
        Reply(
            json=openai_response(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
                output=[openai_message("Hel", "incomplete")],
                usage=usage,
            )
        ),
        Reply(
            chunks=[
                f"data: {json.dumps(delta | {'delta': 'He'})}\n\n".encode(),
                f"data: {json.dumps(delta | {'delta': 'y'})}\n\n".encode(),
                f"data: {json.dumps(completed)}\n\n".encode(),
            ],
            content_type="text/event-stream",
        ),
    )

    async with examples.OpenAIClient("sk-...") as client:
        response = await examples.generate(client)
        text, completed = await examples.generate_streamed(client)

    assert text == "Hey"
    assert (completed.id, completed.model) == ("resp_2", "gpt-6-astra")

    assert (response.id, response.text) == ("resp_1", "Hel")
    assert response.incomplete_reason is IncompleteReason.MAX_TOKENS
    assert response.raw_incomplete_reason == "max_output_tokens"
    assert response.usage is not None
    assert response.usage.total_tokens == 69
    assert response.raw.status == "incomplete"
    first, second = recorder.requests
    assert first.raw_path == second.raw_path == "/v1/responses"
    assert json.loads(first.body) == {
        "model": "gpt-6-astra",
        "input": "Hi",
        "instructions": "Be brief",
        "max_output_tokens": 64,
    }
    assert json.loads(second.body) == {
        "model": "gpt-6-astra",
        "input": "Hi",
        "stream": True,
    }


async def test_generation_parts(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await examples.run(QUICK_START, CLIENT, SHARED_GENERATION, GENERATION_PARTS)
    monkeypatch.setattr(session_factory, "api", api)
    recorder.reply(Reply(json=openai_response()))
    (tmp_path / "chart.png").write_bytes(b"\x89PNG")
    (tmp_path / "report.pdf").write_bytes(b"%PDF")

    async with examples.OpenAIClient("sk-...") as client:
        await examples.describe(
            client, tmp_path / "chart.png", tmp_path / "report.pdf"
        )

    assert json.loads(recorder.last.body)["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Does the chart match the report?",
                },
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw==",
                    "detail": "auto",
                },
                {
                    "type": "input_file",
                    "file_data": "data:application/pdf;base64,JVBERg==",
                    "filename": "report.pdf",
                },
                {
                    "type": "input_image",
                    "image_url": "https://example.com/logo.png",
                    "detail": "auto",
                },
            ],
        }
    ]


async def test_shared_generation_continues_a_conversation(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, CLIENT, SHARED_GENERATION)
    monkeypatch.setattr(session_factory, "api", api)
    answer = openai_response(output=[openai_message("Hello!")])
    recorder.reply(Reply(json=answer), Reply(json=answer))

    async with examples.OpenAIClient("sk-...") as client:
        response = await examples.chat(client)

    assert response.text == "Hello!"
    assert json.loads(recorder.last.body)["input"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Tell me more"},
    ]


async def test_shared_generation_works_with_anthropic(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, CLIENT, SHARED_GENERATION)
    monkeypatch.setattr(anthropic_session_factory, "api", api)
    recorder.reply(
        Reply(
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": "Hel"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 5, "output_tokens": 64},
            }
        )
    )

    async with AnthropicClient("sk-ant-...") as client:
        response = await examples.generate(client)

    assert response.text == "Hel"
    assert response.incomplete_reason is IncompleteReason.MAX_TOKENS
    assert json.loads(recorder.last.body) == {
        "model": "gpt-6-astra",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 64,
        "system": "Be brief",
    }


async def test_tools(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, TOOLS)
    monkeypatch.setattr(session_factory, "api", api)
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
        "status": "completed",
    }
    recorder.reply(
        Reply(json=openai_response(output=[call])),
        Reply(json=openai_response(output=[openai_message("15 C in Paris")])),
    )

    async with OpenAIClient("sk-...") as client:
        response = await examples.ask_with_tools(client, "Weather in Paris?")

    assert response.text == "15 C in Paris"
    first, second = recorder.requests
    assert json.loads(first.body)["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Current weather in a city.",
            "parameters": {
                "properties": {
                    "city": {"title": "City", "type": "string"},
                },
                "required": ["city"],
                "type": "object",
                "additionalProperties": False,
            },
            "strict": None,
        }
    ]
    assert json.loads(second.body)["input"] == [
        {"role": "user", "content": "Weather in Paris?"},
        call,
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"city": "Paris", "celsius": 15}',
        },
    ]


async def test_tools_runner(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, TOOLS_RUNNER)
    monkeypatch.setattr(session_factory, "api", api)
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
        "status": "completed",
    }
    recorder.reply(
        Reply(json=openai_response(output=[call])),
        Reply(json=openai_response(output=[openai_message("15 C in Paris")])),
    )

    async with OpenAIClient("sk-...") as client:
        response = await examples.ask_with_runner(client, "Weather in Paris?")

    assert response.text == "15 C in Paris"
    assert json.loads(recorder.requests[1].body)["input"] == [
        {"role": "user", "content": "Weather in Paris?"},
        call,
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"city": "Paris", "celsius": 15}',
        },
    ]


async def test_tools_runner_stream(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, TOOLS_RUNNER_STREAM)
    monkeypatch.setattr(session_factory, "api", api)
    recorder.reply(sse(openai_called_stream()))

    async with OpenAIClient("sk-...") as client:
        completed, results = await examples.stream_round(
            client, [Turn.user("Weather in Paris?")]
        )

    assert [call.name for call in completed.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert [(result.call_id, result.output) for result in results] == [
        ("call_1", '{"city": "Paris", "celsius": 15}'),
        ("call_2", "3 PM"),
    ]


async def test_tools_model(examples: Examples) -> None:
    await examples.run(QUICK_START, TOOLS, TOOLS_STREAM, TOOLS_MODEL)

    assert examples.CITY_WEATHER.parameters == {
        "type": "object",
        "title": "City",
        "properties": {"city": {"type": "string", "title": "City"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    good = examples.answer(
        ToolCall(id="c1", name="get_weather", arguments='{"city": "Paris"}')
    )
    bad = examples.answer(
        ToolCall(id="c2", name="get_weather", arguments='{"town": "Paris"}')
    )

    assert (good.call_id, good.output) == (
        "c1",
        '{"city": "Paris", "celsius": 15}',
    )
    assert bad.call_id == "c2"
    assert "city" in json.loads(bad.output)["error"]


async def test_tools_stream(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, TOOLS, TOOLS_STREAM)
    monkeypatch.setattr(session_factory, "api", api)
    recorder.reply(sse(openai_called_stream()))
    conversation = [Turn.user("Weather in Paris?")]

    async with OpenAIClient("sk-...") as client:
        items = await examples.stream_round(client, conversation)

    *_, weather, time = items
    assert (weather.call_id, weather.output) == (
        "call_1",
        '{"city": "Paris", "celsius": 15}',
    )
    assert (time.call_id, time.output) == (
        "call_2",
        '{"error": "unknown tool get_time"}',
    )


async def test_tools_stream_fails_after_a_done_call(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, TOOLS, TOOLS_STREAM)
    monkeypatch.setattr(session_factory, "api", api)
    recorder.reply(sse([*openai_called_stream()[:-1], OPENAI_OVERLOADED]))

    async with OpenAIClient("sk-...") as client:
        with pytest.raises(ServiceUnavailableError):
            await examples.stream_round(client, [Turn.user("Weather?")])


async def test_provider_options(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, PROVIDER_OPTIONS)
    monkeypatch.setattr(session_factory, "api", api)
    recorder.reply(
        Reply(json=openai_response(output=[]))
    )

    async with examples.OpenAIClient("sk-...") as client:
        response = await examples.generate_unstored(client)

    assert response.id == "resp_1"
    assert json.loads(recorder.last.body) == {
        "model": "gpt-6-astra",
        "input": "Hi",
        "store": False,
        "temperature": 0.2,
    }


async def test_resources(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    recorder.reply(
        *stream_replies(),
        Reply(
            chunks=[b'data: {"type": "message_stop"}\n\n'],
            content_type="text/event-stream",
        ),
    )
    await examples.run(QUICK_START, STREAMING, EVENTS, RESOURCES)
    client = examples.ExampleClient(session)
    request = GenerationRequest(model="example-1", input="Hi")
    recorder.reply(
        Reply(json={"id": "c-1", "text": "Hello!"}),
        Reply(
            chunks=[
                b'data: {"type": "text_delta", "text": "He"}\n\n',
                b'data: {"type": "text_delta", "text": "y"}\n\n',
                b'data: {"type": "message_stop"}\n\n',
            ],
            content_type="text/event-stream",
        ),
    )

    response = await client.generation.create(request)
    async with client.generation.stream(request) as events:
        received = [event async for event in events]

    assert response.text == "Hello!"
    assert response.id == "c-1"
    assert response.raw.text == "Hello!"
    assert [event.type for event in received] == [
        "text_delta",
        "text_delta",
        "completed",
    ]
    assert [getattr(event, "text", None) for event in received] == [
        "He",
        "y",
        None,
    ]
    assert recorder.requests[-2].body == (
        b'{"model":"example-1",'
        b'"messages":[{"role":"user","content":"Hi"}]}'
    )


async def test_openai_files(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
    tmp_path: Path,
) -> None:
    await examples.run(QUICK_START, CLIENT, OPENAI_FILES)
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    file = {
        "id": "file-1",
        "object": "file",
        "bytes": 8,
        "created_at": 1,
        "expires_at": 3601,
        "filename": "report.pdf",
        "purpose": "user_data",
        "status": "processed",
    }
    page = {"object": "list", "data": [file], "first_id": "file-1"}
    recorder.reply(
        Reply(json=file),
        Reply(json=openai_response(output=[openai_message("A report")])),
        Reply(json={"id": "file-1", "object": "file", "deleted": True}),
        Reply(json=page | {"last_id": "file-1", "has_more": True}),
        Reply(json=page | {"last_id": "file-2", "has_more": False}),
    )

    assert await examples.summarize(session, path) == "A report"
    assert await examples.all_file_ids(session) == ["file-1", "file-1"]
    upload, ask, delete, first, second = recorder.requests
    assert [(part.name, part.data) for part in upload.parts] == [
        ("file", b"%PDF-1.4"),
        ("purpose", b"user_data"),
        ("expires_after[anchor]", b"created_at"),
        ("expires_after[seconds]", b"3600"),
    ]
    assert upload.parts[0].filename == "report.pdf"
    assert json.loads(ask.body)["input"][0]["content"] == [
        {"type": "input_file", "file_id": "file-1"},
        {"type": "input_text", "text": "Summarize it"},
    ]
    assert (delete.method, delete.raw_path) == ("DELETE", "/v1/files/file-1")
    assert dict(first.query) == {"limit": "100"}
    assert dict(second.query) == {"limit": "100", "after": "file-1"}


async def test_anthropic_files(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
    tmp_path: Path,
) -> None:
    await examples.run(QUICK_START, ANTHROPIC_FILES)
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    file = {
        "id": "file_1",
        "type": "file",
        "filename": "report.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 8,
        "created_at": "2026-11-19T00:00:00Z",
        "downloadable": False,
    }
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": "A report"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    recorder.reply(
        Reply(json=file),
        Reply(json=message),
        Reply(json={"id": "file_1", "type": "file_deleted"}),
        Reply(json={"data": [file], "next_page": "page_2"}),
        Reply(json={"data": [file], "next_page": None}),
        Reply(chunks=[b"chart", b" bytes"]),
    )
    target = tmp_path / "chart.png"

    assert await examples.ask_about(session, path) == "A report"
    assert await examples.all_claude_file_ids(session) == ["file_1"] * 2
    assert await examples.download(session, "file_9", target) == 11
    assert target.read_bytes() == b"chart bytes"
    upload, ask, delete, first, second, content = recorder.requests
    assert upload.headers["anthropic-version"] == "2026-10-21"
    assert [(part.name, part.data) for part in upload.parts] == [
        ("file", b"%PDF-1.4"),
        ("expires_in_seconds", b"3600"),
    ]
    assert json.loads(ask.body)["messages"][0]["content"] == [
        {"type": "document", "source": {"type": "file", "file_id": "file_1"}},
        {"type": "text", "text": "Summarize it"},
    ]
    assert (delete.method, delete.raw_path) == ("DELETE", "/v1/files/file_1")
    assert dict(first.query) == {"limit": "100"}
    assert dict(second.query) == {"limit": "100", "page": "page_2"}
    assert content.raw_path == "/v1/files/file_9/content"


def claude_message(text: str) -> dict[str, Any]:
    """A finished Anthropic message with one text block."""
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }


async def test_shared_files(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await examples.run(SHARED_FILES)
    monkeypatch.setattr(anthropic_session_factory, "api", api)
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    file = {
        "id": "file_1",
        "type": "file",
        "filename": "report.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 8,
        "created_at": "2026-11-19T00:00:00Z",
        "expires_at": "2026-11-19T01:00:00Z",
    }
    old = file | {"id": "file_0", "created_at": "2026-11-01T00:00:00Z"}
    recorder.reply(
        Reply(json=file),
        Reply(json=claude_message("A report")),
        Reply(json=claude_message("Two pages")),
        Reply(json={"id": "file_1", "type": "file_deleted"}),
        Reply(json={"data": [file], "next_page": "page_2"}),
        Reply(json={"data": [old], "next_page": None}),
    )

    async with examples.AnthropicClient("sk-ant-...") as client:
        answers = await examples.ask_about(
            client, "claude-opus-5-5", path, ["What is it?", "How long?"]
        )
        old_files = await examples.files_before(
            client, datetime(2026, 11, 10, tzinfo=UTC)
        )

    assert answers == ["A report", "Two pages"]
    assert [ref.id for ref in old_files] == ["file_0"]
    upload, first, second, delete = recorder.requests[:4]
    assert [(part.name, part.data) for part in upload.parts] == [
        ("file", b"%PDF-1.4"),
        ("expires_in_seconds", b"3600"),
    ]
    for ask, question in ((first, "What is it?"), (second, "How long?")):
        assert json.loads(ask.body)["messages"][0]["content"] == [
            {
                "type": "document",
                "source": {"type": "file", "file_id": "file_1"},
            },
            {"type": "text", "text": question},
        ]
    assert (delete.method, delete.raw_path) == ("DELETE", "/v1/files/file_1")


@pytest.fixture
def echo(
    api: LocalAPI, monkeypatch: pytest.MonkeyPatch
) -> EchoClient:
    """Build the client of the echo provider, sent to the local server.

    The built-in tool examples are typed on ``IGenerationClient``, so they
    run on the one provider of the test kit that runs built-in tools.
    """
    monkeypatch.setattr(echo_session_factory, "api", api)
    return EchoClient("sk-...")


async def test_builtin_tools(
    examples: Examples, recorder: Recorder, echo: EchoClient
) -> None:
    recorder.reply(Reply(json=echo_payloads.SEARCHED))
    await examples.run(BUILTIN_TOOLS)

    async with echo:
        response = await examples.research(echo)

    assert [c.kind.value for c in response.server_tool_calls] == ["web_search"]
    assert [c.url for c in response.citations] == ["https://example.com/paris"]
    assert json.loads(recorder.last.body)["tools"] == [
        {
            "type": "web_search",
            "allowed_domains": ["example.com"],
            "user_location": {"country": "FR", "city": "Paris"},
        },
        {"type": "code_execution"},
    ]


async def test_builtin_tools_provider(examples: Examples) -> None:
    await examples.run(BUILTIN_TOOLS_PROVIDER)

    tool = examples.file_search
    assert tool.provider == "openai"
    assert tool.definition["type"] == "file_search"


async def test_builtin_tools_stream(
    examples: Examples,
    recorder: Recorder,
    echo: EchoClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder.reply(sse(echo_payloads.SEARCHED_STREAM))
    await examples.run(BUILTIN_TOOLS_STREAM)

    async with echo:
        completed = await examples.search_streamed(echo)

    assert completed.server_tool_calls[0].id == "search_1"
    assert capsys.readouterr().out.splitlines() == [
        "started web_search",
        "done web_search",
    ]


async def test_builtin_tools_continue(
    examples: Examples, recorder: Recorder, echo: EchoClient
) -> None:
    recorder.reply(
        Reply(json=echo_payloads.PAUSED),
        Reply(json=echo_payloads.SEARCHED),
    )
    await examples.run(BUILTIN_TOOLS_CONTINUE)

    async with echo:
        response = await examples.research_to_the_end(echo, "Weather?")

    assert response.text == "It is sunny."
    first, second = recorder.requests
    assert json.loads(second.body)["input"] == [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "output": echo_payloads.PAUSED_ITEMS},
    ]
    assert len(recorder.requests) == 2
    assert "tools" in json.loads(first.body)


async def test_builtin_tools_files(
    examples: Examples, recorder: Recorder, echo: EchoClient, tmp_path: Path
) -> None:
    table = tmp_path / "sales.csv"
    table.write_bytes(b"month,sales\n")
    stored = {
        "id": "file_1",
        "filename": "sales.csv",
        "bytes": 12,
        "created_at": 1_800_000_000,
    }
    recorder.reply(
        Reply(json=stored),
        Reply(json=echo_payloads.CODED),
        Reply(chunks=[b"\x89PNG"], content_type="image/png"),
    )
    await examples.run(BUILTIN_TOOLS_FILES)

    async with echo:
        await examples.chart(echo, table, tmp_path)

    assert (tmp_path / "chart.png").read_bytes() == b"\x89PNG"
    _, ask, download = recorder.requests
    assert json.loads(ask.body)["tools"] == [
        {"type": "code_execution", "file_ids": ["file_1"]}
    ]
    assert download.raw_path == (
        "/v1/containers/cntr_1/files/file_made_1/content"
    )

