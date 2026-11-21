"""Scenarios every provider's ``client.generation`` passes.

A provider's tests subclass ``GenerationContract`` with a
``GenerationCase`` of its wire format; every scenario then runs against
the provider's client on the local server (``aiollm.testing.plugin``
fixtures), with the provider's own payloads. Only the wire format
differs per provider: the shared results and errors must not.

Example:
    >>> from aiollm.testing.contract import (
    ...     GenerationCase,
    ...     GenerationContract,
    ... )
    >>> EXAMPLE = GenerationCase(provider=PROVIDER, client=ExampleClient, ...)
    >>> class TestExampleGeneration(GenerationContract):
    ...     case = EXAMPLE
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final

import pytest
from pydantic import MISSING

from aiollm.enums import (
    CitationKind,
    Effort,
    IncompleteReason,
    Provider,
    ServerToolKind,
    ServerToolPhase,
    ServerToolStatus,
    ToolChoice,
)
from aiollm.exceptions import RequestValidationError
from aiollm.http.exceptions.status import (
    BadRequestError,
    QuotaExceededError,
    ServiceUnavailableError,
)
from aiollm.http.exceptions.transport import (
    APIConnectionError,
    APIStreamInterruptedError,
    APITimeoutError,
)
from aiollm.resources.generation import GenerationResource
from aiollm.session.retry import RetryPolicy
from aiollm.testing.doubles import TEST_TOKEN
from aiollm.testing.recording import Reply
from aiollm.types.requests import (
    CodeExecution,
    ContentPart,
    DocumentPart,
    ForcedTool,
    GenerationRequest,
    ImagePart,
    JSONSchema,
    NativeOutput,
    ProviderTool,
    TextPart,
    Tool,
    ToolCall,
    ToolCallTurn,
    ToolResult,
    Turn,
    UserLocation,
    WebSearch,
)
from aiollm.types.responses import (
    CompletedEvent,
    ServerToolCallEvent,
    TextDeltaEvent,
    TokenUsage,
    ToolCallDeltaEvent,
    ToolCallDoneEvent,
    ToolCallStartEvent,
)

if TYPE_CHECKING:
    from aiollm.methods.base import BaseStreamMethod
    from aiollm.response.stream import Stream
    from aiollm.session.factory import SessionFactory
    from aiollm.testing.aiohttp import Recorder
    from aiollm.testing.doubles import LocalAPI

REQUEST: Final = GenerationRequest(
    model="model-1",
    input="Hi",
    instructions="Be brief",
    max_output_tokens=64,
)
"""The request of most scenarios."""

SCHEMA: Final[JSONSchema] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
"""Parameters of the tools of the scenarios."""

CALLS: Final[list[tuple[str, str, dict[str, Any]]]] = [
    ("call_1", "get_weather", {"city": "Paris"}),
    ("call_2", "get_time", {}),
]
"""Two calls the model makes in one response: id, tool, arguments."""

TOOLS: Final = [
    Tool(
        name="get_weather",
        description="Current weather",
        parameters=SCHEMA,
        strict=True,
    ),
    Tool(name="get_time", parameters={"type": "object", "properties": {}}),
]
"""A strict tool with a description, and a bare one."""

PNG: Final = b"\x89PNG\r\n"
"""The bytes of the image part of ``PARTS``."""

PDF: Final = b"%PDF-1.7"
"""The bytes of the document parts of ``PARTS``."""

PARTS: Final[list[ContentPart]] = [
    TextPart(text="What is in these?"),
    ImagePart(data=PNG, media_type="image/png"),
    ImagePart(url="https://example.com/cat.jpg"),
    DocumentPart(data=PDF, filename="report.pdf"),
    DocumentPart(data=PDF),
    DocumentPart(url="https://example.com/report.pdf"),
]
"""A user turn of every kind of part and source."""


BUILTIN_TOOLS: Final = [
    WebSearch(
        allowed_domains=["example.com"],
        user_location=UserLocation(country="FR", city="Paris"),
    ),
    CodeExecution(),
]
"""The built-in tools of the scenarios: a search with both its options,
and a code execution."""

SEARCH_ID: Final = "search_1"
"""The id of the web search call the model makes in the scenarios."""

SEARCH_QUERIES: Final = ["weather in Paris"]
"""What that search looks for."""

CITATIONS: Final = [
    (CitationKind.URL, "https://example.com/paris", "Paris weather")
]
"""The citations of the text the search leads to: kind, url, title."""


@dataclass(frozen=True)
class BuiltinCase:
    """The built-in tool scenarios of one provider.

    Every ``dict`` is in the provider's wire format. A provider that
    runs built-in tools gives its ``GenerationCase`` one of these; a
    case without it skips the scenarios.

    Attributes:
        tools: What ``BUILTIN_TOOLS`` become in the body: the keys a
            request with them (and ``ToolChoice.AUTO``) sets.
        mixed: What a function tool, ``TOOLS[1]``, followed by
            ``WebSearch()`` become: the function tool goes first.
        searched: A finished response of one web search, call
            ``SEARCH_ID`` for ``SEARCH_QUERIES``, with the text of
            ``GenerationCase.text`` and the ``CITATIONS``.
        searched_stream: A finished stream of the same.
        failed: A finished response whose web search failed.
        paused: A response that paused its turn in the middle of a
            search, without text; its ``native`` output goes back.
        continued: The body's conversation after its first turn, when
            ``paused`` comes back with the user turn before it.
    """

    tools: dict[str, Any]
    mixed: dict[str, Any]
    searched: dict[str, Any]
    searched_stream: list[dict[str, Any]]
    failed: dict[str, Any]
    paused: dict[str, Any]
    continued: list[Any]


@dataclass(frozen=True)
class GenerationCase:
    """One provider in the generation scenarios.

    Attributes:
        provider: The provider.
        client: Builds the client from a token, or from ``session=``.
        factory: The session factory of the client, sent to the local
            server by replacing its ``api``.
        path: Request path of the endpoint, such as ``/v1/responses``.
        response: A finished buffered response.
        text: Its text.
        usage: Its shared token counts.
        truncated: A response cut off by the token limit, ``text`` kept.
        refused: A response in which the model refused.
        refusal: The refusal's text in it.
        stream: Events of a finished stream, its last event terminal; its
            first event whose ``type`` has ``delta`` in it is a text
            delta.
        deltas: Text deltas of that stream.
        stream_usage: Shared token counts of that stream.
        overloaded: An overloaded error, as the stream sends it.
        quota: An exhausted quota, as the stream sends it.
        bad_request: Body of a 400 response.
        raw_stream: Builds the provider's own stream method.
        options: Valid provider options, sent as they are.
        other_options: Other valid options, for a concurrent call.
        bad_option: A provider option with a value the method refuses.
        effort: What ``effort=Effort.HIGH`` becomes in the body.
        output_schema: What an output schema with ``Effort.HIGH``
            becomes.
        tools: What ``TOOLS`` with ``ToolChoice.REQUIRED`` and no
            parallel calls become.
        forced: What a ``ForcedTool`` becomes.
        only_parallel: What ``parallel_tool_calls=True`` alone becomes.
        called: A finished response with ``CALLS``, text first.
        called_stream: A finished stream with ``CALLS``.
        cut_off: What makes ``called`` stop at the token limit.
        input_field: The body field of the conversation.
        continued: The body's conversation after its first turn, when
            ``called`` comes back with its results.
        continued_shared: The same from the shared calls only.
        continued_streamed: The same when ``called_stream`` comes back.
        parts: What a user turn of ``PARTS`` becomes.
        after_results: The body's conversation after its first two
            items, when a call to ``get_time`` (id ``call_1``) has its
            result ``noon`` and a user turn of ``PARTS[:2]`` follows.
        builtin: The built-in tool scenarios; ``None`` for a provider
            without built-in tools, which skips them.
    """

    provider: Provider
    client: Callable[..., Any]
    factory: SessionFactory[Any, Any, ...]
    path: str
    response: dict[str, Any]
    text: str
    usage: TokenUsage
    truncated: dict[str, Any]
    refused: dict[str, Any]
    refusal: str
    stream: list[dict[str, Any]]
    deltas: list[str]
    stream_usage: TokenUsage
    overloaded: dict[str, Any]
    quota: dict[str, Any]
    bad_request: dict[str, Any]
    raw_stream: Callable[[], BaseStreamMethod[Any]]
    options: dict[str, Any]
    other_options: dict[str, Any]
    bad_option: dict[str, Any]
    effort: dict[str, Any]
    output_schema: dict[str, Any]
    tools: dict[str, Any]
    forced: dict[str, Any]
    only_parallel: dict[str, Any]
    called: dict[str, Any]
    called_stream: list[dict[str, Any]]
    cut_off: dict[str, Any]
    input_field: str
    continued: list[Any]
    continued_shared: list[Any]
    continued_streamed: list[Any]
    parts: dict[str, Any]
    after_results: list[Any]
    builtin: BuiltinCase | None = None

    @property
    def delta(self) -> dict[str, Any]:
        """The first text delta of the stream."""
        return next(e for e in self.stream if "delta" in e["type"])

    @property
    def until_delta(self) -> list[dict[str, Any]]:
        """The stream up to its first text delta, included."""
        return self.stream[: self.stream.index(self.delta) + 1]


def with_tools(tools: list[Any], input: Any = None) -> GenerationRequest:
    """Build the request of most scenarios with tools.

    Args:
        tools: The tools: functions, built-in tools, or both.
        input: The conversation, if it is not ``REQUEST``'s text.

    Returns:
        ``REQUEST`` with the tools, validated.
    """
    fields: dict[str, Any] = {**REQUEST.model_dump(by_alias=True)}
    if input is not None:
        fields["input"] = input
    return GenerationRequest.model_validate({**fields, "tools": tools})


def frame(data: dict[str, Any]) -> bytes:
    """Write one SSE frame with ``event:`` and ``data:``.

    Args:
        data: The event, its ``type`` the frame's ``event``.

    Returns:
        The frame.
    """
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def events(payloads: list[dict[str, Any]]) -> Reply:
    """Build a streamed reply of the given events.

    Args:
        payloads: The events, in order.

    Returns:
        The reply, as ``text/event-stream``.
    """
    return Reply(
        chunks=[frame(payload) for payload in payloads],
        content_type="text/event-stream",
    )


async def concurrently(
    client: Any,
    mode: str,
    calls: list[dict[str, Any]],
) -> None:
    """Run one call per options at once on the client's one resource.

    Args:
        client: The client.
        mode: ``create`` or ``stream``.
        calls: The provider options of each call.
    """

    async def call(options: dict[str, Any]) -> None:
        """Make one call with its options.

        Args:
            options: The call's provider options.
        """
        if mode == "create":
            await client.generation.create(REQUEST, provider_options=options)
        else:
            async with client.generation.stream(
                REQUEST, provider_options=options
            ) as stream:
                await stream.final()

    await asyncio.gather(*(call(options) for options in calls))


def text_of(received: list[Any]) -> list[str]:
    """Take the text deltas among shared events.

    Args:
        received: The events.

    Returns:
        The text of each ``TextDeltaEvent``.
    """
    return [e.text for e in received if isinstance(e, TextDeltaEvent)]


def endless(case: GenerationCase) -> Reply:
    """Build a stream of text deltas that never ends.

    Args:
        case: The provider.

    Returns:
        The reply: the first event, then the first delta over and over.
    """
    return Reply(
        chunks=itertools.chain(
            [frame(case.stream[0])], itertools.repeat(frame(case.delta))
        ),
        chunk_delay=0.001,
        content_type="text/event-stream",
    )


def opener(
    client: Any, case: GenerationCase, mode: str
) -> AbstractAsyncContextManager[Stream[Any]]:
    """Open the raw provider stream or the shared generation stream.

    Args:
        client: The client.
        case: The provider.
        mode: ``raw`` or ``generation``.

    Returns:
        The stream's context manager.
    """
    opened: AbstractAsyncContextManager[Stream[Any]] = (
        client.stream(case.raw_stream())
        if mode == "raw"
        else client.generation.stream(REQUEST)
    )
    return opened


MODES: Final = pytest.mark.parametrize("mode", ["raw", "generation"])
"""Both streams: the provider's own method and the shared one."""


class GenerationContract:
    """The generation scenarios, for a subclass to run on its provider.

    A subclass whose name starts with ``Test`` sets ``case``; pytest
    collects its scenarios. It needs the ``aiollm.testing.plugin``
    fixtures (``recorder``, ``api``).

    Attributes:
        case: The provider the scenarios run on.
    """

    case: ClassVar[GenerationCase]

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

    @pytest.fixture
    def make_client(
        self, api: LocalAPI, monkeypatch: pytest.MonkeyPatch
    ) -> Callable[..., Any]:
        """Build the provider's client with the given session settings.

        Args:
            api: The local server's API.
            monkeypatch: pytest's monkeypatch.

        Returns:
            A function of the session's keyword arguments that builds
            the client on a session of the factory.
        """
        monkeypatch.setattr(self.case.factory, "api", api)
        case = self.case

        def build(**session_kwargs: Any) -> Any:
            """Build the client on a session with these settings.

            Args:
                **session_kwargs: Arguments of the session.

            Returns:
                The client.
            """
            session = case.factory(TEST_TOKEN, **session_kwargs)
            return case.client(session=session)

        return build

    async def test_create_returns_the_text_and_usage(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Create returns the text and usage."""
        case = self.case
        recorder.reply(Reply(json=case.response))

        async with client:
            response = await client.generation.create(REQUEST)

        assert response.provider == case.provider
        assert response.id == case.response["id"]
        assert response.text == case.text
        assert response.usage == case.usage
        assert response.incomplete_reason is None
        assert recorder.last.raw_path == case.path

    async def test_truncated_response_keeps_text_and_reason(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Truncated response keeps text and reason."""
        case = self.case
        recorder.reply(Reply(json=case.truncated))

        async with client:
            response = await client.generation.create(REQUEST)

        assert response.text == case.text
        assert response.incomplete_reason is IncompleteReason.MAX_TOKENS

    async def test_refusal_is_an_answer(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Refusal is an answer."""
        case = self.case
        recorder.reply(Reply(json=case.refused))

        async with client:
            response = await client.generation.create(REQUEST)

        assert response.incomplete_reason is IncompleteReason.REFUSAL
        assert response.refusal == case.refusal

    def test_generation_shares_the_client_session(self, client: Any) -> None:
        """Generation shares the client session."""
        assert isinstance(client.generation, GenerationResource)
        assert client.generation.session is client.session

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_effort_reaches_the_wire(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """Effort reaches the wire."""
        case = self.case
        if mode == "create":
            recorder.reply(Reply(json=case.response))
        else:
            recorder.reply(events(case.stream))
        request = REQUEST.model_copy(update={"effort": Effort.HIGH})

        async with client:
            if mode == "create":
                await client.generation.create(request)
            else:
                async with client.generation.stream(request) as stream:
                    await stream.final()

        body = json.loads(recorder.last.body)
        assert {key: body[key] for key in case.effort} == case.effort

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_an_output_schema_reaches_the_wire_as_it_is(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """An output schema reaches the wire as it is."""
        case = self.case
        # With effort too: Anthropic sends both in one output_config.
        if mode == "create":
            recorder.reply(Reply(json=case.response))
        else:
            recorder.reply(events(case.stream))
        request = REQUEST.model_copy(
            update={
                "effort": Effort.HIGH,
                "output_schema": {"type": "object", "required": ["n"]},
            }
        )

        async with client:
            if mode == "create":
                await client.generation.create(request)
            else:
                async with client.generation.stream(request) as stream:
                    await stream.final()

        body = json.loads(recorder.last.body)
        assert {key: body[key] for key in case.output_schema} == (
            case.output_schema
        )

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_tools_reach_the_wire(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """Tools reach the wire."""
        case = self.case
        if mode == "create":
            recorder.reply(Reply(json=case.response))
        else:
            recorder.reply(events(case.stream))
        request = GenerationRequest.model_validate(
            {
                **REQUEST.model_dump(by_alias=True),
                "tools": TOOLS,
                "tool_choice": ToolChoice.REQUIRED,
                "parallel_tool_calls": False,
            }
        )

        async with client:
            if mode == "create":
                await client.generation.create(request)
            else:
                async with client.generation.stream(request) as stream:
                    await stream.final()

        body = json.loads(recorder.last.body)
        assert {key: body[key] for key in case.tools} == case.tools

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_tool_calls_come_back_alike(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """Tool calls come back alike."""
        case = self.case
        # The arguments' text is the provider's (OpenAI sends it, Anthropic's
        # object is written as JSON), so the parsed arguments are compared.
        if mode == "create":
            recorder.reply(Reply(json=case.called))
        else:
            recorder.reply(events(case.called_stream))
        request = REQUEST.model_copy(update={"tools": TOOLS})

        async with client:
            if mode == "create":
                result: Any = await client.generation.create(request)
            else:
                async with client.generation.stream(request) as stream:
                    result = await stream.final()

        assert [
            (call.id, call.name, call.parse_arguments())
            for call in result.tool_calls
        ] == CALLS
        assert result.incomplete_reason is None
        assert result.raw_incomplete_reason is None

    async def test_tool_calls_stream_as_start_pieces_and_done(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Tool calls stream as start pieces and done."""
        case = self.case
        recorder.reply(events(case.called_stream))
        request = REQUEST.model_copy(update={"tools": TOOLS})

        async with client, client.generation.stream(request) as stream:
            received = [event async for event in stream]

        starts = [e for e in received if isinstance(e, ToolCallStartEvent)]
        done = [e.call for e in received if isinstance(e, ToolCallDoneEvent)]
        assert [(e.call_id, e.name) for e in starts] == [
            (call_id, name) for call_id, name, _ in CALLS
        ]
        assert [
            (call.id, call.name, call.parse_arguments()) for call in done
        ] == CALLS
        for call in done:
            pieces = [
                e.arguments
                for e in received
                if isinstance(e, ToolCallDeltaEvent) and e.call_id == call.id
            ]
            assert not pieces or "".join(pieces) == call.arguments
        completed = received[-1]
        assert isinstance(completed, CompletedEvent)
        assert completed.tool_calls == done
        for call in done:
            opened = next(
                n
                for n, e in enumerate(received)
                if isinstance(e, ToolCallStartEvent) and e.call_id == call.id
            )
            closed = next(
                n
                for n, e in enumerate(received)
                if isinstance(e, ToolCallDoneEvent) and e.call.id == call.id
            )
            assert opened < closed

    async def test_a_stream_can_fail_after_its_calls_are_done(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """A stream can fail after its calls are done."""
        case = self.case
        # A done call is a complete call, not a successful generation: the
        # stream can still fail before its CompletedEvent.
        recorder.reply(events([*case.called_stream[:-1], case.overloaded]))
        request = REQUEST.model_copy(update={"tools": TOOLS})
        received: list[Any] = []

        async with client, client.generation.stream(request) as stream:
            with pytest.raises(ServiceUnavailableError):
                async for event in stream:
                    received.append(event)  # noqa: PERF401 (kept on error)

        done = [e.call for e in received if isinstance(e, ToolCallDoneEvent)]
        assert [call.id for call in done] == [
            call_id for call_id, _, _ in CALLS
        ]
        assert not any(isinstance(e, CompletedEvent) for e in received)

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_a_turn_of_parts_sends_text_images_and_documents(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """A turn of parts sends text images and documents."""
        case = self.case
        if mode == "create":
            recorder.reply(Reply(json=case.response))
        else:
            recorder.reply(events(case.stream))
        request = REQUEST.model_copy(update={"input": [Turn.user(PARTS)]})

        async with client:
            if mode == "create":
                await client.generation.create(request)
            else:
                async with client.generation.stream(request) as stream:
                    await stream.final()

        assert json.loads(recorder.last.body)[case.input_field] == [case.parts]

    async def test_parts_after_tool_results_go_with_them(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Parts after tool results go with them."""
        case = self.case
        # Anthropic puts a user turn after results into their message;
        # OpenAI keeps it a message of its own: the case says which.
        recorder.reply(Reply(json=case.response))
        call = ToolCall(id="call_1", name="get_time", arguments="{}")
        request = REQUEST.model_copy(
            update={
                "tools": TOOLS,
                "input": [
                    Turn.user("Hi"),
                    ToolCallTurn(tool_calls=[call]),
                    ToolResult(call_id="call_1", output="noon"),
                    Turn.user(PARTS[:2]),
                ],
            }
        )

        async with client:
            await client.generation.create(request)

        sent = json.loads(recorder.last.body)[case.input_field]
        assert sent[2:] == case.after_results

    async def test_a_streamed_tool_loop_continues_with_the_results(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """A streamed tool loop continues with the results."""
        case = self.case
        # The completed event's turn carries the stream's own output, put
        # together from its events, and the second request sends it back.
        recorder.reply(events(case.called_stream), Reply(json=case.response))
        request = REQUEST.model_copy(update={"tools": TOOLS})

        async with client:
            async with client.generation.stream(request) as stream:
                completed = await stream.final()
            turn = completed.to_turn()
            results = [
                ToolResult(call_id=call.id, output=f"result of {call.name}")
                for call in completed.tool_calls
            ]
            await client.generation.create(
                request.model_copy(
                    update={"input": [Turn.user("Hi"), turn, *results]}
                )
            )

        assert isinstance(turn, ToolCallTurn)
        assert turn.native is not MISSING
        assert turn.native.provider == case.provider
        sent = json.loads(recorder.last.body)
        assert sent[case.input_field][1:] == case.continued_streamed

    async def test_a_tool_loop_continues_with_the_results(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """A tool loop continues with the results."""
        case = self.case
        # The response's turn of calls goes back with the provider's own
        # output, and the results follow it; the second request ends the
        # loop with text.
        recorder.reply(Reply(json=case.called), Reply(json=case.response))
        request = REQUEST.model_copy(update={"tools": TOOLS})

        async with client:
            called = await client.generation.create(request)
            turn = called.to_turn()
            results = [
                ToolResult(call_id=call.id, output=f"result of {call.name}")
                for call in called.tool_calls
            ]
            answer = await client.generation.create(
                request.model_copy(
                    update={"input": [Turn.user("Hi"), turn, *results]}
                )
            )

        assert isinstance(turn, ToolCallTurn)
        assert turn.native is not MISSING
        assert turn.native.provider == case.provider
        assert answer.text == case.text
        sent = json.loads(recorder.last.body)
        assert sent[case.input_field][1:] == case.continued

    async def test_a_turn_of_calls_from_another_provider_sends_its_calls(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """A turn of calls from another provider sends its calls."""
        case = self.case
        # Another provider's output means nothing here: the shared calls go.
        recorder.reply(Reply(json=case.response))
        foreign = Provider("another")
        turn = ToolCallTurn(
            text="Checking.",
            tool_calls=[
                ToolCall(
                    id=call_id, name=name, arguments=json.dumps(arguments)
                )
                for call_id, name, arguments in CALLS
            ],
            native=NativeOutput(provider=foreign, items=[{"type": "opaque"}]),
        )
        results = [
            ToolResult(call_id=call_id, output=f"result of {name}")
            for call_id, name, _ in CALLS
        ]
        request = REQUEST.model_copy(
            update={"tools": TOOLS, "input": [Turn.user("Hi"), turn, *results]}
        )

        async with client:
            await client.generation.create(request)

        sent = json.loads(recorder.last.body)
        assert sent[case.input_field][1:] == case.continued_shared

    @pytest.mark.parametrize(
        ("items", "match"),
        [
            (["call"], "have no result"),
            (["call", "result-1"], "have no result"),
            (
                ["call", "user", "result-1", "result-2"],
                "comes before the results",
            ),
            (["result-1"], "did not make"),
            (["call", "result-1", "result-1", "result-2"], "did not make"),
            (["twins"], "share an id"),
        ],
        ids=[
            "unanswered",
            "half-answered",
            "text-between",
            "result-alone",
            "result-twice",
            "same-id",
        ],
    )
    async def test_calls_and_results_out_of_order_are_refused(
        self,
        client: Any,
        items: list[str],
        match: str,
    ) -> None:
        """Calls and results out of order are refused."""
        case = self.case
        call = ToolCall(id="call_1", name="get_weather", arguments="{}")
        other = ToolCall(id="call_2", name="get_time", arguments="{}")
        pieces = {
            "call": ToolCallTurn(tool_calls=[call, other]),
            "twins": ToolCallTurn(tool_calls=[call, call]),
            "user": Turn.user("And?"),
            "result-1": ToolResult(call_id="call_1", output="15"),
            "result-2": ToolResult(call_id="call_2", output="noon"),
        }
        conversation = [Turn.user("Hi"), *(pieces[name] for name in items)]
        request = REQUEST.model_copy(update={"input": conversation})

        async with client:
            with pytest.raises(RequestValidationError, match=match) as caught:
                await client.generation.create(request)

        assert caught.value.field == "input"
        assert caught.value.provider == case.provider

    async def test_a_response_that_stopped_mid_call_is_no_turn(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """A response that stopped mid call is no turn."""
        case = self.case
        recorder.reply(Reply(json=case.called | case.cut_off))

        async with client:
            response = await client.generation.create(
                REQUEST.model_copy(update={"tools": TOOLS})
            )

        assert response.incomplete_reason is IncompleteReason.MAX_TOKENS
        assert response.tool_calls
        with pytest.raises(ValueError, match="last tool call may be cut off"):
            response.to_turn()

    async def test_a_response_without_calls_has_none(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """A response without calls has none."""
        case = self.case
        recorder.reply(Reply(json=case.response))

        async with client:
            response = await client.generation.create(REQUEST)

        assert response.tool_calls == []

    @pytest.mark.parametrize(
        ("changes", "expected"),
        [
            ({"tool_choice": ForcedTool(name="get_time")}, "forced"),
            ({"parallel_tool_calls": True}, "only_parallel"),
        ],
        ids=["forced", "only-parallel"],
    )
    async def test_tool_choices_reach_the_wire(
        self,
        client: Any,
        recorder: Recorder,
        changes: dict[str, Any],
        expected: str,
    ) -> None:
        """Tool choices reach the wire."""
        case = self.case
        recorder.reply(Reply(json=case.response))
        request = GenerationRequest.model_validate(
            {"model": "model-1", "input": "Hi", "tools": TOOLS, **changes}
        )

        async with client:
            await client.generation.create(request)

        body = json.loads(recorder.last.body)
        wanted = getattr(case, expected)
        assert {key: body[key] for key in wanted} == wanted

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_provider_options_reach_the_wire_of_their_call_only(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """Provider options reach the wire of their call only."""
        case = self.case
        # The options belong to the call: the next one without them sends none.
        if mode == "create":
            reply = Reply(json=case.response)
        else:
            reply = events(case.stream)
        recorder.reply(reply)
        recorder.reply(reply)

        async with client:
            for options in (case.options, {}):
                if mode == "create":
                    await client.generation.create(
                        REQUEST, provider_options=options
                    )
                else:
                    async with client.generation.stream(
                        REQUEST, provider_options=options
                    ) as stream:
                        await stream.final()

        with_options, without = (json.loads(r.body) for r in recorder.requests)
        assert {key: with_options[key] for key in case.options} == case.options
        assert set(case.options).isdisjoint(without)

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_concurrent_calls_keep_their_own_provider_options(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """Concurrent calls keep their own provider options."""
        case = self.case
        # The calls share the resource and its adapter; each copy carries
        # only its own options, and a later call without any sends none: an
        # adapter that kept a call's options on itself would leak them there.
        calls = [case.options, case.other_options, case.options]
        for _ in [*calls, None]:
            if mode == "create":
                recorder.reply(Reply(json=case.response))
            else:
                recorder.reply(events(case.stream))

        async with client:
            await concurrently(client, mode, calls)
            if mode == "create":
                await client.generation.create(REQUEST)
            else:
                async with client.generation.stream(REQUEST) as stream:
                    await stream.final()

        keys = set(case.options) | set(case.other_options)
        *sent, last = [
            {key: body[key] for key in keys if key in body}
            for body in (json.loads(r.body) for r in recorder.requests)
        ]

        def key(options: dict[str, Any]) -> str:
            return json.dumps(options, sort_keys=True)

        assert sorted(map(key, sent)) == sorted(map(key, calls))
        assert last == {}

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_an_unknown_provider_option_is_refused_before_sending(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """An unknown provider option is refused before sending."""
        case = self.case
        # A shared field is no option: it is set in the request.
        shared = {"model": "m", "input": "x"}

        async with client:
            with pytest.raises(
                RequestValidationError, match="input, model"
            ) as caught:
                if mode == "create":
                    await client.generation.create(
                        REQUEST, provider_options=shared
                    )
                else:
                    async with client.generation.stream(
                        REQUEST, provider_options=shared
                    ):
                        pass  # pragma: no cover - refused first

        assert caught.value.provider == case.provider
        assert caught.value.field == "input"
        assert recorder.requests == []

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_an_invalid_provider_option_is_refused_before_sending(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """An invalid provider option is refused before sending."""
        case = self.case
        async with client:
            with pytest.raises(RequestValidationError) as caught:
                if mode == "create":
                    await client.generation.create(
                        REQUEST, provider_options=case.bad_option
                    )
                else:
                    async with client.generation.stream(
                        REQUEST, provider_options=case.bad_option
                    ):
                        pass  # pragma: no cover - refused first

        assert caught.value.field == next(iter(case.bad_option))
        assert recorder.requests == []

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_conversation_must_end_with_a_user_turn(
        self,
        client: Any,
        recorder: Recorder,
        mode: str,
    ) -> None:
        """Conversation must end with a user turn."""
        case = self.case
        # Refused before anything is sent, by every provider.
        turns = [Turn.user("Hi"), Turn.assistant("Hello")]

        async with client:
            with pytest.raises(
                RequestValidationError, match="last turn"
            ) as caught:
                if mode == "create":
                    await client.generation.create(model="m", input=turns)
                else:
                    async with client.generation.stream(
                        model="m", input=turns
                    ):
                        pass  # pragma: no cover - refused first

        assert caught.value.provider == case.provider
        assert caught.value.field == "input"
        assert recorder.requests == []

    async def test_conversation_continues_with_the_response(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Conversation continues with the response."""
        case = self.case
        recorder.reply(Reply(json=case.response), Reply(json=case.response))

        async with client:
            first = await client.generation.create(REQUEST)
            request = REQUEST.model_copy(
                update={
                    "input": [
                        Turn.user("Hi"),
                        first.to_turn(),
                        Turn.user("More"),
                    ]
                }
            )
            await client.generation.create(request)

        sent = json.loads(recorder.last.body)
        assert sent[case.input_field] == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": case.text},
            {"role": "user", "content": "More"},
        ]

    async def test_stream_yields_deltas_then_completes(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Stream yields deltas then completes."""
        case = self.case
        recorder.reply(events(case.stream))

        async with client, client.generation.stream(REQUEST) as stream:
            received = [event async for event in stream]

        deltas = [e.text for e in received if isinstance(e, TextDeltaEvent)]
        assert deltas == case.deltas
        last = received[-1]
        assert isinstance(last, CompletedEvent)
        assert last.provider == case.provider
        assert last.usage == case.stream_usage
        assert last.incomplete_reason is None
        assert json.loads(recorder.last.body)["stream"] is True

    async def test_stream_error_event_raises_overloaded(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Stream error event raises overloaded."""
        case = self.case
        recorder.reply(events([case.stream[0], case.overloaded]))

        async with client, client.generation.stream(REQUEST) as stream:
            with pytest.raises(ServiceUnavailableError) as caught:
                async for _ in stream:
                    pass

        assert caught.value.provider == case.provider
        assert "Overloaded" in str(caught.value)
        # Only opening is retried: the events already arrived.
        assert len(recorder.requests) == 1

    async def test_stream_quota_error_is_a_quota_error(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Stream quota error is a quota error."""
        case = self.case
        # The same class as over HTTP 429: the stream's 200 does not hide it.
        recorder.reply(events([case.stream[0], case.quota]))

        async with client, client.generation.stream(REQUEST) as stream:
            with pytest.raises(QuotaExceededError) as caught:
                async for _ in stream:
                    pass

        assert caught.value.provider == case.provider
        assert caught.value.transient is False
        assert caught.value.status == 200
        assert len(recorder.requests) == 1

    async def test_stream_cut_before_its_end_raises(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Stream cut before its end raises."""
        case = self.case
        recorder.reply(events(case.stream[:-1]))

        received: list[Any] = []

        async with client, client.generation.stream(REQUEST) as stream:
            with pytest.raises(APIStreamInterruptedError) as caught:
                async for event in stream:
                    received.append(event)  # noqa: PERF401 - fails mid-loop

        assert caught.value.__cause__ is None
        assert caught.value.transient is False
        assert len(received) == len(case.stream) - 1
        assert not any(isinstance(event, CompletedEvent) for event in received)

    async def test_break_closes_the_stream(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Break closes the stream."""
        case = self.case
        recorder.reply(events(case.stream))
        first: list[str] = []

        async with client:
            async with client.generation.stream(REQUEST) as stream:
                async for event in stream:  # pragma: no branch - breaks
                    if isinstance(event, TextDeltaEvent):
                        first.append(event.text)
                        break
            with pytest.raises(RuntimeError):
                await anext(stream)

        assert first == case.deltas[:1]

    async def test_error_status_raises_before_streaming(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Error status raises before streaming."""
        case = self.case
        recorder.reply(
            Reply(status=400, json=case.bad_request),
            Reply(status=400, json=case.bad_request),
        )

        async with client:
            with pytest.raises(BadRequestError) as created:
                await client.generation.create(REQUEST)
            with pytest.raises(BadRequestError):
                async with client.generation.stream(REQUEST):
                    pass  # pragma: no cover

        assert created.value.provider == case.provider
        assert created.value.transient is False
        assert len(recorder.requests) == 2

    async def test_transient_failures_are_retried(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Transient failures are retried."""
        case = self.case
        unavailable = Reply(status=503, json={"error": {"message": "Busy"}})
        recorder.reply(
            unavailable,
            Reply(json=case.response),
            unavailable,
            events(case.stream),
        )

        async with client:
            response = await client.generation.create(REQUEST)
            async with client.generation.stream(REQUEST) as stream:
                completed = await stream.final()

        assert response.text == case.text
        assert completed.usage == case.stream_usage
        assert len(recorder.requests) == 4

    async def test_call_options_reach_the_session(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Call options reach the session."""
        unavailable = Reply(status=503, json={"error": {"message": "Busy"}})
        recorder.reply(unavailable, unavailable)
        once = RetryPolicy(max_attempts=1)

        async with client:
            with pytest.raises(ServiceUnavailableError):
                await client.generation.create(
                    REQUEST, headers={"X-Trace": "t-1"}, retry_policy=once
                )
            with pytest.raises(ServiceUnavailableError):
                async with client.generation.stream(
                    model="model-1",
                    input="Hi",
                    headers={"X-Trace": "t-2"},
                    retry_policy=once,
                ):
                    pass  # pragma: no cover

        assert [r.headers["X-Trace"] for r in recorder.requests] == [
            "t-1",
            "t-2",
        ]

    async def test_a_call_timeout_limits_the_silence_of_a_stream(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """A call timeout limits the silence of a stream."""
        case = self.case
        resume = asyncio.Event()
        frames = [frame(payload) for payload in case.until_delta]
        recorder.reply(
            Reply(
                chunks=[*frames, resume, frame(case.stream[-1])],
                content_type="text/event-stream",
            )
        )

        async with client, client.generation.stream(REQUEST, timeout=0.3) as s:
            with pytest.raises(APIStreamInterruptedError) as caught:
                async for _ in s:
                    pass

        resume.set()

        assert isinstance(caught.value.__cause__, APITimeoutError)

    async def test_connection_drop_keeps_the_text_and_raises(
        self,
        client: Any,
        recorder: Recorder,
    ) -> None:
        """Connection drop keeps the text and raises."""
        case = self.case
        reply = events(case.until_delta)
        recorder.reply(replace(reply, abort=True))
        received: list[Any] = []

        async with client, client.generation.stream(REQUEST) as stream:
            with pytest.raises(APIStreamInterruptedError) as caught:
                async for event in stream:
                    received.append(event)  # noqa: PERF401 - fails mid-loop

        assert text_of(received) == case.deltas[:1]
        assert isinstance(caught.value.__cause__, APIConnectionError)
        assert caught.value.transient is False
        assert len(recorder.requests) == 1

    async def test_silence_keeps_the_text_and_raises(
        self,
        make_client: Callable[..., Any],
        recorder: Recorder,
    ) -> None:
        """Silence keeps the text and raises."""
        case = self.case
        resume = asyncio.Event()
        frames = [frame(payload) for payload in case.until_delta]
        recorder.reply(
            Reply(
                chunks=[*frames, resume, frame(case.stream[-1])],
                content_type="text/event-stream",
            )
        )
        received: list[Any] = []
        from aiohttp import ClientTimeout  # the clients' transport

        client = make_client(stream_timeout=ClientTimeout(sock_read=0.3))

        async with client, client.generation.stream(REQUEST) as stream:
            with pytest.raises(APIStreamInterruptedError) as caught:
                async for event in stream:
                    received.append(event)  # noqa: PERF401 - fails mid-loop

        resume.set()

        assert text_of(received) == case.deltas[:1]
        assert isinstance(caught.value.__cause__, APITimeoutError)

    @MODES
    async def test_cancelling_the_task_while_iterating_releases_the_connection(
        self,
        make_client: Callable[..., Any],
        recorder: Recorder,
        mode: str,
    ) -> None:
        """Cancelling the task while iterating releases the connection."""
        case = self.case
        recorder.reply(endless(case), Reply(json=case.response))
        started = asyncio.Event()

        async def consume(client: Any) -> None:
            async with opener(client, case, mode) as stream:
                async for _ in stream:
                    started.set()

        # With one connection, a leaked one would block the next request.
        async with make_client(max_connections=1) as client:
            task = asyncio.create_task(consume(client))
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            async with asyncio.timeout(2):
                response = await client.generation.create(REQUEST)

        assert response.text == case.text

    @MODES
    async def test_cancelling_the_task_while_opening_releases_the_connection(
        self,
        make_client: Callable[..., Any],
        recorder: Recorder,
        mode: str,
    ) -> None:
        """Cancelling the task while opening releases the connection."""
        case = self.case
        recorder.reply(
            replace(events(case.stream), delay=1),
            Reply(json=case.response),
        )

        async def consume(client: Any) -> None:
            async with opener(
                client, case, mode
            ) as stream:  # pragma: no cover
                async for _ in stream:
                    pass

        async with make_client(max_connections=1) as client:
            task = asyncio.create_task(consume(client))

            async with asyncio.timeout(2):
                while not recorder.requests:
                    await asyncio.sleep(0.01)

            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            async with asyncio.timeout(2):
                response = await client.generation.create(REQUEST)

        assert response.text == case.text

    @pytest.fixture
    def builtin(self) -> BuiltinCase:
        """Take the case's built-in tool scenarios, or skip.

        Returns:
            The case's ``builtin``.
        """
        if self.case.builtin is None:  # pragma: no cover
            pytest.skip("the provider has no built-in tools")
        return self.case.builtin

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_builtin_tools_reach_the_wire(
        self,
        client: Any,
        recorder: Recorder,
        builtin: BuiltinCase,
        mode: str,
    ) -> None:
        """Built-in tools reach the wire."""
        case = self.case
        recorder.reply(
            Reply(json=case.response)
            if mode == "create"
            else events(case.stream)
        )
        request = with_tools(BUILTIN_TOOLS)

        async with client:
            if mode == "create":
                await client.generation.create(request)
            else:
                async with client.generation.stream(request) as stream:
                    await stream.final()

        body = json.loads(recorder.last.body)
        assert {key: body[key] for key in builtin.tools} == builtin.tools

    async def test_builtin_tools_follow_function_tools(
        self,
        client: Any,
        recorder: Recorder,
        builtin: BuiltinCase,
    ) -> None:
        """Built-in tools follow function tools."""
        recorder.reply(Reply(json=self.case.response))
        request = with_tools([WebSearch(), TOOLS[1]])

        async with client:
            await client.generation.create(request)

        body = json.loads(recorder.last.body)
        assert {key: body[key] for key in builtin.mixed} == builtin.mixed

    async def test_a_tool_of_another_provider_is_refused(
        self,
        client: Any,
        recorder: Recorder,
        builtin: BuiltinCase,
    ) -> None:
        """A tool of another provider is refused."""
        other = ProviderTool(
            provider=Provider("someone-else"), definition={"type": "x"}
        )
        request = with_tools([other])

        async with client:
            with pytest.raises(RequestValidationError) as caught:
                await client.generation.create(request)

        assert builtin
        assert caught.value.field == "tools"
        assert caught.value.provider == self.case.provider
        assert not recorder.requests

    @pytest.mark.parametrize("mode", ["create", "stream"])
    async def test_builtin_tool_calls_come_back_alike(
        self,
        client: Any,
        recorder: Recorder,
        builtin: BuiltinCase,
        mode: str,
    ) -> None:
        """Built-in tool calls and citations come back alike."""
        recorder.reply(
            Reply(json=builtin.searched)
            if mode == "create"
            else events(builtin.searched_stream)
        )
        request = with_tools(BUILTIN_TOOLS)

        async with client:
            if mode == "create":
                result: Any = await client.generation.create(request)
            else:
                async with client.generation.stream(request) as stream:
                    result = await stream.final()

        assert [
            (call.id, call.kind, call.status, call.queries)
            for call in result.server_tool_calls
        ] == [
            (
                SEARCH_ID,
                ServerToolKind.WEB_SEARCH,
                ServerToolStatus.COMPLETED,
                SEARCH_QUERIES,
            )
        ]
        assert all(call.raw for call in result.server_tool_calls)
        assert [
            (citation.kind, citation.url, citation.title)
            for citation in result.citations
        ] == CITATIONS
        assert result.tool_calls == []
        assert result.text
        assert result.incomplete_reason is None

    async def test_builtin_tool_calls_stream_as_started_then_done(
        self,
        client: Any,
        recorder: Recorder,
        builtin: BuiltinCase,
    ) -> None:
        """Built-in tool calls stream as started then done."""
        recorder.reply(events(builtin.searched_stream))
        request = with_tools(BUILTIN_TOOLS)

        async with client, client.generation.stream(request) as stream:
            received = [event async for event in stream]

        tool_events = [
            e for e in received if isinstance(e, ServerToolCallEvent)
        ]
        assert [(e.phase, e.call.id) for e in tool_events] == [
            (ServerToolPhase.STARTED, SEARCH_ID),
            (ServerToolPhase.DONE, SEARCH_ID),
        ]
        assert tool_events[0].call.status is ServerToolStatus.IN_PROGRESS
        assert tool_events[1].call.status is ServerToolStatus.COMPLETED
        completed = received[-1]
        assert isinstance(completed, CompletedEvent)
        assert completed.server_tool_calls[0].id == SEARCH_ID
        assert completed.server_tool_calls[0].queries == SEARCH_QUERIES

    async def test_a_failed_builtin_tool_is_a_call_not_an_error(
        self,
        client: Any,
        recorder: Recorder,
        builtin: BuiltinCase,
    ) -> None:
        """A failed built-in tool is a call, not an error."""
        recorder.reply(Reply(json=builtin.failed))
        request = with_tools(BUILTIN_TOOLS)

        async with client:
            response = await client.generation.create(request)

        [call] = response.server_tool_calls
        assert call.kind is ServerToolKind.WEB_SEARCH
        assert call.status is ServerToolStatus.FAILED
        assert call.error

    async def test_a_paused_generation_continues_with_its_native_output(
        self,
        client: Any,
        recorder: Recorder,
        builtin: BuiltinCase,
    ) -> None:
        """A paused generation continues with its native output."""
        case = self.case
        recorder.reply(Reply(json=builtin.paused), Reply(json=case.response))
        request = with_tools(BUILTIN_TOOLS)

        async with client:
            paused = await client.generation.create(request)
            turn = paused.to_turn()
            await client.generation.create(
                with_tools(BUILTIN_TOOLS, [Turn.user("Hi"), turn])
            )

        assert paused.incomplete_reason is IncompleteReason.PAUSED
        assert paused.text is None
        assert isinstance(turn, Turn)
        assert turn.native is not MISSING
        assert turn.native.provider == case.provider
        body = json.loads(recorder.last.body)
        assert body[case.input_field][1:] == builtin.continued
