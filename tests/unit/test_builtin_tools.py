"""Built-in tools, server tool calls and the turns that carry them back."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import MISSING, ValidationError

from aiollm.adapters import check_builtin_tools, check_conversation
from aiollm.enums import (
    CitationKind,
    IncompleteReason,
    Provider,
    Role,
    ServerToolKind,
    ServerToolStatus,
    ToolChoice,
)
from aiollm.exceptions import RequestValidationError
from aiollm.testing.contract.generation import TOOLS
from aiollm.types.requests import (
    CodeExecution,
    ForcedTool,
    GenerationRequest,
    NativeOutput,
    ProviderTool,
    ToolCallTurn,
    Turn,
    UserLocation,
    WebSearch,
)
from aiollm.types.responses import (
    Citation,
    CompletedEvent,
    GenerationResponse,
    ServerToolCall,
)

PROVIDER = Provider("test")
NATIVE = NativeOutput(provider=PROVIDER, items=[{"type": "search"}])
SEARCH = ServerToolCall(
    id="s1", kind=ServerToolKind.WEB_SEARCH, status=ServerToolStatus.COMPLETED
)


def request(**changes: Any) -> GenerationRequest:
    return GenerationRequest.model_validate(
        {"model": "m", "input": "Hi"} | changes
    )


def response(**changes: Any) -> GenerationResponse[None]:
    fields: dict[str, Any] = {
        "provider": PROVIDER,
        "model": "m",
        "text": "Hello",
        "raw": None,
    }
    return GenerationResponse[None](**fields | changes)


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search", "allowed_domains": []},
        {"type": "web_search", "domains": ["a.com"]},
        {"type": "web_search", "user_location": {"country": ""}},
        {"type": "web_search", "user_location": {"state": "CA"}},
        {"type": "code_execution", "memory": 1},
    ],
)
def test_a_shared_tool_refuses_what_no_provider_takes(
    tool: dict[str, Any],
) -> None:
    model = WebSearch if tool["type"] == "web_search" else CodeExecution

    with pytest.raises(ValidationError):
        model.model_validate(tool)


def test_a_provider_tool_needs_a_definition() -> None:
    with pytest.raises(ValidationError):
        ProviderTool(provider=PROVIDER, definition={})


def test_builtin_tools_share_the_list_with_functions() -> None:
    search = WebSearch(user_location=UserLocation(city="Paris"))
    mine = ProviderTool(provider=PROVIDER, definition={"type": "mcp"})

    both = request(tools=[TOOLS[0], search, CodeExecution(), mine])

    assert both.tools == [TOOLS[0]]
    assert both.builtin_tools == [search, CodeExecution(), mine]


def test_a_request_may_hold_builtin_tools_only() -> None:
    only = request(tools=WebSearch(), tool_choice=ToolChoice.REQUIRED)

    assert only.tools is MISSING
    assert only.builtin_tools == [WebSearch()]
    assert request().builtin_tools == []
    assert request(tools=TOOLS[0]).builtin_tools == []


def test_a_request_takes_one_builtin_tool_of_a_kind() -> None:
    with pytest.raises(ValidationError, match="only once"):
        request(tools=[WebSearch(), WebSearch(allowed_domains=["a.com"])])
    with pytest.raises(ValidationError, match="only once"):
        request(tools=[CodeExecution(), CodeExecution()])


def test_provider_tools_may_repeat() -> None:
    def one(label: str) -> ProviderTool:
        return ProviderTool(
            provider=PROVIDER, definition={"type": "mcp", "label": label}
        )

    assert len(request(tools=[one("a"), one("b")]).builtin_tools) == 2


def test_only_a_function_tool_can_be_forced() -> None:
    with pytest.raises(ValidationError, match="not a tool"):
        request(tools=WebSearch(), tool_choice=ForcedTool(name="web_search"))


def test_a_tool_that_is_none_of_them_is_refused() -> None:
    with pytest.raises(ValidationError, match="Expected Tool or callable"):
        request(tools=[WebSearch(), 5])


def test_an_assistant_turn_may_carry_native_output() -> None:
    turn = Turn.assistant("Hi", native=NATIVE)

    assert turn.native == NATIVE
    assert Turn.assistant("Hi").native is MISSING


def test_a_user_turn_has_no_native_output() -> None:
    with pytest.raises(ValidationError, match="native output"):
        Turn(role=Role.USER, content="Hi", native=NATIVE)


def test_a_turn_with_native_output_may_end_a_conversation() -> None:
    last = Turn.assistant("", native=NATIVE)

    check_conversation(request(input=[Turn.user("Hi"), last]), PROVIDER)
    with pytest.raises(RequestValidationError, match="native output"):
        check_conversation(
            request(input=[Turn.user("Hi"), Turn.assistant("Hello")]),
            PROVIDER,
        )


def test_a_turn_keeps_native_output_for_server_tool_calls() -> None:
    turn = response(server_tool_calls=[SEARCH], native=NATIVE).to_turn()

    assert turn == Turn.assistant("Hello", native=NATIVE)


def test_a_turn_drops_native_output_without_server_tool_calls() -> None:
    assert response(native=NATIVE).to_turn() == Turn.assistant("Hello")


def test_a_paused_generation_continues_without_text() -> None:
    paused = response(
        text=None,
        incomplete_reason=IncompleteReason.PAUSED,
        native=NATIVE,
    )

    assert paused.to_turn() == Turn.assistant("", native=NATIVE)


def test_a_paused_generation_without_native_output_continues_as_text() -> None:
    paused = response(incomplete_reason=IncompleteReason.PAUSED)

    assert paused.to_turn() == Turn.assistant("Hello")


@pytest.mark.parametrize(
    "reason", [IncompleteReason.REFUSAL, IncompleteReason.OTHER, None]
)
def test_other_generations_without_text_still_have_no_turn(
    reason: IncompleteReason | None,
) -> None:
    with pytest.raises(ValueError, match="no text"):
        response(text=None, incomplete_reason=reason, native=NATIVE).to_turn()


def test_calls_that_stopped_early_are_still_refused() -> None:
    from aiollm.types.requests import ToolCall  # noqa: PLC0415

    call = ToolCall(id="c1", name="f", arguments="{}")

    with pytest.raises(ValueError, match="stopped early"):
        response(
            tool_calls=[call], incomplete_reason=IncompleteReason.PAUSED
        ).to_turn()
    assert isinstance(response(tool_calls=[call]).to_turn(), ToolCallTurn)


def test_a_completed_event_makes_the_same_turn() -> None:
    event = CompletedEvent[None](
        provider=PROVIDER,
        model="m",
        text="Hello",
        server_tool_calls=[SEARCH],
        native=NATIVE,
        raw=None,
    )

    assert event.to_turn() == Turn.assistant("Hello", native=NATIVE)


def test_the_providers_check_the_builtin_tools_they_run() -> None:
    mine = ProviderTool(provider=PROVIDER, definition={"type": "mcp"})
    theirs = ProviderTool(
        provider=Provider("other"), definition={"type": "mcp"}
    )

    check_builtin_tools(
        request(tools=[WebSearch(), mine]), PROVIDER, [WebSearch]
    )
    check_builtin_tools(request(tools=TOOLS), PROVIDER, [])
    with pytest.raises(RequestValidationError, match="CodeExecution") as e:
        check_builtin_tools(
            request(tools=[CodeExecution()]), PROVIDER, [WebSearch]
        )
    assert (e.value.provider, e.value.field) == (PROVIDER, "tools")
    with pytest.raises(RequestValidationError, match="other") as e:
        check_builtin_tools(request(tools=[theirs]), PROVIDER, [WebSearch])
    assert (e.value.provider, e.value.field) == (PROVIDER, "tools")


def test_a_call_and_a_citation_have_only_what_is_known() -> None:
    assert SEARCH.queries == []
    assert (SEARCH.code, SEARCH.error, SEARCH.raw) == (None, None, [])
    cited = Citation(kind=CitationKind.URL, url="https://example.com")
    assert (cited.title, cited.start_index, cited.file_id) == (None,) * 3
