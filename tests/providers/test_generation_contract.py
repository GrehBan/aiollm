"""Shared generation behaves the same for every provider.

The scenarios of ``aiollm.testing.contract.generation`` run against
``OpenAIClient.generation`` and ``AnthropicClient.generation`` on the
local server, with the payloads of each provider's reference examples.
Only the wire format differs per provider; the shared results and errors
must not.
"""

from __future__ import annotations

import json
from typing import Any

from aiollm.providers.anthropic import AnthropicClient
from aiollm.providers.anthropic.api import PROVIDER as ANTHROPIC_PROVIDER
from aiollm.providers.anthropic.client import (
    session_factory as anthropic_factory,
)
from aiollm.providers.anthropic.methods import StreamMessage
from aiollm.providers.openai import OpenAIClient
from aiollm.providers.openai.api import PROVIDER as OPENAI_PROVIDER
from aiollm.providers.openai.client import session_factory as openai_factory
from aiollm.providers.openai.methods import StreamResponse
from aiollm.testing.contract.generation import (
    CALLS,
    SCHEMA,
    GenerationCase,
    GenerationContract,
)
from aiollm.types.responses import TokenUsage
from tests.providers import test_anthropic_messages as anthropic
from tests.providers import test_openai_responses as openai


def openai_called() -> dict[str, Any]:
    """An OpenAI response with text and ``CALLS``."""
    calls = [
        {
            "type": "function_call",
            "id": f"fc_{n}",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
            "status": "completed",
        }
        for n, (call_id, name, arguments) in enumerate(CALLS)
    ]
    return openai.RESPONSE | {"output": [openai.MESSAGE, *calls]}


def openai_called_stream() -> list[dict[str, Any]]:
    """An OpenAI stream of ``CALLS``, the arguments in two pieces."""
    events: list[dict[str, Any]] = [openai.state("response.created", 0)]
    for n, (call_id, name, arguments) in enumerate(CALLS):
        item = {
            "type": "function_call",
            "id": f"fc_{n}",
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "status": "in_progress",
        }
        text = json.dumps(arguments)
        events += [
            {
                "type": "response.output_item.added",
                "output_index": n,
                "item": item,
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": f"fc_{n}",
                "output_index": n,
                "delta": text[:3],
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": f"fc_{n}",
                "output_index": n,
                "delta": text[3:],
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": f"fc_{n}",
                "output_index": n,
                "arguments": text,
            },
            {
                "type": "response.output_item.done",
                "output_index": n,
                "item": item | {"arguments": text, "status": "completed"},
            },
        ]
    events.append(
        openai.state("response.completed", 0, output=openai_called()["output"])
    )
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return events


def anthropic_called() -> dict[str, Any]:
    """An Anthropic message with text and ``CALLS``."""
    blocks = [
        {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
        for call_id, name, arguments in CALLS
    ]
    return anthropic.MESSAGE | {
        "content": [{"type": "text", "text": "Checking."}, *blocks],
        "stop_reason": "tool_use",
        "stop_details": None,
    }


def anthropic_called_stream() -> list[dict[str, Any]]:
    """An Anthropic stream of ``CALLS``: the first call's arguments in
    two pieces, the second's in none, as Anthropic sends ``{}``."""
    start, *_, stop = anthropic.STREAM
    events: list[dict[str, Any]] = [start]
    for index, (call_id, name, arguments) in enumerate(CALLS):
        events.append(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": {},
                },
            }
        )
        if arguments:
            text = json.dumps(arguments)
            events += [
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": part,
                    },
                }
                for part in (text[:3], text[3:])
            ]
        events.append({"type": "content_block_stop", "index": index})
    events += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 15},
        },
        stop,
    ]
    return events


OPENAI = GenerationCase(
    provider=OPENAI_PROVIDER,
    client=OpenAIClient,
    factory=openai_factory,
    path="/v1/responses",
    response=openai.RESPONSE,
    text=openai.MESSAGE["content"][0]["text"],
    usage=TokenUsage(
        input_tokens=328,
        output_tokens=52,
        total_tokens=380,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    ),
    truncated=openai.RESPONSE
    | {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    },
    refused=openai.RESPONSE
    | {
        "output": [
            openai.MESSAGE
            | {"content": [{"type": "refusal", "refusal": "I can't."}]}
        ]
    },
    refusal="I can't.",
    stream=openai.STREAM,
    deltas=["Hi"],
    stream_usage=TokenUsage(
        input_tokens=328,
        output_tokens=52,
        total_tokens=380,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    ),
    overloaded={
        "type": "error",
        "code": "server_is_overloaded",
        "message": "Overloaded",
        "sequence_number": 1,
    },
    quota={
        "type": "error",
        "code": "insufficient_quota",
        "message": "You exceeded your current quota",
        "sequence_number": 1,
    },
    bad_request={
        "error": {
            "code": "invalid_value",
            "message": "Bad",
            "type": "invalid_request_error",
        }
    },
    raw_stream=lambda: StreamResponse(model="model-1", input="Hi"),
    options={"store": False, "temperature": 0.2},
    other_options={"store": True, "top_p": 0.5},
    bad_option={"temperature": 5},
    effort={"reasoning": {"effort": "high"}},
    output_schema={
        "reasoning": {"effort": "high"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "output",
                "schema": {"type": "object", "required": ["n"]},
                "strict": True,
            }
        },
    },
    tools={
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Current weather",
                "parameters": SCHEMA,
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
                "strict": None,
            },
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    },
    forced={"tool_choice": {"type": "function", "name": "get_time"}},
    only_parallel={"parallel_tool_calls": True},
    called=openai_called(),
    called_stream=openai_called_stream(),
    cut_off={
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    },
    input_field="input",
    continued_streamed=[
        *openai_called()["output"],
        *(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"result of {name}",
            }
            for call_id, name, _ in CALLS
        ),
    ],
    continued=[
        *openai_called()["output"],
        *(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"result of {name}",
            }
            for call_id, name, _ in CALLS
        ),
    ],
    continued_shared=[
        {"role": "assistant", "content": "Checking."},
        *(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments),
            }
            for call_id, name, arguments in CALLS
        ),
        *(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"result of {name}",
            }
            for call_id, name, _ in CALLS
        ),
    ],
    parts={
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What is in these?"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,iVBORw0K",
                "detail": "auto",
            },
            {
                "type": "input_image",
                "image_url": "https://example.com/cat.jpg",
                "detail": "auto",
            },
            {
                "type": "input_file",
                "file_data": "data:application/pdf;base64,JVBERi0xLjc=",
                "filename": "report.pdf",
            },
            {
                "type": "input_file",
                "file_data": "data:application/pdf;base64,JVBERi0xLjc=",
                "filename": "document.pdf",
            },
            {
                "type": "input_file",
                "file_url": "https://example.com/report.pdf",
            },
        ],
    },
    after_results=[
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "noon",
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in these?"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0K",
                    "detail": "auto",
                },
            ],
        },
    ],
)

ANTHROPIC = GenerationCase(
    provider=ANTHROPIC_PROVIDER,
    client=AnthropicClient,
    factory=anthropic_factory,
    path="/v1/messages",
    response=anthropic.MESSAGE,
    text="Hi! My name is Claude.",
    usage=TokenUsage(
        input_tokens=6197,
        output_tokens=503,
        total_tokens=6700,
        cache_read_tokens=2051,
        cache_write_tokens=2051,
        reasoning_tokens=0,
    ),
    truncated=anthropic.MESSAGE | {"stop_reason": "max_tokens"},
    refused=anthropic.MESSAGE | {"stop_reason": "refusal"},
    refusal=anthropic.MESSAGE["stop_details"]["explanation"],
    stream=anthropic.STREAM,
    deltas=["Hello", "!"],
    stream_usage=TokenUsage(
        input_tokens=25, output_tokens=15, total_tokens=40
    ),
    overloaded={
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    },
    quota={
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "Spend cap reached",
            "details": {"error_code": "enforced_spend_limit_reached"},
        },
    },
    bad_request={
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "Bad"},
    },
    raw_stream=lambda: StreamMessage(**anthropic.request()),
    options={"stop_sequences": ["END"]},
    other_options={"stop_sequences": ["STOP", "HALT"]},
    bad_option={"stop_sequences": "END"},
    effort={"output_config": {"effort": "high"}},
    output_schema={
        "output_config": {
            "effort": "high",
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "required": ["n"]},
            },
        }
    },
    tools={
        "tools": [
            {
                "name": "get_weather",
                "description": "Current weather",
                "input_schema": SCHEMA,
                "strict": True,
            },
            {
                "name": "get_time",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
    },
    forced={"tool_choice": {"type": "tool", "name": "get_time"}},
    only_parallel={
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": False}
    },
    called=anthropic_called(),
    called_stream=anthropic_called_stream(),
    cut_off={"stop_reason": "max_tokens"},
    input_field="messages",
    continued_streamed=[
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": arguments,
                }
                for call_id, name, arguments in CALLS
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"result of {name}",
                }
                for call_id, name, _ in CALLS
            ],
        },
    ],
    continued=[
        {"role": "assistant", "content": anthropic_called()["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"result of {name}",
                }
                for call_id, name, _ in CALLS
            ],
        },
    ],
    continued_shared=[
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                *(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": arguments,
                    }
                    for call_id, name, arguments in CALLS
                ),
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"result of {name}",
                }
                for call_id, name, _ in CALLS
            ],
        },
    ],
    parts={
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in these?"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0K",
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/cat.jpg",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjc=",
                },
                "title": "report.pdf",
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjc=",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": "https://example.com/report.pdf",
                },
            },
        ],
    },
    after_results=[
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "noon",
                },
                {"type": "text", "text": "What is in these?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0K",
                    },
                },
            ],
        }
    ],
)


class TestOpenAIGeneration(GenerationContract):
    case = OPENAI


class TestAnthropicGeneration(GenerationContract):
    case = ANTHROPIC
