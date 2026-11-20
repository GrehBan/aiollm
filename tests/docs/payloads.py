"""Payloads of the providers' APIs that the examples of the docs answer with.

The docs show the providers' clients, so their examples run against the
local server with these payloads: OpenAI's reference response and
stream, and Anthropic's reference message. They are copies, kept here so
that the docs need no provider's own tests.
"""

from __future__ import annotations

import json
from typing import Any, Final

from aiollm.testing.contract.generation import CALLS

ANTHROPIC_MODEL: Final = "claude-opus-5-5"

OPENAI_MESSAGE: dict[str, Any] = {
    "type": "message",
    "id": "msg_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41",
    "status": "completed",
    "role": "assistant",
    "content": [
        {
            "type": "output_text",
            "text": "The image depicts a scenic landscape ...",
            "annotations": [],
            "logprobs": [],
        }
    ],
}

OPENAI_USAGE: dict[str, Any] = {
    "input_tokens": 328,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 52,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 380,
}

OPENAI_RESPONSE: dict[str, Any] = {
    "id": "resp_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41",
    "object": "response",
    "access_programs": None,
    "created_at": 1741476777,
    "status": "completed",
    "completed_at": 1741476778,
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "max_output_tokens": None,
    "model": "gpt-6-astra",
    "output": [OPENAI_MESSAGE],
    "parallel_tool_calls": True,
    "previous_response_id": None,
    "reasoning": {"effort": None, "summary": None, "context": None},
    "store": True,
    "temperature": 1,
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": OPENAI_USAGE,
    "user": None,
    "metadata": {},
    "conversation": {"id": "id"},
    "service_tier": "auto",
    "output_text": "The image depicts a scenic landscape ...",
}


def openai_state(kind: str, number: int, **changes: Any) -> dict[str, Any]:
    """A response state event carrying the example response."""
    response = OPENAI_RESPONSE | {"output": []} | changes
    return {"type": kind, "response": response, "sequence_number": number}


ITEM_ID = "msg_67c9"

PART = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}

OPENAI_STREAM: list[dict[str, Any]] = [
    openai_state("response.created", 0, status="in_progress", usage=None),
    openai_state("response.in_progress", 1, status="in_progress", usage=None),
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "id": ITEM_ID,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        },
        "sequence_number": 2,
    },
    {
        "type": "response.content_part.added",
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
        "part": PART,
        "sequence_number": 3,
    },
    {
        "type": "response.output_text.delta",
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
        "delta": "Hi",
        "sequence_number": 4,
        "logprobs": [],
    },
    {
        "type": "response.output_text.annotation.added",
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
        "annotation_index": 0,
        "annotation": {
            "type": "url_citation",
            "url": "https://example.com/",
            "title": "Example",
            "start_index": 0,
            "end_index": 2,
        },
        "sequence_number": 5,
    },
    {
        "type": "response.output_text.done",
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
        "text": "Hi there!",
        "sequence_number": 6,
        "logprobs": [],
    },
    openai_state("response.completed", 7),
]

OPENAI_OVERLOADED: dict[str, Any] = {
    "type": "error",
    "code": "server_is_overloaded",
    "message": "Overloaded",
    "sequence_number": 1,
}
"""An overloaded error, as OpenAI's stream sends it."""


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
    return OPENAI_RESPONSE | {"output": [OPENAI_MESSAGE, *calls]}


def openai_called_stream() -> list[dict[str, Any]]:
    """An OpenAI stream of ``CALLS``, the arguments in two pieces."""
    events: list[dict[str, Any]] = [openai_state("response.created", 0)]
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
        openai_state("response.completed", 0, output=openai_called()["output"])
    )
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return events


ANTHROPIC_MESSAGE: dict[str, Any] = {
    "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
    "container": {
        "id": "container_011CpZohnwH4vuy7gazohgSP",
        "expires_at": "2019-12-27T18:11:19.117Z",
        "skills": [
            {"skill_id": "pdf", "type": "anthropic", "version": "latest"}
        ],
    },
    "content": [
        {
            "citations": [
                {
                    "cited_text": "The grass is green. The sky is blue.",
                    "document_index": 0,
                    "document_title": "My Document",
                    "end_char_index": 0,
                    "file_id": "file_011CNha8iCJcU1wXNR6q4V8w",
                    "start_char_index": 0,
                    "type": "char_location",
                }
            ],
            "text": "Hi! My name is Claude.",
            "type": "text",
        }
    ],
    "diagnostics": {
        "cache_miss_reason": {
            "cache_missed_input_tokens": 0,
            "type": "model_changed",
        }
    },
    "model": ANTHROPIC_MODEL,
    "role": "assistant",
    "stop_details": {
        "category": "cyber",
        "explanation": "This request was declined because it conflicts "
        "with Anthropic's Usage Policy.",
        "type": "refusal",
    },
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "type": "message",
    "usage": {
        "cache_creation": {
            "ephemeral_1h_input_tokens": 0,
            "ephemeral_5m_input_tokens": 0,
        },
        "cache_creation_input_tokens": 2051,
        "cache_read_input_tokens": 2051,
        "inference_geo": "global",
        "input_tokens": 2095,
        "output_tokens": 503,
        "output_tokens_details": {"thinking_tokens": 0},
        "server_tool_use": {"web_fetch_requests": 2, "web_search_requests": 0},
        "service_tier": "standard",
        "speed": "standard",
    },
}
