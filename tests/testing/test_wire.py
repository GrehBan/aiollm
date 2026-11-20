"""The schema checks of ``aiollm.testing.wire`` on a small snapshot."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from pydantic import MISSING

from aiollm.testing.wire import (
    Snapshot,
    allows_null,
    closed,
    errors,
    field_problems,
    is_constant,
    naming_problems,
    provider_models,
    resolve,
    upstream_definitions,
)
from aiollm.types.base import BaseAPIType
from aiollm.types.unknown import BaseUnknown

META = {"type": "object", "properties": {"n": {"type": "integer"}}}
RESPONSE: dict[str, Any] = {
    "title": "Chat",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "meta": META,
        "error": {"$ref": "#/$defs/Error"},
    },
    "required": ["id"],
    "$defs": {
        "Error": {
            "anyOf": [
                {"type": "object", "properties": {"code": {"type": "string"}}},
                {"type": "null"},
            ]
        }
    },
}
REQUEST: dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"$ref": "#/$defs/ChatParams"}},
    "$defs": {
        "ChatParams": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "kind": {"const": "chat"},
                "user": {"type": ["string", "null"]},
                "limit": {"type": "integer"},
            },
            "required": ["model", "kind"],
        }
    },
}


@pytest.fixture
def snapshot(tmp_path: Path) -> Snapshot:
    """A snapshot of one endpoint without a stream."""
    folder = tmp_path / "example" / "chat"
    folder.mkdir(parents=True)
    files = {
        "response.schema.json": RESPONSE,
        "request.schema.json": REQUEST,
        "responses.json": [{"body": {"id": "c1"}}],
    }
    for name, content in files.items():
        (folder / name).write_text(json.dumps(content), encoding="utf-8")
    return Snapshot(tmp_path, "example/chat")


def test_a_snapshot_reads_its_files(snapshot: Snapshot) -> None:
    assert snapshot.has("response.schema.json")
    assert not snapshot.has("stream.schema.json")
    assert snapshot.examples("responses.json") == [{"id": "c1"}]
    assert errors(snapshot.validator("response.schema.json"), {"id": 1}) == [
        "$.id: 1 is not of type 'string'"
    ]


def test_the_request_schema_is_closed(snapshot: Snapshot) -> None:
    request = snapshot.validator("request.schema.json")

    assert errors(request, {"input": {"model": "m", "kind": "chat"}}) == []
    assert errors(request, {"input": {"model": "m", "kind": "chat", "x": 1}})


def test_closed_keeps_what_a_schema_says() -> None:
    schema = {
        "anyOf": [
            {"properties": {}},
            {"properties": {}, "additionalProperties": True},
        ]
    }

    assert closed(schema) == {
        "anyOf": [
            {"properties": {}, "additionalProperties": False},
            {"properties": {}, "additionalProperties": True},
        ]
    }


def test_definitions_name_roots_inline_objects_and_nullable_ones(
    snapshot: Snapshot,
) -> None:
    definitions = upstream_definitions([snapshot])

    assert definitions["Chat"] == RESPONSE
    assert definitions["Chat.meta"] == META
    assert definitions["Error"] == {
        "type": "object",
        "properties": {"code": {"type": "string"}},
    }


@pytest.mark.parametrize(
    ("node", "null"),
    [
        ({}, True),
        ({"type": "null"}, True),
        ({"type": ["string", "null"]}, True),
        ({"type": "string"}, False),
        ({"anyOf": [{"type": "string"}, {"$ref": "#/$defs/Nothing"}]}, True),
        ({"oneOf": [{"type": "string"}]}, False),
        (True, False),
    ],
)
def test_allows_null(node: Any, null: bool) -> None:
    definitions = {"Nothing": {"type": "null"}}

    assert allows_null(node, definitions) is null


def test_resolve_follows_references() -> None:
    definitions = {"A": {"$ref": "#/$defs/B"}, "B": {"type": "string"}}

    assert resolve({"$ref": "#/$defs/A"}, definitions) == {"type": "string"}


class ChatParams(BaseAPIType):
    """A request model as its upstream schema has it."""

    __upstream_schema__ = "ChatParams"
    model: str
    kind: Literal["chat"] = "chat"
    user: str | None | MISSING = MISSING


class Chat(BaseAPIType):
    """A response model."""

    __upstream_schema__ = ("Chat",)
    id: str
    note: str | None = None


class Unnamed(BaseAPIType):
    """A model without an upstream name."""

    id: str


class Unknown(BaseUnknown):
    """Kept whatever it holds; never checked."""


def module(*models: type) -> types.ModuleType:
    """A module that defines the models."""
    found = types.ModuleType("example_types")
    for model in models:
        model.__module__ = found.__name__
        setattr(found, model.__name__, model)
    return found


def test_provider_models_leave_out_unknowns_and_covered_bases() -> None:
    class Method(ChatParams):
        """A method built on a parameter model."""

    types_module = module(ChatParams, Chat, Unknown, Method)

    assert set(provider_models([types_module], [Method])) == {Chat}


def test_matching_models_have_no_problems(snapshot: Snapshot) -> None:
    assert field_problems([snapshot], [Chat], params=False) == []
    assert field_problems([snapshot], [ChatParams], params=True) == []
    assert naming_problems([snapshot], [Chat], params=False) == []


def test_naming_problems(snapshot: Snapshot) -> None:
    class Wrong(BaseAPIType):
        """A model naming a definition that does not exist."""

        __upstream_schema__ = "Nowhere"

    assert naming_problems([snapshot], [Unnamed, Wrong], params=False) == [
        "Unnamed: set __upstream_schema__",
        "Wrong: no upstream definition Nowhere",
    ]


def test_field_problems_both_ways(snapshot: Snapshot) -> None:
    class Params(BaseAPIType):
        """Parameters that disagree with ChatParams."""

        __upstream_schema__ = "ChatParams"
        model: str | None
        kind: str = "chat"
        user: str | MISSING = MISSING
        limit: int
        extra: int = 0

    class Response(BaseAPIType):
        """A response stricter than Chat."""

        __upstream_schema__ = "Chat"
        id: str
        note: str

    where = "Params.{} (ChatParams.{})"
    assert field_problems([snapshot], [Params], params=True) == [
        f"{where.format('model', 'model')}: upstream takes no null",
        f"{where.format('kind', 'kind')}: required upstream",
        f"{where.format('user', 'user')}: upstream allows null",
        f"{where.format('limit', 'limit')}: optional upstream",
        f"{where.format('extra', 'extra')}: not in the upstream schema",
    ]
    assert field_problems([snapshot], [Response], params=False) == [
        "Response.note (Chat.note): upstream allows null",
        "Response.note (Chat.note): optional upstream",
    ]


@pytest.mark.parametrize(
    ("annotation", "default", "constant"),
    [
        (Literal["chat"], "chat", True),
        (Annotated[Literal["chat"], "a tag"], "chat", True),
        (Literal["chat", "x"], "chat", False),
        (str, "chat", False),
    ],
)
def test_is_constant(annotation: Any, default: Any, constant: bool) -> None:
    assert is_constant(annotation, default) is constant
