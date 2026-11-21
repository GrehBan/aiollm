"""JSON types that requests are made of."""

from __future__ import annotations

type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)
"""A JSON value, as ``json.loads`` gives it."""

type JSONSchema = dict[str, JSONValue]
"""A JSON Schema, as a JSON object."""
