"""Argument schemas of tools made from every form of callable."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any, cast

import pytest
from pydantic import ValidationError

from aiollm import tool
from aiollm.types.requests.generation import Tool, ToolCall


def call_with(arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(id="call-1", name="f", arguments=json.dumps(arguments))


def properties(definition: Tool) -> dict[str, Any]:
    return cast(dict[str, Any], definition.parameters["properties"])


def run(obj: Callable[..., Any], arguments: dict[str, Any]) -> Any:
    """Parse the arguments with the callable's tool and call it with them.

    Returns:
        What the callable returns.
    """
    definition = Tool.from_callable(obj)
    args, kwargs = call_with(arguments).parse_arguments(definition)
    return obj(*args, **kwargs)


def echo(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return args, kwargs


def test_positional_only_parameters_are_named_properties() -> None:
    def f(a: int, b: str, /) -> tuple[int, str]:
        return a, b

    definition = Tool.from_callable(f)

    assert definition.parameters == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "string"},
        },
        "required": ["a", "b"],
        "type": "object",
        "additionalProperties": False,
    }
    assert call_with({"a": 1, "b": "x"}).parse_arguments(definition) == (
        (1, "x"),
        {},
    )
    assert run(f, {"a": 1, "b": "x"}) == (1, "x")


def test_keyword_only_parameters_are_keyword_arguments() -> None:
    def f(*, a: int, b: str = "x") -> tuple[int, str]:
        return a, b

    definition = Tool.from_callable(f)

    assert definition.parameters["required"] == ["a"]
    assert call_with({"a": 1}).parse_arguments(definition) == (
        (),
        {"a": 1, "b": "x"},
    )
    assert run(f, {"a": 1, "b": "y"}) == (1, "y")


def test_variadic_positional_parameters_are_an_array() -> None:
    def f(first: int, *rest: int) -> tuple[int, ...]:
        return (first, *rest)

    definition = Tool.from_callable(f)

    assert properties(definition)["rest"] == {
        "items": {"type": "integer"},
        "title": "Rest",
        "type": "array",
    }
    assert definition.parameters["required"] == ["first"]
    assert call_with({"first": 1, "rest": [2, 3]}).parse_arguments(
        definition
    ) == ((1, 2, 3), {})
    assert run(f, {"first": 1}) == (1,)


def test_variadic_keyword_parameters_are_an_object() -> None:
    def f(name: str, **extra: int) -> dict[str, Any]:
        return {"name": name, **extra}

    definition = Tool.from_callable(f)

    assert properties(definition)["extra"] == {
        "additionalProperties": {"type": "integer"},
        "title": "Extra",
        "type": "object",
    }
    assert call_with({"name": "n", "extra": {"x": 1, "y": 2}}).parse_arguments(
        definition
    ) == (("n",), {"x": 1, "y": 2})
    assert run(f, {"name": "n", "extra": {"x": 1}}) == {"name": "n", "x": 1}


def test_every_kind_of_parameter_keeps_its_place() -> None:
    def f(a: int, /, b: int, *args: int, c: int, **kwargs: int) -> None:
        return None

    definition = Tool.from_callable(f)

    assert definition.parameters["required"] == ["a", "b", "c"]
    assert call_with(
        {"a": 1, "b": 2, "args": [3], "c": 4, "kwargs": {"d": 5}}
    ).parse_arguments(definition) == ((1, 2, 3), {"c": 4, "d": 5})


def test_defaults_are_optional_and_filled_in() -> None:
    def f(city: str, days: int = 1, units: str | None = None) -> None:
        return None

    definition = Tool.from_callable(f)

    assert definition.parameters["required"] == ["city"]
    assert properties(definition)["days"]["default"] == 1
    assert call_with({"city": "Paris"}).parse_arguments(definition) == (
        ("Paris", 1, None),
        {},
    )


def test_a_key_the_callable_does_not_take_is_refused() -> None:
    def f(a: int, **kwargs: int) -> None:
        return None

    definition = Tool.from_callable(f)

    assert definition.parameters["additionalProperties"] is False
    with pytest.raises(ValidationError, match="extra_forbidden"):
        call_with({"a": 1, "b": 2}).parse_arguments(definition)


def test_a_callable_object_is_read_through_its_call_method() -> None:
    class Weather:
        """Get the weather in a city."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, city: str, days: int = 1) -> str:
            self.calls.append(city)
            return f"{city}: {days}"

    weather = Weather()
    definition = Tool.from_callable(weather)

    assert definition.name == "Weather"
    assert definition.description == "Get the weather in a city."
    assert definition.parameters["required"] == ["city"]
    assert "self" not in properties(definition)
    assert run(weather, {"city": "Paris", "days": 2}) == "Paris: 2"
    assert weather.calls == ["Paris"]


def test_a_callable_object_without_a_docstring_uses_its_call_method() -> None:
    class Weather:
        def __call__(self, city: str) -> str:
            """Get the weather in a city."""
            return city

    definition = Tool.from_callable(Weather())

    assert definition.description == "Get the weather in a city."


def test_an_async_callable_object_is_read_through_its_call_method() -> None:
    class Weather:
        async def __call__(self, city: str) -> str:
            return city

    definition = Tool.from_callable(Weather())

    assert definition.parameters["required"] == ["city"]


def test_a_partial_leaves_out_its_bound_arguments() -> None:
    def get_weather(api_key: str, city: str, days: int = 1) -> str:
        """Get the weather in a city."""
        return f"{api_key}/{city}/{days}"

    bound = functools.partial(get_weather, "secret")
    definition = Tool.from_callable(bound)

    assert definition.name == "get_weather"
    assert definition.description == "Get the weather in a city."
    assert definition.parameters["required"] == ["city"]
    assert "api_key" not in properties(definition)
    assert run(bound, {"city": "Paris"}) == "secret/Paris/1"


def test_a_partial_keyword_becomes_a_default() -> None:
    bound = functools.partial(echo, units="metric")

    definition = Tool.from_callable(bound)

    assert definition.name == "echo"
    assert call_with({"kwargs": {"units": "imperial"}}).parse_arguments(
        definition
    ) == ((), {"units": "imperial"})


def test_a_partial_of_a_partial_is_named_after_the_function() -> None:
    def f(a: int, b: int, c: int) -> int:
        return a + b + c

    bound = functools.partial(functools.partial(f, 1), 2)
    definition = Tool.from_callable(bound)

    assert definition.name == "f"
    assert definition.parameters["required"] == ["c"]
    assert run(bound, {"c": 3}) == 6


def test_a_partial_of_a_callable_object_is_read_through_its_call() -> None:
    class Add:
        def __call__(self, a: int, b: int) -> int:
            return a + b

    bound = functools.partial(Add(), 1)
    definition = Tool.from_callable(bound)

    assert definition.name == "Add"
    assert definition.parameters["required"] == ["b"]
    assert run(bound, {"b": 2}) == 3


def test_arguments_of_a_callable_object_are_parsed_without_a_tool() -> None:
    class Add:
        def __call__(self, a: int, b: int = 2) -> int:
            return a + b

    assert call_with({"a": 1}).parse_arguments(Add()) == ((1, 2), {})


def strict_from_callable(f: Callable[..., Any]) -> object:
    return Tool.from_callable(f, strict=True)


def strict_decorator(f: Callable[..., Any]) -> object:
    return tool(strict=True)(f)


def strict_existing_wrapper(f: Callable[..., Any]) -> object:
    return Tool.from_callable(tool(f), strict=True)


@pytest.mark.parametrize(
    "make",
    [strict_from_callable, strict_decorator, strict_existing_wrapper],
)
def test_a_strict_tool_cannot_take_variadic_keywords(
    make: Callable[[Callable[..., Any]], object],
) -> None:
    def f(a: int, **kwargs: int) -> None:
        return None

    with pytest.raises(TypeError, match=r"strict tool cannot take \*\*kwargs"):
        make(f)


def test_a_strict_tool_may_take_variadic_positionals() -> None:
    def f(a: int, *args: int) -> None:
        return None

    assert Tool.from_callable(f, strict=True).strict is True
    assert Tool.from_callable(tool(f), strict=False).strict is False
