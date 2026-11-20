"""The checks every adapter runs on a call's provider options."""

from __future__ import annotations

import pytest

from aiollm.adapters import checked_options, merge_options
from aiollm.enums import Provider
from aiollm.exceptions import RequestValidationError
from aiollm.testing.echo import PROVIDER
from aiollm.testing.echo.types import EchoGenerationOptions
from aiollm.types.requests import NoProviderOptions


def test_checked_options_copies_the_known_keys() -> None:
    given: dict[str, object] = {"temperature": 0.5}
    checked = checked_options(given, EchoGenerationOptions, PROVIDER)

    assert checked == given
    assert checked is not given


def test_checked_options_refuses_anything_but_a_mapping() -> None:
    with pytest.raises(RequestValidationError, match="not list") as caught:
        checked_options([], EchoGenerationOptions, PROVIDER)

    assert caught.value.field == "provider_options"


def test_checked_options_needs_a_typed_dict() -> None:
    with pytest.raises(TypeError, match="not a TypedDict"):
        checked_options({}, dict, PROVIDER)


def test_no_provider_options_takes_no_key() -> None:
    with pytest.raises(RequestValidationError, match="store"):
        checked_options({"store": 1}, NoProviderOptions, Provider.GENERIC)


def test_merge_options_fills_the_rest_of_an_object() -> None:
    fields = {"reasoning": {"effort": "low"}, "model": "m"}
    options = {"reasoning": {"summary": "auto"}, "store": False}

    merged = merge_options(fields, options, PROVIDER)

    assert merged == {
        "reasoning": {"effort": "low", "summary": "auto"},
        "model": "m",
        "store": False,
    }
    assert fields == {"reasoning": {"effort": "low"}, "model": "m"}


def test_merge_options_refuses_a_key_both_sides_set() -> None:
    fields = {"reasoning": {"effort": "low"}}

    with pytest.raises(RequestValidationError, match="shared field") as caught:
        merge_options(fields, {"reasoning": {"effort": "high"}}, PROVIDER)

    assert caught.value.field == "reasoning.effort"
