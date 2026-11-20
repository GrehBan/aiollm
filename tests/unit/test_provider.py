"""``Provider``: an open value that every provider package declares."""

from __future__ import annotations

import copy
import pickle
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from aiollm import (
    APIStatusError,
    DocumentPart,
    GenerationRequest,
    Provider,
    ProviderFile,
    RequestValidationError,
    Turn,
)
from aiollm.adapters.validation import check_files
from aiollm.resources.files import FilesResource
from tests.conftest import FakeTransport

ALPHA = Provider("alpha")
BETA = Provider("beta")


class Holder(BaseModel):
    provider: Provider


def test_a_provider_is_its_value() -> None:
    assert ALPHA == "alpha"
    assert Provider("alpha") == ALPHA
    assert ALPHA != BETA
    assert hash(ALPHA) == hash("alpha")
    assert ALPHA.value == "alpha"
    assert type(ALPHA.value) is str
    assert str(ALPHA) == "alpha"
    assert type(str(ALPHA)) is str
    assert f"{ALPHA}" == "alpha"
    assert repr(ALPHA) == "Provider('alpha')"
    assert isinstance(ALPHA, str)


def test_the_core_declares_only_generic() -> None:
    assert Provider.GENERIC == "generic"
    assert isinstance(Provider.GENERIC, Provider)
    assert not hasattr(Provider, "OPENAI")
    assert not hasattr(Provider, "ANTHROPIC")


def test_a_provider_has_no_attributes_of_its_own() -> None:
    with pytest.raises(AttributeError):
        ALPHA.extra = 1  # type: ignore[attr-defined]


def test_a_provider_needs_a_name() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        Provider("")


def test_a_provider_is_made_from_a_str() -> None:
    with pytest.raises(TypeError, match="not int"):
        Provider(cast("Any", 1))


def pickled(provider: Provider) -> Provider:
    restored: Provider = pickle.loads(pickle.dumps(provider))  # noqa: S301
    return restored


@pytest.mark.parametrize(
    "clone",
    [copy.copy, copy.deepcopy, pickled],
    ids=["copy", "deepcopy", "pickle"],
)
def test_a_provider_survives_copy_and_pickle(clone: Any) -> None:
    restored = clone(ALPHA)

    assert type(restored) is Provider
    assert restored == ALPHA


def test_a_field_takes_a_name_and_makes_a_provider() -> None:
    holder = Holder.model_validate({"provider": "alpha"})

    assert type(holder.provider) is Provider
    assert holder.provider == ALPHA
    assert holder.model_dump() == {"provider": ALPHA}
    assert holder.model_dump_json() == '{"provider":"alpha"}'
    assert Holder.model_json_schema()["properties"]["provider"] == {
        "minLength": 1,
        "title": "Provider",
        "type": "string",
    }


def test_a_field_refuses_an_empty_name() -> None:
    with pytest.raises(ValidationError):
        Holder.model_validate({"provider": ""})


@pytest.mark.parametrize(
    "error",
    [
        RequestValidationError("bad", provider=ALPHA, field="input"),
        APIStatusError(status=400, provider=ALPHA),
    ],
    ids=["request", "status"],
)
def test_errors_keep_a_third_party_provider_through_pickle(
    error: RequestValidationError | APIStatusError,
) -> None:
    restored = pickle.loads(pickle.dumps(error))  # noqa: S301

    assert type(restored.provider) is Provider
    assert restored.provider == ALPHA
    assert str(restored) == str(error)


def request_with(file: ProviderFile) -> GenerationRequest:
    return GenerationRequest(
        model="m", input=[Turn.user([DocumentPart(file=file)])]
    )


def test_two_third_party_providers_tell_their_files_apart() -> None:
    request = request_with(ProviderFile(provider=ALPHA, id="file-1"))

    check_files(request, Provider("alpha"))
    with pytest.raises(RequestValidationError) as caught:
        check_files(request, BETA)

    assert caught.value.provider == BETA
    assert caught.value.field == "input"
    assert "stored by alpha, not by beta" in str(caught.value)


class _BetaFiles:
    """A files adapter of ``BETA``; no method is built for a foreign file."""

    provider = BETA

    def to_delete_method(self, file_id: str) -> Any:
        raise AssertionError(file_id)


async def test_the_files_resource_refuses_a_third_party_file() -> None:
    files = FilesResource(FakeTransport(), cast("Any", _BetaFiles()))

    with pytest.raises(RequestValidationError) as caught:
        await files.delete(ProviderFile(provider=ALPHA, id="file-1"))

    assert caught.value.provider == BETA
    assert caught.value.field == "file"
    assert "stored by alpha, not by beta" in str(caught.value)
