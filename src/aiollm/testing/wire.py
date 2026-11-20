"""Checks of a provider's types against the snapshot of its upstream schemas.

A provider package keeps a snapshot of the schemas its provider
publishes (an official OpenAPI file, or the types of an official SDK):
one folder per endpoint, ``<provider>/<endpoint>``, with
``request.schema.json``, ``response.schema.json``, an optional
``stream.schema.json`` and the documented examples ``responses.json``
and ``events.json``. These functions read such a snapshot and compare
the provider's generated models with it, so the library drifting from
the snapshot fails a test; the provider drifting from the library shows
once the snapshot is refreshed. It needs ``jsonschema``, which the
``testing`` extra installs.

Example:
    >>> snapshot = Snapshot(Path("tests/contracts"), "example/chat")
    >>> errors(snapshot.validator("request.schema.json"), body)
    []
    >>> field_problems([snapshot], models, params=False)
    []
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    cast,
    get_args,
    get_origin,
)

from jsonschema import Draft202012Validator

from aiollm.types.base import BaseAPIType
from aiollm.types.unknown import BaseUnknown

if TYPE_CHECKING:
    from types import ModuleType

    from pydantic import BaseModel

REQUEST_SCHEMA: Final = "request.schema.json"
"""The request schema of an endpoint, closed when it is read."""

RESPONSE_FILES: Final = ("response.schema.json", "stream.schema.json")
"""The upstream schemas of the response and event models."""

REQUEST_FILES: Final = (REQUEST_SCHEMA,)
"""The upstream schema of the request parameter models."""


@cache
def _read(path: Path) -> Any:
    """Read and decode one snapshot file, once.

    Args:
        path: The file.

    Returns:
        The decoded JSON.
    """
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Snapshot:
    """The snapshot of one endpoint's upstream schemas and examples.

    Attributes:
        root: The folder of every snapshot of the package.
        name: The endpoint's folder in it, ``<provider>/<endpoint>``.
    """

    root: Path
    name: str

    def has(self, file: str) -> bool:
        """Tell whether the snapshot has a file.

        Args:
            file: The file name, such as ``stream.schema.json``.

        Returns:
            Whether it exists; an endpoint without a stream has no
            stream schema.
        """
        return (self.root / self.name / file).is_file()

    def load(self, file: str) -> Any:
        """Read one file of the snapshot.

        Args:
            file: The file name.

        Returns:
            The decoded JSON.
        """
        return _read(self.root / self.name / file)

    def validator(self, file: str) -> Draft202012Validator:
        """Build a validator of a schema of the snapshot.

        Args:
            file: The schema's file name.

        Returns:
            The validator; the request schema is closed (see ``closed``).
        """
        schema = self.load(file)
        if file == REQUEST_SCHEMA:
            schema = closed(schema)
        return Draft202012Validator(schema)

    def examples(self, file: str) -> list[Any]:
        """Take the documented examples of the snapshot.

        Args:
            file: ``responses.json`` or ``events.json``, a list of
                ``{"body": ...}`` entries.

        Returns:
            The example bodies.
        """
        return [item["body"] for item in self.load(file)]


def closed(schema: Any) -> Any:
    """Close every object of a schema that lists its properties.

    The upstream request schemas leave ``additionalProperties`` out,
    which JSON Schema reads as "anything goes", while the APIs reject
    unknown fields.

    Args:
        schema: A schema or a node of one.

    Returns:
        A copy with ``additionalProperties: false`` wherever an object
        has ``properties`` and does not say otherwise.
    """
    if isinstance(schema, list):
        items: list[Any] = schema
        return [closed(value) for value in items]
    if not isinstance(schema, dict):
        return schema
    result = {
        key: closed(value)
        for key, value in cast("dict[str, Any]", schema).items()
    }
    if "properties" in result and "additionalProperties" not in result:
        result["additionalProperties"] = False
    return result


def errors(validator: Draft202012Validator, value: Any) -> list[str]:
    """List the schema errors of a value with their paths.

    Args:
        validator: The schema's validator.
        value: The value.

    Returns:
        One ``path: message`` line per error; empty for a valid value.
    """
    return [
        f"{error.json_path}: {error.message[:200]}"
        for error in validator.iter_errors(value)
    ]


def snapshot_schemas(
    snapshots: Iterable[Snapshot], files: Sequence[str]
) -> Iterator[dict[str, Any]]:
    """Read the schemas of several snapshots that exist.

    Args:
        snapshots: The endpoints.
        files: Schema file names; one a snapshot lacks is skipped.

    Yields:
        Each schema.
    """
    for snapshot in snapshots:
        for file in files:
            if snapshot.has(file):
                yield snapshot.load(file)


def upstream_definitions(
    snapshots: Iterable[Snapshot], files: Sequence[str] = RESPONSE_FILES
) -> dict[str, Any]:
    """Collect the definitions of endpoints' upstream schemas.

    Args:
        snapshots: The endpoints.
        files: The schemas to read: the response and stream ones, or
            the request one.

    Returns:
        Every ``$defs`` entry of the schemas, and each root schema under
        its ``title`` (such as ``Response``), and every object written
        inline in a property under ``<Owner>.<property>``; a nullable
        object (``anyOf`` of it and ``null``) is the object.
    """
    definitions: dict[str, Any] = {}

    for schema in snapshot_schemas(snapshots, files):
        definitions.update(schema.get("$defs", {}))

        if title := schema.get("title"):
            definitions.setdefault(title, schema)

    # A nullable object (anyOf of it and null, as the official OpenAPI
    # files write ResponseError) is checked as the object.
    for key, definition in list(definitions.items()):
        options = definition.get("anyOf", ())
        objects = [o for o in options if o.get("type") != "null"]
        if (
            len(options) == 2  # noqa: PLR2004 - the object and null
            and len(objects) == 1
            and ("properties" in objects[0])
        ):
            definitions[key] = objects[0]

    # An object written inline in a property has no name upstream (the
    # official OpenAPI files do this); it is named <Owner>.<property>.
    for owner, definition in list(definitions.items()):
        for prop, node in definition.get("properties", {}).items():
            for option in [node, *node.get("anyOf", ())]:
                if isinstance(option, dict) and "properties" in option:
                    definitions.setdefault(f"{owner}.{prop}", option)

    return definitions


def resolve(node: Any, definitions: dict[str, Any]) -> Any:
    """Follow a ``$ref`` to its definition.

    Args:
        node: A schema node.
        definitions: The definitions by name.

    Returns:
        The referenced definition, or the node itself.
    """
    while isinstance(node, dict) and "$ref" in node:
        reference = cast("dict[str, Any]", node)["$ref"]
        node = definitions[str(reference).rsplit("/", 1)[-1]]
    return node


def allows_null(node: Any, definitions: dict[str, Any]) -> bool:
    """Tell whether a schema node accepts ``null``.

    Args:
        node: A schema node.
        definitions: The definitions by name.

    Returns:
        Whether ``null`` is valid for the node.
    """
    node = resolve(node, definitions)

    if not isinstance(node, dict):
        return False
    schema = cast("dict[str, Any]", node)

    constrained = ("type", "anyOf", "oneOf", "allOf", "enum", "const")
    if not any(key in schema for key in constrained):
        return True  # {} takes any value, null too

    kind = schema.get("type")
    kinds: list[Any] = kind if isinstance(kind, list) else [kind]
    if "null" in kinds:
        return True

    return any(
        allows_null(option, definitions)
        for key in ("anyOf", "oneOf")
        for option in schema.get(key, [])
    )


def provider_models(
    modules: Iterable[ModuleType],
    covered: Iterable[type[BaseModel]] = (),
) -> Iterator[type[BaseAPIType]]:
    """Find the models of a provider.

    Args:
        modules: The modules of the models, such as every module of the
            provider's ``types`` package.
        covered: Methods whose own fields a test checks otherwise; the
            parameter models they inherit from are left out.

    Yields:
        Every ``BaseAPIType`` subclass defined in the modules, except the
        ``BaseUnknown`` ones, which keep whatever an unknown ``type``
        sends, and the bases of ``covered``.
    """
    mixins = {base for method in covered for base in method.__mro__}
    for module in modules:
        for _, model in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(model, BaseAPIType)
                and model.__module__ == module.__name__
                and not issubclass(model, BaseUnknown)
                and model not in mixins
            ):
                yield model


def upstream_names(model: type[BaseAPIType]) -> tuple[str, ...]:
    """Read the upstream definitions a model declares it mirrors.

    Args:
        model: A provider model with ``__upstream_schema__``, one name
            or several.

    Returns:
        The names.
    """
    names = model.__dict__["__upstream_schema__"]
    return (names,) if isinstance(names, str) else tuple(names)


def is_constant(annotation: Any, default: Any) -> bool:
    """Tell a single-value tag that defaults to its one value.

    Such a field (``type: Literal["function"] = "function"``) is always
    sent, so a schema that requires it is met without the caller.

    Args:
        annotation: The field's type.
        default: Its default.

    Returns:
        Whether the type is a one-value ``Literal`` of the default.
    """
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return get_origin(annotation) is Literal and get_args(annotation) == (
        default,
    )


def naming_problems(
    snapshots: Sequence[Snapshot],
    models: Sequence[type[BaseAPIType]],
    *,
    params: bool,
) -> list[str]:
    """Check that every model names the upstream definition it mirrors.

    Args:
        snapshots: The endpoints the models serve.
        models: The models, from ``provider_models``.
        params: Whether they are request parameter models, checked
            against the request schema, or response and event models.

    Returns:
        One line per model without ``__upstream_schema__`` or naming a
        definition the schemas lack; empty when all is well.
    """
    definitions = upstream_definitions(
        snapshots, REQUEST_FILES if params else RESPONSE_FILES
    )
    problems = [
        f"{model.__name__}: set __upstream_schema__"
        for model in models
        if "__upstream_schema__" not in model.__dict__
    ]
    problems += [
        f"{model.__name__}: no upstream definition {name}"
        for model in models
        if "__upstream_schema__" in model.__dict__
        for name in upstream_names(model)
        if name not in definitions
    ]
    return problems


def field_problems(
    snapshots: Sequence[Snapshot],
    models: Iterable[type[BaseAPIType]],
    *,
    params: bool,
) -> list[str]:
    """Compare every field of the models with its upstream definition.

    Validating examples sees only the values the docs happen to show;
    this compares every modeled field with the schema itself. A response
    field may only be laxer than the schema, to read what is sent; a
    request field follows it both ways: ``None`` only where ``null`` is
    allowed, required only where it is (a one-value tag that defaults to
    its value counts as given).

    Args:
        snapshots: The endpoints the models serve.
        models: The models, each with ``__upstream_schema__``.
        params: Whether they are request parameter models.

    Returns:
        One line per mismatch; empty when all is well.
    """
    files = REQUEST_FILES if params else RESPONSE_FILES
    definitions = upstream_definitions(snapshots, files)
    # upstream_definitions reads a nullable object as the object; its
    # null is read from the definitions as they are.
    nullable = {
        name: definition
        for schema in snapshot_schemas(snapshots, files)
        for name, definition in schema.get("$defs", {}).items()
    }
    problems: list[str] = []

    for model in models:
        for name in upstream_names(model):
            definition = resolve(definitions[name], definitions)
            problems += _model_problems(
                model,
                name,
                definition,
                nullable if params else definitions,
                params=params,
            )

    return problems


def _model_problems(
    model: type[BaseAPIType],
    name: str,
    definition: dict[str, Any],
    definitions: dict[str, Any],
    *,
    params: bool,
) -> list[str]:
    """Compare the fields of one model with one upstream definition.

    Args:
        model: The model.
        name: The definition's name.
        definition: The definition.
        definitions: The definitions ``null`` is looked up in.
        params: Whether the model is a request parameter model.

    Returns:
        One line per mismatch.
    """
    properties: dict[str, Any] = definition.get("properties", {})
    required = set(definition.get("required", []))
    problems: list[str] = []

    for field_name, field in model.model_fields.items():
        wire = field.serialization_alias or field_name
        where = f"{model.__name__}.{field_name} ({name}.{wire})"

        if wire not in properties:
            problems.append(f"{where}: not in the upstream schema")
            continue

        takes_none = field.annotation is Any or type(None) in get_args(
            field.annotation
        )
        null = allows_null(properties[wire], definitions)
        if null and not takes_none:
            problems.append(f"{where}: upstream allows null")
        if params and takes_none and not null:
            problems.append(f"{where}: upstream takes no null")

        if wire not in required and field.is_required():
            problems.append(f"{where}: optional upstream")
        if (
            params
            and wire in required
            and not field.is_required()
            and not is_constant(field.annotation, field.default)
        ):
            problems.append(f"{where}: required upstream")

    return problems
