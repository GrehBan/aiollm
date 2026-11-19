"""Shared request for text generation."""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Final, Literal, Self, cast, overload

from pydantic import (
    MISSING,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    Tag,
    field_validator,
    model_validator,
)
from pydantic.experimental.arguments_schema import generate_arguments_schema
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import CoreSchema, SchemaValidator
from typing_extensions import TypedDict

from aiollm.enums import Effort, Provider, Role, ToolChoice
from aiollm.types.base import BaseAPIType
from aiollm.types.requests.base import BaseRequest
from aiollm.types.requests.content import ContentPart


class Turn(BaseAPIType):
    """One turn of a conversation: who wrote it and what.

    Immutable, and unknown fields are rejected. System prompts are not
    turns: they go in ``GenerationRequest.instructions``.

    Attributes:
        type: Always ``"text"``; tells the turn apart from the other
            items of a conversation (a turn of parts has it too).
        role: Who wrote the turn.
        content: Text of the turn, or its parts in order: text, images
            and documents, at least one. Only a ``user`` turn takes
            parts; an ``assistant`` turn is text.

    Example:
        >>> Turn.user("Hi")
        Turn(type='text', role=<Role.USER: 'user'>, content='Hi')
        >>> Turn.user([TextPart(text="What is it?"), await
        ...            ImagePart.load("cat.png")])
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    type: Literal["text"] = "text"
    role: Role
    content: str | Annotated[list[ContentPart], Field(min_length=1)]

    @model_validator(mode="after")
    def _parts_from_the_user(self) -> Self:
        """Check that only a ``user`` turn has parts.

        Returns:
            The turn.

        Raises:
            ValueError: A turn of another role has parts.
        """
        if not isinstance(self.content, str) and self.role is not Role.USER:
            message = (
                f"only a {Role.USER.value!r} turn takes parts; "
                f"give a {self.role.value!r} turn its text"
            )
            raise ValueError(message)
        return self

    @classmethod
    def user(cls, content: str | Sequence[ContentPart]) -> Self:
        """Create a turn of the end user.

        Args:
            content: Text of the turn, or its parts.

        Returns:
            The turn.
        """
        if not isinstance(content, str):
            content = list(content)
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Self:
        """Create a turn of the model, such as an earlier answer.

        Args:
            content: Text of the turn.

        Returns:
            The turn.
        """
        return cls(role=Role.ASSISTANT, content=content)


type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)
"""A JSON value, as ``json.loads`` gives it."""

type JSONSchema = dict[str, JSONValue]
"""A JSON Schema, as a JSON object."""


TOOL_NAME_PATTERN: Final = r"^[a-zA-Z0-9_-]{1,128}$"
"""The tool names both providers take: Anthropic's pattern, and the 128
characters OpenAI took in a live check (2026-11-06)."""


class _tool[RT, **P]:
    """Wrap a callable and expose its tool definition.

    The original callable remains directly executable through the
    wrapper. Its metadata is preserved using ``functools.update_wrapper``,
    allowing introspection tools to access the original callable.

    A ``Tool`` definition is generated once during initialization
    and cached for subsequent access through the ``tool`` property.
    The definition can be passed directly to a provider request.

    Attributes:
        callback: The original callable wrapped by this object.
        tool: The generated immutable ``Tool`` definition.
    """

    def __init__(
        self,
        obj: Tool | Callable[P, RT],
        strict: bool | MISSING = MISSING,
        description: str | MISSING = MISSING,
    ) -> None:
        """Initialize the wrapper and generate its tool definition.

        Args:
            obj: The callable to wrap.
            strict: Whether to request strict adherence to the
                generated parameter schema from the provider.
            description: Description of the tool and its purpose.
                Defaults to the callable's docstring. If neither is
                provided, the description is omitted.

        Raises:
            TypeError: The callable cannot be converted into a tool.
        """
        if isinstance(obj, Tool):
            callback = obj._tool_callback  # pyright: ignore[reportPrivateUsage]
            if callback is None:
                raise TypeError("Tool has no associated callable")
            definition = obj
        else:
            callback = obj
            definition = Tool.from_callable(
                callback,
                strict=strict,
                description=description,
            )

        functools.update_wrapper(self, callback)
        self.callback = callback
        self.__tool__ = definition

    def with_updates(
        self,
        strict: bool | MISSING = MISSING,
        description: str | MISSING = MISSING,
    ) -> Tool:
        updates: dict[str, str | bool] = {}
        if strict is not MISSING:
            _check_strict(self.callback, strict)
            updates["strict"] = strict
        if description is not MISSING:
            updates["description"] = description
        return self.tool.model_copy(update=updates)

    @property
    def tool(self) -> Tool:
        """The cached tool definition."""
        return self.__tool__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> RT:
        """Invoke the wrapped callable with the supplied arguments.

        Arguments are forwarded unchanged to the original callable.
        The result is returned without modification.

        Args:
            *args: Positional arguments forwarded to the callable.
            **kwargs: Keyword arguments forwarded to the callable.

        Returns:
            The result of the original callable.

        Raises:
            Exception: Any exception raised by the original callable
                is propagated unchanged.
        """
        return self.callback(*args, **kwargs)


@overload
def tool[RT, **P](
    obj: Tool | Callable[P, RT],
    /,
    *,
    description: str | MISSING = MISSING,
    strict: bool | MISSING = MISSING,
) -> _tool[RT, P]: ...


@overload
def tool[RT, **P](
    *,
    description: str | MISSING = MISSING,
    strict: bool | MISSING = MISSING,
) -> Callable[[Callable[P, RT]], _tool[RT, P]]: ...


def tool[RT, **P](
    obj: Tool | Callable[P, RT] | None = None,
    /,
    *,
    description: str | MISSING = MISSING,
    strict: bool | MISSING = MISSING,
) -> _tool[RT, P] | Callable[[Callable[P, RT]], _tool[RT, P]]:
    """Decorate a callable and expose its tool definition.

    Supports both direct and configured decorator syntax. The
    resulting wrapper preserves the original callable's calling
    interface and metadata.

    A ``Tool`` definition is generated once when the callable is
    decorated and exposed through the wrapper's ``tool`` property.
    The wrapper itself can be passed to ``GenerationRequest`` or
    ``generation.create(tools=...)``; the cached definition is reused.
    Use ``.tool`` when a ``Tool`` object is needed separately.

    The decorator does not execute tool calls returned by the
    provider; it only wraps the callable and provides its definition.

    Args:
        obj: The callable to wrap. Omit it to configure the
            decorator before applying it.
        description: Description of the tool and its purpose.
            Defaults to the callable's docstring. If neither is
            provided, the description is omitted.
        strict: Whether to request strict adherence to the generated
            parameter schema from the provider.

    Returns:
        A callable wrapper when ``obj`` is provided, or a decorator
        accepting a callable otherwise.

    Examples:
        Direct decoration:

        >>> @tool
        ... def get_weather(city: str) -> dict:
        ...     \"\"\"Get current weather in a city.\"\"\"
        ...     return {"city": city}
        ...
        >>> get_weather("Paris")
        {'city': 'Paris'}
        >>> request = GenerationRequest(
        ...     model="gpt-5", input="Weather?", tools=get_weather
        ... )

        Configured decoration:

        >>> @tool(
        ...     description="Search for items matching a query.",
        ...     strict=True,
        ... )
        ... def search(query: str, limit: int) -> list:
        ...     return []
        ...
        >>> request = GenerationRequest(
        ...     model="gpt-5", input="Search", tools=[get_weather, search]
        ... )
    """
    ...
    if obj is None:
        return lambda callback: _tool(
            callback, strict=strict, description=description
        )

    return _tool(obj, strict=strict, description=description)


def _target(obj: Callable[..., Any]) -> Callable[..., Any]:
    """Return the callable a chain of ``functools.partial`` wraps.

    Args:
        obj: A callable, possibly a partial.

    Returns:
        The innermost callable.
    """
    while isinstance(obj, functools.partial):
        obj = obj.func
    return obj


def _arguments_source(obj: Callable[..., Any]) -> Callable[..., Any]:
    """Return the callable whose signature defines a tool's arguments.

    Pydantic reads the signature of functions, methods, classes and
    wrappers (``__wrapped__``), and of a partial of one of them; an
    object with a ``__call__`` method is read through that method, so
    ``self`` is not an argument. A partial is made anew and names the
    wrapped callable's module, where its annotations are evaluated.

    Args:
        obj: The tool's callable.

    Returns:
        ``obj``, or the same callable with every object that has a
        ``__call__`` method replaced by that bound method.
    """
    if isinstance(obj, functools.partial):
        inner = _arguments_source(obj.func)
        source = functools.partial(inner, *obj.args, **obj.keywords)
        # Pydantic evaluates the annotations in the module the callable
        # names, which is functools for a partial; the wrapped
        # function's module resolves them (postponed annotations).
        source.__module__ = _target(inner).__module__
        return source
    if (
        inspect.isroutine(obj)
        or inspect.isclass(obj)
        or hasattr(obj, "__wrapped__")
    ):
        return obj
    call: Callable[..., Any] = obj.__call__  # type: ignore[operator]
    return call


def _check_strict(obj: Callable[..., Any], strict: bool | MISSING) -> None:
    """Refuse a strict tool whose callable takes ``**kwargs``.

    The arguments of ``**kwargs`` are an object of any keys, which a
    strict schema cannot describe: both providers need
    ``additionalProperties: false`` on every object of it.

    Args:
        obj: The tool's callable.
        strict: The tool's ``strict``.

    Raises:
        TypeError: ``strict`` is ``True`` and the callable takes
            ``**kwargs``.
    """
    if strict is not True:
        return
    parameters = inspect.signature(_arguments_source(obj)).parameters
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        message = "A strict tool cannot take **kwargs"
        raise TypeError(message)


def _set_tool_internals(
    obj: Tool,
    *,
    core: CoreSchema,
    callback: Callable[..., Any] | None = None,
) -> None:
    obj._tool_schema_validator = SchemaValidator(core)  # pyright: ignore[reportPrivateUsage]
    obj._tool_callback = callback  # pyright: ignore[reportPrivateUsage]


class Tool(BaseAPIType):
    """A function in your code that the model may call.

    Immutable, and unknown fields are rejected.

    Attributes:
        name: Name the model calls it by: letters, digits, ``_`` and
            ``-``, up to 128 characters; unique in a request.
        description: What the tool does and when to use it.
        parameters: JSON Schema of its arguments, an object schema; a
            tool without arguments takes ``{"type": "object",
            "properties": {}}``. Sent as it is.
        strict: Whether the arguments must follow ``parameters``
            exactly; the schema then needs ``additionalProperties:
            false`` and every property in ``required``. Left out, neither
            provider is strict.

    Example:
        >>> Tool(
        ...     name="get_weather",
        ...     description="Current weather in a city.",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {"city": {"type": "string"}},
        ...         "required": ["city"],
        ...     },
        ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str | MISSING = MISSING
    parameters: JSONSchema
    strict: bool | MISSING = MISSING
    _tool_schema_validator: SchemaValidator | None = PrivateAttr(default=None)
    _tool_callback: Callable[..., Any] | None = PrivateAttr(default=None)
    _tool_arguments_model: type[BaseModel] | None = PrivateAttr(default=None)

    @classmethod
    def from_callable[RT, **P](
        cls,
        obj: Callable[P, RT],
        *,
        description: str | MISSING = MISSING,
        strict: bool | MISSING = MISSING,
    ) -> Self:
        """Create a tool from a callable's signature and annotations.

        The callable's ``__name__`` is used as the tool name; a
        ``functools.partial`` takes the name of the function it wraps,
        and an object with a ``__call__`` method the name of its class.
        Its parameters and annotations are converted to JSON Schema
        using Pydantic's arguments schema generation: a partial's bound
        arguments are left out, an object's ``__call__`` is read without
        ``self``, ``*args`` is an array property named after it and
        ``**kwargs`` an object property named after it. Any other key is
        refused, so the schema has ``additionalProperties: false``.

        If ``description`` is not provided, the callable's docstring
        is used (a partial's function's, an object's class's or else
        its ``__call__``'s). If it has none, the description is omitted.

        The callable is retained as the tool callback. Defining the
        tool does not invoke it; ``generation.tool_runner`` invokes it
        for a matching tool call after a complete response.

        Args:
            obj: The callable whose signature defines the tool's
                arguments.
            description: What the tool does and when to use it.
                Defaults to the callable's docstring.
            strict: Whether the arguments must follow the generated
                schema exactly.

        Returns:
            The tool.

        Raises:
            TypeError: ``strict`` is ``True`` and the callable takes
                ``**kwargs``, which a strict schema cannot describe.

        Example:
            >>> def get_weather(city: str, units: str = "celsius"):
            ...     ...
            >>> tool = Tool.from_callable(
            ...     get_weather,
            ...     description="Get current weather in a city.",
            ... )
        """
        if not isinstance(obj, _tool):
            return cls._from_callable(
                obj=obj, description=description, strict=strict
            )
        return cast(Self, obj.with_updates(strict, description))

    @classmethod
    def _from_callable[RT, **P](
        cls,
        obj: Callable[P, RT],
        *,
        description: str | MISSING = MISSING,
        strict: bool | MISSING = MISSING,
    ) -> Self:
        target = _target(obj)
        name = getattr(target, "__name__", None)
        if name is None:
            name = type(target).__name__

        if description is MISSING:
            description = (
                inspect.getdoc(target)
                or inspect.getdoc(_arguments_source(target))
                or MISSING
            )

        _check_strict(obj, strict)
        core = generate_arguments_schema(_arguments_source(obj))
        schema = GenerateJsonSchema().generate(core)
        schema.setdefault("additionalProperties", False)
        result = cls(
            name=name,
            description=description,
            parameters=schema,
            strict=strict,
        )
        _set_tool_internals(
            result,
            core=core,
            callback=obj,
        )
        return result

    @classmethod
    def from_model(
        cls,
        name: str,
        model: type[BaseModel],
        *,
        description: str | MISSING = MISSING,
        strict: bool | MISSING = MISSING,
    ) -> Self:
        """Create a tool whose arguments are a Pydantic model.

        ``parameters`` is the model's JSON Schema
        (``model.model_json_schema()``), sent as it is, and
        ``ToolCall.parse_arguments(model)`` reads the arguments back
        into the model. For ``strict=True`` the model forbids extra
        fields (``extra="forbid"``) and every field is required, or the
        providers answer 400.

        Args:
            name: Name the model calls it by.
            model: The model of the arguments.
            description: What the tool does and when to use it.
            strict: Whether the arguments must follow the schema
                exactly.

        Returns:
            The tool.

        Example:
            >>> class City(BaseModel):
            ...     model_config = ConfigDict(extra="forbid")
            ...     city: str
            >>> tool = Tool.from_model("get_weather", City, strict=True)
        """
        result = cls(
            name=name,
            description=description,
            parameters=model.model_json_schema(),
            strict=strict,
        )
        _set_tool_internals(
            result,
            core=model.__pydantic_core_schema__,
        )
        return result


class ForcedTool(BaseAPIType):
    """Make the model call this tool.

    The ``tool_choice`` that names one tool, apart from the
    ``ToolChoice`` modes, so a tool named ``auto`` is no mode.

    Attributes:
        name: Name of a tool of the request.

    Example:
        >>> ForcedTool(name="get_weather")
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=TOOL_NAME_PATTERN)


@overload
def parse_model_arguments[ModelT: BaseModel](
    model: type[ModelT],
    /,
    arguments: str,
) -> ModelT: ...


@overload
def parse_model_arguments(
    model: Tool,
    /,
    arguments: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]: ...


@overload
def parse_model_arguments(
    model: Callable[..., Any],
    /,
    arguments: str,
) -> Any: ...


@overload
def parse_model_arguments(
    model: type[BaseModel] | Tool | Callable[..., Any],
    /,
    arguments: str,
) -> Any: ...


def parse_model_arguments(
    model: type[BaseModel] | Tool | Callable[..., Any],
    /,
    arguments: str,
) -> Any:
    """Parse and validate JSON arguments using a model, tool or callable.

    Args:
        model: A Pydantic model, a ``Tool`` with an associated argument
            schema, or a callable whose argument schema can be generated.
        arguments: A JSON string containing the arguments to parse.

    Returns:
        The validated Pydantic model instance when ``model`` is a
        Pydantic model. For a ``Tool`` or callable, returns a tuple
        containing positional and keyword arguments (``args``, ``kwargs``).

    Raises:
        TypeError: The supplied object is not a supported model, tool or
            callable or tool has no associated argument validator.
        ValidationError: The JSON arguments do not satisfy the expected
            schema.
    """
    if isinstance(model, Tool):
        validator = model._tool_schema_validator  # pyright: ignore[reportPrivateUsage]
        if validator is None:
            raise TypeError("Tool has no associated argument validator")
        return validator.validate_json(arguments)

    if inspect.isclass(model) and issubclass(model, BaseModel):
        return model.model_validate_json(arguments)

    if callable(model):
        core = generate_arguments_schema(_arguments_source(model))
        validator = SchemaValidator(core)
        return validator.validate_json(arguments)

    raise TypeError(
        f"Expected a Pydantic model, Tool or callable, "
        f"got {type(model).__name__}"
    )


class ToolCall(BaseAPIType):
    """A call of one of the request's tools.

    Immutable, and unknown fields are rejected. The arguments are the
    JSON text the model wrote, as OpenAI sends them and as both stream
    them; Anthropic's object is written as JSON. Without ``strict`` the
    text may be invalid JSON, so it is parsed only on
    ``parse_arguments()``.

    Attributes:
        id: The call's id, which its result refers to (OpenAI's
            ``call_id``, Anthropic's ``tool_use.id``).
        name: The tool called.
        arguments: The arguments, a JSON object as text.

    Example:
        >>> for call in response.tool_calls:
        ...     result = run(call.name, call.parse_arguments())
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str
    arguments: str

    @overload
    def parse_arguments(self) -> dict[str, JSONValue]: ...

    @overload
    def parse_arguments[ModelT: BaseModel](
        self, model: type[ModelT]
    ) -> ModelT: ...

    @overload
    def parse_arguments(
        self, model: Tool
    ) -> tuple[tuple[Any, ...], dict[str, Any]]: ...

    @overload
    def parse_arguments(self, model: Callable[..., Any]) -> Any: ...

    def parse_arguments(
        self,
        model: type[BaseModel] | Tool | Callable[..., Any] | None = None,
    ) -> Any:
        """Parse the tool call arguments, validating against a schema.

        Args:
            model: An optional Pydantic model, ``Tool`` or callable used to
                parse and validate the arguments. When omitted, the arguments
                are parsed as a JSON object
                without additional schema validation.

        Returns:
            The parsed JSON object when ``model`` is omitted. Otherwise,
            returns the result of ``parse_model_arguments``: a validated
            Pydantic model instance or a tuple of positional and keyword
            arguments (``args``, ``kwargs``).

        Raises:
            TypeError: The supplied tool or callable has no usable argument
                schema.
            ValidationError: The arguments are not satisfy
                the expected schema.
            ValueError: The arguments are not a JSON object.
        """
        if model is None:
            return self._parse_object()
        return parse_model_arguments(model, self.arguments)

    def _parse_object(self) -> dict[str, JSONValue]:
        """Parse the arguments as a JSON object.

        Returns:
            The arguments.

        Raises:
            ValueError: The arguments are not a JSON object.
        """
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError as error:
            message = f"arguments of tool call {self.id} are not JSON"
            raise ValueError(message) from error
        if not isinstance(value, dict):
            message = f"arguments of tool call {self.id} are not an object"
            raise ValueError(message)
        return value


class NativeOutput(BaseAPIType):
    """A response's output as its provider's next request takes it.

    The provider's own record of the turn: OpenAI's output items, its
    reasoning items with ``encrypted_content`` among them, or Anthropic's
    content blocks, thinking with its ``signature`` among them, each as
    the response sent it. The provider's adapter sends it in place of the
    shared fields of the turn, so the model sees its reasoning again;
    another provider's adapter sends the shared fields only, since it
    cannot read it.

    Attributes:
        provider: The provider whose output it is.
        items: The items, in output order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider
    items: Annotated[list[dict[str, JSONValue]], Field(min_length=1)]


class ToolCallTurn(BaseAPIType):
    """A turn of the model that calls tools.

    Made by ``GenerationResponse.to_turn()`` from a response with calls;
    the next items of the conversation are a ``ToolResult`` for each of
    its calls.

    Attributes:
        type: Always ``"tool_calls"``.
        text: Text the model wrote with the calls, if any.
        tool_calls: The calls; at least one.
        native: The response's own output, which its provider gets back
            in place of ``text`` and ``tool_calls``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_calls"] = "tool_calls"
    text: str | MISSING = MISSING
    tool_calls: Annotated[list[ToolCall], Field(min_length=1)]
    native: NativeOutput | MISSING = MISSING


class ToolResult(BaseAPIType):
    """What a tool returned for one call.

    Immutable, and unknown fields are rejected. Describe a failure in
    ``output``: OpenAI has no error flag, and both read the text.

    Attributes:
        type: Always ``"tool_result"``.
        call_id: ``ToolCall.id`` of the call it answers.
        output: The result as text, such as JSON.

    Example:
        >>> ToolResult(call_id=call.id, output='{"celsius": 15}')
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_result"] = "tool_result"
    call_id: str = Field(min_length=1)
    output: str


def _item_tag(value: object) -> object:
    """Tell which item of a conversation a value is.

    A dict without ``type`` is a text turn, as the conversations written
    before the other items were.

    Args:
        value: An item, or a dict of one.

    Returns:
        Its ``type``.
    """
    if isinstance(value, dict):
        return value.get("type", "text")
    return getattr(value, "type", None)


type InputItem = Annotated[
    Annotated[Turn, Tag("text")]
    | Annotated[ToolCallTurn, Tag("tool_calls")]
    | Annotated[ToolResult, Tag("tool_result")],
    Discriminator(_item_tag),
]
"""An item of a conversation: a text turn, a turn of calls or a result."""


class NoProviderOptions(TypedDict, total=False, closed=True):
    """Provider options of a generation resource that takes none.

    The default of ``GenerationResource``'s ``OptionsT``. It is closed
    (PEP 728): only ``{}`` matches it, not even a ``TypedDict`` with keys
    of its own, which an open empty ``TypedDict`` would take. An adapter
    without options still checks what it is given at run time with
    ``checked_options(options, NoProviderOptions, provider)``.
    """


type ToolType = Tool | Callable[..., Any]


class GenerationRequest(BaseRequest):
    """Generate text from an input.

    Attributes:
        model: Model identifier of the provider; not empty.
        input: Text to respond to, taken as one ``user`` turn, or the
            conversation so far, a list of at least one item: text turns,
            turns of tool calls and tool results; an empty text or list
            is refused before anything is sent. The conversation ends
            with a ``user`` turn or a tool result, and the results of a
            turn of calls follow it, one per call; the adapter checks it
            and raises ``RequestValidationError`` otherwise.
        instructions: Instructions for the model given apart from the
            input, such as a system prompt.
        max_output_tokens: Upper bound on generated tokens; positive.
        end_user_id: An opaque id of the end user the request is made
            for, such as a hash, never personal data; providers use it
            to detect abuse. Each provider limits its length.
        effort: How much work the model puts into the response; the
            model's own default when left out.
        output_schema: A JSON Schema the response text must follow; the
            text is then JSON. Each provider takes its own subset of
            JSON Schema and answers 400 to the rest; the schema is sent
            as it is, never rewritten.
        tools: A single ``Tool`` or callable, or a sequence of them,
            describing functions the model may call. Callables are
            converted to ``Tool`` definitions using their signatures
            and annotations; decorated callables reuse their cached
            definitions. The resulting list must contain at least one
            tool, and tool names must be unique.
        tool_choice: Whether the model calls a tool (a ``ToolChoice``) or
            which one (a ``ForcedTool`` naming one of ``tools``); the
            providers' default is ``ToolChoice.AUTO``. Only with
            ``tools``.
        parallel_tool_calls: Whether the model may call several tools in
            one response; both providers allow it by default. Only with
            ``tools``, and not with ``ToolChoice.NONE``.

    An optional field is left out until it is given; ``None`` is not
    taken (it would mean ``null``, which not every provider accepts),
    so leave a field out rather than set it to ``None``.

    Example:
        >>> def get_weather(city: str) -> dict:
        ...     \"\"\"Get current weather in a city.\"\"\"
        ...     ...
        >>> GenerationRequest(
        ...     model="gpt-5",
        ...     input="What is the weather in Paris?",
        ...     tools=get_weather,
        ... )
        >>> GenerationRequest(
        ...     model="gpt-5",
        ...     input="Find the weather.",
        ...     tools=[get_weather, another_tool],
        ... )
    """

    model: str = Field(min_length=1)
    input: (
        Annotated[str, Field(min_length=1)]
        | Annotated[Sequence[InputItem], Field(min_length=1)]
    )
    instructions: str | MISSING = MISSING
    max_output_tokens: int | MISSING = Field(default=MISSING, gt=0)
    end_user_id: str | MISSING = Field(default=MISSING, min_length=1)
    effort: Effort | MISSING = MISSING
    output_schema: JSONSchema | MISSING = MISSING
    tools_: (
        Annotated[Sequence[ToolType] | ToolType, Field(min_length=1)] | MISSING
    ) = Field(
        alias="tools",
        validation_alias="tools",
        serialization_alias="tools",
        default=MISSING,
    )
    tool_choice: ToolChoice | ForcedTool | MISSING = MISSING
    parallel_tool_calls: bool | MISSING = MISSING

    @property
    def tools(self) -> list[Tool] | MISSING:
        """Return the tools after converting callable definitions."""
        return cast(list[Tool] | MISSING, self.tools_)

    @field_validator("tools_", mode="before")
    @classmethod
    def _callables_to_tools(cls, value: Any) -> list[Tool] | MISSING:
        """Normalize tool definitions and callables into a list of tools.

        Args:
            value: A tool, callable, sequence of tools and callables,
                or ``MISSING`` if no tools were provided.

        Returns:
            A list of normalized tool definitions, or ``MISSING``
            if no tools were provided.

        Raises:
            ValueError: The input is not a tool, callable, or sequence
                of tools and callables, or the sequence contains an
                unsupported item.
        """
        if value is MISSING:
            return value
        if isinstance(value, (Tool, _tool)) or callable(value):
            value = [value]
        elif not isinstance(value, Sequence):
            raise ValueError(
                f"Expected a Tool, "
                f"callable or sequence, got {type(value).__name__}"
            )

        result: list[Tool] = []

        for item in value:
            if isinstance(item, Tool):
                result.append(item)
            elif isinstance(item, _tool):
                result.append(item.tool)
            elif callable(item):
                result.append(Tool.from_callable(item))
            else:
                raise ValueError(
                    f"Expected Tool or callable, got {type(item).__name__}"
                )
        return result

    @model_validator(mode="after")
    def _check_tools(self) -> Self:
        """Refuse tool settings that no provider takes.

        Returns:
            The request as it is.

        Raises:
            ValueError: ``tool_choice`` or ``parallel_tool_calls`` comes
                without ``tools``; two tools share a name; a
                ``ForcedTool`` names no tool of the request; or
                ``parallel_tool_calls`` comes with ``ToolChoice.NONE``.
        """
        if self.tools is MISSING:
            if self.tool_choice is not MISSING:
                message = "tool_choice needs tools"
                raise ValueError(message)
            if self.parallel_tool_calls is not MISSING:
                message = "parallel_tool_calls needs tools"
                raise ValueError(message)
            return self
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            message = f"tool names must be unique, got {names}"
            raise ValueError(message)
        choice = self.tool_choice
        if isinstance(choice, ForcedTool) and choice.name not in names:
            message = f"tool_choice names {choice.name!r}, not a tool"
            raise ValueError(message)
        if (
            choice is ToolChoice.NONE
            and self.parallel_tool_calls is not MISSING
        ):
            message = "parallel_tool_calls means nothing with tool_choice none"
            raise ValueError(message)
        return self
