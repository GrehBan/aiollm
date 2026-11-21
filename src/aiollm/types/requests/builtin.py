"""Built-in tools of a request: tools the provider runs itself."""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import MISSING, ConfigDict, Field

from aiollm.enums import Provider
from aiollm.types.base import BaseAPIType
from aiollm.types.requests.schema import JSONValue


class UserLocation(BaseAPIType):
    """Roughly where the end user is, for a web search.

    Immutable, and unknown fields are rejected. Every field is left out
    until it is given.

    Attributes:
        country: The country as a two-letter ISO code, such as ``"US"``.
        region: A region, such as a state: ``"California"``.
        city: A city: ``"San Francisco"``.
        timezone: An IANA time zone: ``"America/Los_Angeles"``.

    Example:
        >>> UserLocation(city="Paris", country="FR")
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    country: str | MISSING = Field(default=MISSING, min_length=1)
    region: str | MISSING = Field(default=MISSING, min_length=1)
    city: str | MISSING = Field(default=MISSING, min_length=1)
    timezone: str | MISSING = Field(default=MISSING, min_length=1)


class WebSearch(BaseAPIType):
    """Let the model search the web, at the provider.

    A built-in tool both providers have, with the options both take;
    the provider's other options (OpenAI's ``search_context_size``,
    Anthropic's ``blocked_domains`` and ``max_uses``, a version of the
    tool) go through a ``ProviderTool`` of the provider's package.
    Immutable, and unknown fields are rejected.

    Attributes:
        type: Always ``"web_search"``.
        allowed_domains: Search only these domains; at least one.
        user_location: Where the user is, to localize the results.

    Example:
        >>> GenerationRequest(
        ...     model="gpt-5",
        ...     input="What happened today?",
        ...     tools=WebSearch(allowed_domains=["example.com"]),
        ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["web_search"] = "web_search"
    allowed_domains: Annotated[list[str], Field(min_length=1)] | MISSING = (
        MISSING
    )
    user_location: UserLocation | MISSING = MISSING


class CodeExecution(BaseAPIType):
    """Let the model run code in the provider's sandbox.

    A built-in tool both providers have. It has no shared options: the
    container, its memory and network are each provider's own, set
    through a ``ProviderTool``. Immutable, and unknown fields are
    rejected.

    Attributes:
        type: Always ``"code_execution"``.

    Example:
        >>> GenerationRequest(
        ...     model="gpt-5",
        ...     input="What is the 50th Fibonacci number?",
        ...     tools=CodeExecution(),
        ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["code_execution"] = "code_execution"


class ProviderTool(BaseAPIType):
    """A built-in tool of one provider, as that provider takes it.

    For a tool that is not shared (file search, web fetch, MCP, ...) or
    that has options the shared ones lack. A provider package makes
    them from its generated tool parameters, so a provider's own
    constructor checks the fields; the core only keeps the object. Only
    that provider's adapter takes it: any other raises
    ``RequestValidationError``, as for a ``ProviderFile``. Immutable,
    and unknown fields are rejected.

    Attributes:
        type: Always ``"provider_tool"``.
        provider: The provider whose tool it is.
        definition: The tool as the provider's request holds it, a JSON
            object, such as ``{"type": "file_search", ...}``.

    Example:
        >>> ProviderTool(
        ...     provider=Provider("openai"),
        ...     definition={
        ...         "type": "web_search",
        ...         "search_context_size": "high",
        ...     },
        ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["provider_tool"] = "provider_tool"
    provider: Provider
    definition: dict[str, JSONValue] = Field(min_length=1)


type BuiltinTool = WebSearch | CodeExecution | ProviderTool
"""A tool the provider runs: a shared one, or one of the provider."""

BUILTIN_TOOLS: Final = (WebSearch, CodeExecution, ProviderTool)
"""The classes of ``BuiltinTool``, for ``isinstance``."""
