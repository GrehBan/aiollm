"""What every provider client does the same way, run for each provider.

The scenarios of ``aiollm.testing.contract.clients`` run against each
provider's client; what differs between providers (the error resolvers,
Anthropic's ``default_max_tokens``) stays in the provider's own module.
"""

from __future__ import annotations

from aiollm.http.auth import APIKeyAuth, BearerAuth
from aiollm.providers.anthropic import (
    ANTHROPIC_API,
    AnthropicClient,
    AnthropicErrorResolver,
)
from aiollm.providers.anthropic.api import PROVIDER as ANTHROPIC_PROVIDER
from aiollm.providers.openai.api import OPENAI_API
from aiollm.providers.openai.api import PROVIDER as OPENAI_PROVIDER
from aiollm.providers.openai.client import OpenAIClient
from aiollm.providers.openai.resolver import OpenAIErrorResolver
from aiollm.testing.contract.clients import ClientCase, ClientContract

OPENAI = ClientCase(
    client=OpenAIClient,
    provider=OPENAI_PROVIDER,
    api=OPENAI_API,
    url=("/models", "https://api.openai.com/v1/models"),
    resolver=OpenAIErrorResolver,
    env="OPENAI_API_KEY",
    auths=(BearerAuth("k"),),
    foreign_resolver=AnthropicErrorResolver,
)

ANTHROPIC = ClientCase(
    client=AnthropicClient,
    provider=ANTHROPIC_PROVIDER,
    api=ANTHROPIC_API,
    url=("/messages", "https://api.anthropic.com/v1/messages"),
    resolver=AnthropicErrorResolver,
    env="ANTHROPIC_API_KEY",
    auths=(BearerAuth("k"), APIKeyAuth("k")),
    foreign_resolver=OpenAIErrorResolver,
)


class TestOpenAIClient(ClientContract):
    case = OPENAI


class TestAnthropicClient(ClientContract):
    case = ANTHROPIC
