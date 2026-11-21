# Built-in tools

A built-in tool (OpenAI calls it hosted, Anthropic a server tool) is a tool the *provider* runs: the model searches the web or runs code, the provider executes it and puts the call and its result into the response. Unlike a function tool ([Tool calls](tools.md)), there is nothing for your code to run and no `ToolResult` to send back. The library has three jobs here: send the tool in the request, show what the provider did in the response and the stream, and keep what the provider needs to go on with the conversation.

!!! note "Provider support"
    The shared types below are in `aiollm-core` 0.16. A provider package takes them from its own release: the built-in tools of `aiollm-openai` and `aiollm-anthropic` are merged and come with their 0.16 releases; 0.15 does not send them. Each provider package also has a `tools` module (`aiollm_openai.tools`, `aiollm_anthropic.tools`) that builds its own tools as a `ProviderTool`, checked against the provider's request parameters. A provider that does not run a tool raises `RequestValidationError` for it rather than leave it out.

## In the request

Built-in tools go in the same `tools=` as functions, one list: `tools=[get_weather, WebSearch()]`. The shared ones are what both providers have:

- `WebSearch(allowed_domains=..., user_location=...)`: a web search. `allowed_domains` limits it to the domains, at least one; `UserLocation(country=, region=, city=, timezone=)` localizes the results (every field is optional).
- `CodeExecution()`: code run in the provider's sandbox. It has no shared options: the container, its memory and its network are each provider's own.
- `ProviderTool(provider=, definition=)`: a tool of one provider as that provider takes it, for everything that is not shared (file search, web fetch, MCP, the options the shared tools lack: OpenAI's `search_context_size`, Anthropic's `blocked_domains` and `max_uses`, the version of an Anthropic tool). The `definition` is a JSON object, the tool as the provider's request holds it. Only that provider takes it; any other raises `RequestValidationError`, as it does for another provider's `ProviderFile`.

```python title="builtin_tools.py"
from typing import Any

from aiollm import (
    CodeExecution,
    GenerationResponse,
    IGenerationClient,
    UserLocation,
    WebSearch,
)


async def research(client: IGenerationClient) -> GenerationResponse[Any]:
    response = await client.generation.create(
        model="gpt-6-astra",
        input="What is the weather in Paris today?",
        tools=[
            WebSearch(
                allowed_domains=["example.com"],
                user_location=UserLocation(country="FR", city="Paris"),
            ),
            CodeExecution(),
        ],
    )
    for call in response.server_tool_calls:
        print(call.kind.value, call.status.value, call.queries)
    for citation in response.citations:
        print(citation.url, citation.title)
    return response
```

```python title="builtin_tools_provider.py"
from aiollm import Provider, ProviderTool

file_search = ProviderTool(
    provider=Provider("openai"),
    definition={"type": "file_search", "vector_store_ids": ["vs_123"]},
)
```

- **One list, two kinds.** `request.tools` holds the function tools and `request.builtin_tools` the built-in ones; a request may have built-in tools only (`request.tools` is then `MISSING`). `WebSearch` and `CodeExecution` come at most once; `ProviderTool`s may repeat.
- **`tool_choice` is for functions.** A `ForcedTool` names a function tool; to make the model use a built-in tool, say so in the prompt.
- **Not left out.** A tool the provider cannot run raises `RequestValidationError` (`field="tools"`) before anything is sent: an answer without the search the caller asked for would be misleading.

## In the response

- **`server_tool_calls`** lists the calls the provider ran, in order: a `ServerToolCall` with the call's `id`, `kind` (`ServerToolKind`: `WEB_SEARCH`, `FILE_SEARCH`, `CODE_EXECUTION`, `WEB_FETCH`, `IMAGE_GENERATION`, `MCP`, `OTHER`), `status` (`ServerToolStatus`: `COMPLETED`, `INCOMPLETE`, `FAILED`; `IN_PROGRESS` only in a stream event), `queries` of a search, the `code` a code execution ran, and the `error` of a failed one. The result itself (the pages found, the output of the code) is not normalized: it is in `call.raw`, the provider's items of the call.
- **A failed tool is a call, not an error.** Both providers answer HTTP 200 with the tool's failure inside the response; so does the library: the call has `status=FAILED` and its `error`, and `create()` returns. Check the calls if the answer depends on them.
- **`tool_calls` is for functions only.** The tool runner and the rule "a response with calls waits for their results" are unchanged by built-in tools.
- **`citations`** lists the sources the text rests on: a `Citation` with a `kind` (`CitationKind`: `URL`, `FILE`, `CONTAINER_FILE`), the `url`, `title`, `file_id`, `container_id`, `filename`, the `cited_text` and where it sits in `text` (`start_index`, `end_index`), as far as the provider tells them. What is the provider's alone (Anthropic's `encrypted_index`) stays in `raw` and `native`.
- **The calls are billed.** Searches, containers and the tokens of what they bring cost money at the provider; the library keeps no prices. `server_tool_calls` is there so that you can count them.

## In a stream

A built-in call is two events: a `ServerToolCallEvent` with `phase` `STARTED` (the call as it is known then, `status` `IN_PROGRESS`), then one with `phase` `DONE` (the whole call). The deltas of the code a tool writes and the partial images stay `OtherEvent`s. `CompletedEvent` has `server_tool_calls` and `citations` as the response does.

```python title="builtin_tools_stream.py"
from typing import Any

from aiollm import (
    CompletedEvent,
    IGenerationClient,
    ServerToolCallEvent,
    WebSearch,
)


async def search_streamed(
    client: IGenerationClient,
) -> CompletedEvent[Any]:
    async with client.generation.stream(
        model="gpt-6-astra",
        input="What is the weather in Paris today?",
        tools=WebSearch(),
    ) as events:
        async for event in events:
            if isinstance(event, ServerToolCallEvent):
                print(event.phase.value, event.call.kind.value)
        return await events.final()
```

## Continuing a conversation

The provider needs its own record of a turn that used built-in tools: Anthropic's search results carry `encrypted_content` that must come back as it was, and OpenAI's search calls and reasoning items are part of its record too. So `response.to_turn()` gives an `assistant` turn with `native` (the response's `native` output) when the response has `server_tool_calls`, and the provider sends it in place of the turn's text; another provider cannot read it and sends the text. A response without built-in calls gives a plain turn, as before.

A long turn of built-in tools can be **paused** by the provider (Anthropic's `pause_turn`): `incomplete_reason` is `IncompleteReason.PAUSED`, the text may be empty, and the turn is not finished. `to_turn()` gives a turn with `native` even without text, and a conversation may *end* with such a turn: send it back, with the same tools, to let the provider go on. The library does not do this by itself, since each round is a paid call of its own; the loop is yours, and bounded.

```python title="builtin_tools_continue.py"
from typing import Any

from aiollm import (
    GenerationResponse,
    IGenerationClient,
    IncompleteReason,
    InputItem,
    Turn,
    WebSearch,
)


async def research_to_the_end(
    client: IGenerationClient, question: str, rounds: int = 5
) -> GenerationResponse[Any]:
    conversation: list[InputItem] = [Turn.user(question)]
    for _ in range(rounds):
        response = await client.generation.create(
            model="gpt-6-astra",
            input=conversation,
            tools=WebSearch(),
        )
        if response.incomplete_reason is not IncompleteReason.PAUSED:
            return response
        conversation.append(response.to_turn())
    raise RuntimeError("the provider keeps pausing")
```

- **The last turn.** The order rules of [Shared generation](generation.md) say a conversation ends with a `user` turn or a tool result; the one other ending is an `assistant` turn with `native`, as above. Sending it after a generation that was not paused is the provider's to answer (Anthropic refuses a final assistant turn on current models).
- **Only with built-in calls or a pause.** An ordinary text answer still goes back as text: `to_turn()` keeps `native` only for what the text cannot carry.

What each shared field becomes for each provider is listed under [Built-in tools](../unification.md#built-in-tools) in the field mapping.
