`tool_runner.create()` and `tool_runner.stream()` take `in_order` and
`buffer_size`. `in_order=False` yields each `ToolResult` as soon as its
callback finishes instead of in tool-call order. `buffer_size` lets that many
finished but unread results wait beyond `concurrency` while further callbacks
start, so a slow first callback no longer holds back the ones after it.
