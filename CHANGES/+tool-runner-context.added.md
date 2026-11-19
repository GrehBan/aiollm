The iterator returned by `tool_runner.create()` is an async context manager:
`async with calls:` cancels and awaits unfinished callbacks when the block is
left early, as `tool_runner.stream()` already did.
