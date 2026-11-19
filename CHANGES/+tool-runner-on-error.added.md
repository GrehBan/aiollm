`tool_runner.create()` and `tool_runner.stream()` take `on_error`, which turns
the exception of a failed callback (or of serializing its output) into that
call's `ToolResult.output`, so the model reads the failure and the other
callbacks go on, for example `on_error=lambda error: f"Error: {error}"`.
Without it a failure is raised as before.
