**Breaking:** the Anthropic types and request parameters follow the
`anthropic` 2.11.0 SDK, which renamed five error objects of the server tool
results: `BashCodeExecutionToolResultError`, `CodeExecutionToolResultError`,
`TextEditorCodeExecutionToolResultError`, `ToolSearchToolResultError` and
`WebSearchToolResultError` are now named with an `Object` suffix
(`CodeExecutionToolResultErrorObject`, ...), in `providers.anthropic.types`
and `providers.anthropic.methods.params`, and
`WebSearchToolRequestErrorParam` is now
`WebSearchToolRequestErrorObjectParam`. Their fields are unchanged. The
`model` literal lists `claude-fable-5-2` and `claude-mythos-5-2`, and a skill
id may be 128 characters long.
