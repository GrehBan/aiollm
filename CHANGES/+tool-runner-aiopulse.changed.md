**Breaking:** the tool runner executes callbacks with aiopulse's
`CallbackExecutor`, a new dependency (pinned to a commit until aiopulse is
on PyPI). A callback failure is raised as a `CallbackExceptionGroup` holding
every failure observed before the callbacks stopped, even when there is only
one, instead of the first failure alone; handle it with `except*` or read
its `failures`. `CallbackExceptionGroup`, `CallbackFailure` and
`CallbackCancelledError` are re-exported from `aiollm`. An iteration of
`tool_runner.create()` still waiting in another task when the iterator is
closed ends with `StopAsyncIteration` instead of `CancelledError`.
