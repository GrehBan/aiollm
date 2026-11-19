A tool callback that ended with a cancellation the runner did not request,
such as awaiting a cancelled task, reached the consumer as `CancelledError`,
which looks like a cancellation of the consumer and is swallowed by its
`TaskGroup`. It is now a `CallbackCancelledError` failure in the raised
`CallbackExceptionGroup`, with the original cancellation as its `__cause__`.
