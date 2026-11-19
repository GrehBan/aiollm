A blocking call that ran on after its caller was cancelled (an upload's
read or open, a synchronous tool callback) and failed after the coroutine
given to `asyncio.run` had returned was dropped silently: `asyncio.run`
cancelled the task that waited for it. The call is no longer a task, so
the loop, which waits for its worker threads before closing, reports the
failure through its exception handler as at any other time.
