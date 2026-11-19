**Breaking:** the tool runner runs a synchronous callback in a worker thread
(`asyncio.to_thread`) instead of in the event loop, so a blocking callback no
longer stops the loop and, with `concurrency` above `1`, synchronous callbacks
run at the same time. A callback that must stay in the loop's thread (one that
uses an `sqlite3` connection or other object bound to its thread, or that is
not thread-safe) should be an `async def`. A cancelled synchronous callback
runs to its end, since a thread cannot be stopped; a failure it then ends with
is reported through the event loop's exception handler.
