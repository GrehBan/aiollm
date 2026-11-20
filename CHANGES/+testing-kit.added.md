`aiollm.testing`, the test kit the library's own tests use, for code built
on aiollm, such as a provider of your own: `FakeTransport` replays queued
responses through the real session, `Recorder` (in `aiollm.testing.aiohttp`)
is a local server that answers only what a test scripted, and the pytest
plugin `aiollm.testing.plugin` gives their fixtures and makes retries
instant. Its dependencies come with the new `testing` extra.
