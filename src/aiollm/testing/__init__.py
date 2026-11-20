"""A test kit for code built on aiollm: doubles, fixtures, a local server.

The doubles need no server: ``FakeTransport`` replays queued responses
through the real ``BaseSession``, ``LocalAPI`` builds URLs and
``make_method`` declares a method class on the fly. The local server
(``Recorder`` in ``aiollm.testing.aiohttp``) answers only what a test
scripted with ``Reply``; the fixtures that run it are in the pytest
plugin ``aiollm.testing.plugin``. Install the ``testing`` extra for
them; importing ``aiollm`` loads none of this.
"""

from __future__ import annotations

from aiollm.testing.doubles import (
    NOW,
    TEST_TOKEN,
    FakeStreamResponse,
    FakeTransport,
    LocalAPI,
    Outcome,
    make_method,
)
from aiollm.testing.recording import (
    Chunk,
    RecordedPart,
    RecordedRequest,
    Reply,
)

__all__ = [
    "Chunk",
    "FakeStreamResponse",
    "FakeTransport",
    "LocalAPI",
    "NOW",
    "Outcome",
    "RecordedPart",
    "RecordedRequest",
    "Reply",
    "TEST_TOKEN",
    "make_method",
]
