"""Shared scenarios that every provider package runs on its client.

A provider's tests subclass ``GenerationContract``, ``FilesContract``
and ``ClientContract`` with a case of its own wire format; the
scenarios run on the local server of ``aiollm.testing.plugin``, so the
plugin must be loaded. The modules import pytest: import them from tests
only.

Example:
    >>> from aiollm.testing.contract import (
    ...     GenerationCase,
    ...     GenerationContract,
    ... )
    >>> class TestExampleGeneration(GenerationContract):
    ...     case = GenerationCase(provider=PROVIDER, ...)
"""

from __future__ import annotations

from aiollm.testing.contract.clients import (
    AnotherErrorResolver,
    ClientCase,
    ClientContract,
)
from aiollm.testing.contract.files import FilesCase, FilesContract
from aiollm.testing.contract.generation import (
    BuiltinCase,
    GenerationCase,
    GenerationContract,
)

__all__ = [
    "AnotherErrorResolver",
    "BuiltinCase",
    "ClientCase",
    "ClientContract",
    "FilesCase",
    "FilesContract",
    "GenerationCase",
    "GenerationContract",
]
