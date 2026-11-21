"""Adapters between shared request and response types and provider methods."""

from __future__ import annotations

from aiollm.adapters.base import (
    BaseAdapter,
    BaseCompositeAdapter,
    BaseConversion,
    BaseFilesAdapter,
    BaseStreamAdapter,
)
from aiollm.adapters.protocols import (
    IAdapter,
    ICompositeAdapter,
    IConversion,
    IStreamAdapter,
)
from aiollm.adapters.stop import Stop
from aiollm.adapters.validation import (
    check_builtin_tools,
    check_conversation,
    check_files,
    checked_options,
    merge_options,
    request_errors,
)

__all__ = [
    "BaseAdapter",
    "BaseCompositeAdapter",
    "BaseConversion",
    "BaseFilesAdapter",
    "BaseStreamAdapter",
    "IAdapter",
    "ICompositeAdapter",
    "IConversion",
    "IStreamAdapter",
    "Stop",
    "check_builtin_tools",
    "check_conversation",
    "check_files",
    "checked_options",
    "merge_options",
    "request_errors",
]
