"""Async-first toolkit for building typed HTTP API clients.

The names most clients need are re-exported here; everything else is
imported from its layer package (``aiollm.http``, ``aiollm.methods``, ...).
Transport code, such as ``aiollm.session.aiohttp``, and provider
packages are imported on their own, so importing ``aiollm`` does not
import a transport.
"""

from __future__ import annotations

import logging

from aiopulse import (
    CallbackCancelledError,
    CallbackExceptionGroup,
    CallbackFailure,
)

from aiollm.__meta__ import (
    __version__,
)
from aiollm.client.base import (
    BaseClient,
    BaseProviderClient,
)
from aiollm.client.binding import (
    bind_resource,
)
from aiollm.enums import (
    BodyFormat,
    CitationKind,
    Effort,
    GenerationEventType,
    IncompleteReason,
    Provider,
    ResponseFormat,
    Role,
    ServerToolKind,
    ServerToolPhase,
    ServerToolStatus,
    ToolChoice,
)
from aiollm.exceptions import (
    AIAPIError,
    RequestValidationError,
    SessionMismatchWarning,
)
from aiollm.http.api import (
    BaseAPI,
)
from aiollm.http.auth import (
    APIKeyAuth,
    BearerAuth,
)
from aiollm.http.exceptions.resolver import (
    BaseErrorResolver,
)
from aiollm.http.exceptions.response import (
    APIResponseError,
)
from aiollm.http.exceptions.status import (
    APIStatusError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    BillingError,
    ConflictError,
    ContentTooLargeError,
    GatewayTimeoutError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    RequestTimeoutError,
    ServiceUnavailableError,
    UnprocessableEntityError,
)
from aiollm.http.exceptions.transport import (
    APIConnectionError,
    APIRedirectError,
    APIRequestError,
    APIStreamInterruptedError,
    APITimeoutError,
    APITLSError,
    APITransportError,
)
from aiollm.http.headers import (
    Headers,
)
from aiollm.methods.base import (
    BaseAPIMethod,
    BaseStreamMethod,
)
from aiollm.resources.protocols import (
    IFilesClient,
    IFilesResource,
    IGenerationAdapter,
    IGenerationClient,
    IGenerationResource,
)
from aiollm.resources.stream import (
    GenerationStream,
)
from aiollm.response.parse.sse import (
    ServerSentEvent,
)
from aiollm.response.stream import (
    Stream,
)
from aiollm.serialization.protocols import (
    IValueFormatter,
)
from aiollm.serialization.values import (
    BaseValueFormatter,
)
from aiollm.session.base import (
    BaseSession,
)
from aiollm.session.retry.policy import (
    FullJitterBackoff,
    RetryPolicy,
)
from aiollm.session.retry.state import (
    RetryState,
)
from aiollm.types.base import (
    BaseAPIType,
    BaseAPITypeExtra,
)
from aiollm.types.files import (
    BytesFile,
    ChunkedFile,
    InputFile,
    IOFile,
    PathFile,
    StreamFile,
)
from aiollm.types.requests.builtin import (
    BuiltinTool,
    CodeExecution,
    ProviderTool,
    UserLocation,
    WebSearch,
)
from aiollm.types.requests.content import (
    ContentPart,
    DocumentPart,
    ImagePart,
    ProviderFile,
    TextPart,
)
from aiollm.types.requests.generation import (
    ForcedTool,
    GenerationRequest,
    InputItem,
    NativeOutput,
    NoProviderOptions,
    Tool,
    ToolCall,
    ToolCallTurn,
    ToolResult,
    Turn,
    tool,
)
from aiollm.types.responses.builtin import (
    Citation,
    ProducedFile,
    ServerToolCall,
)
from aiollm.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    ReasoningDeltaEvent,
    ServerToolCallEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallDoneEvent,
    ToolCallStartEvent,
)
from aiollm.types.responses.files import (
    StoredFile,
)
from aiollm.types.responses.generation import (
    GenerationResponse,
    TokenUsage,
)

# The library configures no logging: its records reach the handlers
# the application sets up, and without them nothing is printed (as
# the logging HOWTO advises libraries).
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AIAPIError",
    "APIConnectionError",
    "APIKeyAuth",
    "APIRedirectError",
    "APIRequestError",
    "APIResponseError",
    "APIStatusError",
    "APIStreamInterruptedError",
    "APITLSError",
    "APITimeoutError",
    "APITransportError",
    "AuthenticationError",
    "BadGatewayError",
    "BadRequestError",
    "BaseAPI",
    "BaseAPIMethod",
    "BaseAPIType",
    "BaseAPITypeExtra",
    "BaseClient",
    "BaseErrorResolver",
    "BaseProviderClient",
    "BaseSession",
    "BaseStreamMethod",
    "BaseValueFormatter",
    "BearerAuth",
    "BillingError",
    "BodyFormat",
    "BuiltinTool",
    "BytesFile",
    "CallbackCancelledError",
    "CallbackExceptionGroup",
    "CallbackFailure",
    "ChunkedFile",
    "Citation",
    "CitationKind",
    "CodeExecution",
    "CompletedEvent",
    "ConflictError",
    "ContentPart",
    "ContentTooLargeError",
    "DocumentPart",
    "Effort",
    "ForcedTool",
    "FullJitterBackoff",
    "GatewayTimeoutError",
    "GenerationEvent",
    "GenerationEventType",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationStream",
    "Headers",
    "IFilesClient",
    "IFilesResource",
    "IGenerationAdapter",
    "IGenerationClient",
    "IGenerationResource",
    "IOFile",
    "IValueFormatter",
    "ImagePart",
    "IncompleteReason",
    "InputFile",
    "InputItem",
    "InternalServerError",
    "NativeOutput",
    "NoProviderOptions",
    "NotFoundError",
    "OtherEvent",
    "PathFile",
    "PermissionDeniedError",
    "ProducedFile",
    "Provider",
    "ProviderFile",
    "ProviderTool",
    "QuotaExceededError",
    "RateLimitError",
    "ReasoningDeltaEvent",
    "RequestTimeoutError",
    "RequestValidationError",
    "ResponseFormat",
    "RetryPolicy",
    "RetryState",
    "Role",
    "ServerSentEvent",
    "ServerToolCall",
    "ServerToolCallEvent",
    "ServerToolKind",
    "ServerToolPhase",
    "ServerToolStatus",
    "ServiceUnavailableError",
    "SessionMismatchWarning",
    "StoredFile",
    "Stream",
    "StreamFile",
    "TextDeltaEvent",
    "TextPart",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolCallDeltaEvent",
    "ToolCallDoneEvent",
    "ToolCallStartEvent",
    "ToolCallTurn",
    "ToolChoice",
    "ToolResult",
    "Turn",
    "UnprocessableEntityError",
    "UserLocation",
    "WebSearch",
    "__version__",
    "bind_resource",
    "tool",
]
