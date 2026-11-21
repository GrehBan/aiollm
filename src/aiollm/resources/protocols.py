"""Interfaces of resources and use cases."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Protocol, Self, Unpack, overload

from pydantic import MISSING

from aiollm.adapters.protocols import ICompositeAdapter
from aiollm.client.protocols import IClient
from aiollm.methods.protocols import IAPIMethod, IStreamMethod
from aiollm.response.stream import Stream
from aiollm.session.protocols import CallOptions, ISession, Timeout
from aiollm.types.requests.base import BaseRequest
from aiollm.types.requests.generation import (
    GenerationRequest,
    NoProviderOptions,
)
from aiollm.types.responses.base import BaseResponse
from aiollm.types.responses.events import BaseResponseEvent, GenerationEvent

if TYPE_CHECKING:
    from os import PathLike

    from aiollm.enums import Effort, Provider, ToolChoice
    from aiollm.http.protocols import HeadersType
    from aiollm.resources.stream import GenerationStream
    from aiollm.session.retry.policy import RetryPolicy
    from aiollm.types.files import BinaryStream, InputFile
    from aiollm.types.requests.content import ProviderFile
    from aiollm.types.requests.generation import (
        ForcedTool,
        InputItem,
        JSONSchema,
        ToolType,
    )
    from aiollm.types.requests.upload import FileUpload
    from aiollm.types.responses.files import StoredFile
    from aiollm.types.responses.generation import GenerationResponse


class ISessionAccess(Protocol):
    """Gives access to the session that sends the requests."""

    @property
    def session(self) -> ISession:
        """The session that sends the requests."""
        ...


class IResource(ISessionAccess, Protocol):
    """A group of use cases of one API area, such as responses."""


class IUseCase[
    ReqT: BaseRequest,
    RespT: BaseResponse,
](ISessionAccess, Protocol):
    """Turns a shared request into a shared response."""

    async def __call__(
        self,
        request: ReqT,
        **options: Unpack[CallOptions],
    ) -> RespT:
        """Send the request and convert the provider's response.

        Args:
            request: The request in shared types.
            **options: Options of the call (``headers``, ``timeout``,
                ``retry_policy``), passed on to the session.

        Returns:
            The response in shared types.
        """
        ...


class IStreamUseCase[
    ReqT: BaseRequest,
    EventT: BaseResponseEvent,
](ISessionAccess, Protocol):
    """Turns a shared request into a stream of shared events."""

    def stream(
        self,
        request: ReqT,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[Stream[EventT]]:
        """Open the stream; use the result with ``async with``.

        Args:
            request: The request in shared types.
            **options: Options of the call (``headers``, ``timeout``,
                ``retry_policy``), passed on to the session.

        Returns:
            A context manager that yields the events and closes the
            stream on exit.
        """
        ...


class ICompositeUseCase[
    ReqT: BaseRequest,
    RespT: BaseResponse,
    EventT: BaseResponseEvent,
](IUseCase[ReqT, RespT], IStreamUseCase[ReqT, EventT], Protocol):
    """A use case that can both respond at once and stream."""


class IGenerationAdapter[
    RawT,
    RawItemT,
    OptionsT: Mapping[str, object] = NoProviderOptions,
](
    ICompositeAdapter[
        GenerationRequest,
        RawT,
        "GenerationResponse[RawT]",
        RawItemT,
        GenerationEvent[RawItemT],
    ],
    Protocol,
):
    """The adapter a ``GenerationResource`` runs on.

    Besides converting, it takes the provider's own options of one call
    (``provider_options``): fields that only its provider has, such as
    OpenAI's ``store``. ``OptionsT`` is a ``TypedDict`` of them.

    Concurrent calls share the adapter and its shallow copies, so an
    adapter, its conversions included, holds no mutable state that
    outlives one method call; what belongs to one call (the options)
    lives only on the copy ``with_options`` makes.
    """

    def with_options(self, options: OptionsT, /) -> Self:
        """Return a copy that adds the options to every method it builds.

        The adapter itself is left as it is, so concurrent calls with
        other options do not meet. The copy may be shallow: it shares
        the conversions, which hold no mutable state. Every key is
        checked, also by an adapter that takes no options
        (``checked_options(options, NoProviderOptions, provider)``), so
        a key is never dropped silently.

        Args:
            options: The provider's options of one call.

        Returns:
            A copy of the adapter with the options.

        Raises:
            RequestValidationError: A key is not one of the provider's
                options; the shared fields are set in the request.
        """
        ...


class IGenerationResource[RawT, RawItemT](IResource, Protocol):
    """Generates text; the same interface for every provider.

    Both methods take the request's fields as keyword arguments, as the
    providers' SDKs do, or a ready ``GenerationRequest``, and the options
    of the call (``headers``, ``timeout``, ``retry_policy``) as
    ``send_request`` does; no request field is named like one of them.
    """

    @overload
    async def create(
        self,
        /,
        *,
        model: str,
        input: str | Sequence[InputItem],
        instructions: str | MISSING = MISSING,
        max_output_tokens: int | MISSING = MISSING,
        end_user_id: str | MISSING = MISSING,
        effort: Effort | MISSING = MISSING,
        output_schema: JSONSchema | MISSING = MISSING,
        tools: ToolType | Sequence[ToolType] | MISSING = MISSING,
        tool_choice: ToolChoice | ForcedTool | MISSING = MISSING,
        parallel_tool_calls: bool | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> GenerationResponse[RawT]: ...

    @overload
    async def create(
        self,
        request: GenerationRequest,
        /,
        **options: Unpack[CallOptions],
    ) -> GenerationResponse[RawT]: ...

    async def create(
        self,
        request: GenerationRequest | None = None,
        /,
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
        **fields: Any,
    ) -> GenerationResponse[RawT]:
        """Generate the whole response at once.

        Args:
            request: What to generate, as a ready request.
            headers: Headers for this request only.
            timeout: Seconds for this request instead of the session's
                limit; ``None`` to keep it.
            retry_policy: Retry policy of this request; ``None`` for the
                session's.
            **fields: What to generate, as the fields of a
                ``GenerationRequest``.

        Returns:
            The generated text with the provider's own response.
        """
        ...

    @overload
    def stream(
        self,
        /,
        *,
        model: str,
        input: str | Sequence[InputItem],
        instructions: str | MISSING = MISSING,
        max_output_tokens: int | MISSING = MISSING,
        end_user_id: str | MISSING = MISSING,
        effort: Effort | MISSING = MISSING,
        output_schema: JSONSchema | MISSING = MISSING,
        tools: ToolType | Sequence[ToolType] | MISSING = MISSING,
        tool_choice: ToolChoice | ForcedTool | MISSING = MISSING,
        parallel_tool_calls: bool | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[GenerationStream[RawItemT]]: ...

    @overload
    def stream(
        self,
        request: GenerationRequest,
        /,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[GenerationStream[RawItemT]]: ...

    def stream(
        self,
        request: GenerationRequest | None = None,
        /,
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
        **fields: Any,
    ) -> AbstractAsyncContextManager[GenerationStream[RawItemT]]:
        """Open a stream of generation events; use with ``async with``.

        Args:
            request: What to generate, as a ready request.
            headers: Headers for this request only.
            timeout: Seconds for this request instead of the session's
                limit; ``None`` to keep it.
            retry_policy: Retry policy of this request; ``None`` for the
                session's.
            **fields: What to generate, as the fields of a
                ``GenerationRequest``.

        Returns:
            A context manager that yields the events, with the
            ``text()`` and ``final()`` helpers, and closes the stream on
            exit.
        """
        ...


class IGenerationClient[RawT = Any, RawItemT = Any](IClient, Protocol):
    """A client that generates text; every provider's client is one.

    ``RawT`` is the provider's response and ``RawItemT`` one item of its
    stream; both default to ``Any`` for code that works with any
    provider. It lives with the resources, not with the clients, since
    it names a resource protocol and the client layer is below this one.

    Example:
        >>> async def ask(client: IGenerationClient, question: str) -> str:
        ...     response = await client.generation.create(
        ...         model=MODEL, input=question
        ...     )
        ...     return response.text or ""
    """

    @property
    def generation(self) -> IGenerationResource[RawT, RawItemT]:
        """Shared text generation over the provider's API."""
        ...


class IFilesAdapter[
    RawFileT,
    RawPageT,
    OptionsT: Mapping[str, object] = NoProviderOptions,
](Protocol):
    """The adapter a ``FilesResource`` runs on.

    It builds the provider's Files methods and converts what they
    return; like every adapter, it sends nothing, holds no session and
    no mutable state that outlives one method call. ``RawFileT`` is the
    provider's file type, ``RawPageT`` one page of its list, and
    ``OptionsT`` the ``TypedDict`` of its own upload options.
    """

    @property
    def provider(self) -> Provider:
        """The provider whose methods the adapter builds."""
        ...

    def with_options(self, options: OptionsT, /) -> Self:
        """Return a copy that adds the options to every upload it builds.

        Args:
            options: The provider's own upload options of one call.

        Returns:
            A copy of the adapter; this one is left as it is.

        Raises:
            RequestValidationError: A key is not one of the provider's
                options.
        """
        ...

    def to_upload_method(self, request: FileUpload) -> IAPIMethod[RawFileT]:
        """Build the method that stores a file.

        Args:
            request: The shared upload.

        Returns:
            The provider's upload method, ready to send.

        Raises:
            RequestValidationError: The provider cannot take the upload,
                such as an expiry outside its range.
        """
        ...

    def to_retrieve_method(self, file_id: str) -> IAPIMethod[RawFileT]:
        """Build the method that reads a file's metadata.

        Args:
            file_id: The provider's id of the file.

        Returns:
            The provider's method, ready to send.
        """
        ...

    def to_list_method(self, cursor: str | None) -> IAPIMethod[RawPageT]:
        """Build the method that reads one page of the stored files.

        Args:
            cursor: Where the page starts, as ``to_page`` gave it;
                ``None`` for the first page.

        Returns:
            The provider's method, ready to send.
        """
        ...

    def to_page(self, raw: RawPageT) -> tuple[Sequence[RawFileT], str | None]:
        """Take the files of a page and where the next one starts.

        Args:
            raw: The page the list method returned.

        Returns:
            The page's files and the cursor of the next page, ``None``
            after the last one.
        """
        ...

    def to_delete_method(self, file_id: str) -> IAPIMethod[Any]:
        """Build the method that deletes a file.

        Args:
            file_id: The provider's id of the file.

        Returns:
            The provider's method, ready to send.
        """
        ...

    def to_download_method(self, file: ProviderFile) -> IStreamMethod[bytes]:
        """Build the method that streams a file's content.

        The whole file, not its id: a ``ProducedFile`` of a container
        (``container_id``) is downloaded from that container where the
        provider has one, and refused where it has none.

        Args:
            file: The file, already known to be the provider's.

        Returns:
            The provider's stream method, ready to open.

        Raises:
            RequestValidationError: The provider cannot download the
                file, such as a file of a container where it has none.
        """
        ...

    def to_stored_file(self, raw: RawFileT) -> StoredFile[RawFileT]:
        """Convert the provider's file into the shared metadata.

        Args:
            raw: The provider's file.

        Returns:
            The shared metadata, ``raw`` included.
        """
        ...


class IFilesResource[RawT](IResource, Protocol):
    """Stores files with a provider; the same interface for every one.

    A stored file is named by its ``ProviderFile`` (``StoredFile.ref``),
    which a part takes as ``file=``; a ``ProviderFile`` of another
    provider raises ``RequestValidationError`` before anything is sent.
    Every method takes the options of the call (``headers``, ``timeout``,
    ``retry_policy``) as ``send_request`` does.
    """

    async def upload(
        self,
        file: InputFile | bytes | PathLike[str] | BinaryStream,
        /,
        *,
        expires_in_seconds: int | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> StoredFile[RawT]:
        """Store a file with the provider.

        Never retried, since a retry after a lost answer would store the
        file twice.

        Args:
            file: The content, with the name the provider keeps.
            expires_in_seconds: Seconds after which the provider deletes
                the file; left out, it is kept until it is deleted.
            **options: Options of the call, passed on to the session.

        Returns:
            The stored file.
        """
        ...

    async def retrieve(
        self,
        file: ProviderFile,
        /,
        **options: Unpack[CallOptions],
    ) -> StoredFile[RawT]:
        """Read a stored file's metadata.

        Args:
            file: The file.
            **options: Options of the call, passed on to the session.

        Returns:
            The stored file.
        """
        ...

    def list(
        self,
        **options: Unpack[CallOptions],
    ) -> AsyncIterator[StoredFile[RawT]]:
        """Go through every stored file, a page at a time.

        Args:
            **options: Options of each page's call, passed on to the
                session.

        Returns:
            The files, in the provider's order; the next page is asked
            for when the last file of a page has been taken.
        """
        ...

    async def delete(
        self,
        file: ProviderFile,
        /,
        **options: Unpack[CallOptions],
    ) -> None:
        """Delete a stored file.

        Args:
            file: The file.
            **options: Options of the call, passed on to the session.
        """
        ...

    def download(
        self,
        file: ProviderFile,
        /,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[Stream[bytes]]:
        """Stream a stored file's content; use with ``async with``.

        Args:
            file: The file.
            **options: Options of opening the stream, passed on to the
                session.

        Returns:
            A context manager that yields the content in chunks
            (``save(path)`` writes it to a file) and closes it on exit.
        """
        ...


class IFilesClient[RawT = Any](IClient, Protocol):
    """A client that stores files; every provider's client is one.

    ``RawT`` is the provider's file type, ``Any`` by default for code
    that works with any provider.

    Example:
        >>> async def store(client: IFilesClient, path: Path) -> ProviderFile:
        ...     stored = await client.files.upload(path)
        ...     return stored.ref
    """

    @property
    def files(self) -> IFilesResource[RawT]:
        """Files stored with the provider."""
        ...


def make_request(
    request: GenerationRequest | None,
    fields: dict[str, Any],
) -> GenerationRequest:
    """Take the request of a call, given ready or as fields.

    Args:
        request: The ready request, if one was given.
        fields: The keyword arguments of the call.

    Returns:
        The request to send.

    Raises:
        TypeError: Both a request and fields are given.
        ValidationError: The fields do not make a valid request, e.g. a
            field name is misspelled.
    """
    if request is None:
        return GenerationRequest(**fields)
    if fields:
        names = ", ".join(sorted(fields))
        message = (
            "Pass either a GenerationRequest or its fields as keyword "
            f"arguments, not both (got the request and {names})"
        )
        raise TypeError(message)
    return request
