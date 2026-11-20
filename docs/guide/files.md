# Provider files

OpenAI and Anthropic keep files you upload and let a request refer to one by its id. A document sent this way is uploaded once instead of being encoded into every request as base64 by the [shared generation](generation.md) parts, which saves both the upload on each turn and the memory a large base64 body takes while it is built.

`client.files` does it the same way for every provider, in shared types; under it, each provider has five methods, `CreateFile`, `ListFiles`, `RetrieveFile`, `DeleteFile` and `DownloadFile`, in its `methods` package, for what the shared layer leaves out. Their fields and result types are generated like the rest of the provider's types (see [Provider types](providers.md#provider-types)).

## On every provider: `client.files`

```python title="shared_files.py"
from datetime import datetime
from pathlib import Path

from aiollm import DocumentPart, IFilesClient, ProviderFile, TextPart, Turn
from aiollm_anthropic import AnthropicClient
from aiollm_openai import OpenAIClient


async def ask_about(
    client: OpenAIClient | AnthropicClient,
    model: str,
    path: Path,
    questions: list[str],
) -> list[str | None]:
    stored = await client.files.upload(path, expires_in_seconds=3600)
    document = DocumentPart(file=stored.ref)
    answers: list[str | None] = []
    try:
        for question in questions:
            response = await client.generation.create(
                model=model,
                input=[Turn.user([document, TextPart(text=question)])],
            )
            answers.append(response.text)
    finally:
        await client.files.delete(stored.ref)
    return answers


async def files_before(
    client: IFilesClient, moment: datetime
) -> list[ProviderFile]:
    return [
        stored.ref
        async for stored in client.files.list()
        if stored.created_at < moment
    ]
```

- **Uploaded once, sent by id.** `upload` takes an `InputFile`, `bytes`, a `pathlib.Path` or a binary stream (see [Uploading files](uploads.md)) and returns a `StoredFile`: `ref`, `filename`, `size`, `created_at` and `expires_at` in UTC, and the provider's own metadata in `raw`. Its `ref`, a `ProviderFile`, goes into `DocumentPart(file=...)` or `ImagePart(file=...)`; each request then carries the id instead of the file in base64.
- **A file belongs to its provider.** A `ProviderFile` names its provider; one of another provider raises `RequestValidationError` before anything is sent, from `client.files` and from `client.generation` alike.
- **Every page, one at a time.** `list()` is an async iterator over all stored files, newest first; the next page is asked for when the last file of a page has been taken.
- **OpenAI's `purpose`.** An upload to OpenAI is a `user_data` file, the purpose of a model's input; `provider_options={"purpose": ...}` sets another. Anthropic's uploads take no options.
- **What is not shared** stays with the provider's methods below: listing by OpenAI's `purpose` or Anthropic's `ids`, and the providers' other fields. [Unification](../unification.md#stored-files) lists what each shared field becomes.

## OpenAI

```python title="openai_files.py"
from pathlib import Path

from aiollm_openai import OpenAIClient
from aiollm_openai.adapters import output_text
from aiollm_openai.methods import (
    CreateFile,
    CreateResponse,
    DeleteFile,
    ListFiles,
)
from aiollm_openai.methods.params import (
    EasyInputMessage,
    FileExpirationAfter,
    InputFileContent,
    InputTextContent,
)


async def summarize(client: OpenAIClient, path: Path) -> str | None:
    file = await client.send_request(
        CreateFile(
            file=path,
            purpose="user_data",
            expires_after=FileExpirationAfter(anchor="created_at", seconds=3600),
        )
    )
    try:
        response = await client.send_request(
            CreateResponse(
                model="gpt-6-astra",
                input=[
                    EasyInputMessage(
                        role="user",
                        content=[
                            InputFileContent(file_id=file.id),
                            InputTextContent(text="Summarize it"),
                        ],
                    )
                ],
            )
        )
    finally:
        await client.send_request(DeleteFile(file_id=file.id))
    return output_text(response)


async def all_file_ids(client: OpenAIClient) -> list[str]:
    page = await client.send_request(ListFiles(limit=100))
    ids = [file.id for file in page.data]
    while page.has_more:
        page = await client.send_request(ListFiles(limit=100, after=page.last_id))
        ids += [file.id for file in page.data]
    return ids
```

- **A file for a model is `user_data`.** `purpose` is required; `user_data` is the purpose for files given to a model as input, and works with the Responses API.
- **`expires_after` is sent key by key.** A multipart form has no nested values, and OpenAI refuses the object as one JSON-encoded field (400); the form spells it out as `expires_after[anchor]` and `expires_after[seconds]` (see [Request body](methods.md#request-body)).
- **One page per call.** Ask for the next page with `after=page.last_id` while `page.has_more` is true.
- **A `user_data` file cannot be downloaded.** `DownloadFile` of one raises `BadRequestError` (400, with no error code; checked live); OpenAI lets only some purposes be downloaded.

## Anthropic

```python title="anthropic_files.py"
from pathlib import Path

from aiollm_anthropic import AnthropicClient
from aiollm_anthropic.adapters import message_text
from aiollm_anthropic.methods import (
    CreateFile,
    CreateMessage,
    DeleteFile,
    DownloadFile,
    ListFiles,
)
from aiollm_anthropic.methods.params import (
    DocumentBlockParam,
    FileDocumentSourceParam,
    TextBlockParam,
    UserMessageParam,
)


async def ask_about(client: AnthropicClient, path: Path) -> str | None:
    file = await client.send_request(CreateFile(file=path, expires_in_seconds=3600))
    try:
        message = await client.send_request(
            CreateMessage(
                model="claude-opus-5-5",
                max_tokens=1024,
                messages=[
                    UserMessageParam(
                        role="user",
                        content=[
                            DocumentBlockParam(
                                source=FileDocumentSourceParam(file_id=file.id)
                            ),
                            TextBlockParam(text="Summarize it"),
                        ],
                    )
                ],
            )
        )
    finally:
        await client.send_request(DeleteFile(file_id=file.id))
    return message_text(message)


async def all_claude_file_ids(client: AnthropicClient) -> list[str]:
    page = await client.send_request(ListFiles(limit=100))
    ids = [file.id for file in page.data]
    while page.next_page is not None:
        page = await client.send_request(ListFiles(limit=100, page=page.next_page))
        ids += [file.id for file in page.data]
    return ids


async def download(client: AnthropicClient, file_id: str, target: Path) -> int:
    async with client.stream(DownloadFile(file_id=file_id)) as content:
        return await content.save(target)
```

- **Pages by cursor.** Pass `page.next_page` as `page` while it is not `None`. `ListFiles(ids=[...])` asks for up to 100 files by id in one page, sent as repeated `ids[]`; it cannot be combined with `page` or `limit`, and the id of a deleted or unknown file is left out, while an id that is not a valid file id fails the whole request with `BadRequestError` (`ids[]: invalid id`, checked live).
- **Only files a tool made can be downloaded.** A file is `downloadable` when code execution or a skill created it; downloading any other raises `BadRequestError` with `code == "file_not_downloadable"`, and an expired file raises `NotFoundError`. `DownloadFile` returns a `Stream[bytes]`: `save(path)` writes it to a file without holding it in memory.
- **The version is part of the method.** Every Files method sends `anthropic-version` (`VERSION_HEADERS`), like `CreateMessage`.
- **Where the schemas come from.** Anthropic's documentation snapshot the library is generated from has no request schemas for Files, so they are a transcription of Anthropic's live API reference, kept with its URLs and the date it was read in `tests/contracts/sources/anthropic-files.openapi.json` of [aiollm_anthropic](https://github.com/aiollm-org/aiollm_anthropic).

## On both providers

- **An upload is never retried.** `CreateFile`, which `client.files.upload` sends, sets `__retryable__ = False`: `POST /files` is not idempotent, and a retry after an answer was lost would store the file twice. A timeout or a broken connection can therefore leave a file stored whose id you never got: give a file an expiry (`expires_after`, `expires_in_seconds`) where that matters. Reading, listing, downloading and deleting are retried like other calls; a deletion retried after its answer was lost raises `NotFoundError`, since the file is already gone.
- **A file id is a server-side reference.** Every key of the OpenAI project or Anthropic workspace can read every file in it by id. Keep the ids your application uploaded and use only those; never take a file id from an end user, who could name somebody else's file.
- **Nothing that names a file is logged.** The transport logs the declared path, `/files/{file_id}`, and never the id or the file name (see [Logging](sessions.md#logging)).
- **Ids are path segments.** A `file_id` is percent-encoded into the path, so an id with a `/` in it cannot reach another endpoint.
