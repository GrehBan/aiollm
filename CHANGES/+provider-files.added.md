OpenAI's and Anthropic's Files APIs: `CreateFile`, `ListFiles`, `RetrieveFile`,
`DeleteFile` and `DownloadFile` in each provider's `methods`, with their
generated types (`OpenAIFile`, `FileMetadata`, the list and deletion results)
and request parameters. Upload a file once and refer to it by id in later
requests instead of sending its base64 each time. An upload is never retried,
since `POST /files` is not idempotent; listing returns one page per call;
`DownloadFile` streams the content (`Stream.save(path)`). See the new [Provider
files](https://grehban.github.io/aiollm/guide/files/) guide.
