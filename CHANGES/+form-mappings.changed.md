**Breaking:** a dict (or a model) in a `FORM` or `MULTIPART` body is now sent
as one field per key with the key in brackets,
`expires_after[anchor]=created_at`, nested dicts likewise, as OpenAI's upload
reads it; it was sent as one JSON-encoded field, which OpenAI refuses with a
400. A list inside a dict now raises `TypeError`, since bracket encodings of it
disagree, and the session's `json_dumps` is no longer used for forms. To keep
sending a JSON-encoded field, declare it as `str` and pass `json.dumps(value)`.
