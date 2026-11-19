`client.files` on `OpenAIClient` and `AnthropicClient`: the same `upload`,
`retrieve`, `list`, `delete` and `download` for every provider, in shared
types. An upload returns a `StoredFile` (`ref`, `filename`, `size`,
`created_at`, `expires_at` in UTC, the provider's metadata in `raw`), and its
`ref`, a `ProviderFile`, goes into the new third source of `ImagePart` and
`DocumentPart`, `file=`, so a request carries the file's id instead of its
base64. A `ProviderFile` of another provider raises `RequestValidationError`
before anything is sent. `list()` goes through every page; OpenAI's uploads are
`user_data` files unless `provider_options={"purpose": ...}` (the new
`OpenAIFileOptions`) says otherwise. New public names: `FilesResource`,
`IFilesResource`, `IFilesClient`, `IFilesAdapter`, `BaseFilesAdapter`,
`OpenAIFilesAdapter`, `AnthropicFilesAdapter`, `FileUpload`, `ProviderFile`,
`StoredFile`.
