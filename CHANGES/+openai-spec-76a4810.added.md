OpenAI's Responses API follows `openapi.yaml` at commit 76a4810: the request
takes `background_options` (`BackgroundOptionsParam`: `completion_window` and
`continuation_id`) for a `background: true` call, and a response has
`input_transformations` (`ReasoningInputTransformation`), the reasoning
inputs withheld from the model.
