**Breaking:** the providers left the core for packages of their own, in
repositories of their own, and the core's distribution is now named
`aiollm-core` (was `aiollm`; `import aiollm` is unchanged). To migrate,
install the provider you use and change its imports:
`aiollm.providers.openai` → `aiollm_openai` (distribution `aiollm-openai`),
`aiollm.providers.anthropic` → `aiollm_anthropic` (distribution
`aiollm-anthropic`). The modules under them keep their names
(`aiollm.providers.openai.methods` → `aiollm_openai.methods`), and a
provider package installs the core it needs. `ToolCall` left the extension
API as a name of `aiollm.types.responses.generation`: a provider takes it
from `aiollm.types.requests.generation`, where it is defined.
