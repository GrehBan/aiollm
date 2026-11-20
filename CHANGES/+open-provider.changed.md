**Breaking:** `Provider` is an open value instead of a closed enum, so a
provider written outside the library names itself and its files and errors
stay apart from another one's. It is a `str` made from the provider's name;
the library declares only `Provider.GENERIC`, and each provider package its
own `PROVIDER`. To migrate, write `aiollm.providers.openai.api.PROVIDER` (or
`Provider("openai")`) for `Provider.OPENAI`,
`aiollm.providers.anthropic.api.PROVIDER` (or `Provider("anthropic")`) for
`Provider.ANTHROPIC`, and `Provider("<name>")` for the `Provider.GENERIC` an
adapter or resolver of your own set; compare providers with `==`, not `is`.
`ANTHROPIC_VERSION`, the header name, moved from `aiollm.http.headers` to
`aiollm.providers.anthropic.api`.
