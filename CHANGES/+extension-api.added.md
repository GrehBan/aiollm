The checks an adapter runs on a shared request, `checked_options`,
`merge_options`, `request_errors`, `check_conversation` and `check_files`,
are exported from `aiollm.adapters`, so a provider written outside the
library takes them from a public package. The
[Extending](https://aiollm-org.github.io/aiollm/guide/extending/) guide
lists the names of the core a provider package may build on.
