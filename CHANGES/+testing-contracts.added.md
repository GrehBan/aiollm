`aiollm.testing` runs a provider of your own through the scenarios every
library provider passes: subclass `GenerationContract`, `FilesContract` and
`ClientContract` (`aiollm.testing.contract`) with a case of your wire format.
`aiollm.testing.extension.check_extension_api` lists the imports of a
package outside the extension API (now `aiollm/testing/extension_api.json`),
`aiollm.testing.wire` compares generated models with a snapshot of the
upstream schemas, and `aiollm.testing.echo` is a whole provider built from
the extension API alone, for an API that exists only on the local server;
its `EchoClient` types the arguments of its session, as a provider client
does. The `testing` extra now installs `jsonschema` too.
