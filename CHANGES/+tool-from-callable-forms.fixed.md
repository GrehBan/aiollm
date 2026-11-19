`Tool.from_callable()` and `@tool` take a `functools.partial` (named after
the function it wraps, its bound arguments left out of the schema) and an
object with a `__call__` method (named after its class, read without
`self`); both raised `TypeError` before. The generated schema now says
`additionalProperties: false`, as the arguments were already validated
that way, and a strict tool whose callable takes `**kwargs`, which no
strict schema can describe, raises `TypeError` instead of getting a 400
from the provider.
