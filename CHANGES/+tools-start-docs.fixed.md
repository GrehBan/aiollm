The tools guide said that the runner starts callbacks only while the
consumer waits for a result; entering `async with calls:` (or the first
read) already starts the first ones, up to `concurrency`.
