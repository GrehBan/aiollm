"""Snapshots of the providers' wire contracts and the script that
refreshes them.

Each ``<provider>/<endpoint>/`` folder holds the upstream JSON schemas of
one endpoint (request, response, stream events) and the examples that
pass them. ``manifest.json`` records where every schema came from.
``tests/providers/test_wire_contracts.py`` checks the library against
them; ``update.py`` rebuilds them from the providers' official OpenAPI
files and SDK types.
"""
