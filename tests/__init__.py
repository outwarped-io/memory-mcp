"""memory-mcp test suite.

Layout:

* ``tests/unit`` — pure-Python tests, no external services.
* ``tests/integration`` — testcontainers-driven, spins up Postgres / Qdrant / Neo4j.
* ``tests/invariants`` — outbox / projection / RBAC / lifecycle invariants.
"""
