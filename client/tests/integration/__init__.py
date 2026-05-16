"""Live integration tests for the memory-mcp Python SDK.

These tests speak to a real running memory-mcp server and exercise one
happy-path call per public namespace. They're gated on the
``MEMORY_MCP_INTEGRATION_URL`` environment variable so unit-test runs
stay hermetic.

Run with::

    MEMORY_MCP_INTEGRATION_URL=http://127.0.0.1:8080/mcp \
        pytest client/tests/integration/

or via ``make integration`` from ``client/``.
"""
