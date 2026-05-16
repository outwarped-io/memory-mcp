"""Integration test: end-to-end MCP transport against a live compose stack.

Skipped unless ``MEMORY_MCP_BASE_URL`` is set, which means an operator (or
CI) has brought up ``docker compose up -d server projection-worker``. We
delegate the actual case logic to ``.tmp/mcp_transport_smoke.py`` to keep
one source of truth for the smoke flows; this wrapper just gives pytest a
discoverable assertion.

Run locally:

    docker compose -f repos/memory-mcp/docker-compose.yml up -d
    MEMORY_MCP_BASE_URL=http://localhost:8080 pytest \
        repos/memory-mcp/tests/integration/test_compose_e2e.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = REPO_ROOT / ".tmp" / "mcp_transport_smoke.py"


@pytest.mark.skipif(
    "MEMORY_MCP_BASE_URL" not in os.environ,
    reason="set MEMORY_MCP_BASE_URL to run integration smoke against a live compose stack",
)
@pytest.mark.skipif(
    not SMOKE_PATH.exists(),
    reason="mcp_transport_smoke.py not found",
)
def test_mcp_transport_smoke_against_live_stack() -> None:
    """Run the 14-case MCP transport smoke as a single pytest assertion.

    The smoke script returns 0 on full pass, 1 on any failure, and prints
    per-case status to stdout (captured by pytest -s).
    """
    spec = spec_from_file_location("mcp_transport_smoke", SMOKE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules["mcp_transport_smoke"] = module
    spec.loader.exec_module(module)

    exit_code = asyncio.run(module.main())  # type: ignore[attr-defined]
    if exit_code != 0:
        failures = "\n".join(
            f"  FAIL :: {name} :: {detail}"
            for name, detail in module.FAILED  # type: ignore[attr-defined]
        )
        pytest.fail(
            f"MCP transport smoke reported {len(module.FAILED)} failures:\n{failures}",  # type: ignore[attr-defined]
        )
