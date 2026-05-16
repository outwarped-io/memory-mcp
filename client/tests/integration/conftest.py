"""Conftest for live integration tests.

Provides:
* A session-scoped ``integration_url`` fixture that auto-skips the
  module if ``MEMORY_MCP_INTEGRATION_URL`` is unset.
* A function-scoped ``live_client`` fixture that opens a fresh
  :class:`MemoryClient` against that URL, with an isolated scratch env
  (auto-created via ``env_create_`` and best-effort cleaned up after).
* Convenience helpers to wait for projection-worker catch-up where
  needed (kept narrow — most tests don't need it).
"""

from __future__ import annotations

import os
import uuid
from typing import AsyncIterator

import pytest

from memory_mcp_client import MemoryClient


pytestmark = pytest.mark.integration


_ENV_VAR = "MEMORY_MCP_INTEGRATION_URL"


@pytest.fixture(scope="session")
def integration_url() -> str:
    url = os.environ.get(_ENV_VAR)
    if not url:
        pytest.skip(f"{_ENV_VAR} not set; skipping live integration tests")
    return url


@pytest.fixture
async def live_client(integration_url: str) -> AsyncIterator[MemoryClient]:
    """Open a MemoryClient against the live server.

    The client is given a generous per-call retry policy so transient
    503s during projection-worker churn don't flake tests.
    """

    async with MemoryClient(integration_url) as client:
        # Cheap probe — if the server isn't ready, fail fast and skip
        # downstream tests with a clean message.
        try:
            await client.ready()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"server at {integration_url} not ready: {exc}")
        yield client


@pytest.fixture
async def scratch_env(live_client: MemoryClient):
    """Create a unique scratch env for this test run; cleanup on teardown.

    Returns the freshly-created :class:`EnvResponse`. The env name is
    namespaced with a UUID4 prefix so concurrent runs don't collide.
    """

    name = f"sdk-it-{uuid.uuid4().hex[:8]}"
    env = await live_client.envs.create(name=name)
    yield env

    # Best-effort cleanup. We intentionally don't fail teardown on
    # error — the server may have already cleaned up, or the test may
    # have hard-deleted the env mid-run.
    try:
        await live_client.envs.delete(env_id=env.id, confirm_destroy=True)
    except Exception:  # noqa: BLE001
        pass
