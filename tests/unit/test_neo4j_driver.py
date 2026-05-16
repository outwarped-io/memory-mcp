"""Unit tests for the Neo4j driver wrapper + ``probe_neo4j`` helper.

We avoid spinning up a real Neo4j by patching the driver factory. The
contract under test is:

* ``probe_neo4j`` returns ``{"status": "skipped"}`` when
  ``GRAPH_BACKEND != "neo4j"``.
* ``probe_neo4j`` returns ``{"status": "ok"}`` when ``verify_connectivity``
  succeeds.
* ``probe_neo4j`` returns ``{"status": "error"}`` on timeout and on any
  other exception (best-effort — never raises).
* ``Neo4jDriver.close`` is idempotent and tolerates double-close.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_mcp.config import Settings
from memory_mcp.db.graph.neo4j import Neo4jDriver, probe_neo4j


def _settings(graph_backend: str = "neo4j") -> Settings:
    return Settings(graph_backend=graph_backend)  # type: ignore[arg-type]


def _patch_driver(verify_side_effect: Any = None) -> Any:
    """Build a MagicMock that quacks like ``neo4j.AsyncDriver``.

    ``verify_connectivity`` is an AsyncMock so we can attach side effects
    that simulate a slow probe or a failure. ``close`` is also async.
    """
    drv = MagicMock()
    drv.verify_connectivity = AsyncMock(side_effect=verify_side_effect)
    drv.close = AsyncMock()
    drv.session = MagicMock()
    return drv


def test_probe_skipped_when_graph_backend_is_postgres() -> None:
    result = asyncio.run(probe_neo4j(_settings(graph_backend="postgres")))
    assert result["status"] == "skipped"
    assert "graph_backend" in result["reason"]


def test_probe_ok_when_driver_verifies() -> None:
    drv = _patch_driver()
    with patch(
        "memory_mcp.db.graph.neo4j.AsyncGraphDatabase.driver",
        return_value=drv,
    ):
        result = asyncio.run(probe_neo4j(_settings(), timeout=1.0))

    assert result == {"status": "ok"}
    drv.verify_connectivity.assert_awaited_once()
    drv.close.assert_awaited_once()


def test_probe_returns_error_on_timeout() -> None:
    async def _slow() -> None:
        await asyncio.sleep(5.0)

    drv = _patch_driver(verify_side_effect=lambda: _slow())
    # Side effects on AsyncMock can't easily simulate a hang; bypass via
    # asyncio.wait_for tripping on a real awaitable.
    drv.verify_connectivity = AsyncMock(side_effect=asyncio.TimeoutError)

    with patch(
        "memory_mcp.db.graph.neo4j.AsyncGraphDatabase.driver",
        return_value=drv,
    ):
        result = asyncio.run(probe_neo4j(_settings(), timeout=0.5))

    assert result["status"] == "error"
    assert "timed out" in result["error"]
    drv.close.assert_awaited_once()


def test_probe_returns_error_on_unexpected_exception() -> None:
    drv = _patch_driver(
        verify_side_effect=RuntimeError("connection refused"),
    )
    with patch(
        "memory_mcp.db.graph.neo4j.AsyncGraphDatabase.driver",
        return_value=drv,
    ):
        result = asyncio.run(probe_neo4j(_settings()))

    assert result["status"] == "error"
    assert "connection refused" in result["error"]
    drv.close.assert_awaited_once()


def test_driver_lazy_construct_and_close_idempotent() -> None:
    drv_mock = _patch_driver()
    with patch(
        "memory_mcp.db.graph.neo4j.AsyncGraphDatabase.driver",
        return_value=drv_mock,
    ):
        wrapper = Neo4jDriver(_settings())

        assert wrapper._driver is None
        first = wrapper.driver
        assert wrapper._driver is first
        # Subsequent access returns the same instance — no second
        # construction.
        assert wrapper.driver is first

        asyncio.run(wrapper.close())
        # Double-close is a no-op; ``close`` was called exactly once on
        # the underlying driver.
        asyncio.run(wrapper.close())
        assert drv_mock.close.await_count == 1


@pytest.mark.parametrize("hide_password", [True, False])
def test_settings_neo4j_url_is_used(hide_password: bool) -> None:
    """Smoke: construction reads settings — guards against typos."""
    s = Settings(neo4j_url="bolt://example:7687", neo4j_user="x", neo4j_password="y")
    drv = Neo4jDriver(s)
    assert drv._settings.neo4j_url == "bolt://example:7687"
    # ``hide_password`` parameter is unused — exists only to make the
    # test name stable in the report (parametrize hooks future log-line
    # assertions if the wrapper grows).
    _ = hide_password
