"""v0.10 Prometheus metric coverage."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from prometheus_client import generate_latest

from memory_mcp import observability


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeSession:
    async def execute(self, stmt):
        sql = str(stmt)
        if "COALESCE(e.name" in sql and "m.kind" in sql:
            return FakeResult([("cdp", "fact", "active", 2), ("cdp", "decision", "superseded", 1)])
        if "COUNT(*) FILTER" in sql:
            return FakeResult([("cdp", 1, 1234)])
        if "FROM tasks" in sql:
            return FakeResult([("pending", 2), ("done", 3)])
        if "kind = 'playbook'" in sql:
            return FakeResult([("active", 4)])
        if "kind = 'decision'" in sql:
            return FakeResult([("active", 5), ("superseded", 1)])
        if "WITH RECURSIVE chain" in sql:
            return FakeResult([(1,), (2,), (4,)])
        if "octet_length(body) AS body_length" in sql:
            return FakeResult([(100, 60, 0.5, 0), (200, None, 0.9, 10)])
        return FakeResult([])


@asynccontextmanager
async def fake_session_scope():
    yield FakeSession()


def _body() -> str:
    return generate_latest(observability.metrics_registry()).decode()


@pytest.mark.asyncio
async def test_v10_metrics_are_registered_and_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    import memory_mcp.db.postgres as postgres

    monkeypatch.setattr(postgres, "session_scope", fake_session_scope)
    monkeypatch.setattr(observability, "_STATS_LAST_REFRESH", 0.0)

    await observability.refresh_metrics_on_scrape_v10(force=True)

    body = _body()
    for name in (
        "mcp_memories_total",
        "mcp_memories_pinned_total",
        "mcp_memories_body_bytes_total",
        "mcp_memory_chain_depth",
        "mcp_memory_age_seconds",
        "mcp_memory_body_length_bytes",
        "mcp_memory_salience",
        "mcp_memory_access_count",
        "mcp_tasks_total",
        "mcp_playbooks_total",
        "mcp_decisions_total",
        "process_resident_memory_bytes",
    ):
        assert name in body
    assert 'env="cdp",kind="fact",status="active"' in body
    assert 'status="pending"' in body


@pytest.mark.asyncio
async def test_v10_refresh_ttl_skips_database(monkeypatch: pytest.MonkeyPatch) -> None:
    import memory_mcp.db.postgres as postgres

    called = False

    @asynccontextmanager
    async def failing_scope():
        nonlocal called
        called = True
        raise AssertionError("should be TTL skipped")
        yield  # pragma: no cover

    monkeypatch.setattr(postgres, "session_scope", failing_scope)
    monkeypatch.setattr(observability, "_STATS_LAST_REFRESH", observability.time.monotonic())
    monkeypatch.setenv("MCP_METRICS_REFRESH_INTERVAL_SECONDS", "60")

    await observability.refresh_metrics_on_scrape_v10(force=False)

    assert called is False
