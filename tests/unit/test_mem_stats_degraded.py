"""Degraded/skip-path coverage for mem_stats."""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from memory_mcp import stats
from memory_mcp.identity import AgentContext
from memory_mcp_schemas.stats import EnvStats, MemoriesStats, MemStatsRequest, OutboxStats, SubstrateStats


class FakeSession:
    async def execute(self, *_args, **_kwargs):
        return None

    async def rollback(self):
        return None


@asynccontextmanager
async def fake_session_scope():
    yield FakeSession()


@pytest.mark.asyncio
async def test_empty_scope_skips_database_work(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_scope():  # pragma: no cover - should never be entered
        raise AssertionError("session_scope should not run")

    monkeypatch.setattr(stats, "session_scope", fail_scope)

    out = await stats.compute_mem_stats(MemStatsRequest(), ctx=AgentContext(agent_id=uuid4(), attached_env_ids=[]))

    assert out.memories.total == 0
    assert out.envs.total == 0


@pytest.mark.asyncio
async def test_distribution_timeout_marks_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    env = uuid4()

    async def env_stats(_session, _env_ids):
        return EnvStats(total=1, active=1), {env: "cdp"}

    async def memory_stats(*_args, **_kwargs):
        return MemoriesStats(total=3)

    async def group_counts(*_args, **_kwargs):
        return {}

    async def playbooks(*_args, **_kwargs):
        return {}

    async def decisions(*_args, **_kwargs):
        return {"total": 0, "by_status": {}}

    async def distributions(*_args, **_kwargs):
        raise TimeoutError("cancelled")

    async def projection(*_args, **_kwargs):
        return [], OutboxStats()

    monkeypatch.setattr(stats, "session_scope", fake_session_scope)
    monkeypatch.setattr(stats, "_env_stats", env_stats)
    monkeypatch.setattr(stats, "_memory_stats", memory_stats)
    monkeypatch.setattr(stats, "_group_counts", group_counts)
    monkeypatch.setattr(stats, "_playbooks", playbooks)
    monkeypatch.setattr(stats, "_decisions", decisions)
    monkeypatch.setattr(stats, "_distributions", distributions)
    monkeypatch.setattr(stats, "_projection_and_outbox", projection)

    out = await stats.compute_mem_stats(MemStatsRequest(env_ids=[env]), ctx=AgentContext(agent_id=uuid4()))

    assert "distributions" in out.degraded_sections
    assert out.distributions is None


@pytest.mark.asyncio
async def test_include_flags_skip_work(monkeypatch: pytest.MonkeyPatch) -> None:
    env = uuid4()
    called = {"distributions": False, "substrate": False, "body": None}

    async def env_stats(_session, _env_ids):
        return EnvStats(total=1, active=1), {env: "cdp"}

    async def memory_stats(_session, _env_ids, _names, *, include_body_bytes, tag_top_k):
        called["body"] = include_body_bytes
        assert tag_top_k == 0
        return MemoriesStats(total=1)

    async def group_counts(*_args, **_kwargs):
        return {}

    async def playbooks(*_args, **_kwargs):
        return {}

    async def decisions(*_args, **_kwargs):
        return {"total": 0, "by_status": {}}

    async def distributions(*_args, **_kwargs):
        called["distributions"] = True
        return None

    async def projection(*_args, **_kwargs):
        return [], OutboxStats()

    async def substrate(*_args, **_kwargs):
        called["substrate"] = True
        return SubstrateStats(), ["qdrant"]

    monkeypatch.setattr(stats, "session_scope", fake_session_scope)
    monkeypatch.setattr(stats, "_env_stats", env_stats)
    monkeypatch.setattr(stats, "_memory_stats", memory_stats)
    monkeypatch.setattr(stats, "_group_counts", group_counts)
    monkeypatch.setattr(stats, "_playbooks", playbooks)
    monkeypatch.setattr(stats, "_decisions", decisions)
    monkeypatch.setattr(stats, "_distributions", distributions)
    monkeypatch.setattr(stats, "_projection_and_outbox", projection)
    monkeypatch.setattr(stats, "_substrate_snapshot", substrate)

    out = await stats.compute_mem_stats(
        MemStatsRequest(env_ids=[env], include_body_bytes=False, include_distributions=False, include_substrates=False, tag_top_k=0),
        ctx=AgentContext(agent_id=uuid4()),
    )

    assert called == {"distributions": False, "substrate": False, "body": False}
    assert out.substrate is None


@pytest.mark.asyncio
async def test_substrate_failures_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    env = uuid4()

    async def env_stats(_session, _env_ids):
        return EnvStats(total=1, active=1), {env: "cdp"}

    async def memory_stats(*_args, **_kwargs):
        return MemoriesStats(total=1)

    async def group_counts(*_args, **_kwargs):
        return {}

    async def playbooks(*_args, **_kwargs):
        return {}

    async def decisions(*_args, **_kwargs):
        return {"total": 0, "by_status": {}}

    async def projection(*_args, **_kwargs):
        return [], OutboxStats()

    async def substrate(*_args, **_kwargs):
        return SubstrateStats(qdrant={"error": "down"}), ["qdrant"]

    monkeypatch.setattr(stats, "session_scope", fake_session_scope)
    monkeypatch.setattr(stats, "_env_stats", env_stats)
    monkeypatch.setattr(stats, "_memory_stats", memory_stats)
    monkeypatch.setattr(stats, "_group_counts", group_counts)
    monkeypatch.setattr(stats, "_playbooks", playbooks)
    monkeypatch.setattr(stats, "_decisions", decisions)
    monkeypatch.setattr(stats, "_projection_and_outbox", projection)
    monkeypatch.setattr(stats, "_substrate_snapshot", substrate)

    out = await stats.compute_mem_stats(
        MemStatsRequest(env_ids=[env], include_distributions=False, include_substrates=True),
        ctx=AgentContext(agent_id=uuid4()),
    )

    assert out.degraded_substrates == ["qdrant"]
    assert out.substrate is not None
