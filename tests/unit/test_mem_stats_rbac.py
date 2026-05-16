"""RBAC/scope coverage for mem_stats."""

from __future__ import annotations

from uuid import uuid4

import pytest

from memory_mcp import rbac, stats
from memory_mcp.identity import AgentContext
from memory_mcp_schemas.stats import MemStatsRequest


def test_default_scope_uses_attached_envs_and_read_rbac(monkeypatch: pytest.MonkeyPatch) -> None:
    env_a = uuid4()
    env_b = uuid4()
    calls: list[tuple[str, object]] = []

    def fake_require(role, env_id, ctx):
        calls.append((role, env_id))

    monkeypatch.setattr(rbac, "require", fake_require)

    out = stats._scope_env_ids(MemStatsRequest(), AgentContext(agent_id=uuid4(), attached_env_ids=[env_a, env_b]))

    assert out == [env_a, env_b]
    assert calls == [("read", env_a), ("read", env_b)]


def test_explicit_scope_dedupes_and_does_not_leak_other_attached_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    env_a = uuid4()
    env_b = uuid4()
    calls: list[tuple[str, object]] = []

    def fake_require(role, env_id, ctx):
        calls.append((role, env_id))

    monkeypatch.setattr(rbac, "require", fake_require)

    out = stats._scope_env_ids(
        MemStatsRequest(env_ids=[env_a, env_a]),
        AgentContext(agent_id=uuid4(), attached_env_ids=[env_a, env_b]),
    )

    assert out == [env_a]
    assert calls == [("read", env_a)]


def test_global_scope_requires_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    def fake_require(role, env_id, ctx):
        calls.append((role, env_id))

    monkeypatch.setattr(rbac, "require", fake_require)

    out = stats._scope_env_ids(MemStatsRequest(global_=True), AgentContext(agent_id=uuid4(), attached_env_ids=[uuid4()]))

    assert out is None
    assert calls == [("admin", None)]


def test_global_scope_propagates_admin_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    class Denied(Exception):
        pass

    def fake_require(role, env_id, ctx):
        raise Denied("no")

    monkeypatch.setattr(rbac, "require", fake_require)

    with pytest.raises(Denied):
        stats._scope_env_ids(MemStatsRequest(global_=True), AgentContext(agent_id=uuid4()))
