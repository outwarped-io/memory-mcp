"""Unit tests for ``mode=auto`` search dispatch."""

from __future__ import annotations

import asyncio
import datetime as dt
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from memory_mcp.config import Settings
from memory_mcp.db.models import Memory
from memory_mcp.errors import InvalidInputError
from memory_mcp.identity import AgentContext
from memory_mcp.search import api as search_api
from memory_mcp.search.api import MemorySearchRequest, MemorySearchResponse, _resolve_auto_mode, memory_search


@pytest.mark.parametrize(
    ("query", "expected_mode"),
    [
        ("550e8400-e29b-41d4-a716-446655440000", "id"),
        ("550e8400", "id"),
        ("550e8400-e29b", "id"),
        ("550E8400", "id"),
        ("john smith", "hybrid"),
        ("deadbe", "hybrid"),
        ("deadbeefz", "hybrid"),
    ],
)
def test_resolve_auto_mode(query: str, expected_mode: str) -> None:
    assert _resolve_auto_mode(query) == expected_mode


@pytest.mark.parametrize("query", ["", "   "])
def test_resolve_auto_mode_rejects_empty_query(query: str) -> None:
    with pytest.raises(InvalidInputError, match="mem_search query cannot be empty"):
        _resolve_auto_mode(query)


def test_search_mode_literal_includes_auto_first() -> None:
    assert search_api.SearchMode.__args__[0] == "auto"


def test_memory_search_auto_uuid_prefix_uses_id_mode_without_db() -> None:
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[uuid4()])
    req = MemorySearchRequest(query="550e8400", mode="auto")

    resp = asyncio.run(memory_search(req, ctx=ctx, settings=_settings()))

    assert resp.mode == "auto"
    assert resp.effective_mode == "id"
    assert resp.hits == []


def test_memory_search_auto_existing_uuid_returns_id_hit(monkeypatch) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    now = dt.datetime.now(dt.UTC)
    memory = Memory(
        id=memory_id,
        env_id=env_id,
        kind="fact",
        status="active",
        title="Auto id hit",
        body="existing memory returned by auto id search",
        salience=0.5,
        confidence=0.9,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        metadata_={},
        version=1,
        created_at=now,
        updated_at=now,
    )
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query=str(memory_id), mode="auto")
    hydrated_ids: list[object] = []

    @asynccontextmanager
    async def fake_session_scope():
        yield MagicMock()

    async def fake_hydrate(_session, ids, env_ids):
        hydrated_ids.extend(ids)
        assert env_ids == [env_id]
        return {memory_id: memory}

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(search_api, "_hydrate_memories", fake_hydrate)
    monkeypatch.setattr(search_api, "_bulk_load_tag_names", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_projection_status", AsyncMock(return_value=[]))

    resp = asyncio.run(memory_search(req, ctx=ctx, settings=_settings()))

    assert hydrated_ids == [memory_id]
    assert resp.mode == "auto"
    assert resp.effective_mode == "id"
    assert len(resp.hits) == 1
    assert resp.hits[0].memory.id == memory_id
    assert resp.hits[0].sources == ["id"]


def test_memory_search_auto_plain_query_uses_hybrid_mode(monkeypatch) -> None:
    env_id = uuid4()
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query="john smith", mode="auto")

    calls: list[str] = []

    async def fake_do_lex(*_args, **_kwargs):
        calls.append("lex")
        return []

    async def fake_do_sem(*_args, **_kwargs):
        calls.append("sem")
        return []

    async def fake_do_graph(*_args, **_kwargs):
        calls.append("graph")
        return []

    @asynccontextmanager
    async def fake_session_scope():
        yield MagicMock()

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(search_api, "_do_lex", fake_do_lex)
    monkeypatch.setattr(search_api, "_do_sem", fake_do_sem)
    monkeypatch.setattr(search_api, "_do_graph", fake_do_graph)
    monkeypatch.setattr(search_api, "_hydrate_memories", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_projection_status", AsyncMock(return_value=[]))

    resp = asyncio.run(memory_search(
        req,
        ctx=ctx,
        settings=_settings(),
        vector_store=MagicMock(),
        embedder=MagicMock(),
        graph_store=MagicMock(),
    ))

    assert resp.mode == "auto"
    assert resp.effective_mode == "hybrid"
    assert sorted(calls) == ["graph", "lex", "sem"]


def test_memory_search_auto_hybrid_can_still_downgrade_for_canonical(monkeypatch) -> None:
    env_id = uuid4()
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query="john smith", mode="auto", consistency="canonical")

    async def fake_do_lex(*_args, **_kwargs):
        return []

    @asynccontextmanager
    async def fake_session_scope():
        yield MagicMock()

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(search_api, "_do_lex", fake_do_lex)
    monkeypatch.setattr(search_api, "_hydrate_memories", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_projection_status", AsyncMock(return_value=[]))

    resp = asyncio.run(memory_search(req, ctx=ctx, settings=_settings()))

    assert resp.mode == "auto"
    assert resp.effective_mode == "lex"
    assert resp.consistency_used == "canonical"


def _settings(**overrides) -> Settings:
    s = Settings()
    object.__setattr__(s, "search_min_per_leg", 1)
    object.__setattr__(s, "search_fresh_max_wait_seconds", 0.01)
    for key, value in overrides.items():
        object.__setattr__(s, key, value)
    return s
