"""Unit tests for v0.13 search expansion presets."""

from __future__ import annotations

import asyncio
import datetime as dt
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from memory_mcp.config import Settings
from memory_mcp.db.models import Memory
from memory_mcp.identity import AgentContext
from memory_mcp.search import ExpansionPreset, MemorySearchRequest, memory_search
from memory_mcp.search import api as search_api
from memory_mcp.search.ranking import RankedHit

NARROW_RESOLVED = {
    "min_score": 0.035,
    "fallback": False,
    "follow_superseded": False,
}
DEFAULT_RESOLVED: dict[str, object] = {}
BROAD_RESOLVED = {
    "fallback": True,
    "follow_superseded": True,
    "include_stale": True,
    "include_archived": True,
}


def _settings(**overrides) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "search_min_per_leg", 1)
    object.__setattr__(settings, "search_fresh_max_wait_seconds", 0.01)
    for key, value in overrides.items():
        object.__setattr__(settings, key, value)
    return settings


def _memory(
    memory_id: UUID,
    env_id: UUID,
    *,
    status: str = "active",
    salience: float = 0.5,
) -> Memory:
    now = dt.datetime.now(dt.UTC)
    return Memory(
        id=memory_id,
        env_id=env_id,
        kind="fact",
        status=status,
        title="t",
        body="b",
        salience=salience,
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


def _patch_search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lex_lists: list[list[RankedHit]],
    memories: dict[UUID, Memory],
    sem_lists: list[list[RankedHit]] | None = None,
    graph_lists: list[list[RankedHit]] | None = None,
) -> None:
    call_idx = {"lex": 0, "sem": 0, "graph": 0}
    sem_lists = sem_lists or []
    graph_lists = graph_lists or []

    async def fake_do_lex(*_args, **_kwargs):
        idx = call_idx["lex"]
        call_idx["lex"] += 1
        return lex_lists[idx] if idx < len(lex_lists) else []

    async def fake_do_sem(*_args, **_kwargs):
        idx = call_idx["sem"]
        call_idx["sem"] += 1
        return sem_lists[idx] if idx < len(sem_lists) else []

    async def fake_do_graph(*_args, **_kwargs):
        idx = call_idx["graph"]
        call_idx["graph"] += 1
        return graph_lists[idx] if idx < len(graph_lists) else []

    @asynccontextmanager
    async def fake_session_scope():
        yield MagicMock()

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(search_api, "_do_lex", fake_do_lex)
    monkeypatch.setattr(search_api, "_do_sem", fake_do_sem)
    monkeypatch.setattr(search_api, "_do_graph", fake_do_graph)
    monkeypatch.setattr(search_api, "_hydrate_memories", AsyncMock(return_value=memories))
    monkeypatch.setattr(search_api, "_bulk_load_tag_names", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_projection_status", AsyncMock(return_value=[]))


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (ExpansionPreset.narrow, NARROW_RESOLVED),
        (ExpansionPreset.default, DEFAULT_RESOLVED),
        (ExpansionPreset.broad, BROAD_RESOLVED),
    ],
)
def test_expansion_resolved_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    preset: ExpansionPreset,
    expected: dict[str, object],
) -> None:
    env_id = uuid4()
    mid = uuid4()
    memories = {mid: _memory(mid, env_id, salience=1.0)}
    ranked = [[RankedHit(memory_id=mid, rank=1, raw_score=1.0, source="lex")]]
    _patch_search(monkeypatch, lex_lists=ranked, sem_lists=ranked, memories=memories)

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query="q", mode="hybrid", expansion=preset)
    resp = asyncio.run(
        memory_search(
            req,
            ctx=ctx,
            settings=_settings(),
            vector_store=MagicMock(),
            embedder=MagicMock(),
            graph_store=MagicMock(),
        )
    )

    assert resp.expansion_resolved == expected


def test_expansion_none_leaves_expansion_resolved_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    mid = uuid4()
    memories = {mid: _memory(mid, env_id, salience=1.0)}
    ranked = [[RankedHit(memory_id=mid, rank=1, raw_score=1.0, source="lex")]]
    _patch_search(monkeypatch, lex_lists=ranked, sem_lists=ranked, memories=memories)

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    resp = asyncio.run(
        memory_search(
            MemorySearchRequest(query="q", mode="hybrid"),
            ctx=ctx,
            settings=_settings(),
            vector_store=MagicMock(),
            embedder=MagicMock(),
            graph_store=MagicMock(),
        )
    )

    assert resp.expansion_resolved is None


def test_expansion_default_matches_omitted_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    mid = uuid4()
    memories = {mid: _memory(mid, env_id, salience=1.0)}
    ranked = [[RankedHit(memory_id=mid, rank=1, raw_score=1.0, source="lex")]]
    _patch_search(monkeypatch, lex_lists=ranked * 2, sem_lists=ranked * 2, memories=memories)

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    default_resp = asyncio.run(
        memory_search(
            MemorySearchRequest(query="q", mode="hybrid", expansion=ExpansionPreset.default),
            ctx=ctx,
            settings=_settings(),
            vector_store=MagicMock(),
            embedder=MagicMock(),
            graph_store=MagicMock(),
        )
    )
    omitted_resp = asyncio.run(
        memory_search(
            MemorySearchRequest(query="q", mode="hybrid"),
            ctx=ctx,
            settings=_settings(),
            vector_store=MagicMock(),
            embedder=MagicMock(),
            graph_store=MagicMock(),
        )
    )

    assert default_resp.model_dump(exclude={"expansion_resolved"}) == omitted_resp.model_dump(
        exclude={"expansion_resolved"},
    )
    assert default_resp.expansion_resolved == {}
    assert omitted_resp.expansion_resolved is None


def test_narrow_rejects_results_below_p90_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    mid = uuid4()
    memories = {mid: _memory(mid, env_id, salience=1.0)}
    ranked = [[RankedHit(memory_id=mid, rank=1, raw_score=1.0, source="lex")]]
    _patch_search(monkeypatch, lex_lists=ranked, memories=memories)

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    resp = asyncio.run(
        memory_search(
            MemorySearchRequest(query="q", mode="lex", expansion=ExpansionPreset.narrow),
            ctx=ctx,
            settings=_settings(),
        )
    )

    assert resp.hits == []
    assert resp.expansion_resolved == NARROW_RESOLVED


def test_broad_does_not_include_retired_rows_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    retired_id = uuid4()
    retired_memory = _memory(retired_id, env_id, status="retired", salience=1.0)
    retired_hit = [RankedHit(memory_id=retired_id, rank=1, raw_score=1.0, source="lex")]
    _patch_search(
        monkeypatch,
        lex_lists=[retired_hit, retired_hit, retired_hit, retired_hit],
        memories={retired_id: retired_memory},
    )

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    resp = asyncio.run(
        memory_search(
            MemorySearchRequest(query="gibberish-zqxjkv", mode="lex", expansion=ExpansionPreset.broad),
            ctx=ctx,
            settings=_settings(),
            vector_store=MagicMock(),
            embedder=MagicMock(),
            graph_store=MagicMock(),
        )
    )

    assert resp.hits == []
    assert resp.expansion_resolved == BROAD_RESOLVED
    assert "include_retired" not in resp.expansion_resolved


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("min_score", 0.01),
        ("fallback", True),
        ("follow_superseded", False),
        ("include_stale", True),
        ("include_archived", True),
        ("include_retired", True),
    ],
)
def test_expansion_rejects_each_mutex_override(field_name: str, value: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        MemorySearchRequest(
            query="q",
            expansion=ExpansionPreset.broad,
            **{field_name: value},
        )

    message = str(exc_info.value)
    assert field_name in message
    assert "bundle" in message


def test_expansion_rejects_mutex_combinations() -> None:
    with pytest.raises(ValueError) as exc_info:
        MemorySearchRequest(
            query="q",
            expansion=ExpansionPreset.default,
            min_score=0.01,
            fallback=True,
        )

    message = str(exc_info.value)
    assert "min_score" in message
    assert "fallback" in message
    assert "bundle" in message
