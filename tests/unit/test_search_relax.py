"""Unit tests for the v0.12 search relax / tighten knobs.

Two new knobs on ``MemorySearchRequest``:

* ``min_score: float | None`` — post-fusion threshold (the *tighten* lever).
* ``fallback: bool`` — auto-broaden cascade on empty results (the *loosen* lever).

These tests cover:

1. ``min_score`` correctly drops sub-threshold hits and is applied after the
   salience boost.
2. The ``fallback`` cascade runs the 4 documented steps in order, gated on
   the prior pass returning 0 hits.
3. ``fallback`` reports which steps fired in ``response.fallback_used``.
4. ``fallback`` is a no-op for ``mode=id``.
5. ``min_score`` interacting with ``fallback`` drives further broadening.
"""

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
from memory_mcp.search import api as search_api
from memory_mcp.search.api import (
    _FALLBACK_STEPS,
    MemorySearchRequest,
    _step_boost_limit,
    _step_drop_filters,
    _step_widen_lifecycle,
    _step_widen_mode,
    memory_search,
)
from memory_mcp.search.ranking import RankedHit


def _settings(**overrides) -> Settings:
    s = Settings()
    object.__setattr__(s, "search_min_per_leg", 1)
    object.__setattr__(s, "search_fresh_max_wait_seconds", 0.01)
    for key, value in overrides.items():
        object.__setattr__(s, key, value)
    return s


def _memory(memory_id: UUID, env_id: UUID, *, salience: float = 0.5) -> Memory:
    now = dt.datetime.now(dt.UTC)
    return Memory(
        id=memory_id,
        env_id=env_id,
        kind="fact",
        status="active",
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


# ---------------------------------------------------------------------------
# Step builders — pure functions, no mocks needed
# ---------------------------------------------------------------------------


def test_step_widen_mode_lex_to_hybrid() -> None:
    req = MemorySearchRequest(query="q", mode="lex")
    out = _step_widen_mode(req)
    assert out is not None
    assert out.mode == "hybrid"


@pytest.mark.parametrize("mode", ["hybrid", "sem", "graph", "auto"])
def test_step_widen_mode_noop_for_broader_modes(mode: str) -> None:
    req = MemorySearchRequest(query="q", mode=mode)  # type: ignore[arg-type]
    assert _step_widen_mode(req) is None


def test_step_drop_filters_drops_all_optional_filters() -> None:
    req = MemorySearchRequest(
        query="q",
        kinds=["fact"],
        tags=["x"],
        created_after=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        created_before=dt.datetime(2026, 2, 1, tzinfo=dt.UTC),
        updated_after=dt.datetime(2026, 1, 15, tzinfo=dt.UTC),
    )
    out = _step_drop_filters(req)
    assert out is not None
    assert out.kinds is None
    assert out.tags is None
    assert out.created_after is None
    assert out.created_before is None
    assert out.updated_after is None


def test_step_drop_filters_noop_when_no_filters() -> None:
    req = MemorySearchRequest(query="q")
    assert _step_drop_filters(req) is None


def test_step_widen_lifecycle_sets_stale_and_archived() -> None:
    req = MemorySearchRequest(query="q")
    out = _step_widen_lifecycle(req)
    assert out is not None
    assert out.include_stale is True
    assert out.include_archived is True


def test_step_widen_lifecycle_noop_when_already_widened() -> None:
    req = MemorySearchRequest(query="q", include_stale=True, include_archived=True)
    assert _step_widen_lifecycle(req) is None


def test_step_boost_limit_drops_follow_superseded_and_boosts_limit() -> None:
    req = MemorySearchRequest(query="q", limit=10, follow_superseded=True)
    out = _step_boost_limit(req)
    assert out is not None
    assert out.follow_superseded is False
    assert out.limit == 50


def test_step_boost_limit_caps_at_100() -> None:
    req = MemorySearchRequest(query="q", limit=50, follow_superseded=False)
    out = _step_boost_limit(req)
    assert out is not None
    assert out.limit == 100


def test_step_boost_limit_noop_when_already_at_cap_and_no_follow() -> None:
    req = MemorySearchRequest(query="q", limit=100, follow_superseded=False)
    assert _step_boost_limit(req) is None


def test_fallback_steps_in_documented_order() -> None:
    names = [name for name, _ in _FALLBACK_STEPS]
    assert names == ["mode->hybrid", "drop_filters", "widen_lifecycle", "boost_limit"]


# ---------------------------------------------------------------------------
# min_score — post-fusion threshold (the *tighten* lever)
# ---------------------------------------------------------------------------


def _patch_lex_only(monkeypatch, ranked_lists: list[list[RankedHit]], memories: dict[UUID, Memory]) -> None:
    call_idx = {"n": 0}

    async def fake_do_lex(*_args, **_kwargs):
        idx = call_idx["n"]
        call_idx["n"] += 1
        return ranked_lists[idx] if idx < len(ranked_lists) else []

    @asynccontextmanager
    async def fake_session_scope():
        yield MagicMock()

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(search_api, "_do_lex", fake_do_lex)
    monkeypatch.setattr(search_api, "_do_sem", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_do_graph", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_hydrate_memories", AsyncMock(return_value=memories))
    monkeypatch.setattr(search_api, "_bulk_load_tag_names", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_projection_status", AsyncMock(return_value=[]))


def test_min_score_drops_sub_threshold_hits(monkeypatch) -> None:
    env_id = uuid4()
    high_id = uuid4()
    low_id = uuid4()
    memories = {
        high_id: _memory(high_id, env_id, salience=1.0),
        low_id: _memory(low_id, env_id, salience=0.0),
    }
    # Build a single lex leg. Rank 1 → RRF score ~1/61 ≈ 0.0164. After
    # salience boost: high → ~0.0246, low → ~0.0164. Threshold 0.02
    # should keep only ``high_id``.
    ranked = [
        [
            RankedHit(memory_id=high_id, rank=1, raw_score=1.0, source="lex"),
            RankedHit(memory_id=low_id, rank=2, raw_score=0.5, source="lex"),
        ]
    ]
    _patch_lex_only(monkeypatch, ranked, memories)

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query="q", mode="lex", min_score=0.02)
    resp = asyncio.run(memory_search(req, ctx=ctx, settings=_settings()))

    assert [h.memory.id for h in resp.hits] == [high_id]


def test_min_score_none_keeps_all_hits(monkeypatch) -> None:
    env_id = uuid4()
    mid_a = uuid4()
    mid_b = uuid4()
    memories = {a: _memory(a, env_id) for a in (mid_a, mid_b)}
    ranked = [
        [
            RankedHit(memory_id=mid_a, rank=1, raw_score=1.0, source="lex"),
            RankedHit(memory_id=mid_b, rank=2, raw_score=0.5, source="lex"),
        ]
    ]
    _patch_lex_only(monkeypatch, ranked, memories)

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query="q", mode="lex")
    resp = asyncio.run(memory_search(req, ctx=ctx, settings=_settings()))

    assert {h.memory.id for h in resp.hits} == {mid_a, mid_b}


def test_min_score_rejects_negative_value() -> None:
    with pytest.raises(ValueError):  # pydantic validation
        MemorySearchRequest(query="q", min_score=-0.1)


# ---------------------------------------------------------------------------
# fallback — broaden-on-empty cascade
# ---------------------------------------------------------------------------


def test_fallback_disabled_returns_empty_without_widening(monkeypatch) -> None:
    env_id = uuid4()
    _patch_lex_only(monkeypatch, [[]], memories={})

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query="q", mode="lex", fallback=False)
    resp = asyncio.run(memory_search(req, ctx=ctx, settings=_settings()))

    assert resp.hits == []
    assert resp.fallback_used == []


def test_fallback_widens_lex_to_hybrid_on_first_step(monkeypatch) -> None:
    """First step: lex→hybrid. Lex empty; hybrid pass returns a hit."""
    env_id = uuid4()
    mid = uuid4()
    memories = {mid: _memory(mid, env_id)}

    pass_modes: list[str] = []

    async def fake_do_lex(_session, req, *_args, **_kwargs):
        pass_modes.append(f"lex:{req.mode}")
        # First lex call (mode=lex) returns nothing.
        # Second lex call (within mode=hybrid) returns a hit.
        if req.mode == "lex":
            return []
        return [RankedHit(memory_id=mid, rank=1, raw_score=1.0, source="lex")]

    async def fake_do_sem(_session, req, *_args, **_kwargs):
        pass_modes.append(f"sem:{req.mode}")
        return []

    async def fake_do_graph(_session, req, *_args, **_kwargs):
        pass_modes.append(f"graph:{req.mode}")
        return []

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
    monkeypatch.setattr(search_api, "_default_vector_store", lambda _s: MagicMock())
    monkeypatch.setattr(search_api, "get_embedder", lambda _s: MagicMock())

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query="q", mode="lex", fallback=True)
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

    assert resp.fallback_used == ["mode->hybrid"]
    assert [h.memory.id for h in resp.hits] == [mid]


def test_fallback_skips_widen_mode_when_already_hybrid(monkeypatch) -> None:
    """mode=hybrid + tags filter — step 1 no-op, step 2 (drop_filters) fires."""
    env_id = uuid4()
    mid = uuid4()
    memories = {mid: _memory(mid, env_id)}
    call_count = {"lex": 0}

    async def fake_do_lex(_session, req, *_args, **_kwargs):
        call_count["lex"] += 1
        # First pass has tag filter; we simulate the tag filter excluding
        # everything. After drop_filters fires, we return a hit.
        if req.tags is not None:
            return []
        return [RankedHit(memory_id=mid, rank=1, raw_score=1.0, source="lex")]

    @asynccontextmanager
    async def fake_session_scope():
        yield MagicMock()

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(search_api, "_do_lex", fake_do_lex)
    monkeypatch.setattr(search_api, "_do_sem", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_do_graph", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_hydrate_memories", AsyncMock(return_value=memories))
    monkeypatch.setattr(search_api, "_bulk_load_tag_names", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_projection_status", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_default_vector_store", lambda _s: MagicMock())
    monkeypatch.setattr(search_api, "get_embedder", lambda _s: MagicMock())

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query="q", mode="hybrid", tags=["x"], fallback=True)
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

    # Step 1 (widen mode) was a no-op so it's NOT in fallback_used.
    # Step 2 (drop_filters) fired and produced hits.
    assert resp.fallback_used == ["drop_filters"]
    assert [h.memory.id for h in resp.hits] == [mid]


def test_fallback_traverses_all_steps_when_each_pass_empty(monkeypatch) -> None:
    """Every cascade pass returns empty — fallback_used lists all 4 steps."""
    env_id = uuid4()
    pass_count = {"n": 0}

    async def fake_do_lex(*_args, **_kwargs):
        pass_count["n"] += 1
        return []

    @asynccontextmanager
    async def fake_session_scope():
        yield MagicMock()

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(search_api, "_do_lex", fake_do_lex)
    monkeypatch.setattr(search_api, "_do_sem", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_do_graph", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_hydrate_memories", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_bulk_load_tag_names", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_projection_status", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_default_vector_store", lambda _s: MagicMock())
    monkeypatch.setattr(search_api, "get_embedder", lambda _s: MagicMock())

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(
        query="q",
        mode="lex",
        tags=["x"],
        fallback=True,
        limit=5,
    )
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

    assert resp.hits == []
    # All 4 steps fired.
    assert resp.fallback_used == [
        "mode->hybrid",
        "drop_filters",
        "widen_lifecycle",
        "boost_limit",
    ]


def test_fallback_does_not_run_when_initial_pass_has_hits(monkeypatch) -> None:
    env_id = uuid4()
    mid = uuid4()
    memories = {mid: _memory(mid, env_id)}
    ranked = [[RankedHit(memory_id=mid, rank=1, raw_score=1.0, source="lex")]]
    _patch_lex_only(monkeypatch, ranked, memories)

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(query="q", mode="lex", fallback=True)
    resp = asyncio.run(memory_search(req, ctx=ctx, settings=_settings()))

    assert [h.memory.id for h in resp.hits] == [mid]
    assert resp.fallback_used == []


def test_fallback_disabled_for_mode_id(monkeypatch) -> None:
    """``mode=id`` never participates in the cascade."""
    env_id = uuid4()
    memories: dict[UUID, Memory] = {}

    @asynccontextmanager
    async def fake_session_scope():
        yield MagicMock()

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(search_api, "_hydrate_memories", AsyncMock(return_value=memories))
    monkeypatch.setattr(search_api, "_bulk_load_tag_names", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_projection_status", AsyncMock(return_value=[]))

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    # Unknown id → empty hits.
    req = MemorySearchRequest(
        query="",
        mode="id",
        ids=[uuid4()],
        fallback=True,
    )
    resp = asyncio.run(memory_search(req, ctx=ctx, settings=_settings()))

    assert resp.hits == []
    assert resp.fallback_used == []


def test_min_score_drives_further_fallback(monkeypatch) -> None:
    """If ``min_score`` empties an otherwise-non-empty pass, fallback continues."""
    env_id = uuid4()
    weak_id = uuid4()
    strong_id = uuid4()
    memories = {
        weak_id: _memory(weak_id, env_id, salience=0.0),
        strong_id: _memory(strong_id, env_id, salience=1.0),
    }

    # Pass 1 (mode=lex): returns the weak hit only (score will be ~0.0164,
    # below min_score=0.02 → filtered → empty → fallback continues).
    # Pass 2 (mode=hybrid after widen): returns the strong hit (score
    # after salience boost ~0.0246 → above 0.02 → kept).
    async def fake_do_lex(_session, req, *_args, **_kwargs):
        if req.mode == "lex":
            return [RankedHit(memory_id=weak_id, rank=1, raw_score=1.0, source="lex")]
        return [RankedHit(memory_id=strong_id, rank=1, raw_score=1.0, source="lex")]

    @asynccontextmanager
    async def fake_session_scope():
        yield MagicMock()

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(search_api, "_do_lex", fake_do_lex)
    monkeypatch.setattr(search_api, "_do_sem", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_do_graph", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_hydrate_memories", AsyncMock(return_value=memories))
    monkeypatch.setattr(search_api, "_bulk_load_tag_names", AsyncMock(return_value={}))
    monkeypatch.setattr(search_api, "_projection_status", AsyncMock(return_value=[]))
    monkeypatch.setattr(search_api, "_default_vector_store", lambda _s: MagicMock())
    monkeypatch.setattr(search_api, "get_embedder", lambda _s: MagicMock())

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
    req = MemorySearchRequest(
        query="q",
        mode="lex",
        fallback=True,
        min_score=0.02,
    )
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

    assert resp.fallback_used == ["mode->hybrid"]
    assert [h.memory.id for h in resp.hits] == [strong_id]
