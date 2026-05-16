"""Unit tests for graph-traversal relax fallback + min-score behavior."""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

import memory_mcp.graph as graph_mod
from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.graph import (
    MemNeighborsRequest,
    MemNeighborsResponse,
    MemRelatedHit,
    MemRelatedRequest,
    MemRelatedResponse,
    NeighborHitResponse,
    NeighborNodeResponse,
)
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryResponse


def _ctx(env_id: UUID) -> AgentContext:
    return AgentContext(
        agent_id=uuid4(),
        agent_name="test",
        attached_env_ids=[env_id],
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(graph_backend="postgres", search_fresh_max_wait_seconds=0)


def _neighbor_hit(env_id: UUID) -> NeighborHitResponse:
    return NeighborHitResponse(
        node=NeighborNodeResponse(
            kind="entity",
            id=uuid4(),
            name="neighbor",
            env_id=env_id,
        ),
        path_length=1,
        path=[],
        score=1.0,
    )


def _memory_response(env_id: UUID, *, title: str) -> MemoryResponse:
    now = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
    return MemoryResponse(
        id=uuid4(),
        env_id=env_id,
        kind=MemoryKind.fact,
        status=MemoryStatus.active,
        title=title,
        body=f"{title} body",
        tags=[],
        metadata={},
        salience=0.5,
        confidence=0.5,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _related_hit(env_id: UUID, *, score: float, title: str) -> MemRelatedHit:
    return MemRelatedHit(
        memory_id=uuid4(),
        score=score,
        shared_entity_ids=None,
        memory=_memory_response(env_id, title=title),
    )


def _patch_memory_neighbors_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    env_id: UUID,
) -> None:
    @asynccontextmanager
    async def fake_session_scope():
        class _Result:
            def scalar_one_or_none(self):
                return object()

        class _Session:
            async def get(self, _model, _memory_id):
                return SimpleNamespace(env_id=env_id)

            async def execute(self, _stmt):
                return _Result()

        yield _Session()

    monkeypatch.setattr(graph_mod, "session_scope", fake_session_scope)
    monkeypatch.setattr(graph_mod.rbac, "require", lambda *_args, **_kwargs: None)


@pytest.mark.asyncio
async def test_memory_neighbors_fallback_false_does_not_widen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    _patch_memory_neighbors_dependencies(monkeypatch, env_id)

    calls: list[tuple[int, tuple[str, ...] | None, bool]] = []

    async def fake_pass(request, *, env_id, graph_store, include_retired=False):
        calls.append(
            (
                request.hops,
                None if request.edge_types is None else tuple(request.edge_types),
                include_retired,
            )
        )
        return MemNeighborsResponse(hits=[], next_cursor=None)

    monkeypatch.setattr(graph_mod, "_memory_neighbors_pass", fake_pass)

    response = await graph_mod.memory_neighbors(
        MemNeighborsRequest(memory_id=uuid4(), hops=1, fallback=False),
        ctx=_ctx(env_id),
        settings=_settings(),
        graph_store=object(),
    )

    assert calls == [(1, None, False)]
    assert response.fallback_used == []


@pytest.mark.asyncio
async def test_memory_neighbors_first_pass_hit_keeps_fallback_used_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    _patch_memory_neighbors_dependencies(monkeypatch, env_id)

    calls: list[int] = []

    async def fake_pass(request, *, env_id, graph_store, include_retired=False):
        calls.append(request.hops)
        return MemNeighborsResponse(hits=[_neighbor_hit(env_id)], next_cursor="strict")

    monkeypatch.setattr(graph_mod, "_memory_neighbors_pass", fake_pass)

    response = await graph_mod.memory_neighbors(
        MemNeighborsRequest(memory_id=uuid4(), hops=1, fallback=True),
        ctx=_ctx(env_id),
        settings=_settings(),
        graph_store=object(),
    )

    assert calls == [1]
    assert response.next_cursor == "strict"
    assert response.fallback_used == []


@pytest.mark.asyncio
async def test_memory_neighbors_widen_hops_fires_on_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    _patch_memory_neighbors_dependencies(monkeypatch, env_id)

    calls: list[int] = []

    async def fake_pass(request, *, env_id, graph_store, include_retired=False):
        calls.append(request.hops)
        if request.hops == 2:
            return MemNeighborsResponse(hits=[_neighbor_hit(env_id)], next_cursor="widened")
        return MemNeighborsResponse(hits=[], next_cursor=None)

    monkeypatch.setattr(graph_mod, "_memory_neighbors_pass", fake_pass)

    response = await graph_mod.memory_neighbors(
        MemNeighborsRequest(memory_id=uuid4(), hops=1, fallback=True),
        ctx=_ctx(env_id),
        settings=_settings(),
        graph_store=object(),
    )

    assert calls == [1, 2]
    assert response.next_cursor == "widened"
    assert response.fallback_used == ["widen_hops"]


@pytest.mark.asyncio
async def test_memory_neighbors_drop_predicate_after_widen_hops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    _patch_memory_neighbors_dependencies(monkeypatch, env_id)

    calls: list[tuple[int, tuple[str, ...] | None, bool]] = []

    async def fake_pass(request, *, env_id, graph_store, include_retired=False):
        calls.append(
            (
                request.hops,
                None if request.edge_types is None else tuple(request.edge_types),
                include_retired,
            )
        )
        if request.hops == 2 and request.edge_types is None:
            return MemNeighborsResponse(hits=[_neighbor_hit(env_id)], next_cursor="predicate-dropped")
        return MemNeighborsResponse(hits=[], next_cursor=None)

    monkeypatch.setattr(graph_mod, "_memory_neighbors_pass", fake_pass)

    response = await graph_mod.memory_neighbors(
        MemNeighborsRequest(
            memory_id=uuid4(),
            hops=1,
            edge_types=["derives_from"],
            fallback=True,
        ),
        ctx=_ctx(env_id),
        settings=_settings(),
        graph_store=object(),
    )

    assert calls == [
        (1, ("derives_from",), False),
        (2, ("derives_from",), False),
        (2, None, False),
    ]
    assert response.next_cursor == "predicate-dropped"
    assert response.fallback_used == ["widen_hops", "drop_predicate"]


@pytest.mark.asyncio
async def test_memory_neighbors_include_retired_fires_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    _patch_memory_neighbors_dependencies(monkeypatch, env_id)

    calls: list[tuple[int, tuple[str, ...] | None, bool]] = []

    async def fake_pass(request, *, env_id, graph_store, include_retired=False):
        calls.append(
            (
                request.hops,
                None if request.edge_types is None else tuple(request.edge_types),
                include_retired,
            )
        )
        if include_retired:
            return MemNeighborsResponse(hits=[_neighbor_hit(env_id)], next_cursor="retired")
        return MemNeighborsResponse(hits=[], next_cursor=None)

    monkeypatch.setattr(graph_mod, "_memory_neighbors_pass", fake_pass)

    response = await graph_mod.memory_neighbors(
        MemNeighborsRequest(
            memory_id=uuid4(),
            hops=1,
            edge_types=["derives_from"],
            fallback=True,
        ),
        ctx=_ctx(env_id),
        settings=_settings(),
        graph_store=object(),
    )

    assert calls == [
        (1, ("derives_from",), False),
        (2, ("derives_from",), False),
        (2, None, False),
        (2, None, True),
    ]
    assert response.next_cursor == "retired"
    assert response.fallback_used == [
        "widen_hops",
        "drop_predicate",
        "include_retired",
    ]


@pytest.mark.asyncio
async def test_memory_neighbors_widen_hops_caps_at_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    _patch_memory_neighbors_dependencies(monkeypatch, env_id)

    calls: list[tuple[int, bool]] = []

    async def fake_pass(request, *, env_id, graph_store, include_retired=False):
        calls.append((request.hops, include_retired))
        return MemNeighborsResponse(hits=[], next_cursor=None)

    monkeypatch.setattr(graph_mod, "_memory_neighbors_pass", fake_pass)

    response = await graph_mod.memory_neighbors(
        MemNeighborsRequest(memory_id=uuid4(), hops=3, fallback=True),
        ctx=_ctx(env_id),
        settings=_settings(),
        graph_store=object(),
    )

    assert calls == [(3, False), (3, True)]
    assert max(hops for hops, _ in calls) == 3
    assert response.fallback_used == ["include_retired"]


def test_mem_related_min_score_rejects_non_semantic_relation() -> None:
    # Current schema exposes ``shared_entity`` as the non-semantic relation.
    with pytest.raises(
        ValidationError,
        match="relation='shared_entity'",
    ):
        MemRelatedRequest(
            memory_id=uuid4(),
            relation="shared_entity",
            min_score=0.5,
        )


@pytest.mark.asyncio
async def test_memory_related_min_score_drops_subthreshold_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()

    async def fake_seed_memory(*_args, **_kwargs):
        return SimpleNamespace(env_id=env_id)

    calls: list[bool] = []

    async def fake_related_pass(request, *, env_id, settings, vector_store, include_retired=False):
        calls.append(include_retired)
        return MemRelatedResponse(
            hits=[
                _related_hit(env_id, score=0.75, title="high"),
                _related_hit(env_id, score=0.40, title="low"),
            ],
            note="ok",
        )

    monkeypatch.setattr(graph_mod, "_resolve_seed_memory", fake_seed_memory)
    monkeypatch.setattr(graph_mod, "_memory_related_pass", fake_related_pass)

    response = await graph_mod.memory_related(
        MemRelatedRequest(
            memory_id=uuid4(),
            relation="semantic",
            min_score=0.5,
        ),
        ctx=_ctx(env_id),
        settings=_settings(),
    )

    assert calls == [False]
    assert [hit.score for hit in response.hits] == [0.75]
    assert response.fallback_used == []


@pytest.mark.asyncio
async def test_memory_related_min_score_empty_triggers_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()

    async def fake_seed_memory(*_args, **_kwargs):
        return SimpleNamespace(env_id=env_id)

    calls: list[bool] = []

    async def fake_related_pass(request, *, env_id, settings, vector_store, include_retired=False):
        calls.append(include_retired)
        score = 0.45 if not include_retired else 0.80
        return MemRelatedResponse(
            hits=[_related_hit(env_id, score=score, title="candidate")],
            note="ok",
        )

    monkeypatch.setattr(graph_mod, "_resolve_seed_memory", fake_seed_memory)
    monkeypatch.setattr(graph_mod, "_memory_related_pass", fake_related_pass)

    response = await graph_mod.memory_related(
        MemRelatedRequest(
            memory_id=uuid4(),
            relation="semantic",
            min_score=0.5,
            fallback=True,
        ),
        ctx=_ctx(env_id),
        settings=_settings(),
    )

    assert calls == [False, True]
    assert [hit.score for hit in response.hits] == [0.80]
    assert response.fallback_used == ["include_retired"]
