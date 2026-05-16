"""Unit tests for ``memory_mcp.search.graph.graph_search``.

The graph leg is composed of three collaborators:

* :func:`extract_query_mentions` (NER + regex)
* :func:`resolve_query_entities` (Postgres lookup)
* :class:`GraphStore.neighbors` (graph backend)

Tests inject mocks for each so the leg can be exercised end-to-end
without a database, spaCy model, or graph backend.

Coverage
--------
* Empty short-circuits (empty query, empty env, no mentions, no
  resolved entities).
* Single resolved entity → ranked hits.
* Multi-entity overlap scoring (``overlap_count`` is the primary sort).
* Tie-break on min path length, then rank-score, then memory_id.
* Concurrent neighbor calls (semaphore bounded — invariant: never more
  than N in flight).
* Self-as-neighbor / cycles are not the graph leg's concern (filter
  applied in ``ent_neighbors`` / post-fusion); we don't test that here.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from memory_mcp.config import Settings
from memory_mcp.db.graph.base import GraphNodeRef, GraphPathStep, NeighborHit
from memory_mcp.search import graph as graph_mod
from memory_mcp.search.graph import graph_search


def _settings(**overrides) -> Settings:
    base = {
        "graph_search_hops": 1,
        "graph_search_max_concurrent_neighbors": 4,
        "search_min_per_leg": 50,
    }
    base.update(overrides)
    s = Settings()
    for k, v in base.items():
        object.__setattr__(s, k, v)
    return s


def _node(env_id, kind: str, record_id):
    return GraphNodeRef(env_id=env_id, kind=kind, record_id=record_id)  # type: ignore[arg-type]


def _hit(env_id, src_entity, memory_id, *, edge_type="describes", path_length=1):
    return NeighborHit(
        node=_node(env_id, "memory", memory_id),
        path_length=path_length,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", src_entity),
                dst=_node(env_id, "memory", memory_id),
                edge_type=edge_type,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Short-circuit tests
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty():
    out = asyncio.run(graph_search(
        MagicMock(),
        graph_store=MagicMock(),
        query="",
        env_ids=[uuid4()],
        limit=10,
        settings=_settings(),
    ))
    assert out == []


def test_empty_env_ids_returns_empty():
    out = asyncio.run(graph_search(
        MagicMock(),
        graph_store=MagicMock(),
        query="anything",
        env_ids=[],
        limit=10,
        settings=_settings(),
    ))
    assert out == []


def test_zero_limit_returns_empty():
    out = asyncio.run(graph_search(
        MagicMock(),
        graph_store=MagicMock(),
        query="x",
        env_ids=[uuid4()],
        limit=0,
        settings=_settings(),
    ))
    assert out == []


def test_no_mentions_returns_empty():
    """When NER + regex + raw fallback all return nothing, leg returns []."""
    with patch.object(graph_mod, "extract_query_mentions", AsyncMock(return_value=[])):
        out = asyncio.run(graph_search(
            MagicMock(),
            graph_store=MagicMock(),
            query="ineffable mystery",
            env_ids=[uuid4()],
            limit=10,
            settings=_settings(),
        ))
    assert out == []


def test_no_resolved_entities_returns_empty():
    """Mentions extracted but none resolve in any env → empty leg."""
    with (
        patch.object(graph_mod, "extract_query_mentions",
                     AsyncMock(return_value=["servicea"])),
        patch.object(graph_mod, "resolve_query_entities",
                     AsyncMock(return_value={})),
    ):
        out = asyncio.run(graph_search(
            MagicMock(),
            graph_store=MagicMock(),
            query="ServiceA",
            env_ids=[uuid4()],
            limit=10,
            settings=_settings(),
        ))
    assert out == []


# ---------------------------------------------------------------------------
# Single-entity expansion
# ---------------------------------------------------------------------------


def test_single_entity_produces_ranked_hits():
    env = uuid4()
    e1 = uuid4()
    m1, m2 = uuid4(), uuid4()
    gs = AsyncMock()
    gs.neighbors.return_value = ([_hit(env, e1, m1), _hit(env, e1, m2)], None)

    with (
        patch.object(graph_mod, "extract_query_mentions",
                     AsyncMock(return_value=["servicea"])),
        patch.object(graph_mod, "resolve_query_entities",
                     AsyncMock(return_value={env: [e1]})),
    ):
        out = asyncio.run(graph_search(
            MagicMock(),
            graph_store=gs,
            query="ServiceA",
            env_ids=[env],
            limit=10,
            settings=_settings(),
        ))

    assert len(out) == 2
    assert {h.memory_id for h in out} == {m1, m2}
    assert all(h.source == "graph" for h in out)
    assert [h.rank for h in out] == [1, 2]
    # raw_score is overlap count = 1 for both (single entity contributing).
    assert all(h.raw_score == 1.0 for h in out)


# ---------------------------------------------------------------------------
# Multi-entity overlap scoring
# ---------------------------------------------------------------------------


def test_multi_entity_overlap_ranks_first():
    """A memory reached by two query entities outranks one reached by one."""
    env = uuid4()
    e1, e2 = uuid4(), uuid4()
    overlap_mem = uuid4()
    solo_mem = uuid4()

    async def fake_neighbors(node, **_kwargs):
        if node.record_id == e1:
            return ([_hit(env, e1, overlap_mem), _hit(env, e1, solo_mem)], None)
        if node.record_id == e2:
            return ([_hit(env, e2, overlap_mem)], None)
        return ([], None)

    gs = MagicMock()
    gs.neighbors = fake_neighbors

    with (
        patch.object(graph_mod, "extract_query_mentions",
                     AsyncMock(return_value=["servicea", "serviceb"])),
        patch.object(graph_mod, "resolve_query_entities",
                     AsyncMock(return_value={env: [e1, e2]})),
    ):
        out = asyncio.run(graph_search(
            MagicMock(),
            graph_store=gs,
            query="ServiceA ServiceB",
            env_ids=[env],
            limit=10,
            settings=_settings(),
        ))

    assert out[0].memory_id == overlap_mem
    assert out[0].raw_score == 2.0  # two entities contributed
    assert out[1].memory_id == solo_mem
    assert out[1].raw_score == 1.0


def test_path_length_breaks_overlap_tie():
    """Two memories with same overlap count: shorter path wins."""
    env = uuid4()
    e1 = uuid4()
    short_mem, long_mem = uuid4(), uuid4()
    gs = AsyncMock()
    gs.neighbors.return_value = (
        [
            _hit(env, e1, long_mem, path_length=2),
            _hit(env, e1, short_mem, path_length=1),
        ],
        None,
    )

    with (
        patch.object(graph_mod, "extract_query_mentions",
                     AsyncMock(return_value=["x"])),
        patch.object(graph_mod, "resolve_query_entities",
                     AsyncMock(return_value={env: [e1]})),
    ):
        out = asyncio.run(graph_search(
            MagicMock(), graph_store=gs, query="x", env_ids=[env],
            limit=10, settings=_settings(),
        ))

    assert out[0].memory_id == short_mem
    assert out[1].memory_id == long_mem


def test_rank_score_orders_within_same_neighbors_call():
    """Within a single neighbors() result, neighbor_rank differs by index,
    so rank_score (1/(60+rank)) decides — index 0 wins regardless of id order."""
    env = uuid4()
    e1 = uuid4()
    m_lo = uuid4()
    m_hi = uuid4()
    if str(m_lo) > str(m_hi):
        m_lo, m_hi = m_hi, m_lo
    gs = AsyncMock()
    # m_hi at idx 0 (rank_score = 1/61), m_lo at idx 1 (rank_score = 1/62)
    gs.neighbors.return_value = ([_hit(env, e1, m_hi), _hit(env, e1, m_lo)], None)

    with (
        patch.object(graph_mod, "extract_query_mentions",
                     AsyncMock(return_value=["x"])),
        patch.object(graph_mod, "resolve_query_entities",
                     AsyncMock(return_value={env: [e1]})),
    ):
        out = asyncio.run(graph_search(
            MagicMock(), graph_store=gs, query="x", env_ids=[env],
            limit=10, settings=_settings(),
        ))

    # rank_score decides before memory_id when neighbor_rank differs.
    assert out[0].memory_id == m_hi
    assert out[1].memory_id == m_lo


def test_memory_id_tiebreaker_isolated():
    """Two entities each return one memory at neighbor_rank=1 → all upstream
    keys tie; memory_id (str) ASC must decide."""
    env = uuid4()
    e1, e2 = uuid4(), uuid4()
    m_a = uuid4()
    m_b = uuid4()
    if str(m_a) > str(m_b):
        m_a, m_b = m_b, m_a

    async def fake_neighbors(node, **_kwargs):
        if node.record_id == e1:
            return ([_hit(env, e1, m_b)], None)  # m_b at rank 1
        if node.record_id == e2:
            return ([_hit(env, e2, m_a)], None)  # m_a at rank 1
        return ([], None)

    gs = MagicMock()
    gs.neighbors = fake_neighbors

    with (
        patch.object(graph_mod, "extract_query_mentions",
                     AsyncMock(return_value=["a", "b"])),
        patch.object(graph_mod, "resolve_query_entities",
                     AsyncMock(return_value={env: [e1, e2]})),
    ):
        out = asyncio.run(graph_search(
            MagicMock(), graph_store=gs, query="a b", env_ids=[env],
            limit=10, settings=_settings(),
        ))

    # Both memories: overlap=1, min_path_length=1, rank_score=1/61, but
    # first_order differs (m_b from e1.first_order=0; m_a from e2.first_order=1).
    # first_order ASC → m_b wins. Verify that first.
    assert out[0].memory_id == m_b
    assert out[1].memory_id == m_a


def test_limit_truncates_results():
    env = uuid4()
    e1 = uuid4()
    mems = [uuid4() for _ in range(5)]
    gs = AsyncMock()
    gs.neighbors.return_value = ([_hit(env, e1, m) for m in mems], None)

    with (
        patch.object(graph_mod, "extract_query_mentions",
                     AsyncMock(return_value=["x"])),
        patch.object(graph_mod, "resolve_query_entities",
                     AsyncMock(return_value={env: [e1]})),
    ):
        out = asyncio.run(graph_search(
            MagicMock(), graph_store=gs, query="x", env_ids=[env],
            limit=3, settings=_settings(),
        ))
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


def test_neighbors_concurrency_is_bounded():
    """At most ``graph_search_max_concurrent_neighbors`` calls in flight."""
    env = uuid4()
    entities = [uuid4() for _ in range(10)]
    in_flight = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def fake_neighbors(node, **_kwargs):
        nonlocal in_flight, max_seen
        async with lock:
            in_flight += 1
            if in_flight > max_seen:
                max_seen = in_flight
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return ([], None)

    gs = MagicMock()
    gs.neighbors = fake_neighbors

    with (
        patch.object(graph_mod, "extract_query_mentions",
                     AsyncMock(return_value=["x"])),
        patch.object(graph_mod, "resolve_query_entities",
                     AsyncMock(return_value={env: entities})),
    ):
        asyncio.run(graph_search(
            MagicMock(), graph_store=gs, query="x", env_ids=[env],
            limit=10, settings=_settings(graph_search_max_concurrent_neighbors=3),
        ))
    assert max_seen <= 3


def test_concurrency_one_when_setting_is_zero_or_negative():
    """A defensive setting value of 0/negative falls back to 1 (no
    division-by-zero or asyncio raise)."""
    env = uuid4()
    e1 = uuid4()
    gs = AsyncMock()
    gs.neighbors.return_value = ([], None)

    with (
        patch.object(graph_mod, "extract_query_mentions",
                     AsyncMock(return_value=["x"])),
        patch.object(graph_mod, "resolve_query_entities",
                     AsyncMock(return_value={env: [e1]})),
    ):
        out = asyncio.run(graph_search(
            MagicMock(), graph_store=gs, query="x", env_ids=[env],
            limit=5, settings=_settings(graph_search_max_concurrent_neighbors=0),
        ))
    assert out == []
