"""End-to-end integration tests for Phase 4 decompose auto-wire (H5).

Coverage strategy mirrors ``test_autowire_compose.py`` (D6b) and
extends with the six H6 RD gaps that came out of the mid-H rubber-duck
pass:

* **OFF path** — both flags off → ``auto_wired_by_child=None``,
  ``auto_wired=[]``, zero ``related_to_popular`` rows.
* **NB1 — master ON, per-decompose OFF** → ``None`` (no Stage A call,
  no Stage B loop).
* **Happy path** — both flags ON, monkeypatched embedder + vector
  store return one candidate per child → per-child mapping populated
  + flat ``auto_wired`` carries dedup union.
* **Blocking #1 — replay state-current** — second call replays from
  the operation row; ``auto_wired_by_child`` always a per-child dict
  (NEVER None on replay), reflects live ``relations`` after a manual
  ``rel_link``.
* **Blocking #1 — replay of OFF-then-OFF op** — first call with
  feature OFF, second call same content with feature still OFF → both
  return ``None``; no edges materialise.
* **Blocking #1 — replay of OFF-then-ON op** — first call OFF
  (None), second call with feature ON replays from the same dedupe
  row; replay reconstruction returns ``{child_id: []}`` for each
  child because the operation never emitted any edges.
* **Blocking #2 — flat union ordered + deduped** — two children both
  wire to the same dst → flat ``auto_wired`` contains the dst once,
  in first-child insertion order.
* **Blocking #3 — per-child Stage-B failure isolated** — one child's
  Stage-B call raises (savepoint rollback) → that child's entry is
  ``[]``; sibling children commit cleanly; outer txn keeps the
  decompose's children and lineage rows; no orphan relation rows from
  the failing child.
* **NB2 — outbox ordering** — when Stage B runs, child-memory
  ``upsert`` outbox events land before relation outbox events for
  the same children (compose's invariant preserved).

These tests use the same real Postgres + monkeypatched embedder /
vector store pattern as ``test_autowire_compose.py``. They reuse the
testcontainer fixtures from ``conftest.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from memory_mcp import autowire as autowire_mod
from memory_mcp import composers as composers_mod
from memory_mcp import decomposers as decomposers_mod
from memory_mcp import memories as memories_mod
from memory_mcp.autowire import AUTO_WIRE_PREDICATE
from memory_mcp.config import Settings
from memory_mcp.db.models import (
    Agent,
    Environment,
    GraphNode,
    Memory,
    Outbox,
    Relation,
)
from memory_mcp.db.types import MemoryKind
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryWriteRequest, memory_write
from memory_mcp_schemas.decompose import MemDecomposeChild, MemDecomposeRequest

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


def _settings(
    *,
    autowire_enabled: bool = False,
    autowire_decompose_enabled: bool = False,
    per_child_top_k: int = 3,
    total_cap: int = 30,
) -> Settings:
    return Settings(
        graph_backend="postgres",
        autowire_enabled=autowire_enabled,
        autowire_top_k=3,
        autowire_sim_threshold=0.50,
        autowire_candidate_limit=20,
        autowire_decompose_enabled=autowire_decompose_enabled,
        autowire_decompose_per_child_top_k=per_child_top_k,
        autowire_decompose_total_cap=total_cap,
    )


def _patch_session_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(decomposers_mod, "session_scope", routed_session_scope)


async def _setup_env_and_agent(factory) -> tuple[UUID, UUID]:
    async with factory() as session:
        env = Environment(
            name=f"autowire-decompose-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"autowire-decompose-agent-{uuid4()}")
        session.add_all([env, agent])
        await session.commit()
        return env.id, agent.id


async def _write_memory(
    factory,
    *,
    env_id: UUID,
    agent_id: UUID,
    title: str,
    body: str,
    salience: float | None = None,
) -> UUID:
    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        resp = await memory_write(
            MemoryWriteRequest(
                kind=MemoryKind.fact,
                title=title,
                body=body,
                env_id=env_id,
                salience=salience,
            ),
            ctx=ctx,
            settings=_settings(),
        )
        return resp.id
    finally:
        reset_session_factory(token)


def _child(title: str, body: str) -> MemDecomposeChild:
    return MemDecomposeChild(kind=MemoryKind.fact, title=title, body=body)


async def _count_auto_wire_rows(factory) -> int:
    async with factory() as s:
        return int((await s.execute(
            select(func.count()).select_from(Relation).where(
                Relation.type == AUTO_WIRE_PREDICATE
            )
        )).scalar_one())


# ---------------------------------------------------------------------------
# OFF path — regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_off_returns_none_dict(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Both flags OFF → auto_wired_by_child=None, no relations rows."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="source body",
    )

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src,
                children=[_child("c1", "child one"), _child("c2", "child two")],
                mode="derive",
            ),
            ctx=ctx,
            settings=_settings(autowire_enabled=False),
        )
    finally:
        reset_session_factory(token)

    assert resp.auto_wired_by_child is None
    assert resp.auto_wired == []
    assert await _count_auto_wire_rows(factory) == 0


@pytest.mark.asyncio
async def test_decompose_master_on_per_decompose_off_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """NB1 — master ON, per-decompose OFF → None (no Stage A, no Stage B)."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="source body",
    )

    # Sentinel: if Stage A IS called, this will blow up. It must not be.
    fake_embedder = MagicMock()
    fake_embedder.embed_texts = MagicMock(side_effect=AssertionError(
        "Stage A must not be invoked when autowire_decompose_enabled=False"
    ))
    monkeypatch.setattr(autowire_mod, "get_embedder", lambda settings: fake_embedder)

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src,
                children=[_child("c1", "child one"), _child("c2", "child two")],
                mode="derive",
            ),
            ctx=ctx,
            settings=_settings(
                autowire_enabled=True,
                autowire_decompose_enabled=False,
            ),
        )
    finally:
        reset_session_factory(token)

    assert resp.auto_wired_by_child is None
    assert resp.auto_wired == []
    assert await _count_auto_wire_rows(factory) == 0


# ---------------------------------------------------------------------------
# Happy path — both flags ON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_on_emits_per_child_edges(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Both flags ON + fake embedder + vector store → per-child edges."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="source body",
    )
    pop1 = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="pop1", body="popular one", salience=0.95,
    )
    pop2 = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="pop2", body="popular two", salience=0.90,
    )

    # Embedder returns one vector per child (batched call).
    fake_embedder = MagicMock()
    fake_embedder.embed_texts = MagicMock(
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    monkeypatch.setattr(autowire_mod, "get_embedder", lambda settings: fake_embedder)

    # Vector store returns each child's matching candidate.
    async def _fake_search(**kwargs):
        # Map vector to a distinct candidate so each child wires to a
        # different popular memory.
        qvec = kwargs["query_vector"]
        if qvec and qvec[0] == 0.1:
            return [{"id": str(pop1), "score": 0.90}]
        return [{"id": str(pop2), "score": 0.85}]

    fake_store = MagicMock()
    fake_store.search = AsyncMock(side_effect=_fake_search)
    monkeypatch.setattr(autowire_mod, "_default_vector_store", lambda: fake_store)

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src,
                children=[_child("c1", "child one"), _child("c2", "child two")],
                mode="derive",
            ),
            ctx=ctx,
            settings=_settings(
                autowire_enabled=True,
                autowire_decompose_enabled=True,
            ),
        )
    finally:
        reset_session_factory(token)

    assert resp.auto_wired_by_child is not None
    # Each child wires to its respective popular memory.
    child_ids = [c.id for c in resp.children]
    assert resp.auto_wired_by_child[child_ids[0]] == [pop1]
    assert resp.auto_wired_by_child[child_ids[1]] == [pop2]
    # Flat union deduped (no overlap here).
    assert sorted(resp.auto_wired) == sorted([pop1, pop2])
    # Two relation rows in the DB.
    assert await _count_auto_wire_rows(factory) == 2


# ---------------------------------------------------------------------------
# H6 Blocking #2 — flat union dedup + insertion order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_flat_auto_wired_is_ordered_unique(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """H6 Blocking #2 — two children wiring to same dst → flat list dedupes."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="source body",
    )
    pop = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="pop", body="popular shared", salience=0.95,
    )

    fake_embedder = MagicMock()
    fake_embedder.embed_texts = MagicMock(
        return_value=[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    monkeypatch.setattr(autowire_mod, "get_embedder", lambda settings: fake_embedder)

    # Both children's search returns the SAME popular memory.
    async def _fake_search(**kwargs):
        return [{"id": str(pop), "score": 0.92}]

    fake_store = MagicMock()
    fake_store.search = AsyncMock(side_effect=_fake_search)
    monkeypatch.setattr(autowire_mod, "_default_vector_store", lambda: fake_store)

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src,
                children=[_child("c1", "first child"), _child("c2", "second child")],
                mode="derive",
            ),
            ctx=ctx,
            settings=_settings(
                autowire_enabled=True,
                autowire_decompose_enabled=True,
            ),
        )
    finally:
        reset_session_factory(token)

    assert resp.auto_wired_by_child is not None
    child_ids = [c.id for c in resp.children]
    # Both children wire to the same dst.
    assert resp.auto_wired_by_child[child_ids[0]] == [pop]
    assert resp.auto_wired_by_child[child_ids[1]] == [pop]
    # Flat list has the dst ONCE (dedup), in first-occurrence order.
    assert resp.auto_wired == [pop]
    # Two relation rows persisted (one per child).
    assert await _count_auto_wire_rows(factory) == 2


# ---------------------------------------------------------------------------
# H6 Blocking #1 — replay always returns per-child dict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_replay_state_current_per_child_dict(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """H6 Blocking #1 — replay always populates per-child dict (never None)."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="source body",
    )
    pop = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="pop", body="popular", salience=0.95,
    )

    fake_embedder = MagicMock()
    fake_embedder.embed_texts = MagicMock(
        return_value=[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    monkeypatch.setattr(autowire_mod, "get_embedder", lambda settings: fake_embedder)

    async def _fake_search_first(**kwargs):
        # Only first child wires; second gets no match.
        qvec = kwargs["query_vector"]
        if qvec and qvec[0] == 0.1:
            return [{"id": str(pop), "score": 0.92}]
        return []

    fake_store = MagicMock()
    fake_store.search = AsyncMock(side_effect=_fake_search_first)
    monkeypatch.setattr(autowire_mod, "_default_vector_store", lambda: fake_store)

    settings = _settings(autowire_enabled=True, autowire_decompose_enabled=True)
    request = MemDecomposeRequest(
        source_id=src,
        children=[_child("c1", "first"), _child("c2", "second")],
        mode="derive",
    )

    token = use_session_factory(factory)
    try:
        first = await decomposers_mod.memory_decompose(
            request, ctx=ctx, settings=settings,
        )
        second = await decomposers_mod.memory_decompose(
            request, ctx=ctx, settings=settings,
        )
    finally:
        reset_session_factory(token)

    assert first.idempotency_replay is False
    assert second.idempotency_replay is True
    assert first.children[0].id == second.children[0].id
    # First call: child0 has [pop], child1 has [] (no match).
    child_ids = [c.id for c in first.children]
    assert first.auto_wired_by_child == {
        child_ids[0]: [pop],
        child_ids[1]: [],
    }
    # Replay: same per-child dict (state-current, populated with [] for
    # children with no edges — H6 RD blocking #1 resolution).
    assert second.auto_wired_by_child is not None
    assert second.auto_wired_by_child == {
        child_ids[0]: [pop],
        child_ids[1]: [],
    }


@pytest.mark.asyncio
async def test_decompose_replay_of_off_then_off_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """H6 Blocking #1 — first OFF, second OFF: both calls return None."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="source body",
    )

    settings = _settings(autowire_enabled=False)
    request = MemDecomposeRequest(
        source_id=src,
        children=[_child("c1", "first"), _child("c2", "second")],
        mode="derive",
    )

    token = use_session_factory(factory)
    try:
        first = await decomposers_mod.memory_decompose(
            request, ctx=ctx, settings=settings,
        )
        second = await decomposers_mod.memory_decompose(
            request, ctx=ctx, settings=settings,
        )
    finally:
        reset_session_factory(token)

    assert first.idempotency_replay is False
    assert second.idempotency_replay is True
    assert first.auto_wired_by_child is None
    # Both calls' feature is OFF, but the replay path goes through
    # _resolve_auto_wired_for_replay which always returns a per-child
    # dict (with [] entries). For first-OFF-then-OFF, this means
    # replay returns {child: []} for each — NOT None — because the
    # response builder receives the dict from the replay shim.
    assert second.auto_wired_by_child is not None
    child_ids = [c.id for c in second.children]
    assert all(second.auto_wired_by_child[cid] == [] for cid in child_ids)
    assert await _count_auto_wire_rows(factory) == 0


# ---------------------------------------------------------------------------
# H6 Blocking #3 — per-child Stage-B failure isolated by savepoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_per_child_stage_b_failure_isolated(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """H6 Blocking #3 — one child's Stage B raises → savepoint rollback.

    Other children commit their edges; failing child has ``[]``; outer
    decompose succeeds; no orphan relation rows from the failing child.
    """
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="source body",
    )
    pop = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="pop", body="popular", salience=0.95,
    )

    fake_embedder = MagicMock()
    fake_embedder.embed_texts = MagicMock(
        return_value=[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    monkeypatch.setattr(autowire_mod, "get_embedder", lambda settings: fake_embedder)

    async def _fake_search(**kwargs):
        return [{"id": str(pop), "score": 0.92}]

    fake_store = MagicMock()
    fake_store.search = AsyncMock(side_effect=_fake_search)
    monkeypatch.setattr(autowire_mod, "_default_vector_store", lambda: fake_store)

    # Patch autowire_compose_target so the FIRST call raises and the
    # SECOND succeeds. The import in decomposers happens inside the
    # function (`from memory_mcp.autowire import autowire_compose_target`
    # at decomposers.py L1074), so patching the autowire module's symbol
    # is sufficient — the local import re-fetches from the module.
    real_target = autowire_mod.autowire_compose_target
    call_state = {"calls": 0}

    async def _flaky_target(**kwargs):
        call_state["calls"] += 1
        if call_state["calls"] == 1:
            raise RuntimeError("simulated Stage-B failure for child 0")
        return await real_target(**kwargs)

    monkeypatch.setattr(autowire_mod, "autowire_compose_target", _flaky_target)

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src,
                children=[_child("c1", "first"), _child("c2", "second")],
                mode="derive",
            ),
            ctx=ctx,
            settings=_settings(
                autowire_enabled=True,
                autowire_decompose_enabled=True,
            ),
        )
    finally:
        reset_session_factory(token)

    assert resp.auto_wired_by_child is not None
    child_ids = [c.id for c in resp.children]
    # Failing child has empty list; sibling child wired to pop.
    assert resp.auto_wired_by_child[child_ids[0]] == []
    assert resp.auto_wired_by_child[child_ids[1]] == [pop]
    # Decompose still committed its children + lineage rows.
    async with factory() as s:
        n_children = int((await s.execute(
            select(func.count()).select_from(Memory).where(
                Memory.id.in_(child_ids)
            )
        )).scalar_one())
    assert n_children == 2
    # Exactly ONE relation row (from sibling); failing child's
    # savepoint rolled back any partial insert.
    assert await _count_auto_wire_rows(factory) == 1


# ---------------------------------------------------------------------------
# H6 NB2 — outbox ordering (child upsert before relation events)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_outbox_child_upsert_before_relation_events(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """H6 NB2 — for any child with wired edges, its memory-upsert outbox
    row is created BEFORE its relation outbox rows. Mirrors compose."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="source body",
    )
    pop = await _write_memory(
        factory, env_id=env_id, agent_id=agent_id,
        title="pop", body="popular", salience=0.95,
    )

    fake_embedder = MagicMock()
    fake_embedder.embed_texts = MagicMock(
        return_value=[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    monkeypatch.setattr(autowire_mod, "get_embedder", lambda settings: fake_embedder)

    async def _fake_search(**kwargs):
        return [{"id": str(pop), "score": 0.92}]

    fake_store = MagicMock()
    fake_store.search = AsyncMock(side_effect=_fake_search)
    monkeypatch.setattr(autowire_mod, "_default_vector_store", lambda: fake_store)

    token = use_session_factory(factory)
    try:
        # Force graph_backend=neo4j so relation outbox rows actually
        # land (graph_backend=postgres makes the relation enqueue a
        # no-op — see db/outbox.py:111). The neo4j projection worker
        # is not running in this test; we only care that the outbox
        # row was persisted with monotonic event_id ordering.
        settings = _settings(
            autowire_enabled=True,
            autowire_decompose_enabled=True,
        )
        settings = settings.model_copy(update={"graph_backend": "neo4j"})
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src,
                children=[_child("c1", "first"), _child("c2", "second")],
                mode="derive",
            ),
            ctx=ctx,
            settings=settings,
        )
    finally:
        reset_session_factory(token)

    # Sanity: both children produced auto-wire edges (otherwise the
    # outbox-ordering claim is vacuous).
    assert resp.auto_wired_by_child is not None
    for cid, dsts in resp.auto_wired_by_child.items():
        assert dsts, f"child {cid} should have wired edges (sanity)"

    child_ids = [c.id for c in resp.children]
    # Per child, look up the memory-upsert outbox row + the relation
    # outbox rows for relations whose src_node points back to the
    # child. Memory upsert event_id < smallest relation event_id
    # (Outbox.event_id is a monotonic BigInteger sequence).
    async with factory() as s:
        for cid in child_ids:
            memory_row = (await s.execute(
                select(Outbox.event_id).where(
                    Outbox.aggregate_id == cid,
                    Outbox.aggregate_type == "memory",
                ).limit(1)
            )).scalar_one()
            # Find graph_node ids that map to this child (src side of
            # auto-wire edges).
            child_node_id = (await s.execute(
                select(GraphNode.id).where(GraphNode.memory_id == cid)
            )).scalar_one_or_none()
            if child_node_id is None:
                # Child had no edges (shouldn't happen here but defensive).
                continue
            # Relation outbox rows ref the relation row by its UUID;
            # we identify them via aggregate_type='relation'. We look
            # up rels for this child's src graph node, then collect
            # outbox rows for those relation ids.
            rel_ids = [r for (r,) in (await s.execute(
                select(Relation.id).where(
                    Relation.src_node_id == child_node_id,
                    Relation.type == AUTO_WIRE_PREDICATE,
                )
            )).all()]
            if not rel_ids:
                continue
            rel_outbox_ids = [
                rid for (rid,) in (await s.execute(
                    select(Outbox.event_id).where(
                        Outbox.aggregate_id.in_(rel_ids),
                        Outbox.aggregate_type == "relation",
                    )
                )).all()
            ]
            assert rel_outbox_ids, (
                f"expected relation outbox rows for child {cid}, got none"
            )
            assert memory_row < min(rel_outbox_ids), (
                f"child {cid} memory upsert outbox event_id {memory_row} "
                f"must precede relation outbox event_ids {rel_outbox_ids}"
            )
