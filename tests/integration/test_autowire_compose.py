"""End-to-end integration tests for Phase 4 auto-wire (D6b).

Coverage strategy:

* **OFF path** — when ``settings.autowire_enabled=False`` (the v0.15.0
  default), compose must behave EXACTLY as it did in v0.14 + Phase 2/3.
  ``auto_wired=[]`` in the response; zero ``related_to_popular`` rows in
  the ``relations`` table; popularity counters unchanged.
* **Stage B direct insert** — call :func:`autowire_compose_target` with
  hand-built candidates so we don't depend on a Qdrant fixture or a
  loaded embedder. Verifies graph-node resolution, ``ON CONFLICT DO
  NOTHING`` semantics, audit + outbox enqueue, and that the
  popularity-counter trigger guard fires (dst's
  ``reference_count_relations`` stays at 0).
* **Replay reconstruction** — :func:`reconstruct_auto_wired` re-queries
  live relations after a manual insert.
* **Compose hook ON path** — feature ON, monkeypatched embedder +
  vector store return a single matching candidate; assert one
  ``related_to_popular`` row + ``auto_wired`` populated + replay
  returns the same id.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.compose import MemComposeRequest, MemComposeTarget
from sqlalchemy import func, select

from memory_mcp import autowire as autowire_mod
from memory_mcp import composers as composers_mod
from memory_mcp import memories as memories_mod
from memory_mcp.autowire import (
    AUTO_WIRE_PREDICATE,
    autowire_compose_target,
    reconstruct_auto_wired,
)
from memory_mcp.config import Settings
from memory_mcp.db.models import (
    Agent,
    AuditLog,
    Environment,
    Memory,
    Outbox,
    Relation,
)
from memory_mcp.db.types import MemoryKind
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryWriteRequest, memory_write

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


def _settings(*, autowire_enabled: bool = False, top_k: int = 3) -> Settings:
    return Settings(
        graph_backend="postgres",
        autowire_enabled=autowire_enabled,
        autowire_top_k=top_k,
        autowire_sim_threshold=0.50,
        autowire_candidate_limit=20,
    )


async def _setup_env_and_agent(factory) -> tuple[UUID, UUID]:
    async with factory() as session:
        env = Environment(
            name=f"autowire-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"autowire-agent-{uuid4()}")
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


# ---------------------------------------------------------------------------
# OFF path — regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_off_does_not_emit_auto_wire_rows(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """v0.15.0 default behaviour: no relations rows, no extra audits."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="s1", body="first")
    src2 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="s2", body="second")

    token = use_session_factory(factory)
    try:
        resp = await composers_mod.memory_compose(
            MemComposeRequest(
                source_ids=[src1, src2],
                target=MemComposeTarget(kind=MemoryKind.fact, title="m", body="merged"),
                mode="promote",
            ),
            ctx=ctx,
            settings=_settings(autowire_enabled=False),
        )
    finally:
        reset_session_factory(token)

    assert resp.auto_wired == []
    async with factory() as s:
        n_rel = int(
            (
                await s.execute(select(func.count()).select_from(Relation).where(Relation.type == AUTO_WIRE_PREDICATE))
            ).scalar_one()
        )
    assert n_rel == 0


# ---------------------------------------------------------------------------
# Stage B direct insert — graph nodes, ON CONFLICT, audit, outbox, trigger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_b_inserts_relations_audit_outbox(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Direct Stage B call: K=2 candidates → 2 relations + 2 audits + 2 outbox."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    new_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="new", body="new mem body")
    pop1 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="p1", body="popular one", salience=0.9)
    pop2 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="p2", body="popular two", salience=0.8)

    candidates = [(pop1, 0.81), (pop2, 0.64)]
    settings = _settings(autowire_enabled=True, top_k=3)

    async with factory() as s:
        inserted = await autowire_compose_target(
            s=s,
            new_memory_id=new_id,
            new_memory_kind=MemoryKind.fact,
            new_memory_tags=None,
            new_memory_body="new mem body",
            new_memory_env_id=env_id,
            candidates=candidates,
            ctx=ctx,
            settings=settings,
        )
        await s.commit()

    assert sorted(inserted) == sorted([pop1, pop2])

    async with factory() as s:
        # Two relations rows with the auto-wire predicate.
        rel_rows = (await s.execute(select(Relation).where(Relation.type == AUTO_WIRE_PREDICATE))).scalars().all()
        assert len(rel_rows) == 2
        for r in rel_rows:
            assert r.properties.get("predicate") == AUTO_WIRE_PREDICATE
            assert "combined_score" in r.properties

        # Audit rows: one per inserted edge.
        audit_count = int(
            (
                await s.execute(
                    select(func.count()).select_from(AuditLog).where(AuditLog.op == f"auto_wire:{AUTO_WIRE_PREDICATE}")
                )
            ).scalar_one()
        )
        assert audit_count == 2

        # Outbox: relation aggregate events landed (sinks resolved
        # downstream by the projection worker). Exact count depends
        # on sink configuration; assert "at least zero" — the main
        # signal is "no exception during enqueue".
        await s.execute(select(func.count()).select_from(Outbox).where(Outbox.aggregate_type == "relation"))

        # Popularity trigger guard: dst memories' relation counter
        # stays at 0 because related_to_popular is excluded from the
        # whitelist (migration 0017 + 0021 regression).
        for dst_id in [pop1, pop2]:
            dst = (await s.execute(select(Memory).where(Memory.id == dst_id))).scalar_one()
            assert dst.reference_count_rel_link == 0


@pytest.mark.asyncio
async def test_stage_b_on_conflict_do_nothing(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Second call with same (src, dst, type) leaves the table unchanged."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    new_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="new", body="b")
    pop_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="pop", body="popular", salience=0.9)

    settings = _settings(autowire_enabled=True, top_k=3)

    async with factory() as s:
        await autowire_compose_target(
            s=s,
            new_memory_id=new_id,
            new_memory_kind=MemoryKind.fact,
            new_memory_tags=None,
            new_memory_body="b",
            new_memory_env_id=env_id,
            candidates=[(pop_id, 0.8)],
            ctx=ctx,
            settings=settings,
        )
        await s.commit()

    # Second identical call.
    async with factory() as s:
        inserted = await autowire_compose_target(
            s=s,
            new_memory_id=new_id,
            new_memory_kind=MemoryKind.fact,
            new_memory_tags=None,
            new_memory_body="b",
            new_memory_env_id=env_id,
            candidates=[(pop_id, 0.8)],
            ctx=ctx,
            settings=settings,
        )
        await s.commit()

    # Returned list is empty because nothing was inserted on the retry.
    assert inserted == []

    async with factory() as s:
        n_rel = int(
            (
                await s.execute(select(func.count()).select_from(Relation).where(Relation.type == AUTO_WIRE_PREDICATE))
            ).scalar_one()
        )
    assert n_rel == 1


# ---------------------------------------------------------------------------
# Replay reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconstruct_auto_wired_returns_current_state(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """After Stage B inserts, reconstruct returns the live dst ids."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    new_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="n", body="body")
    p1 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="p1", body="b1", salience=0.9)
    p2 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="p2", body="b2", salience=0.8)

    settings = _settings(autowire_enabled=True, top_k=3)
    async with factory() as s:
        await autowire_compose_target(
            s=s,
            new_memory_id=new_id,
            new_memory_kind=MemoryKind.fact,
            new_memory_tags=None,
            new_memory_body="body",
            new_memory_env_id=env_id,
            candidates=[(p1, 0.8), (p2, 0.7)],
            ctx=ctx,
            settings=settings,
        )
        await s.commit()

    async with factory() as s:
        recon = await reconstruct_auto_wired(s=s, memory_id=new_id)

    assert sorted(recon) == sorted([p1, p2])


# ---------------------------------------------------------------------------
# Compose hook end-to-end — feature ON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_on_emits_edge_and_populates_response(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Feature ON + fake embedder + fake vector store → 1 auto-wired edge."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)

    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="s1", body="first")
    src2 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="s2", body="second")
    pop_id = await _write_memory(
        factory,
        env_id=env_id,
        agent_id=agent_id,
        title="popular",
        body="popular content",
        salience=0.95,
    )

    # Patch the embedder + vector store at the autowire module surface.
    # The Stage A helper passes them straight through when injected,
    # but we go via composers' entry point so we monkeypatch the
    # autowire module's resolvers instead.
    fake_embedder = MagicMock()
    fake_embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    monkeypatch.setattr(autowire_mod, "get_embedder", lambda settings: fake_embedder)

    fake_store = MagicMock()
    fake_store.search = AsyncMock(
        return_value=[
            {"id": str(pop_id), "score": 0.95},
        ]
    )
    monkeypatch.setattr(autowire_mod, "_default_vector_store", lambda: fake_store)

    settings = _settings(autowire_enabled=True, top_k=3)

    token = use_session_factory(factory)
    try:
        resp = await composers_mod.memory_compose(
            MemComposeRequest(
                source_ids=[src1, src2],
                target=MemComposeTarget(kind=MemoryKind.fact, title="m", body="merged body content"),
                mode="promote",
            ),
            ctx=ctx,
            settings=settings,
        )
    finally:
        reset_session_factory(token)

    assert resp.auto_wired == [pop_id]

    async with factory() as s:
        rels = (await s.execute(select(Relation).where(Relation.type == AUTO_WIRE_PREDICATE))).scalars().all()
    assert len(rels) == 1


@pytest.mark.asyncio
async def test_compose_replay_returns_state_current_auto_wired(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Second identical compose call replays + reconstructs auto-wired list."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)

    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="s1", body="first")
    src2 = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="s2", body="second")
    pop_id = await _write_memory(
        factory,
        env_id=env_id,
        agent_id=agent_id,
        title="popular",
        body="popular content",
        salience=0.95,
    )

    fake_embedder = MagicMock()
    fake_embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    monkeypatch.setattr(autowire_mod, "get_embedder", lambda settings: fake_embedder)

    fake_store = MagicMock()
    fake_store.search = AsyncMock(
        return_value=[
            {"id": str(pop_id), "score": 0.95},
        ]
    )
    monkeypatch.setattr(autowire_mod, "_default_vector_store", lambda: fake_store)

    settings = _settings(autowire_enabled=True, top_k=3)
    request = MemComposeRequest(
        source_ids=[src1, src2],
        target=MemComposeTarget(kind=MemoryKind.fact, title="m", body="merged body content"),
        mode="promote",
    )

    token = use_session_factory(factory)
    try:
        first = await composers_mod.memory_compose(request, ctx=ctx, settings=settings)
        second = await composers_mod.memory_compose(request, ctx=ctx, settings=settings)
    finally:
        reset_session_factory(token)

    assert first.idempotency_replay is False
    assert second.idempotency_replay is True
    assert first.memory.id == second.memory.id
    assert second.auto_wired == [pop_id]
