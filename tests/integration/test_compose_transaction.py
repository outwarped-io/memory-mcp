"""Real-Postgres smoke for ``mem_compose`` transaction body (Phase 2 B3f).

Five baseline cases lock the happy paths; three additional rubber-duck
cases cover the recovery / rejection paths that the unit suite can't
exercise without a live DB:

Baseline (per B3 plan):

* ``smoke_promote_two_sources`` — sources stay active, lineage edges
  use ``promoted_from``, dedupe key persisted.
* ``smoke_merge_two_sources`` — sources transition to ``superseded``,
  lineage edges use ``supersedes``, ``superseded_by`` set on each
  source.
* ``smoke_replay_returns_same_id`` — second identical call returns
  ``idempotency_replay=true`` and the same memory id; no extra mutation.
* ``smoke_replay_after_supersede`` — replay still succeeds after
  sources have been superseded by the original call (dedupe-before-
  state-validation ordering, RD #1).
* ``smoke_caller_idempotency_key`` — caller-supplied key overrides the
  derived hash; subsequent call without it produces a distinct memory.

Rubber-duck additions (B3 mid-flight critique):

* ``smoke_invalid_decision_meta_rejected`` — ``decision_meta`` on a
  non-decision target rejected before any row insert.
* ``smoke_playbook_target_rejected`` — narrow ``MemComposeTarget`` has
  no ``steps`` / ``macro`` fields; surface that as clean rejection.
* ``smoke_replay_mode_mismatch_rejected`` — caller reuses
  ``idempotency_key`` with a different ``mode`` → InvalidInputError
  rather than a misleading replay echo.

The full 29-row matrix lives at B8.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from memory_mcp import composers as composers_mod
from memory_mcp import memories as memories_mod
from memory_mcp.config import Settings
from memory_mcp.db.models import Agent, AuditLog, Environment, Memory, MemoryLineage
from memory_mcp.db.types import LineageRelation, MemoryKind, MemoryStatus
from memory_mcp.errors import InvalidInputError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryWriteRequest, memory_write
from memory_mcp_schemas.compose import MemComposeRequest, MemComposeTarget

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


def _settings() -> Settings:
    return Settings(graph_backend="postgres")


async def _setup_env_and_agent(factory) -> tuple[UUID, UUID]:
    async with factory() as session:
        env = Environment(
            name=f"compose-smoke-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"compose-smoke-agent-{uuid4()}")
        session.add_all([env, agent])
        await session.commit()
        return env.id, agent.id


async def _write_source(
    factory,
    *,
    env_id: UUID,
    agent_id: UUID,
    title: str,
    body: str,
    kind: MemoryKind = MemoryKind.fact,
    tags: list[str] | None = None,
) -> UUID:
    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        resp = await memory_write(
            MemoryWriteRequest(
                kind=kind,
                title=title,
                body=body,
                env_id=env_id,
                tags=tags or [],
            ),
            ctx=ctx,
            settings=_settings(),
        )
        return resp.id
    finally:
        reset_session_factory(token)


async def _fetch_memory(factory, memory_id: UUID) -> Memory:
    async with factory() as session:
        return (await session.execute(
            select(Memory).where(Memory.id == memory_id)
        )).scalar_one()


async def _fetch_lineage(factory, child_id: UUID) -> list[tuple[UUID, str]]:
    async with factory() as session:
        rows = (await session.execute(
            select(MemoryLineage.parent_memory_id, MemoryLineage.relation)
            .where(MemoryLineage.child_memory_id == child_id)
        )).all()
        return [(r[0], r[1]) for r in rows]


async def _count_audits(factory, memory_id: UUID, op_like: str | None = None) -> int:
    async with factory() as session:
        from sqlalchemy import func
        stmt = select(func.count()).select_from(AuditLog).where(
            AuditLog.record_id == memory_id
        )
        if op_like is not None:
            stmt = stmt.where(AuditLog.op.like(op_like))
        return int((await session.execute(stmt)).scalar_one())


# ---------------------------------------------------------------------------
# Baseline smoke cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_promote_two_sources(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Promote mode: sources stay active, lineage uses promoted_from."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="src1", body="first source")
    src2 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="src2", body="second source")

    token = use_session_factory(factory)
    try:
        resp = await composers_mod.memory_compose(
            MemComposeRequest(
                source_ids=[src1, src2],
                target=MemComposeTarget(
                    kind=MemoryKind.fact,
                    title="summary",
                    body="combined summary",
                ),
                mode="promote",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert resp.mode == "promote"
    assert resp.idempotency_replay is False
    assert resp.tag_policy_applied == "target"
    assert resp.dedupe_key
    assert sorted(resp.source_ids) == sorted([src1, src2])
    assert resp.retired_source_ids == []
    assert {r.relation for r in resp.lineage_rows} == {"promoted_from"}

    merged = await _fetch_memory(factory, resp.memory.id)
    assert merged.status == MemoryStatus.active.value
    assert merged.compose_dedupe_key == resp.dedupe_key

    s1 = await _fetch_memory(factory, src1)
    s2 = await _fetch_memory(factory, src2)
    assert s1.status == MemoryStatus.active.value
    assert s2.status == MemoryStatus.active.value
    assert s1.superseded_by is None
    assert s2.superseded_by is None

    lineage = await _fetch_lineage(factory, resp.memory.id)
    assert sorted(p for p, _ in lineage) == sorted([src1, src2])
    assert {rel for _, rel in lineage} == {LineageRelation.promoted_from.value}


@pytest.mark.asyncio
async def test_smoke_merge_two_sources(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Merge mode: sources superseded, lineage uses supersedes, two audit rows for merged."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="src1", body="first")
    src2 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="src2", body="second")

    token = use_session_factory(factory)
    try:
        resp = await composers_mod.memory_compose(
            MemComposeRequest(
                source_ids=[src1, src2],
                target=MemComposeTarget(
                    kind=MemoryKind.fact,
                    title="combined",
                    body="merged body",
                ),
                mode="merge",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert resp.mode == "merge"
    assert resp.tag_policy_applied == "target_plus_union"
    assert sorted(resp.retired_source_ids) == sorted([src1, src2])
    assert {r.relation for r in resp.lineage_rows} == {"supersedes"}

    merged = await _fetch_memory(factory, resp.memory.id)
    assert merged.status == MemoryStatus.active.value

    for sid in (src1, src2):
        src = await _fetch_memory(factory, sid)
        assert src.status == MemoryStatus.superseded.value
        assert src.superseded_by == resp.memory.id

    # Two audit rows on merged: op=create + op=mem_compose:merge.
    n_create = await _count_audits(factory, resp.memory.id, op_like="create")
    n_aggregate = await _count_audits(factory, resp.memory.id, op_like="mem_compose:%")
    assert n_create == 1
    assert n_aggregate == 1

    # Per-source op=supersede audit rows.
    for sid in (src1, src2):
        n_super = await _count_audits(factory, sid, op_like="supersede")
        assert n_super == 1


@pytest.mark.asyncio
async def test_smoke_replay_returns_same_id(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Second identical call returns idempotency_replay=true and same id."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="src1", body="first")
    src2 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="src2", body="second")

    req = MemComposeRequest(
        source_ids=[src1, src2],
        target=MemComposeTarget(kind=MemoryKind.fact, title="dup", body="dup body"),
        mode="promote",
    )

    token = use_session_factory(factory)
    try:
        first = await composers_mod.memory_compose(req, ctx=ctx, settings=_settings())
        second = await composers_mod.memory_compose(req, ctx=ctx, settings=_settings())
    finally:
        reset_session_factory(token)

    assert first.idempotency_replay is False
    assert second.idempotency_replay is True
    assert second.memory.id == first.memory.id
    assert second.dedupe_key == first.dedupe_key
    assert second.mode == first.mode
    # Lineage was not re-inserted.
    lineage = await _fetch_lineage(factory, first.memory.id)
    assert len(lineage) == 2


@pytest.mark.asyncio
async def test_smoke_replay_after_supersede(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Replay still works once sources have been superseded by the original call."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s1", body="b1")
    src2 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s2", body="b2")

    req = MemComposeRequest(
        source_ids=[src1, src2],
        target=MemComposeTarget(kind=MemoryKind.fact, title="merged", body="merged body"),
        mode="merge",
    )

    token = use_session_factory(factory)
    try:
        first = await composers_mod.memory_compose(req, ctx=ctx, settings=_settings())
        # Sources are now superseded; a retry must still replay.
        second = await composers_mod.memory_compose(req, ctx=ctx, settings=_settings())
    finally:
        reset_session_factory(token)

    assert second.idempotency_replay is True
    assert second.memory.id == first.memory.id
    assert sorted(second.retired_source_ids) == sorted([src1, src2])


@pytest.mark.asyncio
async def test_smoke_caller_idempotency_key(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Caller-supplied idempotency_key overrides the derived hash."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s1", body="b1")
    src2 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s2", body="b2")

    target = MemComposeTarget(kind=MemoryKind.fact, title="t", body="b")

    token = use_session_factory(factory)
    try:
        with_key = await composers_mod.memory_compose(
            MemComposeRequest(
                source_ids=[src1, src2],
                target=target,
                mode="promote",
                idempotency_key="caller-key-001",
            ),
            ctx=ctx,
            settings=_settings(),
        )
        # Same content but no caller key → distinct dedupe key, distinct memory.
        without_key = await composers_mod.memory_compose(
            MemComposeRequest(
                source_ids=[src1, src2],
                target=target,
                mode="promote",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert with_key.dedupe_key == "caller-key-001"
    assert without_key.dedupe_key != "caller-key-001"
    assert with_key.memory.id != without_key.memory.id


# ---------------------------------------------------------------------------
# Rubber-duck-driven additions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_invalid_decision_meta_rejected(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``decision_meta`` only valid on kind=decision; non-decision target rejected."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s1", body="b1")
    src2 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s2", body="b2")

    token = use_session_factory(factory)
    try:
        with pytest.raises(InvalidInputError):
            await composers_mod.memory_compose(
                MemComposeRequest(
                    source_ids=[src1, src2],
                    target=MemComposeTarget(
                        kind=MemoryKind.fact,  # not decision
                        title="t",
                        body="b",
                        decision_meta={"context": "x", "decision": "y"},
                    ),
                    mode="promote",
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_smoke_playbook_target_rejected(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """kind=playbook needs steps/macro; MemComposeTarget lacks both — reject up front."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s1", body="b1")
    src2 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s2", body="b2")

    token = use_session_factory(factory)
    try:
        with pytest.raises(InvalidInputError, match="kind=playbook"):
            await composers_mod.memory_compose(
                MemComposeRequest(
                    source_ids=[src1, src2],
                    target=MemComposeTarget(
                        kind=MemoryKind.playbook,
                        title="bad",
                        body="b",
                    ),
                    mode="promote",
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_smoke_replay_mode_mismatch_rejected(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Replay with idempotency_key reused under a different mode is rejected."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src1 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s1", body="b1")
    src2 = await _write_source(factory, env_id=env_id, agent_id=agent_id, title="s2", body="b2")

    token = use_session_factory(factory)
    try:
        # First call: promote with caller key.
        await composers_mod.memory_compose(
            MemComposeRequest(
                source_ids=[src1, src2],
                target=MemComposeTarget(kind=MemoryKind.fact, title="t", body="b"),
                mode="promote",
                idempotency_key="reuse-key-001",
            ),
            ctx=ctx,
            settings=_settings(),
        )
        # Second call: same key but mode=merge → mismatch.
        with pytest.raises(InvalidInputError, match="mode disagrees"):
            await composers_mod.memory_compose(
                MemComposeRequest(
                    source_ids=[src1, src2],
                    target=MemComposeTarget(kind=MemoryKind.fact, title="t", body="b"),
                    mode="merge",
                    idempotency_key="reuse-key-001",
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)
