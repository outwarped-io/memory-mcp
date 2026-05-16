"""Integration coverage for ``memory_hard_delete`` saga (Phase 1.1, v0.11).

Verifies the Postgres-canonical leg of the saga end-to-end:

* canonical row is gone (subsequent ``mem_get`` raises ``NotFoundError``),
* a ``memory_tombstones`` row records audit fields,
* an ``OutboxOp.tombstone`` event is enqueued so the projection worker
  evicts Qdrant / Neo4j out-of-band,
* an ``audit_log`` row records ``hard_delete`` with ``before`` snapshot,
* refs guard rejects hard-delete when another row cites the target.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, update

from memory_mcp import memories as memories_mod
from memory_mcp.config import Settings
from memory_mcp.db.models import (
    Agent,
    AuditLog,
    Environment,
    Memory,
    MemoryLineage,
    MemoryTombstone,
    Outbox,
)
from memory_mcp.db.types import MemoryKind, OutboxOp
from memory_mcp.errors import (
    InvalidInputError,
    NotFoundError,
    VersionConflictError,
)
from memory_mcp.identity import AgentContext
from memory_mcp.memories import (
    MemoryHardDeleteRequest,
    MemoryWriteRequest,
    memory_get,
    memory_hard_delete,
    memory_write,
)

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


def _settings() -> Settings:
    return Settings(graph_backend="postgres")


async def _create_env_and_agent(factory, *, scenario: str) -> tuple[UUID, UUID]:
    async with factory() as session:
        env = Environment(
            name=f"hard-delete-{scenario}-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"hard-delete-agent-{uuid4()}")
        session.add_all([env, agent])
        await session.commit()
        return env.id, agent.id


async def _write_memory(
    factory, *, env_id: UUID, agent_id: UUID, title: str = "to-be-deleted"
) -> UUID:
    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        resp = await memory_write(
            MemoryWriteRequest(
                kind=MemoryKind.fact,
                title=title,
                body="ephemeral body that will be hard-deleted",
                env_id=env_id,
            ),
            ctx=ctx,
            settings=_settings(),
        )
        return resp.id
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_hard_delete_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """create → hard-delete → assert canonical gone, tombstone + outbox + audit present."""

    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    env_id, agent_id = await _create_env_and_agent(factory, scenario="happy")
    memory_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id)

    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        resp = await memory_hard_delete(
            memory_id,
            MemoryHardDeleteRequest(
                expected_version=1,
                reason="integration test cleanup",
                confirm_destroy=True,
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert resp.deleted_id == memory_id
    assert resp.canonical_deleted is True
    assert resp.tombstone_id is not None
    assert resp.projection_eviction.qdrant == "pending"
    assert resp.projection_eviction.neo4j == "pending"

    async with factory() as session:
        row = await session.scalar(select(Memory).where(Memory.id == memory_id))
        assert row is None, "canonical Memory row should be gone"

        tombstone = await session.scalar(
            select(MemoryTombstone).where(MemoryTombstone.id == resp.tombstone_id)
        )
        assert tombstone is not None
        assert tombstone.env_id == env_id
        assert tombstone.deleted_by_agent_id == agent_id
        assert tombstone.reason == "integration test cleanup"
        assert tombstone.original_kind == MemoryKind.fact.value

        outbox_rows = (
            await session.execute(
                select(Outbox).where(Outbox.aggregate_id == memory_id)
            )
        ).scalars().all()
        ops = [r.op for r in outbox_rows]
        assert OutboxOp.tombstone.value in ops, f"expected tombstone outbox event, got {ops}"

        audit_rows = (
            await session.execute(
                select(AuditLog).where(AuditLog.record_id == memory_id)
            )
        ).scalars().all()
        audit_ops = [r.op for r in audit_rows]
        assert "hard_delete" in audit_ops, f"expected hard_delete audit, got {audit_ops}"

    # Subsequent mem_get must surface NotFoundError.
    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        with pytest.raises(NotFoundError):
            await memory_get(memory_id, ctx=ctx)
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_hard_delete_requires_confirm_destroy(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Refuse to hard-delete unless ``confirm_destroy=True`` is set."""

    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    env_id, agent_id = await _create_env_and_agent(factory, scenario="confirm")
    memory_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id)

    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        with pytest.raises(InvalidInputError, match="confirm_destroy"):
            await memory_hard_delete(
                memory_id,
                MemoryHardDeleteRequest(
                    expected_version=1,
                    reason="should be refused",
                    confirm_destroy=False,
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)

    async with factory() as session:
        row = await session.scalar(select(Memory).where(Memory.id == memory_id))
        assert row is not None, "memory should still exist after refused hard-delete"


@pytest.mark.asyncio
async def test_hard_delete_rejects_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Stale ``expected_version`` ⇒ ``VersionConflictError``; row untouched."""

    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    env_id, agent_id = await _create_env_and_agent(factory, scenario="version")
    memory_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id)

    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        with pytest.raises(VersionConflictError):
            await memory_hard_delete(
                memory_id,
                MemoryHardDeleteRequest(
                    expected_version=999,
                    reason="stale version",
                    confirm_destroy=True,
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)

    async with factory() as session:
        row = await session.scalar(select(Memory).where(Memory.id == memory_id))
        assert row is not None


@pytest.mark.asyncio
async def test_hard_delete_rejects_when_refs_exist(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """V1 simplification: refuse hard-delete when another row cites the target.

    A successor memory references the target via ``memory_lineage``. The
    refs guard must reject the call so the lineage isn't orphaned.
    Caller workaround: ``mem_retire`` the dependent first.
    """

    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    env_id, agent_id = await _create_env_and_agent(factory, scenario="refs")
    target_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="target")
    successor_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="successor")

    async with factory() as session:
        session.add(
            MemoryLineage(
                parent_memory_id=target_id,
                child_memory_id=successor_id,
                relation="supersedes",
            )
        )
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        with pytest.raises(Exception) as exc_info:
            await memory_hard_delete(
                target_id,
                MemoryHardDeleteRequest(
                    expected_version=1,
                    reason="should be refused",
                    confirm_destroy=True,
                ),
                ctx=ctx,
                settings=_settings(),
            )
        # ``_ensure_memory_hard_delete_allowed`` raises ConflictError-shaped
        # errors. Accept any MemoryMCPError subclass with code CONFLICT.
        assert getattr(exc_info.value, "code", "") in {"CONFLICT", "REF_EXISTS", "MEMORY_HAS_REFS"} \
            or "ref" in str(exc_info.value).lower() \
            or "lineage" in str(exc_info.value).lower() \
            or "supers" in str(exc_info.value).lower(), \
            f"unexpected error: {exc_info.value!r}"
    finally:
        reset_session_factory(token)

    async with factory() as session:
        row = await session.scalar(select(Memory).where(Memory.id == target_id))
        assert row is not None, "target should still exist after refs-guarded refusal"


@pytest.mark.asyncio
async def test_hard_delete_cascade_real_postgres(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    env_id, agent_id = await _create_env_and_agent(factory, scenario="cascade")
    root_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="root")
    child_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="child")
    leaf_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="leaf")

    async with factory() as session:
        session.add_all([
            MemoryLineage(parent_memory_id=root_id, child_memory_id=child_id, relation="copied_from"),
            MemoryLineage(parent_memory_id=child_id, child_memory_id=leaf_id, relation="summarized_from"),
        ])
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        resp = await memory_hard_delete(
            root_id,
            MemoryHardDeleteRequest(
                expected_version=1,
                reason="cascade integration test",
                confirm_destroy=True,
                cascade=True,
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert resp.canonical_deleted is True
    assert resp.cascade_root is not None
    assert [item.id for item in resp.affected] == [leaf_id, child_id, root_id]

    async with factory() as session:
        survivors = (
            await session.execute(select(Memory).where(Memory.id.in_([root_id, child_id, leaf_id])))
        ).scalars().all()
        assert survivors == []

        tombstones = (
            await session.execute(
                select(MemoryTombstone).where(MemoryTombstone.cascade_root == resp.cascade_root)
            )
        ).scalars().all()
        assert len(tombstones) == 3
        assert {row.cascade_root for row in tombstones} == {resp.cascade_root}

        outbox_rows = (
            await session.execute(
                select(Outbox).where(Outbox.aggregate_id.in_([root_id, child_id, leaf_id]))
            )
        ).scalars().all()
        tombstone_ops = [row for row in outbox_rows if row.op == OutboxOp.tombstone.value]
        assert len(tombstone_ops) == 3


@pytest.mark.asyncio
async def test_hard_delete_cascade_occ_conflict_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    env_id, agent_id = await _create_env_and_agent(factory, scenario="cascade-occ")
    root_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="root")
    child_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="child")

    async with factory() as session:
        session.add(MemoryLineage(parent_memory_id=root_id, child_memory_id=child_id, relation="copied_from"))
        await session.commit()

    original_collect = memories_mod._collect_hard_delete_affected

    async def bump_after_collect(session, root, *, request, ctx):
        affected = await original_collect(session, root, request=request, ctx=ctx)
        async with factory() as other_session:
            await other_session.execute(
                update(Memory)
                .where(Memory.id == child_id)
                .values(version=Memory.version + 1)
            )
            await other_session.commit()
        return affected

    monkeypatch.setattr(memories_mod, "_collect_hard_delete_affected", bump_after_collect)

    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        with pytest.raises(VersionConflictError) as exc:
            await memory_hard_delete(
                root_id,
                MemoryHardDeleteRequest(
                    expected_version=1,
                    reason="cascade occ",
                    confirm_destroy=True,
                    cascade=True,
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)

    assert exc.value.details["memory_id"] == str(child_id)

    async with factory() as session:
        survivors = (
            await session.execute(select(Memory).where(Memory.id.in_([root_id, child_id])))
        ).scalars().all()
        assert {row.id for row in survivors} == {root_id, child_id}
        tombstone_count = await session.scalar(
            select(func.count()).select_from(MemoryTombstone).where(MemoryTombstone.reason == "cascade occ")
        )
        assert tombstone_count == 0


@pytest.mark.asyncio
async def test_hard_delete_cascade_dry_run_leaves_database_identical(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    env_id, agent_id = await _create_env_and_agent(factory, scenario="cascade-dry-run")
    root_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="root")
    child_id = await _write_memory(factory, env_id=env_id, agent_id=agent_id, title="child")

    async with factory() as session:
        session.add(MemoryLineage(parent_memory_id=root_id, child_memory_id=child_id, relation="copied_from"))
        await session.commit()
        before = {
            "memories": await _count_rows(session, Memory),
            "tombstones": await _count_rows(session, MemoryTombstone),
            "outbox": await _count_rows(session, Outbox),
        }

    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        resp = await memory_hard_delete(
            root_id,
            MemoryHardDeleteRequest(
                expected_version=1,
                reason="cascade dry run",
                confirm_destroy=True,
                cascade=True,
                dry_run=True,
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert resp.canonical_deleted is False
    assert resp.cascade_root is not None

    async with factory() as session:
        after = {
            "memories": await _count_rows(session, Memory),
            "tombstones": await _count_rows(session, MemoryTombstone),
            "outbox": await _count_rows(session, Outbox),
        }
    assert after == before


async def _count_rows(session, model) -> int:
    value = await session.scalar(select(func.count()).select_from(model))
    return int(value or 0)
