from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memory_mcp import envs
from memory_mcp.db.models import (
    Agent,
    Entity,
    Environment,
    Memory,
    MemoryLineage,
    MemoryTag,
    Outbox,
    Tag,
    Task,
)
from memory_mcp.env_ops import delete as deleter
from memory_mcp.env_ops.delete import RefsBlockingDeleteError, delete_env
from memory_mcp.envs import env_get
from memory_mcp.errors import InvalidInputError
from memory_mcp.identity import AgentContext
from memory_mcp_schemas.env_ops import EnvDeleteRequest

from tests.env_ops.test_roundtrip import _truncate, postgres_factory


@pytest.fixture
async def delete_db(
    postgres_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncSession, AgentContext]]:
    ctx = AgentContext(agent_id=uuid4(), agent_name="delete-agent")

    @asynccontextmanager
    async def routed_session_scope() -> AsyncIterator[AsyncSession]:
        async with postgres_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr(deleter, "session_scope", routed_session_scope)
    monkeypatch.setattr(envs, "session_scope", routed_session_scope)

    async with postgres_factory() as session:
        await _truncate(session)
        session.add(Agent(id=ctx.agent_id, name="delete-agent"))
        await session.commit()

    async with postgres_factory() as session:
        yield session, ctx

    async with postgres_factory() as session:
        await _truncate(session)


@pytest.mark.asyncio
async def test_delete_requires_confirm() -> None:
    with pytest.raises(InvalidInputError) as exc:
        await delete_env(
            EnvDeleteRequest(env_id=uuid4(), confirm_destroy=False),
            ctx=AgentContext(agent_id=uuid4(), agent_name="delete-agent"),
        )

    assert exc.value.code == "CONFIRM_DESTROY_REQUIRED"


@pytest.mark.asyncio
async def test_delete_purges_env_rows(delete_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = delete_db
    env_id = await _create_env_with_rows(session)

    out = await delete_env(EnvDeleteRequest(env_id=env_id, confirm_destroy=True), ctx=ctx)

    session.expire_all()
    assert out.counts["memories"] == 3
    assert out.counts["tags"] == 2
    assert out.counts["entities"] == 1
    assert out.counts["tasks"] == 1
    counts = await _env_row_counts(session, env_id)
    assert all(count == 0 for count in counts.values())


@pytest.mark.asyncio
async def test_delete_soft_deletes_environment_row(delete_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = delete_db
    env_id = await _create_env_with_rows(session)

    await delete_env(EnvDeleteRequest(env_id=env_id, confirm_destroy=True), ctx=ctx)

    session.expire_all()
    env = await session.scalar(select(Environment).where(Environment.id == env_id))
    assert env is not None
    assert env.status == "deleted"
    assert env.deleted_at is not None


@pytest.mark.asyncio
async def test_delete_blocks_on_external_refs_by_default(delete_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = delete_db
    env_a, memory_a, env_b, memory_b = await _create_external_lineage(session)

    with pytest.raises(RefsBlockingDeleteError) as exc:
        await delete_env(EnvDeleteRequest(env_id=env_a, confirm_destroy=True), ctx=ctx)

    assert exc.value.code == "REFS_BLOCKING_DELETE"
    samples = exc.value.details["samples"]["external_lineage_entry"]
    assert str(memory_b) in samples[0]
    assert str(memory_a) in samples[0]
    assert await _count(session, select(func.count()).select_from(MemoryLineage)) == 1
    assert env_b != env_a


@pytest.mark.asyncio
async def test_delete_with_cascade_drops_external_refs(delete_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = delete_db
    env_a, _memory_a, _env_b, memory_b = await _create_external_lineage(session)

    out = await delete_env(
        EnvDeleteRequest(env_id=env_a, confirm_destroy=True, cascade_external_refs=True),
        ctx=ctx,
    )

    session.expire_all()
    assert out.external_lineage_entry_dropped == 1
    edge = await session.scalar(select(MemoryLineage).where(MemoryLineage.parent_memory_id == memory_b))
    assert edge is None


@pytest.mark.asyncio
async def test_delete_is_idempotent_on_already_deleted(delete_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = delete_db
    env_id = await _create_env(session, "already-deleted")
    env = await session.get(Environment, env_id)
    assert env is not None
    env.status = "deleted"
    await session.commit()

    out = await delete_env(EnvDeleteRequest(env_id=env_id, confirm_destroy=True), ctx=ctx)

    assert out.env_id == env_id
    assert out.confirm_destroy is True
    assert all(count == 0 for count in out.counts.values())


@pytest.mark.asyncio
async def test_delete_emits_outbox_event(delete_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = delete_db
    env_id = await _create_env_with_rows(session)

    await delete_env(EnvDeleteRequest(env_id=env_id, confirm_destroy=True), ctx=ctx)

    session.expire_all()
    rows = (await session.execute(select(Outbox).where(Outbox.env_id == env_id))).scalars().all()
    assert len(rows) == 1
    assert rows[0].payload["event"] == "EnvDeleted"


@pytest.mark.asyncio
async def test_delete_preserves_uuid_for_external_lineage(delete_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = delete_db
    env_a, _memory_a, _env_c, _memory_c = await _create_external_lineage(session)

    await delete_env(
        EnvDeleteRequest(env_id=env_a, confirm_destroy=True, cascade_external_refs=True),
        ctx=ctx,
    )

    resolved = await env_get(env_id=env_a, ctx=ctx, include_deleted=True)
    assert resolved.id == env_a
    assert resolved.status == "deleted"
    assert resolved.deleted_at is not None


async def _create_env(session: AsyncSession, name: str) -> UUID:
    env_id = uuid4()
    session.add(
        Environment(
            id=env_id,
            name=f"{name}-{uuid4().hex[:8]}",
            kind="test",
            retention_policy={},
            default_embedding_model_id="test-model",
        )
    )
    await session.commit()
    return env_id


async def _create_env_with_rows(session: AsyncSession) -> UUID:
    env_id = await _create_env(session, "delete-env")
    memories = [
        Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body=f"memory {idx}", version=1)
        for idx in range(3)
    ]
    tags = [Tag(id=uuid4(), env_id=env_id, name="tag-a"), Tag(id=uuid4(), env_id=env_id, name="tag-b")]
    entity = Entity(
        id=uuid4(),
        env_id=env_id,
        kind="service",
        canonical_name="Delete Service",
        normalized_name="delete service",
        version=1,
    )
    task = Task(id=uuid4(), env_id=env_id, title="delete task", status="pending", priority=50, version=1)
    session.add_all([*memories, *tags, entity, task])
    await session.flush()
    session.add_all(
        [
            MemoryTag(memory_id=memories[0].id, tag_id=tags[0].id, env_id=env_id),
            MemoryTag(memory_id=memories[1].id, tag_id=tags[1].id, env_id=env_id),
        ]
    )
    await session.commit()
    return env_id


async def _create_external_lineage(session: AsyncSession) -> tuple[UUID, UUID, UUID, UUID]:
    env_a = await _create_env(session, "delete-a")
    env_b = await _create_env(session, "delete-b")
    memory_a = Memory(id=uuid4(), env_id=env_a, kind="fact", status="active", body="a", version=1)
    memory_b = Memory(id=uuid4(), env_id=env_b, kind="fact", status="active", body="b", version=1)
    session.add_all([memory_a, memory_b])
    await session.flush()
    session.add(MemoryLineage(parent_memory_id=memory_b.id, child_memory_id=memory_a.id, relation="copied_from"))
    await session.commit()
    return env_a, memory_a.id, env_b, memory_b.id


async def _env_row_counts(session: AsyncSession, env_id: UUID) -> dict[str, int]:
    memory_ids = select(Memory.id).where(Memory.env_id == env_id)
    return {
        "memories": await _count(session, select(func.count()).select_from(Memory).where(Memory.env_id == env_id)),
        "memory_tags": await _count(
            session,
            select(func.count()).select_from(MemoryTag).where(MemoryTag.env_id == env_id),
        ),
        "tags": await _count(session, select(func.count()).select_from(Tag).where(Tag.env_id == env_id)),
        "entities": await _count(session, select(func.count()).select_from(Entity).where(Entity.env_id == env_id)),
        "tasks": await _count(session, select(func.count()).select_from(Task).where(Task.env_id == env_id)),
        "memory_lineage": await _count(
            session,
            select(func.count()).select_from(MemoryLineage).where(
                (MemoryLineage.parent_memory_id.in_(memory_ids))
                | (MemoryLineage.child_memory_id.in_(memory_ids))
            ),
        ),
    }


async def _count(session: AsyncSession, stmt: object) -> int:
    return int((await session.scalar(stmt)) or 0)
