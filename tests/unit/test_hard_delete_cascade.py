from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memory_mcp import memories
from memory_mcp.config import Settings
from memory_mcp.db.models import Agent, Environment, Memory, MemoryLineage, MemoryTombstone
from memory_mcp.errors import BlastRadiusExceededError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryHardDeleteRequest, memory_hard_delete

from tests.env_ops.test_roundtrip import _truncate, postgres_factory


@pytest.fixture
async def hard_delete_db(
    postgres_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncSession, AgentContext, UUID]]:
    ctx = AgentContext(agent_id=uuid4(), agent_name="hard-delete-agent")

    @asynccontextmanager
    async def routed_session_scope() -> AsyncIterator[AsyncSession]:
        async with postgres_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr(memories, "session_scope", routed_session_scope)

    async with postgres_factory() as session:
        await _truncate_hard_delete_state(session)
        env_id = uuid4()
        session.add_all([
            Agent(id=ctx.agent_id, name="hard-delete-agent"),
            Environment(
                id=env_id,
                name=f"hard-delete-{uuid4().hex}",
                retention_policy={},
                default_embedding_model_id="test-embedding",
            ),
        ])
        await session.commit()

    async with postgres_factory() as session:
        yield session, ctx, env_id

    async with postgres_factory() as session:
        await _truncate_hard_delete_state(session)


@pytest.mark.asyncio
async def test_hard_delete_default_single_row_delete_works(
    hard_delete_db: tuple[AsyncSession, AgentContext, UUID],
) -> None:
    session, ctx, env_id = hard_delete_db
    memory_id = await _create_memory(session, env_id=env_id, title="single")

    response = await memory_hard_delete(
        memory_id,
        MemoryHardDeleteRequest(
            expected_version=1,
            reason="single row delete",
            confirm_destroy=True,
        ),
        ctx=ctx,
        settings=_settings(),
    )

    session.expire_all()
    assert response.deleted_id == memory_id
    assert response.canonical_deleted is True
    assert response.cascade_root is None
    assert response.affected == []
    assert await session.get(Memory, memory_id) is None
    assert await session.scalar(select(func.count()).select_from(MemoryTombstone)) == 1


@pytest.mark.asyncio
async def test_hard_delete_default_refs_guard_unchanged(
    hard_delete_db: tuple[AsyncSession, AgentContext, UUID],
) -> None:
    session, ctx, env_id = hard_delete_db
    root_id, child_id, _leaf_id = await _create_chain(session, env_id=env_id, depth=2)

    with pytest.raises(Exception) as exc:
        await memory_hard_delete(
            root_id,
            MemoryHardDeleteRequest(
                expected_version=1,
                reason="refs guard",
                confirm_destroy=True,
            ),
            ctx=ctx,
            settings=_settings(),
        )
    assert getattr(exc.value, "code", None) == "ME_REFERENCED_CANNOT_HARD_DELETE"

    session.expire_all()
    assert await session.get(Memory, root_id) is not None
    assert await session.get(Memory, child_id) is not None


@pytest.mark.asyncio
async def test_hard_delete_cascade_dry_run_reports_affected_without_mutation(
    hard_delete_db: tuple[AsyncSession, AgentContext, UUID],
) -> None:
    session, ctx, env_id = hard_delete_db
    root_id, child_id, leaf_id = await _create_chain(session, env_id=env_id, depth=2)

    response = await memory_hard_delete(
        root_id,
        MemoryHardDeleteRequest(
            expected_version=1,
            reason="dry run",
            confirm_destroy=True,
            cascade=True,
            dry_run=True,
        ),
        ctx=ctx,
        settings=_settings(),
    )

    session.expire_all()
    assert response.canonical_deleted is False
    assert response.deleted_at is None
    assert response.tombstone_id is None
    assert response.projection_eviction is None
    assert response.cascade_root is not None
    assert [item.id for item in response.affected] == [leaf_id, child_id, root_id]
    assert [item.depth for item in response.affected] == [2, 1, 0]
    assert await session.get(Memory, root_id) is not None
    assert await session.get(Memory, child_id) is not None
    assert await session.get(Memory, leaf_id) is not None
    assert await session.scalar(select(func.count()).select_from(MemoryTombstone)) == 0


@pytest.mark.asyncio
async def test_hard_delete_cascade_depth_cap_enforced(
    hard_delete_db: tuple[AsyncSession, AgentContext, UUID],
) -> None:
    session, ctx, env_id = hard_delete_db
    root_id, *_ = await _create_chain(session, env_id=env_id, depth=6)

    with pytest.raises(BlastRadiusExceededError) as exc:
        await memory_hard_delete(
            root_id,
            MemoryHardDeleteRequest(
                expected_version=1,
                reason="depth cap",
                confirm_destroy=True,
                cascade=True,
                dry_run=True,
                max_cascade_depth=5,
            ),
            ctx=ctx,
            settings=_settings(),
        )

    assert exc.value.cap_hit == "depth"
    assert exc.value.limit == 5
    assert exc.value.details["offending_depth"] == 6
    assert len(exc.value.details["affected"]) == 6


@pytest.mark.asyncio
async def test_hard_delete_cascade_count_cap_enforced(
    hard_delete_db: tuple[AsyncSession, AgentContext, UUID],
) -> None:
    session, ctx, env_id = hard_delete_db
    root_id = await _create_memory(session, env_id=env_id, title="root")
    children = [await _create_memory(session, env_id=env_id, title=f"child-{index}") for index in range(21)]
    session.add_all([
        MemoryLineage(parent_memory_id=root_id, child_memory_id=child_id, relation="copied_from")
        for child_id in children
    ])
    await session.commit()

    with pytest.raises(BlastRadiusExceededError) as exc:
        await memory_hard_delete(
            root_id,
            MemoryHardDeleteRequest(
                expected_version=1,
                reason="count cap",
                confirm_destroy=True,
                cascade=True,
                dry_run=True,
                max_cascade_count=20,
            ),
            ctx=ctx,
            settings=_settings(),
        )

    assert exc.value.cap_hit == "count"
    assert exc.value.limit == 20
    assert len(exc.value.details["affected"]) == 20


@pytest.mark.asyncio
async def test_hard_delete_cascade_orders_affected_leaves_first(
    hard_delete_db: tuple[AsyncSession, AgentContext, UUID],
) -> None:
    session, ctx, env_id = hard_delete_db
    root_id = await _create_memory(session, env_id=env_id, title="root")
    child_a = await _create_memory(session, env_id=env_id, title="child-a")
    child_b = await _create_memory(session, env_id=env_id, title="child-b")
    leaf = await _create_memory(session, env_id=env_id, title="leaf")
    session.add_all([
        MemoryLineage(parent_memory_id=root_id, child_memory_id=child_a, relation="copied_from"),
        MemoryLineage(parent_memory_id=root_id, child_memory_id=child_b, relation="copied_from"),
        MemoryLineage(parent_memory_id=child_a, child_memory_id=leaf, relation="summarized_from"),
    ])
    await session.commit()

    response = await memory_hard_delete(
        root_id,
        MemoryHardDeleteRequest(
            expected_version=1,
            reason="ordering",
            confirm_destroy=True,
            cascade=True,
            dry_run=True,
        ),
        ctx=ctx,
        settings=_settings(),
    )

    depths = [item.depth for item in response.affected]
    assert depths == sorted(depths, reverse=True)
    assert response.affected[-1].id == root_id


def test_hard_delete_request_validators() -> None:
    with pytest.raises(ValidationError):
        MemoryHardDeleteRequest(expected_version=1, reason="x", confirm_destroy=True, max_cascade_depth=0)
    with pytest.raises(ValidationError):
        MemoryHardDeleteRequest(expected_version=1, reason="x", confirm_destroy=True, max_cascade_depth=21)
    with pytest.raises(ValidationError):
        MemoryHardDeleteRequest(expected_version=1, reason="x", confirm_destroy=True, max_cascade_count=0)


async def _create_chain(
    session: AsyncSession,
    *,
    env_id: UUID,
    depth: int,
) -> tuple[UUID, ...]:
    ids = [await _create_memory(session, env_id=env_id, title=f"node-{index}") for index in range(depth + 1)]
    session.add_all([
        MemoryLineage(
            parent_memory_id=ids[index],
            child_memory_id=ids[index + 1],
            relation="copied_from",
        )
        for index in range(depth)
    ])
    await session.commit()
    return tuple(ids)


async def _create_memory(session: AsyncSession, *, env_id: UUID, title: str) -> UUID:
    memory = Memory(
        id=uuid4(),
        env_id=env_id,
        kind="fact",
        status="active",
        title=title,
        body=f"body-{title}",
        metadata_={},
    )
    session.add(memory)
    await session.commit()
    return memory.id


async def _truncate_hard_delete_state(session: AsyncSession) -> None:
    await _truncate(session)
    await session.execute(text("TRUNCATE memory_tombstones"))
    await session.commit()


def _settings() -> Settings:
    return Settings(graph_backend="postgres")
