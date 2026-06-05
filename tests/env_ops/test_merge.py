from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.env_ops import EnvMergeRequest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memory_mcp import entities
from memory_mcp.db.models import Agent, Entity, EntityAlias, Environment, Memory, MemoryLineage, MemoryTag, Tag
from memory_mcp.env_ops import merge as merger
from memory_mcp.env_ops.merge import ExternalRefsBlockingError, merge_envs
from memory_mcp.identity import AgentContext
from tests.env_ops.test_roundtrip import _MemoryVectorStore, _truncate, postgres_factory  # noqa: F401


@pytest.fixture
async def merge_db(
    postgres_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncSession, _MemoryVectorStore, AgentContext]]:
    store = _MemoryVectorStore()
    ctx = AgentContext(agent_id=uuid4(), agent_name="merge-agent")

    @asynccontextmanager
    async def routed_session_scope() -> AsyncIterator[AsyncSession]:
        async with postgres_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr(merger, "session_scope", routed_session_scope)
    monkeypatch.setattr(entities, "session_scope", routed_session_scope)
    monkeypatch.setattr(merger, "_merge_vector_store", lambda: store)

    async with postgres_factory() as session:
        await _truncate(session)
        session.add(Agent(id=ctx.agent_id, name="merge-agent"))
        await session.commit()

    async with postgres_factory() as session:
        yield session, store, ctx

    async with postgres_factory() as session:
        await _truncate(session)


@pytest.mark.asyncio
async def test_merge_basic_disjoint_envs(merge_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = merge_db
    src = await _create_env_with_memories(session, "src-basic", 3)
    dst = await _create_env_with_memories(session, "dst-basic", 3)
    src_ids = await _memory_ids(session, src)

    report = await merge_envs(EnvMergeRequest(src_env_id=src, dst_env_id=dst), ctx=ctx)

    assert report.counts["memories"] == 3
    assert await _count(session, Memory, dst) == 6
    assert src_ids.isdisjoint(await _memory_ids(session, dst))
    src_env = await session.scalar(select(Environment).where(Environment.id == src))
    assert src_env is not None
    assert src_env.status == "deleted"


@pytest.mark.asyncio
async def test_merge_with_tag_collision_unions(merge_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = merge_db
    src = await _create_env(session, "src-tags")
    dst = await _create_env(session, "dst-tags")
    src_memory = Memory(id=uuid4(), env_id=src, kind="fact", status="active", body="src tagged", version=1)
    src_tag = Tag(id=uuid4(), env_id=src, name="shared")
    dst_tag = Tag(id=uuid4(), env_id=dst, name="shared")
    session.add_all([src_memory, src_tag, dst_tag])
    await session.flush()
    session.add(MemoryTag(memory_id=src_memory.id, tag_id=src_tag.id, env_id=src))
    await session.commit()

    await merge_envs(EnvMergeRequest(src_env_id=src, dst_env_id=dst), ctx=ctx)

    tags = (await session.execute(select(Tag).where(Tag.env_id == dst, Tag.name == "shared"))).scalars().all()
    assert [tag.id for tag in tags] == [dst_tag.id]
    copied = await session.scalar(select(Memory).where(Memory.env_id == dst, Memory.body == "src tagged"))
    assert copied is not None
    link = await session.scalar(
        select(MemoryTag).where(MemoryTag.memory_id == copied.id, MemoryTag.tag_id == dst_tag.id)
    )
    assert link is not None


@pytest.mark.asyncio
async def test_merge_with_entity_canonical_key_collision_triggers_ent_merge(
    merge_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = merge_db
    src = await _create_env(session, "src-entities")
    dst = await _create_env(session, "dst-entities")
    src_entity = Entity(id=uuid4(), env_id=src, kind="service", canonical_name="X", normalized_name="x")
    dst_entity = Entity(id=uuid4(), env_id=dst, kind="service", canonical_name="X", normalized_name="x")
    session.add_all([src_entity, dst_entity])
    await session.flush()
    session.add(EntityAlias(entity_id=src_entity.id, env_id=src, alias="Source X", normalized_alias="source x"))
    await session.commit()

    report = await merge_envs(EnvMergeRequest(src_env_id=src, dst_env_id=dst), ctx=ctx)

    assert report.entity_merges_performed >= 1
    rows = (
        (await session.execute(select(Entity).where(Entity.env_id == dst, Entity.normalized_name == "x")))
        .scalars()
        .all()
    )
    assert [row.id for row in rows] == [dst_entity.id]
    alias = await session.scalar(
        select(EntityAlias).where(EntityAlias.env_id == dst, EntityAlias.normalized_alias == "source x")
    )
    assert alias is not None
    assert alias.entity_id == dst_entity.id


@pytest.mark.asyncio
async def test_merge_preserves_supersession_chain(
    merge_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = merge_db
    src = await _create_env(session, "src-chain")
    dst = await _create_env(session, "dst-chain")
    a = Memory(id=uuid4(), env_id=src, kind="fact", status="active", body="a", version=1)
    b = Memory(id=uuid4(), env_id=src, kind="fact", status="active", body="b", version=1)
    c = Memory(id=uuid4(), env_id=src, kind="fact", status="active", body="c", version=1)
    session.add_all([a, b, c])
    await session.flush()
    a.superseded_by = b.id
    a.status = "superseded"
    b.superseded_by = c.id
    b.status = "superseded"
    await session.commit()

    await merge_envs(EnvMergeRequest(src_env_id=src, dst_env_id=dst), ctx=ctx)

    copied = {
        row.body: row for row in (await session.execute(select(Memory).where(Memory.env_id == dst))).scalars().all()
    }
    assert copied["a"].superseded_by == copied["b"].id
    assert copied["b"].superseded_by == copied["c"].id


@pytest.mark.asyncio
async def test_merge_with_intra_lineage_only_no_external_edges_succeeds(
    merge_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = merge_db
    src, parent, child = await _create_lineage_src(session)
    dst = await _create_env(session, "dst-intra-lineage")

    await merge_envs(EnvMergeRequest(src_env_id=src, dst_env_id=dst), ctx=ctx)

    copied_parent = await session.scalar(select(Memory).where(Memory.env_id == dst, Memory.body == parent.body))
    copied_child = await session.scalar(select(Memory).where(Memory.env_id == dst, Memory.body == child.body))
    assert copied_parent is not None
    assert copied_child is not None
    edge = await session.scalar(
        select(MemoryLineage).where(
            MemoryLineage.parent_memory_id == copied_parent.id,
            MemoryLineage.child_memory_id == copied_child.id,
        )
    )
    assert edge is not None


@pytest.mark.asyncio
async def test_merge_with_external_lineage_default_blocks(
    merge_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = merge_db
    src, src_memory, _external_memory, dst = await _create_external_lineage(session)

    with pytest.raises(ExternalRefsBlockingError) as exc:
        await merge_envs(EnvMergeRequest(src_env_id=src, dst_env_id=dst, allow_external_ref_rewrite=False), ctx=ctx)

    assert exc.value.code == "EXTERNAL_REFS_BLOCKING"
    assert str(src_memory.id) in exc.value.details["sample_ids"][0]


@pytest.mark.asyncio
async def test_merge_with_external_lineage_rewritten_when_flag_set(
    merge_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = merge_db
    src, src_memory, external_memory, dst = await _create_external_lineage(session)

    report = await merge_envs(
        EnvMergeRequest(src_env_id=src, dst_env_id=dst, allow_external_ref_rewrite=True),
        ctx=ctx,
    )

    copied = await session.scalar(select(Memory).where(Memory.env_id == dst, Memory.body == src_memory.body))
    assert copied is not None
    edge = await session.scalar(
        select(MemoryLineage).where(
            MemoryLineage.parent_memory_id == external_memory.id,
            MemoryLineage.child_memory_id == copied.id,
        )
    )
    assert edge is not None
    assert report.external_refs_rewritten == 1


@pytest.mark.asyncio
async def test_merge_soft_deletes_src_after(merge_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = merge_db
    src = await _create_env_with_memories(session, "src-delete", 1)
    dst = await _create_env(session, "dst-delete")

    await merge_envs(EnvMergeRequest(src_env_id=src, dst_env_id=dst), ctx=ctx)

    env = await session.scalar(select(Environment).where(Environment.id == src))
    assert env is not None
    assert env.status == "deleted"
    assert env.deleted_at is not None


@pytest.mark.asyncio
async def test_merge_rejects_embedding_mismatch_without_flag(
    merge_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = merge_db
    src = await _create_env(session, "src-model", model="model-a")
    dst = await _create_env(session, "dst-model", model="model-b")

    with pytest.raises(Exception) as exc:
        await merge_envs(EnvMergeRequest(src_env_id=src, dst_env_id=dst), ctx=ctx)

    assert getattr(exc.value, "code", None) == "EMBEDDING_MODEL_MISMATCH"


async def _create_env(session: AsyncSession, name: str, *, model: str = "test-model") -> UUID:
    env_id = uuid4()
    session.add(
        Environment(
            id=env_id,
            name=f"{name}-{uuid4().hex}",
            retention_policy={},
            default_embedding_model_id=model,
        )
    )
    await session.commit()
    return env_id


async def _create_env_with_memories(session: AsyncSession, name: str, count: int) -> UUID:
    env_id = await _create_env(session, name)
    session.add_all(
        [
            Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body=f"{name}-{idx}", version=1)
            for idx in range(count)
        ]
    )
    await session.commit()
    return env_id


async def _create_lineage_src(session: AsyncSession) -> tuple[UUID, Memory, Memory]:
    src = await _create_env(session, "src-intra-lineage")
    parent = Memory(id=uuid4(), env_id=src, kind="fact", status="active", body="parent", version=1)
    child = Memory(id=uuid4(), env_id=src, kind="fact", status="active", body="child", version=1)
    session.add_all([parent, child])
    await session.flush()
    session.add(MemoryLineage(parent_memory_id=parent.id, child_memory_id=child.id, relation="copied_from"))
    await session.commit()
    return src, parent, child


async def _create_external_lineage(session: AsyncSession) -> tuple[UUID, Memory, Memory, UUID]:
    src = await _create_env(session, "src-external-lineage")
    dst = await _create_env(session, "dst-external-lineage")
    external = await _create_env(session, "external-lineage")
    src_memory = Memory(id=uuid4(), env_id=src, kind="fact", status="active", body="src external target", version=1)
    external_memory = Memory(
        id=uuid4(), env_id=external, kind="fact", status="active", body="external source", version=1
    )
    session.add_all([src_memory, external_memory])
    await session.flush()
    session.add(
        MemoryLineage(parent_memory_id=external_memory.id, child_memory_id=src_memory.id, relation="copied_from")
    )
    await session.commit()
    return src, src_memory, external_memory, dst


async def _count(session: AsyncSession, model: type, env_id: UUID) -> int:
    return int(await session.scalar(select(func.count()).select_from(model).where(model.env_id == env_id)) or 0)


async def _memory_ids(session: AsyncSession, env_id: UUID) -> set[UUID]:
    return set((await session.execute(select(Memory.id).where(Memory.env_id == env_id))).scalars().all())
