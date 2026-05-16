from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memory_mcp import envs
from memory_mcp.db.models import (
    Agent,
    Entity,
    EnvGrant,
    Environment,
    GraphNode,
    Memory,
    MemoryLineage,
    MemoryTag,
    Relation,
    Tag,
)
from memory_mcp.env_ops import clone as cloner
from memory_mcp.env_ops.clone import clone_env
from memory_mcp.errors import NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp_schemas.browse import MemBrowseRequest
from memory_mcp_schemas.env_ops import EnvCloneRequest
from memory_mcp_schemas.enums import MemoryStatus

from tests.env_ops.test_roundtrip import _MemoryVectorStore, _truncate, postgres_factory


@pytest.fixture
async def clone_db(
    postgres_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncSession, _MemoryVectorStore, AgentContext]]:
    store = _MemoryVectorStore()
    ctx = AgentContext(agent_id=uuid4(), agent_name="clone-agent")

    @asynccontextmanager
    async def routed_session_scope() -> AsyncIterator[AsyncSession]:
        async with postgres_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr(cloner, "session_scope", routed_session_scope)
    monkeypatch.setattr(envs, "session_scope", routed_session_scope)
    monkeypatch.setattr(cloner, "_default_vector_store", lambda: store)

    async with postgres_factory() as session:
        await _truncate(session)
        session.add(Agent(id=ctx.agent_id, name="clone-agent"))
        await session.commit()

    async with postgres_factory() as session:
        yield session, store, ctx

    async with postgres_factory() as session:
        await _truncate(session)


@pytest.mark.asyncio
async def test_clone_full_env_no_filter(clone_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = clone_db
    src_env_id = await _create_full_env(session)

    report = await clone_env(
        EnvCloneRequest(src_env_id=src_env_id, new_name=f"clone-full-{uuid4().hex}", include_embeddings=False),
        ctx=ctx,
    )

    dst = await session.scalar(select(Environment).where(Environment.id == report.dst_env_id))
    assert dst is not None
    assert dst.name == report.dst_env_name
    assert report.counts["memories"] == 3
    assert report.counts["tags"] == 2
    assert report.counts["entities"] == 2
    assert report.counts["relations"] == 1
    assert await _count(session, Memory, src_env_id) == await _count(session, Memory, report.dst_env_id)
    assert (await _ids(session, Memory, src_env_id)).isdisjoint(await _ids(session, Memory, report.dst_env_id))


@pytest.mark.asyncio
async def test_clone_preserves_supersession_chain(
    clone_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = clone_db
    src_env_id = await _create_chain_env(session)

    report = await clone_env(
        EnvCloneRequest(src_env_id=src_env_id, new_name=f"clone-chain-{uuid4().hex}", include_embeddings=False),
        ctx=ctx,
    )

    dst_by_body = {
        row.body: row
        for row in (await session.execute(select(Memory).where(Memory.env_id == report.dst_env_id))).scalars().all()
    }
    assert dst_by_body["A"].superseded_by == dst_by_body["B"].id
    assert dst_by_body["B"].superseded_by == dst_by_body["C"].id


@pytest.mark.asyncio
async def test_clone_filter_seeds_then_expands_closure(
    clone_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = clone_db
    src_env_id = await _create_filtered_env(session)

    report = await clone_env(
        EnvCloneRequest(
            src_env_id=src_env_id,
            new_name=f"clone-filter-{uuid4().hex}",
            include_embeddings=False,
            filter=MemBrowseRequest(
                env_ids=[src_env_id],
                tags=["seed"],
                statuses=[MemoryStatus.active, MemoryStatus.superseded],
            ),
        ),
        ctx=ctx,
    )

    bodies = await _bodies(session, report.dst_env_id)
    assert bodies == {"seed-1", "seed-2", "supersession-target", "lineage-parent"}
    assert report.closure_inclusions["memories"] == 2
    assert report.closure_inclusions["tags"] >= 1


@pytest.mark.asyncio
async def test_clone_with_lineage_depth_zero(
    clone_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = clone_db
    src_env_id = await _create_filtered_env(session)

    report = await clone_env(
        EnvCloneRequest(
            src_env_id=src_env_id,
            new_name=f"clone-depth0-{uuid4().hex}",
            include_embeddings=False,
            lineage_depth=0,
            filter=MemBrowseRequest(
                env_ids=[src_env_id],
                tags=["seed"],
                statuses=[MemoryStatus.active, MemoryStatus.superseded],
            ),
        ),
        ctx=ctx,
    )

    assert "lineage-parent" not in await _bodies(session, report.dst_env_id)


@pytest.mark.asyncio
async def test_clone_name_collision_raises(
    clone_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = clone_db
    src_env_id = await _create_full_env(session)
    existing = f"taken-{uuid4().hex}"
    session.add(Environment(id=uuid4(), name=existing.upper(), retention_policy={}, default_embedding_model_id="test-model"))
    await session.commit()

    with pytest.raises(cloner.ConflictError) as exc:
        await clone_env(
            EnvCloneRequest(src_env_id=src_env_id, new_name=existing.lower(), include_embeddings=False),
            ctx=ctx,
        )
    assert exc.value.code == "ENV_NAME_TAKEN"


@pytest.mark.asyncio
async def test_clone_rejects_deleted_src(clone_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = clone_db
    src_env_id = await _create_full_env(session)
    await session.execute(
        update(Environment).where(Environment.id == src_env_id).values(status="deleted", deleted_at=func.now())
    )
    await session.commit()

    with pytest.raises(NotFoundError) as exc:
        await clone_env(
            EnvCloneRequest(src_env_id=src_env_id, new_name=f"clone-deleted-{uuid4().hex}", include_embeddings=False),
            ctx=ctx,
        )
    assert exc.value.code == "ENV_DELETED"


@pytest.mark.asyncio
async def test_clone_excludes_grants(clone_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = clone_db
    src_env_id = await _create_full_env(session)
    session.add(EnvGrant(env_id=src_env_id, agent_id=ctx.agent_id, role="read"))
    await session.commit()

    report = await clone_env(
        EnvCloneRequest(src_env_id=src_env_id, new_name=f"clone-grants-{uuid4().hex}", include_embeddings=False),
        ctx=ctx,
    )

    grants = await session.scalar(select(func.count()).select_from(EnvGrant).where(EnvGrant.env_id == report.dst_env_id))
    assert grants == 0


@pytest.mark.asyncio
async def test_clone_dst_uuids_fresh(clone_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = clone_db
    src_env_id = await _create_full_env(session)

    report = await clone_env(
        EnvCloneRequest(src_env_id=src_env_id, new_name=f"clone-fresh-{uuid4().hex}", include_embeddings=False),
        ctx=ctx,
    )

    for model in (Memory, Tag, Entity, GraphNode, Relation):
        assert (await _ids(session, model, src_env_id)).isdisjoint(await _ids(session, model, report.dst_env_id))


async def _create_full_env(session: AsyncSession) -> UUID:
    env_id = uuid4()
    session.add(Environment(id=env_id, name=f"src-{uuid4().hex}", retention_policy={}, default_embedding_model_id="test-model"))
    await session.flush()
    memories = [
        Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body=f"memory-{idx}", version=1)
        for idx in range(3)
    ]
    tags = [Tag(id=uuid4(), env_id=env_id, name="tag-a"), Tag(id=uuid4(), env_id=env_id, name="tag-b")]
    entities = [
        Entity(id=uuid4(), env_id=env_id, kind="service", canonical_name="Service A", normalized_name="service a"),
        Entity(id=uuid4(), env_id=env_id, kind="service", canonical_name="Service B", normalized_name="service b"),
    ]
    session.add_all([*memories, *tags, *entities])
    await session.flush()
    session.add_all([
        MemoryTag(memory_id=memories[0].id, tag_id=tags[0].id, env_id=env_id),
        MemoryTag(memory_id=memories[1].id, tag_id=tags[1].id, env_id=env_id),
    ])
    nodes = [
        GraphNode(id=uuid4(), env_id=env_id, node_type="entity", entity_id=entities[0].id),
        GraphNode(id=uuid4(), env_id=env_id, node_type="entity", entity_id=entities[1].id),
    ]
    session.add_all(nodes)
    await session.flush()
    session.add(Relation(id=uuid4(), env_id=env_id, src_node_id=nodes[0].id, dst_node_id=nodes[1].id, type="related_to"))
    await session.commit()
    return env_id


async def _create_chain_env(session: AsyncSession) -> UUID:
    env_id = uuid4()
    session.add(Environment(id=env_id, name=f"chain-{uuid4().hex}", retention_policy={}, default_embedding_model_id="test-model"))
    await session.flush()
    a = Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="A", version=1)
    b = Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="B", version=1)
    c = Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="C", version=1)
    session.add_all([a, b, c])
    await session.flush()
    a.status = "superseded"
    a.superseded_by = b.id
    b.status = "superseded"
    b.superseded_by = c.id
    await session.commit()
    return env_id


async def _create_filtered_env(session: AsyncSession) -> UUID:
    env_id = uuid4()
    session.add(Environment(id=env_id, name=f"filtered-{uuid4().hex}", retention_policy={}, default_embedding_model_id="test-model"))
    await session.flush()
    memories = {
        "seed-1": Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="seed-1", version=1),
        "seed-2": Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="seed-2", version=1),
        "supersession-target": Memory(
            id=uuid4(), env_id=env_id, kind="fact", status="active", body="supersession-target", version=1
        ),
        "lineage-parent": Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="lineage-parent", version=1),
        "unrelated": Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="unrelated", version=1),
    }
    tag = Tag(id=uuid4(), env_id=env_id, name="seed")
    session.add_all([*memories.values(), tag])
    await session.flush()
    memories["seed-2"].status = "superseded"
    memories["seed-2"].superseded_by = memories["supersession-target"].id
    session.add_all([
        MemoryTag(memory_id=memories["seed-1"].id, tag_id=tag.id, env_id=env_id),
        MemoryTag(memory_id=memories["seed-2"].id, tag_id=tag.id, env_id=env_id),
        MemoryLineage(
            parent_memory_id=memories["lineage-parent"].id,
            child_memory_id=memories["seed-1"].id,
            relation="copied_from",
        ),
    ])
    await session.commit()
    return env_id


async def _count(session: AsyncSession, model: type, env_id: UUID) -> int:
    return int(await session.scalar(select(func.count()).select_from(model).where(model.env_id == env_id)) or 0)


async def _ids(session: AsyncSession, model: type, env_id: UUID) -> set[UUID]:
    return set((await session.execute(select(model.id).where(model.env_id == env_id))).scalars().all())


async def _bodies(session: AsyncSession, env_id: UUID) -> set[str]:
    return set((await session.execute(select(Memory.body).where(Memory.env_id == env_id))).scalars().all())
