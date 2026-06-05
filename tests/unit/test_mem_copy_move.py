from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.env_ops import MemCopyRequest, MemMoveRequest
from memory_mcp_schemas.search import MemorySearchRequest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memory_mcp import memories
from memory_mcp.db.models import Agent, Environment, Memory, MemoryLineage, MemorySource, MemoryTag, Tag
from memory_mcp.errors import InvalidInputError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import mem_copy, mem_move, memory_get
from memory_mcp.search import api as search_api
from memory_mcp.search import memory_search
from tests.env_ops.test_roundtrip import _MemoryVectorStore, _truncate, postgres_factory  # noqa: F401


@pytest.fixture
async def mem_copy_db(
    postgres_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncSession, _MemoryVectorStore, AgentContext]]:
    store = _MemoryVectorStore()
    ctx = AgentContext(agent_id=uuid4(), agent_name="mem-copy-agent")

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
    monkeypatch.setattr(search_api, "session_scope", routed_session_scope)
    monkeypatch.setattr(memories, "_default_vector_store", lambda: store)

    async with postgres_factory() as session:
        await _truncate(session)
        session.add(Agent(id=ctx.agent_id, name="mem-copy-agent"))
        await session.commit()

    async with postgres_factory() as session:
        yield session, store, ctx

    async with postgres_factory() as session:
        await _truncate(session)


@pytest.mark.asyncio
async def test_mem_copy_basic_creates_new_memory_in_dst(
    mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, store, ctx = mem_copy_db
    src_env, dst_env, source = await _create_memory_pair(session)
    await store.upsert(env_id=src_env, point_id=source, vector={"body": [0.1, 0.2]}, payload={})

    out = await mem_copy(MemCopyRequest(memory_id=source, dst_env_id=dst_env), ctx=ctx)

    copied = await session.get(Memory, out.dst_memory_id)
    original = await session.get(Memory, source)
    assert copied is not None
    assert original is not None
    assert copied.id != source
    assert copied.env_id == dst_env
    assert copied.body == original.body
    assert copied.kind == original.kind
    assert original.status == "active"
    assert await _lineage_exists(session, source, copied.id, "copied_from")


@pytest.mark.asyncio
async def test_mem_copy_preserves_tags_when_requested(
    mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = mem_copy_db
    _src_env, dst_env, source = await _create_memory_pair(session, tags=["alpha", "beta"])

    out = await mem_copy(MemCopyRequest(memory_id=source, dst_env_id=dst_env, copy_tags=True), ctx=ctx)

    tag_names = await _memory_tag_names(session, out.dst_memory_id)
    assert tag_names == ["alpha", "beta"]
    dst_tag_envs = (await session.execute(select(Tag.env_id).where(Tag.name.in_(tag_names)))).scalars().all()
    assert dst_env in dst_tag_envs


@pytest.mark.asyncio
async def test_mem_copy_skips_tags_when_disabled(
    mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = mem_copy_db
    _src_env, dst_env, source = await _create_memory_pair(session, tags=["alpha", "beta"])

    out = await mem_copy(MemCopyRequest(memory_id=source, dst_env_id=dst_env, copy_tags=False), ctx=ctx)

    count = await session.scalar(
        select(func.count()).select_from(MemoryTag).where(MemoryTag.memory_id == out.dst_memory_id)
    )
    assert count == 0


@pytest.mark.asyncio
async def test_mem_copy_rejects_same_env(mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = mem_copy_db
    src_env, _dst_env, source = await _create_memory_pair(session)

    with pytest.raises(InvalidInputError):
        await mem_copy(MemCopyRequest(memory_id=source, dst_env_id=src_env), ctx=ctx)


@pytest.mark.asyncio
async def test_mem_copy_rejects_deleted_dst(mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = mem_copy_db
    _src_env, dst_env, source = await _create_memory_pair(session, dst_status="deleted")

    with pytest.raises(NotFoundError) as exc:
        await mem_copy(MemCopyRequest(memory_id=source, dst_env_id=dst_env), ctx=ctx)
    assert exc.value.code == "ENV_DELETED"


@pytest.mark.asyncio
async def test_mem_copy_creates_cross_env_lineage_edge(
    mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = mem_copy_db
    _src_env, dst_env, source = await _create_memory_pair(session)

    out = await mem_copy(MemCopyRequest(memory_id=source, dst_env_id=dst_env), ctx=ctx)

    assert await _lineage_exists(session, source, out.dst_memory_id, "copied_from")


@pytest.mark.asyncio
async def test_mem_copy_rejects_embedding_mismatch_without_flag(
    mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = mem_copy_db
    _src_env, dst_env, source = await _create_memory_pair(session, src_model="model-a", dst_model="model-b")

    with pytest.raises(Exception) as exc:
        await mem_copy(MemCopyRequest(memory_id=source, dst_env_id=dst_env), ctx=ctx)
    assert getattr(exc.value, "code", None) == "EMBEDDING_MODEL_MISMATCH"


@pytest.mark.asyncio
async def test_mem_move_supersedes_source(mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext]) -> None:
    session, _store, ctx = mem_copy_db
    _src_env, dst_env, source = await _create_memory_pair(session)

    out = await mem_move(MemMoveRequest(memory_id=source, dst_env_id=dst_env), ctx=ctx)

    session.expire_all()
    original = await session.get(Memory, source)
    assert original is not None
    assert original.status == "superseded"
    assert original.superseded_by == out.dst_memory_id


@pytest.mark.asyncio
async def test_mem_move_preserves_source_uuid_as_tombstone(
    mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = mem_copy_db
    src_env, dst_env, source = await _create_memory_pair(session, body="tombstone source body")

    await mem_move(MemMoveRequest(memory_id=source, dst_env_id=dst_env), ctx=ctx)

    fetched = await memory_get(source, ctx=ctx)
    assert fetched.id == source
    assert fetched.status.value == "superseded"
    search = await memory_search(
        MemorySearchRequest(query="tombstone source body", env_ids=[src_env], mode="lex", consistency="canonical"),
        ctx=ctx,
    )
    assert source not in {hit.memory.id for hit in search.hits}


@pytest.mark.asyncio
async def test_mem_move_hard_delete_when_unreferenced(
    mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = mem_copy_db
    _src_env, dst_env, source = await _create_memory_pair(session)

    out = await mem_move(
        MemMoveRequest(memory_id=source, dst_env_id=dst_env, redirect_source=False, create_lineage_edge=False),
        ctx=ctx,
    )

    assert out.source_memory_status == "deleted"
    assert await session.get(Memory, source) is None


@pytest.mark.asyncio
async def test_mem_move_hard_delete_blocked_when_referenced(
    mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = mem_copy_db
    _src_env, dst_env, source = await _create_memory_pair(session)
    ref = Memory(env_id=dst_env, kind="fact", status="active", title="ref", body="ref")
    session.add(ref)
    await session.flush()
    session.add(MemoryLineage(parent_memory_id=source, child_memory_id=ref.id, relation="copied_from"))
    await session.commit()

    with pytest.raises(Exception) as exc:
        await mem_move(
            MemMoveRequest(memory_id=source, dst_env_id=dst_env, redirect_source=False, create_lineage_edge=False),
            ctx=ctx,
        )
    assert getattr(exc.value, "code", None) == "ME_REFERENCED_CANNOT_HARD_DELETE"


@pytest.mark.asyncio
async def test_mem_move_does_not_trigger_mem_supersede_guard(
    mem_copy_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = mem_copy_db
    _src_env, dst_env, source = await _create_memory_pair(session)

    out = await mem_move(MemMoveRequest(memory_id=source, dst_env_id=dst_env), ctx=ctx)

    session.expire_all()
    original = await session.get(Memory, source)
    assert original is not None
    assert original.superseded_by == out.dst_memory_id


async def _create_memory_pair(
    session: AsyncSession,
    *,
    body: str = "source body",
    tags: list[str] | None = None,
    src_model: str = "test-model",
    dst_model: str = "test-model",
    dst_status: str = "active",
) -> tuple[UUID, UUID, UUID]:
    src_env = uuid4()
    dst_env = uuid4()
    session.add_all(
        [
            Environment(
                id=src_env, name=f"src-{uuid4().hex}", retention_policy={}, default_embedding_model_id=src_model
            ),
            Environment(
                id=dst_env,
                name=f"dst-{uuid4().hex}",
                retention_policy={},
                default_embedding_model_id=dst_model,
                status=dst_status,
            ),
        ]
    )
    memory = Memory(
        id=uuid4(),
        env_id=src_env,
        kind="fact",
        status="active",
        title="source",
        body=body,
        metadata_={"payload": "same"},
        version=1,
    )
    session.add(memory)
    await session.flush()
    session.add(MemorySource(memory_id=memory.id, source_type="agent", source_ref="test", agent_id=None))
    if tags:
        for name in tags:
            tag = Tag(env_id=src_env, name=name)
            session.add(tag)
            await session.flush()
            session.add(MemoryTag(memory_id=memory.id, tag_id=tag.id, env_id=src_env))
    await session.commit()
    return src_env, dst_env, memory.id


async def _memory_tag_names(session: AsyncSession, memory_id: UUID) -> list[str]:
    rows = await session.execute(
        select(Tag.name)
        .join(MemoryTag, MemoryTag.tag_id == Tag.id)
        .where(MemoryTag.memory_id == memory_id)
        .order_by(Tag.name)
    )
    return list(rows.scalars().all())


async def _lineage_exists(session: AsyncSession, parent_id: UUID, child_id: UUID, relation: str) -> bool:
    row = await session.scalar(
        select(MemoryLineage.parent_memory_id).where(
            MemoryLineage.parent_memory_id == parent_id,
            MemoryLineage.child_memory_id == child_id,
            MemoryLineage.relation == relation,
        )
    )
    return row is not None
