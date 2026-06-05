from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.browse import MemBrowseRequest
from memory_mcp_schemas.enums import MemoryKind
from memory_mcp_schemas.env_ops import EnvMigrateRequest, MigrationMode
from memory_mcp_schemas.memories import MemorySupersedeRequest, MemoryWriteRequest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memory_mcp import memories
from memory_mcp.db.models import Agent, Environment, Memory
from memory_mcp.env_ops import migrate as migrator
from memory_mcp.env_ops.migrate import migrate_env
from memory_mcp.errors import InvalidInputError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import memory_supersede, memory_write
from tests.env_ops.test_roundtrip import _MemoryVectorStore, _truncate, postgres_factory  # noqa: F401


@pytest.fixture
async def migrate_db(
    postgres_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncSession, _MemoryVectorStore, AgentContext]]:
    store = _MemoryVectorStore()
    ctx = AgentContext(agent_id=uuid4(), agent_name="migrate-agent")

    @asynccontextmanager
    async def routed_session_scope() -> AsyncIterator[AsyncSession]:
        async with postgres_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr(migrator, "session_scope", routed_session_scope)
    monkeypatch.setattr(memories, "session_scope", routed_session_scope)
    monkeypatch.setattr(memories, "_default_vector_store", lambda: store)

    async with postgres_factory() as session:
        await _truncate(session)
        session.add(Agent(id=ctx.agent_id, name="migrate-agent"))
        await session.commit()

    async with postgres_factory() as session:
        yield session, store, ctx

    async with postgres_factory() as session:
        await _truncate(session)


@pytest.mark.asyncio
async def test_migrate_copy_no_filter_full_env(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = migrate_db
    src_env, dst_env, source_ids = await _create_env_pair(session, count=3)

    out = await migrate_env(EnvMigrateRequest(src_env_id=src_env, dst_env_id=dst_env), ctx=ctx)

    assert out.attempted == 3
    assert out.succeeded == 3
    assert out.failed == 0
    assert len(out.remap) == 3
    assert set(out.remap) == set(source_ids)
    assert await _memory_count(session, dst_env) == 3
    assert await _memory_count(session, src_env) == 3


@pytest.mark.asyncio
async def test_migrate_move_supersedes_sources(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = migrate_db
    src_env, dst_env, source_ids = await _create_env_pair(session, count=3)

    out = await migrate_env(
        EnvMigrateRequest(src_env_id=src_env, dst_env_id=dst_env, mode=MigrationMode.move),
        ctx=ctx,
    )

    session.expire_all()
    rows = (await session.execute(select(Memory).where(Memory.id.in_(source_ids)))).scalars().all()
    assert {row.id for row in rows} == set(source_ids)
    for row in rows:
        assert row.status == "superseded"
        assert row.superseded_by == out.remap[row.id]


@pytest.mark.asyncio
async def test_migrate_with_filter_selects_subset(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = migrate_db
    src_env, dst_env, _source_ids = await _create_env_pair(
        session,
        kinds=["fact", "fact", "fact", "decision", "decision"],
    )

    out = await migrate_env(
        EnvMigrateRequest(
            src_env_id=src_env,
            dst_env_id=dst_env,
            filter=MemBrowseRequest(env_ids=[src_env], kinds=[MemoryKind.fact]),
        ),
        ctx=ctx,
    )

    assert out.attempted == 3
    assert out.succeeded == 3
    assert await _memory_count(session, dst_env) == 3
    assert await _kind_count(session, dst_env, "fact") == 3


@pytest.mark.asyncio
async def test_migrate_preserves_supersession_chain(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = migrate_db
    src_env, dst_env = await _create_empty_env_pair(session)
    m1, m2, m3 = await _create_chain_with_tools(src_env, ctx)

    out = await migrate_env(
        EnvMigrateRequest(
            src_env_id=src_env,
            dst_env_id=dst_env,
            filter=MemBrowseRequest(env_ids=[src_env], tags=["chain-tip"]),
        ),
        ctx=ctx,
    )

    assert out.attempted == 3
    assert out.succeeded == 3
    assert out.closure_inclusions == 2
    dst_m1 = await session.get(Memory, out.remap[m1])
    dst_m2 = await session.get(Memory, out.remap[m2])
    dst_m3 = await session.get(Memory, out.remap[m3])
    assert dst_m1 is not None and dst_m2 is not None and dst_m3 is not None
    assert dst_m1.superseded_by == dst_m2.id
    assert dst_m2.superseded_by == dst_m3.id


@pytest.mark.asyncio
async def test_migrate_fail_fast_aborts_on_first_error(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _store, ctx = migrate_db
    src_env, dst_env, source_ids = await _create_env_pair(session, count=3)
    failing = set(source_ids)
    real_mem_copy = migrator.mem_copy

    async def flaky_mem_copy(request, *, ctx):  # type: ignore[no-untyped-def]
        if request.memory_id in failing:
            raise InvalidInputError("forced first failure")
        return await real_mem_copy(request, ctx=ctx)

    monkeypatch.setattr(migrator, "mem_copy", flaky_mem_copy)

    with pytest.raises(InvalidInputError):
        await migrate_env(
            EnvMigrateRequest(src_env_id=src_env, dst_env_id=dst_env, fail_fast=True),
            ctx=ctx,
        )
    assert await _memory_count(session, dst_env) == 0


@pytest.mark.asyncio
async def test_migrate_continue_on_failure_default(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _store, ctx = migrate_db
    src_env, dst_env, source_ids = await _create_env_pair(session, count=3)
    failing = source_ids[0]
    real_mem_copy = migrator.mem_copy

    async def flaky_mem_copy(request, *, ctx):  # type: ignore[no-untyped-def]
        if request.memory_id == failing:
            raise InvalidInputError("forced failure")
        return await real_mem_copy(request, ctx=ctx)

    monkeypatch.setattr(migrator, "mem_copy", flaky_mem_copy)

    out = await migrate_env(EnvMigrateRequest(src_env_id=src_env, dst_env_id=dst_env), ctx=ctx)

    assert out.attempted == 3
    assert out.failed == 1
    assert out.succeeded == 2
    assert out.failures[0].memory_id == failing
    assert out.failures[0].code == "INVALID_INPUT"
    assert await _memory_count(session, dst_env) == 2


@pytest.mark.asyncio
async def test_migrate_rejects_same_env(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = migrate_db
    src_env, _dst_env, _source_ids = await _create_env_pair(session, count=1)

    with pytest.raises(InvalidInputError):
        await migrate_env(EnvMigrateRequest(src_env_id=src_env, dst_env_id=src_env), ctx=ctx)


@pytest.mark.asyncio
async def test_migrate_rejects_deleted_dst(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = migrate_db
    src_env, dst_env, _source_ids = await _create_env_pair(session, count=1, dst_status="deleted")

    with pytest.raises(NotFoundError) as exc:
        await migrate_env(EnvMigrateRequest(src_env_id=src_env, dst_env_id=dst_env), ctx=ctx)
    assert exc.value.code == "ENV_DELETED"


@pytest.mark.asyncio
async def test_migrate_remap_table_correct(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = migrate_db
    src_env, dst_env, source_ids = await _create_env_pair(session, bodies=["alpha", "beta", "gamma"])

    out = await migrate_env(EnvMigrateRequest(src_env_id=src_env, dst_env_id=dst_env), ctx=ctx)

    assert set(out.remap) == set(source_ids)
    for src_id, dst_id in out.remap.items():
        assert src_id != dst_id
        src = await session.get(Memory, src_id)
        dst = await session.get(Memory, dst_id)
        assert src is not None and dst is not None
        assert dst.env_id == dst_env
        assert dst.body == src.body


@pytest.mark.asyncio
async def test_migrate_copy_mode_leaves_source_active(
    migrate_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = migrate_db
    src_env, dst_env, source_ids = await _create_env_pair(session, count=3)

    await migrate_env(EnvMigrateRequest(src_env_id=src_env, dst_env_id=dst_env), ctx=ctx)

    rows = (await session.execute(select(Memory).where(Memory.id.in_(source_ids)))).scalars().all()
    assert {row.status for row in rows} == {"active"}
    assert all(row.superseded_by is None for row in rows)


async def _create_empty_env_pair(
    session: AsyncSession,
    *,
    src_model: str = "test-model",
    dst_model: str = "test-model",
    dst_status: str = "active",
) -> tuple[UUID, UUID]:
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
    await session.commit()
    return src_env, dst_env


async def _create_env_pair(
    session: AsyncSession,
    *,
    count: int | None = None,
    kinds: list[str] | None = None,
    bodies: list[str] | None = None,
    dst_status: str = "active",
) -> tuple[UUID, UUID, list[UUID]]:
    src_env, dst_env = await _create_empty_env_pair(session, dst_status=dst_status)
    if bodies is not None:
        items = [(kind, body) for kind, body in zip(kinds or ["fact"] * len(bodies), bodies, strict=True)]
    else:
        assert count is not None or kinds is not None
        values = kinds or ["fact"] * int(count)
        items = [(kind, f"body {idx}") for idx, kind in enumerate(values)]

    memory_ids: list[UUID] = []
    for idx, (kind, body) in enumerate(items):
        memory = Memory(
            id=uuid4(),
            env_id=src_env,
            kind=kind,
            status="active",
            title=f"memory {idx}",
            body=body,
            version=1,
        )
        session.add(memory)
        await session.flush()
        memory_ids.append(memory.id)
    await session.commit()
    return src_env, dst_env, memory_ids


async def _create_chain_with_tools(src_env: UUID, ctx: AgentContext) -> tuple[UUID, UUID, UUID]:
    m1 = await memory_write(
        MemoryWriteRequest(kind=MemoryKind.fact, title="m1", body="chain m1", env_id=src_env),
        ctx=ctx,
    )
    _old1, m2 = await memory_supersede(
        m1.id,
        MemorySupersedeRequest(
            expected_version=m1.version,
            new=MemoryWriteRequest(kind=MemoryKind.fact, title="m2", body="chain m2", env_id=src_env),
        ),
        ctx=ctx,
    )
    _old2, m3 = await memory_supersede(
        m2.id,
        MemorySupersedeRequest(
            expected_version=m2.version,
            new=MemoryWriteRequest(
                kind=MemoryKind.fact,
                title="m3",
                body="chain m3",
                env_id=src_env,
                tags=["chain-tip"],
            ),
        ),
        ctx=ctx,
    )
    return m1.id, m2.id, m3.id


async def _memory_count(session: AsyncSession, env_id: UUID) -> int:
    return int(await session.scalar(select(func.count()).select_from(Memory).where(Memory.env_id == env_id)) or 0)


async def _kind_count(session: AsyncSession, env_id: UUID, kind: str) -> int:
    return int(
        await session.scalar(
            select(func.count()).select_from(Memory).where(Memory.env_id == env_id, Memory.kind == kind),
        )
        or 0
    )
