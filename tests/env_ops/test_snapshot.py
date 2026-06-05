from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from memory_mcp_schemas.env_ops import EnvRestoreRequest, EnvSnapshotRequest, RestoreMode
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer

from memory_mcp import envs
from memory_mcp.db.models import Agent, Environment, Memory, MemoryLineage, Snapshot
from memory_mcp.env_ops import export as exporter
from memory_mcp.env_ops import import_ as importer
from memory_mcp.env_ops import snapshot as snapshot_ops
from memory_mcp.env_ops.snapshot import create_snapshot, restore_snapshot
from memory_mcp.identity import AgentContext

REPO_ROOT = Path(__file__).resolve().parents[2]


class _MemoryVectorStore:
    def __init__(self) -> None:
        self.vectors: dict[tuple[UUID, UUID, str], list[float]] = {}

    async def ensure_env_collection(self, *, env_id: UUID, dimension: int) -> None:
        return None

    async def upsert(
        self,
        *,
        env_id: UUID,
        point_id: UUID,
        vector: Sequence[float] | Mapping[str, Sequence[float]],
        payload: Mapping[str, Any],
    ) -> None:
        if isinstance(vector, Mapping):
            for name, values in vector.items():
                self.vectors[(env_id, point_id, str(name))] = list(values)
        else:
            self.vectors[(env_id, point_id, "body")] = list(vector)

    async def delete(self, *, env_id: UUID, point_id: UUID) -> None:
        return None

    async def search(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def get_vector(self, *, env_id: UUID, id: str, vector_name: str = "body") -> list[float] | None:
        return self.vectors.get((env_id, UUID(id), vector_name))

    async def get_vectors(
        self, *, env_id: UUID, ids: list[UUID], vector_name: str = "body"
    ) -> dict[UUID, list[float] | None]:
        return {memory_id: self.vectors.get((env_id, memory_id, vector_name)) for memory_id in ids}

    async def close(self) -> None:
        return None


@pytest.fixture(scope="session")
def postgres_factory() -> Iterator[async_sessionmaker[AsyncSession]]:
    container = PostgresContainer("postgres:16-alpine", username="memory", password="memory", dbname="memory")
    try:
        container.start()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres testcontainer unavailable; Docker is required for snapshot tests: {exc!r}")

    engine = None
    try:
        sync_url = container.get_connection_url(driver="psycopg2")
        async_url = container.get_connection_url(driver="asyncpg")
        config = Config(str(REPO_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
        config.set_main_option("sqlalchemy.url", sync_url)
        command.upgrade(config, "head")
        engine = create_async_engine(async_url, pool_pre_ping=True, poolclass=NullPool)
        yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    finally:
        if engine is not None:
            asyncio.run(engine.dispose())
        with suppress(Exception):
            container.stop()


@pytest.fixture
async def snapshot_db(
    postgres_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncSession, _MemoryVectorStore, AgentContext, Path]]:
    store = _MemoryVectorStore()
    ctx = AgentContext(agent_id=uuid4(), agent_name="snapshot-agent")
    data_root = Path(".tmp") / "snapshot-tests" / uuid4().hex
    data_root.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def routed_session_scope() -> AsyncIterator[AsyncSession]:
        async with postgres_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    settings = SimpleNamespace(data_root=data_root)
    monkeypatch.setattr(exporter, "session_scope", routed_session_scope)
    monkeypatch.setattr(importer, "session_scope", routed_session_scope)
    monkeypatch.setattr(snapshot_ops, "session_scope", routed_session_scope)
    monkeypatch.setattr(envs, "session_scope", routed_session_scope)
    monkeypatch.setattr(snapshot_ops, "get_settings", lambda: settings)
    monkeypatch.setattr(exporter, "_default_vector_store", lambda: store)
    monkeypatch.setattr(importer, "_default_vector_store", lambda: store)
    monkeypatch.setattr(snapshot_ops, "_default_vector_store", lambda: store)

    async with postgres_factory() as session:
        await _truncate(session)
        session.add(Agent(id=ctx.agent_id, name="snapshot-agent"))
        await session.commit()

    async with postgres_factory() as session:
        yield session, store, ctx, data_root

    async with postgres_factory() as session:
        await _truncate(session)
    shutil.rmtree(data_root, ignore_errors=True)


@pytest.mark.asyncio
async def test_snapshot_creates_archive_and_db_row(
    snapshot_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext, Path],
) -> None:
    session, _store, ctx, _root = snapshot_db
    env_id = await _create_env_with_memories(session, count=2)

    response = await create_snapshot(
        EnvSnapshotRequest(env_id=env_id, label="test1", include_embeddings=False), ctx=ctx
    )

    assert Path(response.path).is_file()
    row = await session.scalar(select(Snapshot).where(Snapshot.id == response.snapshot_id))
    assert row is not None
    assert row.path == response.path
    assert row.checksum_sha256 == response.checksum


@pytest.mark.asyncio
async def test_snapshot_label_unique_per_env(
    snapshot_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext, Path],
) -> None:
    session, _store, ctx, _root = snapshot_db
    env_id = await _create_env_with_memories(session, count=1)
    await create_snapshot(EnvSnapshotRequest(env_id=env_id, label="dup", include_embeddings=False), ctx=ctx)

    with pytest.raises(Exception) as exc:
        await create_snapshot(EnvSnapshotRequest(env_id=env_id, label="dup", include_embeddings=False), ctx=ctx)
    assert getattr(exc.value, "code", "") == "ALREADY_EXISTS"


@pytest.mark.asyncio
async def test_snapshot_label_can_repeat_across_envs(
    snapshot_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext, Path],
) -> None:
    session, _store, ctx, _root = snapshot_db
    env_a = await _create_env_with_memories(session, name="env-a", count=1)
    env_b = await _create_env_with_memories(session, name="env-b", count=1)

    snap_a = await create_snapshot(EnvSnapshotRequest(env_id=env_a, label="x", include_embeddings=False), ctx=ctx)
    snap_b = await create_snapshot(EnvSnapshotRequest(env_id=env_b, label="x", include_embeddings=False), ctx=ctx)

    assert snap_a.snapshot_id != snap_b.snapshot_id


@pytest.mark.asyncio
async def test_restore_to_new_env(snapshot_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext, Path]) -> None:
    session, _store, ctx, _root = snapshot_db
    env_id = await _create_env_with_memories(session, count=3)
    original_ids = set((await session.execute(select(Memory.id).where(Memory.env_id == env_id))).scalars().all())
    snap = await create_snapshot(EnvSnapshotRequest(env_id=env_id, label="new-env", include_embeddings=False), ctx=ctx)
    await session.execute(
        update(Environment).where(Environment.id == env_id).values(status="deleted", deleted_at=func.now())
    )
    await session.commit()

    out = await restore_snapshot(
        EnvRestoreRequest(snapshot_id=snap.snapshot_id, mode=RestoreMode.restore_to_new_env, new_env_name="restored"),
        ctx=ctx,
    )

    restored_ids = set(
        (await session.execute(select(Memory.id).where(Memory.env_id == out.target_env_id))).scalars().all()
    )
    assert len(restored_ids) == 3
    assert restored_ids.isdisjoint(original_ids)


@pytest.mark.asyncio
async def test_restore_in_place_preserves_uuids(
    snapshot_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext, Path],
) -> None:
    session, _store, ctx, _root = snapshot_db
    env_id = await _create_env_with_memories(session, count=3)
    original_ids = list(
        (await session.execute(select(Memory.id).where(Memory.env_id == env_id).order_by(Memory.title))).scalars().all()
    )
    snap = await create_snapshot(EnvSnapshotRequest(env_id=env_id, label="in-place", include_embeddings=False), ctx=ctx)
    u4 = uuid4()
    session.add(_memory(u4, env_id, "extra"))
    await session.execute(delete(Memory).where(Memory.id == original_ids[0]))
    await session.commit()

    await restore_snapshot(
        EnvRestoreRequest(snapshot_id=snap.snapshot_id, mode=RestoreMode.replace_env_in_place, confirm_destroy=True),
        ctx=ctx,
    )

    restored = set((await session.execute(select(Memory.id).where(Memory.env_id == env_id))).scalars().all())
    assert restored == set(original_ids)
    assert u4 not in restored


@pytest.mark.asyncio
async def test_restore_in_place_requires_confirm(
    snapshot_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext, Path],
) -> None:
    session, _store, ctx, _root = snapshot_db
    env_id = await _create_env_with_memories(session, count=1)
    snap = await create_snapshot(EnvSnapshotRequest(env_id=env_id, label="confirm", include_embeddings=False), ctx=ctx)

    with pytest.raises(Exception) as exc:
        await restore_snapshot(
            EnvRestoreRequest(snapshot_id=snap.snapshot_id, mode=RestoreMode.replace_env_in_place), ctx=ctx
        )
    assert getattr(exc.value, "code", "") == "CONFIRM_DESTROY_REQUIRED"


@pytest.mark.asyncio
async def test_restore_in_place_rebuilds_external_refs(
    snapshot_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext, Path],
) -> None:
    session, _store, ctx, _root = snapshot_db
    env_a = await _create_env_with_memories(session, name="env-a", count=3)
    env_c = await _create_env_with_memories(session, name="env-c", count=1)
    u2 = (
        await session.execute(select(Memory.id).where(Memory.env_id == env_a).order_by(Memory.title).offset(1).limit(1))
    ).scalar_one()
    c_mem = (await session.execute(select(Memory.id).where(Memory.env_id == env_c).limit(1))).scalar_one()
    session.add(MemoryLineage(parent_memory_id=c_mem, child_memory_id=u2, relation="copied_from"))
    await session.commit()
    snap = await create_snapshot(EnvSnapshotRequest(env_id=env_a, label="external", include_embeddings=False), ctx=ctx)
    await session.execute(delete(Memory).where(Memory.id == u2))
    await session.commit()

    await restore_snapshot(
        EnvRestoreRequest(snapshot_id=snap.snapshot_id, mode=RestoreMode.replace_env_in_place, confirm_destroy=True),
        ctx=ctx,
    )

    assert await session.scalar(select(Memory.id).where(Memory.id == u2)) == u2
    edge_count = await session.scalar(
        select(func.count())
        .select_from(MemoryLineage)
        .where(
            MemoryLineage.parent_memory_id == c_mem,
            MemoryLineage.child_memory_id == u2,
            MemoryLineage.relation == "copied_from",
        )
    )
    assert edge_count == 1


@pytest.mark.asyncio
async def test_snapshot_size_warning_at_threshold(
    snapshot_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _store, ctx, _root = snapshot_db
    env_id = await _create_env_with_memories(session, count=1)

    async def too_large(_root: Path) -> int:
        return 11 * 1024 * 1024 * 1024

    warnings: list[str] = []

    def capture_warning(message: str, *args: object) -> None:
        warnings.append(message % args)

    monkeypatch.setattr(snapshot_ops, "_snapshot_tree_size", too_large)
    monkeypatch.setattr(snapshot_ops.log, "warning", capture_warning)
    await create_snapshot(EnvSnapshotRequest(env_id=env_id, label="warn", include_embeddings=False), ctx=ctx)

    assert any("snapshot storage" in message for message in warnings)


async def _create_env_with_memories(session: AsyncSession, *, name: str = "snap-env", count: int) -> UUID:
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
    await session.flush()
    for index in range(count):
        session.add(_memory(uuid4(), env_id, f"memory-{index}"))
    await session.commit()
    return env_id


def _memory(memory_id: UUID, env_id: UUID, title: str) -> Memory:
    return Memory(
        id=memory_id,
        env_id=env_id,
        kind="fact",
        status="active",
        title=title,
        body=f"body for {title}",
        salience=0.5,
        confidence=0.5,
        pinned=False,
        metadata_={},
        version=1,
    )


async def _truncate(session: AsyncSession) -> None:
    await session.execute(
        text(
            "TRUNCATE snapshots, outbox_delivery, outbox, projection_state, audit_log, env_grants, "
            "dream_proposals, dream_runs, memory_sources, memory_lineage, memory_tags, "
            "relations, graph_nodes, tasks, tags, entity_aliases, entities, memories, "
            "sessions, tokens, agents, environments CASCADE"
        )
    )
    await session.commit()
