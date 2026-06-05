from __future__ import annotations

import asyncio
import json
import os
import tarfile
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from memory_mcp_schemas.env_ops import EnvExportRequest, EnvImportRequest, ExportFormat, ImportMode
from memory_mcp_schemas.envs import EnvCreateRequest
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer

from memory_mcp import envs
from memory_mcp.db.models import (
    Agent,
    Entity,
    EntityAlias,
    Environment,
    GraphNode,
    Memory,
    MemoryLineage,
    MemorySource,
    MemoryTag,
    Relation,
    Tag,
    Task,
)
from memory_mcp.env_ops import export as exporter
from memory_mcp.env_ops import import_ as importer
from memory_mcp.env_ops._checksums import verify_checksums_file
from memory_mcp.env_ops.export import export_env
from memory_mcp.env_ops.import_ import import_env
from memory_mcp.envs import env_create
from memory_mcp.errors import NotFoundError
from memory_mcp.identity import AgentContext

REPO_ROOT = Path(__file__).resolve().parents[2]
ROW_TABLES = (
    "memories",
    "memory_tags",
    "tags",
    "entities",
    "entity_aliases",
    "relations",
    "graph_nodes",
    "tasks",
    "memory_lineage",
    "memory_sources",
)


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
        for key in list(self.vectors):
            if key[0] == env_id and key[1] == point_id:
                self.vectors.pop(key, None)

    async def search(
        self,
        *,
        env_id: UUID,
        query_vector: Sequence[float],
        limit: int,
        filters: Mapping[str, Any] | None = None,
        vector_name: str = "body",
    ) -> list[dict[str, Any]]:
        return []

    async def get_vector(self, *, env_id: UUID, id: str, vector_name: str = "body") -> list[float] | None:
        return self.vectors.get((env_id, UUID(id), vector_name))

    async def get_vectors(
        self,
        *,
        env_id: UUID,
        ids: list[UUID],
        vector_name: str = "body",
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
        pytest.skip(f"Postgres testcontainer unavailable; Docker is required for env_ops roundtrip tests: {exc!r}")

    engine = None
    try:
        sync_url = container.get_connection_url(driver="psycopg2")
        async_url = container.get_connection_url(driver="asyncpg")
        config = Config(str(REPO_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
        old_url = os.environ.get("POSTGRES_URL")
        os.environ["POSTGRES_URL"] = sync_url
        try:
            command.upgrade(config, "head")
        finally:
            if old_url is None:
                os.environ.pop("POSTGRES_URL", None)
            else:
                os.environ["POSTGRES_URL"] = old_url

        engine = create_async_engine(async_url, pool_pre_ping=True, poolclass=NullPool)
        yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    finally:
        if engine is not None:
            asyncio.run(engine.dispose())
        with contextlib_suppress():
            container.stop()


@pytest.fixture
async def roundtrip_db(
    postgres_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncSession, _MemoryVectorStore, AgentContext]]:
    store = _MemoryVectorStore()
    ctx = AgentContext(agent_id=uuid4(), agent_name="roundtrip-agent")

    @asynccontextmanager
    async def routed_session_scope() -> AsyncIterator[AsyncSession]:
        async with postgres_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr(exporter, "session_scope", routed_session_scope)
    monkeypatch.setattr(importer, "session_scope", routed_session_scope)
    monkeypatch.setattr(envs, "session_scope", routed_session_scope)
    monkeypatch.setattr(exporter, "_default_vector_store", lambda: store)
    monkeypatch.setattr(importer, "_default_vector_store", lambda: store)

    async with postgres_factory() as session:
        await _truncate(session)
        session.add(Agent(id=ctx.agent_id, name="roundtrip-agent"))
        await session.commit()

    async with postgres_factory() as session:
        yield session, store, ctx

    async with postgres_factory() as session:
        await _truncate(session)


def contextlib_suppress():
    from contextlib import suppress

    return suppress(Exception)


async def create_canonical_env(session: AsyncSession, *, name: str = "rt-canonical") -> UUID:
    """Build the canonical roundtrip fixture and return env_id.

    The current Task schema has no parent_task_id column, so the task fixture is
    represented by a single root task with no parent relationship.
    """

    env_id = uuid4()
    env = Environment(
        id=env_id,
        name=f"{name}-{uuid4().hex[:8]}",
        kind="test",
        retention_policy={"days": 90},
        default_embedding_model_id="test-model",
    )
    session.add(env)
    await session.flush()

    fact = Memory(
        id=uuid4(),
        env_id=env_id,
        kind="fact",
        status="active",
        title="Roundtrip fact",
        body="canonical fact body for roundtrip",
        salience=0.7,
        confidence=0.9,
        pinned=True,
        metadata_={"ordinal": 1},
        version=2,
    )
    decision = Memory(
        id=uuid4(),
        env_id=env_id,
        kind="decision",
        status="active",
        title="Roundtrip decision",
        body="canonical decision body for roundtrip",
        salience=0.6,
        confidence=0.8,
        pinned=False,
        metadata_={"ordinal": 2},
        decision_meta={"status": "accepted", "rationale": "roundtrip fixture"},
        version=3,
    )
    playbook = Memory(
        id=uuid4(),
        env_id=env_id,
        kind="playbook",
        status="active",
        title="Roundtrip playbook",
        body="canonical playbook body for roundtrip",
        steps=["export", "import", "export"],
        macro="roundtrip-demo",
        salience=0.5,
        confidence=0.75,
        pinned=False,
        metadata_={"ordinal": 3},
        version=1,
    )
    session.add_all([fact, decision, playbook])
    await session.flush()
    fact.status = "superseded"
    fact.superseded_by = decision.id

    tag_a = Tag(id=uuid4(), env_id=env_id, name="shared")
    tag_b = Tag(id=uuid4(), env_id=env_id, name="roundtrip")
    session.add_all([tag_a, tag_b])
    await session.flush()
    session.add_all(
        [
            MemoryTag(memory_id=fact.id, tag_id=tag_a.id, env_id=env_id),
            MemoryTag(memory_id=decision.id, tag_id=tag_a.id, env_id=env_id),
            MemoryTag(memory_id=decision.id, tag_id=tag_b.id, env_id=env_id),
            MemoryTag(memory_id=playbook.id, tag_id=tag_b.id, env_id=env_id),
        ]
    )

    entity_a = Entity(
        id=uuid4(),
        env_id=env_id,
        kind="service",
        canonical_name="Roundtrip Service",
        normalized_name="roundtrip service",
        metadata_={"tier": "test"},
        version=1,
    )
    entity_b = Entity(
        id=uuid4(),
        env_id=env_id,
        kind="repo",
        canonical_name="Roundtrip Repo",
        normalized_name="roundtrip repo",
        metadata_={"language": "python"},
        version=1,
    )
    session.add_all([entity_a, entity_b])
    await session.flush()
    session.add_all(
        [
            EntityAlias(entity_id=entity_a.id, env_id=env_id, alias="RT Service", normalized_alias="rt service"),
            EntityAlias(entity_id=entity_b.id, env_id=env_id, alias="rt-repo", normalized_alias="rt-repo"),
        ]
    )

    node_a = GraphNode(id=uuid4(), env_id=env_id, node_type="entity", entity_id=entity_a.id)
    node_b = GraphNode(id=uuid4(), env_id=env_id, node_type="entity", entity_id=entity_b.id)
    session.add_all([node_a, node_b])
    await session.flush()
    session.add(
        Relation(
            id=uuid4(),
            env_id=env_id,
            src_node_id=node_a.id,
            dst_node_id=node_b.id,
            type="documented_by",
            properties={"strength": "fixture"},
            version=1,
        )
    )

    session.add_all(
        [
            MemoryLineage(parent_memory_id=fact.id, child_memory_id=decision.id, relation="supersedes"),
            MemoryLineage(parent_memory_id=decision.id, child_memory_id=playbook.id, relation="copied_from"),
            Task(id=uuid4(), env_id=env_id, title="Roundtrip root task", status="pending", priority=20, version=1),
            MemorySource(memory_id=fact.id, source_type="other", source_ref="canonical:fact", evidence_span="fact"),
            MemorySource(
                memory_id=decision.id,
                source_type="other",
                source_ref="canonical:decision",
                evidence_span="decision",
            ),
        ]
    )
    await session.commit()
    return env_id


def memory_signature(m: dict[str, Any]) -> tuple[str, str]:
    return (m["kind"], m["body"][:200])


@pytest.mark.asyncio
async def test_roundtrip_directory_format_preserves_all_tables(
    tmp_path: Path,
    roundtrip_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = roundtrip_db
    src_env_id = await create_canonical_env(session)

    src = await _export(src_env_id, tmp_path / "src_export", ExportFormat.directory, ctx)
    target = await _create_target_env("rt-target", ctx)
    await import_env(
        EnvImportRequest(source_path=src.output_path, target_env_id=target.id, mode=ImportMode.fail, dry_run=False),
        ctx=ctx,
    )
    dst = await _export(target.id, tmp_path / "dst_export", ExportFormat.directory, ctx)

    src_root = Path(src.output_path)
    dst_root = Path(dst.output_path)
    assert _relative_files(src_root) == _relative_files(dst_root)
    for rel_path in _relative_files(src_root):
        if rel_path.endswith(".jsonl"):
            assert _jsonl_count(src_root / rel_path) == _jsonl_count(dst_root / rel_path)

    src_memories = _jsonl(src_root / "memories.jsonl")
    dst_memories = _jsonl(dst_root / "memories.jsonl")
    assert Counter(m["kind"] for m in src_memories) == Counter(m["kind"] for m in dst_memories)
    assert {memory_signature(m): (m["kind"], m["body"], m["status"]) for m in src_memories} == {
        memory_signature(m): (m["kind"], m["body"], m["status"]) for m in dst_memories
    }
    _assert_supersession_chain_preserved(src_memories, dst_memories)
    assert _stable_manifest_counts(src_root) == _stable_manifest_counts(dst_root)


@pytest.mark.asyncio
async def test_roundtrip_archive_format_preserves_checksums(
    tmp_path: Path,
    roundtrip_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = roundtrip_db
    src_env_id = await create_canonical_env(session)

    first = await _export(src_env_id, tmp_path / "first_archive", ExportFormat.archive, ctx)
    first_root = _unpack(Path(first.output_path), tmp_path / "first_unpacked")
    assert await verify_checksums_file(first_root / "checksums.sha256", first_root)

    imported = await import_env(
        EnvImportRequest(source_path=first.output_path, target_env_name=f"rt-archive-{uuid4().hex[:8]}", dry_run=False),
        ctx=ctx,
    )
    second = await _export(imported.target_env_id, tmp_path / "second_archive", ExportFormat.archive, ctx)
    second_root = _unpack(Path(second.output_path), tmp_path / "second_unpacked")
    assert await verify_checksums_file(second_root / "checksums.sha256", second_root)

    chained = await import_env(
        EnvImportRequest(
            source_path=second.output_path, target_env_name=f"rt-archive-chain-{uuid4().hex[:8]}", dry_run=False
        ),
        ctx=ctx,
    )
    assert chained.counts["memories"] == 3


@pytest.mark.asyncio
async def test_roundtrip_with_embeddings_include_flag_off(
    tmp_path: Path,
    roundtrip_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, store, ctx = roundtrip_db
    src_env_id = await create_canonical_env(session)
    src_ids = (await session.execute(select(Memory.id).where(Memory.env_id == src_env_id).limit(2))).scalars().all()
    for index, memory_id in enumerate(src_ids, start=1):
        store.vectors[(src_env_id, memory_id, "body")] = [float(index), float(index) + 0.1]

    src = await _export(
        src_env_id,
        tmp_path / "src_no_embeddings",
        ExportFormat.directory,
        ctx,
        include_embeddings=False,
    )
    assert not (Path(src.output_path) / "embeddings").exists()

    target = await _create_target_env("rt-no-embeddings", ctx)
    report = await import_env(
        EnvImportRequest(source_path=src.output_path, target_env_id=target.id, mode=ImportMode.fail, dry_run=False),
        ctx=ctx,
    )
    target_ids = (await session.execute(select(Memory.id).where(Memory.env_id == target.id))).scalars().all()
    vectors = await store.get_vectors(env_id=target.id, ids=list(target_ids), vector_name="body")
    assert report.counts["memories"] == 3
    assert target_ids
    assert all(vector is None for vector in vectors.values())


@pytest.mark.asyncio
async def test_roundtrip_dry_run_creates_nothing(
    tmp_path: Path,
    roundtrip_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = roundtrip_db
    src_env_id = await create_canonical_env(session)
    src = await _export(src_env_id, tmp_path / "src_dry_run", ExportFormat.directory, ctx)
    target = await _create_target_env("rt-dry-run", ctx)

    report = await import_env(
        EnvImportRequest(source_path=src.output_path, target_env_id=target.id, mode=ImportMode.fail, dry_run=True),
        ctx=ctx,
    )

    assert report.dry_run is True
    assert report.counts["memories"] == 3
    assert report.remap_table_size > 0
    for table, count in (await _count_env_rows(session, target.id)).items():
        assert count == 0, table


@pytest.mark.asyncio
async def test_roundtrip_skip_mode_handles_tag_collision(
    tmp_path: Path,
    roundtrip_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = roundtrip_db
    src_env_id = await create_canonical_env(session)
    src = await _export(src_env_id, tmp_path / "src_tag_collision", ExportFormat.directory, ctx)
    target = await _create_target_env("rt-tag-collision", ctx)
    original_tag = Tag(id=uuid4(), env_id=target.id, name="shared")
    session.add(original_tag)
    await session.commit()

    report = await import_env(
        EnvImportRequest(source_path=src.output_path, target_env_id=target.id, mode=ImportMode.skip, dry_run=False),
        ctx=ctx,
    )

    shared_count = await session.scalar(
        select(func.count()).select_from(Tag).where(Tag.env_id == target.id, Tag.name == "shared")
    )
    assert report.conflicts["tags"] >= 1
    assert shared_count == 1
    linked_shared = await session.scalar(
        select(func.count())
        .select_from(MemoryTag)
        .where(MemoryTag.env_id == target.id, MemoryTag.tag_id == original_tag.id)
    )
    assert linked_shared == 0


@pytest.mark.asyncio
async def test_roundtrip_two_pass_supersession_resolves(
    tmp_path: Path,
    roundtrip_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = roundtrip_db
    env_id = await _create_supersession_chain(session)
    src = await _export(env_id, tmp_path / "src_chain", ExportFormat.directory, ctx)
    imported = await import_env(
        EnvImportRequest(source_path=src.output_path, target_env_name=f"rt-chain-{uuid4().hex[:8]}", dry_run=False),
        ctx=ctx,
    )
    dst = await _export(imported.target_env_id, tmp_path / "dst_chain", ExportFormat.directory, ctx)

    src_memories = _jsonl(Path(src.output_path) / "memories.jsonl")
    dst_memories = _jsonl(Path(dst.output_path) / "memories.jsonl")
    _assert_supersession_chain_preserved(src_memories, dst_memories, expected_edges=2)
    source_ids = {UUID(m["id"]) for m in src_memories}
    assert all(UUID(m["superseded_by"]) not in source_ids for m in dst_memories if m.get("superseded_by"))


@pytest.mark.asyncio
async def test_roundtrip_deleted_env_cannot_be_exported(
    tmp_path: Path,
    roundtrip_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = roundtrip_db
    env_id = await create_canonical_env(session)
    await session.execute(
        update(Environment).where(Environment.id == env_id).values(status="deleted", deleted_at=func.now())
    )
    await session.commit()

    with pytest.raises(NotFoundError) as exc:
        await _export(env_id, tmp_path / "deleted_export", ExportFormat.directory, ctx)
    assert exc.value.code == "ENV_DELETED"


@pytest.mark.asyncio
async def test_roundtrip_counts_match_actual_rows(
    tmp_path: Path,
    roundtrip_db: tuple[AsyncSession, _MemoryVectorStore, AgentContext],
) -> None:
    session, _store, ctx = roundtrip_db
    env_id = await create_canonical_env(session)
    exported = await _export(env_id, tmp_path / "counts_export", ExportFormat.directory, ctx)

    manifest_counts = json.loads((Path(exported.output_path) / "manifest.json").read_text(encoding="utf-8"))["counts"]
    db_counts = await _count_env_rows(session, env_id)
    db_counts["env"] = 1
    for table, count in db_counts.items():
        assert manifest_counts[table] == count, table


async def _create_target_env(name: str, ctx: AgentContext):
    return await env_create(
        EnvCreateRequest(
            name=f"{name}-{uuid4().hex[:8]}",
            kind="test",
            retention_policy={},
            default_embedding_model_id="test-model",
        ),
        ctx=ctx,
    )


async def _export(
    env_id: UUID,
    target_path: Path,
    format_: ExportFormat,
    ctx: AgentContext,
    **kwargs: Any,
):
    return await export_env(
        EnvExportRequest(
            env_id=env_id,
            format=format_,
            target_path=str(target_path),
            include_embeddings=kwargs.pop("include_embeddings", False),
            **kwargs,
        ),
        ctx=ctx,
    )


async def _truncate(session: AsyncSession) -> None:
    await session.execute(
        text(
            "TRUNCATE outbox_delivery, outbox, projection_state, audit_log, env_grants, "
            "dream_proposals, dream_runs, memory_sources, memory_lineage, memory_tags, "
            "relations, graph_nodes, tasks, tags, entity_aliases, entities, memories, "
            "sessions, tokens, agents, environments CASCADE"
        )
    )
    await session.commit()


async def _count_env_rows(session: AsyncSession, env_id: UUID) -> dict[str, int]:
    counts: dict[str, int] = {}
    counts["memories"] = await _count(session, select(func.count()).select_from(Memory).where(Memory.env_id == env_id))
    counts["memory_tags"] = await _count(
        session, select(func.count()).select_from(MemoryTag).where(MemoryTag.env_id == env_id)
    )
    counts["tags"] = await _count(session, select(func.count()).select_from(Tag).where(Tag.env_id == env_id))
    counts["entities"] = await _count(session, select(func.count()).select_from(Entity).where(Entity.env_id == env_id))
    counts["entity_aliases"] = await _count(
        session, select(func.count()).select_from(EntityAlias).where(EntityAlias.env_id == env_id)
    )
    counts["relations"] = await _count(
        session, select(func.count()).select_from(Relation).where(Relation.env_id == env_id)
    )
    counts["graph_nodes"] = await _count(
        session, select(func.count()).select_from(GraphNode).where(GraphNode.env_id == env_id)
    )
    counts["tasks"] = await _count(session, select(func.count()).select_from(Task).where(Task.env_id == env_id))
    counts["memory_sources"] = await _count(
        session,
        select(func.count()).select_from(MemorySource).join(Memory).where(Memory.env_id == env_id),
    )
    parent = select(Memory.id).where(Memory.env_id == env_id).subquery()
    child = select(Memory.id).where(Memory.env_id == env_id).subquery()
    counts["memory_lineage"] = await _count(
        session,
        select(func.count())
        .select_from(MemoryLineage)
        .where(MemoryLineage.parent_memory_id.in_(select(parent.c.id)))
        .where(MemoryLineage.child_memory_id.in_(select(child.c.id))),
    )
    return counts


async def _count(session: AsyncSession, stmt: Any) -> int:
    value = await session.scalar(stmt)
    return int(value or 0)


async def _create_supersession_chain(session: AsyncSession) -> UUID:
    env_id = uuid4()
    env = Environment(
        id=env_id,
        name=f"rt-chain-src-{uuid4().hex[:8]}",
        retention_policy={},
        default_embedding_model_id="test-model",
    )
    session.add(env)
    await session.flush()
    a = Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="chain A", version=1)
    b = Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="chain B", version=1)
    c = Memory(id=uuid4(), env_id=env_id, kind="fact", status="active", body="chain C", version=1)
    session.add_all([a, b, c])
    await session.flush()
    a.status = "superseded"
    a.superseded_by = b.id
    b.status = "superseded"
    b.superseded_by = c.id
    await session.commit()
    return env_id


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonl_count(path: Path) -> int:
    return len(_jsonl(path))


def _relative_files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _stable_manifest_counts(root: Path) -> dict[str, int]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))["counts"]


def _assert_supersession_chain_preserved(
    src_memories: list[dict[str, Any]],
    dst_memories: list[dict[str, Any]],
    *,
    expected_edges: int = 1,
) -> None:
    src_by_id = {UUID(m["id"]): m for m in src_memories}
    dst_by_sig = {memory_signature(m): m for m in dst_memories}
    edges = 0
    for src in src_memories:
        if not src.get("superseded_by"):
            continue
        dst = dst_by_sig[memory_signature(src)]
        src_target = src_by_id[UUID(src["superseded_by"])]
        dst_target = dst_by_sig[memory_signature(src_target)]
        assert UUID(dst["superseded_by"]) == UUID(dst_target["id"])
        edges += 1
    assert edges == expected_edges


def _unpack(archive_path: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target)
    return target
