"""Snapshot / restore implementation for v0.8 environment operations."""

from __future__ import annotations

import logging
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from memory_mcp_schemas.env_ops import (
    EnvExportRequest,
    EnvImportReport,
    EnvImportRequest,
    EnvRestoreRequest,
    EnvRestoreResponse,
    EnvSnapshotRequest,
    EnvSnapshotResponse,
    ExportFormat,
    ImportMode,
    MemoryVectorRecord,
    RestoreMode,
)
from sqlalchemy import delete, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp.config import get_settings
from memory_mcp.db.models import (
    DreamProposal,
    DreamRun,
    Entity,
    EntityAlias,
    EnvGrant,
    Environment,
    GraphNode,
    Memory,
    MemoryLineage,
    MemorySource,
    MemoryTag,
    Relation,
    Snapshot,
    Tag,
    Task,
)
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.env_ops._checksums import sha256_file
from memory_mcp.env_ops._io import JsonlReader, JsonlWriter
from memory_mcp.env_ops.export import export_env
from memory_mcp.env_ops.import_ import (
    _extract_archive,
    _json_nullable,
    _json_obj,
    _load_verified_manifest,
    _maybe_datetime,
    _required_uuid,
)
from memory_mcp.errors import AlreadyExistsError, InvalidInputError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import _projection_payload

SCHEMA_VERSION = "0.8.0"
_SNAPSHOT_WARN_BYTES = 10 * 1024 * 1024 * 1024
log = logging.getLogger(__name__)
VectorStoreFactory = Callable[[], VectorStore]


async def create_snapshot(request: EnvSnapshotRequest, *, ctx: AgentContext) -> EnvSnapshotResponse:
    """Snapshot an env to a retained archive under ``data_root/snapshots``."""

    snapshot_id = uuid4()
    settings = get_settings()
    snapshot_dir = Path(settings.data_root).expanduser() / "snapshots" / str(request.env_id)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    target_base = snapshot_dir / f"{snapshot_id}.memarchive"

    async with session_scope() as session:
        env = await _load_active_env(session, request.env_id)
        rbac.require("write", env.id, ctx)

    exported = await export_env(
        EnvExportRequest(
            env_id=request.env_id,
            format=ExportFormat.archive,
            target_path=str(target_base),
            include_embeddings=request.include_embeddings,
            include_provenance=True,
            include_grants=True,
            include_dream_history=True,
        ),
        ctx=ctx,
    )
    archive_path = Path(exported.output_path).resolve()
    await _augment_archive_with_external_lineage(archive_path, request.env_id)
    size_bytes = archive_path.stat().st_size
    checksum = await sha256_file(archive_path)

    snapshot = Snapshot(
        id=snapshot_id,
        env_id=request.env_id,
        label=request.label,
        created_by_agent_id=ctx.agent_id,
        path=str(archive_path),
        size_bytes=size_bytes,
        checksum_sha256=checksum,
        schema_version=exported.manifest.schema_version,
        notes=getattr(request, "notes", None),
    )
    try:
        async with session_scope() as session:
            session.add(snapshot)
            await session.flush()
            await session.refresh(snapshot)
    except IntegrityError as exc:
        raise AlreadyExistsError(
            f"snapshot label {request.label!r} already exists for env {request.env_id}",
            env_id=str(request.env_id),
            label=request.label,
        ) from exc

    snapshots_root = Path(settings.data_root).expanduser() / "snapshots"
    total = await _snapshot_tree_size(snapshots_root)
    if total > _SNAPSHOT_WARN_BYTES:
        log.warning("snapshot storage under %s is %d bytes (>10 GB)", snapshots_root, total)

    return EnvSnapshotResponse(
        snapshot_id=snapshot.id,
        env_id=snapshot.env_id,
        label=snapshot.label,
        created_at=snapshot.created_at,
        path=snapshot.path,
        size_bytes=snapshot.size_bytes,
        checksum=snapshot.checksum_sha256,
    )


async def restore_snapshot(request: EnvRestoreRequest, *, ctx: AgentContext) -> EnvRestoreResponse:
    """Restore a retained snapshot into a new env or back into its original env."""

    async with session_scope() as session:
        snapshot = await _load_snapshot(session, request.snapshot_id)
        if request.mode == RestoreMode.replace_env_in_place:
            env = await _load_active_env(session, snapshot.env_id)
            rbac.require("write", env.id, ctx)
        else:
            rbac.require("admin", None, ctx)

    archive_path = Path(snapshot.path)
    await _verify_snapshot_archive(archive_path, snapshot)

    if request.mode == RestoreMode.restore_to_new_env:
        if not request.new_env_name:
            raise InvalidInputError("new_env_name is required for restore_to_new_env")
        report: EnvImportReport = await import_env(
            EnvImportRequest(
                source_path=str(archive_path),
                target_env_name=request.new_env_name,
                mode=ImportMode.fail,
                dry_run=False,
            ),
            ctx=ctx,
        )
        return EnvRestoreResponse(
            snapshot_id=snapshot.id,
            mode=request.mode,
            target_env_id=report.target_env_id,
            counts_restored=dict(report.counts),
            import_report=report,
            pending_vector_rebuild=report.pending_vector_rebuild,
            re_embed_count=report.re_embed_count,
        )

    if not request.confirm_destroy:
        exc = InvalidInputError("CONFIRM_DESTROY_REQUIRED: replace_env_in_place requires confirm_destroy=True")
        exc.code = "CONFIRM_DESTROY_REQUIRED"
        raise exc

    tmp: tempfile.TemporaryDirectory[str] | None = None
    try:
        root, tmp = await _open_archive_root(archive_path)
        manifest = await _load_verified_manifest(root)
        if manifest.source.env_id != snapshot.env_id:
            raise InvalidInputError(
                "snapshot source env does not match snapshot row",
                snapshot_env_id=str(snapshot.env_id),
                manifest_env_id=str(manifest.source.env_id),
            )
        async with session_scope() as session:
            counts, inserted, tag_names = await _restore_in_place(root, snapshot.env_id, session=session, ctx=ctx)
        pending = await _restore_embeddings(
            root,
            env_id=snapshot.env_id,
            target_model_id=manifest.source.default_embedding_model_id,
            inserted_memories=inserted,
            tag_names=tag_names,
        )
        return EnvRestoreResponse(
            snapshot_id=snapshot.id,
            mode=request.mode,
            target_env_id=snapshot.env_id,
            counts_restored=dict(counts),
            pending_vector_rebuild=pending,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


async def _load_active_env(session: AsyncSession, env_id: UUID) -> Environment:
    env = (await session.execute(select(Environment).where(Environment.id == env_id))).scalars().first()
    if env is None:
        raise NotFoundError(f"environment {env_id} not found", env_id=str(env_id))
    if getattr(env, "status", "active") == "deleted":
        exc = NotFoundError(f"environment {env_id} is deleted", env_id=str(env_id))
        exc.code = "ENV_DELETED"
        raise exc
    return env


async def _load_snapshot(session: AsyncSession, snapshot_id: UUID) -> Snapshot:
    snapshot = (await session.execute(select(Snapshot).where(Snapshot.id == snapshot_id))).scalars().first()
    if snapshot is None:
        raise NotFoundError(f"snapshot {snapshot_id} not found", snapshot_id=str(snapshot_id))
    return snapshot


async def _verify_snapshot_archive(path: Path, snapshot: Snapshot) -> None:
    if not path.is_file():
        raise NotFoundError(f"snapshot archive missing: {path}", path=str(path), snapshot_id=str(snapshot.id))
    checksum = await sha256_file(path)
    if checksum != snapshot.checksum_sha256:
        raise InvalidInputError("snapshot archive checksum mismatch", snapshot_id=str(snapshot.id), path=str(path))


async def _open_archive_root(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None
    scratch = Path(".tmp") / "env-snapshot-restore"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(prefix="archive-", dir=scratch)
    root = Path(tmp.name)
    _extract_archive(path, root)
    return root, tmp


async def _restore_in_place(
    root: Path,
    env_id: UUID,
    *,
    session: AsyncSession,
    ctx: AgentContext,
) -> tuple[Mapping[str, int], dict[UUID, Memory], Mapping[UUID, list[str]]]:
    counts: defaultdict[str, int] = defaultdict(int)
    target_ids = set((await session.execute(select(Memory.id).where(Memory.env_id == env_id))).scalars().all())
    external_lineage = await _external_lineage_rows(session, target_ids)
    await _delete_env_rows(session, env_id, target_ids)

    await _apply_tags(root, session, env_id, counts)
    await _apply_entities(root, session, env_id, counts)
    await _apply_entity_aliases(root, session, env_id, counts)
    inserted = await _apply_memories(root, session, env_id, counts)
    tag_names = await _apply_memory_tags(root, session, env_id, counts)
    await _apply_memory_sources(root, session, counts)
    await _apply_tasks(root, session, env_id, ctx, counts)
    await _apply_graph_nodes(root, session, env_id, counts)
    await _apply_relations(root, session, env_id, counts)
    await _apply_memory_lineage(root, session, counts)
    await _apply_external_memory_lineage(root, session, env_id, counts)
    await _restore_external_lineage(session, external_lineage, counts)
    await _apply_dream_runs(root, session, env_id, counts)
    await _apply_dream_proposals(root, session, env_id, counts)
    await _apply_grants(root, session, env_id, counts)
    await _apply_superseded_by(root, session, counts)
    return counts, inserted, tag_names


async def _delete_env_rows(session: AsyncSession, env_id: UUID, memory_ids: set[UUID]) -> None:
    await session.execute(delete(Relation).where(Relation.env_id == env_id))
    await session.execute(delete(MemoryTag).where(MemoryTag.env_id == env_id))
    await session.execute(delete(EntityAlias).where(EntityAlias.env_id == env_id))
    await session.execute(delete(GraphNode).where(GraphNode.env_id == env_id))
    await session.execute(delete(Task).where(Task.env_id == env_id))
    await session.execute(delete(DreamProposal).where(DreamProposal.env_id == env_id))
    await session.execute(delete(DreamRun).where(DreamRun.env_id == env_id))
    if memory_ids:
        await session.execute(delete(MemorySource).where(MemorySource.memory_id.in_(memory_ids)))
        await session.execute(
            delete(MemoryLineage).where(
                or_(MemoryLineage.parent_memory_id.in_(memory_ids), MemoryLineage.child_memory_id.in_(memory_ids))
            )
        )
    await session.execute(delete(Memory).where(Memory.env_id == env_id))
    await session.execute(delete(Tag).where(Tag.env_id == env_id))
    await session.execute(delete(Entity).where(Entity.env_id == env_id))
    await session.execute(delete(EnvGrant).where(EnvGrant.env_id == env_id))
    await session.flush()


async def _external_lineage_rows(session: AsyncSession, memory_ids: set[UUID]) -> list[dict[str, Any]]:
    if not memory_ids:
        return []
    rows = (
        (
            await session.execute(
                select(MemoryLineage).where(
                    or_(MemoryLineage.parent_memory_id.in_(memory_ids), MemoryLineage.child_memory_id.in_(memory_ids))
                )
            )
        )
        .scalars()
        .all()
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        parent_in = row.parent_memory_id in memory_ids
        child_in = row.child_memory_id in memory_ids
        if parent_in != child_in:
            out.append(_row(row, ["parent_memory_id", "child_memory_id", "relation", "created_at"]))
    return out


async def _restore_external_lineage(
    session: AsyncSession, rows: list[dict[str, Any]], counts: defaultdict[str, int]
) -> None:
    for row in rows:
        if not await _memories_exist(
            session, _required_uuid(row, "parent_memory_id"), _required_uuid(row, "child_memory_id")
        ):
            continue
        session.add(
            MemoryLineage(
                parent_memory_id=_required_uuid(row, "parent_memory_id"),
                child_memory_id=_required_uuid(row, "child_memory_id"),
                relation=row["relation"],
                created_at=_maybe_datetime(row.get("created_at")),
            )
        )
        await session.flush()
        counts["external_memory_lineage"] += 1


async def _memories_exist(session: AsyncSession, *ids: UUID) -> bool:
    found = set((await session.execute(select(Memory.id).where(Memory.id.in_(ids)))).scalars().all())
    return set(ids) <= found


async def _apply_tags(root: Path, session: AsyncSession, env_id: UUID, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "tags"):
        session.add(Tag(id=_required_uuid(row, "id"), env_id=env_id, name=row["name"]))
        await session.flush()
        counts["tags"] += 1


async def _apply_entities(root: Path, session: AsyncSession, env_id: UUID, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "entities"):
        session.add(
            Entity(
                id=_required_uuid(row, "id"),
                env_id=env_id,
                kind=row["kind"],
                canonical_name=row["canonical_name"],
                normalized_name=row["normalized_name"],
                metadata_=_json_obj(row.get("metadata") or row.get("metadata_")),
                created_at=_maybe_datetime(row.get("created_at")),
                updated_at=_maybe_datetime(row.get("updated_at")),
                version=int(row.get("version", 1)),
            )
        )
        await session.flush()
        counts["entities"] += 1


async def _apply_entity_aliases(root: Path, session: AsyncSession, env_id: UUID, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "entity_aliases"):
        session.add(
            EntityAlias(
                entity_id=_required_uuid(row, "entity_id"),
                env_id=env_id,
                alias=row["alias"],
                normalized_alias=row["normalized_alias"],
                created_at=_maybe_datetime(row.get("created_at")),
            )
        )
        await session.flush()
        counts["entity_aliases"] += 1


async def _apply_memories(
    root: Path,
    session: AsyncSession,
    env_id: UUID,
    counts: defaultdict[str, int],
) -> dict[UUID, Memory]:
    inserted: dict[UUID, Memory] = {}
    async for row in _rows(root, "memories"):
        memory_id = _required_uuid(row, "id")
        obj = Memory(
            id=memory_id,
            env_id=env_id,
            kind=row["kind"],
            status="active"
            if row.get("status") == "superseded" and row.get("superseded_by")
            else row.get("status", "active"),
            title=row.get("title"),
            body=row["body"],
            trigger_description=row.get("trigger_description"),
            steps=row.get("steps"),
            macro=row.get("macro"),
            salience=float(row.get("salience", 0.5)),
            confidence=float(row.get("confidence", 0.5)),
            access_count=int(row.get("access_count", 0)),
            last_accessed_at=_maybe_datetime(row.get("last_accessed_at")),
            pinned=bool(row.get("pinned", False)),
            negative_feedback_count=int(row.get("negative_feedback_count", 0)),
            verified_at=_maybe_datetime(row.get("verified_at")),
            created_at=_maybe_datetime(row.get("created_at")),
            updated_at=_maybe_datetime(row.get("updated_at")),
            expires_at=_maybe_datetime(row.get("expires_at")),
            superseded_by=None,
            metadata_=_json_obj(row.get("metadata") or row.get("metadata_")),
            decision_meta=_json_nullable(row.get("decision_meta")),
            version=int(row.get("version", 1)),
        )
        session.add(obj)
        await session.flush()
        inserted[memory_id] = obj
        counts["memories"] += 1
    return inserted


async def _apply_memory_tags(
    root: Path,
    session: AsyncSession,
    env_id: UUID,
    counts: defaultdict[str, int],
) -> Mapping[UUID, list[str]]:
    tag_names: defaultdict[UUID, list[str]] = defaultdict(list)
    async for row in _rows(root, "memory_tags"):
        memory_id = _required_uuid(row, "memory_id")
        tag_id = _required_uuid(row, "tag_id")
        session.add(MemoryTag(memory_id=memory_id, tag_id=tag_id, env_id=env_id))
        await session.flush()
        name = (await session.execute(select(Tag.name).where(Tag.id == tag_id))).scalar_one_or_none()
        if name:
            tag_names[memory_id].append(name)
        counts["memory_tags"] += 1
    return tag_names


async def _apply_memory_sources(root: Path, session: AsyncSession, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "memory_sources"):
        kwargs: dict[str, Any] = {
            "memory_id": _required_uuid(row, "memory_id"),
            "source_type": row["source_type"],
            "source_ref": row.get("source_ref"),
            "agent_id": None,
            "created_at": _maybe_datetime(row.get("created_at")),
            "evidence_span": row.get("evidence_span"),
        }
        if row.get("id") is not None:
            kwargs["id"] = int(row["id"])
        session.add(MemorySource(**kwargs))
        await session.flush()
        counts["memory_sources"] += 1


async def _apply_tasks(
    root: Path, session: AsyncSession, env_id: UUID, ctx: AgentContext, counts: defaultdict[str, int]
) -> None:
    async for row in _rows(root, "tasks"):
        session.add(
            Task(
                id=_required_uuid(row, "id"),
                env_id=env_id,
                title=row["title"],
                description=row.get("description"),
                status=row.get("status", "pending"),
                priority=int(row.get("priority", 50)),
                playbook_id=_maybe_uuid(row.get("playbook_id")),
                version=int(row.get("version", 1)),
                created_at=_maybe_datetime(row.get("created_at")),
                updated_at=_maybe_datetime(row.get("updated_at")),
                created_by_agent_id=ctx.agent_id,
            )
        )
        await session.flush()
        counts["tasks"] += 1


async def _apply_graph_nodes(root: Path, session: AsyncSession, env_id: UUID, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "graph_nodes"):
        session.add(
            GraphNode(
                id=_required_uuid(row, "id"),
                env_id=env_id,
                node_type=row["node_type"],
                memory_id=_maybe_uuid(row.get("memory_id")),
                entity_id=_maybe_uuid(row.get("entity_id")),
                task_id=_maybe_uuid(row.get("task_id")),
                created_at=_maybe_datetime(row.get("created_at")),
            )
        )
        await session.flush()
        counts["graph_nodes"] += 1


async def _apply_relations(root: Path, session: AsyncSession, env_id: UUID, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "relations"):
        session.add(
            Relation(
                id=_required_uuid(row, "id"),
                env_id=env_id,
                src_node_id=_required_uuid(row, "src_node_id"),
                dst_node_id=_required_uuid(row, "dst_node_id"),
                type=row["type"],
                properties=_json_obj(row.get("properties")),
                created_at=_maybe_datetime(row.get("created_at")),
                updated_at=_maybe_datetime(row.get("updated_at")),
                version=int(row.get("version", 1)),
            )
        )
        await session.flush()
        counts["relations"] += 1


async def _apply_memory_lineage(root: Path, session: AsyncSession, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "memory_lineage"):
        session.add(
            MemoryLineage(
                parent_memory_id=_required_uuid(row, "parent_memory_id"),
                child_memory_id=_required_uuid(row, "child_memory_id"),
                relation=row["relation"],
                created_at=_maybe_datetime(row.get("created_at")),
            )
        )
        await session.flush()
        counts["memory_lineage"] += 1


async def _apply_external_memory_lineage(
    root: Path,
    session: AsyncSession,
    env_id: UUID,
    counts: defaultdict[str, int],
) -> None:
    _ = env_id
    async for row in _rows(root, "external_memory_lineage"):
        if not await _memories_exist(
            session, _required_uuid(row, "parent_memory_id"), _required_uuid(row, "child_memory_id")
        ):
            continue
        session.add(
            MemoryLineage(
                parent_memory_id=_required_uuid(row, "parent_memory_id"),
                child_memory_id=_required_uuid(row, "child_memory_id"),
                relation=row["relation"],
                created_at=_maybe_datetime(row.get("created_at")),
            )
        )
        await session.flush()
        counts["external_memory_lineage"] += 1


async def _apply_dream_runs(root: Path, session: AsyncSession, env_id: UUID, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "dream_runs"):
        session.add(
            DreamRun(
                id=_required_uuid(row, "id"),
                env_id=env_id,
                mode=row["mode"],
                status=row.get("status", "running"),
                started_at=_maybe_datetime(row.get("started_at")),
                ended_at=_maybe_datetime(row.get("ended_at")),
                triggered_by=row.get("triggered_by", "restore"),
                summarizer_kind=row.get("summarizer_kind"),
                summary=_json_obj(row.get("summary")),
                last_error=row.get("last_error"),
            )
        )
        await session.flush()
        counts["dream_runs"] += 1


async def _apply_dream_proposals(
    root: Path, session: AsyncSession, env_id: UUID, counts: defaultdict[str, int]
) -> None:
    async for row in _rows(root, "dream_proposals"):
        session.add(
            DreamProposal(
                id=_required_uuid(row, "id"),
                env_id=env_id,
                kind=row["kind"],
                status=row.get("status", "open"),
                payload=_json_obj(row.get("payload")),
                summarizer_kind=row.get("summarizer_kind"),
                llm_failed=bool(row.get("llm_failed", False)),
                dedupe_key=row.get("dedupe_key"),
                dream_run_id=_maybe_uuid(row.get("dream_run_id")),
                created_at=_maybe_datetime(row.get("created_at")),
                updated_at=_maybe_datetime(row.get("updated_at")),
                reviewed_at=_maybe_datetime(row.get("reviewed_at")),
                reviewed_by_agent_id=None,
                review_action=row.get("review_action"),
                review_notes=row.get("review_notes"),
            )
        )
        await session.flush()
        counts["dream_proposals"] += 1


async def _apply_grants(root: Path, session: AsyncSession, env_id: UUID, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "grants"):
        if row.get("agent_id") is None:
            continue
        session.add(
            EnvGrant(
                env_id=env_id,
                agent_id=_required_uuid(row, "agent_id"),
                role=row["role"],
                granted_at=_maybe_datetime(row.get("granted_at")),
            )
        )
        await session.flush()
        counts["grants"] += 1


async def _apply_superseded_by(root: Path, session: AsyncSession, counts: defaultdict[str, int]) -> None:
    async for row in _rows(root, "memories"):
        target = _maybe_uuid(row.get("superseded_by"))
        if target is None:
            continue
        await session.execute(
            update(Memory)
            .where(Memory.id == _required_uuid(row, "id"))
            .values(status=row.get("status", "superseded"), superseded_by=target)
        )
        counts["memories_superseded_updates"] += 1


async def _restore_embeddings(
    root: Path,
    *,
    env_id: UUID,
    target_model_id: str,
    inserted_memories: Mapping[UUID, Memory],
    tag_names: Mapping[UUID, list[str]],
    vector_store_factory: VectorStoreFactory | None = None,
) -> int:
    path = root / "embeddings" / "memory_vectors.jsonl"
    if not path.is_file():
        return 0
    pending = 0
    store: VectorStore | None = None
    try:
        async for raw in JsonlReader(path):
            record = MemoryVectorRecord.model_validate(raw)
            memory = inserted_memories.get(record.memory_id)
            if memory is None or memory.version != record.memory_version or record.model_id != target_model_id:
                pending += 1
                continue
            if store is None:
                store = vector_store_factory() if vector_store_factory else _default_vector_store()
            await store.ensure_env_collection(env_id=env_id, dimension=record.dimension)
            await store.upsert(
                env_id=env_id,
                point_id=record.memory_id,
                vector={record.vector_name: record.vector},
                payload=_projection_payload(
                    memory, tag_names=tag_names.get(record.memory_id, []), embedding_model_id=target_model_id
                ),
            )
    finally:
        if store is not None:
            await store.close()
    return pending


def _default_vector_store() -> VectorStore:
    from memory_mcp.db.vector.qdrant import QdrantVectorStore

    return QdrantVectorStore(get_settings())


async def _augment_archive_with_external_lineage(archive_path: Path, env_id: UUID) -> None:
    async with session_scope() as session:
        memory_ids = set((await session.execute(select(Memory.id).where(Memory.env_id == env_id))).scalars().all())
        rows = await _external_lineage_rows(session, memory_ids)
    if not rows:
        return
    scratch = Path(".tmp") / "env-snapshot-augment"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="archive-", dir=scratch) as tmp_name:
        root = Path(tmp_name)
        _extract_archive(archive_path, root)
        with JsonlWriter(root / "external_memory_lineage.jsonl") as writer:
            for row in rows:
                writer.write(row)
        with tarfile.open(archive_path, mode="w:gz") as tar:
            for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
                tar.add(file_path, arcname=file_path.relative_to(root))


async def _snapshot_tree_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


async def _rows(root: Path, table: str) -> AsyncIterator[dict[str, Any]]:
    path = root / f"{table}.jsonl"
    if not path.is_file():
        return
    async for row in JsonlReader(path):
        yield row


def _maybe_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _row(obj: object, names: list[str]) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in names}


# Import is intentionally late to keep the in-place path separate from import_env.
from memory_mcp.env_ops.import_ import import_env  # noqa: E402

__all__ = ["create_snapshot", "restore_snapshot"]
