"""Environment export implementation for v0.8 env operations."""

from __future__ import annotations

import json
import shutil
import tarfile
import tomllib
from contextlib import suppress
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import exists, func, inspect, select
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
    Tag,
    Task,
)
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.db.vector.qdrant import QdrantVectorStore
from memory_mcp.errors import MemoryMCPError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp_schemas.env_ops import (
    EnvExportRequest,
    EnvExportResponse,
    ExportFormat,
    ExportManifest,
    IncludeFlags,
    MemoryVectorRecord,
    SourceMetadata,
)

from ._checksums import sha256_file, write_checksums_file
from ._io import JsonlWriter

SCHEMA_VERSION = "0.8.0"
_HIDDEN_VECTOR_STATUSES = {"archived", "retired", "superseded"}


class ConflictError(MemoryMCPError):
    """Export target conflicts with an existing non-empty path."""

    code = "EXPORT_TARGET_NOT_EMPTY"


async def export_env(request: EnvExportRequest, *, ctx: AgentContext) -> EnvExportResponse:
    """Implementation of the env_export tool. See plan.md §4 + §17."""

    target_dir = _prepare_target_dir(request)
    counts: dict[str, int] = {}
    counts_by_kind: dict[str, int] = {}

    async with session_scope() as session:
        await _set_repeatable_read(session)
        env = await _load_environment(session, request.env_id)
        rbac.require("read", request.env_id, ctx)

        counts["env"] = _write_single_row(target_dir / "env.json", env)
        counts["memories"] = await _stream_table(
            session,
            select(Memory).where(Memory.env_id == request.env_id),
            target_dir / "memories.jsonl",
            request.chunk_size,
        )
        counts_by_kind = await _count_memories_by_kind(session, request.env_id)
        counts["memory_tags"] = await _stream_table(
            session,
            select(MemoryTag).where(MemoryTag.env_id == request.env_id),
            target_dir / "memory_tags.jsonl",
            request.chunk_size,
        )
        counts["tags"] = await _stream_table(
            session,
            select(Tag).where(Tag.env_id == request.env_id),
            target_dir / "tags.jsonl",
            request.chunk_size,
        )
        counts["entities"] = await _stream_table(
            session,
            select(Entity).where(Entity.env_id == request.env_id),
            target_dir / "entities.jsonl",
            request.chunk_size,
        )
        counts["entity_aliases"] = await _stream_table(
            session,
            select(EntityAlias).where(EntityAlias.env_id == request.env_id),
            target_dir / "entity_aliases.jsonl",
            request.chunk_size,
        )
        counts["relations"] = await _stream_table(
            session,
            select(Relation).where(Relation.env_id == request.env_id),
            target_dir / "relations.jsonl",
            request.chunk_size,
        )
        counts["graph_nodes"] = await _stream_table(
            session,
            select(GraphNode).where(GraphNode.env_id == request.env_id),
            target_dir / "graph_nodes.jsonl",
            request.chunk_size,
        )
        counts["tasks"] = await _stream_table(
            session,
            select(Task).where(Task.env_id == request.env_id),
            target_dir / "tasks.jsonl",
            request.chunk_size,
            transform=_null_agent_fields("created_by_agent_id"),
        )
        counts["memory_lineage"] = await _stream_table(
            session,
            _memory_lineage_stmt(request.env_id),
            target_dir / "memory_lineage.jsonl",
            request.chunk_size,
        )

        if request.include_provenance:
            counts["memory_sources"] = await _stream_table(
                session,
                select(MemorySource)
                .join(Memory, MemorySource.memory_id == Memory.id)
                .where(Memory.env_id == request.env_id),
                target_dir / "memory_sources.jsonl",
                request.chunk_size,
                transform=_null_agent_fields("agent_id", "created_by_agent_id"),
            )
        if request.include_grants:
            counts["grants"] = await _stream_table(
                session,
                select(EnvGrant).where(EnvGrant.env_id == request.env_id),
                target_dir / "grants.jsonl",
                request.chunk_size,
                transform=_null_agent_fields("agent_id"),
            )
        if request.include_dream_history:
            counts["dream_runs"] = await _stream_table(
                session,
                select(DreamRun).where(DreamRun.env_id == request.env_id),
                target_dir / "dream_runs.jsonl",
                request.chunk_size,
            )
            counts["dream_proposals"] = await _stream_table(
                session,
                select(DreamProposal).where(DreamProposal.env_id == request.env_id),
                target_dir / "dream_proposals.jsonl",
                request.chunk_size,
                transform=_null_agent_fields("reviewed_by_agent_id"),
            )

        if request.include_embeddings:
            vector_counts = await _export_memory_vectors(
                session,
                env_id=request.env_id,
                model_id=env.default_embedding_model_id,
                target_dir=target_dir,
                chunk_size=request.chunk_size,
            )
            counts.update(vector_counts)

        manifest = ExportManifest(
            memory_mcp_version=_read_project_version(),
            source=SourceMetadata(
                env_id=env.id,
                env_name=env.name,
                default_embedding_model_id=env.default_embedding_model_id,
                # TODO(envops): replace with sha256 of instance bootstrap config when the config source is stable.
                instance_fingerprint="unknown",
            ),
            exported_at=datetime.now(UTC),
            exported_by_agent=str(ctx.agent_id) if getattr(ctx, "agent_id", None) else None,
            include_flags=IncludeFlags(
                embeddings=request.include_embeddings,
                provenance=request.include_provenance,
                dream_history=request.include_dream_history,
                grants=request.include_grants,
            ),
            counts=counts,
            checksums={},
        )

    checksums = await _write_manifest_and_checksums(target_dir, manifest)
    manifest = manifest.model_copy(update={"checksums": checksums})
    _write_manifest(target_dir / "manifest.json", manifest)
    await write_checksums_file(await _compute_checksums(target_dir), target_dir / "checksums.sha256")

    output_path = target_dir
    if request.format == ExportFormat.archive:
        output_path = await _archive_and_remove(target_dir)

    return EnvExportResponse(
        manifest=manifest,
        output_path=str(output_path),
        byte_size=_byte_size(output_path),
        counts_by_kind=counts_by_kind,
    )


async def _set_repeatable_read(session: AsyncSession) -> None:
    try:
        await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
    except AttributeError:
        return


async def _load_environment(session: AsyncSession, env_id: UUID) -> Environment:
    env = (await session.execute(select(Environment).where(Environment.id == env_id))).scalars().first()
    if env is None:
        raise NotFoundError(f"environment {env_id} not found", env_id=str(env_id))
    if getattr(env, "status", "active") == "deleted":
        exc = NotFoundError(f"environment {env_id} is deleted", env_id=str(env_id))
        exc.code = "ENV_DELETED"
        raise exc
    return env


def _prepare_target_dir(request: EnvExportRequest) -> Path:
    target_dir = Path(request.target_path).expanduser()
    if request.format == ExportFormat.archive and Path(f"{target_dir}.tar.gz").exists():
        raise ConflictError(f"archive target already exists: {target_dir}.tar.gz", path=f"{target_dir}.tar.gz")
    if target_dir.exists() and not target_dir.is_dir():
        raise ConflictError(f"export target is not a directory: {target_dir}", path=str(target_dir))
    target_dir.mkdir(parents=True, exist_ok=True)
    if any(target_dir.iterdir()):
        raise ConflictError(f"export target directory is not empty: {target_dir}", path=str(target_dir))
    return target_dir


async def _stream_table(
    session: AsyncSession,
    stmt: Any,
    path: Path,
    chunk_size: int,
    *,
    transform: Any | None = None,
) -> int:
    count = 0
    offset = 0
    ordered = _with_primary_key_order(stmt)
    with JsonlWriter(path) as writer:
        while True:
            rows = (await session.execute(ordered.limit(chunk_size).offset(offset))).scalars().all()
            if not rows:
                break
            for row in rows:
                payload = _row_dict(row)
                if transform is not None:
                    payload = transform(payload)
                writer.write(payload)
            count += len(rows)
            if len(rows) < chunk_size:
                break
            offset += chunk_size
    return count


def _with_primary_key_order(stmt: Any) -> Any:
    descriptions = getattr(stmt, "column_descriptions", None) or []
    entity = descriptions[0].get("entity") if descriptions else None
    if entity is None or not hasattr(entity, "__mapper__"):
        return stmt
    return stmt.order_by(*entity.__mapper__.primary_key)


def _write_single_row(path: Path, row: Any) -> int:
    with JsonlWriter(path) as writer:
        writer.write(_row_dict(row))
    return 1


def _row_dict(row: Any) -> dict[str, Any]:
    mapper = inspect(row).mapper
    payload: dict[str, Any] = {}
    for column in mapper.columns:
        attr_key = mapper.get_property_by_column(column).key
        payload[column.name] = getattr(row, attr_key)
    return payload


def _null_agent_fields(*field_names: str):
    def transform(payload: dict[str, Any]) -> dict[str, Any]:
        for field_name in field_names:
            if field_name in payload:
                payload[field_name] = None
        return payload

    return transform


def _memory_lineage_stmt(env_id: UUID) -> Any:
    parent_in_env = exists(select(Memory.id).where(Memory.id == MemoryLineage.parent_memory_id, Memory.env_id == env_id))
    child_in_env = exists(select(Memory.id).where(Memory.id == MemoryLineage.child_memory_id, Memory.env_id == env_id))
    return select(MemoryLineage).where(parent_in_env, child_in_env)


async def _count_memories_by_kind(session: AsyncSession, env_id: UUID) -> dict[str, int]:
    rows = (
        await session.execute(
            select(Memory.kind, func.count()).where(Memory.env_id == env_id).group_by(Memory.kind)
        )
    ).all()
    return {str(kind): int(count) for kind, count in rows}


async def _export_memory_vectors(
    session: AsyncSession,
    *,
    env_id: UUID,
    model_id: str,
    target_dir: Path,
    chunk_size: int,
) -> dict[str, int]:
    embeddings_dir = target_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    path = embeddings_dir / "memory_vectors.jsonl"
    vector_store = _default_vector_store()
    written = 0
    skipped = 0
    offset = 0

    try:
        with JsonlWriter(path) as writer:
            while True:
                memories = (
                    await session.execute(
                        select(Memory)
                        .where(Memory.env_id == env_id)
                        .order_by(Memory.id)
                        .limit(chunk_size)
                        .offset(offset)
                    )
                ).scalars().all()
                if not memories:
                    break

                visible = [m for m in memories if m.status not in _HIDDEN_VECTOR_STATUSES]
                skipped += len(memories) - len(visible)
                body_vectors = await _safe_get_vectors(vector_store, env_id, [m.id for m in visible], "body")
                trigger_ids = [m.id for m in visible if getattr(m, "trigger_description", None)]
                trigger_vectors = await _safe_get_vectors(vector_store, env_id, trigger_ids, "trigger")

                for memory in visible:
                    vector = body_vectors.get(memory.id)
                    if vector:
                        writer.write(_vector_record(memory, model_id, "body", vector))
                        written += 1
                    else:
                        skipped += 1
                    trigger_vector = trigger_vectors.get(memory.id)
                    if trigger_vector:
                        writer.write(_vector_record(memory, model_id, "trigger", trigger_vector))
                        written += 1

                if len(memories) < chunk_size:
                    break
                offset += chunk_size
    finally:
        with suppress(Exception):
            await vector_store.close()

    return {"memory_vectors": written, "memory_vectors_skipped": skipped}


async def _safe_get_vectors(
    vector_store: VectorStore,
    env_id: UUID,
    ids: list[UUID],
    vector_name: str,
) -> dict[UUID, list[float] | None]:
    if not ids:
        return {}
    try:
        return await vector_store.get_vectors(env_id=env_id, ids=ids, vector_name=vector_name)
    except Exception:
        return {memory_id: None for memory_id in ids}


def _vector_record(memory: Memory, model_id: str, vector_name: str, vector: Sequence[float]) -> MemoryVectorRecord:
    return MemoryVectorRecord(
        memory_id=memory.id,
        memory_version=memory.version,
        model_id=model_id,
        vector_name=vector_name,  # type: ignore[arg-type]
        dimension=len(vector),
        vector=list(vector),
    )


def _default_vector_store() -> VectorStore:
    return QdrantVectorStore(get_settings())


async def _write_manifest_and_checksums(target_dir: Path, manifest: ExportManifest) -> dict[str, str]:
    _write_manifest(target_dir / "manifest.json", manifest)
    checksums = await _compute_checksums(target_dir, exclude={"manifest.json"})
    await write_checksums_file(checksums, target_dir / "checksums.sha256")
    return checksums


def _write_manifest(path: Path, manifest: ExportManifest) -> None:
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


async def _compute_checksums(target_dir: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    checksums: dict[str, str] = {}
    for file_path in sorted(path for path in target_dir.rglob("*") if path.is_file()):
        rel_path = file_path.relative_to(target_dir).as_posix()
        if rel_path == "checksums.sha256" or rel_path in excluded:
            continue
        checksums[rel_path] = await sha256_file(file_path)
    return checksums


async def _archive_and_remove(target_dir: Path) -> Path:
    archive_path = Path(f"{target_dir}.tar.gz")
    with tarfile.open(archive_path, mode="w:gz") as tar:
        for file_path in sorted(path for path in target_dir.rglob("*") if path.is_file()):
            tar.add(file_path, arcname=file_path.relative_to(target_dir))
    with tarfile.open(archive_path, mode="r:gz") as tar:
        if not tar.getmembers():
            raise RuntimeError(f"archive verification failed: {archive_path} is empty")
    shutil.rmtree(target_dir)
    return archive_path


def _byte_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _read_project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as f:
            payload = tomllib.load(f)
        version = payload.get("project", {}).get("version")
        return str(version) if version else "0.8.0-dev"
    except Exception:
        return "0.8.0-dev"


__all__ = ["ConflictError", "export_env"]
