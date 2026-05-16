"""Import implementation for v0.8 environment operations."""

from __future__ import annotations

import datetime as dt
import json
import logging
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp.config import get_settings
from memory_mcp.db.models import (
    DreamProposal,
    DreamRun,
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
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.env_ops._checksums import verify_checksums_file
from memory_mcp.env_ops._embed import maybe_re_embed
from memory_mcp.env_ops._io import JsonlReader
from memory_mcp.errors import MemoryMCPError, NotFoundError
from memory_mcp.envs import get_env_by_id
from memory_mcp.entities import entity_merge
from memory_mcp.identity import AgentContext
from memory_mcp.memories import _projection_payload
from memory_mcp_schemas.entities import EntityMergeRequest
from memory_mcp_schemas.env_ops import (
    ArchiveVersionDecision,
    EnvImportReport,
    EnvImportRequest,
    ExportManifest,
    ImportMode,
    MemoryVectorRecord,
    RemapTable,
)

SCHEMA_VERSION = "0.8.0"
_REEMBED_SYNC_LIMIT = 10_000

log = logging.getLogger(__name__)

T = TypeVar("T")
VectorStoreFactory = Callable[[], VectorStore]


class ChecksumMismatchError(MemoryMCPError):
    """Archive checksum validation failed."""

    code = "IMPORT_CHECKSUM_MISMATCH"


class ArchiveVersionError(MemoryMCPError):
    """Archive schema version is not import-compatible with this server."""

    code = "IMPORT_ARCHIVE_VERSION"


class BulkReembedBlocked(MemoryMCPError):
    """Import would trigger a large inline re-embedding run."""

    code = "IMPORT_BULK_REEMBED_BLOCKED"


async def import_env(request: EnvImportRequest, *, ctx: AgentContext) -> EnvImportReport:
    """Implementation of the env_import tool. See plan.md §5 + §17.6.

    ``mode=overwrite`` is destructive, so it requires the forward-compatible
    RBAC admin gate. In v1 that helper is local-only/no-op; deployments can
    monkey-patch or v1.5 can enforce it centrally without changing this code.
    """

    if request.mode == ImportMode.overwrite:
        rbac.require("admin", env_id=None, ctx=ctx)
    if (request.target_env_name is None) == (request.target_env_id is None):
        raise ValueError("exactly one of target_env_name or target_env_id must be set")

    source = Path(request.source_path)
    tmp: tempfile.TemporaryDirectory[str] | None = None
    try:
        root = _open_source(source)
        if root is None:
            tmp_root = Path(".tmp") / "env-import"
            tmp_root.mkdir(parents=True, exist_ok=True)
            tmp = tempfile.TemporaryDirectory(prefix="archive-", dir=tmp_root)
            root = Path(tmp.name)
            _extract_archive(source, root)

        manifest = await _load_verified_manifest(root)
        decision = _version_decision(manifest.schema_version)
        if decision in {
            ArchiveVersionDecision.reject_too_old,
            ArchiveVersionDecision.reject_too_new,
        }:
            raise ArchiveVersionError(
                f"archive schema_version {manifest.schema_version!r} is not compatible with {SCHEMA_VERSION!r}: "
                f"{decision.value}",
                archive_schema_version=manifest.schema_version,
                current_schema_version=SCHEMA_VERSION,
                decision=decision.value,
            )

        remap = await _build_remap(root)
        target = await _resolve_target(request, manifest=manifest, ctx=ctx)
        needs_reembed = (
            request.re_embed_if_model_mismatch
            and manifest.source.default_embedding_model_id != target.default_embedding_model_id
        )
        memory_count = manifest.counts.get("memories")
        if memory_count is None:
            memory_count = await _count_jsonl(root / "memories.jsonl")
        if needs_reembed and memory_count > _REEMBED_SYNC_LIMIT and not request.allow_bulk_reembed:
            raise BulkReembedBlocked(
                f"Import would re-embed an estimated {memory_count} memories; set allow_bulk_reembed=True to accept "
                "pricing and rate-limit risk.",
                estimated_count=memory_count,
            )

        if request.dry_run:
            return EnvImportReport(
                target_env_id=target.id,
                dry_run=True,
                mode=request.mode,
                counts=dict(manifest.counts),
                conflicts={table: 0 for table in manifest.counts},
                sample_conflicts={},
                remap_table_size=_remap_size(remap),
                archive_version_decision=decision,
                pending_vector_rebuild=0,
                re_embed_count=0,
            )

        state = _ImportState(target_env_id=target.id, remap=remap, mode=request.mode)
        async with session_scope() as session:
            if request.target_env_name is not None:
                session.add(
                    Environment(
                        id=target.id,
                        name=request.target_env_name,
                        default_embedding_model_id=manifest.source.default_embedding_model_id,
                        kind=None,
                        retention_policy={},
                    )
                )
                await session.flush()
            await _apply_rows(root, session=session, ctx=ctx, manifest=manifest, state=state)
        await _perform_entity_merges(state, ctx=ctx)

        pending_vector_rebuild, re_embed_count = await _apply_embeddings(
            root,
            manifest=manifest,
            target_model_id=target.default_embedding_model_id,
            re_embed_if_model_mismatch=request.re_embed_if_model_mismatch,
            state=state,
        )

        return EnvImportReport(
            target_env_id=target.id,
            dry_run=False,
            mode=request.mode,
            counts=dict(state.counts),
            conflicts=dict(state.conflicts),
            sample_conflicts={k: v[:5] for k, v in state.sample_conflicts.items()},
            remap_table_size=_remap_size(remap),
            archive_version_decision=decision,
            pending_vector_rebuild=pending_vector_rebuild,
            re_embed_count=re_embed_count,
            entity_merges_performed=state.entity_merges_performed,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


def _open_source(source: Path) -> Path | None:
    if source.is_file() or str(source).endswith(".tar.gz"):
        return None
    manifest = source / "manifest.json"
    checksums = source / "checksums.sha256"
    if not manifest.is_file() or not checksums.is_file():
        missing = [p.name for p in (manifest, checksums) if not p.is_file()]
        raise FileNotFoundError(f"missing import archive file(s): {', '.join(missing)}")
    return source


def _extract_archive(source: Path, target: Path) -> None:
    with tarfile.open(source, mode="r:gz") as archive:
        base = target.resolve()
        for member in archive.getmembers():
            destination = (target / member.name).resolve()
            if base != destination and base not in destination.parents:
                raise ValueError(f"unsafe archive member path: {member.name!r}")
        archive.extractall(target)

    children = [child for child in target.iterdir() if child.is_dir()]
    if not (target / "manifest.json").is_file() and len(children) == 1:
        for child in children[0].iterdir():
            child.rename(target / child.name)
        children[0].rmdir()


async def _load_verified_manifest(root: Path) -> ExportManifest:
    if not await verify_checksums_file(root / "checksums.sha256", root):
        raise ChecksumMismatchError("import archive checksum validation failed")
    raw = await _read_json(root / "manifest.json")
    raw_version = str(raw.get("schema_version", ""))
    decision = _version_decision(raw_version)
    if decision in {
        ArchiveVersionDecision.reject_too_old,
        ArchiveVersionDecision.reject_too_new,
    }:
        return ExportManifest.model_construct(  # type: ignore[call-arg]
            schema_version=raw_version,
            memory_mcp_version=str(raw.get("memory_mcp_version", "")),
            source=raw.get("source"),
            exported_at=raw.get("exported_at"),
            exported_by_agent=raw.get("exported_by_agent"),
            include_flags=raw.get("include_flags"),
            counts=raw.get("counts", {}),
            checksums=raw.get("checksums", {}),
        )
    try:
        return ExportManifest.model_validate(raw)
    except ValidationError as exc:
        raise ArchiveVersionError("archive manifest is invalid", errors=exc.errors()) from exc


async def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(await _to_thread(path.read_text, encoding="utf-8"))


async def _to_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


def _version_decision(version: str) -> ArchiveVersionDecision:
    current_major, current_minor, _ = _parse_version(SCHEMA_VERSION)
    major, minor, _patch = _parse_version(version)
    if (major, minor) == (current_major, current_minor):
        return ArchiveVersionDecision.accept
    if major > current_major or (major == current_major and minor > current_minor):
        return ArchiveVersionDecision.reject_too_new
    if major < current_major or minor < current_minor:
        return (
            ArchiveVersionDecision.accept_with_migration
            if _has_migration(version, SCHEMA_VERSION)
            else ArchiveVersionDecision.reject_too_old
        )
    return ArchiveVersionDecision.reject_too_new


def _parse_version(version: str) -> tuple[int, int, int]:
    try:
        major, minor, patch = version.split(".", 2)
        return int(major), int(minor), int(patch)
    except ValueError as exc:
        raise ArchiveVersionError(f"invalid archive schema_version {version!r}") from exc


def _has_migration(source_version: str, target_version: str) -> bool:
    migrations = Path(__file__).with_name("_migrations")
    return (migrations / f"{source_version}_to_{target_version}.py").is_file()


class _TargetEnv:
    def __init__(self, *, id: UUID, default_embedding_model_id: str) -> None:
        self.id = id
        self.default_embedding_model_id = default_embedding_model_id


async def _resolve_target(
    request: EnvImportRequest,
    *,
    manifest: ExportManifest,
    ctx: AgentContext,
) -> _TargetEnv:
    if request.target_env_name is not None:
        rbac.require("admin", env_id=None, ctx=ctx)
        return _TargetEnv(
            id=uuid4(),
            default_embedding_model_id=manifest.source.default_embedding_model_id,
        )

    assert request.target_env_id is not None
    row = await get_env_by_id(request.target_env_id, include_deleted=False)
    if row is None:
        raise NotFoundError(f"environment {request.target_env_id} not found", env_id=str(request.target_env_id))
    rbac.require("write", env_id=row.id, ctx=ctx)
    return _TargetEnv(id=row.id, default_embedding_model_id=row.default_embedding_model_id)


async def _build_remap(root: Path) -> RemapTable:
    remap = RemapTable()
    for table, attr in {
        "tags": "tags",
        "entities": "entities",
        "memories": "memories",
        "graph_nodes": "graph_nodes",
        "relations": "relations",
        "tasks": "tasks",
        "dream_runs": "dream_runs",
        "dream_proposals": "dream_proposals",
    }.items():
        path = root / f"{table}.jsonl"
        if not path.is_file():
            continue
        mapping: dict[UUID, UUID] = getattr(remap, attr)
        async for row in JsonlReader(path):
            source_id = _maybe_uuid(row.get("id"))
            if source_id is not None:
                mapping[source_id] = uuid4()
    return remap


async def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    async for _ in JsonlReader(path):
        count += 1
    return count


class _ImportState:
    def __init__(self, *, target_env_id: UUID, remap: RemapTable, mode: ImportMode) -> None:
        self.target_env_id = target_env_id
        self.remap = remap
        self.mode = mode
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.conflicts: defaultdict[str, int] = defaultdict(int)
        self.sample_conflicts: defaultdict[str, list[str]] = defaultdict(list)
        self.inserted_memories: dict[UUID, Memory] = {}
        self.memory_tags: defaultdict[UUID, list[str]] = defaultdict(list)
        self.pending_entity_merges: list[tuple[UUID, UUID]] = []
        self.entity_merges_performed = 0

    def conflict(self, table: str, source_id: object) -> None:
        self.conflicts[table] += 1
        if len(self.sample_conflicts[table]) < 5:
            self.sample_conflicts[table].append(str(source_id))


async def _apply_rows(
    root: Path,
    *,
    session: AsyncSession,
    ctx: AgentContext,
    manifest: ExportManifest,
    state: _ImportState,
) -> None:
    await _apply_tags(root, session, state)
    await _apply_entities(root, session, state)
    await _apply_entity_aliases(root, session, state)
    await _apply_memories(root, session, ctx, state)
    await _apply_memory_tags(root, session, state)
    await _apply_memory_sources(root, session, state)
    await _apply_graph_nodes(root, session, state)
    await _apply_relations(root, session, state)
    await _apply_memory_lineage(root, session, state)
    await _apply_tasks(root, session, ctx, state)
    await _apply_superseded_by(root, session, state)
    if manifest.include_flags.dream_history:
        await _apply_dream_runs(root, session, state)
        await _apply_dream_proposals(root, session, state)
    if (root / "grants.jsonl").is_file() and manifest.include_flags.grants:
        log.warning("env_import: grants import is skipped in v0.8 Phase 2")


async def _apply_tags(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "tags"):
        old_id = _required_uuid(row, "id")
        new_id = state.remap.tags[old_id]
        if state.mode == ImportMode.skip and await _tag_exists(session, state.target_env_id, str(row["name"])):
            state.remap.tags.pop(old_id, None)
            state.conflict("tags", old_id)
            continue
        existing_id = await _tag_id_by_name(session, state.target_env_id, str(row["name"]))
        if existing_id is not None and state.mode == ImportMode.merge:
            state.remap.tags[old_id] = existing_id
            state.conflict("tags", old_id)
            continue
        if existing_id is not None and state.mode == ImportMode.overwrite:
            await session.execute(delete(Tag).where(Tag.id == existing_id))
            await session.flush()
            state.conflict("tags", old_id)
        await _add(session, Tag(id=new_id, env_id=state.target_env_id, name=row["name"]), state, "tags", old_id)


async def _apply_entities(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "entities"):
        old_id = _required_uuid(row, "id")
        new_id = state.remap.entities[old_id]
        normalized = str(row["normalized_name"])
        if state.mode == ImportMode.skip and await _entity_exists(session, state.target_env_id, normalized):
            state.remap.entities.pop(old_id, None)
            state.conflict("entities", old_id)
            continue
        existing_id = await _entity_id_by_normalized(session, state.target_env_id, normalized)
        if existing_id is not None and state.mode == ImportMode.overwrite:
            await session.execute(delete(Entity).where(Entity.id == existing_id))
            await session.flush()
            state.conflict("entities", old_id)
        if existing_id is not None and state.mode == ImportMode.merge:
            state.pending_entity_merges.append((existing_id, new_id))
            state.conflict("entities", old_id)
            normalized = f"{normalized}__import_merge__{new_id}"
        obj = Entity(
            id=new_id,
            env_id=state.target_env_id,
            kind=row["kind"],
            canonical_name=row["canonical_name"],
            normalized_name=normalized,
            metadata_=_json_obj(row.get("metadata") or row.get("metadata_")),
            version=int(row.get("version", 1)),
        )
        await _add(session, obj, state, "entities", old_id)


async def _apply_entity_aliases(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "entity_aliases"):
        old_entity_id = _required_uuid(row, "entity_id")
        entity_id = state.remap.entities.get(old_entity_id)
        if entity_id is None:
            continue
        normalized = str(row["normalized_alias"])
        if state.mode in {ImportMode.skip, ImportMode.merge} and await _entity_alias_exists(
            session,
            state.target_env_id,
            normalized,
        ):
            state.conflict("entity_aliases", old_entity_id)
            continue
        obj = EntityAlias(
            entity_id=entity_id,
            env_id=state.target_env_id,
            alias=row["alias"],
            normalized_alias=normalized,
        )
        await _add(session, obj, state, "entity_aliases", old_entity_id)


async def _apply_memories(root: Path, session: AsyncSession, ctx: AgentContext, state: _ImportState) -> None:
    async for row in _rows(root, "memories"):
        old_id = _required_uuid(row, "id")
        new_id = state.remap.memories[old_id]
        source_status = row.get("status", "active")
        insert_status = "active" if source_status == "superseded" and row.get("superseded_by") else source_status
        obj = Memory(
            id=new_id,
            env_id=state.target_env_id,
            kind=row["kind"],
            status=insert_status,
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
            expires_at=_maybe_datetime(row.get("expires_at")),
            superseded_by=None,
            metadata_=_json_obj(row.get("metadata") or row.get("metadata_")),
            decision_meta=_json_nullable(row.get("decision_meta")),
            version=int(row.get("version", 1)),
        )
        try:
            await _add(session, obj, state, "memories", old_id)
        except IntegrityError:
            state.remap.memories.pop(old_id, None)
            raise
        state.inserted_memories[old_id] = obj


async def _apply_memory_tags(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "memory_tags"):
        memory_id = state.remap.memories.get(_required_uuid(row, "memory_id"))
        tag_id = state.remap.tags.get(_required_uuid(row, "tag_id"))
        if memory_id is None or tag_id is None:
            continue
        tag_name = await _tag_name(session, tag_id)
        if tag_name is not None:
            state.memory_tags[memory_id].append(tag_name)
        await _add(
            session,
            MemoryTag(memory_id=memory_id, tag_id=tag_id, env_id=state.target_env_id),
            state,
            "memory_tags",
            f"{memory_id}:{tag_id}",
        )


async def _apply_memory_sources(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "memory_sources"):
        memory_id = state.remap.memories.get(_required_uuid(row, "memory_id"))
        if memory_id is None:
            continue
        obj = MemorySource(
            memory_id=memory_id,
            source_type=row["source_type"],
            source_ref=row.get("source_ref"),
            agent_id=None,
            evidence_span=row.get("evidence_span"),
        )
        await _add(session, obj, state, "memory_sources", row.get("id", memory_id))


async def _apply_graph_nodes(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "graph_nodes"):
        old_id = _required_uuid(row, "id")
        memory_id = _remapped_optional(state.remap.memories, row.get("memory_id"))
        entity_id = _remapped_optional(state.remap.entities, row.get("entity_id"))
        task_id = _remapped_optional(state.remap.tasks, row.get("task_id"))
        node_type = row["node_type"]
        if (node_type == "memory" and memory_id is None) or (node_type == "entity" and entity_id is None):
            state.remap.graph_nodes.pop(old_id, None)
            continue
        obj = GraphNode(
            id=state.remap.graph_nodes[old_id],
            env_id=state.target_env_id,
            node_type=node_type,
            memory_id=memory_id,
            entity_id=entity_id,
            task_id=task_id,
        )
        await _add(session, obj, state, "graph_nodes", old_id)


async def _apply_relations(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "relations"):
        old_id = _required_uuid(row, "id")
        src = state.remap.graph_nodes.get(_required_uuid(row, "src_node_id"))
        dst = state.remap.graph_nodes.get(_required_uuid(row, "dst_node_id"))
        if src is None or dst is None:
            state.remap.relations.pop(old_id, None)
            continue
        if state.mode == ImportMode.skip and await _relation_exists(session, src, dst, str(row["type"])):
            state.remap.relations.pop(old_id, None)
            state.conflict("relations", old_id)
            continue
        obj = Relation(
            id=state.remap.relations[old_id],
            env_id=state.target_env_id,
            src_node_id=src,
            dst_node_id=dst,
            type=row["type"],
            properties=_json_obj(row.get("properties")),
            version=int(row.get("version", 1)),
        )
        await _add(session, obj, state, "relations", old_id)


async def _apply_memory_lineage(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "memory_lineage"):
        parent = state.remap.memories.get(_required_uuid(row, "parent_memory_id"))
        child = state.remap.memories.get(_required_uuid(row, "child_memory_id"))
        if parent is None or child is None:
            state.conflicts["memory_lineage"] += 1
            continue
        obj = MemoryLineage(parent_memory_id=parent, child_memory_id=child, relation=row["relation"])
        await _add(session, obj, state, "memory_lineage", f"{parent}:{child}:{row['relation']}")


async def _apply_tasks(root: Path, session: AsyncSession, ctx: AgentContext, state: _ImportState) -> None:
    async for row in _rows(root, "tasks"):
        old_id = _required_uuid(row, "id")
        new_id = state.remap.tasks[old_id]
        playbook_id = _remapped_optional(state.remap.memories, row.get("playbook_id"))
        obj = Task(
            id=new_id,
            env_id=state.target_env_id,
            title=row["title"],
            description=row.get("description"),
            status=row.get("status", "pending"),
            priority=int(row.get("priority", 50)),
            playbook_id=playbook_id,
            version=int(row.get("version", 1)),
            created_by_agent_id=ctx.agent_id,
        )
        await _add(session, obj, state, "tasks", old_id)


async def _apply_superseded_by(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "memories"):
        old_id = _required_uuid(row, "id")
        old_target = _maybe_uuid(row.get("superseded_by"))
        if old_target is None:
            continue
        new_id = state.remap.memories.get(old_id)
        new_target = state.remap.memories.get(old_target)
        if new_id is None or new_target is None:
            continue
        await session.execute(
            update(Memory)
            .where(Memory.id == new_id)
            .values(status=row.get("status", "superseded"), superseded_by=new_target)
        )
        memory = state.inserted_memories.get(old_id)
        if memory is not None:
            memory.status = row.get("status", "superseded")
            memory.superseded_by = new_target


async def _apply_dream_runs(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "dream_runs"):
        old_id = _required_uuid(row, "id")
        obj = DreamRun(
            id=state.remap.dream_runs[old_id],
            env_id=state.target_env_id,
            mode=row["mode"],
            status=row.get("status", "running"),
            triggered_by=row.get("triggered_by", "import"),
            summarizer_kind=row.get("summarizer_kind"),
            summary=_remap_payload(_json_obj(row.get("summary")), state.remap),
            last_error=row.get("last_error"),
        )
        await _add(session, obj, state, "dream_runs", old_id)


async def _apply_dream_proposals(root: Path, session: AsyncSession, state: _ImportState) -> None:
    async for row in _rows(root, "dream_proposals"):
        old_id = _required_uuid(row, "id")
        run_id = _remapped_optional(state.remap.dream_runs, row.get("dream_run_id"))
        obj = DreamProposal(
            id=state.remap.dream_proposals[old_id],
            env_id=state.target_env_id,
            kind=row["kind"],
            status=row.get("status", "open"),
            payload=_remap_payload(_json_obj(row.get("payload")), state.remap),
            summarizer_kind=row.get("summarizer_kind"),
            llm_failed=bool(row.get("llm_failed", False)),
            dedupe_key=row.get("dedupe_key"),
            dream_run_id=run_id,
            reviewed_at=_maybe_datetime(row.get("reviewed_at")),
            reviewed_by_agent_id=None,
            review_action=row.get("review_action"),
            review_notes=row.get("review_notes"),
        )
        await _add(session, obj, state, "dream_proposals", old_id)


async def _rows(root: Path, table: str) -> AsyncIterator[dict[str, Any]]:
    path = root / f"{table}.jsonl"
    if not path.is_file():
        return
    async for row in JsonlReader(path):
        yield row


async def _add(session: AsyncSession, obj: object, state: _ImportState, table: str, source_id: object) -> None:
    try:
        session.add(obj)
        await session.flush()
    except IntegrityError:
        if state.mode == ImportMode.skip:
            state.conflict(table, source_id)
            return
        raise
    state.counts[table] += 1


async def _perform_entity_merges(state: _ImportState, *, ctx: AgentContext) -> None:
    for keep_id, merge_id in state.pending_entity_merges:
        async with session_scope() as session:
            rows = (
                await session.execute(
                    select(Entity.id, Entity.version).where(Entity.id.in_({keep_id, merge_id}))
                )
            ).all()
            expected_versions = {entity_id: int(version) for entity_id, version in rows}
        if keep_id not in expected_versions or merge_id not in expected_versions:
            continue
        await entity_merge(
            EntityMergeRequest(
                keep_id=keep_id,
                merge_ids=[merge_id],
                expected_versions=expected_versions,
            ),
            ctx=ctx,
        )
        state.entity_merges_performed += 1


async def _tag_id_by_name(session: AsyncSession, env_id: UUID, name: str) -> UUID | None:
    result = await session.execute(select(Tag.id).where(Tag.env_id == env_id, Tag.name == name))
    return result.scalar_one_or_none()


async def _tag_exists(session: AsyncSession, env_id: UUID, name: str) -> bool:
    return await _tag_id_by_name(session, env_id, name) is not None


async def _tag_name(session: AsyncSession, tag_id: UUID) -> str | None:
    result = await session.execute(select(Tag.name).where(Tag.id == tag_id))
    return result.scalar_one_or_none()


async def _entity_exists(session: AsyncSession, env_id: UUID, normalized_name: str) -> bool:
    return await _entity_id_by_normalized(session, env_id, normalized_name) is not None


async def _entity_id_by_normalized(session: AsyncSession, env_id: UUID, normalized_name: str) -> UUID | None:
    result = await session.execute(
        select(Entity.id).where(Entity.env_id == env_id, Entity.normalized_name == normalized_name)
    )
    return result.scalar_one_or_none()


async def _entity_alias_exists(session: AsyncSession, env_id: UUID, normalized_alias: str) -> bool:
    result = await session.execute(
        select(EntityAlias.entity_id).where(
            EntityAlias.env_id == env_id,
            EntityAlias.normalized_alias == normalized_alias,
        )
    )
    return result.scalar_one_or_none() is not None


async def _relation_exists(session: AsyncSession, src: UUID, dst: UUID, type_: str) -> bool:
    result = await session.execute(
        select(Relation.id).where(
            Relation.src_node_id == src,
            Relation.dst_node_id == dst,
            Relation.type == type_,
        )
    )
    return result.scalar_one_or_none() is not None


async def _apply_embeddings(
    root: Path,
    *,
    manifest: ExportManifest,
    target_model_id: str,
    re_embed_if_model_mismatch: bool,
    state: _ImportState,
    vector_store_factory: VectorStoreFactory | None = None,
) -> tuple[int, int]:
    vectors_path = root / "embeddings" / "memory_vectors.jsonl"
    if not vectors_path.is_file():
        return 0, 0

    pending = 0
    reembedded = 0
    store: VectorStore | None = None
    source_model = manifest.source.default_embedding_model_id
    try:
        async for raw in JsonlReader(vectors_path):
            record = MemoryVectorRecord.model_validate(raw)
            new_memory_id = state.remap.memories.get(record.memory_id)
            memory = state.inserted_memories.get(record.memory_id)
            if new_memory_id is None or memory is None:
                continue
            if record.memory_version != memory.version:
                pending += 1
                continue
            if store is None:
                store = vector_store_factory() if vector_store_factory else _default_vector_store()
            payload = _projection_payload(
                memory,
                tag_names=state.memory_tags.get(new_memory_id, []),
                embedding_model_id=target_model_id,
            )
            if record.model_id == target_model_id:
                await store.ensure_env_collection(env_id=state.target_env_id, dimension=record.dimension)
                await store.upsert(
                    env_id=state.target_env_id,
                    point_id=new_memory_id,
                    vector={record.vector_name: record.vector},
                    payload=payload,
                )
                continue
            if re_embed_if_model_mismatch:
                if record.vector_name != "body":
                    pending += 1
                    continue
                vectors = await maybe_re_embed([memory], source_model, target_model_id, session=None)  # type: ignore[arg-type]
                vector = vectors.get(new_memory_id)
                if vector is None:
                    pending += 1
                    continue
                await store.ensure_env_collection(env_id=state.target_env_id, dimension=len(vector))
                await store.upsert(
                    env_id=state.target_env_id,
                    point_id=new_memory_id,
                    vector={"body": vector},
                    payload=payload,
                )
                reembedded += 1
            else:
                pending += 1
    finally:
        if store is not None:
            await store.close()
    return pending, reembedded


def _default_vector_store() -> VectorStore:
    from memory_mcp.db.vector.qdrant import QdrantVectorStore

    return QdrantVectorStore(get_settings())


def _remap_payload(value: Any, remap: RemapTable) -> Any:
    if isinstance(value, dict):
        return {k: _remap_payload(v, remap) for k, v in value.items()}
    if isinstance(value, list):
        return [_remap_payload(v, remap) for v in value]
    source_id = _maybe_uuid(value)
    if source_id is None:
        return value
    for mapping in (remap.memories, remap.entities, remap.tasks, remap.tags, remap.graph_nodes):
        if source_id in mapping:
            return str(mapping[source_id])
    return value


def _required_uuid(row: Mapping[str, Any], key: str) -> UUID:
    value = _maybe_uuid(row.get(key))
    if value is None:
        raise ValueError(f"row is missing UUID field {key!r}")
    return value


def _maybe_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _remapped_optional(mapping: Mapping[UUID, UUID], value: object) -> UUID | None:
    source_id = _maybe_uuid(value)
    return None if source_id is None else mapping.get(source_id)


def _maybe_datetime(value: object) -> dt.datetime | None:
    if value is None or isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _json_obj(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"expected object payload, got {type(value).__name__}")


def _json_nullable(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"expected nullable object payload, got {type(value).__name__}")


def _remap_size(remap: RemapTable) -> int:
    return sum(len(value) for value in remap.model_dump().values())


__all__ = [
    "ArchiveVersionError",
    "BulkReembedBlocked",
    "ChecksumMismatchError",
    "SCHEMA_VERSION",
    "import_env",
]
