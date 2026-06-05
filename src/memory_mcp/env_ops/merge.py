"""Environment merge implementation for v0.8 env operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID, uuid4

from memory_mcp_schemas.entities import EntityMergeRequest
from memory_mcp_schemas.env_ops import (
    EntityMergeStrategy,
    EnvMergeRequest,
    EnvMergeResponse,
    RemapTable,
    TagMergeStrategy,
)
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from memory_mcp import rbac
from memory_mcp.config import get_settings
from memory_mcp.db.models import (
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
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import OutboxAggregateType, OutboxOp
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.db.vector.qdrant import QdrantVectorStore
from memory_mcp.entities import entity_merge
from memory_mcp.env_ops.clone import _default_vector_store
from memory_mcp.errors import InvalidInputError, MemoryMCPError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import _projection_payload

log = logging.getLogger(__name__)


class ExternalRefsBlockingError(MemoryMCPError):
    """Cross-env lineage touching a third env blocks merge unless explicitly allowed."""

    code = "EXTERNAL_REFS_BLOCKING"


@dataclass(frozen=True)
class _LineageRef:
    parent_memory_id: UUID
    child_memory_id: UUID
    relation: str
    parent_env_id: UUID
    child_env_id: UUID

    @property
    def sample_id(self) -> str:
        return f"{self.parent_memory_id}:{self.child_memory_id}:{self.relation}"


@dataclass
class _MergeResult:
    counts: dict[str, int]
    remap: RemapTable
    dst_memories: dict[UUID, Memory]
    tag_names: dict[UUID, list[str]]
    entity_collisions: list[tuple[UUID, UUID]]


async def merge_envs(request: EnvMergeRequest, *, ctx: AgentContext) -> EnvMergeResponse:
    """Pairwise merge src_env into dst_env. See plan §6 + §17.8-17.11."""

    if request.src_env_id == request.dst_env_id:
        raise InvalidInputError("src_env_id and dst_env_id must differ")

    src, dst = await _load_env_pair(request.src_env_id, request.dst_env_id)
    rbac.require("write", request.dst_env_id, ctx)
    rbac.require("read", request.src_env_id, ctx)
    rbac.require("write", request.src_env_id, ctx)

    if src.default_embedding_model_id != dst.default_embedding_model_id and not request.allow_embedding_mismatch:
        exc = InvalidInputError(
            "source and destination default embedding models differ",
            src_env_id=str(request.src_env_id),
            dst_env_id=str(request.dst_env_id),
            src_default_embedding_model_id=src.default_embedding_model_id,
            dst_default_embedding_model_id=dst.default_embedding_model_id,
        )
        exc.code = "EMBEDDING_MODEL_MISMATCH"
        raise exc

    external_refs = await _scan_external_lineage(
        src_env_id=request.src_env_id,
        dst_env_id=request.dst_env_id,
    )
    if external_refs and not request.allow_external_ref_rewrite:
        samples = [ref.sample_id for ref in external_refs[:20]]
        raise ExternalRefsBlockingError(
            "cross-env lineage touching a third environment blocks env_merge",
            sample_ids=samples,
            count=len(external_refs),
        )

    if request.dry_run:
        async with session_scope() as session:
            counts = await _source_counts(session, request.src_env_id)
        return EnvMergeResponse(
            dst_env_id=request.dst_env_id,
            src_env_id=request.src_env_id,
            delete_src_after=False,
            counts=counts,
            entity_merges_performed=0,
            external_refs_rewritten=0,
            pending_vector_rebuild=counts.get("memories", 0),
            remap_table_size=0,
        )

    async with session_scope() as session:
        result = await _apply_rows(session, request=request)

        if request.allow_external_ref_rewrite:
            external_refs_rewritten = await _rewrite_cross_env_lineage(
                session,
                src_env_id=request.src_env_id,
                remap=result.remap,
            )
        else:
            external_refs_rewritten = 0

        if request.delete_src_after:
            await session.execute(
                update(Environment)
                .where(Environment.id == request.src_env_id)
                .values(status="deleted", deleted_at=func.now())
            )

        await enqueue_event(
            session,
            aggregate_type=OutboxAggregateType.env,
            aggregate_id=request.dst_env_id,
            aggregate_version=1,
            env_id=request.dst_env_id,
            op=OutboxOp.update,
            payload={
                "event": "EnvMerged",
                "src_env_id": str(request.src_env_id),
                "dst_env_id": str(request.dst_env_id),
                "delete_src_after": request.delete_src_after,
                "counts": dict(result.counts),
                "external_refs_rewritten": external_refs_rewritten,
            },
        )

    entity_merges_performed = 0
    if request.entity_strategy == EntityMergeStrategy.by_canonical_key:
        for keep_id, merge_id in result.entity_collisions:
            versions = await _entity_versions([keep_id, merge_id])
            await entity_merge(
                EntityMergeRequest(
                    keep_id=keep_id,
                    merge_ids=[merge_id],
                    expected_versions=versions,
                ),
                ctx=ctx,
                settings=None,
            )
            entity_merges_performed += 1

    pending_vector_rebuild = await _copy_vectors(
        src_env_id=request.src_env_id,
        dst_env_id=request.dst_env_id,
        remap=result.remap,
        memories=result.dst_memories,
        tag_names=result.tag_names,
        model_id=dst.default_embedding_model_id,
    )

    return EnvMergeResponse(
        dst_env_id=request.dst_env_id,
        src_env_id=request.src_env_id,
        delete_src_after=request.delete_src_after,
        counts=dict(result.counts),
        entity_merges_performed=entity_merges_performed,
        external_refs_rewritten=external_refs_rewritten,
        pending_vector_rebuild=pending_vector_rebuild,
        remap_table_size=_remap_size(result.remap),
    )


async def _load_env_pair(src_env_id: UUID, dst_env_id: UUID) -> tuple[Environment, Environment]:
    async with session_scope() as session:
        rows = (
            (await session.execute(select(Environment).where(Environment.id.in_([src_env_id, dst_env_id]))))
            .scalars()
            .all()
        )
    by_id = {row.id: row for row in rows}
    for env_id in (src_env_id, dst_env_id):
        env = by_id.get(env_id)
        if env is None:
            raise NotFoundError(f"environment {env_id} not found", env_id=str(env_id))
        if getattr(env, "status", "active") == "deleted":
            exc = NotFoundError(f"environment {env_id} is deleted", env_id=str(env_id))
            exc.code = "ENV_DELETED"
            raise exc
    return by_id[src_env_id], by_id[dst_env_id]


async def _scan_external_lineage(*, src_env_id: UUID, dst_env_id: UUID) -> list[_LineageRef]:
    async with session_scope() as session:
        refs = await _lineage_touching_src(session, src_env_id=src_env_id)
    return [
        ref
        for ref in refs
        if (
            (ref.parent_env_id == src_env_id and ref.child_env_id not in {src_env_id, dst_env_id})
            or (ref.child_env_id == src_env_id and ref.parent_env_id not in {src_env_id, dst_env_id})
        )
    ]


async def _lineage_touching_src(session: AsyncSession, *, src_env_id: UUID) -> list[_LineageRef]:
    parent = aliased(Memory)
    child = aliased(Memory)
    rows = (
        await session.execute(
            select(
                MemoryLineage.parent_memory_id,
                MemoryLineage.child_memory_id,
                MemoryLineage.relation,
                parent.env_id,
                child.env_id,
            )
            .join(parent, parent.id == MemoryLineage.parent_memory_id)
            .join(child, child.id == MemoryLineage.child_memory_id)
            .where(or_(parent.env_id == src_env_id, child.env_id == src_env_id))
        )
    ).all()
    return [
        _LineageRef(
            parent_memory_id=parent_memory_id,
            child_memory_id=child_memory_id,
            relation=relation,
            parent_env_id=parent_env_id,
            child_env_id=child_env_id,
        )
        for parent_memory_id, child_memory_id, relation, parent_env_id, child_env_id in rows
    ]


async def _source_counts(session: AsyncSession, src_env_id: UUID) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, model in (
        ("tags", Tag),
        ("entities", Entity),
        ("graph_nodes", GraphNode),
        ("memories", Memory),
        ("memory_tags", MemoryTag),
        ("tasks", Task),
        ("relations", Relation),
    ):
        counts[name] = int(
            await session.scalar(select(func.count()).select_from(model).where(model.env_id == src_env_id)) or 0
        )
    entity_ids = select(Entity.id).where(Entity.env_id == src_env_id)
    memory_ids = select(Memory.id).where(Memory.env_id == src_env_id)
    counts["entity_aliases"] = int(
        await session.scalar(select(func.count()).select_from(EntityAlias).where(EntityAlias.entity_id.in_(entity_ids)))
        or 0
    )
    counts["memory_sources"] = int(
        await session.scalar(
            select(func.count()).select_from(MemorySource).where(MemorySource.memory_id.in_(memory_ids))
        )
        or 0
    )
    counts["memory_lineage"] = int(
        await session.scalar(
            select(func.count())
            .select_from(MemoryLineage)
            .where(MemoryLineage.parent_memory_id.in_(memory_ids))
            .where(MemoryLineage.child_memory_id.in_(memory_ids))
        )
        or 0
    )
    return counts


async def _apply_rows(session: AsyncSession, *, request: EnvMergeRequest) -> _MergeResult:
    counts = {
        "tags": 0,
        "entities": 0,
        "entity_aliases": 0,
        "graph_nodes": 0,
        "memories": 0,
        "memory_tags": 0,
        "memory_sources": 0,
        "relations": 0,
        "memory_lineage": 0,
        "tasks": 0,
    }
    remap = RemapTable()
    dst_memories: dict[UUID, Memory] = {}
    tag_names: dict[UUID, list[str]] = {}
    entity_collisions: list[tuple[UUID, UUID]] = []

    await _copy_tags(session, request=request, counts=counts, remap=remap)
    entity_collisions = await _copy_entities(session, request=request, counts=counts, remap=remap)
    await _copy_entity_aliases(session, request=request, counts=counts, remap=remap)
    await _copy_memories(session, request=request, counts=counts, remap=remap, dst_memories=dst_memories)
    await _copy_memory_tags(session, request=request, counts=counts, remap=remap, tag_names=tag_names)
    await _copy_memory_sources(session, request=request, counts=counts, remap=remap)
    await _copy_tasks(session, request=request, counts=counts, remap=remap)
    await _copy_graph_nodes(session, request=request, counts=counts, remap=remap)
    await _copy_relations(session, request=request, counts=counts, remap=remap)
    await _copy_intra_src_lineage(session, request=request, counts=counts, remap=remap)
    await _update_superseded_by(session, remap=remap)

    return _MergeResult(
        counts=counts,
        remap=remap,
        dst_memories=dst_memories,
        tag_names=tag_names,
        entity_collisions=entity_collisions,
    )


async def _copy_tags(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
) -> None:
    src_tags = (
        (await session.execute(select(Tag).where(Tag.env_id == request.src_env_id).order_by(Tag.name))).scalars().all()
    )
    dst_by_name = {
        name: tag_id
        for tag_id, name in (
            await session.execute(select(Tag.id, Tag.name).where(Tag.env_id == request.dst_env_id))
        ).all()
    }
    for row in src_tags:
        if request.tag_strategy == TagMergeStrategy.union and row.name in dst_by_name:
            remap.tags[row.id] = dst_by_name[row.name]
            continue
        new_id = uuid4()
        remap.tags[row.id] = new_id
        session.add(Tag(id=new_id, env_id=request.dst_env_id, name=row.name))
        counts["tags"] += 1
    await session.flush()


async def _copy_entities(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
) -> list[tuple[UUID, UUID]]:
    collisions: list[tuple[UUID, UUID]] = []
    dst_by_normalized = {
        normalized_name: entity_id
        for entity_id, normalized_name in (
            await session.execute(select(Entity.id, Entity.normalized_name).where(Entity.env_id == request.dst_env_id))
        ).all()
    }
    src_entities = (
        (
            await session.execute(
                select(Entity).where(Entity.env_id == request.src_env_id).order_by(Entity.created_at, Entity.id)
            )
        )
        .scalars()
        .all()
    )
    for row in src_entities:
        new_id = uuid4()
        remap.entities[row.id] = new_id
        normalized_name = row.normalized_name
        if normalized_name in dst_by_normalized:
            if request.entity_strategy == EntityMergeStrategy.by_canonical_key:
                collisions.append((dst_by_normalized[normalized_name], new_id))
            normalized_name = f"{normalized_name} merge {new_id.hex}"
        session.add(
            Entity(
                id=new_id,
                env_id=request.dst_env_id,
                kind=row.kind,
                canonical_name=row.canonical_name,
                normalized_name=normalized_name,
                metadata_=dict(row.metadata_ or {}),
                created_at=row.created_at,
                updated_at=row.updated_at,
                version=row.version,
            )
        )
        counts["entities"] += 1
    await session.flush()
    return collisions


async def _copy_entity_aliases(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
) -> None:
    if not remap.entities:
        return
    dst_aliases = set(
        (await session.execute(select(EntityAlias.normalized_alias).where(EntityAlias.env_id == request.dst_env_id)))
        .scalars()
        .all()
    )
    dst_canonicals = set(
        (await session.execute(select(Entity.normalized_name).where(Entity.env_id == request.dst_env_id)))
        .scalars()
        .all()
    )
    rows = (
        (
            await session.execute(
                select(EntityAlias)
                .where(EntityAlias.env_id == request.src_env_id)
                .where(EntityAlias.entity_id.in_(remap.entities.keys()))
                .order_by(EntityAlias.created_at)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        if row.normalized_alias in dst_aliases or row.normalized_alias in dst_canonicals:
            continue
        session.add(
            EntityAlias(
                entity_id=remap.entities[row.entity_id],
                env_id=request.dst_env_id,
                alias=row.alias,
                normalized_alias=row.normalized_alias,
                created_at=row.created_at,
            )
        )
        dst_aliases.add(row.normalized_alias)
        counts["entity_aliases"] += 1
    await session.flush()


async def _copy_memories(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
    dst_memories: dict[UUID, Memory],
) -> None:
    rows = (
        (
            await session.execute(
                select(Memory).where(Memory.env_id == request.src_env_id).order_by(Memory.created_at, Memory.id)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        new_id = uuid4()
        remap.memories[row.id] = new_id
        insert_status = "active" if row.status == "superseded" and row.superseded_by is not None else row.status
        memory = Memory(
            id=new_id,
            env_id=request.dst_env_id,
            kind=row.kind,
            status=insert_status,
            title=row.title,
            body=row.body,
            trigger_description=row.trigger_description,
            steps=list(row.steps) if row.steps is not None else None,
            macro=row.macro,
            salience=float(row.salience),
            confidence=float(row.confidence),
            access_count=row.access_count,
            last_accessed_at=row.last_accessed_at,
            pinned=row.pinned,
            negative_feedback_count=row.negative_feedback_count,
            verified_at=row.verified_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
            expires_at=row.expires_at,
            superseded_by=None,
            metadata_=dict(row.metadata_ or {}),
            decision_meta=dict(row.decision_meta) if row.decision_meta is not None else None,
            version=row.version,
        )
        session.add(memory)
        dst_memories[row.id] = memory
        counts["memories"] += 1
    await session.flush()


async def _copy_memory_tags(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
    tag_names: dict[UUID, list[str]],
) -> None:
    if not remap.memories or not remap.tags:
        return
    tag_name_by_old = dict((await session.execute(select(Tag.id, Tag.name).where(Tag.id.in_(remap.tags.keys())))).all())
    rows = (
        (
            await session.execute(
                select(MemoryTag)
                .where(MemoryTag.env_id == request.src_env_id)
                .where(MemoryTag.memory_id.in_(remap.memories.keys()))
                .where(MemoryTag.tag_id.in_(remap.tags.keys()))
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        new_memory_id = remap.memories[row.memory_id]
        new_tag_id = remap.tags[row.tag_id]
        await session.execute(
            pg_insert(MemoryTag)
            .values(memory_id=new_memory_id, tag_id=new_tag_id, env_id=request.dst_env_id)
            .on_conflict_do_nothing(index_elements=["memory_id", "tag_id"])
        )
        tag_names.setdefault(row.memory_id, []).append(tag_name_by_old[row.tag_id])
        counts["memory_tags"] += 1
    await session.flush()


async def _copy_memory_sources(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
) -> None:
    if not remap.memories:
        return
    rows = (
        (
            await session.execute(
                select(MemorySource).where(MemorySource.memory_id.in_(remap.memories.keys())).order_by(MemorySource.id)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        session.add(
            MemorySource(
                memory_id=remap.memories[row.memory_id],
                source_type=row.source_type,
                source_ref=row.source_ref,
                agent_id=None,
                created_at=row.created_at,
                evidence_span=row.evidence_span,
            )
        )
        counts["memory_sources"] += 1
    await session.flush()


async def _copy_tasks(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
) -> None:
    rows = (
        (
            await session.execute(
                select(Task).where(Task.env_id == request.src_env_id).order_by(Task.created_at, Task.id)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        new_id = uuid4()
        remap.tasks[row.id] = new_id
        session.add(
            Task(
                id=new_id,
                env_id=request.dst_env_id,
                title=row.title,
                description=row.description,
                status=row.status,
                priority=row.priority,
                playbook_id=remap.memories.get(row.playbook_id) if row.playbook_id is not None else None,
                version=row.version,
                created_at=row.created_at,
                updated_at=row.updated_at,
                created_by_agent_id=None,
            )
        )
        counts["tasks"] += 1
    await session.flush()


async def _copy_graph_nodes(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
) -> None:
    rows = (
        (
            await session.execute(
                select(GraphNode)
                .where(GraphNode.env_id == request.src_env_id)
                .order_by(GraphNode.created_at, GraphNode.id)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        memory_id = remap.memories.get(row.memory_id) if row.memory_id is not None else None
        entity_id = remap.entities.get(row.entity_id) if row.entity_id is not None else None
        task_id = remap.tasks.get(row.task_id) if row.task_id is not None else None
        if (
            (row.node_type == "memory" and memory_id is None)
            or (row.node_type == "entity" and entity_id is None)
            or (row.node_type == "task" and task_id is None)
        ):
            continue
        new_id = uuid4()
        remap.graph_nodes[row.id] = new_id
        session.add(
            GraphNode(
                id=new_id,
                env_id=request.dst_env_id,
                node_type=row.node_type,
                memory_id=memory_id,
                entity_id=entity_id,
                task_id=task_id,
                created_at=row.created_at,
            )
        )
        counts["graph_nodes"] += 1
    await session.flush()


async def _copy_relations(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
) -> None:
    if not remap.graph_nodes:
        return
    rows = (
        (
            await session.execute(
                select(Relation)
                .where(Relation.env_id == request.src_env_id)
                .where(Relation.src_node_id.in_(remap.graph_nodes.keys()))
                .where(Relation.dst_node_id.in_(remap.graph_nodes.keys()))
                .order_by(Relation.created_at, Relation.id)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        new_id = uuid4()
        remap.relations[row.id] = new_id
        session.add(
            Relation(
                id=new_id,
                env_id=request.dst_env_id,
                src_node_id=remap.graph_nodes[row.src_node_id],
                dst_node_id=remap.graph_nodes[row.dst_node_id],
                type=row.type,
                properties=dict(row.properties or {}),
                created_at=row.created_at,
                updated_at=row.updated_at,
                version=row.version,
            )
        )
        counts["relations"] += 1
    await session.flush()


async def _copy_intra_src_lineage(
    session: AsyncSession,
    *,
    request: EnvMergeRequest,
    counts: dict[str, int],
    remap: RemapTable,
) -> None:
    if not remap.memories:
        return
    rows = (
        (
            await session.execute(
                select(MemoryLineage)
                .where(MemoryLineage.parent_memory_id.in_(remap.memories.keys()))
                .where(MemoryLineage.child_memory_id.in_(remap.memories.keys()))
                .order_by(MemoryLineage.created_at)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        await session.execute(
            pg_insert(MemoryLineage)
            .values(
                parent_memory_id=remap.memories[row.parent_memory_id],
                child_memory_id=remap.memories[row.child_memory_id],
                relation=row.relation,
                created_at=row.created_at,
            )
            .on_conflict_do_nothing(index_elements=["parent_memory_id", "child_memory_id", "relation"])
        )
        counts["memory_lineage"] += 1
    await session.flush()


async def _update_superseded_by(session: AsyncSession, *, remap: RemapTable) -> None:
    if not remap.memories:
        return
    rows = (
        await session.execute(
            select(Memory.id, Memory.status, Memory.superseded_by).where(Memory.id.in_(remap.memories.keys()))
        )
    ).all()
    for old_id, old_status, old_target in rows:
        if old_target is None or old_target not in remap.memories:
            continue
        await session.execute(
            update(Memory)
            .where(Memory.id == remap.memories[old_id])
            .values(status=old_status, superseded_by=remap.memories[old_target])
        )


async def _rewrite_cross_env_lineage(
    session: AsyncSession,
    *,
    src_env_id: UUID,
    remap: RemapTable,
) -> int:
    refs = await _lineage_touching_src(session, src_env_id=src_env_id)
    rewritten = 0
    for ref in refs:
        new_parent = remap.memories.get(ref.parent_memory_id, ref.parent_memory_id)
        new_child = remap.memories.get(ref.child_memory_id, ref.child_memory_id)
        if new_parent == ref.parent_memory_id and new_child == ref.child_memory_id:
            continue
        if ref.parent_env_id == src_env_id and ref.child_env_id == src_env_id:
            continue
        await session.execute(
            pg_insert(MemoryLineage)
            .values(
                parent_memory_id=new_parent,
                child_memory_id=new_child,
                relation=ref.relation,
                created_at=func.now(),
            )
            .on_conflict_do_nothing(index_elements=["parent_memory_id", "child_memory_id", "relation"])
        )
        await session.execute(
            delete(MemoryLineage).where(
                MemoryLineage.parent_memory_id == ref.parent_memory_id,
                MemoryLineage.child_memory_id == ref.child_memory_id,
                MemoryLineage.relation == ref.relation,
            )
        )
        rewritten += 1
    return rewritten


async def _entity_versions(entity_ids: list[UUID]) -> dict[UUID, int]:
    async with session_scope() as session:
        rows = (await session.execute(select(Entity.id, Entity.version).where(Entity.id.in_(entity_ids)))).all()
    return dict(rows)


def _remap_size(remap: RemapTable) -> int:
    return (
        len(remap.memories)
        + len(remap.entities)
        + len(remap.tags)
        + len(remap.graph_nodes)
        + len(remap.relations)
        + len(remap.tasks)
    )


async def _copy_vectors(
    *,
    src_env_id: UUID,
    dst_env_id: UUID,
    remap: RemapTable,
    memories: dict[UUID, Memory],
    tag_names: dict[UUID, list[str]],
    model_id: str,
) -> int:
    if not remap.memories:
        return 0

    store = _merge_vector_store()
    pending = 0
    try:
        old_ids = list(remap.memories)
        body_vectors = await store.get_vectors(env_id=src_env_id, ids=old_ids, vector_name="body")
        trigger_vectors = await store.get_vectors(env_id=src_env_id, ids=old_ids, vector_name="trigger")
        ensured_dimension: int | None = None
        for old_id, new_id in remap.memories.items():
            body = body_vectors.get(old_id)
            if body is None:
                pending += 1
                continue
            if ensured_dimension is None:
                ensured_dimension = len(body)
                await store.ensure_env_collection(env_id=dst_env_id, dimension=ensured_dimension)
            vectors: dict[str, list[float]] = {"body": body}
            trigger = trigger_vectors.get(old_id)
            if trigger is not None:
                vectors["trigger"] = trigger
            await store.upsert(
                env_id=dst_env_id,
                point_id=new_id,
                vector=vectors,
                payload=_projection_payload(
                    memories[old_id],
                    tag_names=tag_names.get(old_id, []),
                    embedding_model_id=model_id,
                ),
            )
        return pending
    except Exception:
        log.exception("env_merge: vector copy failed; marking all memories pending")
        return len(remap.memories)
    finally:
        await store.close()


def _merge_vector_store() -> VectorStore:
    try:
        return _default_vector_store()
    except Exception:
        return QdrantVectorStore(get_settings())


__all__ = ["ExternalRefsBlockingError", "merge_envs"]
