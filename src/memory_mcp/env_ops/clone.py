"""Environment clone implementation for v0.8 env operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
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
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.db.vector.qdrant import QdrantVectorStore
from memory_mcp.envs import env_create, get_env_by_name
from memory_mcp.errors import AlreadyExistsError, MemoryMCPError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import _projection_payload

from memory_mcp_schemas.browse import MemBrowseRequest
from memory_mcp_schemas.env_ops import EnvCloneRequest, EnvCloneResponse
from memory_mcp_schemas.envs import EnvCreateRequest


class ConflictError(MemoryMCPError):
    """Clone destination environment name is already in use."""

    code = "ENV_NAME_TAKEN"


@dataclass
class _Closure:
    seed_memories: set[UUID] = field(default_factory=set)
    memories: set[UUID] = field(default_factory=set)
    entities: set[UUID] = field(default_factory=set)
    tags: set[UUID] = field(default_factory=set)
    graph_nodes: set[UUID] = field(default_factory=set)
    tasks: set[UUID] = field(default_factory=set)


@dataclass
class _Remap:
    memories: dict[UUID, UUID] = field(default_factory=dict)
    entities: dict[UUID, UUID] = field(default_factory=dict)
    tags: dict[UUID, UUID] = field(default_factory=dict)
    graph_nodes: dict[UUID, UUID] = field(default_factory=dict)
    relations: dict[UUID, UUID] = field(default_factory=dict)
    tasks: dict[UUID, UUID] = field(default_factory=dict)


async def clone_env(request: EnvCloneRequest, *, ctx: AgentContext) -> EnvCloneResponse:
    """Duplicate src env into a fresh dst env. See plan §3.1 + §17.12.

    Source UUIDs are never reused. Filtered clones first resolve seed memories
    with browse semantics, then expand closure for supersession chains, lineage
    parents (``lineage_depth`` is schema-limited to 0..5), required tags, and
    FK-traceable graph nodes. ``MemorySource`` currently has no ``entity_id``
    column, so referenced-entity closure is limited to explicit graph/entity
    rows; TODO: extend this if a future schema adds structured memory-entity
    provenance. Embeddings are copied verbatim after commit; callers that want
    a different embedding model should run a separate re-embed flow.
    """

    src = await _load_src_env(request.src_env_id)
    rbac.require("read", request.src_env_id, ctx)
    await _ensure_name_available(request.new_name)

    try:
        dst = await env_create(
            EnvCreateRequest(
                name=request.new_name,
                kind=src.kind,
                retention_policy=dict(src.retention_policy or {}),
                default_embedding_model_id=src.default_embedding_model_id,
            ),
            ctx=ctx,
        )
    except AlreadyExistsError as exc:
        raise ConflictError(f"environment name already exists: {request.new_name!r}", name=request.new_name) from exc

    async with session_scope() as session:
        closure = await _resolve_closure(session, request)
        copied = await _copy_closure(session, src_env_id=request.src_env_id, dst_env_id=dst.id, closure=closure, ctx=ctx)

    pending_vector_rebuild = await _copy_vectors(
        src_env_id=request.src_env_id,
        dst_env_id=dst.id,
        remap=copied.remap,
        memories=copied.dst_memories,
        tag_names=copied.tag_names,
        model_id=src.default_embedding_model_id,
        include_embeddings=request.include_embeddings,
    )

    return EnvCloneResponse(
        dst_env_id=dst.id,
        dst_env_name=dst.name,
        new_env_id=dst.id,
        counts=dict(copied.counts),
        closure_inclusions=_closure_inclusions(closure),
        pending_vector_rebuild=pending_vector_rebuild,
        remap_table_size=_remap_size(copied.remap),
    )


async def _load_src_env(env_id: UUID) -> Environment:
    async with session_scope() as session:
        env = (await session.execute(select(Environment).where(Environment.id == env_id))).scalar_one_or_none()
    if env is None:
        raise NotFoundError(f"environment {env_id} not found", env_id=str(env_id))
    if getattr(env, "status", "active") == "deleted":
        exc = NotFoundError(f"environment {env_id} is deleted", env_id=str(env_id))
        exc.code = "ENV_DELETED"
        raise exc
    return env


async def _ensure_name_available(name: str) -> None:
    existing = await get_env_by_name(name, include_deleted=True)
    if existing is not None:
        raise ConflictError(f"environment name already exists: {name!r}", name=name)

    async with session_scope() as session:
        collision = await session.scalar(
            select(Environment.id).where(func.lower(Environment.name) == name.lower()).limit(1)
        )
    if collision is not None:
        raise ConflictError(f"environment name already exists: {name!r}", name=name)


async def _resolve_closure(session: AsyncSession, request: EnvCloneRequest) -> _Closure:
    seed = await _seed_memory_ids(session, request)
    closure = _Closure(seed_memories=set(seed), memories=set(seed))
    if request.filter is not None:
        await _expand_supersession_chain(session, env_id=request.src_env_id, memory_ids=closure.memories)
        await _expand_lineage_parents(
            session,
            memory_ids=closure.memories,
            seed_ids=set(seed),
            depth=request.lineage_depth,
        )

    closure.tags = set((await session.execute(
        select(MemoryTag.tag_id).where(MemoryTag.env_id == request.src_env_id, MemoryTag.memory_id.in_(closure.memories))
    )).scalars().all()) if closure.memories else set()

    if request.filter is None:
        closure.entities = set((await session.execute(
            select(Entity.id).where(Entity.env_id == request.src_env_id)
        )).scalars().all())
        closure.graph_nodes = set((await session.execute(
            select(GraphNode.id).where(GraphNode.env_id == request.src_env_id)
        )).scalars().all())
        closure.tasks = set((await session.execute(
            select(Task.id).where(Task.env_id == request.src_env_id)
        )).scalars().all())
        return closure

    if request.include_referenced_entities:
        memory_node = aliased(GraphNode)
        entity_node = aliased(GraphNode)
        closure.entities.update((await session.execute(
            select(entity_node.entity_id)
            .select_from(Relation)
            .join(memory_node, memory_node.id == Relation.src_node_id)
            .join(entity_node, entity_node.id == Relation.dst_node_id)
            .where(Relation.env_id == request.src_env_id)
            .where(memory_node.memory_id.in_(closure.memories))
            .where(entity_node.entity_id.is_not(None))
        )).scalars().all())

    closure.tasks = set((await session.execute(
        select(Task.id)
        .where(Task.env_id == request.src_env_id)
        .where(Task.playbook_id.is_not(None), Task.playbook_id.in_(closure.memories))
    )).scalars().all()) if closure.memories else set()

    node_clauses: list[Any] = []
    if closure.memories:
        node_clauses.append(GraphNode.memory_id.in_(closure.memories))
    if closure.entities:
        node_clauses.append(GraphNode.entity_id.in_(closure.entities))
    if closure.tasks:
        node_clauses.append(GraphNode.task_id.in_(closure.tasks))
    if node_clauses:
        from sqlalchemy import or_

        closure.graph_nodes = set((await session.execute(
            select(GraphNode.id).where(GraphNode.env_id == request.src_env_id).where(or_(*node_clauses))
        )).scalars().all())
    return closure


async def _seed_memory_ids(session: AsyncSession, request: EnvCloneRequest) -> set[UUID]:
    if request.filter is None:
        return set((await session.execute(
            select(Memory.id).where(Memory.env_id == request.src_env_id)
        )).scalars().all())

    browse_filter = _filter_for_src(request.filter, request.src_env_id)
    statuses = [s.value for s in browse_filter.statuses] if browse_filter.statuses else ["proposed", "active"]
    stmt = (
        select(Memory.id)
        .where(Memory.env_id == request.src_env_id)
        .where(Memory.status.in_(statuses))
    )
    if browse_filter.kinds:
        stmt = stmt.where(Memory.kind.in_([k.value for k in browse_filter.kinds]))
    if browse_filter.created_after is not None:
        stmt = stmt.where(Memory.created_at >= browse_filter.created_after)
    if browse_filter.created_before is not None:
        stmt = stmt.where(Memory.created_at < browse_filter.created_before)
    if browse_filter.updated_after is not None:
        stmt = stmt.where(Memory.updated_at >= browse_filter.updated_after)
    if browse_filter.tags:
        stmt = stmt.where(
            select(MemoryTag.memory_id)
            .join(Tag, Tag.id == MemoryTag.tag_id)
            .where(
                MemoryTag.env_id == request.src_env_id,
                MemoryTag.memory_id == Memory.id,
                Tag.name.in_(browse_filter.tags),
            )
            .exists()
        )
    return set((await session.execute(stmt)).scalars().all())


def _filter_for_src(raw: MemBrowseRequest, src_env_id: UUID) -> MemBrowseRequest:
    return raw.model_copy(update={"env_ids": [src_env_id], "cursor": None})


async def _expand_supersession_chain(session: AsyncSession, *, env_id: UUID, memory_ids: set[UUID]) -> None:
    frontier = set(memory_ids)
    while frontier:
        rows = (await session.execute(
            select(Memory.id, Memory.superseded_by)
            .where(Memory.env_id == env_id)
            .where((Memory.id.in_(frontier)) | (Memory.superseded_by.in_(frontier)))
        )).all()
        discovered: set[UUID] = set()
        for memory_id, superseded_by in rows:
            discovered.add(memory_id)
            if superseded_by is not None:
                discovered.add(superseded_by)
        frontier = discovered - memory_ids
        memory_ids.update(discovered)


async def _expand_lineage_parents(
    session: AsyncSession,
    *,
    memory_ids: set[UUID],
    seed_ids: set[UUID],
    depth: int,
) -> None:
    frontier = set(seed_ids)
    for _ in range(depth):
        if not frontier:
            return
        parents = set((await session.execute(
            select(MemoryLineage.parent_memory_id).where(MemoryLineage.child_memory_id.in_(frontier))
        )).scalars().all())
        frontier = parents - memory_ids
        memory_ids.update(parents)


@dataclass
class _CopyResult:
    counts: dict[str, int]
    remap: _Remap
    dst_memories: dict[UUID, Memory]
    tag_names: dict[UUID, list[str]]


async def _copy_closure(
    session: AsyncSession,
    *,
    src_env_id: UUID,
    dst_env_id: UUID,
    closure: _Closure,
    ctx: AgentContext,
) -> _CopyResult:
    counts = {
        "env": 1,
        "tags": 0,
        "entities": 0,
        "entity_aliases": 0,
        "memories": 0,
        "memory_tags": 0,
        "memory_sources": 0,
        "tasks": 0,
        "graph_nodes": 0,
        "relations": 0,
        "memory_lineage": 0,
    }
    remap = _Remap()
    dst_memories: dict[UUID, Memory] = {}
    tag_names: dict[UUID, list[str]] = {}

    for row in await _rows_by_ids(session, Tag, closure.tags):
        remap.tags[row.id] = uuid4()
        session.add(Tag(id=remap.tags[row.id], env_id=dst_env_id, name=row.name))
        counts["tags"] += 1

    for row in await _rows_by_ids(session, Entity, closure.entities):
        remap.entities[row.id] = uuid4()
        session.add(Entity(
            id=remap.entities[row.id],
            env_id=dst_env_id,
            kind=row.kind,
            canonical_name=row.canonical_name,
            normalized_name=row.normalized_name,
            metadata_=dict(row.metadata_ or {}),
            created_at=row.created_at,
            updated_at=row.updated_at,
            version=row.version,
        ))
        counts["entities"] += 1
    await session.flush()

    if closure.entities:
        aliases = (await session.execute(
            select(EntityAlias).where(EntityAlias.env_id == src_env_id, EntityAlias.entity_id.in_(closure.entities))
        )).scalars().all()
        for row in aliases:
            session.add(EntityAlias(
                entity_id=remap.entities[row.entity_id],
                env_id=dst_env_id,
                alias=row.alias,
                normalized_alias=row.normalized_alias,
                created_at=row.created_at,
            ))
            counts["entity_aliases"] += 1

    for row in await _rows_by_ids(session, Memory, closure.memories):
        remap.memories[row.id] = uuid4()
        insert_status = "active" if row.status == "superseded" and row.superseded_by is not None else row.status
        memory = Memory(
            id=remap.memories[row.id],
            env_id=dst_env_id,
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

    if closure.memories and closure.tags:
        memory_tags = (await session.execute(
            select(MemoryTag)
            .where(MemoryTag.env_id == src_env_id)
            .where(MemoryTag.memory_id.in_(closure.memories), MemoryTag.tag_id.in_(closure.tags))
        )).scalars().all()
        tag_names_by_old_id = await _tag_names(session, closure.tags)
        for row in memory_tags:
            new_memory_id = remap.memories[row.memory_id]
            new_tag_id = remap.tags[row.tag_id]
            session.add(MemoryTag(memory_id=new_memory_id, tag_id=new_tag_id, env_id=dst_env_id))
            tag_names.setdefault(row.memory_id, []).append(tag_names_by_old_id[row.tag_id])
            counts["memory_tags"] += 1

    if closure.memories:
        sources = (await session.execute(
            select(MemorySource).where(MemorySource.memory_id.in_(closure.memories))
        )).scalars().all()
        for row in sources:
            session.add(MemorySource(
                memory_id=remap.memories[row.memory_id],
                source_type=row.source_type,
                source_ref=row.source_ref,
                agent_id=None,
                created_at=row.created_at,
                evidence_span=row.evidence_span,
            ))
            counts["memory_sources"] += 1

    task_rows = await _rows_by_ids(session, Task, closure.tasks)
    for row in task_rows:
        remap.tasks[row.id] = uuid4()
        playbook_id = remap.memories.get(row.playbook_id) if row.playbook_id is not None else None
        session.add(Task(
            id=remap.tasks[row.id],
            env_id=dst_env_id,
            title=row.title,
            description=row.description,
            status=row.status,
            priority=row.priority,
            playbook_id=playbook_id,
            version=row.version,
            created_at=row.created_at,
            updated_at=row.updated_at,
            created_by_agent_id=ctx.agent_id,
        ))
        counts["tasks"] += 1
    await session.flush()

    for row in await _rows_by_ids(session, GraphNode, closure.graph_nodes):
        memory_id = remap.memories.get(row.memory_id) if row.memory_id is not None else None
        entity_id = remap.entities.get(row.entity_id) if row.entity_id is not None else None
        task_id = remap.tasks.get(row.task_id) if row.task_id is not None else None
        if (
            (row.node_type == "memory" and memory_id is None)
            or (row.node_type == "entity" and entity_id is None)
            or (row.node_type == "task" and task_id is None)
        ):
            continue
        remap.graph_nodes[row.id] = uuid4()
        session.add(GraphNode(
            id=remap.graph_nodes[row.id],
            env_id=dst_env_id,
            node_type=row.node_type,
            memory_id=memory_id,
            entity_id=entity_id,
            task_id=task_id,
            created_at=row.created_at,
        ))
        counts["graph_nodes"] += 1
    await session.flush()

    if remap.graph_nodes:
        relations = (await session.execute(
            select(Relation)
            .where(Relation.env_id == src_env_id)
            .where(Relation.src_node_id.in_(remap.graph_nodes), Relation.dst_node_id.in_(remap.graph_nodes))
        )).scalars().all()
        for row in relations:
            remap.relations[row.id] = uuid4()
            session.add(Relation(
                id=remap.relations[row.id],
                env_id=dst_env_id,
                src_node_id=remap.graph_nodes[row.src_node_id],
                dst_node_id=remap.graph_nodes[row.dst_node_id],
                type=row.type,
                properties=dict(row.properties or {}),
                created_at=row.created_at,
                updated_at=row.updated_at,
                version=row.version,
            ))
            counts["relations"] += 1

    if closure.memories:
        lineage = (await session.execute(
            select(MemoryLineage)
            .where(MemoryLineage.parent_memory_id.in_(closure.memories))
            .where(MemoryLineage.child_memory_id.in_(closure.memories))
        )).scalars().all()
        for row in lineage:
            session.add(MemoryLineage(
                parent_memory_id=remap.memories[row.parent_memory_id],
                child_memory_id=remap.memories[row.child_memory_id],
                relation=row.relation,
                created_at=row.created_at,
            ))
            counts["memory_lineage"] += 1

    for old_id, memory in dst_memories.items():
        source_status, source_target = (await session.execute(
            select(Memory.status, Memory.superseded_by).where(Memory.id == old_id)
        )).one()
        if source_target is None or source_target not in remap.memories:
            continue
        new_target = remap.memories[source_target]
        await session.execute(
            update(Memory)
            .where(Memory.id == memory.id)
            .values(status=source_status, superseded_by=new_target)
        )
        memory.status = source_status
        memory.superseded_by = new_target

    return _CopyResult(counts=counts, remap=remap, dst_memories=dst_memories, tag_names=tag_names)


async def _rows_by_ids(session: AsyncSession, model: Any, ids: set[UUID]) -> list[Any]:
    if not ids:
        return []
    return list((await session.execute(select(model).where(model.id.in_(ids)).order_by(model.id))).scalars().all())


async def _tag_names(session: AsyncSession, tag_ids: set[UUID]) -> dict[UUID, str]:
    rows = (await session.execute(select(Tag.id, Tag.name).where(Tag.id.in_(tag_ids)))).all()
    return {tag_id: name for tag_id, name in rows}


def _closure_inclusions(closure: _Closure) -> dict[str, int]:
    return {
        "memories": max(0, len(closure.memories - closure.seed_memories)),
        "entities": len(closure.entities),
        "tags": len(closure.tags),
        "graph_nodes": len(closure.graph_nodes),
    }


def _remap_size(remap: _Remap) -> int:
    return (
        len(remap.memories)
        + len(remap.entities)
        + len(remap.tags)
        + len(remap.graph_nodes)
        + len(remap.relations)
        + len(remap.tasks)
    )


def _default_vector_store() -> VectorStore:
    return QdrantVectorStore(get_settings())


async def _copy_vectors(
    *,
    src_env_id: UUID,
    dst_env_id: UUID,
    remap: _Remap,
    memories: dict[UUID, Memory],
    tag_names: dict[UUID, list[str]],
    model_id: str,
    include_embeddings: bool,
) -> int:
    if not include_embeddings:
        return len(remap.memories)
    if not remap.memories:
        return 0

    store = _default_vector_store()
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
            memory = memories[old_id]
            await store.upsert(
                env_id=dst_env_id,
                point_id=new_id,
                vector=vectors,
                payload=_projection_payload(
                    memory,
                    tag_names=tag_names.get(old_id, []),
                    embedding_model_id=model_id,
                ),
            )
        return pending
    except Exception:
        return len(remap.memories)
    finally:
        await store.close()


__all__ = ["ConflictError", "clone_env"]
