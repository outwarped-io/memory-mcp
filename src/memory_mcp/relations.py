"""Relation tools — directed, typed edges between graph_nodes for v1.

The ``relations`` table backs both the canonical graph (v1 Postgres-only;
recursive CTE traversal) and the Phase 2 Neo4j projection. Each relation
edge points at two ``graph_nodes`` rows whose ``node_type`` is either
``entity``, ``memory``, or ``task``.

Phase 1 surface:

* :func:`relation_link` — idempotent insert-or-update by
  ``(src_node_id, dst_node_id, type)``. Creates ``graph_nodes`` rows
  for the endpoints on demand if they don't exist yet.

Outbox routing for ``aggregate_type=relation`` is determined by the
``graph_backend`` setting — ``GRAPH_BACKEND=postgres`` is a no-op (no
Neo4j projection), ``neo4j`` enqueues to the neo4j sink.

Endpoint resolution
-------------------

Callers identify endpoints by ``(kind, id)`` where ``kind ∈ {entity,
memory}`` and ``id`` is the canonical record id (``entities.id`` or
``memories.id``). On insert, we look up the matching ``graph_nodes``
row via the partial-unique indexes ``graph_nodes_{entity,memory}_uniq``;
if no row exists we INSERT one. The referenced record must exist in
the same env.

Self-loops (src == dst) are rejected.

Optimistic concurrency
----------------------

Same pattern as ``entity_upsert``: SELECT, version-check, UPDATE WHERE
version=:expected, raise :class:`VersionConflictError` on rowcount=0.
A re-link with identical ``properties`` is idempotent — no UPDATE,
no version bump, no outbox event.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any
from uuid import UUID

from memory_mcp_schemas.relations import (
    RelationBrowseHit,
    RelationBrowseRequest,
    RelationBrowseResponse,
    RelationEndpoint,
    RelationLinkRequest,
    RelationResponse,
)
from sqlalchemy import Select, and_, func, select, tuple_, update
from sqlalchemy.exc import IntegrityError

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import (
    AuditLog,
    Entity,
    GraphNode,
    Memory,
    Relation,
    Task,
)
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import OutboxAggregateType, OutboxOp
from memory_mcp.errors import (
    CycleDetectedError,
    EnvAmbiguousError,
    InvalidCursorError,
    InvalidInputError,
    NotFoundError,
    VersionConflictError,
)
from memory_mcp.identity import AgentContext
from memory_mcp.pagination import (
    Direction,
    compute_filter_fingerprint,
    decode_cursor,
    encode_cursor,
)
from memory_mcp.tasks.api import _acquire_dep_lock
from memory_mcp.tasks.cycles import would_cycle

log = logging.getLogger(__name__)

__all__ = [
    "RelationBrowseHit",
    "RelationBrowseRequest",
    "RelationBrowseResponse",
    "RelationEndpoint",
    "RelationLinkRequest",
    "RelationResponse",
    "relation_browse",
    "relation_link",
]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_env_id(*, explicit: UUID | None, ctx: AgentContext) -> UUID:
    if explicit is not None:
        return explicit
    attached = list(dict.fromkeys(ctx.attached_env_ids))
    if len(attached) == 1:
        return attached[0]
    raise EnvAmbiguousError(
        "no unambiguous env to write to; pass env_id or attach exactly one env",
        attached=[str(e) for e in attached],
    )


async def _create_or_get_graph_node(
    session,
    *,
    candidate: GraphNode,
    re_select_stmt: Select,
) -> GraphNode:
    """Idempotent INSERT for a ``graph_nodes`` row.

    The three partial-unique indexes ``graph_nodes_{memory,entity,task}_uniq``
    can fire under concurrent ``rel_link`` calls that share an endpoint
    (e.g. fan-out from one src to N dsts on N parallel sessions). Each
    session SELECTs ``None``, both attempt INSERT, the loser hits the
    unique violation. We absorb that race by wrapping the INSERT in a
    SAVEPOINT and re-fetching the winner's row on ``IntegrityError``.

    Caller passes the prepared ``candidate`` (already constructed with
    env/type/target fields) and a ``re_select_stmt`` that locates the
    winner by the same partial-unique predicate. The candidate is
    added *inside* the savepoint so its lifecycle is fully contained;
    on ``IntegrityError`` the savepoint rolls back cleanly and the
    re-SELECT runs under ``no_autoflush`` (sync context manager on
    ``AsyncSession``) so the ORM does not attempt to re-fire the failed
    INSERT.
    """
    try:
        async with session.begin_nested():
            session.add(candidate)
            await session.flush()
    except IntegrityError:
        # Savepoint rollback should expunge ``candidate`` (it was added
        # inside the nested block); guard defensively in case the
        # dialect leaves it attached.
        if candidate in session:
            session.expunge(candidate)
        with session.no_autoflush:
            return (await session.execute(re_select_stmt)).scalar_one()
    await session.refresh(candidate)
    return candidate


async def _ensure_graph_node(
    session,
    *,
    env_id: UUID,
    endpoint: RelationEndpoint,
) -> GraphNode:
    """Fetch the ``graph_nodes`` row for an endpoint, creating one if absent.

    Verifies the referenced record (entity or memory) exists in
    ``env_id``. Cross-env endpoints raise ``ValueError``.
    Missing record raises :class:`NotFoundError`.
    """
    if endpoint.kind == "entity":
        ent = (await session.execute(select(Entity).where(Entity.id == endpoint.id))).scalar_one_or_none()
        if ent is None:
            raise NotFoundError(f"entity {endpoint.id} not found", entity_id=str(endpoint.id))
        if ent.env_id != env_id:
            raise ValueError(f"entity {endpoint.id} is in env {ent.env_id}, not relation env {env_id}")
        stmt = select(GraphNode).where(GraphNode.entity_id == endpoint.id)
        node = (await session.execute(stmt)).scalar_one_or_none()
        if node is None:
            node = await _create_or_get_graph_node(
                session,
                candidate=GraphNode(
                    env_id=env_id,
                    node_type="entity",
                    entity_id=endpoint.id,
                ),
                re_select_stmt=stmt,
            )
        return node

    if endpoint.kind == "memory":
        mem = (await session.execute(select(Memory).where(Memory.id == endpoint.id))).scalar_one_or_none()
        if mem is None:
            raise NotFoundError(f"memory {endpoint.id} not found", memory_id=str(endpoint.id))
        if mem.env_id != env_id:
            raise ValueError(f"memory {endpoint.id} is in env {mem.env_id}, not relation env {env_id}")
        stmt = select(GraphNode).where(GraphNode.memory_id == endpoint.id)
        node = (await session.execute(stmt)).scalar_one_or_none()
        if node is None:
            node = await _create_or_get_graph_node(
                session,
                candidate=GraphNode(
                    env_id=env_id,
                    node_type="memory",
                    memory_id=endpoint.id,
                ),
                re_select_stmt=stmt,
            )
        return node

    stmt = select(GraphNode).where(GraphNode.task_id == endpoint.id)
    task = (await session.execute(select(Task).where(Task.id == endpoint.id))).scalar_one_or_none()
    if task is None:
        raise NotFoundError(f"task {endpoint.id} not found", task_id=str(endpoint.id))
    if task.env_id != env_id:
        raise ValueError(f"task {endpoint.id} is in env {task.env_id}, not relation env {env_id}")
    node = (await session.execute(stmt)).scalar_one_or_none()
    if node is None:
        node = await _create_or_get_graph_node(
            session,
            candidate=GraphNode(env_id=env_id, node_type="task", task_id=endpoint.id),
            re_select_stmt=stmt,
        )
    return node


def _endpoint_for_node(node: GraphNode) -> RelationEndpoint:
    if node.node_type == "entity":
        return RelationEndpoint(kind="entity", id=node.entity_id)  # type: ignore[arg-type]
    if node.node_type == "task":
        return RelationEndpoint(kind="task", id=node.task_id)  # type: ignore[arg-type]
    return RelationEndpoint(kind="memory", id=node.memory_id)  # type: ignore[arg-type]


def _validate_relation_type_for_endpoints(
    relation_type: str,
    src: RelationEndpoint,
    dst: RelationEndpoint,
) -> None:
    if relation_type == "depends_on":
        if src.kind != "task" or dst.kind != "task":
            raise InvalidInputError("depends_on relations require task-to-task endpoints")
        return
    if (src.kind == "task" or dst.kind == "task") and relation_type not in {
        "motivated_by",
        "produces",
        "references",
    }:
        raise InvalidInputError("task endpoint relations must use motivated_by, produces, or references")


def _relation_payload(
    relation: Relation,
    src_node: GraphNode,
    dst_node: GraphNode,
) -> dict[str, Any]:
    src_ep = _endpoint_for_node(src_node)
    dst_ep = _endpoint_for_node(dst_node)
    return {
        "relation_id": str(relation.id),
        "env_id": str(relation.env_id),
        "type": relation.type,
        "properties": dict(relation.properties or {}),
        "src": {
            "kind": src_ep.kind,
            "id": str(src_ep.id),
            "node_id": str(src_node.id),
        },
        "dst": {
            "kind": dst_ep.kind,
            "id": str(dst_ep.id),
            "node_id": str(dst_node.id),
        },
        "version": relation.version,
        "created_at": relation.created_at.isoformat() if relation.created_at else None,
        "updated_at": relation.updated_at.isoformat() if relation.updated_at else None,
    }


def _relation_to_response(
    relation: Relation,
    src_node: GraphNode,
    dst_node: GraphNode,
) -> RelationResponse:
    return RelationResponse(
        id=relation.id,
        env_id=relation.env_id,
        src=_endpoint_for_node(src_node),
        dst=_endpoint_for_node(dst_node),
        src_node_id=src_node.id,
        dst_node_id=dst_node.id,
        type=relation.type,
        properties=dict(relation.properties or {}),
        version=relation.version,
        created_at=relation.created_at,
        updated_at=relation.updated_at,
    )


async def _record_relation_audit(
    session,
    *,
    op: str,
    relation_id: UUID,
    env_id: UUID,
    by_agent_id: UUID | None,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> None:
    audit = AuditLog(
        record_type="relation",
        record_id=relation_id,
        env_id=env_id,
        op=op,
        by_agent_id=by_agent_id,
        before=before,
        after=after,
    )
    session.add(audit)
    await session.flush()


# ---------------------------------------------------------------------------
# relation_link
# ---------------------------------------------------------------------------


async def relation_link(
    request: RelationLinkRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> RelationResponse:
    """Create or update a typed edge between two graph nodes.

    Behavior:

    * Both endpoints must reference records in the same env (the
      relation's env). Cross-env edges are rejected.
    * Self-loops (``src == dst``) are rejected.
    * If no relation row exists for ``(src_node_id, dst_node_id, type)``
      → INSERT new row (version=1) + audit (``op=create``) + outbox
      (``op=upsert``).
    * If an existing row's ``properties`` matches the request → no-op
      (no UPDATE, no version bump, no outbox event). Idempotent.
    * If properties differ → optimistic UPDATE WHERE version=:expected;
      bump version; audit (``op=update``); outbox (``op=update``).
      Mismatch on ``expected_version`` (when supplied) or rowcount=0
      → :class:`VersionConflictError`.

    Outbox sink resolution depends on ``settings.graph_backend``;
    when ``postgres`` (default in v1 minimal deployment) the outbox
    write is a no-op.
    """
    settings = settings or get_settings()
    env_id = _resolve_env_id(explicit=request.env_id, ctx=ctx)
    rbac.require("write", env_id, ctx)

    if request.src.kind == request.dst.kind and request.src.id == request.dst.id:
        raise ValueError("self-loop relations are not allowed")
    _validate_relation_type_for_endpoints(request.type, request.src, request.dst)

    async with session_scope() as s:
        src_node = await _ensure_graph_node(s, env_id=env_id, endpoint=request.src)
        dst_node = await _ensure_graph_node(s, env_id=env_id, endpoint=request.dst)
        if request.type == "depends_on" and request.src.kind == "task" and request.dst.kind == "task":
            await _acquire_dep_lock(s, env_id)
            if await would_cycle(s, env_id, request.src.id, request.dst.id):
                raise CycleDetectedError("relation_link depends_on would create a dependency cycle")

        existing = (
            await s.execute(
                select(Relation).where(
                    Relation.src_node_id == src_node.id,
                    Relation.dst_node_id == dst_node.id,
                    Relation.type == request.type,
                )
            )
        ).scalar_one_or_none()

        is_create = existing is None
        emit_outbox = False
        outbox_op = OutboxOp.upsert
        before_snap: dict[str, Any] | None = None

        if existing is None:
            relation = Relation(
                env_id=env_id,
                src_node_id=src_node.id,
                dst_node_id=dst_node.id,
                type=request.type,
                properties=request.properties,
            )
            s.add(relation)
            await s.flush()
            await s.refresh(relation)
            emit_outbox = True
            outbox_op = OutboxOp.upsert
        else:
            relation = existing
            before_snap = _relation_payload(relation, src_node, dst_node)
            if request.expected_version is not None and relation.version != request.expected_version:
                raise VersionConflictError(
                    expected=request.expected_version,
                    actual=relation.version,
                )

            if dict(relation.properties or {}) != request.properties:
                expected_v = relation.version
                result = await s.execute(
                    update(Relation)
                    .where(
                        Relation.id == relation.id,
                        Relation.version == expected_v,
                    )
                    .values(
                        {
                            Relation.properties: request.properties,
                            Relation.version: expected_v + 1,
                            Relation.updated_at: func.now(),
                        }
                    )
                )
                if result.rowcount == 0:  # type: ignore[attr-defined]
                    raise VersionConflictError(expected=expected_v, actual=expected_v + 1)
                await s.refresh(relation)
                emit_outbox = True
                outbox_op = OutboxOp.update

        after_snap = _relation_payload(relation, src_node, dst_node)
        if is_create or emit_outbox or before_snap != after_snap:
            await _record_relation_audit(
                s,
                op="create" if is_create else "update",
                relation_id=relation.id,
                env_id=env_id,
                by_agent_id=ctx.agent_id,
                before=before_snap,
                after=after_snap,
            )

        if emit_outbox:
            await enqueue_event(
                s,
                aggregate_type=OutboxAggregateType.relation,
                aggregate_id=relation.id,
                aggregate_version=relation.version,
                env_id=env_id,
                op=outbox_op,
                payload=after_snap,
                settings=settings,
            )

    return _relation_to_response(relation, src_node, dst_node)


# ---------------------------------------------------------------------------
# relation_browse (Sprint A exploration API)
# ---------------------------------------------------------------------------


from sqlalchemy.orm import aliased  # noqa: E402

"""Max number of distinct edge ``type`` values accepted by
:func:`relation_browse` in one call. Mirrors the cap used by
:class:`memory_mcp.graph.EntityNeighborsRequest`.
"""


def _resolve_browse_env_ids(
    explicit: list[UUID] | None,
    ctx: AgentContext,
) -> list[UUID]:
    if explicit:
        return list(dict.fromkeys(explicit))
    return list(dict.fromkeys(ctx.attached_env_ids))


def _relation_browse_filter_dict(
    req: RelationBrowseRequest,
    env_ids: list[UUID],
) -> dict[str, Any]:
    return {
        "env_ids": list(env_ids),
        "types": sorted(set(req.types)) if req.types else None,
        "src_kind": req.src_kind,
        "dst_kind": req.dst_kind,
        "src_id": str(req.src_id) if req.src_id else None,
        "dst_id": str(req.dst_id) if req.dst_id else None,
        "created_after": req.created_after,
        "descending": req.descending,
    }


async def relation_browse(
    request: RelationBrowseRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> RelationBrowseResponse:
    """Keyset-paginated listing of relations across attached envs.

    Backed by a single SQL query joining ``relations`` to two aliases
    of ``graph_nodes`` (one per endpoint). The new
    ``idx_relations_env_type_id`` covers the common (env, type, id)
    filter path; tiebreak is on ``relations.id`` (UUID monotonic
    ordering is irrelevant — only stability matters for cursors).
    """
    _ = settings or get_settings()
    env_ids = _resolve_browse_env_ids(request.env_ids, ctx)
    for env_id in env_ids:
        rbac.require("read", env_id, ctx)

    if not env_ids:
        return RelationBrowseResponse(hits=[], next_cursor=None, has_more=False)

    filter_dict = _relation_browse_filter_dict(request, env_ids)
    fingerprint = compute_filter_fingerprint(filter_dict)
    direction: Direction = "desc" if request.descending else "asc"

    cursor_value: dt.datetime | None = None
    cursor_id: UUID | None = None
    if request.cursor:
        cur = decode_cursor(
            request.cursor,
            expected_fingerprint=fingerprint,
            expected_order_field="created_at",
            expected_direction=direction,
        )
        cursor_id = cur.tiebreak_id
        try:
            cursor_value = dt.datetime.fromisoformat(cur.order_value)
        except ValueError as exc:
            raise InvalidCursorError(
                f"INVALID_CURSOR: cursor order_value is not ISO-8601: {cur.order_value!r}",
            ) from exc

    src_node = aliased(GraphNode, name="src_node")
    dst_node = aliased(GraphNode, name="dst_node")

    async with session_scope() as session:
        stmt: Select[Any] = (
            select(Relation, src_node, dst_node)
            .join(src_node, src_node.id == Relation.src_node_id)
            .join(dst_node, dst_node.id == Relation.dst_node_id)
            .where(Relation.env_id.in_(env_ids))
        )
        if request.types:
            stmt = stmt.where(Relation.type.in_(list(set(request.types))))
        if request.src_kind is not None:
            stmt = stmt.where(src_node.node_type == request.src_kind)
        if request.dst_kind is not None:
            stmt = stmt.where(dst_node.node_type == request.dst_kind)
        if request.src_id is not None:
            stmt = stmt.where(
                and_(
                    src_node.node_type == (request.src_kind or src_node.node_type),
                    (src_node.entity_id == request.src_id)
                    | (src_node.memory_id == request.src_id)
                    | (src_node.task_id == request.src_id),
                )
            )
        if request.dst_id is not None:
            stmt = stmt.where(
                and_(
                    dst_node.node_type == (request.dst_kind or dst_node.node_type),
                    (dst_node.entity_id == request.dst_id)
                    | (dst_node.memory_id == request.dst_id)
                    | (dst_node.task_id == request.dst_id),
                )
            )
        if request.created_after is not None:
            stmt = stmt.where(Relation.created_at >= request.created_after)

        if cursor_value is not None and cursor_id is not None:
            if request.descending:
                stmt = stmt.where(tuple_(Relation.created_at, Relation.id) < tuple_(cursor_value, cursor_id))
            else:
                stmt = stmt.where(tuple_(Relation.created_at, Relation.id) > tuple_(cursor_value, cursor_id))
        if request.descending:
            stmt = stmt.order_by(Relation.created_at.desc(), Relation.id.desc())
        else:
            stmt = stmt.order_by(Relation.created_at.asc(), Relation.id.asc())

        stmt = stmt.limit(request.limit + 1)

        rows = (await session.execute(stmt)).all()
        page = list(rows[: request.limit])
        has_more = len(rows) > request.limit

    hits: list[RelationBrowseHit] = []
    for rel, sn, dn in page:
        src_kind = sn.node_type
        dst_kind = dn.node_type
        src_id = sn.entity_id if src_kind == "entity" else sn.memory_id
        dst_id = dn.entity_id if dst_kind == "entity" else dn.memory_id
        if src_id is None or dst_id is None:
            # graph_nodes CHECK constraint should make this impossible.
            continue
        hits.append(
            RelationBrowseHit(
                id=rel.id,
                env_id=rel.env_id,
                type=rel.type,
                src_kind=src_kind,
                src_id=src_id,
                dst_kind=dst_kind,
                dst_id=dst_id,
                properties=dict(rel.properties or {}),
                created_at=rel.created_at,
                updated_at=rel.updated_at,
            )
        )

    next_cursor: str | None = None
    if has_more and page:
        last_rel = page[-1][0]
        next_cursor = encode_cursor(
            filter_fingerprint=fingerprint,
            order_field="created_at",
            order_value=last_rel.created_at,
            tiebreak_id=last_rel.id,
            direction=direction,
        )

    return RelationBrowseResponse(hits=hits, next_cursor=next_cursor, has_more=has_more)
