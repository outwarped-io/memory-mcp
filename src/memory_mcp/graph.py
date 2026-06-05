"""Graph-traversal MCP tools — entity neighborhood exploration.

The ``ent_neighbors`` tool wraps :meth:`GraphStore.neighbors` for client
agents that want to walk the projected entity graph. The underlying
backend (Neo4j or Postgres recursive CTE) is selected by configuration;
this module handles the request/response shaping, RBAC, env-scoping,
canonical-name resolution, lifecycle filtering, and cursor-error
translation.

Design notes
------------

* **Real-edge path orientation.** Each :class:`NeighborPathStepResponse`
  has ``src`` and ``dst`` fields matching the actual relation direction,
  *not* the traversal order. For ``direction="in"`` walks the path
  reads naturally as ``terminal → ... → start`` in real-edge terms.
  This contract is enforced for both backends (the Postgres
  ``WITH RECURSIVE`` step always carries ``e.src_node_id`` and
  ``e.dst_node_id`` regardless of traversal direction).

* **Memory lifecycle filter.** Hits whose terminal *or* path-transit
  contains a memory in a non-default-visible status (``archived``,
  ``superseded``, ``retired``) — or a memory whose canonical row is
  missing from Postgres — are dropped from the response. This matches
  the search-default visibility rule and prevents leaking IDs of
  hidden memories through the graph leg.

* **Sparse pages.** ``limit`` is the *pre-filter* cap on backend
  results; the response may carry fewer hits when many neighbors are
  filtered out by the lifecycle rule. Clients should iterate
  ``next_cursor`` until ``None`` rather than treat a short page as
  end-of-stream.

* **Self-as-neighbor.** Cycles can surface the start entity as its own
  neighbor (``A → B → A`` paths for ``hops >= 2``). These are
  unconditionally suppressed.

* **Cursor errors.** Backend ``ValueError`` from
  :meth:`GraphStore.neighbors` (invalid base64, query-shape mismatch,
  out-of-range params) is translated to
  :class:`memory_mcp.errors.InvalidCursorError` so MCP callers see a
  stable ``INVALID_CURSOR`` code rather than ``INTERNAL``.

* **Singleton lifecycle.** The configured :class:`GraphStore` is held
  as a process-wide singleton (matching the pattern in
  :mod:`memory_mcp.search.api`). First-call construction is guarded by
  an :class:`asyncio.Lock` so two concurrent callers cannot leak a
  Neo4j driver pool. The FastAPI lifespan calls
  :func:`_close_default_graph_store` on shutdown.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any, Literal
from uuid import UUID

from memory_mcp_schemas.graph import (
    EntityNeighborsRequest,
    EntityNeighborsResponse,
    MemNeighborsRequest,
    MemNeighborsResponse,
    MemRelatedHit,
    MemRelatedRequest,
    MemRelatedResponse,
    NeighborHitResponse,
    NeighborNodeResponse,
    NeighborPathStepResponse,
)
from sqlalchemy import and_, distinct, func, or_, select, tuple_
from sqlalchemy.orm import aliased

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.graph import (
    GraphNodeRef,
    GraphStore,
    NeighborHit,
    get_graph_store,
)
from memory_mcp.db.models import Entity, GraphNode, Memory, MemoryTag, Relation, Tag
from memory_mcp.db.postgres import get_session_factory, session_scope
from memory_mcp.db.types import OutboxSink
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.errors import InvalidCursorError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryResponse, _to_response
from memory_mcp.pagination import (
    compute_filter_fingerprint,
    decode_cursor,
    encode_cursor,
)
from memory_mcp.search.api import (
    _capture_watermarks,
    _default_vector_store,
    _wait_for_watermarks,
)

log = logging.getLogger(__name__)

# Memories in these statuses are visible by default to graph-traversal
# callers. Mirrors the default visibility used by ``mem_search`` (see
# ``memory_mcp.search.api``). Hits whose terminal or path-transit
# references a memory in a non-default-visible status are dropped.
_DEFAULT_VISIBLE_MEMORY_STATUSES: frozenset[str] = frozenset({"proposed", "active", "stale"})


__all__ = [
    "EntityNeighborsRequest",
    "EntityNeighborsResponse",
    "MemNeighborsRequest",
    "MemNeighborsResponse",
    "MemRelatedHit",
    "MemRelatedRequest",
    "MemRelatedResponse",
    "NeighborHitResponse",
    "NeighborNodeResponse",
    "NeighborPathStepResponse",
    "entity_neighbors",
    "memory_neighbors",
    "memory_related",
]


# ---------------------------------------------------------------------------
# Internal projection helpers (pure — testable without a DB)
# ---------------------------------------------------------------------------


def _collect_record_ids(
    hits: Sequence[NeighborHit],
) -> tuple[set[UUID], set[UUID]]:
    """Walk every hit's terminal + path nodes and bucket record ids by kind."""
    entity_ids: set[UUID] = set()
    memory_ids: set[UUID] = set()
    for hit in hits:
        target = entity_ids if hit.node.kind == "entity" else memory_ids
        target.add(hit.node.record_id)
        for step in hit.path:
            (entity_ids if step.src.kind == "entity" else memory_ids).add(step.src.record_id)
            (entity_ids if step.dst.kind == "entity" else memory_ids).add(step.dst.record_id)
    return entity_ids, memory_ids


def _is_memory_visible(
    mem_id: UUID,
    memories_by_id: dict[UUID, tuple[str | None, str]],
    *,
    include_retired: bool = False,
) -> bool:
    info = memories_by_id.get(mem_id)
    if info is None:
        return False
    if include_retired:
        return True
    _, status = info
    return status in _DEFAULT_VISIBLE_MEMORY_STATUSES


def _project_hits(
    *,
    hits: Sequence[NeighborHit],
    start_entity_id: UUID | None = None,
    start_memory_id: UUID | None = None,
    env_id: UUID,
    entities_by_id: dict[UUID, str],
    memories_by_id: dict[UUID, tuple[str | None, str]],
    include_retired: bool = False,
) -> list[NeighborHitResponse]:
    """Apply lifecycle filter, self-cycle removal, and name resolution.

    Pure function. Drops a hit if any of these are true:

    * Terminal is the start entity (self-cycle from ``hops >= 2``).
    * Terminal or any path-transit node is a memory whose canonical
      status is not in ``_DEFAULT_VISIBLE_MEMORY_STATUSES`` (or whose
      canonical row is missing entirely).
    * Terminal is an entity whose canonical row is missing.

    Surviving hits are projected to the wire-shape with real-edge
    orientation in path steps.
    """
    out: list[NeighborHitResponse] = []
    for hit in hits:
        if hit.node.kind == "entity" and start_entity_id is not None and hit.node.record_id == start_entity_id:
            continue
        if hit.node.kind == "memory" and start_memory_id is not None and hit.node.record_id == start_memory_id:
            continue

        if hit.node.kind == "memory" and not _is_memory_visible(
            hit.node.record_id,
            memories_by_id,
            include_retired=include_retired,
        ):
            continue

        path_visible = True
        for step in hit.path:
            if step.src.kind == "memory" and not _is_memory_visible(
                step.src.record_id,
                memories_by_id,
                include_retired=include_retired,
            ):
                path_visible = False
                break
            if step.dst.kind == "memory" and not _is_memory_visible(
                step.dst.record_id,
                memories_by_id,
                include_retired=include_retired,
            ):
                path_visible = False
                break
        if not path_visible:
            continue

        if hit.node.kind == "entity":
            name = entities_by_id.get(hit.node.record_id)
            if name is None:
                log.debug(
                    "entity_neighbors: terminal entity %s missing canonical row",
                    hit.node.record_id,
                )
                continue
        else:
            name = memories_by_id[hit.node.record_id][0]

        out.append(
            NeighborHitResponse(
                node=NeighborNodeResponse(
                    kind=hit.node.kind,
                    id=hit.node.record_id,
                    name=name,
                    env_id=env_id,
                ),
                path_length=hit.path_length,
                path=[
                    NeighborPathStepResponse(
                        src_kind=step.src.kind,
                        src_id=step.src.record_id,
                        dst_kind=step.dst.kind,
                        dst_id=step.dst.record_id,
                        edge_type=step.edge_type,
                    )
                    for step in hit.path
                ],
                score=hit.score,
            )
        )
    return out


async def _finalize_neighbor_hits(
    hits: Sequence[NeighborHit],
    *,
    env_id: UUID,
    start_entity_id: UUID | None = None,
    start_memory_id: UUID | None = None,
    include_retired: bool = False,
) -> list[NeighborHitResponse]:
    """Resolve graph hits against Postgres, then filter and project them.

    Shared by ``entity_neighbors`` and ``memory_neighbors`` so the lifecycle
    filter, canonical-row suppression, and path projection stay identical for
    entity-rooted and memory-rooted graph walks.
    """
    entity_ids_to_lookup, memory_ids_to_lookup = _collect_record_ids(hits)

    entities_by_id: dict[UUID, str] = {}
    memories_by_id: dict[UUID, tuple[str | None, str]] = {}
    if entity_ids_to_lookup or memory_ids_to_lookup:
        async with session_scope() as s:
            if entity_ids_to_lookup:
                rows = (
                    await s.execute(
                        select(Entity.id, Entity.canonical_name).where(
                            Entity.id.in_(entity_ids_to_lookup),
                            Entity.env_id == env_id,
                        )
                    )
                ).all()
                entities_by_id = {row[0]: row[1] for row in rows}
            if memory_ids_to_lookup:
                rows = (
                    await s.execute(
                        select(Memory.id, Memory.title, Memory.status).where(
                            Memory.id.in_(memory_ids_to_lookup),
                            Memory.env_id == env_id,
                        )
                    )
                ).all()
                memories_by_id = {row[0]: (row[1], str(row[2])) for row in rows}

    return _project_hits(
        hits=hits,
        start_entity_id=start_entity_id,
        start_memory_id=start_memory_id,
        env_id=env_id,
        entities_by_id=entities_by_id,
        memories_by_id=memories_by_id,
        include_retired=include_retired,
    )


def _parse_shared_entity_cursor_value(raw: str) -> tuple[int, dt.datetime]:
    try:
        overlap_raw, updated_at_raw = raw.split("|", 1)
        return int(overlap_raw), dt.datetime.fromisoformat(updated_at_raw)
    except (ValueError, TypeError) as exc:
        raise InvalidCursorError(
            f"INVALID_CURSOR: bad shared_entity order_value: {raw!r}",
        ) from exc


def _decode_shared_entity_cursor(
    raw: str,
    *,
    fingerprint: str,
) -> tuple[int, dt.datetime, UUID]:
    cur = decode_cursor(
        raw,
        expected_fingerprint=fingerprint,
        expected_order_field="overlap_updated_at",
        expected_direction="desc",
    )
    overlap, updated_at = _parse_shared_entity_cursor_value(cur.order_value)
    return overlap, updated_at, cur.tiebreak_id


def _decode_semantic_cursor(
    raw: str,
    *,
    fingerprint: str,
) -> tuple[int, UUID]:
    cur = decode_cursor(
        raw,
        expected_fingerprint=fingerprint,
        expected_order_field="score_bucket",
        expected_direction="desc",
    )
    try:
        return int(cur.order_value), cur.tiebreak_id
    except ValueError as exc:
        raise InvalidCursorError(
            f"INVALID_CURSOR: bad semantic score bucket: {cur.order_value!r}",
        ) from exc


async def _hydrate_memory_responses(
    session: Any,
    memory_ids: Sequence[UUID],
    *,
    env_id: UUID,
    include_retired: bool = False,
) -> dict[UUID, MemoryResponse]:
    """Bulk-load memories plus tags and project to ``MemoryResponse``."""
    ids = list(dict.fromkeys(memory_ids))
    if not ids:
        return {}
    clauses = [
        Memory.id.in_(ids),
        Memory.env_id == env_id,
    ]
    if not include_retired:
        clauses.append(Memory.status.in_(list(_DEFAULT_VISIBLE_MEMORY_STATUSES)))
    rows = (await session.execute(select(Memory).where(*clauses))).scalars().all()
    memories_by_id = {m.id: m for m in rows}
    tag_rows = (
        await session.execute(
            select(MemoryTag.memory_id, Tag.name)
            .join(Tag, Tag.id == MemoryTag.tag_id)
            .where(MemoryTag.memory_id.in_(list(memories_by_id)))
            .order_by(MemoryTag.memory_id, Tag.name)
        )
    ).all()
    tags_by_id: dict[UUID, list[str]] = {mid: [] for mid in memories_by_id}
    for mid, name in tag_rows:
        tags_by_id[mid].append(name)
    return {mid: _to_response(memory, tags_by_id.get(mid, [])) for mid, memory in memories_by_id.items()}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GraphStore singleton
# ---------------------------------------------------------------------------


_DEFAULT_GRAPH_STORE: GraphStore | None = None
_DEFAULT_GRAPH_STORE_LOCK: asyncio.Lock = asyncio.Lock()


async def _get_default_graph_store(settings: Settings) -> GraphStore:
    """Lazily construct (and cache) the configured :class:`GraphStore`.

    The lock prevents two concurrent first-callers from each
    instantiating a Neo4j driver pool — the loser would be GC'd
    without :meth:`GraphStore.close`. The double-check around the lock
    keeps the steady-state path lock-free.
    """
    global _DEFAULT_GRAPH_STORE
    if _DEFAULT_GRAPH_STORE is not None:
        return _DEFAULT_GRAPH_STORE
    async with _DEFAULT_GRAPH_STORE_LOCK:
        if _DEFAULT_GRAPH_STORE is None:
            _DEFAULT_GRAPH_STORE = get_graph_store(settings)
        return _DEFAULT_GRAPH_STORE


async def _close_default_graph_store() -> None:
    """Release the singleton's resources (driver pools, sessions).

    Called from the FastAPI lifespan shutdown hook. Idempotent.
    """
    global _DEFAULT_GRAPH_STORE
    if _DEFAULT_GRAPH_STORE is not None:
        store, _DEFAULT_GRAPH_STORE = _DEFAULT_GRAPH_STORE, None
        try:
            await store.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            log.exception("graph store close() failed during shutdown")


def _reset_default_graph_store_for_tests() -> None:
    """Test hook — clear the singleton so tests can inject fakes.

    Tests pass an explicit ``graph_store=`` kwarg to
    :func:`entity_neighbors`; this hook is for tests that exercise the
    lazy-init path itself.
    """
    global _DEFAULT_GRAPH_STORE
    _DEFAULT_GRAPH_STORE = None


def _step_widen_hops(
    request: MemNeighborsRequest | MemRelatedRequest,
) -> MemNeighborsRequest | MemRelatedRequest | None:
    hops = getattr(request, "hops", None)
    if not isinstance(hops, int) or hops >= 3:
        return None
    return request.model_copy(update={"hops": min(hops + 1, 3)})


def _step_drop_predicate(
    request: MemNeighborsRequest | MemRelatedRequest,
) -> MemNeighborsRequest | MemRelatedRequest | None:
    for field_name in ("predicate", "relation_types", "edge_types"):
        if not hasattr(request, field_name):
            continue
        value = getattr(request, field_name)
        if value is None or value == [] or value == "":
            continue
        return request.model_copy(update={field_name: None})
    return None


def _apply_related_min_score(
    response: MemRelatedResponse,
    *,
    min_score: float | None,
) -> MemRelatedResponse:
    if min_score is None:
        return response
    hits = [hit for hit in response.hits if hit.score >= min_score]
    if len(hits) == len(response.hits):
        return response
    return response.model_copy(update={"hits": hits})


# ---------------------------------------------------------------------------
# entity_neighbors
# ---------------------------------------------------------------------------


async def entity_neighbors(
    request: EntityNeighborsRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
    graph_store: GraphStore | None = None,
) -> EntityNeighborsResponse:
    """Walk the projected graph from ``request.entity_id``.

    Steps:

    1. Resolve the start entity to ``(env_id, canonical_name)``;
       :class:`NotFoundError` if missing or if a non-matching
       ``request.env_id`` was supplied.
    2. Call :meth:`GraphStore.neighbors` with the request parameters.
       Translate backend ``ValueError`` (cursor mismatch, malformed
       cursor) to :class:`InvalidCursorError`.
    3. Bulk-resolve canonical names for every entity / memory id
       referenced in the result set (terminal node + every path step's
       endpoints).
    4. Filter hits whose terminal or path-transit references a memory
       outside the default-visible status set (``proposed``, ``active``,
       ``stale``), or whose canonical row is missing.
    5. Drop self-as-neighbor hits (start entity reached via cycles).
    6. Project ``GraphPathStep`` instances to
       :class:`NeighborPathStepResponse` preserving real-edge orientation.
    """
    settings = settings or get_settings()

    async with session_scope() as s:
        entity = await s.get(Entity, request.entity_id)
        if entity is None:
            raise NotFoundError(
                f"entity {request.entity_id} not found",
                entity_id=str(request.entity_id),
            )
        env_id = entity.env_id
        if request.env_id is not None and request.env_id != env_id:
            # Don't leak which env the entity actually lives in.
            raise NotFoundError(
                f"entity {request.entity_id} not found",
                entity_id=str(request.entity_id),
            )

    rbac.require("read", env_id, ctx)

    kinds: list[Literal["entity", "memory"]] | None = None if request.kind == "both" else [request.kind]

    gs = graph_store or await _get_default_graph_store(settings)
    start = GraphNodeRef(
        env_id=env_id,
        kind="entity",
        record_id=request.entity_id,
    )

    # consistency=fresh: snapshot the env's outbox watermark and wait
    # for the neo4j sink to catch up. Only relevant for the Neo4j
    # backend — the Postgres recursive-CTE fallback reads the canonical
    # ``relations`` table directly and is therefore already consistent.
    if request.consistency == "fresh" and settings.graph_backend == "neo4j":
        async with session_scope() as ws:
            watermarks = await _capture_watermarks(ws, [env_id])
        await _wait_for_watermarks(
            get_session_factory(),
            watermarks,
            max_wait_seconds=settings.search_fresh_max_wait_seconds,
            sinks=(OutboxSink.neo4j,),
        )

    try:
        hits, next_cursor = await gs.neighbors(
            start,
            hops=request.hops,
            direction=request.direction,
            edge_types=request.edge_types,
            kinds=kinds,
            limit=request.limit,
            cursor=request.cursor,
        )
    except ValueError as exc:
        raise InvalidCursorError(
            f"cursor or pagination parameter rejected: {exc}",
            cursor=request.cursor,
        ) from exc

    response_hits = await _finalize_neighbor_hits(
        hits,
        env_id=env_id,
        start_entity_id=request.entity_id,
    )

    return EntityNeighborsResponse(
        hits=response_hits,
        next_cursor=next_cursor,
    )


# ---------------------------------------------------------------------------
# memory_neighbors
# ---------------------------------------------------------------------------


async def _memory_neighbors_pass(
    request: MemNeighborsRequest,
    *,
    env_id: UUID,
    graph_store: GraphStore,
    include_retired: bool = False,
) -> MemNeighborsResponse:
    kinds: list[Literal["entity", "memory"]] | None = None if request.kind == "both" else [request.kind]
    start = GraphNodeRef(
        env_id=env_id,
        kind="memory",
        record_id=request.memory_id,
    )

    try:
        hits, next_cursor = await graph_store.neighbors(
            start,
            hops=request.hops,
            direction=request.direction,
            edge_types=request.edge_types,
            kinds=kinds,
            limit=request.limit,
            cursor=request.cursor,
        )
    except ValueError as exc:
        raise InvalidCursorError(
            f"cursor or pagination parameter rejected: {exc}",
            cursor=request.cursor,
        ) from exc

    response_hits = await _finalize_neighbor_hits(
        hits,
        env_id=env_id,
        start_memory_id=request.memory_id,
        include_retired=include_retired,
    )
    return MemNeighborsResponse(hits=response_hits, next_cursor=next_cursor)


async def memory_neighbors(
    request: MemNeighborsRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
    graph_store: GraphStore | None = None,
) -> MemNeighborsResponse:
    """Walk the projected graph from ``request.memory_id``.

    Mirrors :func:`entity_neighbors`, including RBAC, env-mismatch
    suppression, ``consistency=fresh`` Neo4j watermark waiting, backend cursor
    translation, default-visible memory lifecycle filtering, and self-cycle
    suppression. A missing ``graph_nodes`` row is reported distinctly because
    memory provenance can still be inspected via source/lineage tools.
    """
    settings = settings or get_settings()

    async with session_scope() as s:
        memory = await s.get(Memory, request.memory_id)
        if memory is None:
            raise NotFoundError(
                f"memory {request.memory_id} not found",
                memory_id=str(request.memory_id),
            )
        env_id = memory.env_id
        if request.env_id is not None and request.env_id != env_id:
            raise NotFoundError(
                f"memory {request.memory_id} not found",
                memory_id=str(request.memory_id),
            )

        graph_node = (
            await s.execute(
                select(GraphNode).where(
                    GraphNode.env_id == env_id,
                    GraphNode.node_type == "memory",
                    GraphNode.memory_id == request.memory_id,
                )
            )
        ).scalar_one_or_none()
        if graph_node is None:
            raise NotFoundError(
                f"memory {request.memory_id} has no graph edges — use mem_sources_browse / mem_lineage instead",
                memory_id=str(request.memory_id),
            )

    rbac.require("read", env_id, ctx)
    gs = graph_store or await _get_default_graph_store(settings)

    if request.consistency == "fresh" and settings.graph_backend == "neo4j":
        async with session_scope() as ws:
            watermarks = await _capture_watermarks(ws, [env_id])
        await _wait_for_watermarks(
            get_session_factory(),
            watermarks,
            max_wait_seconds=settings.search_fresh_max_wait_seconds,
            sinks=(OutboxSink.neo4j,),
        )

    response = await _memory_neighbors_pass(
        request,
        env_id=env_id,
        graph_store=gs,
    )
    if response.hits or not request.fallback:
        return response

    fallback_used: list[str] = []
    cascade_request = request
    include_retired = False

    for step_name, builder in (
        ("widen_hops", _step_widen_hops),
        ("drop_predicate", _step_drop_predicate),
    ):
        next_request = builder(cascade_request)
        if next_request is None:
            continue
        cascade_request = next_request
        fallback_used.append(step_name)
        response = await _memory_neighbors_pass(
            cascade_request,
            env_id=env_id,
            graph_store=gs,
            include_retired=include_retired,
        )
        if response.hits:
            break

    if not response.hits and not include_retired:
        include_retired = True
        fallback_used.append("include_retired")
        response = await _memory_neighbors_pass(
            cascade_request,
            env_id=env_id,
            graph_store=gs,
            include_retired=True,
        )

    return response.model_copy(update={"fallback_used": fallback_used})


# ---------------------------------------------------------------------------
# memory_related
# ---------------------------------------------------------------------------


async def _resolve_seed_memory(
    memory_id: UUID,
    *,
    env_id: UUID | None,
    ctx: AgentContext,
) -> Memory:
    async with session_scope() as s:
        memory = await s.get(Memory, memory_id)
        if memory is None or (env_id is not None and memory.env_id != env_id):
            raise NotFoundError(
                f"memory {memory_id} not found",
                memory_id=str(memory_id),
            )
        resolved = memory

    rbac.require("read", resolved.env_id, ctx)
    return resolved


async def _memory_related_shared_entity(
    request: MemRelatedRequest,
    *,
    env_id: UUID,
    include_retired: bool = False,
) -> MemRelatedResponse:
    filter_fingerprint = compute_filter_fingerprint(
        {
            "memory_id": request.memory_id,
            "relation": request.relation,
            "env_id": env_id,
            "limit": request.limit,
        }
    )

    cursor_overlap: int | None = None
    cursor_updated_at: dt.datetime | None = None
    cursor_id: UUID | None = None
    if request.cursor:
        cursor_overlap, cursor_updated_at, cursor_id = _decode_shared_entity_cursor(
            request.cursor,
            fingerprint=filter_fingerprint,
        )

    seed_mem = aliased(GraphNode)
    seed_entity = aliased(GraphNode)
    other_mem = aliased(GraphNode)
    shared_entity = aliased(GraphNode)

    async with session_scope() as s:
        seed_entity_rows = (
            await s.execute(
                select(distinct(seed_entity.entity_id))
                .select_from(seed_mem)
                .join(
                    Relation,
                    and_(
                        Relation.env_id == env_id,
                        or_(
                            Relation.src_node_id == seed_mem.id,
                            Relation.dst_node_id == seed_mem.id,
                        ),
                    ),
                )
                .join(
                    seed_entity,
                    and_(
                        seed_entity.env_id == env_id,
                        seed_entity.node_type == "entity",
                        or_(
                            and_(
                                Relation.src_node_id == seed_entity.id,
                                Relation.dst_node_id == seed_mem.id,
                            ),
                            and_(
                                Relation.dst_node_id == seed_entity.id,
                                Relation.src_node_id == seed_mem.id,
                            ),
                        ),
                    ),
                )
                .where(
                    seed_mem.env_id == env_id,
                    seed_mem.node_type == "memory",
                    seed_mem.memory_id == request.memory_id,
                )
            )
        ).all()
        seed_entity_ids = [row[0] for row in seed_entity_rows if row[0] is not None]
        if not seed_entity_ids:
            return MemRelatedResponse(hits=[], next_cursor=None, note="ok")

        overlap = func.count(distinct(shared_entity.entity_id))
        shared_ids = func.array_agg(distinct(shared_entity.entity_id))

        stmt = (
            select(
                Memory.id.label("memory_id"),
                Memory.updated_at.label("updated_at"),
                overlap.label("overlap"),
                shared_ids.label("shared_ids"),
            )
            .select_from(Memory)
            .join(
                other_mem,
                and_(
                    other_mem.env_id == Memory.env_id,
                    other_mem.node_type == "memory",
                    other_mem.memory_id == Memory.id,
                ),
            )
            .join(
                Relation,
                and_(
                    Relation.env_id == env_id,
                    or_(
                        Relation.src_node_id == other_mem.id,
                        Relation.dst_node_id == other_mem.id,
                    ),
                ),
            )
            .join(
                shared_entity,
                and_(
                    shared_entity.env_id == env_id,
                    shared_entity.node_type == "entity",
                    shared_entity.entity_id.in_(seed_entity_ids),
                    or_(
                        and_(
                            Relation.src_node_id == shared_entity.id,
                            Relation.dst_node_id == other_mem.id,
                        ),
                        and_(
                            Relation.dst_node_id == shared_entity.id,
                            Relation.src_node_id == other_mem.id,
                        ),
                    ),
                ),
            )
            .where(
                Memory.env_id == env_id,
                Memory.id != request.memory_id,
            )
            .group_by(Memory.id, Memory.updated_at)
        )
        if not include_retired:
            stmt = stmt.where(Memory.status.in_(list(_DEFAULT_VISIBLE_MEMORY_STATUSES)))
        if cursor_overlap is not None and cursor_updated_at is not None and cursor_id is not None:
            stmt = stmt.having(
                or_(
                    overlap < cursor_overlap,
                    and_(
                        overlap == cursor_overlap,
                        tuple_(Memory.updated_at, Memory.id) < tuple_(cursor_updated_at, cursor_id),
                    ),
                )
            )

        stmt = stmt.order_by(overlap.desc(), Memory.updated_at.desc(), Memory.id.desc())
        rows = (await s.execute(stmt.limit(request.limit + 1))).all()
        page = rows[: request.limit]
        has_more = len(rows) > request.limit

        responses = await _hydrate_memory_responses(
            s,
            [row.memory_id for row in page],
            env_id=env_id,
            include_retired=include_retired,
        )

    hits = [
        MemRelatedHit(
            memory_id=row.memory_id,
            score=float(row.overlap),
            shared_entity_ids=list(row.shared_ids or []),
            memory=responses[row.memory_id],
        )
        for row in page
        if row.memory_id in responses
    ]

    next_cursor: str | None = None
    if has_more and page:
        last = page[-1]
        next_cursor = encode_cursor(
            filter_fingerprint=filter_fingerprint,
            order_field="overlap_updated_at",
            order_value=f"{int(last.overlap)}|{last.updated_at.isoformat()}",
            tiebreak_id=last.memory_id,
            direction="desc",
        )

    return MemRelatedResponse(hits=hits, next_cursor=next_cursor, note="ok")


async def _memory_related_semantic(
    request: MemRelatedRequest,
    *,
    env_id: UUID,
    settings: Settings,
    vector_store: VectorStore | None,
    include_retired: bool = False,
) -> MemRelatedResponse:
    if request.cursor:
        raise InvalidCursorError("INVALID_CURSOR: semantic mode does not support cursor pagination yet")

    try:
        vs = vector_store or _default_vector_store(settings)
        vec = await vs.get_vector(env_id=env_id, id=str(request.memory_id))
        if vec is None:
            return MemRelatedResponse(
                hits=[],
                next_cursor=None,
                note="no_embedding",
            )
        raw_results = await vs.search(
            env_id=env_id,
            query_vector=vec,
            limit=request.limit + 1,
            filters=(None if include_retired else {"status": list(_DEFAULT_VISIBLE_MEMORY_STATUSES)}),
        )
    except Exception as exc:  # noqa: BLE001 — vector search degrades to a note
        log.warning(
            "memory_related semantic vector store unavailable: %s",
            type(exc).__name__,
        )
        return MemRelatedResponse(
            hits=[],
            next_cursor=None,
            note="vector_store_unavailable",
        )

    scored: list[tuple[UUID, float, int]] = []
    for hit in raw_results:
        mid = UUID(str(hit["id"]))
        if mid == request.memory_id:
            continue
        score = float(hit["score"])
        bucket = int(round(score * 10_000))
        scored.append((mid, score, bucket))

    scored.sort(key=lambda item: (item[2], item[0]), reverse=True)
    page = scored[: request.limit]
    async with session_scope() as s:
        responses = await _hydrate_memory_responses(
            s,
            [mid for mid, _, _ in page],
            env_id=env_id,
            include_retired=include_retired,
        )

    hits = [
        MemRelatedHit(
            memory_id=mid,
            score=score,
            shared_entity_ids=None,
            memory=responses[mid],
        )
        for mid, score, _bucket in page
        if mid in responses
    ]

    return MemRelatedResponse(hits=hits, next_cursor=None, note="ok")


async def _memory_related_pass(
    request: MemRelatedRequest,
    *,
    env_id: UUID,
    settings: Settings,
    vector_store: VectorStore | None,
    include_retired: bool = False,
) -> MemRelatedResponse:
    if request.relation == "shared_entity":
        return await _memory_related_shared_entity(
            request,
            env_id=env_id,
            include_retired=include_retired,
        )
    return await _memory_related_semantic(
        request,
        env_id=env_id,
        settings=settings,
        vector_store=vector_store,
        include_retired=include_retired,
    )


async def memory_related(
    request: MemRelatedRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
    vector_store: VectorStore | None = None,
) -> MemRelatedResponse:
    """Find memories related to ``request.memory_id``.

    ``shared_entity`` uses canonical Postgres graph-node/relation rows and a
    keyset cursor over overlap + updated_at + id. ``semantic`` uses the seed's
    stored vector via :meth:`VectorStore.get_vector`; it never calls an embedder.
    Semantic cursor pagination is a v1 limitation: pass a larger ``limit``
    (up to 500) instead.
    """
    settings = settings or get_settings()
    seed = await _resolve_seed_memory(
        request.memory_id,
        env_id=request.env_id,
        ctx=ctx,
    )
    env_id = seed.env_id

    response = _apply_related_min_score(
        await _memory_related_pass(
            request,
            env_id=env_id,
            settings=settings,
            vector_store=vector_store,
        ),
        min_score=request.min_score,
    )
    if response.hits or not request.fallback:
        return response

    fallback_used: list[str] = []
    cascade_request = request
    include_retired = False

    for step_name, builder in (
        ("widen_hops", _step_widen_hops),
        ("drop_predicate", _step_drop_predicate),
    ):
        next_request = builder(cascade_request)
        if next_request is None:
            continue
        cascade_request = next_request
        fallback_used.append(step_name)
        response = _apply_related_min_score(
            await _memory_related_pass(
                cascade_request,
                env_id=env_id,
                settings=settings,
                vector_store=vector_store,
                include_retired=include_retired,
            ),
            min_score=cascade_request.min_score,
        )
        if response.hits:
            break

    if not response.hits and not include_retired:
        include_retired = True
        fallback_used.append("include_retired")
        response = _apply_related_min_score(
            await _memory_related_pass(
                cascade_request,
                env_id=env_id,
                settings=settings,
                vector_store=vector_store,
                include_retired=True,
            ),
            min_score=cascade_request.min_score,
        )

    return response.model_copy(update={"fallback_used": fallback_used})
