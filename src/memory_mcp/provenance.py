"""Memory provenance and lineage tools — Sprint B.

Two tools share this module because both answer the question
"where did this memory come from?":

* :func:`memory_lineage` walks the ``memory_lineage`` table for the
  full provenance graph (promoted_from / summarized_from / supersedes
  / copied_from / moved_from) starting from a seed memory.
* :func:`memory_sources_browse` paginates the ``memory_sources`` table
  (session / file / url / llm / dream provenance records).

Both reuse the keyset-cursor + filter-fingerprint scaffolding from
:mod:`memory_mcp.pagination` and the RBAC contract from
:mod:`memory_mcp.rbac` (require ``read`` on every scoped env).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Select, and_, select, text, tuple_

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import Memory, MemoryLineage, MemorySource, MemoryTag, Tag
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import MemoryStatus
from memory_mcp.errors import InvalidCursorError, InvalidInputError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryResponse, _to_response
from memory_mcp.pagination import Direction, compute_filter_fingerprint, decode_cursor, encode_cursor

from memory_mcp_schemas.provenance import (
    MemLineageEdge,
    MemLineageRequest,
    MemLineageResponse,
    MemSourceHit,
    MemSourcesBrowseRequest,
    MemSourcesBrowseResponse,
)

__all__ = [
    "MemLineageEdge",
    "MemLineageRequest",
    "MemLineageResponse",
    "MemSourceHit",
    "MemSourcesBrowseRequest",
    "MemSourcesBrowseResponse",
    "memory_lineage",
    "memory_sources_browse",
]


_MAX_BIGINT = 9_223_372_036_854_775_807
_LINEAGE_TABLE = MemoryLineage.__tablename__


_VISIBLE_FOR_SOURCE_HYDRATION = [
    MemoryStatus.proposed.value,
    MemoryStatus.active.value,
    MemoryStatus.stale.value,
]


def _resolve_env_ids(explicit: list[UUID] | None, ctx: AgentContext) -> list[UUID]:
    if explicit:
        return list(dict.fromkeys(explicit))
    return list(dict.fromkeys(ctx.attached_env_ids))


def _dedupe(values: list[Any] | None) -> list[Any] | None:
    if not values:
        return None
    return list(dict.fromkeys(values))


def _lineage_relation_clause(relations: list[str] | None, *, column: str = "relation") -> str:
    return f"AND {column} = ANY(:relations)" if relations else ""


def _lineage_params(request: MemLineageRequest) -> dict[str, Any]:
    params: dict[str, Any] = {
        "seed": request.memory_id,
        "max_depth": request.max_depth + 1,
    }
    if request.relations:
        params["relations"] = list(dict.fromkeys(request.relations))
    return params


def _lineage_sql(request: MemLineageRequest, *, ancestors: bool) -> str:
    relation_clause = _lineage_relation_clause(request.relations)
    recursive_relation_clause = _lineage_relation_clause(request.relations, column="ml.relation")
    if ancestors:
        return f"""
            WITH RECURSIVE ancestors AS (
                SELECT parent_memory_id, child_memory_id, relation, created_at, 1 AS depth
                FROM {_LINEAGE_TABLE}
                WHERE child_memory_id = :seed
                {relation_clause}
                UNION ALL
                SELECT ml.parent_memory_id, ml.child_memory_id, ml.relation, ml.created_at, a.depth + 1
                FROM {_LINEAGE_TABLE} ml
                JOIN ancestors a ON ml.child_memory_id = a.parent_memory_id
                WHERE a.depth < :max_depth
                {recursive_relation_clause}
            ) CYCLE child_memory_id SET is_cycle USING path
            SELECT parent_memory_id, child_memory_id, relation, created_at, depth
            FROM ancestors
            WHERE NOT is_cycle
            ORDER BY depth ASC, created_at ASC, parent_memory_id ASC, child_memory_id ASC
        """

    return f"""
        WITH RECURSIVE descendants AS (
            SELECT parent_memory_id, child_memory_id, relation, created_at, 1 AS depth
            FROM {_LINEAGE_TABLE}
            WHERE parent_memory_id = :seed
            {relation_clause}
            UNION ALL
            SELECT ml.parent_memory_id, ml.child_memory_id, ml.relation, ml.created_at, d.depth + 1
            FROM {_LINEAGE_TABLE} ml
            JOIN descendants d ON ml.parent_memory_id = d.child_memory_id
            WHERE d.depth < :max_depth
            {recursive_relation_clause}
        ) CYCLE parent_memory_id SET is_cycle USING path
        SELECT parent_memory_id, child_memory_id, relation, created_at, depth
        FROM descendants
        WHERE NOT is_cycle
        ORDER BY depth ASC, created_at ASC, parent_memory_id ASC, child_memory_id ASC
    """


def _lineage_edges_from_rows(
    rows: list[Mapping[str, Any]],
    *,
    visible_max_depth: int,
) -> tuple[list[MemLineageEdge], bool]:
    edges: list[MemLineageEdge] = []
    truncated = False
    for row in rows:
        depth = int(row["depth"])
        if depth > visible_max_depth:
            truncated = True
            continue
        edges.append(
            MemLineageEdge(
                parent_memory_id=row["parent_memory_id"],
                child_memory_id=row["child_memory_id"],
                relation=row["relation"],
                created_at=row["created_at"],
                depth=depth,
            )
        )
    return edges, truncated


async def _lineage_edges(request: MemLineageRequest, *, ancestors: bool) -> tuple[list[MemLineageEdge], bool]:
    """Walk ``memory_lineage`` using PostgreSQL 14+ ``CYCLE`` recursion."""
    sql = _lineage_sql(request, ancestors=ancestors)
    async with session_scope() as session:
        rows = (await session.execute(text(sql), _lineage_params(request))).mappings().all()
    return _lineage_edges_from_rows(rows, visible_max_depth=request.max_depth)


def _apply_lineage_edge_cap(
    ancestors: list[MemLineageEdge],
    descendants: list[MemLineageEdge],
    *,
    max_edges: int,
) -> tuple[list[MemLineageEdge], list[MemLineageEdge], bool]:
    combined: list[tuple[str, int, MemLineageEdge]] = [
        *[("ancestor", idx, edge) for idx, edge in enumerate(ancestors)],
        *[("descendant", idx, edge) for idx, edge in enumerate(descendants)],
    ]
    if len(combined) <= max_edges:
        return ancestors, descendants, False

    combined.sort(
        key=lambda item: (
            item[2].depth,
            item[2].created_at,
            str(item[2].parent_memory_id),
            str(item[2].child_memory_id),
            item[0],
            item[1],
        )
    )
    kept = combined[:max_edges]
    kept_ancestors = [edge for kind, _idx, edge in kept if kind == "ancestor"]
    kept_descendants = [edge for kind, _idx, edge in kept if kind == "descendant"]
    sort_key = lambda edge: (edge.depth, edge.created_at, str(edge.parent_memory_id), str(edge.child_memory_id))
    kept_ancestors.sort(key=sort_key)
    kept_descendants.sort(key=sort_key)
    return kept_ancestors, kept_descendants, True


async def _hydrate_memory_responses(
    memory_ids: list[UUID],
    *,
    statuses: list[str] | None = None,
) -> dict[UUID, MemoryResponse]:
    if not memory_ids:
        return {}

    async with session_scope() as session:
        stmt: Select[Any] = select(Memory).where(Memory.id.in_(memory_ids))
        if statuses is not None:
            stmt = stmt.where(Memory.status.in_(statuses))
        rows = (await session.execute(stmt)).scalars().all()
        by_id: dict[UUID, Memory] = {memory.id: memory for memory in rows}

        tag_rows = await session.execute(
            select(MemoryTag.memory_id, Tag.name)
            .join(Tag, Tag.id == MemoryTag.tag_id)
            .where(MemoryTag.memory_id.in_(list(by_id)))
            .order_by(MemoryTag.memory_id, Tag.name)
        )
        tags_by_id: dict[UUID, list[str]] = {memory_id: [] for memory_id in by_id}
        for memory_id, tag_name in tag_rows.all():
            tags_by_id[memory_id].append(tag_name)

    return {
        memory_id: _to_response(memory, tags_by_id.get(memory_id, []))
        for memory_id, memory in by_id.items()
    }


async def memory_lineage(
    request: MemLineageRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemLineageResponse:
    """Return the provenance graph around a seed memory.

    The recursive traversal uses PostgreSQL 14+'s native ``CYCLE`` clause;
    memory-mcp's Sprint B provenance tools require that database version or
    newer. Node hydration intentionally bypasses default visibility so
    archived/superseded/retired forensic ancestors remain inspectable.
    """
    _ = settings or get_settings()

    async with session_scope() as session:
        seed = await session.get(Memory, request.memory_id)
        if seed is None or (request.env_id is not None and seed.env_id != request.env_id):
            raise NotFoundError(f"memory {request.memory_id} not found", memory_id=str(request.memory_id))
        rbac.require("read", seed.env_id, ctx)

    ancestors: list[MemLineageEdge] = []
    descendants: list[MemLineageEdge] = []
    ancestors_truncated = False
    descendants_truncated = False
    if request.direction in {"ancestors", "both"}:
        ancestors, ancestors_truncated = await _lineage_edges(request, ancestors=True)
    if request.direction in {"descendants", "both"}:
        descendants, descendants_truncated = await _lineage_edges(request, ancestors=False)

    ancestors, descendants, edge_cap_truncated = _apply_lineage_edge_cap(
        ancestors,
        descendants,
        max_edges=request.max_edges,
    )

    memory_ids = {request.memory_id}
    for edge in [*ancestors, *descendants]:
        memory_ids.add(edge.parent_memory_id)
        memory_ids.add(edge.child_memory_id)

    nodes = await _hydrate_memory_responses(list(memory_ids))
    for env_id in {node.env_id for node in nodes.values()}:
        rbac.require("read", env_id, ctx)

    seed_response = nodes.get(request.memory_id)
    if seed_response is None:
        raise NotFoundError(f"memory {request.memory_id} not found", memory_id=str(request.memory_id))

    truncated = ancestors_truncated or descendants_truncated or edge_cap_truncated
    return MemLineageResponse(
        seed=seed_response,
        ancestors=ancestors,
        descendants=descendants,
        nodes=nodes,
        truncated=truncated,
    )


def _sources_filter_dict(request: MemSourcesBrowseRequest, env_ids: list[UUID]) -> dict[str, Any]:
    return {
        "env_ids": list(env_ids),
        "memory_ids": _dedupe(request.memory_ids),
        "source_types": _dedupe(request.source_types),
        "source_refs": _dedupe(request.source_refs),
        "agent_ids": _dedupe(request.agent_ids),
        "created_after": request.created_after,
        "created_before": request.created_before,
        "descending": request.descending,
        "order_by": "created_at",
    }


def _source_id_to_cursor_uuid(source_id: int) -> UUID:
    """Pack positive ``BigInteger`` ids into the existing UUID cursor slot."""
    if source_id < 0:
        raise InvalidInputError(f"INVALID_INPUT: memory source id must be non-negative: {source_id}")
    return UUID(int=source_id)


def _cursor_uuid_to_source_id(source_uuid: UUID) -> int:
    """Unpack a ``MemorySource.id`` carried in ``KeysetCursor.tiebreak_id``."""
    source_id = source_uuid.int
    if source_id > _MAX_BIGINT:
        raise InvalidCursorError("INVALID_CURSOR: memory source id exceeds signed BigInteger range")
    return source_id


def _decode_sources_cursor(raw: str, *, fingerprint: str, direction: Direction) -> tuple[dt.datetime, int]:
    cur = decode_cursor(
        raw,
        expected_fingerprint=fingerprint,
        expected_order_field="created_at",
        expected_direction=direction,
    )
    try:
        created_at = dt.datetime.fromisoformat(cur.order_value)
    except ValueError as exc:
        raise InvalidCursorError(
            f"INVALID_CURSOR: cursor order_value is not ISO-8601 datetime: {cur.order_value!r}",
        ) from exc
    return created_at, _cursor_uuid_to_source_id(cur.tiebreak_id)


def _apply_sources_filters(
    stmt: Select[Any],
    request: MemSourcesBrowseRequest,
    *,
    env_ids: list[UUID],
) -> Select[Any]:
    stmt = stmt.where(
        Memory.env_id.in_(env_ids),
        Memory.status.in_(_VISIBLE_FOR_SOURCE_HYDRATION),
    )
    if request.memory_ids:
        stmt = stmt.where(MemorySource.memory_id.in_(list(dict.fromkeys(request.memory_ids))))
    if request.source_types:
        stmt = stmt.where(MemorySource.source_type.in_(list(dict.fromkeys(request.source_types))))
    if request.source_refs:
        stmt = stmt.where(MemorySource.source_ref.in_(list(dict.fromkeys(request.source_refs))))
    if request.agent_ids:
        stmt = stmt.where(MemorySource.agent_id.in_(list(dict.fromkeys(request.agent_ids))))
    if request.created_after is not None:
        stmt = stmt.where(MemorySource.created_at >= request.created_after)
    if request.created_before is not None:
        stmt = stmt.where(MemorySource.created_at < request.created_before)
    return stmt


def _apply_sources_keyset(
    stmt: Select[Any],
    *,
    descending: bool,
    cursor_value: dt.datetime | None,
    cursor_id: int | None,
) -> Select[Any]:
    if cursor_value is not None and cursor_id is not None:
        keyset = tuple_(MemorySource.created_at, MemorySource.id)
        cursor_keyset = tuple_(cursor_value, cursor_id)
        stmt = stmt.where(keyset < cursor_keyset if descending else keyset > cursor_keyset)
    if descending:
        return stmt.order_by(MemorySource.created_at.desc(), MemorySource.id.desc())
    return stmt.order_by(MemorySource.created_at.asc(), MemorySource.id.asc())


async def memory_sources_browse(
    request: MemSourcesBrowseRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemSourcesBrowseResponse:
    """Keyset-paginate provenance source rows scoped through ``memories.env_id``."""
    _ = settings or get_settings()
    env_ids = _resolve_env_ids(request.env_ids, ctx)
    for env_id in env_ids:
        rbac.require("read", env_id, ctx)

    if not env_ids:
        return MemSourcesBrowseResponse(hits=[], next_cursor=None, nodes={} if request.hydrate_memories else None)

    fingerprint = compute_filter_fingerprint(_sources_filter_dict(request, env_ids))
    direction: Direction = "desc" if request.descending else "asc"

    cursor_value: dt.datetime | None = None
    cursor_id: int | None = None
    if request.cursor:
        cursor_value, cursor_id = _decode_sources_cursor(
            request.cursor,
            fingerprint=fingerprint,
            direction=direction,
        )

    async with session_scope() as session:
        stmt: Select[Any] = (
            select(MemorySource, Memory.env_id)
            .join(Memory, Memory.id == MemorySource.memory_id)
        )
        stmt = _apply_sources_filters(stmt, request, env_ids=env_ids)
        stmt = _apply_sources_keyset(
            stmt,
            descending=request.descending,
            cursor_value=cursor_value,
            cursor_id=cursor_id,
        ).limit(request.limit + 1)

        rows = (await session.execute(stmt)).all()

    page = rows[: request.limit]
    has_more = len(rows) > request.limit
    hits = [
        MemSourceHit(
            id=int(source.id),
            memory_id=source.memory_id,
            env_id=env_id,
            source_type=source.source_type,
            source_ref=source.source_ref,
            agent_id=source.agent_id,
            created_at=source.created_at,
            evidence_span=source.evidence_span,
        )
        for source, env_id in page
    ]

    next_cursor: str | None = None
    if has_more and hits:
        last = hits[-1]
        next_cursor = encode_cursor(
            filter_fingerprint=fingerprint,
            order_field="created_at",
            order_value=last.created_at,
            tiebreak_id=_source_id_to_cursor_uuid(last.id),
            direction=direction,
        )

    nodes: dict[UUID, MemoryResponse] | None = None
    if request.hydrate_memories:
        memory_ids = list(dict.fromkeys(hit.memory_id for hit in hits))
        nodes = await _hydrate_memory_responses(memory_ids, statuses=_VISIBLE_FOR_SOURCE_HYDRATION)
        for env_id in {node.env_id for node in nodes.values()}:
            rbac.require("read", env_id, ctx)

    return MemSourcesBrowseResponse(hits=hits, next_cursor=next_cursor, nodes=nodes)
