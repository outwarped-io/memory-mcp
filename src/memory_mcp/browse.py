"""Row-level browse + facet tools for open-ended exploration.

Sprint A introduces two MCP tools backed by pure-Postgres queries on the
canonical ``memories`` (+ ``memory_tags``) tables:

* :func:`memory_browse` — keyset-paginated listing of memories without a
  free-text query. Closes the empty-query gap in ``mem_search`` (which
  intentionally returns 0 hits when ``query`` is empty in both lex and
  sem modes).

* :func:`memory_facets` — distinct-value + count aggregation over kind /
  status / tag / month buckets. Closes the tag-enumeration and
  pre-flight-an-env gaps surfaced by the exploration gap-analysis.

Both tools mirror :class:`memory_mcp.search.api.MemorySearchRequest`'s
filter shape (env_ids, kinds, tags, statuses, time windows) so the
contract is consistent across exploration tools.

Design notes
------------

* **Browse is not ranking.** No relevance score, no RRF, no semantic
  fan-out. Browse returns rows in deterministic ``(order_value, id)``
  order so paginated cursors are stable.
* **Default visibility** matches ``mem_search``: ``proposed`` + ``active``
  only unless the caller opts into stale / archived / retired.
* **Tag filter semantics** mirror ``mem_search`` post-filter: a memory
  matches when **ANY** listed tag is present (OR). Empty list = no
  filter.
* **RBAC**: ``rbac.require("read", env_id, ctx)`` for every requested
  env; no silent cross-env filtering. v1 is a no-op; v1.5 raises on
  missing grants.
* **Facet cost discipline.** Facet queries enforce a statement timeout
  and an optional ``max_rows`` budget. When the budget is exceeded the
  response carries ``approximate=True`` so the caller knows the counts
  are bounded.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from memory_mcp_schemas.browse import (
    BrowseOrderField,
    FacetBucket,
    MemBrowseRequest,
    MemBrowseResponse,
    MemFacetsRequest,
    MemFacetsResponse,
)
from sqlalchemy import (
    Select,
    and_,
    func,
    select,
    text,
    tuple_,
)
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp._filters import exclude_expired_clause
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import Memory, MemoryTag, Tag
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.errors import InvalidCursorError, InvalidInputError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import _to_response
from memory_mcp.pagination import (
    Direction,
    compute_filter_fingerprint,
    decode_cursor,
    encode_cursor,
)

log = logging.getLogger(__name__)


__all__ = [
    "FacetBucket",
    "MemBrowseRequest",
    "MemBrowseResponse",
    "MemFacetsRequest",
    "MemFacetsResponse",
    "memory_browse",
    "memory_facets",
]


# ---------------------------------------------------------------------------
# Shared visibility helpers (mirror search.api._VISIBLE_DEFAULT etc.)
# ---------------------------------------------------------------------------


_VISIBLE_DEFAULT: list[str] = ["proposed", "active"]
_VALID_STATUSES = {s.value for s in MemoryStatus}


def _resolve_browse_env_ids(
    explicit: list[UUID] | None,
    ctx: AgentContext,
) -> list[UUID]:
    """Browse env-resolution: explicit list > caller's attached envs."""
    if explicit:
        return list(dict.fromkeys(explicit))
    return list(dict.fromkeys(ctx.attached_env_ids))


def _resolve_statuses(
    explicit: list[MemoryStatus] | None,
) -> list[str]:
    """Resolve the status filter; default = ``[proposed, active]``."""
    if not explicit:
        return list(_VISIBLE_DEFAULT)
    out: list[str] = []
    seen: set[str] = set()
    for s in explicit:
        val = s.value if isinstance(s, MemoryStatus) else str(s)
        if val not in _VALID_STATUSES:
            raise InvalidInputError(f"INVALID_INPUT: unknown status: {val!r}")
        if val not in seen:
            seen.add(val)
            out.append(val)
    return out


# ---------------------------------------------------------------------------
# mem_browse
# ---------------------------------------------------------------------------


def _browse_filter_dict(req: MemBrowseRequest, env_ids: list[UUID]) -> dict[str, Any]:
    """Pure-data view of the filter set — feeds the keyset-cursor fingerprint."""
    return {
        "env_ids": list(env_ids),
        "kinds": sorted(k.value for k in req.kinds) if req.kinds else None,
        "tags": sorted(req.tags) if req.tags else None,
        "statuses": _resolve_statuses(req.statuses),
        "created_after": req.created_after,
        "created_before": req.created_before,
        "updated_after": req.updated_after,
        "order_by": req.order_by,
        "descending": req.descending,
        "include_expired": req.include_expired,
    }


def _direction(req: MemBrowseRequest) -> Direction:
    return "desc" if req.descending else "asc"


def _apply_browse_filters(
    stmt: Select[Any],
    *,
    env_ids: Sequence[UUID],
    statuses: Sequence[str],
    kinds: Sequence[MemoryKind] | None,
    tags: Sequence[str] | None,
    created_after: dt.datetime | None,
    created_before: dt.datetime | None,
    updated_after: dt.datetime | None,
    include_expired: bool = False,
) -> Select[Any]:
    stmt = stmt.where(Memory.env_id.in_(list(env_ids)))
    stmt = stmt.where(Memory.status.in_(list(statuses)))
    if not include_expired:
        stmt = stmt.where(exclude_expired_clause())
    if kinds:
        stmt = stmt.where(Memory.kind.in_([k.value for k in kinds]))
    if created_after is not None:
        stmt = stmt.where(Memory.created_at >= created_after)
    if created_before is not None:
        stmt = stmt.where(Memory.created_at < created_before)
    if updated_after is not None:
        stmt = stmt.where(Memory.updated_at >= updated_after)
    if tags:
        # OR semantics (mirror search.api._post_filter): a memory matches
        # if ANY listed tag is present. Single EXISTS subquery suffices.
        sub = (
            select(MemoryTag.memory_id)
            .join(Tag, Tag.id == MemoryTag.tag_id)
            .where(
                Tag.name.in_(list(tags)),
                MemoryTag.env_id == Memory.env_id,
                MemoryTag.memory_id == Memory.id,
            )
        )
        stmt = stmt.where(sub.exists())
    return stmt


def _apply_browse_keyset(
    stmt: Select[Any],
    *,
    order_by: BrowseOrderField,
    descending: bool,
    cursor_value: dt.datetime | None,
    cursor_id: UUID | None,
) -> Select[Any]:
    """Apply the keyset (cursor) constraint and final ordering."""
    order_col = Memory.updated_at if order_by == "updated_at" else Memory.created_at
    if cursor_value is not None and cursor_id is not None:
        if descending:
            stmt = stmt.where(tuple_(order_col, Memory.id) < tuple_(cursor_value, cursor_id))
        else:
            stmt = stmt.where(tuple_(order_col, Memory.id) > tuple_(cursor_value, cursor_id))
    if descending:
        stmt = stmt.order_by(order_col.desc(), Memory.id.desc())
    else:
        stmt = stmt.order_by(order_col.asc(), Memory.id.asc())
    return stmt


async def _hydrate_tags(session: AsyncSession, memory_ids: list[UUID]) -> dict[UUID, list[str]]:
    if not memory_ids:
        return {}
    rows = await session.execute(
        select(MemoryTag.memory_id, Tag.name)
        .join(Tag, Tag.id == MemoryTag.tag_id)
        .where(MemoryTag.memory_id.in_(memory_ids))
        .order_by(MemoryTag.memory_id, Tag.name)
    )
    out: dict[UUID, list[str]] = {mid: [] for mid in memory_ids}
    for mid, name in rows.all():
        out[mid].append(name)
    return out


def _decode_browse_cursor(
    raw: str,
    *,
    fingerprint: str,
    order_by: BrowseOrderField,
    direction: Direction,
) -> tuple[dt.datetime, UUID]:
    cur = decode_cursor(
        raw,
        expected_fingerprint=fingerprint,
        expected_order_field=order_by,
        expected_direction=direction,
    )
    try:
        order_value = dt.datetime.fromisoformat(cur.order_value)
    except ValueError as exc:
        raise InvalidCursorError(
            f"INVALID_CURSOR: cursor order_value is not ISO-8601 datetime: {cur.order_value!r}",
        ) from exc
    return order_value, cur.tiebreak_id


async def memory_browse(
    req: MemBrowseRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemBrowseResponse:
    """Keyset-paginated listing of memories with no relevance ranking.

    Backed by a single SQL query on ``memories`` + (when tag filters are
    present) per-tag ``EXISTS`` subqueries. Reuses the same default
    visibility (``proposed`` + ``active``) and filter shape as
    ``mem_search`` so callers don't need to learn a second filter
    language.
    """
    _ = settings or get_settings()
    env_ids = _resolve_browse_env_ids(req.env_ids, ctx)
    for env_id in env_ids:
        rbac.require("read", env_id, ctx)

    if not env_ids:
        return MemBrowseResponse(hits=[], next_cursor=None, has_more=False)

    statuses = _resolve_statuses(req.statuses)
    filter_dict = _browse_filter_dict(req, env_ids)
    fingerprint = compute_filter_fingerprint(filter_dict)
    direction: Direction = _direction(req)

    cursor_value: dt.datetime | None = None
    cursor_id: UUID | None = None
    if req.cursor:
        cursor_value, cursor_id = _decode_browse_cursor(
            req.cursor,
            fingerprint=fingerprint,
            order_by=req.order_by,
            direction=direction,
        )

    async with session_scope() as session:
        stmt: Select[Any] = select(Memory)
        stmt = _apply_browse_filters(
            stmt,
            env_ids=env_ids,
            statuses=statuses,
            kinds=req.kinds,
            tags=req.tags,
            created_after=req.created_after,
            created_before=req.created_before,
            updated_after=req.updated_after,
            include_expired=req.include_expired,
        )
        stmt = _apply_browse_keyset(
            stmt,
            order_by=req.order_by,
            descending=req.descending,
            cursor_value=cursor_value,
            cursor_id=cursor_id,
        )
        # Over-fetch by 1 to detect "has more" without a separate count.
        stmt = stmt.limit(req.limit + 1)

        rows = (await session.execute(stmt)).scalars().all()
        page = rows[: req.limit]
        has_more = len(rows) > req.limit

        memory_ids = [m.id for m in page]
        tags_by_id = await _hydrate_tags(session, memory_ids)

    hits = [_to_response(m, tags_by_id.get(m.id, [])) for m in page]

    next_cursor: str | None = None
    if has_more and page:
        last = page[-1]
        order_value = last.updated_at if req.order_by == "updated_at" else last.created_at
        next_cursor = encode_cursor(
            filter_fingerprint=fingerprint,
            order_field=req.order_by,
            order_value=order_value,
            tiebreak_id=last.id,
            direction=direction,
        )

    return MemBrowseResponse(hits=hits, next_cursor=next_cursor, has_more=has_more)


# ---------------------------------------------------------------------------
# mem_facets
# ---------------------------------------------------------------------------


def _facet_filter_clause(
    *,
    env_ids: Sequence[UUID],
    statuses: Sequence[str],
    kinds: Sequence[MemoryKind] | None,
    tags: Sequence[str] | None,
    created_after: dt.datetime | None,
    created_before: dt.datetime | None,
    updated_after: dt.datetime | None,
    include_expired: bool = False,
) -> Any:
    clauses = [
        Memory.env_id.in_(list(env_ids)),
        Memory.status.in_(list(statuses)),
    ]
    if not include_expired:
        clauses.append(exclude_expired_clause())
    if kinds:
        clauses.append(Memory.kind.in_([k.value for k in kinds]))
    if created_after is not None:
        clauses.append(Memory.created_at >= created_after)
    if created_before is not None:
        clauses.append(Memory.created_at < created_before)
    if updated_after is not None:
        clauses.append(Memory.updated_at >= updated_after)
    if tags:
        # OR semantics (parity with mem_search post-filter).
        sub = (
            select(MemoryTag.memory_id)
            .join(Tag, Tag.id == MemoryTag.tag_id)
            .where(
                Tag.name.in_(list(tags)),
                MemoryTag.env_id == Memory.env_id,
                MemoryTag.memory_id == Memory.id,
            )
        )
        clauses.append(sub.exists())
    return and_(*clauses)


async def _facet_total_and_by_env(
    session: AsyncSession,
    *,
    filter_clause: Any,
    env_ids: Sequence[UUID],
) -> tuple[int, dict[UUID, int]]:
    rows = await session.execute(
        select(Memory.env_id, func.count(Memory.id)).where(filter_clause).group_by(Memory.env_id)
    )
    by_env: dict[UUID, int] = dict.fromkeys(env_ids, 0)
    total = 0
    for env_id, count in rows.all():
        by_env[env_id] = int(count)
        total += int(count)
    return total, by_env


async def _facet_groupby_kind(
    session: AsyncSession,
    filter_clause: Any,
) -> list[FacetBucket]:
    rows = await session.execute(
        select(Memory.kind, func.count(Memory.id))
        .where(filter_clause)
        .group_by(Memory.kind)
        .order_by(func.count(Memory.id).desc(), Memory.kind)
    )
    return [FacetBucket(value=k, count=int(c)) for k, c in rows.all()]


async def _facet_groupby_status(
    session: AsyncSession,
    filter_clause: Any,
) -> list[FacetBucket]:
    rows = await session.execute(
        select(Memory.status, func.count(Memory.id))
        .where(filter_clause)
        .group_by(Memory.status)
        .order_by(func.count(Memory.id).desc(), Memory.status)
    )
    return [FacetBucket(value=s, count=int(c)) for s, c in rows.all()]


async def _facet_groupby_tag(
    session: AsyncSession,
    filter_clause: Any,
    *,
    tag_limit: int,
) -> list[FacetBucket]:
    rows = await session.execute(
        select(Tag.name, func.count(MemoryTag.memory_id))
        .join(MemoryTag, MemoryTag.tag_id == Tag.id)
        .join(Memory, and_(Memory.id == MemoryTag.memory_id, Memory.env_id == MemoryTag.env_id))
        .where(filter_clause)
        .group_by(Tag.name)
        .order_by(func.count(MemoryTag.memory_id).desc(), Tag.name)
        .limit(tag_limit)
    )
    return [FacetBucket(value=t, count=int(c)) for t, c in rows.all()]


async def _facet_groupby_month(
    session: AsyncSession,
    filter_clause: Any,
) -> list[FacetBucket]:
    """Bucket by month of ``created_at`` (UTC date_trunc)."""
    month_expr = func.date_trunc("month", Memory.created_at)
    rows = await session.execute(
        select(month_expr.label("month"), func.count(Memory.id))
        .where(filter_clause)
        .group_by(month_expr)
        .order_by(month_expr.desc())
    )
    out: list[FacetBucket] = []
    for month, count in rows.all():
        if isinstance(month, dt.datetime):
            # Normalize to UTC ISO YYYY-MM
            if month.tzinfo is None:
                month = month.replace(tzinfo=dt.UTC)
            value = month.astimezone(dt.UTC).strftime("%Y-%m")
        else:
            value = str(month)
        out.append(FacetBucket(value=value, count=int(count)))
    return out


async def memory_facets(
    req: MemFacetsRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemFacetsResponse:
    """Distinct-value + count aggregation across requested facets.

    Statement timeout: enforced server-side (``settings.facet_query_timeout_seconds``,
    default 2.0). On timeout we return whatever facets have completed with
    ``approximate=True``. The ``max_rows`` knob lets the caller place an
    explicit upper bound; when the underlying ``COUNT(*)`` exceeds it we
    flag ``approximate`` even if all queries completed.
    """
    settings = settings or get_settings()
    env_ids = _resolve_browse_env_ids(req.env_ids, ctx)
    for env_id in env_ids:
        rbac.require("read", env_id, ctx)

    if not env_ids:
        return MemFacetsResponse(total=0, by_env={}, facets={})

    statuses = _resolve_statuses(req.statuses)
    filter_clause = _facet_filter_clause(
        env_ids=env_ids,
        statuses=statuses,
        kinds=req.kinds,
        tags=req.tags,
        created_after=req.created_after,
        created_before=req.created_before,
        updated_after=req.updated_after,
        include_expired=req.include_expired,
    )

    timeout_seconds = getattr(settings, "facet_query_timeout_seconds", 2.0)
    timeout_ms = max(1, int(timeout_seconds * 1000))

    approximate = False
    facets_out: dict[str, list[FacetBucket]] = {}
    total = 0
    by_env: dict[UUID, int] = {}

    async with session_scope() as session:
        await session.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))

        try:
            total, by_env = await _facet_total_and_by_env(
                session,
                filter_clause=filter_clause,
                env_ids=env_ids,
            )
        except Exception:  # noqa: BLE001 — timeout / cancel surfaces here
            await session.rollback()
            log.warning("memory_facets: total aggregation timed out", extra={"env_ids": [str(e) for e in env_ids]})
            return MemFacetsResponse(
                total=0,
                by_env=dict.fromkeys(env_ids, 0),
                facets={},
                approximate=True,
                sampled_rows=0,
            )

        sampled_rows = total
        if total > req.max_rows:
            # Budget exceeded — return totals + per-env breakdown but skip
            # the heavier per-facet GROUP BY. Caller is expected to narrow
            # filters (time window / kinds) and retry.
            return MemFacetsResponse(
                total=total,
                by_env=by_env,
                facets={},
                approximate=True,
                sampled_rows=sampled_rows,
            )

        for facet in dict.fromkeys(req.facets):  # dedupe, preserve order
            try:
                if facet == "kind":
                    facets_out["kind"] = await _facet_groupby_kind(session, filter_clause)
                elif facet == "status":
                    facets_out["status"] = await _facet_groupby_status(session, filter_clause)
                elif facet == "tag":
                    facets_out["tag"] = await _facet_groupby_tag(
                        session,
                        filter_clause,
                        tag_limit=req.tag_limit,
                    )
                elif facet == "month":
                    facets_out["month"] = await _facet_groupby_month(session, filter_clause)
            except Exception:  # noqa: BLE001
                await session.rollback()
                log.warning(
                    "memory_facets: facet %s timed out",
                    facet,
                    extra={"env_ids": [str(e) for e in env_ids]},
                )
                approximate = True
                break

    return MemFacetsResponse(
        total=total,
        by_env=by_env,
        facets=facets_out,
        approximate=approximate,
        sampled_rows=sampled_rows,
    )
