"""Top-of-the-board ranking — Phase 1 ``mem_top`` MCP tool.

Returns the highest-ranked memories in the requested env(s) by one of
five metrics: ``salience`` / ``access_count`` / ``reference_count`` /
``reference_velocity`` / ``reference_authority``.

All metrics share a stable tie-breaker — ``(metric DESC, created_at DESC,
id DESC)`` — so a deterministic top-N is reproducible across runs.

Tag filter semantics:

* ``tag_match="any"`` (default) — OR (mirrors ``mem_search`` /
  ``mem_browse``); a memory matches if ANY listed tag is present.
* ``tag_match="all"`` — AND; a memory must carry every listed tag.

Status filter defaults to ``[active]`` — top-of-the-board is a live
signal; ``proposed`` and ``stale`` rows are excluded unless the caller
opts them in explicitly.

Reference-velocity (``by="reference_velocity"``) is computed lazily over
the requested ``velocity_window_days`` window from the ``relations`` +
``memory_lineage`` edge tables. Playbook embeds are *excluded* from
velocity because they have no per-edge timestamp; that limitation is
documented in the parent plan's S7.

Reference-authority (``by="reference_authority"``) is the **weighted**
citation footprint — ``Σ source.salience`` over inbound citations,
maintained by the recount pass. The metric is **knob-gated**: when
``Settings.dream_popularity_authority_weighted`` is ``False`` (default),
the tool raises ``AUTHORITY_DISABLED`` before any DB work. When ON,
zero-authority rows are excluded from ``items`` (mirroring
``reference_velocity`` semantics) but counted in ``total_examined``.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from memory_mcp_schemas.top import (
    MemTopItem,
    MemTopRequest,
    MemTopResponse,
)
from sqlalchemy import (
    Select,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp._filters import exclude_expired_clause
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import (
    GraphNode,
    Memory,
    MemoryLineage,
    MemoryTag,
    Relation,
    Tag,
)
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.errors import AuthorityDisabledError, InvalidInputError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import _to_response

log = logging.getLogger(__name__)


__all__ = [
    "MemTopItem",
    "MemTopRequest",
    "MemTopResponse",
    "memory_top",
]


_DEFAULT_STATUS: list[str] = ["active"]
_VALID_STATUSES = {s.value for s in MemoryStatus}

_LINEAGE_VELOCITY_WHITELIST = (
    "summarized_from",
    "promoted_from",
    "derives_from",
    # ``split_from`` was forward-listed here before Migration 0021 but
    # was removed from the popularity whitelist (rows of that relation
    # connect to a retired parent and must not contribute to velocity).
    "derived_from",
)


def _resolve_env_ids(explicit: list[UUID] | None, ctx: AgentContext) -> list[UUID]:
    if explicit:
        return list(dict.fromkeys(explicit))
    return list(dict.fromkeys(ctx.attached_env_ids))


def _resolve_statuses(explicit: list[MemoryStatus] | None) -> list[str]:
    if not explicit:
        return list(_DEFAULT_STATUS)
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


def _apply_base_filters(
    stmt: Select[Any],
    *,
    env_ids: Sequence[UUID],
    statuses: Sequence[str],
    kinds: Sequence[MemoryKind] | None,
    tags: Sequence[str] | None,
    tag_match: str,
    include_expired: bool = False,
) -> Select[Any]:
    stmt = stmt.where(Memory.env_id.in_(list(env_ids)))
    stmt = stmt.where(Memory.status.in_(list(statuses)))
    if not include_expired:
        # v0.17 — opt-in expired exposure; default-hide.
        stmt = stmt.where(exclude_expired_clause())
    if kinds:
        stmt = stmt.where(Memory.kind.in_([k.value for k in kinds]))
    if tags:
        if tag_match == "all":
            # AND semantics — one EXISTS per tag.
            for tag_name in tags:
                sub = (
                    select(MemoryTag.memory_id)
                    .join(Tag, Tag.id == MemoryTag.tag_id)
                    .where(
                        Tag.name == tag_name,
                        MemoryTag.env_id == Memory.env_id,
                        MemoryTag.memory_id == Memory.id,
                    )
                )
                stmt = stmt.where(sub.exists())
        else:
            # OR semantics — single EXISTS over the tag set.
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


async def _count_total_examined(
    session: AsyncSession,
    *,
    env_ids: Sequence[UUID],
    statuses: Sequence[str],
    kinds: Sequence[MemoryKind] | None,
    tags: Sequence[str] | None,
    tag_match: str,
    include_expired: bool = False,
) -> int:
    stmt: Select[Any] = select(func.count(Memory.id))
    stmt = _apply_base_filters(
        stmt,
        env_ids=env_ids,
        statuses=statuses,
        kinds=kinds,
        tags=tags,
        tag_match=tag_match,
        include_expired=include_expired,
    )
    return int((await session.execute(stmt)).scalar_one() or 0)


def _column_for_metric(by: str) -> Any:
    if by == "salience":
        return Memory.salience
    if by == "access_count":
        return Memory.access_count
    if by == "reference_count":
        return Memory.reference_count
    if by == "reference_authority":
        return Memory.reference_authority
    raise InvalidInputError(f"INVALID_INPUT: unknown column metric: {by!r}")


async def _rank_by_column(
    session: AsyncSession,
    *,
    env_ids: Sequence[UUID],
    statuses: Sequence[str],
    kinds: Sequence[MemoryKind] | None,
    tags: Sequence[str] | None,
    tag_match: str,
    by: str,
    limit: int,
    include_expired: bool = False,
) -> tuple[list[Memory], list[float]]:
    """Salience / access_count / reference_count / reference_authority —
    all stored on memories. ``reference_authority`` additionally applies
    a ``metric > 0`` WHERE clause so zero-authority rows are excluded
    from results (mirrors ``reference_velocity`` semantics — a metric
    that's only meaningful for memories with positive citation weight).
    """
    metric_col = _column_for_metric(by)
    stmt: Select[Any] = select(Memory, metric_col.label("metric_value"))
    stmt = _apply_base_filters(
        stmt,
        env_ids=env_ids,
        statuses=statuses,
        kinds=kinds,
        tags=tags,
        tag_match=tag_match,
        include_expired=include_expired,
    )
    if by == "reference_authority":
        stmt = stmt.where(metric_col > 0)
    stmt = stmt.order_by(
        metric_col.desc(),
        Memory.created_at.desc(),
        Memory.id.desc(),
    ).limit(limit)
    rows = (await session.execute(stmt)).all()
    memories = [r[0] for r in rows]
    metric_values = [float(r[1] or 0) for r in rows]
    return memories, metric_values


async def _velocity_counts(
    session: AsyncSession,
    *,
    env_ids: Sequence[UUID],
    since: dt.datetime,
) -> dict[UUID, int]:
    """Count rel_link + task + lineage edges newly pointing at each memory
    in the given env(s) since ``since``.

    Returns ``{memory_id: count}`` with only memories that have at least
    one new edge in the window. Excludes ``related_to_popular`` rel_links
    (Phase 4 auto-wire reservation) and ``supersedes`` lineage parents
    (matches the counter triggers' whitelist).
    """
    out: dict[UUID, int] = {}

    # rel_link + task — both come through the ``relations`` table; the
    # counter triggers separate them by src node_type, but velocity treats
    # them uniformly (citation = citation).
    rel_q = (
        select(GraphNode.memory_id, func.count(Relation.id).label("c"))
        .join(GraphNode, GraphNode.id == Relation.dst_node_id)
        .where(
            GraphNode.memory_id.is_not(None),
            GraphNode.env_id.in_(list(env_ids)),
            Relation.env_id.in_(list(env_ids)),
            Relation.created_at >= since,
            Relation.type != "related_to_popular",
        )
        .group_by(GraphNode.memory_id)
    )
    for mem_id, count in (await session.execute(rel_q)).all():
        if mem_id is None:
            continue
        out[mem_id] = out.get(mem_id, 0) + int(count)

    # lineage parents in the whitelist.
    lin_q = (
        select(MemoryLineage.parent_memory_id, func.count().label("c"))
        .where(
            MemoryLineage.created_at >= since,
            MemoryLineage.relation.in_(list(_LINEAGE_VELOCITY_WHITELIST)),
        )
        .group_by(MemoryLineage.parent_memory_id)
    )
    for parent_id, count in (await session.execute(lin_q)).all():
        out[parent_id] = out.get(parent_id, 0) + int(count)

    return out


async def _rank_by_velocity(
    session: AsyncSession,
    *,
    env_ids: Sequence[UUID],
    statuses: Sequence[str],
    kinds: Sequence[MemoryKind] | None,
    tags: Sequence[str] | None,
    tag_match: str,
    window_days: int,
    limit: int,
    include_expired: bool = False,
) -> tuple[list[Memory], list[float], dict[UUID, int]]:
    """Reference-velocity ranking — recent citation arrival rate."""
    now = dt.datetime.now(dt.UTC)
    since = now - dt.timedelta(days=window_days)
    velocity = await _velocity_counts(session, env_ids=env_ids, since=since)

    if not velocity:
        return [], [], {}

    candidate_ids = list(velocity.keys())

    stmt: Select[Any] = select(Memory).where(Memory.id.in_(candidate_ids))
    stmt = _apply_base_filters(
        stmt,
        env_ids=env_ids,
        statuses=statuses,
        kinds=kinds,
        tags=tags,
        tag_match=tag_match,
        include_expired=include_expired,
    )
    candidates = list((await session.execute(stmt)).scalars().all())

    candidates.sort(
        key=lambda m: (velocity.get(m.id, 0), m.created_at, m.id),
        reverse=True,
    )
    top = candidates[:limit]
    metric_values = [float(velocity.get(m.id, 0)) for m in top]
    return top, metric_values, velocity


async def memory_top(
    req: MemTopRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemTopResponse:
    """Compute the top-N memories under the requested ranking metric."""
    settings = settings or get_settings()

    # Knob gate for the reference_authority metric. Fires immediately,
    # before env resolution / RBAC / DB, so callers get a clean "metric
    # unavailable" signal without spurious load when the authority
    # signal is dormant.
    if req.by == "reference_authority" and not settings.dream_popularity_authority_weighted:
        raise AuthorityDisabledError(
            "reference_authority metric requires "
            "dream_popularity_authority_weighted=True; the authority "
            "signal is dormant under this env's current settings."
        )

    env_ids = _resolve_env_ids(req.env_ids, ctx)
    for env_id in env_ids:
        rbac.require("read", env_id, ctx)

    if not env_ids:
        return MemTopResponse(items=[], by=req.by, total_examined=0)

    statuses = _resolve_statuses(req.statuses)

    async with session_scope() as session:
        total_examined = await _count_total_examined(
            session,
            env_ids=env_ids,
            statuses=statuses,
            kinds=req.kinds,
            tags=req.tags,
            tag_match=req.tag_match,
            include_expired=req.include_expired,
        )

        velocity_lookup: dict[UUID, int] = {}
        if req.by == "reference_velocity":
            memories, metric_values, velocity_lookup = await _rank_by_velocity(
                session,
                env_ids=env_ids,
                statuses=statuses,
                kinds=req.kinds,
                tags=req.tags,
                tag_match=req.tag_match,
                window_days=req.velocity_window_days,
                limit=req.limit,
                include_expired=req.include_expired,
            )
        else:
            memories, metric_values = await _rank_by_column(
                session,
                env_ids=env_ids,
                statuses=statuses,
                kinds=req.kinds,
                tags=req.tags,
                tag_match=req.tag_match,
                by=req.by,
                limit=req.limit,
                include_expired=req.include_expired,
            )

        tags_by_id = await _hydrate_tags(session, [m.id for m in memories])

    items: list[MemTopItem] = []
    for m, metric in zip(memories, metric_values, strict=True):
        rv: int | None = None
        if req.by == "reference_velocity":
            rv = int(velocity_lookup.get(m.id, 0))
        items.append(
            MemTopItem(
                memory=_to_response(m, tags_by_id.get(m.id, []), reference_velocity=rv),
                metric_value=metric,
            )
        )

    return MemTopResponse(items=items, by=req.by, total_examined=total_examined)
