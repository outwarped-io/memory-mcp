"""Postgres FTS retrieval (``mode=lex`` and the lex leg of ``mode=hybrid``).

Uses the ``body_tsv`` ``tsvector`` column on ``memories`` (computed +
indexed; see :mod:`memory_mcp.db.models`). Ranking via ``ts_rank_cd``.

Filters honored
---------------

* env_ids (always — required)
* status set (derived from ``include_stale/include_archived/include_retired``)
* kinds (optional)
* tags (optional — EXISTS to avoid distinct blow-up)
* created_after / created_before / updated_after

Limit
-----

The lex leg returns up to ``limit_lex`` rows. Hybrid uses a default of
``2 * final_limit`` to give RRF enough recall.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import String

from memory_mcp._filters import exclude_expired_raw_sql
from memory_mcp.search.ranking import RankedHit


async def lex_search(
    session: AsyncSession,
    *,
    query: str,
    env_ids: Sequence[UUID],
    statuses: Sequence[str],
    kinds: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    created_after: dt.datetime | None = None,
    created_before: dt.datetime | None = None,
    updated_after: dt.datetime | None = None,
    limit: int = 50,
    include_expired: bool = False,
) -> list[RankedHit]:
    """Postgres FTS retrieval. Returns 1-indexed ranked hits."""
    if not query.strip():
        return []
    if not env_ids:
        return []

    where: list[str] = [
        "m.body_tsv @@ q",
        "m.env_id = ANY(:env_ids)",
        "m.status = ANY(:statuses)",
    ]
    if not include_expired:
        where.append(exclude_expired_raw_sql("m"))
    params: dict[str, Any] = {
        "q_text": query,
        "env_ids": [str(e) for e in env_ids],
        "statuses": list(statuses),
        "limit": limit,
    }

    if kinds:
        where.append("m.kind = ANY(:kinds)")
        params["kinds"] = list(kinds)
    if tags:
        where.append(
            "EXISTS ("
            "  SELECT 1 FROM memory_tags mt JOIN tags t ON t.id = mt.tag_id "
            "  WHERE mt.memory_id = m.id AND t.name = ANY(:tags)"
            ")"
        )
        params["tags"] = list(tags)
    if created_after is not None:
        where.append("m.created_at >= :created_after")
        params["created_after"] = created_after
    if created_before is not None:
        where.append("m.created_at < :created_before")
        params["created_before"] = created_before
    if updated_after is not None:
        where.append("m.updated_at >= :updated_after")
        params["updated_after"] = updated_after

    sql = text(
        "WITH q AS (SELECT websearch_to_tsquery('english', :q_text) AS q) "
        "SELECT m.id, ts_rank_cd(m.body_tsv, q.q) AS lex_score "
        "FROM memories m, q "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY lex_score DESC, m.updated_at DESC "
        "LIMIT :limit"
    ).bindparams(
        bindparam("env_ids", type_=ARRAY(PG_UUID(as_uuid=True))),
        bindparam("statuses", type_=ARRAY(String)),
    )
    if "kinds" in params:
        sql = sql.bindparams(bindparam("kinds", type_=ARRAY(String)))
    if "tags" in params:
        sql = sql.bindparams(bindparam("tags", type_=ARRAY(String)))

    result = await session.execute(sql, params)
    rows = result.mappings().all()

    return [
        RankedHit(
            memory_id=r["id"],
            rank=i + 1,
            raw_score=float(r["lex_score"] or 0.0),
            source="lex",
        )
        for i, r in enumerate(rows)
    ]


__all__ = ["lex_search"]
