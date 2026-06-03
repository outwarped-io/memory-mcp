"""Shared read-path filter helpers (v0.17 Phase B).

Single source for the ``expires_at`` default-exclusion clause used across
every read path (``mem_search`` / ``mem_browse`` / ``mem_facets`` / ``mem_top``
/ ``mem_context_pack`` / ``mem_resume`` / ``mem_digest`` / ``mem_auto_context``
/ ``mem_inbox``). The inbox feature (v0.17) introduced caller-visible
TTLs; this module makes the default-exclusion uniform across the rest
of the surface so expired memories disappear consistently.

Two complementary helpers:

* :func:`exclude_expired_clause` — SQLAlchemy where-clause fragment for
  ORM / raw-SQL select builders. Uses PostgreSQL server time
  (``now()``) so the filter is evaluated by the database and the
  partial index ``memories_expires_idx`` on ``(expires_at) WHERE
  expires_at IS NOT NULL`` can participate.
* :func:`is_expired` — Python-side check for in-memory ``Memory`` rows
  after they have been hydrated (e.g. inside
  :func:`memory_mcp.search.api._passes_post_filters`). Uses
  ``datetime.now(UTC)``; the small Python-vs-PG clock-skew window is
  acceptable for a post-filter that runs milliseconds after the
  hydrating query.

Identity-style lookups (``mem_get``, ``mem_get_many``, ``mem_lineage``)
deliberately do NOT apply this filter — they are forensic / lineage
operations where the caller has already asserted a specific id.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import text
from sqlalchemy.sql.elements import ColumnElement

from memory_mcp.db.models import Memory


def exclude_expired_clause() -> ColumnElement[bool]:
    """Return a SQLAlchemy where-clause that excludes expired memories.

    Pattern mirrors :mod:`memory_mcp.inbox`:

    .. code-block:: python

        (Memory.expires_at.is_(None)) | (Memory.expires_at > text("now()"))

    Always uses ``text("now()")`` — server-side PG time — so caller and
    database agree on a single clock and the partial index
    ``memories_expires_idx`` is eligible.
    """
    return (Memory.expires_at.is_(None)) | (Memory.expires_at > text("now()"))


_EXPIRED_RAW_SQL_CLAUSE = "(m.expires_at IS NULL OR m.expires_at > now())"


def exclude_expired_raw_sql(table_alias: str = "m") -> str:
    """Return a raw-SQL fragment for builders that do string concatenation.

    Used by :mod:`memory_mcp.search.lex` where the SELECT is assembled
    as a list of WHERE fragments joined with ``AND``.
    """
    if table_alias == "m":
        return _EXPIRED_RAW_SQL_CLAUSE
    return f"({table_alias}.expires_at IS NULL OR {table_alias}.expires_at > now())"


def is_expired(memory: Memory, *, now: dt.datetime | None = None) -> bool:
    """Return True if ``memory`` has a past ``expires_at``.

    ``now`` defaults to ``datetime.now(UTC)``. Memories with
    ``expires_at is None`` are never expired.
    """
    if memory.expires_at is None:
        return False
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    return memory.expires_at <= now


__all__ = (
    "exclude_expired_clause",
    "exclude_expired_raw_sql",
    "is_expired",
)
