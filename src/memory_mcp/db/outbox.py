"""Transactional outbox writer.

Single source of truth for emitting projection events. Canonical writers
(``memory_write``, ``entity_upsert``, ``relation_link``, …) call
:func:`enqueue_event` *inside* their own transaction so the canonical
mutation and the outbox row commit atomically. The outbox tables are
defined in migration ``0001_v1_initial`` (see also :mod:`memory_mcp.db.models`).

Sink routing (v1)
-----------------

Sinks are derived from ``aggregate_type`` × backend selection:

============  ==================================  =====================
Aggregate     Backend setting                     Sink
============  ==================================  =====================
``memory``    ``vector_backend=qdrant``           ``qdrant``
``memory``    ``vector_backend=pgvector``         ``pgvector``
``entity``    ``graph_backend=neo4j``             ``neo4j``
``entity``    ``graph_backend=postgres``          *(none — recursive CTE)*
``relation``  ``graph_backend=neo4j``             ``neo4j``
``relation``  ``graph_backend=postgres``          *(none — recursive CTE)*
``task``      ``graph_backend=neo4j``             ``neo4j``
``task``      ``graph_backend=postgres``          *(none — recursive CTE)*
``env``       *(any)*                             *(none — metadata only)*
============  ==================================  =====================

When no sinks subscribe to a given aggregate type, :func:`enqueue_event`
becomes a no-op (returns ``EnqueuedEvent(event_id=None, sinks=())``).
The canonical row is still written by the caller; admin tools can rebuild
the projection from canonical Postgres on demand
(``memory_admin_rebuild_neo4j``, ``memory_admin_rebuild_qdrant``).

``projection_state`` row creation
---------------------------------

For every ``(sink, env_id)`` pair touched by an enqueue,
:func:`enqueue_event` performs an idempotent ``INSERT … ON CONFLICT DO
NOTHING`` to ensure a watermark row exists. The projection-worker is
responsible for advancing ``last_event_id`` / ``last_success_at`` /
``lag_seconds``; enqueue never disturbs those fields.

Idempotency / duplicate enqueues
--------------------------------

The Postgres unique constraint on
``(aggregate_type, aggregate_id, aggregate_version)`` makes duplicate
enqueues atomic-fail with :class:`sqlalchemy.exc.IntegrityError`. This is
treated as a writer bug — caller code paths must not enqueue the same
``(type, id, version)`` twice. We re-raise the SQLAlchemy error unchanged
so the canonical transaction rolls back.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import Outbox, OutboxDelivery, ProjectionState
from memory_mcp.db.types import OutboxAggregateType, OutboxOp, OutboxSink

__all__ = [
    "EnqueuedEvent",
    "enqueue_event",
    "resolve_sinks_for",
]


@dataclass(frozen=True)
class EnqueuedEvent:
    """Result of an enqueue.

    ``event_id`` is ``None`` when no sinks subscribe to the aggregate type
    under the current backend configuration. ``sinks`` is the tuple of
    sinks that received delivery rows (empty when ``event_id`` is ``None``).
    """

    event_id: int | None
    sinks: tuple[OutboxSink, ...]


def resolve_sinks_for(
    aggregate_type: OutboxAggregateType,
    settings: Settings,
) -> tuple[OutboxSink, ...]:
    """Return the sinks that should receive deliveries for ``aggregate_type``.

    The mapping is derived from :class:`Settings` so flipping
    ``vector_backend`` / ``graph_backend`` automatically rewires routing
    without touching the writer.
    """
    if aggregate_type == OutboxAggregateType.memory:
        if settings.vector_backend == "qdrant":
            return (OutboxSink.qdrant,)
        if settings.vector_backend == "pgvector":
            return (OutboxSink.pgvector,)
        return ()

    if aggregate_type in (
        OutboxAggregateType.entity,
        OutboxAggregateType.relation,
        OutboxAggregateType.task,
    ):
        if settings.graph_backend == "neo4j":
            return (OutboxSink.neo4j,)
        # graph_backend=postgres uses a recursive-CTE fallback; no sink.
        return ()

    # ``env`` aggregate type carries metadata only — no projection.
    return ()


async def enqueue_event(
    session: AsyncSession,
    *,
    aggregate_type: OutboxAggregateType,
    aggregate_id: UUID,
    aggregate_version: int,
    env_id: UUID,
    op: OutboxOp,
    payload: Mapping[str, Any],
    sinks: Iterable[OutboxSink] | None = None,
    settings: Settings | None = None,
) -> EnqueuedEvent:
    """Insert an outbox row + delivery rows + ensure projection_state rows.

    Performs *no* commit — must run within the caller's transaction so the
    canonical mutation and the outbox row are durable together.

    Parameters
    ----------
    session
        Active async session bound to the caller's transaction.
    aggregate_type
        Domain object kind (``memory`` / ``entity`` / ``relation`` / ``env``).
    aggregate_id
        UUID of the canonical row.
    aggregate_version
        Monotonic per-aggregate version (must be ``> 0``; the
        Postgres-side trigger enforces strict monotonic increase per
        aggregate).
    env_id
        Environment owning the row. Stored on the outbox row directly so
        workers, replay, and per-env ``projection_state`` queries do not
        need to parse the JSON payload.
    op
        ``upsert`` / ``update`` / ``tombstone``.
    payload
        Self-contained snapshot the projection worker uses to materialize
        the row in the sink. Must be JSON-serializable.
    sinks
        Optional explicit sink list (overrides ``resolve_sinks_for``).
        Useful for admin tools that target a single sink rebuild.
    settings
        Optional ``Settings`` override; defaults to ``get_settings()``.

    Returns
    -------
    :class:`EnqueuedEvent`
        ``event_id`` is the freshly assigned ``bigserial``; ``sinks`` is
        the tuple of sinks that received delivery rows. When no sinks
        subscribe the call is a no-op (``event_id=None, sinks=()``).

    Raises
    ------
    sqlalchemy.exc.IntegrityError
        When ``(aggregate_type, aggregate_id, aggregate_version)``
        already exists. This is a writer bug; the canonical transaction
        is expected to roll back.
    """
    settings = settings or get_settings()
    resolved_sinks: tuple[OutboxSink, ...] = (
        tuple(sinks) if sinks is not None else resolve_sinks_for(aggregate_type, settings)
    )

    if not resolved_sinks:
        return EnqueuedEvent(event_id=None, sinks=())

    insert_outbox = (
        insert(Outbox)
        .values(
            aggregate_type=aggregate_type.value,
            aggregate_id=aggregate_id,
            aggregate_version=aggregate_version,
            env_id=env_id,
            op=op.value,
            payload=dict(payload),
        )
        .returning(Outbox.event_id)
    )
    result = await session.execute(insert_outbox)
    event_id: int = result.scalar_one()

    await session.execute(
        insert(OutboxDelivery),
        [
            {"event_id": event_id, "sink": sink.value}
            for sink in resolved_sinks
        ],
    )

    # Ensure a projection_state row exists for each (sink, env_id) pair.
    # The worker advances last_event_id / last_success_at; we never touch
    # those columns from the writer side.
    upsert_state = (
        pg_insert(ProjectionState)
        .values(
            [
                {"sink": sink.value, "env_id": env_id}
                for sink in resolved_sinks
            ]
        )
        .on_conflict_do_nothing(index_elements=["sink", "env_id"])
    )
    await session.execute(upsert_state)

    return EnqueuedEvent(event_id=event_id, sinks=resolved_sinks)


async def get_projection_state(
    session: AsyncSession,
    *,
    sink: OutboxSink,
    env_id: UUID,
) -> ProjectionState | None:
    """Convenience read for ``(sink, env_id)`` watermark.

    Used by integration tests and the future
    ``memory_admin_projection_status`` tool.
    """
    stmt = select(ProjectionState).where(
        ProjectionState.sink == sink.value,
        ProjectionState.env_id == env_id,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
