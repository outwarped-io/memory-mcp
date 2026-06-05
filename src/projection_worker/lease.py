"""Outbox lease/release/fail SQL operations for the projection worker.

Concurrency model
-----------------

The projection worker uses **per-aggregate ordering with at-least-once
delivery**. Multiple worker instances may run concurrently; the
``locked_by``/``locked_until`` columns on ``outbox_delivery`` provide
the lease primitive. A failed processing attempt releases the lock with
``locked_until`` advanced by an exponential backoff.

Lease query (high level)::

    UPDATE outbox_delivery od
       SET locked_by = :worker_id,
           locked_until = now() + :lease_ttl
     WHERE (od.event_id, od.sink) IN (
        SELECT od2.event_id, od2.sink
          FROM outbox_delivery od2
          JOIN outbox o ON o.event_id = od2.event_id
         WHERE od2.sink = :sink
           AND od2.status = 'pending'
           AND (od2.locked_until IS NULL OR od2.locked_until < now())
           AND NOT EXISTS (
              SELECT 1
                FROM outbox o_prior
                JOIN outbox_delivery od_prior
                  ON od_prior.event_id = o_prior.event_id
               WHERE o_prior.aggregate_id = o.aggregate_id
                 AND o_prior.aggregate_version < o.aggregate_version
                 AND od_prior.sink = od2.sink
                 AND od_prior.status <> 'done'
           )
         ORDER BY o.event_id
         LIMIT :batch_size
         FOR UPDATE OF od2 SKIP LOCKED
     )
     RETURNING od.event_id, od.sink, od.attempt_count;

The ``NOT EXISTS`` clause enforces per-aggregate ordering: an event for
``aggregate_id=X version=N`` cannot be leased while any earlier-version
event for X is still ``pending`` / ``in_flight``. ``SKIP LOCKED`` is the
standard recipe for queue-style row leasing in Postgres.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp.db.models import OutboxDelivery
from memory_mcp.db.types import OutboxDeliveryStatus, OutboxSink

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeasedEvent:
    """A row pulled from ``outbox`` + ``outbox_delivery`` for processing."""

    event_id: int
    sink: OutboxSink
    aggregate_type: str
    aggregate_id: UUID
    aggregate_version: int
    env_id: UUID
    op: str
    payload: dict[str, Any]
    attempt_count: int
    created_at: dt.datetime


# ---------------------------------------------------------------------------
# Lease
# ---------------------------------------------------------------------------


_LEASE_SQL = text("""
WITH leased AS (
    SELECT od.event_id, od.sink
      FROM outbox_delivery od
      JOIN outbox o ON o.event_id = od.event_id
     WHERE od.sink = :sink
       AND od.status = :pending
       AND (od.locked_until IS NULL OR od.locked_until < now())
       AND NOT EXISTS (
            SELECT 1
              FROM outbox o_prior
              JOIN outbox_delivery od_prior
                ON od_prior.event_id = o_prior.event_id
             WHERE o_prior.aggregate_id = o.aggregate_id
               AND o_prior.aggregate_version < o.aggregate_version
               AND od_prior.sink = od.sink
               AND od_prior.status <> :done
       )
     ORDER BY o.event_id
     LIMIT :batch_size
     FOR UPDATE OF od SKIP LOCKED
)
UPDATE outbox_delivery upd
   SET locked_by = :worker_id,
       locked_until = now() + make_interval(secs => :lease_ttl_s),
       status = :in_flight
  FROM leased
 WHERE upd.event_id = leased.event_id
   AND upd.sink = leased.sink
RETURNING upd.event_id, upd.sink, upd.attempt_count
""")


async def lease_batch(
    session: AsyncSession,
    *,
    worker_id: str,
    sink: OutboxSink,
    batch_size: int = 16,
    lease_ttl_seconds: int = 60,
) -> list[LeasedEvent]:
    """Lease up to ``batch_size`` events for ``sink``; commit on caller side.

    Marks each leased delivery as ``in_flight`` to make crash recovery
    deterministic (a crashed worker's rows return to ``pending`` once
    ``locked_until`` expires; another worker re-leases and resumes).
    """
    result = await session.execute(
        _LEASE_SQL,
        {
            "sink": sink.value,
            "pending": OutboxDeliveryStatus.pending.value,
            "in_flight": OutboxDeliveryStatus.in_flight.value,
            "done": OutboxDeliveryStatus.done.value,
            "batch_size": batch_size,
            "worker_id": worker_id,
            "lease_ttl_s": lease_ttl_seconds,
        },
    )
    rows = result.mappings().all()
    if not rows:
        return []

    # Hydrate with outbox payload for the leased event_ids.
    event_ids = [r["event_id"] for r in rows]
    fetch = await session.execute(
        text(
            "SELECT event_id, aggregate_type, aggregate_id, aggregate_version, "
            "env_id, op, payload, created_at "
            "FROM outbox WHERE event_id = ANY(:ids)"
        ),
        {"ids": event_ids},
    )
    by_id = {r["event_id"]: r for r in fetch.mappings().all()}

    leased: list[LeasedEvent] = []
    for r in rows:
        outbox_row = by_id[r["event_id"]]
        leased.append(
            LeasedEvent(
                event_id=outbox_row["event_id"],
                sink=OutboxSink(r["sink"]),
                aggregate_type=outbox_row["aggregate_type"],
                aggregate_id=outbox_row["aggregate_id"],
                aggregate_version=outbox_row["aggregate_version"],
                env_id=outbox_row["env_id"],
                op=outbox_row["op"],
                payload=dict(outbox_row["payload"] or {}),
                attempt_count=r["attempt_count"],
                created_at=outbox_row["created_at"],
            )
        )
    return leased


# ---------------------------------------------------------------------------
# Mark done / fail / dead-letter
# ---------------------------------------------------------------------------


async def mark_done(
    session: AsyncSession,
    *,
    event_id: int,
    sink: OutboxSink,
    env_id: UUID,
    event_created_at: dt.datetime,
) -> None:
    """Finalize a successful delivery + advance projection_state watermark.

    Called from inside the caller's transaction.
    """
    now = dt.datetime.now(tz=dt.UTC)
    await session.execute(
        update(OutboxDelivery)
        .where(
            OutboxDelivery.event_id == event_id,
            OutboxDelivery.sink == sink.value,
        )
        .values(
            status=OutboxDeliveryStatus.done.value,
            done_at=now,
            locked_by=None,
            locked_until=None,
            last_error=None,
        )
    )

    # Advance watermark only if event_id is newer than what's recorded.
    lag = (now - event_created_at).total_seconds()
    await session.execute(
        text("""
            UPDATE projection_state
               SET last_event_id = GREATEST(COALESCE(last_event_id, 0), :event_id),
                   last_success_at = :now,
                   lag_seconds = :lag,
                   status = 'healthy',
                   last_error = NULL
             WHERE sink = :sink AND env_id = :env_id
        """),
        {
            "event_id": event_id,
            "now": now,
            "lag": lag,
            "sink": sink.value,
            "env_id": env_id,
        },
    )


def _backoff_seconds(attempt_count: int) -> int:
    """Capped exponential backoff: 2, 4, 8, 16, …, max 600 seconds."""
    base = 2 ** min(attempt_count, 10)
    return min(base, 600)


async def mark_fail(
    session: AsyncSession,
    *,
    event_id: int,
    sink: OutboxSink,
    env_id: UUID,
    error: str,
    max_attempts: int = 8,
) -> bool:
    """Increment attempt; release lease with backoff; dead-letter if exhausted.

    Returns True if the row was dead-lettered.
    """
    # Read current attempt + decide.
    row = (
        (
            await session.execute(
                text("SELECT attempt_count FROM outbox_delivery WHERE event_id = :e AND sink = :s FOR UPDATE"),
                {"e": event_id, "s": sink.value},
            )
        )
        .mappings()
        .first()
    )
    attempt_count = (row["attempt_count"] if row else 0) + 1
    dead = attempt_count >= max_attempts

    truncated_error = (error or "")[:2000]
    if dead:
        await session.execute(
            update(OutboxDelivery)
            .where(
                OutboxDelivery.event_id == event_id,
                OutboxDelivery.sink == sink.value,
            )
            .values(
                status=OutboxDeliveryStatus.dead.value,
                attempt_count=attempt_count,
                locked_by=None,
                locked_until=None,
                last_error=truncated_error,
            )
        )
        await session.execute(
            text("""
                UPDATE projection_state
                   SET status = 'degraded',
                       last_error = :err
                 WHERE sink = :sink AND env_id = :env_id
            """),
            {"err": truncated_error, "sink": sink.value, "env_id": env_id},
        )
        log.warning(
            "projection-worker dead-lettered event_id=%s sink=%s after %s attempts",
            event_id,
            sink.value,
            attempt_count,
        )
        return True

    backoff = _backoff_seconds(attempt_count)
    await session.execute(
        text("""
            UPDATE outbox_delivery
               SET status = :pending,
                   attempt_count = :ac,
                   locked_by = NULL,
                   locked_until = now() + make_interval(secs => :b),
                   last_error = :err
             WHERE event_id = :e AND sink = :s
        """),
        {
            "pending": OutboxDeliveryStatus.pending.value,
            "ac": attempt_count,
            "b": backoff,
            "err": truncated_error,
            "e": event_id,
            "s": sink.value,
        },
    )
    log.info(
        "projection-worker requeued event_id=%s sink=%s attempt=%s backoff=%ss",
        event_id,
        sink.value,
        attempt_count,
        backoff,
    )
    return False


__all__: Sequence[str] = (
    "LeasedEvent",
    "lease_batch",
    "mark_done",
    "mark_fail",
)
