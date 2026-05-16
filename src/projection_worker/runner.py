"""Projection worker entrypoint — drains the outbox into Qdrant + Neo4j.

Phase 2.1 scope:

* Two sinks: ``qdrant`` (memory aggregates) and ``neo4j`` (entity +
  relation aggregates).
* Per-aggregate ordering enforced by the lease query in :mod:`lease`.
* At-least-once delivery with leasing + dead-lettering.

The worker is runnable in two ways:

* ``python -m projection_worker`` — long-running loop, signal-handled
  graceful shutdown, sleep when **both** queues are empty.
* ``await drain_once(...)`` — process one batch and return; used by
  integration smokes and tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import uuid
from dataclasses import dataclass

from memory_mcp.config import Settings, get_settings
from memory_mcp.db.graph import get_graph_store
from memory_mcp.db.graph.base import GraphStore
from memory_mcp.db.postgres import dispose_engine, init_engine, session_scope
from memory_mcp.db.types import OutboxSink
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.db.vector.qdrant import QdrantVectorStore
from memory_mcp.embeddings.base import Embedder, get_embedder
from projection_worker.handlers.neo4j import handle_neo4j_event
from projection_worker.handlers.qdrant import handle_qdrant_event
from projection_worker.lease import LeasedEvent, lease_batch, mark_done, mark_fail

logger = logging.getLogger("projection_worker")


# ---------------------------------------------------------------------------
# Drain-once API (used by tests + the long-running loop)
# ---------------------------------------------------------------------------


@dataclass
class DrainStats:
    leased: int = 0
    succeeded: int = 0
    failed: int = 0
    dead_lettered: int = 0


async def drain_once(
    *,
    sink: OutboxSink,
    vector_store: VectorStore | None = None,
    graph_store: GraphStore | None = None,
    embedder: Embedder | None = None,
    worker_id: str | None = None,
    batch_size: int = 16,
    lease_ttl_seconds: int = 60,
    max_attempts: int = 8,
) -> DrainStats:
    """Lease a batch from ``sink`` and process to completion.

    Required arguments depend on ``sink``:

    * ``OutboxSink.qdrant`` — needs ``vector_store`` + ``embedder``.
    * ``OutboxSink.neo4j`` — needs ``graph_store``.

    Each event is processed in **its own transaction**. The lease is
    taken in a separate short transaction so a crash mid-processing
    doesn't lose the lease commit.
    """
    worker_id = worker_id or _default_worker_id()
    stats = DrainStats()

    # Step 1: lease + commit.
    async with session_scope() as s:
        events = await lease_batch(
            s,
            worker_id=worker_id,
            sink=sink,
            batch_size=batch_size,
            lease_ttl_seconds=lease_ttl_seconds,
        )
    stats.leased = len(events)
    if not events:
        return stats

    # Step 2: process each event; per-event txn for finalization.
    for event in events:
        try:
            await _dispatch_event(
                event,
                sink=sink,
                vector_store=vector_store,
                graph_store=graph_store,
                embedder=embedder,
            )
        except Exception as exc:
            logger.exception(
                "projection-worker handler failed event_id=%s sink=%s "
                "aggregate=%s/%s/%s",
                event.event_id, sink.value, event.aggregate_type,
                event.aggregate_id, event.aggregate_version,
            )
            async with session_scope() as s:
                dead = await mark_fail(
                    s,
                    event_id=event.event_id,
                    sink=sink,
                    env_id=event.env_id,
                    error=f"{type(exc).__name__}: {exc}",
                    max_attempts=max_attempts,
                )
            if dead:
                stats.dead_lettered += 1
            else:
                stats.failed += 1
            continue

        async with session_scope() as s:
            await mark_done(
                s,
                event_id=event.event_id,
                sink=sink,
                env_id=event.env_id,
                event_created_at=event.created_at,
            )
        stats.succeeded += 1

    return stats


async def _dispatch_event(
    event: LeasedEvent,
    *,
    sink: OutboxSink,
    vector_store: VectorStore | None,
    graph_store: GraphStore | None,
    embedder: Embedder | None,
) -> None:
    """Route ``event`` to the per-sink handler with a clear arg-validation error."""
    if sink == OutboxSink.qdrant:
        if vector_store is None or embedder is None:
            raise ValueError(
                "drain_once(sink=qdrant) requires vector_store + embedder"
            )
        await handle_qdrant_event(
            event,
            vector_store=vector_store,
            embedder=embedder,
        )
        return
    if sink == OutboxSink.neo4j:
        if graph_store is None:
            raise ValueError("drain_once(sink=neo4j) requires graph_store")
        await handle_neo4j_event(event, graph_store=graph_store)
        return
    raise NotImplementedError(f"sink {sink.value!r} not handled")


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Long-running loop
# ---------------------------------------------------------------------------


async def _run(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    init_engine(settings)
    embedder = get_embedder(settings)
    vector_store: VectorStore = QdrantVectorStore(settings)
    graph_store: GraphStore = get_graph_store(settings)
    worker_id = _default_worker_id()
    logger.info(
        "projection-worker started worker_id=%s qdrant=%s graph_backend=%s",
        worker_id, settings.qdrant_url, settings.graph_backend,
    )

    # Initialize neo4j schema on startup so the first relation event
    # has its constraints already in place. PostgresGraphStore returns
    # a no-op.
    try:
        if hasattr(graph_store, "init_schema"):
            await graph_store.init_schema()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        # Don't crash the worker on transient Neo4j unavailability —
        # the per-event handler will surface real errors via
        # dead-lettering.
        logger.exception("projection-worker init_schema failed; continuing")

    stop = asyncio.Event()

    def _handle(*_: object) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle)

    idle_sleep_s = float(os.environ.get("PROJECTION_WORKER_IDLE_SLEEP_S", "1.0"))

    try:
        while not stop.is_set():
            qstats = await drain_once(
                sink=OutboxSink.qdrant,
                vector_store=vector_store,
                embedder=embedder,
                worker_id=worker_id,
            )
            nstats = await drain_once(
                sink=OutboxSink.neo4j,
                graph_store=graph_store,
                worker_id=worker_id,
            )
            total_leased = qstats.leased + nstats.leased
            if total_leased == 0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=idle_sleep_s)
            else:
                logger.info(
                    "projection-worker drained "
                    "qdrant(leased=%s succeeded=%s failed=%s dead=%s) "
                    "neo4j(leased=%s succeeded=%s failed=%s dead=%s)",
                    qstats.leased, qstats.succeeded, qstats.failed, qstats.dead_lettered,
                    nstats.leased, nstats.succeeded, nstats.failed, nstats.dead_lettered,
                )
    finally:
        await vector_store.close()
        await graph_store.close()
        await dispose_engine()
        logger.info("projection-worker shutting down worker_id=%s", worker_id)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(_run())


if __name__ == "__main__":
    main()
