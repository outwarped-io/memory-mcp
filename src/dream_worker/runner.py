"""Dream worker entrypoint — APScheduler runner driving cadence-driven passes.

This is the long-running process spec'd in the Phase 2.2 plan
(``p2.2-runner``). It:

1. Loads :class:`memory_mcp.config.Settings` and initializes the Postgres
   engine.
2. Builds **once-per-process** singletons for the summarizer, embedder,
   vector store, and (lazily) the LLM client.
3. Resolves the dream-worker's :class:`AgentContext` via the public
   :class:`memory_mcp.identity.IdentityResolver` API (``resolve(None,
   None, None)`` returns the server-default agent).
4. If ``settings.dream_enabled`` is true, starts a
   :class:`dream_worker.scheduler.DreamScheduler` (3 jobs: decay /
   dedupe / promote). Otherwise enters a heartbeat-only idle loop so
   the container can run without scheduling work (useful for tests +
   dry-run smokes).
5. On SIGTERM/SIGINT: shuts down the scheduler (waiting for in-flight
   passes to drain), closes the vector store, resets the LLM-client
   singleton, and disposes the engine — in that order. Reversing the
   order risks pool errors mid-pass.

Operational safety
------------------

* Cadence defaults to "off" in tests/dev (``dream_enabled=False``).
  Production compose flips it on.
* If the scheduler raises during ``start()``, the process exits with
  a non-zero code. Misconfigurations should fail fast.
* If a pass raises mid-tick, the scheduler tick wrapper catches it
  (per-env isolation). The worker keeps running.
* Engine disposal happens **after** the scheduler has fully drained,
  so no pass holds a connection from the pool when the engine closes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal

from dream_worker.scheduler import DreamScheduler
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.postgres import dispose_engine, init_engine
from memory_mcp.db.vector.qdrant import QdrantVectorStore
from memory_mcp.dream.summarizer import build_summarizer
from memory_mcp.embeddings.base import get_embedder
from memory_mcp.identity import get_identity_resolver
from memory_mcp.llm.base import get_llm_client, reset_llm_client

logger = logging.getLogger("dream_worker")


async def _run(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    init_engine(settings)

    # Build the LLM client first when summarizer is LLM-backed so the
    # summarizer can reuse the singleton (connection-pool reuse). When
    # summarizer is template-backed, ``build_summarizer`` short-circuits
    # and never touches the LLM module.
    llm_client = None
    if settings.dream_summarizer == "llm":
        llm_client = await get_llm_client(settings)
    summarizer = build_summarizer(settings, llm_client=llm_client)

    embedder = get_embedder(settings)
    vector_store = QdrantVectorStore(settings)

    # Identity: the dream worker writes outbox/audit events as the
    # default agent. ``resolve(None, None, None)`` is the public path
    # to bootstrap-or-read the default agent file.
    resolver = get_identity_resolver(settings)
    default_ctx = await resolver.resolve(
        agent_id_header=None,
        agent_name_header=None,
        session_id_header=None,
    )

    logger.info(
        "dream_worker starting enabled=%s summarizer=%s llm_backend=%s agent_id=%s",
        settings.dream_enabled,
        settings.dream_summarizer,
        settings.llm_backend,
        default_ctx.agent_id,
    )

    scheduler: DreamScheduler | None = None
    if settings.dream_enabled:
        scheduler = DreamScheduler(
            settings,
            summarizer=summarizer,
            embedder=embedder,
            vector_store=vector_store,
            agent_id=default_ctx.agent_id,
            agent_name=default_ctx.agent_name,
        )
        scheduler.start()
    else:
        logger.info(
            "dream_worker DREAM_ENABLED=false — entering heartbeat-only idle loop (no jobs registered)",
        )

    # SIGTERM/SIGINT handling.
    stop = asyncio.Event()

    def _handle_signal(*_: object) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_signal)

    try:
        # Idle until signal. The scheduler runs in the background on
        # the same event loop; we just need to keep this coroutine
        # alive so the loop doesn't exit.
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=60)
            if scheduler is None:
                logger.debug("dream_worker heartbeat (idle)")
    finally:
        # Shutdown order matters: scheduler first (drain in-flight
        # passes; release advisory locks; finalize dream_runs), then
        # external clients, finally the DB engine.
        await _shutdown(scheduler=scheduler, vector_store=vector_store)
        logger.info("dream_worker shutdown complete")


async def _shutdown(
    *,
    scheduler: DreamScheduler | None,
    vector_store: QdrantVectorStore,
) -> None:
    """Drain scheduler then close all per-process resources.

    Each step is wrapped in try/except so a failure in one cleanup
    path does not block others. The Postgres engine MUST be the last
    thing disposed: while the scheduler is draining, in-flight passes
    are still using pooled connections.
    """
    if scheduler is not None:
        try:
            # APScheduler's .shutdown is sync; the wrapper sets
            # ``stopping`` first so any active env loop bails between
            # envs, then waits for the active pass.
            scheduler.shutdown(wait=True)
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception("dream_worker scheduler shutdown raised")

    try:
        await vector_store.close()
    except Exception:  # noqa: BLE001
        logger.exception("dream_worker vector_store.close raised")

    try:
        await reset_llm_client()
    except Exception:  # noqa: BLE001
        logger.exception("dream_worker reset_llm_client raised")

    try:
        await dispose_engine()
    except Exception:  # noqa: BLE001
        logger.exception("dream_worker dispose_engine raised")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(_run())


if __name__ == "__main__":
    main()
