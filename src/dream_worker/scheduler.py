"""Dream-worker scheduler — APScheduler binding for cadence-driven passes.

This module owns the **periodic invocation** of :func:`dream_worker.jobs.run_dream_pass`.
It does not own the per-pass lifecycle (dream_runs row, advisory lock,
heartbeat) — that lives in :mod:`dream_worker.jobs`. The scheduler's
responsibility is purely:

1. Register dream jobs with cadences
   from :class:`memory_mcp.config.Settings`.
2. On each tick, refresh the env list and iterate envs sequentially,
   calling :func:`run_dream_pass` per ``(env, mode)``.
3. Surface graceful shutdown: a ``stopping`` flag halts the env loop
   between envs (so SIGTERM doesn't have to wait for an entire 100-env
   tick to finish before draining), and ``shutdown(wait=True)`` lets
   the active pass complete.

Design notes (rubber-duck-validated)
------------------------------------

* No ``asyncio.wait_for`` around ``run_dream_pass``. ``CancelledError``
  bypasses the dream_runs finalize / advisory-unlock path. The
  ``dream_pass_timeout_seconds`` config is **observability only** in v1.
* APScheduler's ``shutdown(wait=...)`` is a synchronous method on
  ``AsyncIOScheduler``. Do not ``await`` it.
* ``coalesce=True`` collapses backlog into one run if the loop is
  blocked. ``misfire_grace_time = cadence_seconds`` tolerates startup
  / GC / DB pause without dropping ticks.
* ``max_instances=settings.dream_scheduler_max_instances`` (default
  ``1``) prevents two ticks from racing for the same env's advisory
  lock and wasting cycles.
* Per-env ``AgentContext`` (least privilege, matches the
  ``run_dream_pass`` invariant that ``env_id ∈ attached_env_ids``).
* Failure isolation: one env raising does not stop the env loop for
  later envs in the same tick.

Shutdown order on SIGTERM (owned by :mod:`dream_worker.runner`)
---------------------------------------------------------------

1. ``DreamScheduler.shutdown(wait=True)`` — APScheduler stops accepting
   new jobs; ``stopping`` flag halts the env loop; in-flight pass
   completes naturally and releases its advisory lock.
2. Close vector store + reset LLM client singleton.
3. Dispose the Postgres engine.

Reversing this order risks pool errors mid-pass.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from dream_worker.jobs import (
    DreamMode,
    DreamPassReport,
    list_active_envs,
    refresh_proposals_open_gauge,
    run_dream_pass,
)
from memory_mcp.identity import AgentContext

if TYPE_CHECKING:
    from memory_mcp.config import Settings
    from memory_mcp.db.vector.base import VectorStore
    from memory_mcp.dream.summarizer import DreamSummarizer
    from memory_mcp.embeddings.base import Embedder

log = logging.getLogger(__name__)


# Stable APScheduler job IDs — used by tests to assert registration and
# by `dream_status` to look up next-run-time / next-fire-time.
JOB_ID_DECAY = "dream-decay"
JOB_ID_DEDUPE = "dream-dedupe"
JOB_ID_PROMOTE = "dream-promote"
JOB_ID_DECISION_CONFLICTS = "dream-decision-conflicts"
JOB_ID_METRICS_REFRESH = "dream-metrics-refresh"


class DreamScheduler:
    """Owns the APScheduler instance and the per-mode tick wrappers.

    The scheduler does not own the engine, vector store, or LLM client
    lifecycles — :mod:`dream_worker.runner` builds those, hands them in,
    and tears them down after :meth:`shutdown` returns.

    Parameters
    ----------
    settings:
        Caller-provided settings. Cadences and ``dream_scheduler_max_instances``
        are read from here.
    summarizer:
        Built once per worker process via
        :func:`memory_mcp.dream.summarizer.build_summarizer`. Reused
        across all passes so the LLM client connection pool (if any) is
        shared.
    embedder:
        Required by ``dedupe``. Decay and promote ignore it.
    vector_store:
        Required by ``dedupe``. Decay and promote ignore it.
    agent_id:
        UUID of the agent the worker attributes outbox/audit events to.
        Resolved by :mod:`dream_worker.runner` via the public
        ``IdentityResolver.resolve(None, None, None)`` path.
    agent_name:
        Human-readable name; carried in :class:`AgentContext` for logs.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        summarizer: DreamSummarizer,
        embedder: Embedder,
        vector_store: VectorStore,
        agent_id: UUID,
        agent_name: str | None = None,
    ) -> None:
        self._settings = settings
        self._summarizer = summarizer
        self._embedder = embedder
        self._vector_store = vector_store
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._scheduler = AsyncIOScheduler()
        # Set on shutdown initiation. Each tick checks this between
        # envs so SIGTERM doesn't have to wait for the full env loop
        # before APScheduler can drain the active job.
        self._stopping = False
        # Tracks whether ``start()`` has been called. Re-entry is a bug.
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register the three jobs and start the scheduler.

        Safe to call once. Subsequent calls raise ``RuntimeError`` —
        the scheduler is owned by a single runner.
        """
        if self._started:
            raise RuntimeError("DreamScheduler.start() already called")
        self._register_jobs()
        self._scheduler.start()
        self._started = True
        log.info(
            "dream_worker scheduler started "
            "decay=%ss dedupe=%ss promote=%ss decision_conflicts=%ss max_instances=%s",
            self._settings.dream_decay_cadence_seconds,
            self._settings.dream_dedupe_cadence_seconds,
            self._settings.dream_promote_cadence_seconds,
            self._settings.dream_decision_conflicts_cadence_seconds,
            self._settings.dream_scheduler_max_instances,
        )

    def shutdown(self, wait: bool = True) -> None:
        """Stop the scheduler and drain in-flight jobs.

        APScheduler 3.x's ``AsyncIOScheduler.shutdown`` is synchronous —
        do **not** ``await`` this method. The ``stopping`` flag is set
        first so any active env loop bails out between envs rather than
        running another env after shutdown was requested.
        """
        self._stopping = True
        if not self._started:
            return
        # APScheduler 3.x .shutdown is sync — must not be awaited.
        self._scheduler.shutdown(wait=wait)
        log.info("dream_worker scheduler shutdown complete (wait=%s)", wait)

    @property
    def stopping(self) -> bool:
        return self._stopping

    @property
    def scheduler(self) -> AsyncIOScheduler:
        """Exposed for tests — assert job registration / introspection."""
        return self._scheduler

    # ------------------------------------------------------------------
    # Manual trigger (used by `dream_run` MCP tool, sync-mode tests)
    # ------------------------------------------------------------------

    async def trigger_now(
        self,
        env_id: UUID,
        mode: DreamMode,
        *,
        triggered_by: str = "manual",
    ) -> DreamPassReport:
        """Run a single pass immediately, bypassing the cadence.

        Used by ``dream_run`` (manual trigger) and by tests in
        ``wait=true`` mode. Goes through the same advisory-lock + dream_runs
        + heartbeat lifecycle as scheduled passes.
        """
        ctx = self._build_actor_ctx(env_id)
        return await run_dream_pass(
            env_id,
            mode,
            actor_ctx=ctx,
            summarizer=self._summarizer,
            embedder=self._embedder if mode is DreamMode.dedupe else None,
            vector_store=(
                self._vector_store
                if mode in {DreamMode.dedupe, DreamMode.decision_conflicts}
                else None
            ),
            settings=self._settings,
            triggered_by=triggered_by,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _register_jobs(self) -> None:
        """Register one APScheduler job per :class:`DreamMode`."""
        max_instances = self._settings.dream_scheduler_max_instances
        for mode, job_id, cadence in self._mode_specs():
            self._scheduler.add_job(
                self._make_tick_func(mode),
                trigger="interval",
                seconds=cadence,
                id=job_id,
                name=f"dream:{mode.value}",
                coalesce=True,
                max_instances=max_instances,
                # Tolerate up to one cadence worth of slip (startup, GC,
                # transient DB pause). With ``coalesce=True`` we still
                # collapse a backlog into one run.
                misfire_grace_time=cadence,
                replace_existing=True,
            )
        # 4th job: metrics gauge refresher. Lightweight SQL aggregate;
        # disabled when cadence is 0 so tests / minimal deployments
        # don't pay for it.
        refresh_cadence = self._settings.dream_metrics_refresh_seconds
        if refresh_cadence > 0:
            self._scheduler.add_job(
                refresh_proposals_open_gauge,
                trigger="interval",
                seconds=refresh_cadence,
                id=JOB_ID_METRICS_REFRESH,
                name="dream:metrics-refresh",
                coalesce=True,
                max_instances=1,
                misfire_grace_time=refresh_cadence,
                replace_existing=True,
            )

    def _mode_specs(self) -> list[tuple[DreamMode, str, int]]:
        return [
            (
                DreamMode.decay,
                JOB_ID_DECAY,
                self._settings.dream_decay_cadence_seconds,
            ),
            (
                DreamMode.dedupe,
                JOB_ID_DEDUPE,
                self._settings.dream_dedupe_cadence_seconds,
            ),
            (
                DreamMode.promote,
                JOB_ID_PROMOTE,
                self._settings.dream_promote_cadence_seconds,
            ),
            (
                DreamMode.decision_conflicts,
                JOB_ID_DECISION_CONFLICTS,
                self._settings.dream_decision_conflicts_cadence_seconds,
            ),
        ]

    def _make_tick_func(self, mode: DreamMode):  # noqa: ANN202 — APScheduler callable
        """Return an async callable APScheduler can register for ``mode``.

        The closure captures ``mode`` so the same wrapper logic serves
        all three modes; tests can introspect by mode without registering
        the scheduler.
        """
        async def _tick() -> None:
            await self._run_mode_tick(mode)

        _tick.__name__ = f"_tick_{mode.value}"
        _tick.__qualname__ = f"DreamScheduler._tick_{mode.value}"
        return _tick

    async def _run_mode_tick(self, mode: DreamMode) -> None:
        """One tick of ``mode``: iterate envs, dispatch, isolate failures.

        Per-env failure isolation: an exception in env N does not stop
        the loop for envs N+1..N+k. Each pass already has its own
        try/except/finalize in :func:`run_dream_pass`; this wrapper
        catches anything that escapes (e.g. lock-acquisition errors,
        DB connection issues outside the pass body).

        Cooperative shutdown: between envs, check ``self._stopping`` so
        SIGTERM can drain quickly. APScheduler's ``shutdown(wait=True)``
        will still wait for the in-flight env's pass to finish.

        Env discovery is per-tick: a new env created mid-flight is
        picked up on the next tick automatically.
        """
        try:
            env_ids = await list_active_envs()
        except Exception:  # noqa: BLE001 — log + bail; next tick will retry
            log.exception(
                "dream_worker tick mode=%s: list_active_envs() failed; "
                "skipping this tick",
                mode.value,
            )
            return

        if not env_ids:
            log.debug("dream_worker tick mode=%s: no envs; skipping", mode.value)
            return

        log.debug(
            "dream_worker tick mode=%s envs=%d", mode.value, len(env_ids),
        )

        for env_id in env_ids:
            if self._stopping:
                log.info(
                    "dream_worker tick mode=%s halted by shutdown "
                    "(remaining envs skipped)",
                    mode.value,
                )
                return
            await self._dispatch_one(env_id, mode)

    async def _dispatch_one(self, env_id: UUID, mode: DreamMode) -> None:
        """Dispatch one ``(env_id, mode)`` pass with per-env isolation.

        Any exception escaping :func:`run_dream_pass` is logged and
        swallowed; a future env in the same tick must not be poisoned
        by a transient failure.
        """
        try:
            await run_dream_pass(
                env_id,
                mode,
                actor_ctx=self._build_actor_ctx(env_id),
                summarizer=self._summarizer,
                embedder=self._embedder if mode is DreamMode.dedupe else None,
                vector_store=(
                    self._vector_store
                    if mode in {DreamMode.dedupe, DreamMode.decision_conflicts}
                    else None
                ),
                settings=self._settings,
                triggered_by="scheduler",
            )
        except asyncio.CancelledError:
            # Surface cancellation upward — the scheduler is shutting
            # down. The pass itself catches non-Cancelled exceptions and
            # finalizes its dream_run row; cancellation here means the
            # whole event loop is going down.
            raise
        except Exception:  # noqa: BLE001 — per-env isolation
            log.exception(
                "dream_worker tick mode=%s env=%s: unexpected error escaped "
                "run_dream_pass; continuing with next env",
                mode.value, env_id,
            )

    def _build_actor_ctx(self, env_id: UUID) -> AgentContext:
        """Per-env :class:`AgentContext` matching ``run_dream_pass`` invariant.

        The pass requires ``env_id ∈ ctx.attached_env_ids`` (decay's
        env-visibility check). We attach exactly the one env the pass
        operates on, both to satisfy the check and to enforce
        least-privilege on outbox/audit events.
        """
        return AgentContext(
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            session_id=None,
            attached_env_ids=[env_id],
            is_default_agent=True,
        )


__all__ = [
    "JOB_ID_DECAY",
    "JOB_ID_DEDUPE",
    "JOB_ID_DECISION_CONFLICTS",
    "JOB_ID_PROMOTE",
    "DreamScheduler",
]
