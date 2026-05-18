"""Dream-worker job orchestration.

This module owns the **lifecycle** of a single dream pass tick — the
``dream_runs`` row management, advisory-lock coordination, per-env
:class:`AgentContext` construction, dispatch to the per-mode pass
function, and heartbeat update.

The :class:`~dream_worker.scheduler.DreamScheduler` calls
:func:`run_dream_pass` from APScheduler triggers; tests + the
``dream_run`` MCP tool call it directly.

Lifecycle
---------

1. **Acquire advisory lock** ``(mode_key, env_id_hash)``. If another
   worker replica holds the lock, return ``DreamPassOutcome.skipped``
   without writing a ``dream_runs`` row.
2. **INSERT ``dream_runs``** with ``status='running'``, ``started_at=now``,
   ``triggered_by``, ``summarizer_kind``.
3. **Run the pass** — dispatch by ``mode``. The pass is wrapped in a
   ``try``/``except`` that captures any exception into ``last_error``
   and marks the run ``failed``.
4. **UPDATE ``dream_runs``** with ``ended_at``, ``status``,
   ``summary`` (the pass-result dict), ``last_error`` (if failed).
5. **Update heartbeat** in ``projection_state`` so ``/readyz`` can
   report dream-worker liveness without scanning ``dream_runs``.
6. **Release advisory lock**.

Idempotency
-----------

The advisory lock is per ``(mode, env_id)`` — two replicas attempting
the same env+mode at the same tick produce one ``dream_runs`` row, not
two. The lock is session-scoped (held for the whole pass) using
``pg_try_advisory_lock`` / ``pg_advisory_unlock`` rather than the
xact-level variant because passes contain many internal transactions.

If the worker process dies mid-pass the lock is released by Postgres
when the connection closes; the orphaned ``dream_runs`` row is left
``status='running'`` and surfaces in observability via the partial
index ``dream_runs_running_idx``. A future pruning job can mark stale
runs ``failed`` (out of scope for v1; documented as a known limit).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import time
import uuid as uuidlib
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from memory_mcp_schemas.dream import DreamMode, DreamPassOutcome
from dream_worker.decision_conflicts import run_decision_conflict_pass
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import DreamRun, Environment, ProjectionState
from memory_mcp.db.postgres import session_scope
from memory_mcp.dream.passes.decay import run_decay
from memory_mcp.dream.passes.dedupe import run_dedupe
from memory_mcp.dream.passes.promote import run_promote
from memory_mcp.dream.passes.recount import run_recount
from memory_mcp.identity import AgentContext

if TYPE_CHECKING:
    from memory_mcp.db.vector.base import VectorStore
    from memory_mcp.dream.summarizer import DreamSummarizer
    from memory_mcp.embeddings.base import Embedder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------




# Stable advisory-lock namespace ids (first 32-bit slot). Different per
# mode so `(decay, env_X)` and `(dedupe, env_X)` can run concurrently.
_LOCK_NS_BASE = 0x4D454D  # "MEM"
_MODE_LOCK_KEY: dict[DreamMode, int] = {
    DreamMode.decay: _LOCK_NS_BASE | (1 << 24),
    DreamMode.dedupe: _LOCK_NS_BASE | (2 << 24),
    DreamMode.promote: _LOCK_NS_BASE | (3 << 24),
    DreamMode.decision_conflicts: _LOCK_NS_BASE | (4 << 24),
    DreamMode.recount: _LOCK_NS_BASE | (5 << 24),
}

# Synthetic sink name for the heartbeat row. Re-uses the existing
# ``projection_state`` table so ``/readyz`` can probe one schema.
HEARTBEAT_SINK = "dream_worker"


@dataclass(frozen=True)
class DreamPassReport:
    """Per-call report — what the runner returns to its caller."""

    env_id: UUID
    mode: DreamMode
    outcome: DreamPassOutcome
    dream_run_id: UUID | None = None
    summary: dict[str, Any] | None = None
    last_error: str | None = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Advisory lock helpers
# ---------------------------------------------------------------------------


def _env_lock_key(env_id: UUID) -> int:
    """Map a UUID to a deterministic signed 32-bit int.

    Postgres advisory locks accept either one bigint or two ints; we
    pass ``(mode_key, env_lock_key)``. ``env_lock_key`` is derived from
    the UUID by XOR-folding to 31 bits so the result fits in a signed
    int32 with no sign issues.
    """
    h = env_id.int
    folded = 0
    while h:
        folded ^= h & 0x7FFFFFFF
        h >>= 31
    return folded


async def _try_acquire_lock_in_session(
    session: Any, *, mode: DreamMode, env_id: UUID,
) -> bool:
    """Try to take the per-(mode, env) session-scoped advisory lock.

    Returns ``True`` on success. The lock MUST be released on the same
    SQLAlchemy connection via :func:`_release_lock_in_session`. Postgres
    auto-releases when the connection closes, so a worker crash mid-pass
    cannot orphan the lock.
    """
    mode_key = _MODE_LOCK_KEY[mode]
    env_key = _env_lock_key(env_id)
    stmt = text("SELECT pg_try_advisory_lock(:m, :e) AS locked")
    row = (await session.execute(stmt, {"m": mode_key, "e": env_key})).first()
    return bool(row and row.locked)


async def _release_lock_in_session(
    session: Any, *, mode: DreamMode, env_id: UUID,
) -> None:
    mode_key = _MODE_LOCK_KEY[mode]
    env_key = _env_lock_key(env_id)
    stmt = text("SELECT pg_advisory_unlock(:m, :e)")
    await session.execute(stmt, {"m": mode_key, "e": env_key})


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


async def update_heartbeat(
    *,
    env_id: UUID,
    mode: DreamMode,
    success: bool,
    now: dt.datetime,
    last_error: str | None = None,
) -> None:
    """Upsert a row into ``projection_state`` for the dream-worker sink.

    On success: bump ``last_success_at`` and reset ``lag_seconds`` to
    zero. On failure: PRESERVE prior ``last_success_at`` and
    ``lag_seconds`` (so a long unbroken success streak doesn't get
    erased by one failure), update only ``status`` and ``last_error``.
    """
    sink = f"{HEARTBEAT_SINK}:{mode.value}"
    # `projection_state.status` is constrained to {healthy, degraded, down,
    # rebuilding} by a CHECK constraint shared with the projection-worker
    # sinks, so we map dream-worker outcomes onto that vocabulary.
    status = "healthy" if success else "down"
    if success:
        set_clause: dict[str, Any] = {
            "last_success_at": now,
            "lag_seconds": 0,
            "status": status,
            "last_error": None,
        }
    else:
        # Preserve last_success_at + lag_seconds; only mark error.
        set_clause = {
            "status": status,
            "last_error": last_error,
        }

    async with session_scope() as s:
        stmt = (
            pg_insert(ProjectionState)
            .values(
                sink=sink,
                env_id=env_id,
                last_success_at=now if success else None,
                lag_seconds=0 if success else None,
                status=status,
                last_error=last_error,
            )
            .on_conflict_do_update(
                index_elements=["sink", "env_id"],
                set_=set_clause,
            )
        )
        await s.execute(stmt)


# ---------------------------------------------------------------------------
# Dream run lifecycle
# ---------------------------------------------------------------------------


async def _insert_dream_run(
    *,
    env_id: UUID,
    mode: DreamMode,
    triggered_by: str,
    summarizer_kind: str | None,
    started_at: dt.datetime,
) -> UUID:
    run_id = uuidlib.uuid4()
    async with session_scope() as s:
        await s.execute(
            DreamRun.__table__.insert().values(
                id=run_id,
                env_id=env_id,
                mode=mode.value,
                status="running",
                started_at=started_at,
                triggered_by=triggered_by,
                summarizer_kind=summarizer_kind,
                summary={},
            )
        )
    return run_id


async def _finalize_dream_run(
    *,
    run_id: UUID,
    status: str,
    summary: dict[str, Any],
    ended_at: dt.datetime,
    last_error: str | None,
) -> None:
    async with session_scope() as s:
        await s.execute(
            update(DreamRun)
            .where(DreamRun.id == run_id)
            .values(
                status=status,
                ended_at=ended_at,
                summary=summary,
                last_error=last_error,
            )
        )


# ---------------------------------------------------------------------------
# Per-mode dispatch
# ---------------------------------------------------------------------------


async def _dispatch_pass(
    *,
    mode: DreamMode,
    env_id: UUID,
    actor_ctx: AgentContext,
    summarizer: DreamSummarizer,
    embedder: Embedder | None,
    vector_store: VectorStore | None,
    settings: Settings,
    now: dt.datetime,
    dream_run_id: UUID,
) -> dict[str, Any]:
    """Call the right pass and return its result-as-dict.

    Each pass returns a frozen dataclass; we serialize via ``asdict``
    and stringify UUIDs / replace enum-likes so the ``summary`` JSONB
    is fully serializable.
    """
    if mode is DreamMode.decay:
        result = await run_decay(
            env_id,
            actor_ctx=actor_ctx,
            settings=settings,
            now=now,
        )
    elif mode is DreamMode.dedupe:
        if vector_store is None or embedder is None:
            raise RuntimeError("dedupe pass requires vector_store + embedder")
        result = await run_dedupe(
            env_id,
            qdrant=vector_store,
            embedder=embedder,
            summarizer=summarizer,
            settings=settings,
            now=now,
            dream_run_id=dream_run_id,
        )
    elif mode is DreamMode.promote:
        result = await run_promote(
            env_id,
            summarizer=summarizer,
            settings=settings,
            now=now,
            dream_run_id=dream_run_id,
        )
    elif mode is DreamMode.decision_conflicts:
        if vector_store is None:
            raise RuntimeError("decision_conflicts pass requires vector_store")
        result = await run_decision_conflict_pass(
            env_id,
            actor_ctx=actor_ctx,
            qdrant=vector_store,
            threshold=settings.decision_conflict_cosine_threshold,
            dream_run_id=dream_run_id,
        )
    elif mode is DreamMode.recount:
        # Phase 1 (v0.14): reconcile reference counters against canonical
        # edge tables + playbook macro scan. No external resources beyond
        # Postgres — vector_store / embedder / summarizer all unused.
        result = await run_recount(
            env_id,
            actor_ctx=actor_ctx,
            settings=settings,
            now=now,
        )
    else:
        raise RuntimeError(f"unknown dream mode: {mode}")

    return _result_to_dict(result)


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a pass result dataclass to a JSON-serializable dict."""
    raw = asdict(result)
    return {k: _coerce_value(v) for k, v in raw.items()}


def _coerce_value(v: Any) -> Any:
    if isinstance(v, UUID):
        return str(v)
    if isinstance(v, dt.datetime):
        return v.isoformat()
    if isinstance(v, list):
        return [_coerce_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _coerce_value(val) for k, val in v.items()}
    return v


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run_dream_pass(
    env_id: UUID,
    mode: DreamMode,
    *,
    actor_ctx: AgentContext,
    summarizer: DreamSummarizer,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    settings: Settings | None = None,
    triggered_by: str = "scheduler",
    now: dt.datetime | None = None,
) -> DreamPassReport:
    """Run one dream pass for ``env_id`` and ``mode`` end to end.

    Acquires the per-(mode, env) advisory lock; if another replica holds
    it, returns ``DreamPassOutcome.skipped`` without writing a
    ``dream_runs`` row. Otherwise: writes ``dream_runs`` (running),
    dispatches to the pass function, updates ``dream_runs`` with the
    result, updates the heartbeat row, and releases the lock.

    Args:
        env_id: scope.
        mode: which pass to run.
        actor_ctx: identity attributed to outbox/audit events. Decay
            mutates state via ``memory_update`` and requires
            ``env_id ∈ ctx.attached_env_ids``; dedupe and promote do
            not, but the runner passes a per-env context anyway for
            uniformity.
        summarizer: pre-built summarizer (constructed once per worker
            process).
        embedder: required for dedupe; ignored otherwise.
        vector_store: required for dedupe; ignored otherwise.
        settings: caller may override for tests.
        triggered_by: ``"scheduler"`` for cadence ticks; ``"manual"``
            for ``dream_run`` MCP tool calls; ``"test"`` for tests.
        now: caller may override for tests.

    Returns:
        :class:`DreamPassReport` with outcome, dream_run_id, and the
        pass summary (or ``last_error`` on failure).
    """
    settings = settings or get_settings()
    now = now or dt.datetime.now(dt.UTC)
    started = time.perf_counter()

    if env_id not in actor_ctx.attached_env_ids:
        # Decay needs this; we enforce it for all modes for uniformity.
        # The runner constructs per-env contexts so this is the runner's
        # bug if it ever fires.
        raise ValueError(
            f"run_dream_pass: actor_ctx.attached_env_ids must include {env_id}"
        )

    summarizer_kind = (
        summarizer.kind.value if hasattr(summarizer, "kind") else None
    )

    # 1. Acquire advisory lock. We hold a dedicated session for the
    # WHOLE pass duration so the lock is bound to a single connection.
    # The lock is released in the same session in the ``finally`` arm.
    async with session_scope() as lock_session:
        locked = await _try_acquire_lock_in_session(
            lock_session, mode=mode, env_id=env_id,
        )
        if not locked:
            log.info(
                "dream_worker: %s/%s lock held by another worker — skipping",
                mode.value, env_id,
            )
            return DreamPassReport(
                env_id=env_id,
                mode=mode,
                outcome=DreamPassOutcome.skipped,
                duration_seconds=time.perf_counter() - started,
            )

        # 2. Insert dream_runs row.
        dream_run_id = await _insert_dream_run(
            env_id=env_id,
            mode=mode,
            triggered_by=triggered_by,
            summarizer_kind=summarizer_kind,
            started_at=now,
        )

        summary: dict[str, Any] = {}
        last_error: str | None = None
        outcome = DreamPassOutcome.done
        try:
            # 3. Dispatch.
            summary = await _dispatch_pass(
                mode=mode,
                env_id=env_id,
                actor_ctx=actor_ctx,
                summarizer=summarizer,
                embedder=embedder,
                vector_store=vector_store,
                settings=settings,
                now=now,
                dream_run_id=dream_run_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception(
                "dream_worker: %s/%s pass failed", mode.value, env_id,
            )
            outcome = DreamPassOutcome.failed
            last_error = f"{type(exc).__name__}: {exc}"

        # 4. Finalize dream_runs.
        await _finalize_dream_run(
            run_id=dream_run_id,
            status=outcome.value,
            summary=summary,
            ended_at=dt.datetime.now(dt.UTC),
            last_error=last_error,
        )

        # 5. Heartbeat.
        await update_heartbeat(
            env_id=env_id,
            mode=mode,
            success=outcome is DreamPassOutcome.done,
            now=now,
            last_error=last_error,
        )

        # 6. Release lock (still in same session).
        await _release_lock_in_session(
            lock_session, mode=mode, env_id=env_id,
        )

    duration = time.perf_counter() - started
    try:
        from memory_mcp.observability import (
            dream_pass_items_processed_total,
            dream_run_duration_seconds,
            dream_runs_total,
        )
        dream_run_duration_seconds.labels(mode=mode.value).observe(duration)
        dream_runs_total.labels(mode=mode.value, outcome=outcome.value).inc()
        # Pull "items_processed" out of pass summaries when available.
        # Pass authors record `inspected`/`processed` shapes; fall back
        # to 0 when the field is absent (e.g. failed pass).
        items = summary.get("items_processed") if isinstance(summary, dict) else 0
        if isinstance(items, int) and items:
            dream_pass_items_processed_total.labels(
                mode=mode.value,
            ).inc(items)
    except Exception:  # noqa: BLE001 — observability must never poison the pass
        log.exception("dream_worker: failed to record metrics")

    return DreamPassReport(
        env_id=env_id,
        mode=mode,
        outcome=outcome,
        dream_run_id=dream_run_id,
        summary=summary,
        last_error=last_error,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Env discovery
# ---------------------------------------------------------------------------


async def list_active_envs() -> list[UUID]:
    """Return the set of env IDs the worker should iterate this tick.

    v1 returns ALL envs; envs created mid-tick are picked up next tick.
    A future iteration could filter to envs with non-zero memory rows
    (cheap activity probe) so empty envs don't waste cycles.
    """
    async with session_scope() as s:
        rows = (await s.execute(select(Environment.id))).all()
    return [r[0] for r in rows]


async def refresh_proposals_open_gauge() -> None:
    """Refresh the ``mcp_dream_proposals_open`` gauge from canonical SQL.

    Counts open ``dream_proposals`` grouped by ``(kind, summarizer_kind)``
    and sets one gauge sample per group. Called periodically by the
    dream-worker scheduler (cadence: ``dream_metrics_refresh_seconds``).

    Observability errors never escape — the gauge can lag a tick if
    Prometheus or the DB hiccups, but the dream-worker keeps running.
    """
    try:
        from sqlalchemy import func

        from memory_mcp.db.models import DreamProposal
        from memory_mcp.observability import dream_proposals_open

        async with session_scope() as s:
            stmt = (
                select(
                    DreamProposal.kind,
                    DreamProposal.summarizer_kind,
                    func.count(DreamProposal.id),
                )
                .where(DreamProposal.status == "open")
                .group_by(DreamProposal.kind, DreamProposal.summarizer_kind)
            )
            rows = (await s.execute(stmt)).all()

        # Reset all label combinations we know about so labels removed
        # by reviewer activity don't keep stale values. The simplest
        # correct approach: clear and re-set. ``Gauge._metrics`` is
        # internal but stable across prometheus_client releases; if
        # this becomes brittle, switch to per-(kind, summarizer_kind)
        # tracked-set bookkeeping.
        with contextlib.suppress(Exception):
            dream_proposals_open.clear()

        for kind, summarizer_kind, count in rows:
            dream_proposals_open.labels(
                kind=kind,
                summarizer_kind=summarizer_kind or "unknown",
            ).set(count)
    except Exception:  # noqa: BLE001
        log.exception("refresh_proposals_open_gauge failed")


__all__ = [
    "DreamMode",
    "DreamPassOutcome",
    "DreamPassReport",
    "HEARTBEAT_SINK",
    "list_active_envs",
    "refresh_proposals_open_gauge",
    "run_dream_pass",
    "update_heartbeat",
]
