"""Decay pass — ``active → stale → archived`` lifecycle transitions.

The decay pass is *structural only*: it never invokes the summarizer
(unlike dedupe and promote) and never produces proposals. Its job is to
walk memories down the lifecycle ladder when they've gone cold:

* **Leg 1** — for each ``active`` memory in the env that hasn't been
  accessed in ``DREAM_DECAY_INACTIVE_DAYS`` days, recompute salience
  using the same formula as the on-read access path (so the result is
  consistent across reads and decay). If salience drops below
  ``DREAM_DECAY_STALE_THRESHOLD``, transition the row to ``stale``.
* **Leg 2** — for each ``stale`` memory in the env, recompute salience.
  If it drops below ``DREAM_DECAY_ARCHIVE_THRESHOLD`` (tighter), transition
  to ``archived``.

Skipped on both legs: ``pinned`` (never auto-archived), ``retired`` and
``superseded`` (terminal states). All transitions go through
:func:`memory_mcp.memories.memory_update` so the standard outbox events
fire — the projection workers don't need to know about the dream worker
at all.

Idempotency
-----------

Re-running the decay pass over an *unchanged* dataset converges:

* Active memories accessed inside the window are filtered out by SQL
  (no salience recompute, no UPDATE).
* Active memories below threshold transition to stale once; the next
  run sees them in the stale leg.
* Stale memories above the archive threshold stay stale (the row is
  unchanged across runs, so no outbox spam).
* Concurrent agent reads can race with our UPDATE — the optimistic-lock
  ``expected_version`` check inside :func:`memory_update` raises
  :class:`VersionConflictError`, which we treat as "the row was just
  touched; not stale anymore" and silently skip. The next decay tick
  will re-evaluate.

Per-pass cap
------------

``DREAM_DECAY_BATCH_CAP`` bounds how many rows each leg processes per
run. Operators tune this to bound wall-clock time and outbox pressure
on environments with very large memory tables. Hitting the cap is
tracked on the result (:attr:`DecayPassResult.items_capped`) so
observability can flag environments that aren't keeping up.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, or_, select

from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import Memory
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import MemoryStatus
from memory_mcp.dream.salience import (
    SalienceInputs,
    SalienceWeights,
    compute_salience,
    salience_weights_from_settings,
)
from memory_mcp.errors import VersionConflictError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryUpdatePatch, memory_update

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecayCandidateRow:
    """Subset of memory fields the decay pass needs.

    Decoupled from :class:`memory_mcp.db.models.Memory` so the loader
    can be mocked in unit tests without an ORM instance. The
    :class:`SalienceInputs` projection is identical for both legs.
    """

    id: UUID
    version: int
    status: MemoryStatus
    salience_inputs: SalienceInputs
    reference_count: int = 0


@dataclass(frozen=True)
class DecayPassResult:
    """Per-pass observability payload.

    Surfaced in ``dream_status`` and the ``dream_run_*`` Prometheus
    metrics added in :mod:`memory_mcp.dream.passes` (``p2.2-observability``).
    """

    env_id: UUID
    examined_active: int = 0
    examined_stale: int = 0
    transitioned_to_stale: int = 0
    transitioned_to_archived: int = 0
    skipped_version_conflicts: int = 0
    skipped_above_threshold: int = 0
    # Phase 1 (v0.14): active rows held back from staling because their
    # graph-citation count meets ``dream_decay_reference_floor``. Distinct
    # from ``skipped_above_threshold`` so the two reasons remain
    # observable individually.
    skipped_reference_floor: int = 0
    items_capped_active_leg: bool = False
    items_capped_stale_leg: bool = False
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Row loaders — extracted so unit tests can mock the SELECT machinery.
# ---------------------------------------------------------------------------


async def _load_active_candidates(
    *,
    env_id: UUID,
    cutoff: dt.datetime,
    cap: int,
) -> list[DecayCandidateRow]:
    """Load active rows in env that haven't been accessed since ``cutoff``.

    "Never accessed" rows (``last_accessed_at IS NULL``) are included
    only if they were also created before the cutoff, so brand-new
    memories created inside the inactivity window aren't immediately
    eligible for staling on the first scan.
    """

    async with session_scope() as s:
        stmt = (
            select(
                Memory.id,
                Memory.version,
                Memory.status,
                Memory.access_count,
                Memory.last_accessed_at,
                Memory.confidence,
                Memory.pinned,
                Memory.negative_feedback_count,
                Memory.verified_at,
                Memory.created_at,
                Memory.reference_count,
                Memory.reference_count_rel_link,
                Memory.reference_count_lineage,
                Memory.reference_count_task,
                Memory.reference_count_playbook,
                Memory.reference_authority,
            )
            .where(
                and_(
                    Memory.env_id == env_id,
                    Memory.status == MemoryStatus.active.value,
                    Memory.pinned.is_(False),
                    or_(
                        Memory.last_accessed_at < cutoff,
                        and_(
                            Memory.last_accessed_at.is_(None),
                            Memory.created_at < cutoff,
                        ),
                    ),
                )
            )
            .order_by(Memory.last_accessed_at.asc().nullsfirst())
            .limit(cap)
        )
        rows = (await s.execute(stmt)).all()

    return [_row_to_candidate(r) for r in rows]


async def _load_stale_candidates(
    *,
    env_id: UUID,
    cap: int,
) -> list[DecayCandidateRow]:
    """Load stale rows in env (any age — archive threshold is the gate)."""

    async with session_scope() as s:
        stmt = (
            select(
                Memory.id,
                Memory.version,
                Memory.status,
                Memory.access_count,
                Memory.last_accessed_at,
                Memory.confidence,
                Memory.pinned,
                Memory.negative_feedback_count,
                Memory.verified_at,
                Memory.created_at,
                Memory.reference_count,
                Memory.reference_count_rel_link,
                Memory.reference_count_lineage,
                Memory.reference_count_task,
                Memory.reference_count_playbook,
                Memory.reference_authority,
            )
            .where(
                and_(
                    Memory.env_id == env_id,
                    Memory.status == MemoryStatus.stale.value,
                    Memory.pinned.is_(False),
                )
            )
            .order_by(Memory.updated_at.asc())
            .limit(cap)
        )
        rows = (await s.execute(stmt)).all()

    return [_row_to_candidate(r) for r in rows]


def _row_to_candidate(row: object) -> DecayCandidateRow:
    """Project a SQLAlchemy ``Row`` into the decay-pass DTO.

    Extracted as a helper so tests can build candidate rows directly
    without going through SQLAlchemy.
    """

    return DecayCandidateRow(
        id=row[0],
        version=row[1],
        status=MemoryStatus(row[2]),
        salience_inputs=SalienceInputs(
            access_count=row[3],
            last_accessed_at=row[4],
            confidence=float(row[5]),
            pinned=row[6],
            negative_feedback_count=row[7],
            verified_at=row[8],
            created_at=row[9],
            reference_count_rel_link=int(row[11] or 0),
            reference_count_lineage=int(row[12] or 0),
            reference_count_task=int(row[13] or 0),
            reference_count_playbook=int(row[14] or 0),
            # Phase 1e-d — decay reads ``reference_authority`` so the
            # authority term contributes to the salience used for
            # decay-threshold decisions. Decay does NOT stamp
            # ``salience_formula_version`` (recount owns that); decay-pass
            # UPDATEs go via direct SQL today and only set
            # ``salience`` / ``status``. Next recount picks the row up via
            # the formula-version mismatch path if applicable.
            reference_authority=float(row[15] or 0),
        ),
        reference_count=int(row[10] or 0),
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run_decay(
    env_id: UUID,
    *,
    actor_ctx: AgentContext,
    settings: Settings | None = None,
    now: dt.datetime | None = None,
) -> DecayPassResult:
    """Walk active → stale → archived for a single env.

    Args:
        env_id: scope. Each env runs in its own pass so a noisy env
            doesn't slow down a quiet one.
        actor_ctx: identity attributed to the resulting outbox /
            audit-log events. The runner constructs this once at startup
            (typically the server-default agent or a dedicated
            "dream-worker" agent) and passes it into every pass call.
            ``ctx.attached_env_ids`` MUST include ``env_id`` so the
            internal :func:`memory_update` calls pass env-visibility
            checks.
        settings: caller may override for tests; otherwise resolved via
            :func:`get_settings`.
        now: caller may override for tests; otherwise wall clock UTC.

    Idempotent: safe to re-run on the same env without coordinating with
    other ticks. The advisory lock acquired by the runner (``p2.2-runner``)
    prevents two replicas from running the *same* env in parallel; this
    function does not assume the lock and would still converge if called
    re-entrantly.
    """

    settings = settings or get_settings()
    now = now or dt.datetime.now(dt.UTC)
    weights = salience_weights_from_settings(settings)
    inactive_cutoff = now - dt.timedelta(days=settings.dream_decay_inactive_days)
    cap = settings.dream_decay_batch_cap

    if env_id not in actor_ctx.attached_env_ids:
        raise ValueError(
            "run_decay: actor_ctx.attached_env_ids must include env_id "
            f"({env_id}) for memory_update env-visibility checks to pass"
        )

    started = time.perf_counter()

    result_active = await _run_leg(
        env_id=env_id,
        actor_ctx=actor_ctx,
        settings=settings,
        now=now,
        weights=weights,
        candidates=await _load_active_candidates(
            env_id=env_id,
            cutoff=inactive_cutoff,
            cap=cap,
        ),
        cap=cap,
        target_status=MemoryStatus.stale,
        threshold=settings.dream_decay_stale_threshold,
        reference_count_floor=settings.dream_decay_reference_floor,
    )
    result_stale = await _run_leg(
        env_id=env_id,
        actor_ctx=actor_ctx,
        settings=settings,
        now=now,
        weights=weights,
        candidates=await _load_stale_candidates(env_id=env_id, cap=cap),
        cap=cap,
        target_status=MemoryStatus.archived,
        threshold=settings.dream_decay_archive_threshold,
        # Stale → archived is not gated by reference floor — once a row is
        # stale, decay to archived is the natural lifecycle. Surface the
        # gate only on the active → stale leg.
        reference_count_floor=0,
    )

    return DecayPassResult(
        env_id=env_id,
        examined_active=result_active["examined"],
        examined_stale=result_stale["examined"],
        transitioned_to_stale=result_active["transitioned"],
        transitioned_to_archived=result_stale["transitioned"],
        skipped_version_conflicts=(result_active["version_conflicts"] + result_stale["version_conflicts"]),
        skipped_above_threshold=(result_active["above_threshold"] + result_stale["above_threshold"]),
        skipped_reference_floor=result_active["reference_floor"],
        items_capped_active_leg=result_active["capped"],
        items_capped_stale_leg=result_stale["capped"],
        duration_seconds=time.perf_counter() - started,
    )


async def _run_leg(  # noqa: PLR0913 — explicit args document the contract
    *,
    env_id: UUID,
    actor_ctx: AgentContext,
    settings: Settings,
    now: dt.datetime,
    weights: SalienceWeights,
    candidates: list[DecayCandidateRow],
    cap: int,
    target_status: MemoryStatus,
    threshold: float,
    reference_count_floor: int = 0,
) -> dict[str, int | bool]:
    """Generic decay leg — recompute salience, transition if below threshold.

    The optional ``reference_count_floor`` adds a structural-popularity
    gate (Phase 1 v0.14): when ``reference_count_floor > 0`` and a
    candidate has ``reference_count >= reference_count_floor``, the
    transition is skipped (counted in ``reference_floor``) regardless of
    salience. This protects highly-cited memories from being archived
    just because nobody read them recently — graph-citation is a
    structural use signal even when access is dormant.
    """

    examined = 0
    transitioned = 0
    above_threshold = 0
    version_conflicts = 0
    reference_floor_skipped = 0

    for cand in candidates:
        examined += 1
        if reference_count_floor > 0 and cand.reference_count >= reference_count_floor:
            reference_floor_skipped += 1
            continue
        salience = compute_salience(
            cand.salience_inputs,
            now=now,
            weights=weights,
        )
        if salience >= threshold:
            above_threshold += 1
            continue
        try:
            await memory_update(
                cand.id,
                MemoryUpdatePatch(
                    expected_version=cand.version,
                    status=target_status,
                ),
                ctx=actor_ctx,
                settings=settings,
            )
        except VersionConflictError:
            version_conflicts += 1
            log.debug(
                "decay: version conflict on memory %s (env %s); skipping",
                cand.id,
                env_id,
            )
            continue
        transitioned += 1

    return {
        "examined": examined,
        "transitioned": transitioned,
        "above_threshold": above_threshold,
        "version_conflicts": version_conflicts,
        "reference_floor": reference_floor_skipped,
        "capped": len(candidates) >= cap,
    }


__all__ = [
    "DecayCandidateRow",
    "DecayPassResult",
    "run_decay",
]
