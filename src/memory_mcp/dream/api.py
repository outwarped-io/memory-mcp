"""MCP tool surface for dream-mode operations.

Exposes four tools:

* :func:`dream_run` — kick a pass on demand. With ``wait=True`` blocks
  until the pass completes; otherwise spawns the work as a background
  task tracked in a process-local registry so shutdown can drain
  in-flight passes.
* :func:`dream_status` — list recent dream_runs, open-proposal counts,
  scheduler heartbeats, summarizer kind, and bounded LLM probe.
* :func:`dream_proposals_list` — paginated, filterable browse of the
  ``dream_proposals`` table. Cursor uses keyset ``(created_at, id)``
  for stable ordering across batched inserts.
* :func:`dream_review` — terminal action on a proposal: ``accept``
  (dispatches to the merge or promotion accept handler atomically),
  ``reject``, ``defer``. ``amend`` is reserved for v1.5.

Concurrency safety
------------------

* The dream-worker holds per-(env, mode) advisory locks during pass
  execution; ``dream_run`` calls go through the same lock so manual
  triggers cannot race the scheduler.
* ``dream_review`` acquires a row-level lock on the proposal
  (``SELECT FOR UPDATE``) before doing any work, then locks all
  involved memory rows in **deterministic UUID order** to avoid
  deadlocks when overlapping merge proposals are accepted concurrently.
* All accept-path mutations (merge fan-in, promotion fan-out) commit
  in a single transaction with their lineage, audit, and outbox events.

V1 limitations (documented in payload / error messages)
-------------------------------------------------------

* ``amend`` action is not implemented; accept/reject/defer only.
* Merge accept does NOT auto-rewire entity relations from old memories
  to the new merged memory. Search hits via ``follow_superseded`` work,
  but graph-leg traversal degrades for the merged content. Tracked for
  v1.5.
* Open overlapping proposals are NOT auto-expired when one is accepted;
  the second accept will surface ``INVALID_TRANSITION`` because the
  candidates are already superseded. Reviewers must `reject` overlaps
  themselves. Tracked for v1.5.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from collections.abc import Iterable
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dream_worker.jobs import (
    DreamMode,
    DreamPassOutcome,
    DreamPassReport,
    list_active_envs,
    run_dream_pass,
)
from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import (
    DreamProposal,
    DreamRun,
    Memory,
    MemoryLineage,
    MemorySource,
    ProjectionState,
)
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import (
    LineageRelation,
    MemorySourceType,
    MemoryStatus,
    OutboxAggregateType,
)
from memory_mcp.db.vector.qdrant import QdrantVectorStore
from memory_mcp.dream.summarizer import build_summarizer
from memory_mcp.embeddings.base import get_embedder
from memory_mcp.errors import (
    InvalidInputError,
    InvalidTransitionError,
    NotFoundError,
    VersionConflictError,
)
from memory_mcp.identity import AgentContext
from memory_mcp.memories import (
    MemoryResponse,
    _audit_snapshot,
    _load_env_embedding_model,
    _load_tag_names,
    _outbox_op_for,
    _projection_payload,
    _record_audit,
    _to_response,
)

from memory_mcp_schemas.dream import (
    DreamHeartbeatEntry,
    DreamProposalEntry,
    DreamProposalsListRequest,
    DreamProposalsListResponse,
    DreamReviewPatch,
    DreamReviewRequest,
    DreamReviewResponse,
    DreamRunReport,
    DreamRunRequest,
    DreamRunResponse,
    DreamRunScheduledItem,
    DreamRunSummaryEntry,
    DreamStatusRequest,
    DreamStatusResponse,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# `dream_run` — manual trigger (and tests)
# ---------------------------------------------------------------------------


# Module-level registry so background coordinators (wait=False) survive
# the request that spawned them; FastAPI does not track create_task results.
# Tasks are removed via add_done_callback on completion.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _track_background_task(task: asyncio.Task[Any]) -> None:
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    task.add_done_callback(_log_failed_task)


def _log_failed_task(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.exception(
            "dream_run background coordinator raised", exc_info=exc,
        )


def get_active_background_tasks() -> set[asyncio.Task[Any]]:
    """Test/observability helper — exposes the in-flight registry."""
    return set(_BACKGROUND_TASKS)


_ALL_MODES: tuple[DreamMode, ...] = (
    DreamMode.decay,
    DreamMode.dedupe,
    DreamMode.promote,
    DreamMode.decision_conflicts,
    DreamMode.recount,
)


async def dream_run(
    req: DreamRunRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> DreamRunResponse:
    """Trigger one or more dream passes synchronously or in the background.

    Resolves the env list from ``req.env_id`` (single env) or from
    :func:`list_active_envs` (all envs). Modes default to all manual modes.

    With ``wait=True``, awaits all passes inline; resources are
    constructed and torn down per call. With ``wait=False`` (default),
    spawns a coordinator that owns the resources and discards results;
    the response lists which (env, mode) pairs were scheduled.
    """
    settings = settings or get_settings()

    env_ids = await _resolve_env_ids_for_dream_run(req.env_id, ctx)
    if not env_ids:
        return DreamRunResponse(scheduled=[], reports=[])
    for env_id in env_ids:
        rbac.require("write", env_id, ctx)

    modes: list[DreamMode] = list(req.modes) if req.modes else list(_ALL_MODES)

    pairs: list[tuple[UUID, DreamMode]] = [
        (env_id, mode) for env_id in env_ids for mode in modes
    ]

    if req.wait:
        reports = await _run_pairs_with_resources(
            pairs, settings=settings, agent_id=ctx.agent_id,
            agent_name=ctx.agent_name, triggered_by=req.triggered_by,
        )
        return DreamRunResponse(
            scheduled=[],
            reports=[DreamRunReport(**_dream_pass_report_to_dict(r)) for r in reports],
        )

    # wait=False — fire and forget.
    coordinator = asyncio.create_task(
        _run_pairs_with_resources(
            pairs, settings=settings, agent_id=ctx.agent_id,
            agent_name=ctx.agent_name, triggered_by=req.triggered_by,
        ),
        name=f"dream_run-coordinator-{dt.datetime.now(dt.UTC).isoformat()}",
    )
    _track_background_task(coordinator)
    return DreamRunResponse(
        scheduled=[
            DreamRunScheduledItem(env_id=env_id, mode=mode)
            for env_id, mode in pairs
        ],
        reports=[],
    )


async def _resolve_env_ids_for_dream_run(
    explicit_env_id: UUID | None,
    ctx: AgentContext,
) -> list[UUID]:
    """Pick the env list. Single env if given; else all attached; else all."""
    if explicit_env_id is not None:
        return [explicit_env_id]
    if ctx.attached_env_ids:
        return list(dict.fromkeys(ctx.attached_env_ids))
    # No envs attached — fall back to all envs the worker would iterate.
    return await list_active_envs()


async def _run_pairs_with_resources(
    pairs: Iterable[tuple[UUID, DreamMode]],
    *,
    settings: Settings,
    agent_id: UUID,
    agent_name: str | None,
    triggered_by: str,
) -> list[DreamPassReport]:
    """Build per-call resources, fan out to ``run_dream_pass``, close in finally.

    Resources (vector store, summarizer-owned LLM client) live for the
    duration of the call so concurrent passes share connection pools.
    Per-pair errors are caught and surfaced as failed reports rather
    than raised, so one bad env doesn't poison the whole batch.
    """
    summarizer = build_summarizer(settings)
    embedder = get_embedder(settings)
    vector_store = QdrantVectorStore(settings)

    reports: list[DreamPassReport] = []
    try:
        for env_id, mode in pairs:
            ctx = AgentContext(
                agent_id=agent_id,
                agent_name=agent_name,
                session_id=None,
                attached_env_ids=[env_id],
                is_default_agent=False,
            )
            try:
                report = await run_dream_pass(
                    env_id, mode,
                    actor_ctx=ctx,
                    summarizer=summarizer,
                    embedder=embedder if mode is DreamMode.dedupe else None,
                    vector_store=(
                        vector_store
                        if mode in {DreamMode.dedupe, DreamMode.decision_conflicts}
                        else None
                    ),
                    settings=settings,
                    triggered_by=triggered_by,
                )
            except Exception as exc:  # noqa: BLE001 — per-pair isolation
                log.exception(
                    "dream_run: env=%s mode=%s raised", env_id, mode.value,
                )
                report = DreamPassReport(
                    env_id=env_id,
                    mode=mode,
                    outcome=DreamPassOutcome.failed,
                    last_error=f"{type(exc).__name__}: {exc}",
                )
            reports.append(report)
    finally:
        try:
            await vector_store.close()
        except Exception:  # noqa: BLE001
            log.exception("dream_run: vector_store.close raised")
    return reports


def _dream_pass_report_to_dict(report: DreamPassReport) -> dict[str, Any]:
    """Coerce :class:`DreamPassReport` to a Pydantic-friendly dict."""
    summary = report.summary
    if summary and not isinstance(summary, dict):
        summary = {"_raw": str(summary)}
    return {
        "env_id": report.env_id,
        "mode": report.mode,
        "outcome": report.outcome,
        "dream_run_id": report.dream_run_id,
        "summary": summary or {},
        "last_error": report.last_error,
        "duration_seconds": report.duration_seconds,
    }


# ---------------------------------------------------------------------------
# `dream_status`
# ---------------------------------------------------------------------------


async def dream_status(
    req: DreamStatusRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> DreamStatusResponse:
    """Aggregate dream-worker state for an env (or all envs)."""
    settings = settings or get_settings()
    if req.env_id is not None:
        rbac.require("read", req.env_id, ctx)

    async with session_scope() as s:
        last_runs = await _load_last_runs_per_mode(
            s, env_id=req.env_id, runs_per_mode=req.runs_per_mode,
        )
        open_counts = await _load_open_proposal_counts(s, env_id=req.env_id)
        heartbeats = await _load_dream_heartbeats(s, env_id=req.env_id)

    llm_status = await _bounded_llm_probe(settings)

    return DreamStatusResponse(
        last_runs=last_runs,
        open_proposal_counts=open_counts,
        summarizer_kind=settings.dream_summarizer,
        llm_backend=settings.llm_backend,
        llm_status=llm_status,
        heartbeats=heartbeats,
    )


async def _load_last_runs_per_mode(
    s: AsyncSession,
    *,
    env_id: UUID | None,
    runs_per_mode: int,
) -> list[DreamRunSummaryEntry]:
    """Return up to ``runs_per_mode`` most recent runs per mode."""
    out: list[DreamRunSummaryEntry] = []
    for mode in _ALL_MODES:
        stmt = (
            select(DreamRun)
            .where(DreamRun.mode == mode.value)
            .order_by(DreamRun.started_at.desc())
            .limit(runs_per_mode)
        )
        if env_id is not None:
            stmt = stmt.where(DreamRun.env_id == env_id)
        rows = (await s.execute(stmt)).scalars().all()
        for r in rows:
            out.append(_dream_run_to_entry(r))
    return out


async def _load_open_proposal_counts(
    s: AsyncSession,
    *,
    env_id: UUID | None,
) -> dict[str, int]:
    stmt = (
        select(DreamProposal.kind, func.count())
        .where(DreamProposal.status == "open")
        .group_by(DreamProposal.kind)
    )
    if env_id is not None:
        stmt = stmt.where(DreamProposal.env_id == env_id)
    rows = (await s.execute(stmt)).all()
    counts = {
        "merge_candidate": 0,
        "promotion_candidate": 0,
        "decay_candidate": 0,
        "decision_conflict_candidate": 0,
    }
    for kind, n in rows:
        counts[kind] = int(n)
    return counts


async def _load_dream_heartbeats(
    s: AsyncSession,
    *,
    env_id: UUID | None,
) -> list[DreamHeartbeatEntry]:
    stmt = select(ProjectionState).where(ProjectionState.sink.like("dream_worker:%"))
    if env_id is not None:
        stmt = stmt.where(ProjectionState.env_id == env_id)
    rows = (await s.execute(stmt)).scalars().all()
    return [
        DreamHeartbeatEntry(
            sink=r.sink,
            env_id=r.env_id,
            last_success_at=r.last_success_at,
            lag_seconds=float(r.lag_seconds) if r.lag_seconds is not None else None,
            status=r.status,
            last_error=r.last_error,
        )
        for r in rows
    ]


async def _bounded_llm_probe(settings: Settings) -> dict[str, Any]:
    """Best-effort ``probe_llm`` with a 2-second cap.

    The dream-worker's ``/readyz`` already runs the same probe; we
    duplicate it here so ``dream_status`` is self-contained for clients
    that don't have access to ``/readyz`` (MCP-only consumers).
    """
    from memory_mcp.llm.base import probe_llm  # lazy import — see llm/base.py

    try:
        return await asyncio.wait_for(probe_llm(settings), timeout=2.0)
    except TimeoutError:
        return {
            "status": "error",
            "backend": settings.llm_backend,
            "error": "probe timed out after 2.0s",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "backend": settings.llm_backend,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# `dream_proposals_list`
# ---------------------------------------------------------------------------


async def dream_proposals_list(
    req: DreamProposalsListRequest,
    *,
    ctx: AgentContext,
) -> DreamProposalsListResponse:
    """Keyset-paginated browse over ``dream_proposals``.

    Order: ``created_at DESC, id DESC``. Cursor is an opaque token
    encoding the last seen ``(created_at, id)`` and the filter
    fingerprint — reusing the cursor across different filters yields
    ``INVALID_INPUT``.
    """
    if req.env_id is not None:
        rbac.require("read", req.env_id, ctx)

    cursor_state = _decode_cursor(req.cursor) if req.cursor else None
    if (
        cursor_state is not None
        and cursor_state["filters_hash"] != _filters_hash(req)
    ):
        raise InvalidInputError(
            "cursor was issued for a different filter set; "
            "drop the cursor and re-issue the query",
        )

    stmt = select(DreamProposal)
    if req.env_id is not None:
        stmt = stmt.where(DreamProposal.env_id == req.env_id)
    if req.status is not None:
        stmt = stmt.where(DreamProposal.status == req.status)
    if req.kind is not None:
        stmt = stmt.where(DreamProposal.kind == req.kind)
    if req.summarizer_kind is not None:
        stmt = stmt.where(DreamProposal.summarizer_kind == req.summarizer_kind)

    if cursor_state is not None:
        last_created_at = dt.datetime.fromisoformat(cursor_state["created_at"])
        last_id = UUID(cursor_state["id"])
        stmt = stmt.where(
            _keyset_lt(DreamProposal.created_at, DreamProposal.id, last_created_at, last_id)
        )

    stmt = stmt.order_by(
        DreamProposal.created_at.desc(), DreamProposal.id.desc()
    ).limit(req.limit + 1)

    async with session_scope() as s:
        rows = (await s.execute(stmt)).scalars().all()

    has_more = len(rows) > req.limit
    rows = rows[: req.limit]
    items = [_dream_proposal_to_entry(r) for r in rows]
    next_cursor: str | None = None
    if has_more and items:
        last = rows[-1]
        next_cursor = _encode_cursor(
            {
                "created_at": last.created_at.isoformat(),
                "id": str(last.id),
                "filters_hash": _filters_hash(req),
            },
        )

    return DreamProposalsListResponse(items=items, next_cursor=next_cursor)


def _keyset_lt(created_at_col, id_col, last_created_at, last_id):  # noqa: ANN001
    """``(created_at, id) < (a, b)`` keyset filter for DESC ordering.

    The Postgres tuple comparison ``(a, b) < (c, d)`` is equivalent to:

    .. code-block:: sql

        a < c OR (a = c AND b < d)
    """
    return (created_at_col < last_created_at) | (
        and_(created_at_col == last_created_at, id_col < last_id)
    )


def _filters_hash(req: DreamProposalsListRequest) -> str:
    """Stable fingerprint of cursor-relevant filters."""
    parts = [
        str(req.env_id) if req.env_id is not None else "",
        req.status or "",
        req.kind or "",
        req.summarizer_kind or "",
    ]
    return "|".join(parts)


def _encode_cursor(state: dict[str, str]) -> str:
    return json.dumps(state, separators=(",", ":"))


def _decode_cursor(cursor: str) -> dict[str, str]:
    try:
        out = json.loads(cursor)
    except (json.JSONDecodeError, ValueError) as exc:
        raise InvalidInputError(f"malformed cursor: {exc}") from exc
    if not isinstance(out, dict) or {"created_at", "id", "filters_hash"} - set(out):
        raise InvalidInputError("cursor missing required fields")
    return out


# ---------------------------------------------------------------------------
# `dream_review` — accept / reject / defer / amend
# ---------------------------------------------------------------------------


async def dream_review(
    req: DreamReviewRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> DreamReviewResponse:
    """Apply a terminal review action to an open proposal.

    Locks the proposal row first (``SELECT FOR UPDATE``), then dispatches
    by ``(action, kind)``. All accept-path mutations commit in a single
    transaction with the proposal status update.
    """
    settings = settings or get_settings()

    if req.action == "amend":
        raise InvalidInputError(
            "dream_review action='amend' is not implemented in v1; "
            "use 'reject' and re-run dream_run to regenerate the proposal",
        )

    async with session_scope() as s:
        proposal = await _lock_proposal(s, req.proposal_id)
        rbac.require("write", proposal.env_id, ctx)

        if proposal.status != "open":
            raise InvalidTransitionError(
                src=f"proposal.status={proposal.status}",
                dst=req.action,
            )

        accepted_memory: Memory | None = None
        accepted_tag_names: list[str] = []
        superseded_ids: list[UUID] = []

        if req.action == "accept":
            if proposal.kind == "merge_candidate":
                accepted_memory, accepted_tag_names, superseded_ids = (
                    await _accept_merge(
                        s,
                        proposal=proposal,
                        ctx=ctx,
                        patch=req.patch,
                        expected_versions=req.expected_versions or {},
                        settings=settings,
                    )
                )
            elif proposal.kind == "promotion_candidate":
                accepted_memory, accepted_tag_names = await _accept_promotion(
                    s,
                    proposal=proposal,
                    ctx=ctx,
                    patch=req.patch,
                    expected_versions=req.expected_versions or {},
                    settings=settings,
                )
            elif proposal.kind == "decay_candidate":
                # Decay candidate accept is a no-op (decay is structural).
                # We just mark the proposal accepted with a note.
                pass
            elif proposal.kind == "decision_conflict_candidate":
                # Conflict candidates are informational reviewer prompts.
                pass
            else:
                raise InvalidInputError(
                    f"unknown proposal kind: {proposal.kind!r}",
                )

        await _finalize_proposal_status(
            s,
            proposal=proposal,
            action=req.action,
            notes=req.notes,
            agent_id=ctx.agent_id,
            accepted_memory_id=(
                accepted_memory.id if accepted_memory is not None else None
            ),
            superseded_ids=superseded_ids,
        )

        await s.refresh(proposal)
        accepted_response: MemoryResponse | None = None
        if accepted_memory is not None:
            accepted_response = _to_response(accepted_memory, accepted_tag_names)

    return DreamReviewResponse(
        proposal=_dream_proposal_to_entry(proposal),
        accepted_memory=accepted_response,
        superseded_memory_ids=superseded_ids,
    )


async def _lock_proposal(s: AsyncSession, proposal_id: UUID) -> DreamProposal:
    """``SELECT ... FOR UPDATE`` on the proposal row.

    Required so two concurrent ``dream_review`` calls on the same
    proposal serialize — only one runs the accept handler; the second
    observes ``status != 'open'`` and raises :class:`InvalidTransitionError`.
    """
    stmt = (
        select(DreamProposal)
        .where(DreamProposal.id == proposal_id)
        .with_for_update()
    )
    proposal = (await s.execute(stmt)).scalar_one_or_none()
    if proposal is None:
        raise NotFoundError(
            f"dream_proposal {proposal_id} not found",
        )
    return proposal


async def _finalize_proposal_status(
    s: AsyncSession,
    *,
    proposal: DreamProposal,
    action: DreamReviewAction,
    notes: str | None,
    agent_id: UUID,
    accepted_memory_id: UUID | None,
    superseded_ids: list[UUID],
) -> None:
    """Write the proposal status + reviewed_* fields + payload.accepted_*."""
    new_status = {
        "accept": "accepted",
        "reject": "rejected",
        "defer": "deferred",
    }[action]
    new_payload = dict(proposal.payload or {})
    if accepted_memory_id is not None:
        new_payload["accepted_memory_id"] = str(accepted_memory_id)
    if superseded_ids:
        new_payload["superseded_memory_ids"] = [str(x) for x in superseded_ids]
    await s.execute(
        update(DreamProposal)
        .where(DreamProposal.id == proposal.id)
        .values(
            status=new_status,
            review_action=action,
            review_notes=notes,
            reviewed_at=func.now(),
            reviewed_by_agent_id=agent_id,
            payload=new_payload,
        )
    )


# ---------------------------------------------------------------------------
# Accept handler — merge_candidate (N→1)
# ---------------------------------------------------------------------------


async def _accept_merge(
    s: AsyncSession,
    *,
    proposal: DreamProposal,
    ctx: AgentContext,
    patch: DreamReviewPatch | None,
    expected_versions: dict[UUID, int],
    settings: Settings,
) -> tuple[Memory, list[str], list[UUID]]:
    """N→1 merge: insert merged memory + supersede primary + candidates.

    Steps (single transaction):

    1. Parse ``primary_id`` + ``candidate_ids`` from payload.
    2. Lock all involved memory rows in deterministic UUID order (avoids
       deadlock under overlapping merge proposals).
    3. Validate: all rows exist, all in ``active``/``stale``, all share
       the same ``env_id`` and ``kind``, none already superseded.
    4. Verify ``expected_versions`` (when provided).
    5. Insert the merged memory in ``active`` status with content from
       (caller patch) → (suggested_merged_*) → (primary's title/body)
       fallback chain. Tags = union of all source memories' tags.
    6. Mark each source memory ``superseded → merged.id`` with
       ``version+=1``; insert lineage (parent=source, child=merged,
       relation=supersedes); record audit entries.
    7. Outbox: ``upsert`` for merged; ``tombstone`` for each superseded.
    """
    payload = proposal.payload or {}
    try:
        primary_id = UUID(str(payload["primary_id"]))
        candidate_ids = [UUID(str(x)) for x in payload.get("candidate_ids", [])]
    except (KeyError, ValueError, TypeError) as exc:
        raise InvalidInputError(
            f"merge_candidate payload missing or malformed: {exc}",
        ) from exc

    if not candidate_ids:
        raise InvalidInputError(
            "merge_candidate payload has no candidate_ids",
        )
    if primary_id in candidate_ids:
        raise InvalidInputError(
            "merge_candidate payload includes primary_id in candidate_ids",
        )

    # Deterministic UUID order — avoids deadlock when overlapping merge
    # proposals are accepted concurrently.
    all_ids_sorted = sorted({primary_id, *candidate_ids})

    rows = await _lock_memories(s, all_ids_sorted)
    if len(rows) != len(all_ids_sorted):
        found = {r.id for r in rows}
        missing = [str(x) for x in all_ids_sorted if x not in found]
        raise NotFoundError(
            f"memory rows not found: {missing}",
        )

    by_id: dict[UUID, Memory] = {r.id: r for r in rows}
    primary = by_id[primary_id]
    sources: list[Memory] = [primary] + [by_id[c] for c in candidate_ids]

    # Validate envelope: same env, same kind, all transitionable.
    env_id = primary.env_id
    kind = primary.kind
    for src in sources:
        if src.env_id != env_id:
            raise InvalidTransitionError(
                src=f"env={src.env_id}",
                dst=f"env={env_id}",
            )
        if src.kind != kind:
            raise InvalidTransitionError(
                src=f"kind={src.kind}",
                dst=f"kind={kind}",
            )
        if src.status not in ("active", "stale"):
            raise InvalidTransitionError(
                src=f"{src.id}.status={src.status}",
                dst="superseded",
            )

    # Optional optimistic-lock check.
    for src in sources:
        expected = expected_versions.get(src.id)
        if expected is not None and expected != src.version:
            raise VersionConflictError(
                expected=expected, actual=src.version,
            )

    # Resolve merged content.
    title = (
        (patch.title if patch else None)
        or payload.get("suggested_merged_title")
        or primary.title
    )
    body = (
        (patch.body if patch else None)
        or payload.get("suggested_merged_body")
        or primary.body
    )

    # Union of source tags.
    tag_names_per_src = {
        src.id: await _load_tag_names(s, src.id) for src in sources
    }
    merged_tag_names = sorted(
        {n for ts in tag_names_per_src.values() for n in ts}
    )

    # Embedding model (per env) — every source already shares env_id.
    embedding_model_id = await _load_env_embedding_model(s, env_id)

    # 1. Insert merged memory.
    merged = Memory(
        env_id=env_id,
        kind=kind,
        status=MemoryStatus.active.value,
        title=title,
        body=body,
        # Carry forward primary's metadata as a baseline; reviewer can
        # always patch later via memory_update.
        metadata_=dict(primary.metadata_ or {}),
    )
    s.add(merged)
    await s.flush()
    await s.refresh(merged)

    if merged_tag_names:
        from memory_mcp.memories import _replace_memory_tags, _upsert_tags
        tag_map = await _upsert_tags(s, env_id=env_id, names=merged_tag_names)
        await _replace_memory_tags(
            s,
            memory_id=merged.id,
            env_id=env_id,
            tag_ids=[tag_map[n] for n in merged_tag_names],
        )

    # Provenance: dream-run reference.
    await s.execute(
        insert(MemorySource).values(
            memory_id=merged.id,
            source_type=MemorySourceType.dream.value,
            source_ref=str(proposal.dream_run_id) if proposal.dream_run_id else str(proposal.id),
            agent_id=ctx.agent_id,
        )
    )

    # 2. Audit + outbox for new merged.
    await _record_audit(
        s,
        op="create",
        memory=merged,
        by_agent_id=ctx.agent_id,
        before=None,
        after=_audit_snapshot(merged, tag_names=merged_tag_names),
        extra_after={"merged_from": [str(s_.id) for s_ in sources]},
    )
    await enqueue_event(
        s,
        aggregate_type=OutboxAggregateType.memory,
        aggregate_id=merged.id,
        aggregate_version=merged.version,
        env_id=env_id,
        op=_outbox_op_for(MemoryStatus.active, is_create=True),
        payload=_projection_payload(
            merged, tag_names=merged_tag_names,
            embedding_model_id=embedding_model_id,
        ),
        settings=settings,
    )

    # 3. Per-source: status=superseded, superseded_by=merged.id, version+=1.
    superseded_ids: list[UUID] = []
    for src in sources:
        old_tag_names = tag_names_per_src[src.id]
        old_before = _audit_snapshot(src, tag_names=old_tag_names)
        result = await s.execute(
            update(Memory)
            .where(
                Memory.id == src.id,
                Memory.version == src.version,
            )
            .values(
                status=MemoryStatus.superseded.value,
                superseded_by=merged.id,
                version=src.version + 1,
                updated_at=func.now(),
            )
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]
            # The FOR UPDATE lock should make this impossible, but if it
            # does happen we surface a clean conflict.
            raise VersionConflictError(
                expected=src.version,
                actual=src.version + 1,
            )
        await s.refresh(src)
        await s.execute(
            insert(MemoryLineage).values(
                parent_memory_id=src.id,
                child_memory_id=merged.id,
                relation=LineageRelation.supersedes.value,
            )
        )
        await _record_audit(
            s,
            op="supersede",
            memory=src,
            by_agent_id=ctx.agent_id,
            before=old_before,
            after=_audit_snapshot(src, tag_names=old_tag_names),
            extra_after={"superseded_by": str(merged.id)},
        )
        await enqueue_event(
            s,
            aggregate_type=OutboxAggregateType.memory,
            aggregate_id=src.id,
            aggregate_version=src.version,
            env_id=env_id,
            op=_outbox_op_for(MemoryStatus.superseded, is_create=False),
            payload=_projection_payload(
                src, tag_names=old_tag_names,
                embedding_model_id=embedding_model_id,
            ),
            settings=settings,
        )
        superseded_ids.append(src.id)

    return merged, merged_tag_names, superseded_ids


# ---------------------------------------------------------------------------
# Accept handler — promotion_candidate
# ---------------------------------------------------------------------------


async def _accept_promotion(
    s: AsyncSession,
    *,
    proposal: DreamProposal,
    ctx: AgentContext,
    patch: DreamReviewPatch | None,
    expected_versions: dict[UUID, int],
    settings: Settings,
) -> tuple[Memory, list[str]]:
    """Create a new ``active`` memory + lineage rows tying it to observations.

    Steps:

    1. Parse ``observation_ids`` + ``target_kind`` from payload.
    2. Lock the observation memories in deterministic order (read-lock
       only — observations are not mutated).
    3. Validate: all exist, share the same env, are not already retired/
       superseded.
    4. Insert the new memory with content from
       (caller patch) → (suggested_*) → fallback.
    5. Insert ``memory_lineage`` rows: parent=observation,
       child=new_memory, relation=``promoted_from``.
    6. Audit + outbox event for the new memory.
    """
    payload = proposal.payload or {}
    try:
        observation_ids = [UUID(str(x)) for x in payload.get("observation_ids", [])]
        target_kind = str(payload.get("target_kind", "fact"))
    except (KeyError, ValueError, TypeError) as exc:
        raise InvalidInputError(
            f"promotion_candidate payload malformed: {exc}",
        ) from exc

    if not observation_ids:
        raise InvalidInputError(
            "promotion_candidate payload has no observation_ids",
        )

    rows = await _lock_memories(s, sorted(set(observation_ids)))
    if len(rows) != len(set(observation_ids)):
        found = {r.id for r in rows}
        missing = [str(x) for x in set(observation_ids) if x not in found]
        raise NotFoundError(f"observation memories not found: {missing}")

    # All observations must be in the same env (already true if dedupe pass
    # produced this proposal, but verified for malformed payloads).
    env_ids = {r.env_id for r in rows}
    if len(env_ids) != 1:
        raise InvalidTransitionError(
            src=f"observations span envs={[str(e) for e in env_ids]}",
            dst="promote",
        )
    env_id = next(iter(env_ids))
    if proposal.env_id != env_id:
        raise InvalidTransitionError(
            src=f"proposal.env={proposal.env_id}",
            dst=f"observations.env={env_id}",
        )

    for r in rows:
        if r.status in ("retired", "superseded"):
            raise InvalidTransitionError(
                src=f"{r.id}.status={r.status}",
                dst="promote",
            )
        expected = expected_versions.get(r.id)
        if expected is not None and expected != r.version:
            raise VersionConflictError(
                expected=expected, actual=r.version,
            )

    title = (
        (patch.title if patch else None)
        or payload.get("suggested_title")
        or "Promoted observation"
    )
    body = (
        (patch.body if patch else None)
        or payload.get("suggested_body")
        or ""
    )
    confidence = (
        (patch.confidence if patch else None)
        or payload.get("suggested_confidence")
    )

    embedding_model_id = await _load_env_embedding_model(s, env_id)

    new_memory = Memory(
        env_id=env_id,
        kind=target_kind,
        status=MemoryStatus.active.value,
        title=title,
        body=body,
    )
    if confidence is not None:
        new_memory.confidence = float(confidence)
    s.add(new_memory)
    await s.flush()
    await s.refresh(new_memory)

    # Provenance: dream-run reference.
    await s.execute(
        insert(MemorySource).values(
            memory_id=new_memory.id,
            source_type=MemorySourceType.dream.value,
            source_ref=str(proposal.dream_run_id) if proposal.dream_run_id else str(proposal.id),
            agent_id=ctx.agent_id,
        )
    )

    # Lineage: each observation → new memory (relation=promoted_from).
    for obs in rows:
        await s.execute(
            insert(MemoryLineage).values(
                parent_memory_id=obs.id,
                child_memory_id=new_memory.id,
                relation=LineageRelation.promoted_from.value,
            )
        )

    # Audit + outbox for the new memory.
    new_tag_names: list[str] = []
    await _record_audit(
        s,
        op="create",
        memory=new_memory,
        by_agent_id=ctx.agent_id,
        before=None,
        after=_audit_snapshot(new_memory, tag_names=new_tag_names),
        extra_after={"promoted_from": [str(o.id) for o in rows]},
    )
    await enqueue_event(
        s,
        aggregate_type=OutboxAggregateType.memory,
        aggregate_id=new_memory.id,
        aggregate_version=new_memory.version,
        env_id=env_id,
        op=_outbox_op_for(MemoryStatus.active, is_create=True),
        payload=_projection_payload(
            new_memory, tag_names=new_tag_names,
            embedding_model_id=embedding_model_id,
        ),
        settings=settings,
    )

    return new_memory, new_tag_names


async def _lock_memories(s: AsyncSession, ids: list[UUID]) -> list[Memory]:
    """``SELECT ... FOR UPDATE`` on memory rows in input (sorted) order.

    Returns rows in the same order as ``ids`` when found; missing rows
    are simply absent. Caller checks for missing.
    """
    if not ids:
        return []
    stmt = (
        select(Memory)
        .where(Memory.id.in_(ids))
        .order_by(Memory.id)
        .with_for_update()
    )
    rows = (await s.execute(stmt)).scalars().all()
    return list(rows)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _dream_run_to_entry(r: DreamRun) -> DreamRunSummaryEntry:
    return DreamRunSummaryEntry(
        id=r.id,
        env_id=r.env_id,
        mode=DreamMode(r.mode),
        status=r.status,
        started_at=r.started_at,
        ended_at=r.ended_at,
        triggered_by=r.triggered_by,
        summarizer_kind=r.summarizer_kind,
        summary=dict(r.summary or {}),
        last_error=r.last_error,
    )


def _dream_proposal_to_entry(p: DreamProposal) -> DreamProposalEntry:
    return DreamProposalEntry(
        id=p.id,
        env_id=p.env_id,
        kind=p.kind,  # type: ignore[arg-type]
        status=p.status,  # type: ignore[arg-type]
        summarizer_kind=p.summarizer_kind,
        llm_failed=p.llm_failed,
        payload=dict(p.payload or {}),
        dream_run_id=p.dream_run_id,
        created_at=p.created_at,
        updated_at=p.updated_at,
        reviewed_at=p.reviewed_at,
        reviewed_by_agent_id=p.reviewed_by_agent_id,
        review_action=p.review_action,
        review_notes=p.review_notes,
    )


__all__ = [
    "DreamHeartbeatEntry",
    "DreamProposalEntry",
    "DreamProposalsListRequest",
    "DreamProposalsListResponse",
    "DreamReviewPatch",
    "DreamReviewRequest",
    "DreamReviewResponse",
    "DreamRunReport",
    "DreamRunRequest",
    "DreamRunResponse",
    "DreamRunScheduledItem",
    "DreamRunSummaryEntry",
    "DreamStatusRequest",
    "DreamStatusResponse",
    "dream_proposals_list",
    "dream_review",
    "dream_run",
    "dream_status",
    "get_active_background_tasks",
]
