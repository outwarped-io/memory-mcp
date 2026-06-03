"""Public ``memory_search`` tool — orchestrates lex + sem + graph + fusion.

Modes
-----

* ``auto`` — opt-in dispatcher for callers that do not want to pick a
  backend. In v0.5 this is intentionally minimal: UUID-shaped queries
  (8+ hex chars, with optional dashed suffixes) route to ``id``; everything
  else routes to ``hybrid``. The request default remains ``hybrid`` for
  v0.5; ``auto`` is expected to become the default in v0.6 after real-traffic
  validation.
* ``hybrid`` (default) — RRF fusion of ``lex`` + ``sem`` + ``graph``.
  The graph leg is best-effort: a missing spaCy model, missing entity
  resolution, or graph-store outage degrades silently to lex+sem.
* ``lex`` — Postgres FTS only.
* ``sem`` — Qdrant only.
* ``graph`` — graph leg only. Unlike its role in hybrid, ``mode=graph``
  *does* propagate backend errors as ``GRAPH_BACKEND_UNAVAILABLE`` so
  callers can distinguish "no graph hits" from "graph subsystem down".
* ``id`` — explicit ``ids`` lookup; canonical Postgres read.

Consistency modes
-----------------

* ``default`` — read whatever the projections currently have (fast).
* ``fresh`` — wait up to ``settings.search_fresh_max_wait_seconds`` for
  every involved env's relevant projection ``last_event_id`` to catch
  up. When the request involves the **graph leg**, this waits on BOTH
  the ``qdrant`` and ``neo4j`` sinks simultaneously — a sem-only or
  hybrid-without-graph-resolution request still only waits on
  ``qdrant``. If any required sink times out the request degrades to
  ``canonical`` (lex-only) — preserves the existing read-after-write
  truthfulness invariant.
* ``canonical`` — never query Qdrant or Neo4j; lex-only against
  Postgres.

``include_*`` flags
-------------------

By default only ``proposed`` + ``active`` are visible. ``include_stale``
adds ``stale``; ``include_archived`` and ``include_retired`` widen
further. Note that archived/retired/superseded memories are tombstoned
in Qdrant, so widening those flags only affects ``lex`` and
``canonical`` — never ``sem``. The graph leg does not pre-filter on
status; lifecycle filtering happens post-fusion against canonical
Postgres.

``follow_superseded``
---------------------

When True (default), a hit pointing at a memory whose ``superseded_by``
is set is *replaced* with the successor (if visible to the caller),
preserving the original score & sources. The original is dropped from
the result set so callers don't see both unless they ask for it
(``include_archived=True`` AND ``follow_superseded=False``).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp.config import Settings, get_settings
from memory_mcp.db.graph.base import GraphStore
from memory_mcp.db.models import Environment, Memory, Outbox, ProjectionState, Tag
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import MemoryKind, OutboxSink
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.embeddings.base import Embedder, get_embedder
from memory_mcp.errors import GraphBackendUnavailableError, InvalidInputError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryResponse, _to_response
from memory_mcp._filters import is_expired
from memory_mcp.search.graph import graph_search
from memory_mcp.search.lex import lex_search
from memory_mcp.search.ranking import (
    FusedHit,
    RankedHit,
    apply_salience_boost,
    reciprocal_rank_fuse,
    sort_hits,
)
from memory_mcp.search.sem import sem_search

from memory_mcp_schemas.search import (
    ConsistencyMode,
    ExpansionPreset,
    MemorySearchHit,
    MemorySearchRequest,
    MemorySearchResponse,
    ProjectionStatusEntry,
    SearchMode,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VISIBLE_DEFAULT: list[str] = ["proposed", "active"]
_AUTO_ID_QUERY_RE = re.compile(r"^[0-9a-fA-F]{8,32}(-[0-9a-fA-F]+)*$")


def _expansion_preset_updates(req: MemorySearchRequest) -> dict[str, Any] | None:
    if req.expansion is None:
        return None
    if req.expansion == ExpansionPreset.default:
        return {}
    if req.expansion == ExpansionPreset.narrow:
        return {
            "min_score": 0.035,
            "fallback": False,
            "follow_superseded": False,
        }
    if req.expansion == ExpansionPreset.broad:
        return {
            "fallback": True,
            "follow_superseded": True,
            "include_stale": True,
            "include_archived": True,
        }
    raise ValueError(f"unsupported expansion preset: {req.expansion!r}")


def _resolve_expansion_preset(req: MemorySearchRequest) -> MemorySearchRequest:
    updates = _expansion_preset_updates(req)
    if updates in (None, {}):
        return req
    return req.model_copy(update=updates)


def _with_expansion_resolved(
    response: MemorySearchResponse,
    expansion_resolved: dict[str, Any] | None,
) -> MemorySearchResponse:
    if expansion_resolved is None:
        return response
    return response.model_copy(update={"expansion_resolved": expansion_resolved})


def _resolve_auto_mode(query: str) -> SearchMode:
    """Resolve ``mode="auto"`` to the concrete search mode to execute.

    The v0.5 heuristic is intentionally minimal and conservative while we
    gather real traffic: UUID-shaped queries route to ``id``; all other
    non-empty queries route to ``hybrid``. Smarter dispatch (short keyword vs.
    phrase, proper-noun detection, and related heuristics) is deferred.
    """
    trimmed = query.strip()
    if not trimmed:
        raise InvalidInputError("INVALID_INPUT: mem_search query cannot be empty")
    if _AUTO_ID_QUERY_RE.fullmatch(trimmed):
        return "id"
    return "hybrid"


def _with_auto_resolved_id(req: MemorySearchRequest) -> MemorySearchRequest:
    """Thread a full UUID query into the explicit id lookup path."""
    if req.ids:
        return req
    try:
        memory_id = UUID(req.query.strip())
    except ValueError:
        return req
    return req.model_copy(update={"ids": [memory_id]})


def _statuses_to_query(req: MemorySearchRequest) -> list[str]:
    s = list(_VISIBLE_DEFAULT)
    if req.include_stale:
        s.append("stale")
    if req.include_archived:
        s.append("archived")
    if req.include_retired:
        s.append("retired")
        s.append("superseded")
    return s


def _resolve_env_ids(
    explicit: list[UUID] | None,
    ctx: AgentContext,
) -> list[UUID]:
    if explicit:
        return list(dict.fromkeys(explicit))
    attached = list(dict.fromkeys(ctx.attached_env_ids))
    return attached  # may be empty (callers without an attached env see no sem results)


async def _projection_status(
    session: AsyncSession,
    env_ids: list[UUID],
    *,
    sinks: Sequence[OutboxSink] = (OutboxSink.qdrant,),
) -> list[ProjectionStatusEntry]:
    if not env_ids:
        return []
    sink_values = [s.value for s in sinks]
    rows = (await session.execute(
        select(
            ProjectionState.env_id,
            ProjectionState.sink,
            ProjectionState.last_event_id,
            ProjectionState.lag_seconds,
            ProjectionState.status,
        ).where(
            ProjectionState.env_id.in_(env_ids),
            ProjectionState.sink.in_(sink_values),
        )
    )).all()
    return [
        ProjectionStatusEntry(
            env_id=r[0],
            sink=r[1],
            last_event_id=r[2],
            lag_seconds=float(r[3]) if r[3] is not None else None,
            status=r[4],
        )
        for r in rows
    ]


async def _capture_watermarks(
    session: AsyncSession,
    env_ids: list[UUID],
) -> dict[UUID, int | None]:
    """Snapshot the latest outbox event id per env at search-start time.

    Returns ``{env_id: max_event_id_or_None}``. ``None`` means the env has
    never emitted an outbox event, which is automatically "caught up".
    """
    if not env_ids:
        return {}
    rows = (await session.execute(
        select(Outbox.env_id, func.max(Outbox.event_id))
        .where(Outbox.env_id.in_(env_ids))
        .group_by(Outbox.env_id)
    )).all()
    by_env: dict[UUID, int | None] = dict.fromkeys(env_ids, None)
    for env_id, max_id in rows:
        by_env[env_id] = int(max_id) if max_id is not None else None
    return by_env


async def _wait_for_watermarks(
    session_factory,
    watermarks: dict[UUID, int | None],
    *,
    max_wait_seconds: float,
    sinks: Sequence[OutboxSink] = (OutboxSink.qdrant,),
    poll_interval: float = 0.1,
) -> bool:
    """Wait until ``projection_state.last_event_id`` meets each env's watermark
    on **every** required sink.

    Envs whose snapshot watermark is ``None`` (no outbox events) are caught
    up by definition. Returns True if all (env, sink) pairs caught up,
    False on timeout. Multi-sink support (rubber-duck BLOCKER 1): a
    request that involves the graph leg waits on both qdrant and neo4j;
    if any sink lags out, the whole request degrades to canonical so the
    response's ``consistency_used`` remains truthful.
    """
    pending = {eid: wm for eid, wm in watermarks.items() if wm is not None}
    if not pending or not sinks:
        return True
    sink_values = [s.value for s in sinks]
    deadline = asyncio.get_running_loop().time() + max_wait_seconds
    while True:
        async with session_factory() as s:
            rows = (await s.execute(
                select(
                    ProjectionState.env_id,
                    ProjectionState.sink,
                    ProjectionState.last_event_id,
                )
                .where(
                    ProjectionState.env_id.in_(list(pending.keys())),
                    ProjectionState.sink.in_(sink_values),
                )
            )).all()
        # Build ``{(env_id, sink): last_event_id}`` snapshot.
        seen: dict[tuple[UUID, str], int | None] = {
            (r[0], r[1]): r[2] for r in rows
        }
        all_ok = True
        for env_id, target in pending.items():
            for sink_value in sink_values:
                last = seen.get((env_id, sink_value))
                if last is None or last < target:
                    all_ok = False
                    break
            if not all_ok:
                break
        if all_ok:
            return True
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(poll_interval)


async def _hydrate_memories(
    session: AsyncSession,
    memory_ids: list[UUID],
    visible_env_ids: list[UUID] | None,
) -> dict[UUID, Memory]:
    """Bulk-load memories. Filter by env visibility if non-empty."""
    if not memory_ids:
        return {}
    stmt = select(Memory).where(Memory.id.in_(memory_ids))
    if visible_env_ids:
        stmt = stmt.where(Memory.env_id.in_(visible_env_ids))
    rows = (await session.execute(stmt)).scalars().all()
    return {m.id: m for m in rows}


async def _bulk_load_tag_names(
    session: AsyncSession,
    memory_ids: list[UUID],
) -> dict[UUID, list[str]]:
    """Single round-trip ``{memory_id: [tag_names…]}``."""
    if not memory_ids:
        return {}
    from memory_mcp.db.models import MemoryTag  # local to avoid circular

    rows = (await session.execute(
        select(MemoryTag.memory_id, Tag.name)
        .join(Tag, MemoryTag.tag_id == Tag.id)
        .where(MemoryTag.memory_id.in_(memory_ids))
        .order_by(MemoryTag.memory_id, Tag.name)
    )).all()
    out: dict[UUID, list[str]] = {mid: [] for mid in memory_ids}
    for mid, name in rows:
        out[mid].append(name)
    return out


def _passes_post_filters(
    memory: Memory,
    *,
    statuses: list[str],
    kinds: list[str] | None,
    tag_names: list[str],
    tags: list[str] | None,
    created_after: dt.datetime | None,
    created_before: dt.datetime | None,
    updated_after: dt.datetime | None,
    include_expired: bool = False,
) -> bool:
    if memory.status not in statuses:
        return False
    if kinds and memory.kind not in kinds:
        return False
    if tags and not (set(tags) & set(tag_names)):
        return False
    if created_after and memory.created_at < created_after:
        return False
    if created_before and memory.created_at >= created_before:
        return False
    if updated_after and memory.updated_at < updated_after:
        return False
    return include_expired or not is_expired(memory)


# ---------------------------------------------------------------------------
# Per-mode dispatch
# ---------------------------------------------------------------------------


async def _do_lex(
    session: AsyncSession, req: MemorySearchRequest, env_ids: list[UUID], statuses: list[str], leg_limit: int,
) -> list[RankedHit]:
    return await lex_search(
        session,
        query=req.query,
        env_ids=env_ids,
        statuses=statuses,
        kinds=[k.value for k in req.kinds] if req.kinds else None,
        tags=req.tags,
        created_after=req.created_after,
        created_before=req.created_before,
        updated_after=req.updated_after,
        limit=leg_limit,
        include_expired=req.include_expired,
    )


async def _do_sem(
    session: AsyncSession,
    req: MemorySearchRequest,
    env_ids: list[UUID],
    statuses: list[str],
    leg_limit: int,
    *,
    vector_store: VectorStore,
    embedder: Embedder,
) -> list[RankedHit]:
    # Sem ignores archived/retired/superseded automatically (tombstoned).
    # We restrict statuses to the projection-visible set to make it explicit.
    sem_statuses = [s for s in statuses if s in {"proposed", "active", "stale"}]
    if not sem_statuses:
        return []
    return await sem_search(
        session,
        vector_store=vector_store,
        embedder=embedder,
        query=req.query,
        env_ids=env_ids,
        statuses=sem_statuses,
        kinds=[k.value for k in req.kinds] if req.kinds else None,
        tags=req.tags,
        limit=leg_limit,
    )


async def _search_by_trigger(
    task_desc: str,
    env_id: UUID,
    top_k: int,
    *,
    settings: Settings | None = None,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> list[tuple[UUID, float]]:
    """Search the per-env Qdrant ``trigger`` named vector for auto-context.

    Trigger vectors live on the same memory point as the body vector, but under
    a separate named vector (``trigger``). Normal semantic search always queries
    ``body``; this helper queries only ``trigger`` and filters to points whose
    payload says a trigger description exists, so authoring triggers never
    pollute body recall.
    """
    query = task_desc.strip()
    if not query or top_k <= 0:
        return []

    settings = settings or get_settings()
    vs = vector_store or _default_vector_store(settings)
    emb = embedder or get_embedder(settings)

    async with session_scope() as session:
        model_id = (await session.execute(
            select(Environment.default_embedding_model_id).where(Environment.id == env_id)
        )).scalar_one_or_none()
        if model_id is None:
            return []
        if model_id != emb.model_id:
            from memory_mcp.errors import EmbeddingModelMismatchError

            raise EmbeddingModelMismatchError(expected=str(model_id), actual=emb.model_id)

    vectors = await asyncio.get_running_loop().run_in_executor(
        None,
        emb.embed_texts,
        [query],
    )
    qvec = vectors[0]
    raw_hits = await vs.search(
        env_id=env_id,
        query_vector=qvec,
        limit=max(top_k * 4, top_k),
        filters={
            "status": ["proposed", "active", "stale"],
            "has_trigger_description": True,
        },
        vector_name="trigger",
    )

    scores: dict[UUID, float] = {}
    for hit in raw_hits:
        memory_id = UUID(str(hit["id"]))
        score = float(hit["score"])
        if score > scores.get(memory_id, float("-inf")):
            scores[memory_id] = score
    if not scores:
        return []

    async with session_scope() as session:
        rows = (await session.execute(
            select(Memory).where(
                Memory.id.in_(list(scores.keys())),
                Memory.env_id == env_id,
                Memory.trigger_description.is_not(None),
                Memory.status.in_(["proposed", "active", "stale"]),
            )
        )).scalars().all()

    ordered = sorted(
        rows,
        key=lambda m: (scores[m.id], float(m.salience)),
        reverse=True,
    )
    return [(m.id, scores[m.id]) for m in ordered[:top_k]]


async def _do_graph(
    session: AsyncSession,
    req: MemorySearchRequest,
    env_ids: list[UUID],
    leg_limit: int,
    *,
    graph_store: GraphStore,
    settings: Settings,
) -> list[RankedHit]:
    """Run the graph leg. Backend errors propagate to the caller."""
    if not req.query.strip():
        return []
    return await graph_search(
        session,
        graph_store=graph_store,
        query=req.query,
        env_ids=env_ids,
        limit=leg_limit,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def memory_search(
    req: MemorySearchRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    graph_store: GraphStore | None = None,
) -> MemorySearchResponse:
    """Top-level ``mem_search`` entry point.

    Orchestrates the optional ``fallback`` relaxation ladder around the
    core pass implemented in ``_memory_search_pass``. The first pass
    always runs the caller's request as-is; if it returns 0 hits and
    ``req.fallback`` is True, the orchestrator re-runs with progressively
    broader scope until something matches (or all steps are exhausted).
    ``mode=id`` does not participate in the cascade — its hit set is
    explicit and broadening it would silently expand the lookup.
    """
    resolved_req = _resolve_expansion_preset(req)
    expansion_resolved = _expansion_preset_updates(req)

    # First pass: caller's request as given (after expansion preset resolution).
    response = await _memory_search_pass(
        resolved_req,
        ctx=ctx,
        settings=settings,
        vector_store=vector_store,
        embedder=embedder,
        graph_store=graph_store,
    )
    if response.hits or not resolved_req.fallback or resolved_req.mode == "id":
        return _with_expansion_resolved(response, expansion_resolved)

    # Build the cascade. Each step yields (name, relaxed_request) and is
    # gated on the previous pass returning 0 hits. Steps deliberately
    # build on each other rather than resetting — once we relax filters,
    # subsequent passes inherit that relaxation.
    fallback_used: list[str] = []
    cascade_req = resolved_req

    for step_name, builder in _FALLBACK_STEPS:
        next_req = builder(cascade_req)
        if next_req is None:
            # Step is a no-op for this request shape (e.g. mode already
            # hybrid). Skip without recording so ``fallback_used`` only
            # carries steps that actually changed the request.
            continue
        cascade_req = next_req
        fallback_used.append(step_name)
        response = await _memory_search_pass(
            cascade_req,
            ctx=ctx,
            settings=settings,
            vector_store=vector_store,
            embedder=embedder,
            graph_store=graph_store,
        )
        if response.hits:
            break

    return _with_expansion_resolved(
        response.model_copy(update={"fallback_used": fallback_used}),
        expansion_resolved,
    )


def _step_widen_mode(req: MemorySearchRequest) -> MemorySearchRequest | None:
    """Step 1: widen ``mode=lex`` to ``hybrid``. No-op for broader modes."""
    if req.mode == "lex":
        return req.model_copy(update={"mode": "hybrid"})
    return None


def _step_drop_filters(req: MemorySearchRequest) -> MemorySearchRequest | None:
    """Step 2: drop ``kinds`` / ``tags`` / time bounds. No-op when none set."""
    if not (
        req.kinds
        or req.tags
        or req.created_after is not None
        or req.created_before is not None
        or req.updated_after is not None
    ):
        return None
    return req.model_copy(update={
        "kinds": None,
        "tags": None,
        "created_after": None,
        "created_before": None,
        "updated_after": None,
    })


def _step_widen_lifecycle(req: MemorySearchRequest) -> MemorySearchRequest | None:
    """Step 3: include stale + archived. No-op when already widened."""
    if req.include_stale and req.include_archived:
        return None
    return req.model_copy(update={
        "include_stale": True,
        "include_archived": True,
    })


def _step_boost_limit(req: MemorySearchRequest) -> MemorySearchRequest | None:
    """Step 4: drop ``follow_superseded`` + boost ``limit`` 5x (cap 100)."""
    new_limit = min(req.limit * 5, 100)
    if not req.follow_superseded and new_limit == req.limit:
        return None
    return req.model_copy(update={
        "follow_superseded": False,
        "limit": new_limit,
    })


_FALLBACK_STEPS: tuple[
    tuple[str, Callable[[MemorySearchRequest], MemorySearchRequest | None]],
    ...,
] = (
    ("mode->hybrid", _step_widen_mode),
    ("drop_filters", _step_drop_filters),
    ("widen_lifecycle", _step_widen_lifecycle),
    ("boost_limit", _step_boost_limit),
)


async def _memory_search_pass(
    req: MemorySearchRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    graph_store: GraphStore | None = None,
) -> MemorySearchResponse:
    settings = settings or get_settings()
    requested_mode: SearchMode = req.mode
    dispatched_mode: SearchMode = (
        _resolve_auto_mode(req.query) if requested_mode == "auto" else requested_mode
    )
    env_ids = _resolve_env_ids(req.env_ids, ctx)
    final_statuses = _statuses_to_query(req)
    # Retrieval-time statuses: when follow_superseded is on, we MUST include
    # `superseded` in the lex/canonical search even if the caller's visibility
    # set excludes it — otherwise the superseded predecessor never enters the
    # rewrite stage. The post-filter pass (against final_statuses) drops
    # any superseded hit that wasn't successfully rewritten.
    retrieval_statuses: list[str] = list(final_statuses)
    if req.follow_superseded and "superseded" not in retrieval_statuses:
        retrieval_statuses.append("superseded")
    consistency_used: ConsistencyMode = req.consistency

    # ---- mode=id short-circuit ------------------------------------------
    if dispatched_mode == "id":
        id_req = _with_auto_resolved_id(req) if requested_mode == "auto" else req
        return await _serve_by_ids(
            id_req,
            env_ids,
            final_statuses,
            response_mode=requested_mode,
        )

    # ---- determine which sinks the request needs ------------------------
    # - Hybrid waits on qdrant always (sem leg) and additionally on neo4j
    #   when the graph leg can contribute.
    # - sem alone waits on qdrant.
    # - graph alone waits on neo4j.
    # - lex doesn't wait (canonical only).
    fresh_sinks: list[OutboxSink] = []
    if dispatched_mode in ("hybrid", "sem"):
        fresh_sinks.append(OutboxSink.qdrant)
    if dispatched_mode in ("hybrid", "graph"):
        fresh_sinks.append(OutboxSink.neo4j)

    # ---- consistency: fresh — capture watermark, wait, or degrade -------
    if (
        dispatched_mode in ("hybrid", "sem", "graph")
        and req.consistency == "fresh"
        and env_ids
    ):
        async with session_scope() as ws:
            watermarks = await _capture_watermarks(ws, env_ids)
        ok = await _wait_for_watermarks(
            session_scope,
            watermarks,
            max_wait_seconds=settings.search_fresh_max_wait_seconds,
            sinks=fresh_sinks,
        )
        if not ok:
            log.info(
                "memory_search consistency=fresh timed out across sinks=%s; "
                "degrading to canonical",
                [s.value for s in fresh_sinks],
            )
            consistency_used = "canonical"

    # ---- consistency: canonical — force lex-only ------------------------
    effective_mode: SearchMode = dispatched_mode
    if consistency_used == "canonical" and dispatched_mode in ("hybrid", "sem", "graph"):
        effective_mode = "lex"

    leg_limit = max(2 * req.limit, settings.search_min_per_leg)
    if not env_ids:
        # No envs visible to caller. Return an empty response.
        return MemorySearchResponse(
            hits=[], mode=req.mode, effective_mode=effective_mode,
            consistency_used=consistency_used, projection_status=[],
        )

    # ---- retrieve --------------------------------------------------------
    if effective_mode == "lex":
        async with session_scope() as session:
            ranked_lists = [await _do_lex(
                session, req, env_ids, retrieval_statuses, leg_limit,
            )]
    elif effective_mode == "sem":
        vs = vector_store or _default_vector_store(settings)
        emb = embedder or get_embedder(settings)
        async with session_scope() as session:
            ranked_lists = [await _do_sem(
                session, req, env_ids, retrieval_statuses, leg_limit,
                vector_store=vs, embedder=emb,
            )]
    elif effective_mode == "graph":
        # Rubber-duck BLOCKER 2: mode=graph propagates backend errors.
        try:
            gs = graph_store or await _default_graph_store(settings)
        except Exception as exc:
            raise GraphBackendUnavailableError(
                f"graph store unavailable: {exc}",
            ) from exc
        try:
            async with session_scope() as session:
                ranked_lists = [await _do_graph(
                    session, req, env_ids, leg_limit,
                    graph_store=gs, settings=settings,
                )]
        except GraphBackendUnavailableError:
            raise
        except Exception as exc:
            raise GraphBackendUnavailableError(
                f"graph leg failed: {exc}",
            ) from exc
    elif effective_mode == "hybrid":
        vs = vector_store or _default_vector_store(settings)
        emb = embedder or get_embedder(settings)

        async def _lex_leg() -> list[RankedHit]:
            async with session_scope() as s:
                return await _do_lex(
                    s, req, env_ids, retrieval_statuses, leg_limit,
                )

        async def _sem_leg() -> list[RankedHit]:
            async with session_scope() as s:
                return await _do_sem(
                    s, req, env_ids, retrieval_statuses, leg_limit,
                    vector_store=vs, embedder=emb,
                )

        async def _graph_leg() -> list[RankedHit]:
            # Best-effort: any backend / store-acquisition failure
            # degrades the leg to empty. Hybrid keeps lex+sem.
            try:
                gs = graph_store or await _default_graph_store(settings)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "memory_search hybrid: graph store unavailable (%s); "
                    "graph leg disabled for this request",
                    exc,
                )
                return []
            try:
                async with session_scope() as s:
                    return await _do_graph(
                        s, req, env_ids, leg_limit,
                        graph_store=gs, settings=settings,
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "memory_search hybrid: graph leg failed (%s); "
                    "continuing with lex+sem",
                    exc,
                )
                return []

        ranked_lists = await asyncio.gather(
            _lex_leg(), _sem_leg(), _graph_leg(),
        )
    else:
        raise ValueError(f"unsupported mode: {effective_mode!r}")

    async with session_scope() as session:
        # ---- fuse + post-process ---------------------------------------
        fused = reciprocal_rank_fuse(lists=ranked_lists)
        memory_ids = list(fused.keys())

        memories = await _hydrate_memories(session, memory_ids, env_ids)
        # Drop hits whose memory was deleted / not visible.
        for mid in list(fused.keys()):
            if mid not in memories:
                del fused[mid]

        # Optional follow_superseded: replace pointers (chain-aware).
        if req.follow_superseded:
            await _follow_superseded(session, fused, memories, env_ids)

        # Bulk pre-load tags up front when needed for filtering OR output —
        # avoids N+1 round-trips against Postgres.
        tag_map: dict[UUID, list[str]] = {}
        if req.tags or fused:
            tag_map = await _bulk_load_tag_names(session, list(fused.keys()))

        # Apply post-filters against the caller's intended visibility set
        # (final_statuses). Any superseded predecessor that didn't get
        # successfully rewritten is dropped here.
        for mid in list(fused.keys()):
            m = memories.get(mid)
            if m is None:
                del fused[mid]
                continue
            if not _passes_post_filters(
                m,
                statuses=final_statuses,
                kinds=[k.value for k in req.kinds] if req.kinds else None,
                tag_names=tag_map.get(mid, []),
                tags=req.tags,
                created_after=req.created_after,
                created_before=req.created_before,
                updated_after=req.updated_after,
                include_expired=req.include_expired,
            ):
                del fused[mid]

        # Hydrate scoring-relevant fields + sort.
        for mid, hit in fused.items():
            m = memories[mid]
            hit.salience = float(m.salience)
            hit.pinned = bool(m.pinned)
            hit.updated_at = m.updated_at

        apply_salience_boost(fused.values())

        # Post-fusion ``min_score`` threshold (the *tighten* lever).
        # Applied after salience boost so the threshold reflects the
        # caller-visible final ``score``.
        if req.min_score is not None:
            for mid in list(fused.keys()):
                if fused[mid].score < req.min_score:
                    del fused[mid]

        sorted_hits = sort_hits(fused.values())[: req.limit]

        out_hits: list[MemorySearchHit] = []
        for h in sorted_hits:
            m = memories[h.memory_id]
            out_hits.append(MemorySearchHit(
                memory=_to_response(m, tag_map.get(h.memory_id, [])),
                score=h.score,
                sources=h.sources,
                raw_scores=h.raw_scores,
            ))

        # Multi-sink projection status — include both qdrant and neo4j
        # when the graph leg can contribute, so the client can see the
        # actual freshness of every consulted projection.
        status_sinks: list[OutboxSink] = []
        if dispatched_mode in ("hybrid", "sem"):
            status_sinks.append(OutboxSink.qdrant)
        if dispatched_mode in ("hybrid", "graph"):
            status_sinks.append(OutboxSink.neo4j)
        if not status_sinks:
            status_sinks.append(OutboxSink.qdrant)
        proj = await _projection_status(session, env_ids, sinks=status_sinks)

    return MemorySearchResponse(
        hits=out_hits,
        mode=req.mode,
        effective_mode=effective_mode,
        consistency_used=consistency_used,
        projection_status=proj,
        truncated=len(fused) > req.limit,
    )


async def _serve_by_ids(
    req: MemorySearchRequest,
    env_ids: list[UUID],
    statuses: list[str],
    *,
    response_mode: SearchMode = "id",
) -> MemorySearchResponse:
    """``mode=id`` — canonical lookup; ranking by request order."""
    ids = list(dict.fromkeys(req.ids or []))
    if not ids:
        return MemorySearchResponse(
            hits=[], mode=response_mode, effective_mode="id",
            consistency_used="canonical", projection_status=[],
        )
    async with session_scope() as session:
        memories = await _hydrate_memories(session, ids, env_ids)
        resolved: dict[UUID, UUID] = {}
        if req.follow_superseded:
            # Build a fake fused map to reuse follow logic.
            fused: dict[UUID, FusedHit] = {
                mid: FusedHit(memory_id=mid, score=1.0, sources=["id"])
                for mid in memories
            }
            resolved = await _follow_superseded(session, fused, memories, env_ids)
        # Resolved-id sequence: each requested id maps to its successor (if
        # any) or itself. De-duplicate while preserving request order so a
        # fan-in (multiple olds → one new) shows up exactly once.
        seen_resolved: set[UUID] = set()
        ordered: list[Memory] = []
        for mid in ids:
            target = resolved.get(mid, mid)
            if target in seen_resolved:
                continue
            m = memories.get(target)
            if m is None or m.status not in statuses:
                continue
            if not req.include_expired and is_expired(m):
                continue
            seen_resolved.add(target)
            ordered.append(m)
        tag_map = await _bulk_load_tag_names(session, [m.id for m in ordered])
        proj = await _projection_status(session, env_ids)

    hits = [
        MemorySearchHit(
            memory=_to_response(m, tag_map.get(m.id, [])),
            score=1.0,
            sources=["id"],
            raw_scores={"id": 1.0},
        )
        for m in ordered[: req.limit]
    ]
    return MemorySearchResponse(
        hits=hits, mode=response_mode, effective_mode="id",
        consistency_used="canonical", projection_status=proj,
        truncated=len(ordered) > req.limit,
    )


_FOLLOW_SUPERSEDED_MAX_HOPS = 8


async def _follow_superseded(
    session: AsyncSession,
    fused: dict[UUID, FusedHit],
    memories: dict[UUID, Memory],
    env_ids: list[UUID],
) -> dict[UUID, UUID]:
    """Rewrite ``superseded_by`` pointers chain-wise to their final successors.

    Returns ``{original_id: resolved_id}`` for every old id that was
    successfully redirected. Originals whose successor is not visible (e.g.
    cross-env without grant) are left untouched and **omitted** from the
    returned map. Cycle and depth guards bound the walk at
    ``_FOLLOW_SUPERSEDED_MAX_HOPS`` hops.
    """
    # Step 1: resolve each old id to the end of its visible chain.
    starts = [m.id for m in memories.values() if m.superseded_by is not None]
    if not starts:
        return {}

    resolution: dict[UUID, UUID] = {}
    pending_lookup: set[UUID] = set()

    def _enqueue(mid: UUID) -> None:
        if mid not in memories:
            pending_lookup.add(mid)

    for sid in starts:
        _enqueue(memories[sid].superseded_by)  # type: ignore[arg-type]

    # Hydrate the first batch of successors so the loop has something to walk.
    while pending_lookup:
        batch = await _hydrate_memories(session, list(pending_lookup), env_ids)
        memories.update(batch)
        pending_lookup.clear()
        # If any successors are themselves superseded and not yet hydrated,
        # queue them up. Bounded by the chain depth guard below.
        for m in batch.values():
            if m.superseded_by is not None and m.superseded_by not in memories:
                pending_lookup.add(m.superseded_by)

    for original in starts:
        seen: set[UUID] = {original}
        cursor = original
        for _ in range(_FOLLOW_SUPERSEDED_MAX_HOPS):
            curr = memories.get(cursor)
            if curr is None or curr.superseded_by is None:
                break
            nxt = curr.superseded_by
            if nxt in seen:
                # Cycle — give up; treat as no rewrite.
                cursor = original
                break
            seen.add(nxt)
            if nxt not in memories:
                # Successor not visible — keep predecessor (cursor unchanged).
                break
            cursor = nxt
        if cursor != original:
            resolution[original] = cursor

    # Step 2: apply the resolution to the fused map.
    for old_id, new_id in resolution.items():
        old_hit = fused.pop(old_id, None)
        if old_hit is None:
            continue
        existing = fused.get(new_id)
        if existing is None:
            old_hit.memory_id = new_id
            fused[new_id] = old_hit
        else:
            existing.score += old_hit.score
            for src in old_hit.sources:
                if src not in existing.sources:
                    existing.sources.append(src)
            existing.raw_scores.update(old_hit.raw_scores)
    return resolution


_DEFAULT_VECTOR_STORE: VectorStore | None = None


def _default_vector_store(settings: Settings) -> VectorStore:
    """Lazy-init a process-wide vector store (Qdrant)."""
    global _DEFAULT_VECTOR_STORE
    if _DEFAULT_VECTOR_STORE is None:
        if settings.vector_backend == "qdrant":
            from memory_mcp.db.vector.qdrant import QdrantVectorStore

            _DEFAULT_VECTOR_STORE = QdrantVectorStore(settings)
        else:
            raise NotImplementedError(
                f"vector_backend={settings.vector_backend!r} not implemented in v1"
            )
    return _DEFAULT_VECTOR_STORE


def _reset_default_vector_store_for_tests() -> None:
    """Test hook — clear the singleton so tests can inject fakes."""
    global _DEFAULT_VECTOR_STORE
    _DEFAULT_VECTOR_STORE = None


async def _default_graph_store(settings: Settings) -> GraphStore:
    """Lazy-init a process-wide graph store (Neo4j or Postgres CTE).

    Delegates to the canonical singleton in ``memory_mcp.graph`` — this
    is just a thin re-export so the search module doesn't pull in the
    full graph-tools layer for unrelated tests.
    """
    from memory_mcp.graph import _get_default_graph_store

    return await _get_default_graph_store(settings)


__all__ = [
    "ConsistencyMode",
    "ExpansionPreset",
    "MemorySearchHit",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "ProjectionStatusEntry",
    "SearchMode",
    "_search_by_trigger",
    "memory_search",
]
