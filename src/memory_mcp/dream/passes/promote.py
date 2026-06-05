"""Promote pass — emit ``promotion_candidate`` proposals from journal observations.

Goal
----

Cluster recent observation memories by which entity they reference;
when an entity has been the subject of at least
``DREAM_PROMOTE_MIN_CLUSTER_SIZE`` distinct observations within
``DREAM_PROMOTE_WINDOW_DAYS``, emit one ``promotion_candidate`` proposal
that — once accepted via ``dream_review`` — becomes a ``proposed`` fact
memory through the existing :func:`memory_mcp.memories.memory_promote`
plumbing.

Pipeline
--------

1. **Load candidate observations** — recent ``active`` observations in
   the env, ordered by ``created_at`` DESC, capped at ``cap × 5`` for
   over-fetch.
2. **Resolve entity references** — one query joining ``relations`` →
   ``graph_nodes`` (memory side) → ``graph_nodes`` (entity side) →
   ``entities``. Edge direction-agnostic (memory→entity OR
   entity→memory). **Distinct ``(memory_id, entity_id)`` pairs** so
   relation multiplicity (e.g., a memory linked to one entity via
   ``MENTIONS`` + ``DESCRIBES``) cannot inflate the cluster size — see
   rubber-duck blocker #2 from the p2.2-promote design critique.
3. **Cluster by entity** — flip the per-observation refs dict so each
   entity points at the *set* of observations referencing it. Drop
   entity-clusters smaller than ``DREAM_PROMOTE_MIN_CLUSTER_SIZE``.
   Clustering is intentionally **non-transitive**: an observation
   referencing entities A and B contributes to BOTH clusters, but the
   pass emits two separate proposals (one per entity) rather than a
   joint (A, B) cluster.
4. **Truncate evidence** — keep the most-recent
   ``DREAM_PROMOTE_OBSERVATIONS_PER_CLUSTER`` observations for
   summarization. The full set still goes into the proposal payload as
   ``all_observation_ids``; the summarizer sees only
   ``evidence_observation_ids``.
5. **Build dedupe key** — ``promote:entity=<id>:evidence=<sorted ids>``.
   Stable across runs that share the same most-recent N observations;
   advances when fresh observations arrive.
6. **3-layer idempotency** — in-run ``emitted_keys`` set; pre-summarize
   ``_open_proposal_exists`` (checks **any status**, not just open, so
   accepted/rejected proposals over the same exact evidence don't
   re-emit — see rubber-duck blocker #1); DB ``ON CONFLICT DO NOTHING``
   on the partial unique index over open proposals.
7. **Summarize** — ``await summarizer.summarize_promotion(cluster)``.
8. **Insert** — write a ``dream_proposal(promotion_candidate)`` row.

The pass **never mutates canonical state**. Acceptance happens via
``dream_review(action=accept)`` which dispatches to
:func:`memory_mcp.memories.memory_promote`.

Per-run cap
-----------

``DREAM_PROMOTE_BATCH_CAP`` bounds proposal emissions per run. Hitting
the cap is recorded on the result so observability can flag environments
that aren't keeping up. ``proposals_skipped_capped`` exposes how many
*eligible* clusters were left for the next run, completing the
accounting invariant:

``entity_clusters_found = proposals_emitted + proposals_skipped_existing
+ proposals_skipped_capped``.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import uuid as uuidlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import and_, exists, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import (
    DreamProposal,
    Entity,
    GraphNode,
    Memory,
    Relation,
)
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import (
    DreamProposalKind,
    DreamProposalStatus,
    MemoryKind,
    MemoryStatus,
)
from memory_mcp.dream.summarizer import (
    DreamSummarizer,
    PromotionCluster,
    PromotionClusterObservation,
    PromotionSummary,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result + per-cluster shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotePassResult:
    """Per-pass observability payload.

    Invariant: ``entity_clusters_found == proposals_emitted +
    proposals_skipped_existing + proposals_skipped_capped`` (modulo
    summarizer-internal failures, which surface via ``llm_failed``
    flags on each emitted proposal rather than by suppressing the
    proposal entirely).
    """

    env_id: UUID
    observations_examined: int = 0
    entity_clusters_found: int = 0
    proposals_emitted: int = 0
    proposals_skipped_existing: int = 0
    proposals_skipped_capped: int = 0
    items_capped: bool = False
    summarizer_kind: str | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class _ObservationRow:
    """Raw observation row (memory of kind ``observation``)."""

    id: UUID
    body: str
    created_at: dt.datetime


@dataclass(frozen=True)
class _EntityRefRow:
    """One ``(memory, entity)`` reference resolved from ``relations``.

    Distinct at the SQL level — relation multiplicity (multiple edge
    types between the same memory/entity pair) is collapsed before this
    row is constructed.
    """

    memory_id: UUID
    entity_id: UUID
    entity_name: str


# ---------------------------------------------------------------------------
# Loader seam helpers (extracted so tests can mock at module top level).
# ---------------------------------------------------------------------------


async def _load_observation_rows(
    *,
    env_id: UUID,
    cutoff: dt.datetime,
    cap: int,
) -> list[_ObservationRow]:
    """Load active observations in env created since ``cutoff``.

    Ordered by ``created_at DESC``. Capped at ``cap`` so a flood doesn't
    blow up memory.
    """

    async with session_scope() as s:
        stmt = (
            select(Memory.id, Memory.body, Memory.created_at)
            .where(
                and_(
                    Memory.env_id == env_id,
                    Memory.kind == MemoryKind.observation.value,
                    Memory.status == MemoryStatus.active.value,
                    Memory.created_at >= cutoff,
                )
            )
            .order_by(Memory.created_at.desc())
            .limit(cap)
        )
        rows = (await s.execute(stmt)).all()

    return [_ObservationRow(id=r[0], body=r[1] or "", created_at=r[2]) for r in rows]


async def _load_observation_entity_refs(
    *,
    env_id: UUID,
    observation_ids: Sequence[UUID],
) -> list[_EntityRefRow]:
    """Resolve memory→entity (and entity→memory) refs for observations.

    Returns DISTINCT ``(memory_id, entity_id, entity_name)`` triples —
    relation multiplicity cannot inflate cluster size.

    The query joins ``relations`` to two ``graph_nodes`` rows (src and
    dst), figures out which side is the memory and which is the entity,
    and emits one row per distinct ``(memory_id, entity_id)`` pair. The
    ``entity_name`` comes from a final join against ``entities``.
    """

    if not observation_ids:
        return []

    obs_set = list(observation_ids)
    async with session_scope() as s:
        # Two graph_nodes aliases — one per relation endpoint.
        src_node = GraphNode.__table__.alias("src_node")
        dst_node = GraphNode.__table__.alias("dst_node")

        # Edge can go memory→entity OR entity→memory. Capture both.
        memory_to_entity = and_(
            src_node.c.node_type == "memory",
            src_node.c.memory_id.in_(obs_set),
            dst_node.c.node_type == "entity",
        )
        entity_to_memory = and_(
            dst_node.c.node_type == "memory",
            dst_node.c.memory_id.in_(obs_set),
            src_node.c.node_type == "entity",
        )

        stmt = select(
            # Project (memory_id, entity_id) regardless of direction.
            # CASE WHEN src is memory THEN src.memory_id ELSE dst.memory_id END
            # is what we need — express as func/literal_column or use
            # SQLAlchemy's case().
        )
        # Build with case() expressions for portability.
        from sqlalchemy import case

        memory_id_col = case(
            (src_node.c.node_type == "memory", src_node.c.memory_id),
            else_=dst_node.c.memory_id,
        ).label("memory_id")
        entity_id_col = case(
            (src_node.c.node_type == "entity", src_node.c.entity_id),
            else_=dst_node.c.entity_id,
        ).label("entity_id")

        join_clause = Relation.__table__.join(
            src_node,
            src_node.c.id == Relation.src_node_id,
        ).join(
            dst_node,
            dst_node.c.id == Relation.dst_node_id,
        )
        ref_subq = (
            select(memory_id_col, entity_id_col)
            .select_from(join_clause)
            .where(
                and_(
                    Relation.env_id == env_id,
                    or_(memory_to_entity, entity_to_memory),
                )
            )
            .distinct()
            .subquery()
        )

        # Final join to entities for canonical_name. Name is captured at
        # emission time — staleness is acceptable; acceptance re-reads
        # canonical state.
        stmt = select(
            ref_subq.c.memory_id,
            ref_subq.c.entity_id,
            Entity.canonical_name,
        ).join(Entity, Entity.id == ref_subq.c.entity_id)

        rows = (await s.execute(stmt)).all()

    return [_EntityRefRow(memory_id=r[0], entity_id=r[1], entity_name=r[2]) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dedupe_key(
    *,
    entity_id: UUID,
    evidence_observation_ids: Sequence[UUID],
) -> str:
    """Stable canonical key for a promotion cluster.

    Built from the **truncated evidence set** (the actual observations
    the summarizer sees). When fresh observations arrive, the truncated
    set advances and the key changes — desirable: more evidence = a new
    proposal worth surfacing.

    The key uses sorted-by-string UUIDs for canonicalization, mirroring
    the dedupe pass.
    """

    sorted_ids = sorted(str(o) for o in evidence_observation_ids)
    return f"promote:entity={entity_id}:evidence=" + ",".join(sorted_ids)


async def _open_proposal_exists(
    *,
    env_id: UUID,
    dedupe_key: str,
) -> bool:
    """Cross-status existence check for ``promotion_candidate`` proposals.

    Returns ``True`` iff *any* proposal (open / accepted / rejected /
    amended / deferred / expired) already exists with this exact
    ``dedupe_key``. This is a stricter check than the dedupe pass uses
    — see rubber-duck blocker #1 from the p2.2-promote critique:
    accepting a promotion does NOT remove the source observations, so a
    pure ``status='open'`` filter would re-emit the same cluster on the
    next run. The cross-status check ensures identical evidence sets
    only ever produce one proposal in the system's lifetime.

    Different evidence sets (= different ``dedupe_key``) still flow
    through normally — fresh observations arriving over time produce
    fresh proposals.
    """

    async with session_scope() as s:
        stmt = select(
            exists().where(
                and_(
                    DreamProposal.env_id == env_id,
                    DreamProposal.kind == DreamProposalKind.promotion_candidate.value,
                    DreamProposal.dedupe_key == dedupe_key,
                )
            )
        )
        return bool((await s.execute(stmt)).scalar())


async def _insert_proposal(
    *,
    env_id: UUID,
    dream_run_id: UUID | None,
    dedupe_key: str,
    payload: dict[str, Any],
    summary: PromotionSummary,
) -> bool:
    """Insert a single ``promotion_candidate`` proposal.

    Returns ``True`` on insert; ``False`` if the partial unique index
    rejected as duplicate. Uses ``INSERT ... ON CONFLICT DO NOTHING``
    against the partial unique index so the transaction is never left
    aborted (mirrors the dedupe-pass pattern; see rubber-duck blocker #1
    from the p2.2-dedupe review).

    Note that the partial index covers only ``status='open'`` rows —
    cross-status idempotency is provided by the pre-summarize
    :func:`_open_proposal_exists` check, NOT by this insert. This
    insert layer guards against the rare interleaved-worker race where
    two passes try to write the same open proposal at the same time.
    """

    async with session_scope() as s:
        stmt = (
            pg_insert(DreamProposal)
            .values(
                id=uuidlib.uuid4(),
                env_id=env_id,
                kind=DreamProposalKind.promotion_candidate.value,
                status=DreamProposalStatus.open.value,
                payload=payload,
                summarizer_kind=summary.summarizer_kind.value,
                llm_failed=summary.llm_failed,
                dedupe_key=dedupe_key,
                dream_run_id=dream_run_id,
            )
            .on_conflict_do_nothing(
                index_elements=["env_id", "kind", "dedupe_key"],
                index_where=text("status = 'open' AND dedupe_key IS NOT NULL"),
            )
            .returning(DreamProposal.id)
        )
        result = await s.execute(stmt)
        inserted_id = result.scalar()

    if inserted_id is None:
        log.debug(
            "promote: open proposal already exists (env %s, dedupe_key %s)",
            env_id,
            dedupe_key,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run_promote(
    env_id: UUID,
    *,
    summarizer: DreamSummarizer,
    settings: Settings | None = None,
    now: dt.datetime | None = None,
    dream_run_id: UUID | None = None,
) -> PromotePassResult:
    """Run one promote pass for a single env.

    Args:
        env_id: scope. One pass per env so a noisy env doesn't slow
            down a quiet one.
        summarizer: produces ``PromotionSummary`` per cluster.
            Constructed once per worker process by the runner.
        settings: caller may override for tests.
        now: caller may override for tests.
        dream_run_id: id of the ``dream_runs`` row that initiated this
            pass; recorded on each proposal for observability joins.

    Returns:
        :class:`PromotePassResult` with per-stage counters.
    """

    settings = settings or get_settings()
    now = now or dt.datetime.now(dt.UTC)
    cutoff = now - dt.timedelta(days=settings.dream_promote_window_days)
    min_size = settings.dream_promote_min_cluster_size
    cap = settings.dream_promote_batch_cap
    obs_per_cluster = settings.dream_promote_observations_per_cluster

    started = time.perf_counter()

    # 1. Load recent observations.
    observations = await _load_observation_rows(
        env_id=env_id,
        cutoff=cutoff,
        cap=cap * 5,
    )
    if not observations:
        return PromotePassResult(
            env_id=env_id,
            summarizer_kind=summarizer.kind.value,
            duration_seconds=time.perf_counter() - started,
        )
    obs_index: dict[UUID, _ObservationRow] = {o.id: o for o in observations}

    # 2. Resolve entity refs (DISTINCT memory/entity pairs).
    refs = await _load_observation_entity_refs(
        env_id=env_id,
        observation_ids=list(obs_index.keys()),
    )

    # 3. Cluster by entity (set semantics — relation multiplicity already
    # collapsed at SQL level, but we belt-and-suspenders in Python too).
    entity_to_obs: dict[UUID, set[UUID]] = {}
    entity_names: dict[UUID, str] = {}
    for r in refs:
        # Refs against observations NOT in our recent window are dropped
        # (the JOIN can return them via memory_id, but if we didn't load
        # the observation we can't include it).
        if r.memory_id not in obs_index:
            continue
        entity_to_obs.setdefault(r.entity_id, set()).add(r.memory_id)
        entity_names[r.entity_id] = r.entity_name

    eligible_entities = [eid for eid, obs_ids in entity_to_obs.items() if len(obs_ids) >= min_size]

    # Deterministic processing order: by entity_id string. Reviewers
    # should not see a randomized cluster order across runs.
    eligible_entities.sort(key=str)

    proposals_emitted = 0
    proposals_skipped_existing = 0
    proposals_skipped_capped = 0
    entity_clusters_found = len(eligible_entities)
    items_capped = False
    emitted_keys: set[str] = set()

    for entity_id in eligible_entities:
        if proposals_emitted >= cap:
            # Remaining eligible clusters are explicitly capped.
            proposals_skipped_capped = entity_clusters_found - (proposals_emitted + proposals_skipped_existing)
            items_capped = True
            break

        full_obs_ids = sorted(
            entity_to_obs[entity_id],
            key=lambda mid: obs_index[mid].created_at,
            reverse=True,
        )
        evidence_obs_ids = full_obs_ids[:obs_per_cluster]

        dedupe_key = _build_dedupe_key(
            entity_id=entity_id,
            evidence_observation_ids=evidence_obs_ids,
        )

        # In-run dedup. Guards against pathological re-entry (shouldn't
        # happen given deterministic ordering, but cheap).
        if dedupe_key in emitted_keys:
            proposals_skipped_existing += 1
            continue

        # Pre-summarize cross-status existence check.
        if await _open_proposal_exists(env_id=env_id, dedupe_key=dedupe_key):
            proposals_skipped_existing += 1
            continue

        # Build summarizer input. Evidence is chronological-DESC (most
        # recent first) — matches what reviewers find easiest to scan.
        cluster_input = PromotionCluster(
            source_entity_id=entity_id,
            source_entity_name=entity_names[entity_id],
            observations=[
                PromotionClusterObservation(
                    memory_id=mid,
                    body=obs_index[mid].body,
                    created_at=obs_index[mid].created_at,
                )
                for mid in evidence_obs_ids
            ],
        )

        summary = await _instrumented_summarize_promotion(summarizer, cluster_input)

        # Payload separates "all evidence" from "summarized evidence" —
        # see rubber-duck finding #4 from the p2.2-promote critique.
        # Both lists chronological DESC.
        payload: dict[str, Any] = {
            "all_observation_ids": [str(mid) for mid in full_obs_ids],
            "evidence_observation_ids": [str(mid) for mid in evidence_obs_ids],
            "observation_count": len(full_obs_ids),
            "evidence_observation_count": len(evidence_obs_ids),
            "target_kind": MemoryKind.fact.value,
            "source_entity_id": str(entity_id),
            "source_entity_name": entity_names[entity_id],
            "suggested_title": summary.suggested_title,
            "suggested_body": summary.suggested_body,
            "suggested_confidence": summary.suggested_confidence,
            "summarizer_kind": summary.summarizer_kind.value,
            "llm_failed": summary.llm_failed,
            "llm_model_id": summary.llm_model_id,
        }

        inserted = await _insert_proposal(
            env_id=env_id,
            dream_run_id=dream_run_id,
            dedupe_key=dedupe_key,
            payload=payload,
            summary=summary,
        )
        if inserted:
            proposals_emitted += 1
            emitted_keys.add(dedupe_key)
        else:
            # Race: another worker beat us to it. Count as existing.
            proposals_skipped_existing += 1

    # If we didn't break out of the loop, there are no capped remainders.
    if not items_capped:
        proposals_skipped_capped = 0

    return PromotePassResult(
        env_id=env_id,
        observations_examined=len(observations),
        entity_clusters_found=entity_clusters_found,
        proposals_emitted=proposals_emitted,
        proposals_skipped_existing=proposals_skipped_existing,
        proposals_skipped_capped=proposals_skipped_capped,
        items_capped=items_capped,
        summarizer_kind=summarizer.kind.value,
        duration_seconds=time.perf_counter() - started,
    )


async def _instrumented_summarize_promotion(
    summarizer: Any,
    cluster: PromotionCluster,
) -> PromotionSummary:
    """Wrap ``summarize_promotion`` with Prometheus instrumentation.

    Mirror of ``dedupe._instrumented_summarize_merge``: counts +
    latency labelled by summarizer kind; ``llm_failed`` increments
    ``dream_llm_fallbacks_total{pass="promote"}``. Observability
    failures never poison the pass.
    """
    started = time.perf_counter()
    kind_label = summarizer.kind.value
    outcome = "ok"
    try:
        summary = await summarizer.summarize_promotion(cluster)
    except Exception:
        outcome = "error"
        try:
            from memory_mcp.observability import (
                dream_summarizer_calls_total,
                dream_summarizer_latency_seconds,
            )

            dream_summarizer_calls_total.labels(
                kind=kind_label,
                outcome=outcome,
            ).inc()
            dream_summarizer_latency_seconds.labels(
                kind=kind_label,
            ).observe(time.perf_counter() - started)
        except Exception:  # noqa: BLE001
            pass
        raise

    try:
        from memory_mcp.observability import (
            dream_llm_fallbacks_total,
            dream_summarizer_calls_total,
            dream_summarizer_latency_seconds,
        )

        dream_summarizer_latency_seconds.labels(kind=kind_label).observe(
            time.perf_counter() - started,
        )
        if getattr(summary, "llm_failed", False):
            outcome = "fallback"
            dream_llm_fallbacks_total.labels(**{"pass": "promote"}).inc()
        dream_summarizer_calls_total.labels(
            kind=kind_label,
            outcome=outcome,
        ).inc()
    except Exception:  # noqa: BLE001
        pass
    return summary
