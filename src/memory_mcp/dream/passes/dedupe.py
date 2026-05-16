"""Dedupe pass — emit ``merge_candidate`` proposals for near-duplicate memories.

Pipeline (per env, per run):

1. **Seed selection** — pick recently-updated ``active`` memories as
   cluster seeds. Memories outside the window are still *eligible
   members* (a freshly-written memory can cluster with a year-old one),
   but only seeds drive the search loop, so re-runs are cheap.
2. **Neighbor query** — for each seed, embed its ``title + body``,
   query Qdrant for the top-K nearest in the same env+kind with
   ``status=active``, and keep neighbors above
   ``DREAM_DEDUPE_THRESHOLD``.
3. **Clustering** — seed-local expansion: each seed becomes a cluster
   together with its above-threshold Qdrant neighbors. Clustering is
   intentionally **non-transitive** (we do not run union-find), so a
   "bridge" duplicate (A↔B above threshold, B↔C above threshold,
   A↔C below) does NOT merge into a single ``{A,B,C}`` proposal — that
   conservatism prevents false-positive mega-merges. Two seeds may
   yield clusters that overlap; the in-run dedupe-key set absorbs
   identical sets, but proper-subset / proper-superset overlaps surface
   as separate reviewer proposals (intended for v1; revisit when
   reviewer feedback lands).
4. **Primary selection** — within each cluster, the member with the
   highest salience becomes the "canonical primary"; lex order on UUID
   is the tiebreaker. ``dedupe_key`` is built from the sorted member
   set so the same cluster always produces the same key regardless of
   which seed surfaced it.
5. **Idempotency** — three layers of dedup, in this order:
   (a) the in-run ``emitted_keys`` set short-circuits sibling seeds;
   (b) a pre-summarizer ``SELECT EXISTS`` against open proposals
   short-circuits cross-run duplicates BEFORE paying summarizer/LLM
   cost; (c) the DB ``ON CONFLICT DO NOTHING`` against the partial
   unique index on ``dream_proposals(env_id, kind, dedupe_key) WHERE
   status='open'`` covers the rare interleaved-worker race.
6. **Summarization** — call ``summarizer.summarize_merge(cluster)``.
7. **Insert** — write a ``dream_proposal(merge_candidate)`` row.

The pass **never mutates canonical state**. Acceptance happens via
``dream_review(action=accept)`` in :mod:`memory_mcp.dream_tools`, which
dispatches to :func:`memory_mcp.memories.memory_supersede`.

Per-run cap
-----------

``DREAM_DEDUPE_BATCH_CAP`` bounds how many new proposals each run
emits. The cap exists to bound LLM call volume in ``llm`` mode and
reviewer cognitive load on a busy env. Hitting the cap is recorded on
the result so observability can flag environments that aren't keeping
up; the next run picks up where this one stopped (clusters not yet
covered by an open proposal).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
import uuid as uuidlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from sqlalchemy import and_, exists, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import DreamProposal, Memory
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import (
    DreamProposalKind,
    DreamProposalStatus,
    MemoryStatus,
)
from memory_mcp.dream.summarizer import (
    DreamSummarizer,
    MergeCluster,
    MergeClusterMember,
    MergeSummary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from memory_mcp.embeddings.base import Embedder

log = logging.getLogger(__name__)


class _SearchableVectorStore(Protocol):
    """Narrow surface this pass uses — keeps mocks tiny in tests."""

    async def search(
        self,
        *,
        env_id: UUID,
        query_vector: Sequence[float],
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Result + per-cluster shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DedupePassResult:
    """Per-pass observability payload."""

    env_id: UUID
    seeds_examined: int = 0
    clusters_found: int = 0
    proposals_emitted: int = 0
    proposals_skipped_existing: int = 0
    proposals_skipped_below_min_size: int = 0
    items_capped: bool = False
    summarizer_kind: str | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class _SeedRow:
    """Raw seed pulled from Postgres before Qdrant lookup."""

    id: UUID
    title: str | None
    body: str
    salience: float
    kind: str
    created_at: dt.datetime


@dataclass
class _Cluster:
    """Mutable working cluster — frozen into a ``MergeCluster`` later."""

    members: dict[UUID, MergeClusterMember] = field(default_factory=dict)
    cosine_scores: dict[UUID, float] = field(default_factory=dict)

    def add(self, member: MergeClusterMember, score: float) -> None:
        # If a member already appears (it can if it's both a seed and a
        # neighbor of another seed) keep the highest cosine score.
        existing = self.cosine_scores.get(member.memory_id, 0.0)
        if score > existing:
            self.members[member.memory_id] = member
            self.cosine_scores[member.memory_id] = score


# ---------------------------------------------------------------------------
# Seed loader (extracted so tests can mock).
# ---------------------------------------------------------------------------


async def _load_seed_rows(
    *,
    env_id: UUID,
    cutoff: dt.datetime,
    cap: int,
) -> list[_SeedRow]:
    """Load active rows in env updated since ``cutoff`` ordered by recency.

    Caps at ``cap`` × small constant so the seed list reflects the busiest
    region of the env without unbounded scan; the union-find clustering
    naturally absorbs less-recently-updated rows that show up as
    neighbors of an in-window seed.
    """

    async with session_scope() as s:
        stmt = (
            select(
                Memory.id,
                Memory.title,
                Memory.body,
                Memory.salience,
                Memory.kind,
                Memory.created_at,
            )
            .where(
                and_(
                    Memory.env_id == env_id,
                    Memory.status == MemoryStatus.active.value,
                    Memory.updated_at >= cutoff,
                )
            )
            .order_by(Memory.updated_at.desc())
            .limit(cap)
        )
        rows = (await s.execute(stmt)).all()

    return [
        _SeedRow(
            id=r[0],
            title=r[1],
            body=r[2],
            salience=float(r[3]),
            kind=str(r[4]),
            created_at=r[5],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embed_text_sync(embedder: Embedder, text: str) -> list[float]:
    """Embed a single text. Local embedder is sync; wrap once here."""
    return embedder.embed_texts([text])[0]


def _payload_to_member(
    *,
    memory_id: UUID,
    payload: Mapping[str, Any],
    fallback_created_at: dt.datetime,
) -> MergeClusterMember:
    """Project a Qdrant payload into a :class:`MergeClusterMember`.

    Qdrant stores the per-memory payload assembled by
    :func:`memory_mcp._projection_payload` (see ``memories.py``), which
    contains everything we need without a Postgres round-trip.
    """

    created_at_raw = payload.get("created_at")
    created_at: dt.datetime
    if isinstance(created_at_raw, str):
        try:
            created_at = dt.datetime.fromisoformat(created_at_raw)
        except ValueError:
            created_at = fallback_created_at
    else:
        created_at = fallback_created_at

    return MergeClusterMember(
        memory_id=memory_id,
        title=payload.get("title"),
        body=str(payload.get("body", "")),
        salience=float(payload.get("salience", 0.0)),
        created_at=created_at,
    )


def _build_dedupe_key(member_ids: Sequence[UUID]) -> str:
    """Stable canonical key for a cluster — sorted UUIDs joined.

    The partial unique index on ``dream_proposals(env_id, kind, dedupe_key)``
    where ``status='open'`` blocks duplicate open proposals over the same
    set. Sorting ensures a cluster of {A, B, C} produces the same key
    regardless of which member was the seed.
    """

    return "merge:" + ":".join(sorted(str(m) for m in member_ids))


def _select_primary_id(members: dict[UUID, MergeClusterMember]) -> UUID:
    """Pick the canonical "winning" memory — highest salience, lex tiebreak.

    Reviewers typically want the most-trusted member to become the
    survivor of a merge. Lexically-lowest UUID is a deterministic
    tiebreaker so the same cluster always nominates the same primary
    across runs.
    """

    best = next(iter(members.values()))
    for m in members.values():
        if m.salience > best.salience or (
            m.salience == best.salience and str(m.memory_id) < str(best.memory_id)
        ):
            best = m
    return best.memory_id


def _materialize_cluster(
    cluster: _Cluster,
) -> tuple[MergeCluster, list[UUID]]:
    """Convert a working cluster to the immutable ``MergeCluster`` shape.

    Returns ``(MergeCluster, sorted_member_ids)`` — the second tuple
    element drives ``dedupe_key`` construction.
    """
    sorted_ids = sorted(cluster.members.keys(), key=str)
    members_list = [cluster.members[mid] for mid in sorted_ids]
    scores_list = [cluster.cosine_scores[mid] for mid in sorted_ids]
    primary_id = _select_primary_id(cluster.members)
    return (
        MergeCluster(
            primary_id=primary_id,
            members=members_list,
            cosine_scores=scores_list,
        ),
        sorted_ids,
    )


async def _open_proposal_exists(*, env_id: UUID, dedupe_key: str) -> bool:
    """Cheap pre-summarizer existence check.

    Returns ``True`` iff an open ``merge_candidate`` proposal already
    covers this exact member set. Used to short-circuit before paying
    summarizer cost on repeat runs — see rubber-duck finding #2 from the
    p2.2-dedupe review.
    """

    async with session_scope() as s:
        stmt = select(
            exists().where(
                and_(
                    DreamProposal.env_id == env_id,
                    DreamProposal.kind == DreamProposalKind.merge_candidate.value,
                    DreamProposal.status == DreamProposalStatus.open.value,
                    DreamProposal.dedupe_key == dedupe_key,
                )
            )
        )
        return bool((await s.execute(stmt)).scalar())


async def _insert_proposal(
    *,
    env_id: UUID,
    dream_run_id: UUID | None,
    cluster: MergeCluster,
    sorted_member_ids: list[UUID],
    summary: MergeSummary,
) -> bool:
    """Insert a single ``merge_candidate`` proposal.

    Returns ``True`` on insert; ``False`` if the partial unique index
    rejected as duplicate (idempotency hit). Uses ``INSERT ... ON
    CONFLICT DO NOTHING`` against the partial unique index so we never
    have to parse driver-specific ``IntegrityError`` text and so the
    transaction is never left in an aborted state (rubber-duck finding
    #1 from the p2.2-dedupe review).
    """

    dedupe_key = _build_dedupe_key(sorted_member_ids)
    # API contract (see ``dream_api._accept_merge``): ``candidate_ids`` is
    # the set of to-be-superseded members and MUST NOT include
    # ``primary_id``. Cluster member ids include the primary, so we filter
    # it out here. Cosine scores are kept aligned with ``sorted_member_ids``
    # (the primary's self-similarity stays in the array for debugging).
    candidate_ids = [
        str(mid) for mid in sorted_member_ids if mid != cluster.primary_id
    ]
    payload: dict[str, Any] = {
        "primary_id": str(cluster.primary_id),
        "candidate_ids": candidate_ids,
        "cosine_scores": list(cluster.cosine_scores),
        "suggested_merged_title": summary.suggested_merged_title,
        "suggested_merged_body": summary.suggested_merged_body,
        "summarizer_kind": summary.summarizer_kind.value,
        "llm_failed": summary.llm_failed,
        "llm_model_id": summary.llm_model_id,
    }

    async with session_scope() as s:
        stmt = (
            pg_insert(DreamProposal)
            .values(
                id=uuidlib.uuid4(),
                env_id=env_id,
                kind=DreamProposalKind.merge_candidate.value,
                status=DreamProposalStatus.open.value,
                payload=payload,
                summarizer_kind=summary.summarizer_kind.value,
                llm_failed=summary.llm_failed,
                dedupe_key=dedupe_key,
                dream_run_id=dream_run_id,
            )
            # ON CONFLICT DO NOTHING against the partial unique index —
            # the index name is referenced explicitly so a future schema
            # change that adds another unique constraint won't silently
            # short-circuit a different collision.
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
            "dedupe: open proposal already exists for cluster "
            "(env %s, dedupe_key %s)", env_id, dedupe_key,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run_dedupe(
    env_id: UUID,
    *,
    qdrant: _SearchableVectorStore,
    embedder: Embedder,
    summarizer: DreamSummarizer,
    settings: Settings | None = None,
    now: dt.datetime | None = None,
    dream_run_id: UUID | None = None,
) -> DedupePassResult:
    """Run one dedupe pass for a single env.

    Args:
        env_id: scope. One pass per env so a noisy env doesn't slow
            down a quiet one.
        qdrant: vector store with ``search`` capability. The runner
            shares the singleton from app state.
        embedder: text embedder. Singleton from app state.
        summarizer: produces ``MergeSummary`` per cluster. Constructed
            once per worker process by the runner; passes never build
            their own.
        settings: caller may override for tests.
        now: caller may override for tests.
        dream_run_id: id of the ``dream_runs`` row that initiated this
            pass; recorded on each proposal so observability can join.

    Returns:
        :class:`DedupePassResult` with per-stage counters.
    """

    settings = settings or get_settings()
    now = now or dt.datetime.now(dt.UTC)
    cutoff = now - dt.timedelta(days=settings.dream_dedupe_window_days)
    threshold = settings.dream_dedupe_threshold
    top_k = settings.dream_dedupe_top_k
    cap = settings.dream_dedupe_batch_cap

    started = time.perf_counter()

    # Seeds: most-recently-updated active memories. Generous over-fetch
    # (3x cap) so we still find clusters when many seeds happen to fall
    # into the same cluster (which would only emit one proposal).
    seeds = await _load_seed_rows(env_id=env_id, cutoff=cutoff, cap=cap * 3)

    # Track which (sorted member-id tuple) we've already proposed inside
    # *this* run — bare DB uniqueness blocks duplicates across runs but
    # not within a single run, since the cap counter would inflate.
    emitted_keys: set[str] = set()

    proposals_emitted = 0
    proposals_skipped_existing = 0
    proposals_skipped_below_min_size = 0
    clusters_found = 0
    seeds_examined = 0
    items_capped = False

    for seed in seeds:
        if proposals_emitted >= cap:
            items_capped = True
            break
        seeds_examined += 1

        cluster = await _build_cluster_for_seed(
            seed=seed,
            env_id=env_id,
            qdrant=qdrant,
            embedder=embedder,
            top_k=top_k,
            threshold=threshold,
        )

        if cluster is None or len(cluster.members) < 2:
            proposals_skipped_below_min_size += 1
            continue

        clusters_found += 1
        merge_cluster, sorted_ids = _materialize_cluster(cluster)
        dedupe_key = _build_dedupe_key(sorted_ids)
        if dedupe_key in emitted_keys:
            # Two seeds in the same cluster — already emitted via the
            # earlier seed.
            continue

        # Cross-run idempotency: if an open proposal already covers this
        # exact set, skip BEFORE paying summarizer cost (which can be an
        # expensive LLM call). The DB ON CONFLICT below still races-safely
        # absorbs the rare case where two workers emit the same cluster
        # within microseconds.
        if await _open_proposal_exists(env_id=env_id, dedupe_key=dedupe_key):
            proposals_skipped_existing += 1
            emitted_keys.add(dedupe_key)
            continue

        summary = await _instrumented_summarize_merge(summarizer, merge_cluster)

        inserted = await _insert_proposal(
            env_id=env_id,
            dream_run_id=dream_run_id,
            cluster=merge_cluster,
            sorted_member_ids=sorted_ids,
            summary=summary,
        )
        if inserted:
            proposals_emitted += 1
            emitted_keys.add(dedupe_key)
        else:
            proposals_skipped_existing += 1
            emitted_keys.add(dedupe_key)

    return DedupePassResult(
        env_id=env_id,
        seeds_examined=seeds_examined,
        clusters_found=clusters_found,
        proposals_emitted=proposals_emitted,
        proposals_skipped_existing=proposals_skipped_existing,
        proposals_skipped_below_min_size=proposals_skipped_below_min_size,
        items_capped=items_capped,
        summarizer_kind=summarizer.kind.value,
        duration_seconds=time.perf_counter() - started,
    )


async def _build_cluster_for_seed(
    *,
    seed: _SeedRow,
    env_id: UUID,
    qdrant: _SearchableVectorStore,
    embedder: Embedder,
    top_k: int,
    threshold: float,
) -> _Cluster | None:
    """Embed the seed, query Qdrant, return the cluster if any neighbors hit.

    Returns ``None`` if no neighbors above threshold (so no cluster is
    formed). Otherwise returns a populated cluster including the seed.
    """

    text = _seed_text(seed)
    if not text:
        return None
    # ``embed_texts`` may be CPU-heavy (local sentence-transformers).
    # Off-load to a worker thread so the dream-worker event loop can
    # process the rest of the seed list while the embedding runs.
    vector = await asyncio.to_thread(_embed_text_sync, embedder, text)

    hits = await qdrant.search(
        env_id=env_id,
        query_vector=vector,
        # ``top_k + 1`` so the seed's own point (returned at score 1.0)
        # doesn't consume a slot that should hold a real neighbor —
        # rubber-duck finding #8 from the p2.2-dedupe review.
        limit=top_k + 1,
        filters={
            "kind": seed.kind,
            "status": MemoryStatus.active.value,
        },
    )

    cluster = _Cluster()
    cluster.add(
        MergeClusterMember(
            memory_id=seed.id,
            title=seed.title,
            body=seed.body,
            salience=seed.salience,
            created_at=seed.created_at,
        ),
        score=1.0,
    )
    for hit in hits:
        try:
            hit_id = UUID(hit["id"])
        except (KeyError, ValueError, TypeError):
            log.warning("dedupe: malformed qdrant hit id %r; skipping", hit.get("id"))
            continue
        if hit_id == seed.id:
            continue
        score = float(hit.get("score", 0.0))
        if score < threshold:
            # Hits come back sorted descending — first below-threshold
            # hit means we're done.
            break
        member = _payload_to_member(
            memory_id=hit_id,
            payload=hit.get("payload", {}),
            fallback_created_at=seed.created_at,
        )
        cluster.add(member, score)

    if len(cluster.members) < 2:
        return None
    return cluster


def _seed_text(seed: _SeedRow) -> str:
    """Compose the embedding input for a seed.

    Mirrors what ``memories.py`` uses when first projecting the memory
    so the seed's neighbor query lands on the same vector geometry as
    the original write.
    """
    if seed.title:
        return f"{seed.title}\n\n{seed.body}".strip()
    return seed.body.strip()


async def _instrumented_summarize_merge(
    summarizer: Any,
    cluster: MergeCluster,
) -> MergeSummary:
    """Wrap ``summarize_merge`` with Prometheus instrumentation.

    Records call count + latency labelled by summarizer kind. When the
    LLM summarizer falls back to template content (``llm_failed=True``)
    increments ``dream_llm_fallbacks_total{pass="dedupe"}``. Observability
    failures never poison the pass — wrapped in try/except.
    """
    started = time.perf_counter()
    kind_label = summarizer.kind.value
    outcome = "ok"
    try:
        summary = await summarizer.summarize_merge(cluster)
    except Exception:
        outcome = "error"
        try:
            from memory_mcp.observability import (
                dream_summarizer_calls_total,
                dream_summarizer_latency_seconds,
            )
            dream_summarizer_calls_total.labels(
                kind=kind_label, outcome=outcome,
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
            dream_llm_fallbacks_total.labels(**{"pass": "dedupe"}).inc()
        dream_summarizer_calls_total.labels(
            kind=kind_label, outcome=outcome,
        ).inc()
    except Exception:  # noqa: BLE001
        pass
    return summary


__all__ = [
    "DedupePassResult",
    "run_dedupe",
]
