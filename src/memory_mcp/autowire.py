"""Phase 4 auto-wire — bounded ``related_to_popular`` edges on ``mem_compose``.

Two-stage design (see plan Stage D, RD-D blocker):

* **Stage A** — read-only, executed *before* the compose transaction opens.
  Embeds the new memory body, queries Qdrant for top-K semantic neighbours
  in the same env, intersects with Postgres top-by-salience candidates,
  excludes the source set's lineage ancestors. Returns at most
  ``settings.autowire_top_k`` ``(memory_id, combined_score)`` tuples. Any
  embedder / vector-store failure degrades to ``[]`` — auto-wire never
  blocks compose.

* **Stage B** — runs inside ``_compose_in_session`` just before the response
  builder. Resolves ``graph_nodes`` for the new memory + each candidate,
  inserts one ``relations`` row per pair with ``ON CONFLICT DO NOTHING``
  on the ``(src_node_id, dst_node_id, type)`` unique constraint, emits
  one audit row + one outbox event per inserted edge. Returns the dst
  ``memory_id`` list in deterministic order.

On replay (idempotency hit or savepoint race-loss), Stage B is skipped.
``reconstruct_auto_wired`` re-queries the live ``relations`` rows for the
replayed memory so the response stays consistent with the persisted
state. Per RD-D blocker resolution this is **state-current, not
operation-exact** — a manual ``rel_link`` of ``related_to_popular``
between the same nodes would surface in the replay output. Documented
on ``MemComposeResponse.auto_wired``.

The predicate ``related_to_popular`` is excluded from the
popularity-trigger whitelist (migration 0017 lines 180-183, 213-215),
the recount canonical writer (``dream/passes/recount.py:172``), and the
velocity windows (``top.py:260, 277``). Auto-wired edges therefore never
amplify the popularity score of their dst memory — see Stage D §D-pre
for the regression coverage.

Decompose auto-wire is deferred to v0.16 because
``MemDecomposeResponse.auto_wired: list[UUID]`` cannot disambiguate
per-child mapping; widening the schema is a breaking change.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import GraphNode, Memory, Relation
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.db.types import MemoryKind, OutboxAggregateType, OutboxOp
from memory_mcp.embeddings.base import Embedder, get_embedder
from memory_mcp.errors import EmbeddingModelMismatchError
from memory_mcp.identity import AgentContext
from memory_mcp.relations import _ensure_graph_node, _record_relation_audit
from memory_mcp_schemas.relations import RelationEndpoint

log = logging.getLogger(__name__)

AUTO_WIRE_PREDICATE = "related_to_popular"
"""The relation ``type`` value emitted by Phase 4 auto-wire.

Excluded from popularity-counter triggers, recount, and velocity. See
``migrations/0017_popularity_counters.py:180-183, 213-215``,
``dream/passes/recount.py:172``, ``top.py:260, 277``.
"""

# Skip kinds (set of MemoryKind enum *values*). Playbooks are templates,
# not standalone content — auto-wire would create misleading "similar"
# edges to them. Conservative v1 list per RD-D resolution.
_SKIP_KINDS: frozenset[str] = frozenset({MemoryKind.playbook.value})

# Tag-prefix skip: any tag starting with this prefix marks the memory as
# an active directive. Directives are tag-based (per workspace policy
# §17.5), not a MemoryKind.
_SKIP_TAG_PREFIX = "directive:active"


__all__ = [
    "AUTO_WIRE_PREDICATE",
    "autowire_fetch_candidates",
    "autowire_compose_target",
    "reconstruct_auto_wired",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_vector_store() -> Any:
    """Local copy of :func:`memory_mcp.memories._default_vector_store`.

    Duplicated rather than imported to keep autowire's surface narrow
    and avoid an autowire→memories edge. Three lines; cheap.
    """
    from memory_mcp.db.vector.qdrant import QdrantVectorStore

    return QdrantVectorStore(get_settings())


def _should_skip_target(
    *,
    kind: MemoryKind | str,
    tags: list[str] | None,
    body: str,
) -> bool:
    """Early-skip filter for both Stage A and Stage B.

    Skips when: kind is in ``_SKIP_KINDS``, any tag starts with
    ``directive:active``, or body is empty / whitespace-only.
    """
    if not body or not body.strip():
        return True
    kind_value = kind.value if isinstance(kind, MemoryKind) else str(kind)
    if kind_value in _SKIP_KINDS:
        return True
    if tags:
        for t in tags:
            if t.startswith(_SKIP_TAG_PREFIX):
                return True
    return False


# ---------------------------------------------------------------------------
# Stage A — read-only candidate fetch (pre-txn)
# ---------------------------------------------------------------------------


async def autowire_fetch_candidates(
    *,
    s: AsyncSession,
    env_id: UUID,
    source_ids: list[UUID] | tuple[UUID, ...],
    body: str,
    new_kind: MemoryKind | str,
    new_tags: list[str] | None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    vector_store: Any | None = None,
) -> list[tuple[UUID, float]]:
    """Read-only Stage A: pick up to ``top_k`` popular semantic neighbours.

    Algorithm:

    * A1 — early skip on kind / tag / empty body.
    * A2 — pull ``candidate_limit`` top-by-salience active candidates
      from Postgres (excluding ``self`` and skip-listed kinds).
    * A3 — exclude any candidate that is a lineage ancestor of any
      ``source_ids`` member (recursive CTE, depth ≤ 20).
    * A4 — embed ``body`` off-thread; mismatch / failure → ``[]``.
    * A5 — Qdrant similarity search (``vector_name="body"``,
      ``filters={"status": ["active"]}``); failure → ``[]``.
    * A6 — combine: ``combined = salience * sim_score`` for ids present
      in both sets with ``sim_score >= autowire_sim_threshold``; sort
      ``(combined DESC, id DESC)``; take top-K.

    Returns
    -------
    list of ``(memory_id, combined_score)``, ordered most-relevant first.
    Empty list when feature OFF, body unsuitable, or any external
    dependency fails.
    """
    settings = settings or get_settings()

    if not settings.autowire_enabled:
        return []

    if _should_skip_target(kind=new_kind, tags=new_tags, body=body):
        return []

    top_k = int(settings.autowire_top_k)
    candidate_limit = int(settings.autowire_candidate_limit)
    sim_threshold = float(settings.autowire_sim_threshold)

    # A2 — top-by-salience candidates from Postgres. We exclude only
    # explicit self-references via NOT IN on source_ids (sources are
    # not the target memory but we still don't want to wire to them).
    # The new memory's row is committed *after* this query in Stage A
    # contexts, so id != :new_id is not needed here.
    skip_kind_values = list(_SKIP_KINDS)
    rows = (
        await s.execute(
            select(Memory.id, Memory.salience).where(
                Memory.env_id == env_id,
                Memory.status == "active",
                Memory.kind.not_in(skip_kind_values),
            ).order_by(
                Memory.salience.desc(),
                Memory.created_at.desc(),
                Memory.id.desc(),
            ).limit(candidate_limit)
        )
    ).all()
    if not rows:
        return []

    pg_candidates: dict[UUID, float] = {row[0]: float(row[1] or 0.0) for row in rows}

    # Always exclude source ids themselves.
    for sid in source_ids:
        pg_candidates.pop(sid, None)
    if not pg_candidates:
        return []

    # A3 — exclude lineage ancestors of the source set (depth ≤ 20).
    if source_ids:
        excluded = await _collect_lineage_ancestors(s, list(source_ids))
        for ancestor_id in excluded:
            pg_candidates.pop(ancestor_id, None)
    if not pg_candidates:
        return []

    # A4 — embed body.
    embedder = embedder or get_embedder(settings)
    try:
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(None, embedder.embed_texts, [body])
    except EmbeddingModelMismatchError as exc:
        log.warning("autowire: embedding model mismatch (%s); skipping", exc)
        return []
    except Exception as exc:  # noqa: BLE001 — degrade silently
        log.warning("autowire: embedder failure (%s); skipping", exc)
        return []
    if not vectors or not vectors[0]:
        return []
    qvec = vectors[0]

    # A5 — Qdrant similarity. Pull more than top_k to leave room for the
    # PG-intersection + threshold cut.
    vector_store = vector_store or _default_vector_store()
    try:
        results = await vector_store.search(
            env_id=env_id,
            query_vector=qvec,
            limit=max(candidate_limit, top_k * 2),
            filters={"status": ["active"]},
            vector_name="body",
        )
    except Exception as exc:  # noqa: BLE001 — degrade silently
        log.warning("autowire: vector_store.search failure (%s); skipping", exc)
        return []

    sim_scores: dict[UUID, float] = {}
    for hit in results or []:
        try:
            mid = UUID(str(hit["id"]))
            score = float(hit["score"])
        except (KeyError, ValueError, TypeError):
            continue
        if score >= sim_threshold:
            # Keep the highest score per id (Qdrant returns one row each
            # but defensive against future fan-out).
            if score > sim_scores.get(mid, float("-inf")):
                sim_scores[mid] = score

    # A6 — combine + rank.
    combined: list[tuple[UUID, float]] = []
    for mid, salience in pg_candidates.items():
        sim = sim_scores.get(mid)
        if sim is None:
            continue
        combined.append((mid, salience * sim))

    combined.sort(key=lambda x: (-x[1], -int(x[0].int)))
    top = combined[:top_k]

    log.debug(
        "autowire candidates: k_pg=%d k_sem=%d k_combined=%d k_returned=%d",
        len(pg_candidates), len(sim_scores), len(combined), len(top),
    )
    return top


async def _collect_lineage_ancestors(
    s: AsyncSession,
    seeds: list[UUID],
    max_depth: int = 20,
) -> set[UUID]:
    """Recursive CTE returning all ancestors of ``seeds`` plus the seeds.

    Walks ``memory_lineage`` parent_memory_id → child relationships (we
    treat parent as ancestor of child). Depth-capped at 20 — lineage
    chains are shallow by design.
    """
    if not seeds:
        return set()
    cte_sql = text(
        """
        WITH RECURSIVE ancestors(id, depth) AS (
            SELECT id, 0 FROM unnest(:seeds ::uuid[]) AS id
            UNION
            SELECT ml.parent_memory_id, a.depth + 1
              FROM memory_lineage ml
              JOIN ancestors a ON ml.child_memory_id = a.id
             WHERE a.depth < :max_depth
        )
        SELECT DISTINCT id FROM ancestors
        """
    )
    result = await s.execute(
        cte_sql.bindparams(seeds=[str(u) for u in seeds], max_depth=max_depth)
    )
    return {row[0] for row in result.all()}


# ---------------------------------------------------------------------------
# Stage B — in-transaction insert
# ---------------------------------------------------------------------------


async def autowire_compose_target(
    *,
    s: AsyncSession,
    new_memory_id: UUID,
    new_memory_kind: MemoryKind | str,
    new_memory_tags: list[str] | None,
    new_memory_body: str,
    new_memory_env_id: UUID,
    candidates: list[tuple[UUID, float]],
    ctx: AgentContext,
    settings: Settings | None = None,
) -> list[UUID]:
    """In-transaction Stage B: insert relations + audit + outbox.

    ``candidates`` comes from :func:`autowire_fetch_candidates`. The
    caller is responsible for the read-only pre-fetch outside the lock
    window; this function only does graph-node resolution, INSERTs with
    ``ON CONFLICT DO NOTHING`` on the relations unique constraint, and
    enqueues outbox + audit rows for actually-inserted edges.

    Returns the inserted dst ``memory_id`` list in deterministic order
    (sorted by ``dst_node_id`` ascending — the order the INSERT statements
    fired in, which is the same order we'd recover via replay).
    """
    settings = settings or get_settings()

    if not settings.autowire_enabled:
        return []
    if not candidates:
        return []

    # Re-apply skip filter at Stage B too: defensive against caller
    # threading the wrong inputs and against pre-computed candidates
    # surviving across a feature-flag flip.
    if _should_skip_target(
        kind=new_memory_kind, tags=new_memory_tags, body=new_memory_body
    ):
        return []

    # Resolve src graph_node once.
    try:
        src_node = await _ensure_graph_node(
            s,
            env_id=new_memory_env_id,
            endpoint=RelationEndpoint(kind="memory", id=new_memory_id),
        )
    except Exception as exc:  # noqa: BLE001 — never block compose on autowire
        log.warning(
            "autowire: failed to resolve src graph_node for %s (%s); skipping",
            new_memory_id, exc,
        )
        return []

    # Resolve dst graph_nodes; build (combined_score, dst_node_id,
    # dst_memory_id, salience_at_link, sim_score) tuples. We've lost
    # the per-pair salience+sim split because Stage A returns the
    # product; persist that as one payload field for forensics.
    resolved: list[tuple[UUID, UUID, float]] = []
    for dst_memory_id, combined_score in candidates:
        try:
            dst_node = await _ensure_graph_node(
                s,
                env_id=new_memory_env_id,
                endpoint=RelationEndpoint(kind="memory", id=dst_memory_id),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "autowire: failed to resolve dst graph_node for %s (%s); skipping edge",
                dst_memory_id, exc,
            )
            continue
        resolved.append((dst_node.id, dst_memory_id, combined_score))

    if not resolved:
        return []

    # Sort by dst_node_id ascending for deterministic lock-acquisition
    # order across concurrent compose calls (deadlock safety).
    resolved.sort(key=lambda x: int(x[0].int))

    inserted: list[UUID] = []
    for dst_node_id, dst_memory_id, combined_score in resolved:
        # Skip self-loop defensively (should never happen because Stage A
        # excludes the new memory id, but the dst graph_node could in
        # principle equal src if the candidate computation is fed a stale
        # source set — guard at the cheapest layer).
        if dst_node_id == src_node.id:
            continue

        payload = {
            "predicate": AUTO_WIRE_PREDICATE,
            "combined_score": combined_score,
        }
        properties_jsonb = payload

        # INSERT ... ON CONFLICT DO NOTHING RETURNING id. Using raw SQL
        # for the conflict clause; sqlalchemy's ON CONFLICT helpers exist
        # but raw text keeps the constraint name explicit.
        insert_sql = text(
            """
            INSERT INTO relations
                (id, env_id, src_node_id, dst_node_id, type, properties,
                 created_at, updated_at, version)
            VALUES
                (gen_random_uuid(), CAST(:env_id AS uuid),
                 CAST(:src_node_id AS uuid), CAST(:dst_node_id AS uuid),
                 :rel_type, CAST(:properties AS jsonb), now(), now(), 1)
            ON CONFLICT (src_node_id, dst_node_id, type) DO NOTHING
            RETURNING id
            """
        )
        import json as _json
        row = (
            await s.execute(
                insert_sql.bindparams(
                    env_id=str(new_memory_env_id),
                    src_node_id=str(src_node.id),
                    dst_node_id=str(dst_node_id),
                    rel_type=AUTO_WIRE_PREDICATE,
                    properties=_json.dumps(properties_jsonb),
                )
            )
        ).first()

        if row is None:
            # Conflict — relation already exists. Skip outbox + audit so we
            # don't double-count; replay will surface it via state-current
            # reconstruction.
            continue

        relation_id = row[0]
        if isinstance(relation_id, str):
            relation_id = UUID(relation_id)

        # Refresh the inserted row so we have version + timestamps for
        # the outbox payload.
        relation = (
            await s.execute(
                select(Relation).where(Relation.id == relation_id)
            )
        ).scalar_one()

        outbox_payload = {
            "relation_id": str(relation.id),
            "env_id": str(relation.env_id),
            "type": relation.type,
            "properties": dict(relation.properties or {}),
            "src": {
                "kind": "memory",
                "id": str(new_memory_id),
                "node_id": str(src_node.id),
            },
            "dst": {
                "kind": "memory",
                "id": str(dst_memory_id),
                "node_id": str(dst_node_id),
            },
            "version": relation.version,
            "created_at": (
                relation.created_at.isoformat() if relation.created_at else None
            ),
            "updated_at": (
                relation.updated_at.isoformat() if relation.updated_at else None
            ),
            "auto_wire": True,
        }

        await _record_relation_audit(
            s,
            op=f"auto_wire:{AUTO_WIRE_PREDICATE}",
            relation_id=relation.id,
            env_id=new_memory_env_id,
            by_agent_id=ctx.agent_id,
            before=None,
            after=outbox_payload,
        )

        await enqueue_event(
            s,
            aggregate_type=OutboxAggregateType.relation,
            aggregate_id=relation.id,
            aggregate_version=relation.version,
            env_id=new_memory_env_id,
            op=OutboxOp.upsert,
            payload=outbox_payload,
            settings=settings,
        )

        inserted.append(dst_memory_id)

    return inserted


# ---------------------------------------------------------------------------
# Replay reconstruction
# ---------------------------------------------------------------------------


async def reconstruct_auto_wired(
    *,
    s: AsyncSession,
    memory_id: UUID,
) -> list[UUID]:
    """Re-query live ``related_to_popular`` edges for ``memory_id``.

    Used by the compose response builder on the replay path (dedupe
    hit or savepoint race-loss recovery). Returns dst ``memory_id``
    values from ``graph_nodes`` joined through the ``relations`` table.

    State-current semantics — see module docstring + the
    ``MemComposeResponse.auto_wired`` field docstring. A manually-added
    ``related_to_popular`` edge between the same nodes will appear here.
    """
    query = text(
        """
        SELECT gn_dst.memory_id
          FROM relations r
          JOIN graph_nodes gn_src ON r.src_node_id = gn_src.id
          JOIN graph_nodes gn_dst ON r.dst_node_id = gn_dst.id
         WHERE gn_src.memory_id = CAST(:memory_id AS uuid)
           AND r.type = :rel_type
         ORDER BY gn_dst.id ASC
        """
    )
    result = await s.execute(
        query.bindparams(memory_id=str(memory_id), rel_type=AUTO_WIRE_PREDICATE)
    )
    out: list[UUID] = []
    for row in result.all():
        val = row[0]
        if val is None:
            continue
        if isinstance(val, str):
            val = UUID(val)
        out.append(val)
    return out
