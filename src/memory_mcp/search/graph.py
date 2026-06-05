"""Graph leg of ``mem_search`` — third leg alongside ``lex`` and ``sem``.

Pipeline
--------

1. **NER + identifier extraction** on the query
   (:func:`memory_mcp.search.ner.extract_query_mentions`).
2. **Entity resolution** against ``entities.normalized_name`` +
   ``entity_aliases.normalized_alias``
   (:func:`memory_mcp.search.entity_resolution.resolve_query_entities`).
3. **Graph expansion** — for each resolved ``(env_id, entity_id)``, call
   ``GraphStore.neighbors(entity_id, hops=1, kinds=["memory"], ...)``.
   Concurrent calls are bounded by an ``asyncio.Semaphore``.
4. **Scoring** — count distinct query-entities reaching each memory
   (overlap), break ties by min-path-length and the sum of reciprocal
   per-entity neighbor ranks.
5. **Emit** at most ``leg_limit`` ``RankedHit`` rows with
   ``source="graph"``. Lifecycle / kind / tag filters are applied
   post-fusion against canonical Postgres state by ``api.py``.

Empty-input handling
--------------------

Any of these short-circuits to an empty result list (graceful
degradation; hybrid then falls back to lex+sem):

* empty query
* spaCy unavailable AND no identifier-like tokens (e.g. natural-language
  question with only common words)
* no mention resolves to a canonical entity in any env
* graph store unavailable (caller decides whether to swallow or raise —
  see :func:`graph_search_or_raise` vs :func:`graph_search_best_effort`)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp.config import Settings
from memory_mcp.db.graph.base import GraphNodeRef, GraphStore
from memory_mcp.search.entity_resolution import resolve_query_entities
from memory_mcp.search.ner import extract_query_mentions
from memory_mcp.search.ranking import RankedHit

log = logging.getLogger(__name__)


# RRF damping constant used internally for our tie-breaking score.
# We reuse the same default that ``reciprocal_rank_fuse`` uses so the
# graph leg's internal ranking is consistent with downstream fusion.
_INTERNAL_K = 60


@dataclass
class _MemoryAggregate:
    """Per-memory aggregate built up across all ``neighbors`` calls."""

    contributing_entities: set[UUID] = field(default_factory=set)
    min_path_length: int = 1 << 30  # effectively +inf
    rank_score: float = 0.0
    first_order: int = 1 << 30


async def graph_search(
    session: AsyncSession,
    *,
    graph_store: GraphStore,
    query: str,
    env_ids: Sequence[UUID],
    limit: int,
    settings: Settings,
) -> list[RankedHit]:
    """Run the graph leg end-to-end. May raise backend errors."""
    if not query or not env_ids or limit <= 0:
        return []

    mentions = await extract_query_mentions(query, settings=settings)
    if not mentions:
        return []

    resolved = await resolve_query_entities(
        session,
        mentions=mentions,
        env_ids=env_ids,
        settings=settings,
    )
    if not resolved:
        return []

    # Per-entity over-fetch budget. Rubber-duck MAJOR 3: post-filtering
    # in api.py can drop hits, so we over-fetch up front to avoid
    # under-filling the leg. ``2 * limit`` mirrors the lex/sem leg's
    # over-fetch and is bounded by the ``graph_search_max_resolved_*``
    # caps applied during resolution.
    per_entity_limit = max(2 * limit, settings.search_min_per_leg)

    semaphore = asyncio.Semaphore(
        max(1, settings.graph_search_max_concurrent_neighbors),
    )

    # ``per_call`` records: which entity sourced each fetch and that
    # entity's resolution order, so we can reconstruct deterministic
    # tie-breaks when two entities both reach the same memory.
    per_call: list[tuple[UUID, UUID, int]] = []  # (env_id, entity_id, order)
    order = 0
    for env_id, entity_ids in resolved.items():
        for entity_id in entity_ids:
            per_call.append((env_id, entity_id, order))
            order += 1

    async def _fetch_one(
        env_id: UUID,
        entity_id: UUID,
        ent_order: int,
    ) -> list[tuple[UUID, UUID, int, int, int]]:
        """One ``neighbors`` call.

        Returns ``(memory_id, entity_id, ent_order, neighbor_rank, path_length)``
        per memory hit. ``neighbor_rank`` is 1-indexed within this entity's
        result page (deterministic per the ``GraphStore`` contract).
        """
        async with semaphore:
            hits, _cursor = await graph_store.neighbors(
                GraphNodeRef(env_id=env_id, kind="entity", record_id=entity_id),
                hops=settings.graph_search_hops,
                direction="both",
                kinds=["memory"],
                limit=per_entity_limit,
            )
        return [(hit.node.record_id, entity_id, ent_order, idx + 1, hit.path_length) for idx, hit in enumerate(hits)]

    # Run all fetches concurrently (bounded by semaphore).
    fetched_lists = await asyncio.gather(
        *[_fetch_one(env_id, entity_id, ent_order) for env_id, entity_id, ent_order in per_call]
    )

    # ---- aggregate per memory ------------------------------------------
    # For each memory id collect:
    #   - distinct contributing entity_ids   (overlap count for sort key)
    #   - min path_length                    (tighter paths preferred)
    #   - sum 1/(K + neighbor_rank)          (rank-based tie-breaker)
    #   - first observed ent_order           (deterministic last resort)
    agg: dict[UUID, _MemoryAggregate] = {}
    for rows in fetched_lists:
        for memory_id, entity_id, ent_order, neighbor_rank, path_length in rows:
            a = agg.get(memory_id)
            if a is None:
                a = _MemoryAggregate()
                agg[memory_id] = a
            a.contributing_entities.add(entity_id)
            if path_length < a.min_path_length:
                a.min_path_length = path_length
            a.rank_score += 1.0 / (_INTERNAL_K + neighbor_rank)
            if ent_order < a.first_order:
                a.first_order = ent_order

    if not agg:
        return []

    # ---- deterministic ordering ----------------------------------------
    # Primary: overlap count DESC. Ties: min_path_length ASC, rank_score
    # DESC, first_order ASC, memory_id (str) ASC. The final memory_id
    # tiebreak guarantees stable ordering across runs.
    ordered = sorted(
        agg.items(),
        key=lambda kv: (
            -len(kv[1].contributing_entities),
            kv[1].min_path_length,
            -kv[1].rank_score,
            kv[1].first_order,
            str(kv[0]),
        ),
    )[:limit]

    return [
        RankedHit(
            memory_id=memory_id,
            rank=i + 1,
            raw_score=float(len(a.contributing_entities)),
            source="graph",
        )
        for i, (memory_id, a) in enumerate(ordered)
    ]


__all__ = ["graph_search"]
