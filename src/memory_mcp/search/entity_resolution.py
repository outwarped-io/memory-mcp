"""Resolve query mentions to canonical entity ids.

Used by the graph leg of ``mem_search`` to translate a list of normalized
strings (output of :func:`memory_mcp.search.ner.extract_query_mentions`)
into a per-env list of entity UUIDs that can be expanded through
``GraphStore.neighbors``.

Resolution sources, in order:

1. ``entities.normalized_name`` — exact canonical-name match.
2. ``entity_aliases.normalized_alias`` — exact alias match.

Both are exact-string lookups — no fuzzy / trigram matching in v1
(deferred to v1.x once entity volume warrants it). Multi-mention queries
with multiple matches per env produce a deduplicated list per env.

Hard caps (from settings) bound the resolved set so a query that
matches every entity in a large env can't flood the graph leg:

* ``graph_search_max_resolved_entities_per_env`` — per env cap.
* ``graph_search_max_resolved_entities_total`` — total cap across envs.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp.config import Settings
from memory_mcp.db.models import Entity, EntityAlias


async def resolve_query_entities(
    session: AsyncSession,
    *,
    mentions: Sequence[str],
    env_ids: Sequence[UUID],
    settings: Settings,
) -> dict[UUID, list[UUID]]:
    """Return ``{env_id: [entity_id, ...]}`` for matched mentions.

    Mentions are expected to be already normalized (lowercase, NFKC,
    stripped). Empty inputs return ``{}``.

    Order within each env's list is **deterministic** and follows the
    order in ``mentions`` — first match wins for the per-env cap, so a
    deliberate ordering on the caller side influences which entities
    survive truncation.
    """
    norms = [m for m in dict.fromkeys(mentions) if m]
    if not norms or not env_ids:
        return {}

    env_id_list = list(dict.fromkeys(env_ids))

    # Two parallel-shape queries — kept separate to preserve provenance
    # (canonical vs alias) for tie-breaking. SQLAlchemy `.in_()` handles
    # parameter binding safely (rubber-duck MAJOR 9).
    canonical_rows = (
        await session.execute(
            select(Entity.id, Entity.env_id, Entity.normalized_name).where(
                Entity.normalized_name.in_(norms),
                Entity.env_id.in_(env_id_list),
            )
        )
    ).all()
    alias_rows = (
        await session.execute(
            select(
                EntityAlias.entity_id,
                EntityAlias.env_id,
                EntityAlias.normalized_alias,
            ).where(
                EntityAlias.normalized_alias.in_(norms),
                EntityAlias.env_id.in_(env_id_list),
            )
        )
    ).all()

    # Build per-env, per-mention candidate map preserving mention order.
    # Insertion-ordered dicts give deterministic iteration in Python 3.7+.
    by_env_mention: dict[UUID, dict[str, list[UUID]]] = {eid: {m: [] for m in norms} for eid in env_id_list}
    for entity_id, env_id, norm_name in canonical_rows:
        by_env_mention[env_id][norm_name].append(entity_id)
    for entity_id, env_id, norm_alias in alias_rows:
        existing = by_env_mention[env_id][norm_alias]
        if entity_id not in existing:
            existing.append(entity_id)

    per_env_cap = settings.graph_search_max_resolved_entities_per_env
    total_cap = settings.graph_search_max_resolved_entities_total

    out: dict[UUID, list[UUID]] = {}
    total = 0
    # Walk envs in their input order, mentions in insertion order, then
    # candidates in DB-row order. Stop emitting per-env when the per-env
    # cap is hit; stop globally when the total cap is hit.
    for env_id in env_id_list:
        env_seen: set[UUID] = set()
        env_entities: list[UUID] = []
        for norm in norms:
            for entity_id in by_env_mention[env_id][norm]:
                if entity_id in env_seen:
                    continue
                env_seen.add(entity_id)
                env_entities.append(entity_id)
                total += 1
                if len(env_entities) >= per_env_cap or total >= total_cap:
                    break
            if len(env_entities) >= per_env_cap or total >= total_cap:
                break
        if env_entities:
            out[env_id] = env_entities
        if total >= total_cap:
            break

    return out


__all__ = ["resolve_query_entities"]
