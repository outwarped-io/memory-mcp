"""Vector retrieval — ``mode=sem`` and the sem leg of ``mode=hybrid``.

Embeds the query with the env's default embedding model and queries the
per-env Qdrant collection. Per-env collections decouple model dimension
choices: each env may set its own ``default_embedding_model_id``.

If an env's model id differs from the configured embedder we raise
``EmbeddingModelMismatchError`` (rebuild required). Multiple envs are
queried in parallel — each gets its own embedding so that a future
multi-model deployment can fan out.

V1: identical model across envs (single embedder process), so we embed
ONCE and reuse the vector across all envs. The mismatch check still
runs per env.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp.db.models import Environment
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.embeddings.base import Embedder
from memory_mcp.errors import EmbeddingModelMismatchError, NotFoundError
from memory_mcp.search.ranking import RankedHit


async def _check_env_models(
    session: AsyncSession,
    env_ids: Sequence[UUID],
    embedder: Embedder,
) -> None:
    """Hard-fail if any env expects a different embedding model than ours."""
    rows = (await session.execute(
        select(Environment.id, Environment.default_embedding_model_id)
        .where(Environment.id.in_(list(env_ids)))
    )).all()
    found = {row[0]: row[1] for row in rows}
    missing = [e for e in env_ids if e not in found]
    if missing:
        raise NotFoundError(
            f"environments not found: {missing}", env_ids=[str(e) for e in missing],
        )
    for _env_id, model_id in found.items():
        if model_id != embedder.model_id:
            raise EmbeddingModelMismatchError(
                expected=str(model_id), actual=embedder.model_id,
            )


def _build_qdrant_filter(
    *,
    statuses: Sequence[str],
    kinds: Sequence[str] | None,
    tags: Sequence[str] | None,
    created_after: dt.datetime | None,
    created_before: dt.datetime | None,
    updated_after: dt.datetime | None,
) -> dict[str, Any]:
    flt: dict[str, Any] = {"status": list(statuses)}
    if kinds:
        flt["kind"] = list(kinds)
    if tags:
        flt["tags"] = list(tags)
    # Range filters are not modelled in the simple ``MatchAny`` path —
    # we apply them client-side after fetch (see api.py).
    return flt


async def sem_search(
    session: AsyncSession,
    *,
    vector_store: VectorStore,
    embedder: Embedder,
    query: str,
    env_ids: Sequence[UUID],
    statuses: Sequence[str],
    kinds: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    created_after: dt.datetime | None = None,
    created_before: dt.datetime | None = None,
    updated_after: dt.datetime | None = None,
    limit: int = 50,
) -> list[RankedHit]:
    """Vector retrieval across ``env_ids``. Returns 1-indexed ranked hits.

    Each env contributes its own top-K; results are merged and re-sorted
    by raw cosine score. The fusion stage (RRF) re-ranks against lex.
    """
    if not query.strip() or not env_ids:
        return []

    await _check_env_models(session, env_ids, embedder)

    import asyncio

    vectors = await asyncio.get_running_loop().run_in_executor(
        None, embedder.embed_texts, [query],
    )
    qvec = vectors[0]

    qfilter = _build_qdrant_filter(
        statuses=statuses,
        kinds=kinds,
        tags=tags,
        created_after=created_after,
        created_before=created_before,
        updated_after=updated_after,
    )

    # Fan out across envs.
    per_env = await asyncio.gather(
        *[
            vector_store.search(
                env_id=env_id,
                query_vector=qvec,
                limit=limit,
                filters=qfilter,
                vector_name="body",
            )
            for env_id in env_ids
        ]
    )

    merged: list[tuple[UUID, float]] = []
    for env_results in per_env:
        for hit in env_results:
            mid = UUID(hit["id"])
            merged.append((mid, float(hit["score"])))

    # Dedupe (keeps highest score) + descending sort by score.
    best: dict[UUID, float] = {}
    for mid, score in merged:
        if score > best.get(mid, float("-inf")):
            best[mid] = score
    ordered = sorted(best.items(), key=lambda x: x[1], reverse=True)[:limit]

    return [
        RankedHit(memory_id=mid, rank=i + 1, raw_score=score, source="sem")
        for i, (mid, score) in enumerate(ordered)
    ]


__all__ = ["sem_search"]
