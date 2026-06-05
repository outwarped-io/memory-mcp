"""Trigger-conditioned auto-context retrieval."""

from __future__ import annotations

from uuid import UUID

from memory_mcp_schemas.search import (
    AutoContextHit,
    AutoContextResponse,
)
from sqlalchemy import select

from memory_mcp._filters import exclude_expired_clause
from memory_mcp.config import Settings
from memory_mcp.db.models import Memory
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.embeddings.base import Embedder
from memory_mcp.errors import InvalidInputError
from memory_mcp.search.api import _search_by_trigger


async def memory_auto_context(
    *,
    task_desc: str,
    env_id: UUID,
    top_k: int = 8,
    settings: Settings | None = None,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> AutoContextResponse:
    """Return memories whose authoring trigger semantically matches a task."""
    task_desc_used = task_desc.strip()
    if not task_desc_used:
        raise InvalidInputError("INVALID_INPUT: mem_auto_context task_desc cannot be empty")
    if top_k < 1:
        raise InvalidInputError("INVALID_INPUT: mem_auto_context top_k must be at least 1")
    top_k = min(top_k, 50)

    ranked_ids = await _search_by_trigger(
        task_desc_used,
        env_id,
        top_k,
        settings=settings,
        vector_store=vector_store,
        embedder=embedder,
    )
    if not ranked_ids:
        return AutoContextResponse(hits=[], task_desc_used=task_desc_used)

    ids = [memory_id for memory_id, _score in ranked_ids]
    scores = dict(ranked_ids)
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(Memory).where(
                        Memory.id.in_(ids),
                        Memory.env_id == env_id,
                        # v0.17 — convenience surface always default-excludes
                        # expired memories. No opt-out.
                        exclude_expired_clause(),
                    )
                )
            )
            .scalars()
            .all()
        )
    by_id = {m.id: m for m in rows}

    hits: list[AutoContextHit] = []
    for memory_id in ids:
        memory = by_id.get(memory_id)
        if memory is None or not memory.trigger_description:
            continue
        hits.append(
            AutoContextHit(
                memory_id=memory.id,
                title=memory.title or "",
                body=memory.body,
                trigger_description=memory.trigger_description,
                score=scores[memory.id],
                salience=float(memory.salience),
                kind=str(memory.kind),
            )
        )

    return AutoContextResponse(hits=hits, task_desc_used=task_desc_used)


__all__ = ["AutoContextHit", "AutoContextResponse", "memory_auto_context"]
