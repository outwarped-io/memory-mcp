"""Re-embedding adapter for v0.8 environment operations."""

from __future__ import annotations

import asyncio
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp.config import get_settings
from memory_mcp.db.models import Memory
from memory_mcp.embeddings.base import get_embedder
from memory_mcp.errors import EmbeddingModelMismatchError
from memory_mcp.memories import _load_env_embedding_model


async def maybe_re_embed(
    memories: list[Memory],
    source_model_id: str,
    target_model_id: str,
    *,
    session: AsyncSession,
) -> dict[UUID, list[float]]:
    """Re-embed memory bodies when source and target env models differ."""

    del session
    if source_model_id == target_model_id:
        return {}

    embedder = get_embedder(get_settings())
    if embedder.model_id != target_model_id:
        raise EmbeddingModelMismatchError(expected=target_model_id, actual=embedder.model_id)

    vectors: dict[UUID, list[float]] = {}
    for memory in memories:
        embedded = await asyncio.to_thread(embedder.embed_texts, [memory.body])
        vectors[memory.id] = embedded[0]
    return vectors


async def load_env_embedding_model(env_id: UUID, *, session: AsyncSession) -> str:
    """Return the environment's default embedding model id."""

    return await _load_env_embedding_model(session, env_id)


__all__ = ["load_env_embedding_model", "maybe_re_embed"]
