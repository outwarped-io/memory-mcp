"""``VectorStore`` protocol — backend-agnostic surface for vector projections.

Implemented by:

* :class:`memory_mcp.db.vector.qdrant.QdrantVectorStore` (default, v1)
* (future) ``PgvectorVectorStore`` — Postgres-only fallback for small deployments

The projection-worker calls into a ``VectorStore`` to upsert / delete memory
points; ``memory_search`` calls ``search`` for the semantic stage of hybrid
ranking.

Per-env collections
-------------------

Each environment owns its own collection (``memory-mcp-{env_id}``). This lets
operators rebuild a single env, change its embedding model, or purge it
without touching siblings. Collections are created lazily by
:meth:`ensure_env_collection` on the worker's first write.

Point ids
---------

Phase 1 uses ``memory_id`` (a UUID) as the point id directly. Each new
version of a memory **overwrites** the same point — per-aggregate ordering
is enforced at the lease level, so the LATEST version always wins under
healthy operation. Tombstones (status ∈ {archived, superseded, retired})
issue a ``delete`` on the same point id.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class VectorStore(Protocol):
    """Backend-agnostic vector store surface."""

    async def ensure_env_collection(
        self,
        *,
        env_id: UUID,
        dimension: int,
    ) -> None:
        """Idempotently ensure the per-env collection exists with ``dimension``.

        If the collection exists with a different dimension this is a hard
        error — operators must rebuild via the admin tool.
        """

    async def upsert(
        self,
        *,
        env_id: UUID,
        point_id: UUID,
        vector: Sequence[float] | Mapping[str, Sequence[float]],
        payload: Mapping[str, Any],
    ) -> None:
        """Upsert a single point (idempotent).

        ``payload`` is filterable metadata: env_id, kind, status, tags,
        embedding_model_id, version, created_at, updated_at, etc.
        """

    async def delete(
        self,
        *,
        env_id: UUID,
        point_id: UUID,
    ) -> None:
        """Idempotently remove a point (no-op if absent)."""

    async def search(
        self,
        *,
        env_id: UUID,
        query_vector: Sequence[float],
        limit: int,
        filters: Mapping[str, Any] | None = None,
        vector_name: str = "body",
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` matching points sorted by similarity.

        Each result is ``{"id": str, "score": float, "payload": {...}}``.
        ``vector_name`` selects the named vector (``body`` for normal memory
        search, ``trigger`` for auto-context). ``filters`` is interpreted by
        the backend (Qdrant ``must``-style filter for v1).
        """

    async def get_vector(self, *, env_id: UUID, id: str, vector_name: str = "body") -> list[float] | None:
        """Return the stored vector for ``id`` in ``env_id`` if available.

        Returns ``None`` when the point does not exist or was upserted without
        a vector. Callers MUST treat ``None`` as "no embedding available" and
        MUST NOT fall back to a fresh embed; this preserves the cost guarantee
        for ``mem_related semantic``.
        """

    async def get_vectors(
        self,
        *,
        env_id: UUID,
        ids: list[UUID],
        vector_name: str = "body",
    ) -> dict[UUID, list[float] | None]:
        """Return stored vectors for many ids in one backend call."""

    async def close(self) -> None:
        """Release any held resources (HTTP pools, gRPC channels, …)."""


__all__ = ["VectorStore"]
