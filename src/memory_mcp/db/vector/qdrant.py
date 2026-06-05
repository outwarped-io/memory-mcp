"""Qdrant-backed :class:`VectorStore` implementation.

Wraps :class:`qdrant_client.AsyncQdrantClient` with a small per-env
collection management layer. The collection name is
``memory-mcp-{env_id}`` (lowercased UUID) so admin-side ops can find it
deterministically.

Per-env collections
-------------------

Created lazily on first :meth:`ensure_env_collection`. We do NOT cache
"collection exists" forever in process memory — each env's first
event-of-the-restart re-runs the idempotent ``recreate_collection``
guarded by ``collection_exists``. Subsequent calls within the same
process memoize via ``self._known``.

Filter semantics
----------------

The default search filters are payload ``status`` IN [active, stale]
unless overridden. Worker-side payload contains ``status`` so search
can include / exclude superseded / archived rows without re-reading
canonical Postgres.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse
from sqlalchemy import select

from memory_mcp.config import Settings
from memory_mcp.db.models import Environment, Memory, MemoryTag, Tag
from memory_mcp.db.postgres import session_scope
from memory_mcp.embeddings.base import get_embedder

log = logging.getLogger(__name__)

_QDRANT_VISIBLE_STATUSES: tuple[str, ...] = ("proposed", "active", "stale")


def _collection_name(env_id: UUID) -> str:
    return f"memory-mcp-{env_id}"


def _embed_text(title: str | None, body: str) -> str:
    parts = [part for part in (title, body) if part]
    return "\n\n".join(str(part) for part in parts).strip()


class QdrantVectorStore:
    """Concrete vector store backed by Qdrant (v1 default)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncQdrantClient | None = None
        # Per-env (collection_name → expected_dimension) memo so we only
        # call ensure_collection at most once per env per worker process.
        self._known: dict[str, int] = {}

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self._settings.qdrant_url,
                api_key=self._settings.qdrant_api_key,
            )
        return self._client

    async def ensure_env_collection(
        self,
        *,
        env_id: UUID,
        dimension: int,
    ) -> None:
        name = _collection_name(env_id)
        cached = self._known.get(name)
        if cached is not None:
            if cached != dimension:
                raise RuntimeError(
                    f"qdrant collection {name!r} cached at dim={cached}, caller asked for dim={dimension}"
                )
            return

        try:
            info = await self.client.get_collection(collection_name=name)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict):
                body_cfg = vectors_config.get("body")
                trigger_cfg = vectors_config.get("trigger")
                existing_dim = getattr(body_cfg, "size", None)
                trigger_dim = getattr(trigger_cfg, "size", None)
            else:
                existing_dim = getattr(vectors_config, "size", None)
                trigger_dim = None
            if existing_dim == dimension and trigger_dim is None:
                await self._rebuild_legacy_single_vector_collection(
                    collection_name=name,
                    env_id=env_id,
                    dimension=dimension,
                )
                self._known[name] = dimension
                return
            if existing_dim != dimension or trigger_dim != dimension:
                raise RuntimeError(
                    f"qdrant collection {name!r} exists with body_dim={existing_dim} "
                    f"trigger_dim={trigger_dim}, caller expected dim={dimension}; "
                    "admin-rebuild required"
                )
            self._known[name] = dimension
            return
        except (UnexpectedResponse, ValueError):
            # Collection does not exist — create it.
            pass

        await self._create_env_collection(name, dimension)
        self._known[name] = dimension

    async def _create_env_collection(self, name: str, dimension: int) -> None:
        await self.client.create_collection(
            collection_name=name,
            vectors_config={
                "body": qm.VectorParams(
                    size=dimension,
                    distance=qm.Distance.COSINE,
                ),
                "trigger": qm.VectorParams(
                    size=dimension,
                    distance=qm.Distance.COSINE,
                ),
            },
        )
        # Payload indexes for the predicates the search path filters on.
        for field, schema in (
            ("status", qm.PayloadSchemaType.KEYWORD),
            ("kind", qm.PayloadSchemaType.KEYWORD),
            ("env_id", qm.PayloadSchemaType.KEYWORD),
            ("has_trigger_description", qm.PayloadSchemaType.BOOL),
        ):
            try:
                await self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                )
            except UnexpectedResponse as exc:
                log.warning(
                    "qdrant create_payload_index(%s, %s) failed: %s",
                    name,
                    field,
                    exc,
                )

    async def _rebuild_legacy_single_vector_collection(
        self,
        *,
        collection_name: str,
        env_id: UUID,
        dimension: int,
    ) -> None:
        """Replace pre-v0.6 unnamed-vector collections with named vectors.

        The collection is a rebuildable projection, so first access after a
        v0.6 upgrade can safely recreate it and backfill visible memory body
        vectors from Postgres.
        """
        log.warning(
            "qdrant collection %s uses legacy unnamed vectors; rebuilding with named body/trigger vectors",
            collection_name,
        )
        try:
            await self.client.delete_collection(collection_name=collection_name)
        except UnexpectedResponse as exc:
            log.warning("qdrant delete legacy collection %s failed during rebuild: %s", collection_name, exc)
        await self._create_env_collection(collection_name, dimension)
        backfilled = await self._backfill_env_body_vectors(env_id=env_id, dimension=dimension)
        log.warning(
            "qdrant collection %s rebuilt with named vectors; backfilled %s body vectors",
            collection_name,
            backfilled,
        )

    async def _backfill_env_body_vectors(self, *, env_id: UUID, dimension: int) -> int:
        embedder = get_embedder(self._settings)
        if embedder.dimension != dimension:
            raise RuntimeError(
                f"qdrant rebuild for env {env_id} expected embedder dimension {dimension}, got {embedder.dimension}"
            )

        async with session_scope() as session:
            model_id = (
                await session.execute(select(Environment.default_embedding_model_id).where(Environment.id == env_id))
            ).scalar_one_or_none()
            if model_id is None:
                return 0
            if model_id != embedder.model_id:
                raise RuntimeError(
                    f"qdrant rebuild for env {env_id} requires model {model_id!r}, "
                    f"configured embedder is {embedder.model_id!r}"
                )
            rows = (
                (
                    await session.execute(
                        select(Memory)
                        .where(Memory.env_id == env_id, Memory.status.in_(list(_QDRANT_VISIBLE_STATUSES)))
                        .order_by(Memory.updated_at.desc(), Memory.id.desc())
                    )
                )
                .scalars()
                .all()
            )
            memory_ids = [m.id for m in rows]
            tag_rows = (
                (
                    await session.execute(
                        select(MemoryTag.memory_id, Tag.name)
                        .join(Tag, MemoryTag.tag_id == Tag.id)
                        .where(MemoryTag.env_id == env_id, MemoryTag.memory_id.in_(memory_ids))
                        .order_by(MemoryTag.memory_id, Tag.name)
                    )
                ).all()
                if memory_ids
                else []
            )

        tags_by_memory: dict[UUID, list[str]] = {memory_id: [] for memory_id in memory_ids}
        for memory_id, tag_name in tag_rows:
            tags_by_memory.setdefault(memory_id, []).append(str(tag_name))

        backfilled = 0
        for memory in rows:
            body_text = _embed_text(memory.title, memory.body)
            if not body_text:
                continue
            trigger_text = str(memory.trigger_description or "").strip()
            texts = [body_text, trigger_text] if trigger_text else [body_text]
            vectors = embedder.embed_texts(texts)
            vector_payload: dict[str, list[float]] = {"body": list(vectors[0])}
            if trigger_text:
                vector_payload["trigger"] = list(vectors[1])
            await self.upsert(
                env_id=env_id,
                point_id=memory.id,
                vector=vector_payload,
                payload={
                    "memory_id": str(memory.id),
                    "env_id": str(env_id),
                    "kind": memory.kind,
                    "status": memory.status,
                    "title": memory.title,
                    "trigger_description": trigger_text or None,
                    "has_trigger_description": bool(trigger_text),
                    "tags": tags_by_memory.get(memory.id, []),
                    "salience": float(memory.salience),
                    "confidence": float(memory.confidence),
                    "pinned": memory.pinned,
                    "version": memory.version,
                    "embedding_model_id": embedder.model_id,
                    "created_at": memory.created_at.isoformat(),
                    "updated_at": memory.updated_at.isoformat(),
                },
            )
            backfilled += 1
        return backfilled

    async def upsert(
        self,
        *,
        env_id: UUID,
        point_id: UUID,
        vector: Sequence[float] | Mapping[str, Sequence[float]],
        payload: Mapping[str, Any],
    ) -> None:
        name = _collection_name(env_id)
        if isinstance(vector, Mapping):
            vector_payload: list[float] | dict[str, list[float]] = {
                vector_name: list(values) for vector_name, values in vector.items()
            }
        else:
            vector_payload = {"body": list(vector)}
        await self.client.upsert(
            collection_name=name,
            points=[
                qm.PointStruct(
                    id=str(point_id),
                    vector=vector_payload,
                    payload=dict(payload),
                )
            ],
        )

    async def delete(
        self,
        *,
        env_id: UUID,
        point_id: UUID,
    ) -> None:
        name = _collection_name(env_id)
        try:
            await self.client.delete(
                collection_name=name,
                points_selector=qm.PointIdsList(points=[str(point_id)]),
            )
        except UnexpectedResponse as exc:
            # Collection-missing on tombstone is benign (e.g., env never
            # got an upsert before being purged). We log and swallow.
            log.warning(
                "qdrant delete on missing collection %s for point %s: %s",
                name,
                point_id,
                exc,
            )

    async def search(
        self,
        *,
        env_id: UUID,
        query_vector: Sequence[float],
        limit: int,
        filters: Mapping[str, Any] | None = None,
        vector_name: str = "body",
    ) -> list[dict[str, Any]]:
        name = _collection_name(env_id)
        qfilter = self._build_filter(filters or {})
        try:
            result = await self.client.query_points(
                collection_name=name,
                query=list(query_vector),
                using=vector_name,
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            log.info("qdrant search on env %s returned %s", env_id, exc)
            return []

        out: list[dict[str, Any]] = []
        for hit in result.points:
            out.append(
                {
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "payload": dict(hit.payload or {}),
                }
            )
        return out

    async def get_vector(self, *, env_id: UUID, id: str, vector_name: str = "body") -> list[float] | None:
        records = await self.client.retrieve(
            collection_name=_collection_name(env_id),
            ids=[id],
            with_vectors=True,
            with_payload=False,
        )
        if not records:
            return None

        vector = records[0].vector
        if vector is None:
            return None
        if isinstance(vector, dict):
            vector = vector.get(vector_name)
            if vector is None:
                return None
        return list(vector)

    async def get_vectors(
        self,
        *,
        env_id: UUID,
        ids: list[UUID],
        vector_name: str = "body",
    ) -> dict[UUID, list[float] | None]:
        """Fetch named vectors for many memory ids in one Qdrant call.

        Missing ids map to None.
        """
        out: dict[UUID, list[float] | None] = dict.fromkeys(ids)
        if not ids:
            return out
        try:
            records = await self.client.retrieve(
                collection_name=_collection_name(env_id),
                ids=[str(memory_id) for memory_id in ids],
                with_vectors=True,
                with_payload=False,
            )
        except UnexpectedResponse as exc:
            log.info("qdrant batch vector retrieve on env %s returned %s", env_id, exc)
            return out

        for record in records:
            try:
                memory_id = UUID(str(record.id))
            except (TypeError, ValueError):
                log.debug("qdrant batch vector retrieve returned non-UUID id %r", getattr(record, "id", None))
                continue
            vector = getattr(record, "vector", None)
            if vector is None:
                out[memory_id] = None
                continue
            if isinstance(vector, dict):
                vector = vector.get(vector_name)
                if vector is None:
                    out[memory_id] = None
                    continue
            out[memory_id] = list(vector)
        return out

    def _build_filter(self, filters: Mapping[str, Any]) -> qm.Filter | None:
        """Translate a flat ``{field: value | [values]}`` dict to Qdrant filter.

        Lists become ``MatchAny``; scalars become ``MatchValue``.
        """
        must: list[qm.Condition] = []
        for field, value in filters.items():
            if isinstance(value, list):
                must.append(
                    qm.FieldCondition(
                        key=field,
                        match=qm.MatchAny(any=list(value)),
                    )
                )
            else:
                must.append(
                    qm.FieldCondition(
                        key=field,
                        match=qm.MatchValue(value=value),
                    )
                )
        if not must:
            return None
        return qm.Filter(must=must)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


__all__ = ["QdrantVectorStore"]
