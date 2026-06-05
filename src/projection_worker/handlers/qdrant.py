"""Qdrant handler for the projection worker.

Translates leased ``LeasedEvent`` rows into Qdrant ``upsert`` / ``delete``
calls. Embeds memory text on the worker side (centralized batching, keeps
the canonical writer light).

For ``aggregate_type=memory``:

* ``op=upsert`` with status ∈ {proposed, active, stale} → embed body
  (or ``title + body``) and upsert the point.
* ``op=update`` with same statuses → re-embed (text may have changed) and
  re-upsert.
* ``op=tombstone`` (status ∈ {archived, superseded, retired}) → delete the
  point.
* Any other ``aggregate_type`` reaching this handler is a routing bug —
  raises ``ValueError``.

Embedding model id check
------------------------

The outbox payload carries ``embedding_model_id`` (snapshotted at write
time per the env's default). If it doesn't match the currently configured
embedder, we raise ``EmbeddingModelMismatchError`` and let the worker
mark the delivery failed; operators must run ``rebuild_qdrant`` to switch
models.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from memory_mcp.db.types import OutboxAggregateType, OutboxOp
from memory_mcp.db.vector.base import VectorStore
from memory_mcp.embeddings.base import Embedder
from memory_mcp.errors import EmbeddingModelMismatchError
from projection_worker.lease import LeasedEvent

log = logging.getLogger(__name__)


def _embed_text(payload: dict[str, Any]) -> str:
    """Concatenate title + body for embedding (title-only and body-only fine)."""
    parts: list[str] = []
    title = payload.get("title")
    if title:
        parts.append(str(title))
    body = payload.get("body")
    if body:
        parts.append(str(body))
    return "\n\n".join(parts).strip()


def _trigger_text(payload: dict[str, Any]) -> str:
    value = payload.get("trigger_description")
    return str(value).strip() if value else ""


_TOMBSTONE_STATUSES: frozenset[str] = frozenset({"archived", "superseded", "retired"})


async def handle_qdrant_event(
    event: LeasedEvent,
    *,
    vector_store: VectorStore,
    embedder: Embedder,
) -> None:
    """Apply a single leased event to the Qdrant collection for ``event.env_id``.

    Idempotent on retry: ``upsert`` overwrites, ``delete`` is no-op-on-missing.
    """
    if event.aggregate_type != OutboxAggregateType.memory.value:
        # Defense in depth — Qdrant sink only handles memory aggregates.
        raise ValueError(f"qdrant handler received aggregate_type={event.aggregate_type!r}; expected 'memory'")

    payload = event.payload
    point_id = event.aggregate_id

    is_tombstone = event.op == OutboxOp.tombstone.value or str(payload.get("status", "")).lower() in _TOMBSTONE_STATUSES

    if is_tombstone:
        await vector_store.delete(env_id=event.env_id, point_id=point_id)
        log.debug(
            "qdrant tombstone applied: event_id=%s memory_id=%s",
            event.event_id,
            point_id,
        )
        return

    payload_model_id = payload.get("embedding_model_id")
    if payload_model_id and payload_model_id != embedder.model_id:
        raise EmbeddingModelMismatchError(
            expected=str(payload_model_id),
            actual=embedder.model_id,
        )

    body_text = _embed_text(payload)
    trigger_text = _trigger_text(payload)
    if not body_text:
        # No content to embed — treat as a tombstone so we don't leave a
        # stale point. (Rare: fully empty memory; the schema permits empty
        # title with non-empty body.)
        await vector_store.delete(env_id=event.env_id, point_id=point_id)
        log.warning(
            "qdrant empty-text fallback delete: event_id=%s memory_id=%s",
            event.event_id,
            point_id,
        )
        return

    # ``embed_texts`` is sync (may be CPU-heavy); run in a thread pool to
    # avoid blocking the asyncio loop.
    vectors = await asyncio.get_running_loop().run_in_executor(
        None,
        embedder.embed_texts,
        [body_text, trigger_text] if trigger_text else [body_text],
    )
    vector_payload: dict[str, list[float]] = {"body": vectors[0]}
    if trigger_text:
        vector_payload["trigger"] = vectors[1]

    await vector_store.ensure_env_collection(
        env_id=event.env_id,
        dimension=embedder.dimension,
    )

    qdrant_payload = {
        "memory_id": str(point_id),
        "env_id": str(event.env_id),
        "kind": payload.get("kind"),
        "status": payload.get("status"),
        "title": payload.get("title"),
        "trigger_description": trigger_text or None,
        "has_trigger_description": bool(trigger_text),
        "tags": payload.get("tags") or [],
        "salience": payload.get("salience"),
        "confidence": payload.get("confidence"),
        "pinned": payload.get("pinned"),
        "version": event.aggregate_version,
        "embedding_model_id": embedder.model_id,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }

    await vector_store.upsert(
        env_id=event.env_id,
        point_id=point_id,
        vector=vector_payload,
        payload=qdrant_payload,
    )
    log.debug(
        "qdrant upsert applied: event_id=%s memory_id=%s op=%s",
        event.event_id,
        point_id,
        event.op,
    )


__all__ = ["handle_qdrant_event"]
