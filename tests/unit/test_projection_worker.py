"""Unit tests for projection-worker components.

These are pure unit tests — no Postgres / Qdrant required. The
end-to-end smoke (``.tmp/projection_smoke.py``) covers the lease SQL
against real Postgres.
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from uuid import uuid4

import pytest

from memory_mcp.db.types import OutboxAggregateType, OutboxOp
from memory_mcp.errors import EmbeddingModelMismatchError
from projection_worker.handlers.qdrant import handle_qdrant_event
from projection_worker.lease import LeasedEvent, _backoff_seconds

# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        (1, 2),
        (2, 4),
        (3, 8),
        (4, 16),
        (5, 32),
        (10, 600),  # capped
        (20, 600),  # capped
    ],
)
def test_backoff_caps_at_600s(attempt: int, expected: int) -> None:
    assert _backoff_seconds(attempt) == expected


# ---------------------------------------------------------------------------
# Qdrant handler — fake vector store + fake embedder
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self) -> None:
        self.ensured: list[tuple[Any, int]] = []
        self.upserts: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []

    async def ensure_env_collection(self, *, env_id: Any, dimension: int) -> None:
        self.ensured.append((env_id, dimension))

    async def upsert(self, *, env_id: Any, point_id: Any, vector: Any, payload: Any) -> None:
        vector_out = (
            {name: list(values) for name, values in vector.items()} if isinstance(vector, dict) else list(vector)
        )
        self.upserts.append(
            {
                "env_id": env_id,
                "point_id": point_id,
                "vector": vector_out,
                "payload": dict(payload),
            }
        )

    async def delete(self, *, env_id: Any, point_id: Any) -> None:
        self.deletes.append({"env_id": env_id, "point_id": point_id})

    async def search(self, *, env_id: Any, query_vector: Any, limit: int, filters=None) -> list:
        return []

    async def get_vector(self, *, env_id: Any, id: str) -> list[float] | None:
        return None

    async def close(self) -> None:
        pass


class _FakeEmbedder:
    def __init__(self, *, model_id: str = "all-MiniLM-L6-v2", dimension: int = 384) -> None:
        self._model_id = model_id
        self._dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[0.1] * self._dimension for _ in texts]


def _make_event(
    *,
    op: str = "upsert",
    payload: dict[str, Any] | None = None,
    aggregate_type: str = "memory",
) -> LeasedEvent:
    base_payload = {
        "title": "Hello",
        "body": "World",
        "kind": "fact",
        "status": "active",
        "tags": ["t1"],
        "salience": 0.5,
        "confidence": 0.5,
        "pinned": False,
        "embedding_model_id": "all-MiniLM-L6-v2",
        "version": 1,
        "created_at": "2026-05-06T00:00:00+00:00",
        "updated_at": "2026-05-06T00:00:00+00:00",
    }
    if payload:
        base_payload.update(payload)
    return LeasedEvent(
        event_id=1,
        sink="qdrant",  # type: ignore[arg-type]
        aggregate_type=aggregate_type,
        aggregate_id=uuid4(),
        aggregate_version=1,
        env_id=uuid4(),
        op=op,
        payload=base_payload,
        attempt_count=0,
        created_at=dt.datetime.now(tz=dt.UTC),
    )


@pytest.mark.asyncio
async def test_handler_upsert_active_memory_calls_embed_and_upsert() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event()
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)

    assert embedder.calls == [["Hello\n\nWorld"]]
    assert len(store.ensured) == 1
    assert store.ensured[0] == (ev.env_id, 384)
    assert len(store.upserts) == 1
    point = store.upserts[0]
    assert point["point_id"] == ev.aggregate_id
    assert point["env_id"] == ev.env_id
    assert point["vector"] == {"body": [0.1] * 384}
    assert point["payload"]["embedding_model_id"] == "all-MiniLM-L6-v2"
    assert point["payload"]["status"] == "active"
    assert point["payload"]["title"] == "Hello"


@pytest.mark.asyncio
async def test_handler_op_tombstone_calls_delete() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event(op=OutboxOp.tombstone.value, payload={"status": "retired"})
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)

    assert embedder.calls == []
    assert store.upserts == []
    assert len(store.deletes) == 1
    assert store.deletes[0] == {"env_id": ev.env_id, "point_id": ev.aggregate_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden_status", ["archived", "superseded", "retired"])
async def test_handler_hidden_status_acts_as_tombstone(hidden_status: str) -> None:
    """Even with op=upsert/update, a hidden status results in a delete."""
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event(op="update", payload={"status": hidden_status})
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)

    assert store.upserts == []
    assert len(store.deletes) == 1


@pytest.mark.asyncio
async def test_handler_model_mismatch_raises() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder(model_id="other-model")
    ev = _make_event()
    with pytest.raises(EmbeddingModelMismatchError) as excinfo:
        await handle_qdrant_event(ev, vector_store=store, embedder=embedder)
    assert excinfo.value.expected == "all-MiniLM-L6-v2"
    assert excinfo.value.actual == "other-model"
    assert store.upserts == []
    assert store.deletes == []


@pytest.mark.asyncio
async def test_handler_empty_text_falls_back_to_delete() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event(payload={"title": "", "body": ""})
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)

    assert embedder.calls == []
    assert store.upserts == []
    assert len(store.deletes) == 1


@pytest.mark.asyncio
async def test_handler_title_only_embeds_title() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event(payload={"title": "JustTitle", "body": ""})
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)
    assert embedder.calls == [["JustTitle"]]
    assert len(store.upserts) == 1


@pytest.mark.asyncio
async def test_handler_trigger_description_adds_named_trigger_vector() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event(payload={"trigger_description": "deploy prod"})
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)

    assert embedder.calls == [["Hello\n\nWorld", "deploy prod"]]
    point = store.upserts[0]
    assert set(point["vector"]) == {"body", "trigger"}
    assert point["payload"]["trigger_description"] == "deploy prod"
    assert point["payload"]["has_trigger_description"] is True


@pytest.mark.asyncio
async def test_handler_body_only_embeds_body() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event(payload={"title": "", "body": "JustBody"})
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)
    assert embedder.calls == [["JustBody"]]
    assert len(store.upserts) == 1


@pytest.mark.asyncio
async def test_handler_rejects_non_memory_aggregate() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event(aggregate_type=OutboxAggregateType.entity.value)
    with pytest.raises(ValueError, match="aggregate_type"):
        await handle_qdrant_event(ev, vector_store=store, embedder=embedder)
    assert store.upserts == []
    assert store.deletes == []


@pytest.mark.asyncio
async def test_handler_no_model_id_in_payload_skips_check() -> None:
    """Older events / bare-bones payloads (no embedding_model_id) still process."""
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event()
    ev.payload.pop("embedding_model_id")
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)
    assert len(store.upserts) == 1
    assert store.upserts[0]["payload"]["embedding_model_id"] == embedder.model_id


@pytest.mark.asyncio
async def test_handler_proposed_status_is_indexed() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event(payload={"status": "proposed"})
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)
    assert len(store.upserts) == 1
    assert store.deletes == []


@pytest.mark.asyncio
async def test_handler_stale_status_is_indexed() -> None:
    store, embedder = _FakeStore(), _FakeEmbedder()
    ev = _make_event(payload={"status": "stale"})
    await handle_qdrant_event(ev, vector_store=store, embedder=embedder)
    assert len(store.upserts) == 1
    assert store.deletes == []
