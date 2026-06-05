from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from memory_mcp.config import Settings
from memory_mcp.db.models import Memory
from memory_mcp.db.vector import qdrant as qdrant_mod
from memory_mcp.db.vector.qdrant import QdrantVectorStore


class _FakeEmbedder:
    model_id = "all-MiniLM-L6-v2"
    dimension = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):  # type: ignore[no-untyped-def]
        self.calls.append(list(texts))
        return [[float(i + 1)] * self.dimension for i, _text in enumerate(texts)]


class _FakeResult:
    def __init__(self, *, scalar=None, rows=None) -> None:  # type: ignore[no-untyped-def]
        self._scalar = scalar
        self._rows = rows or []

    def scalar_one_or_none(self):  # type: ignore[no-untyped-def]
        return self._scalar

    def scalars(self):  # type: ignore[no-untyped-def]
        return self

    def all(self):  # type: ignore[no-untyped-def]
        return self._rows


class _FakeSession:
    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self.calls = 0

    async def execute(self, _stmt):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return _FakeResult(scalar="all-MiniLM-L6-v2")
        if self.calls == 2:
            return _FakeResult(rows=[self.memory])
        return _FakeResult(rows=[(self.memory.id, "tag-one")])


class _FakeClient:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.created: list[dict[str, object]] = []
        self.indexes: list[tuple[str, object]] = []
        self.upserts: list[object] = []

    async def get_collection(self, *, collection_name: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=3)),
            ),
        )

    async def delete_collection(self, *, collection_name: str) -> None:
        self.deleted.append(collection_name)

    async def create_collection(self, *, collection_name: str, vectors_config) -> None:  # type: ignore[no-untyped-def]
        self.created.append({"name": collection_name, "vectors_config": vectors_config})

    async def create_payload_index(self, *, collection_name: str, field_name: str, field_schema) -> None:  # type: ignore[no-untyped-def]
        self.indexes.append((field_name, field_schema))

    async def upsert(self, *, collection_name: str, points) -> None:  # type: ignore[no-untyped-def]
        self.upserts.extend(points)


class _FakeRetrieveClient:
    def __init__(self, records) -> None:  # type: ignore[no-untyped-def]
        self.records = records
        self.retrieve_calls: list[dict[str, object]] = []

    async def retrieve(self, **kwargs):  # type: ignore[no-untyped-def]
        self.retrieve_calls.append(kwargs)
        return self.records


@pytest.mark.asyncio
async def test_ensure_env_collection_rebuilds_legacy_single_vector_and_backfills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    memory = Memory(
        id=uuid4(),
        env_id=env_id,
        kind="fact",
        status="active",
        title="Title",
        body="Body",
        trigger_description="when deploying",
        salience=0.7,
        confidence=0.8,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        metadata_={},
        version=2,
        created_at=dt.datetime(2026, 5, 12, tzinfo=dt.UTC),
        updated_at=dt.datetime(2026, 5, 12, 1, tzinfo=dt.UTC),
    )
    fake_session = _FakeSession(memory)
    fake_embedder = _FakeEmbedder()

    @asynccontextmanager
    async def fake_session_scope():
        yield fake_session

    monkeypatch.setattr(qdrant_mod, "session_scope", fake_session_scope)
    monkeypatch.setattr(qdrant_mod, "get_embedder", lambda _settings: fake_embedder)

    store = QdrantVectorStore(Settings(_env_file=None))  # type: ignore[arg-type]
    client = _FakeClient()
    store._client = client  # noqa: SLF001

    await store.ensure_env_collection(env_id=env_id, dimension=3)

    assert client.deleted == [f"memory-mcp-{env_id}"]
    assert set(client.created[0]["vectors_config"]) == {"body", "trigger"}
    assert ("has_trigger_description", qdrant_mod.qm.PayloadSchemaType.BOOL) in client.indexes
    assert fake_embedder.calls == [["Title\n\nBody", "when deploying"]]
    assert len(client.upserts) == 1
    point = client.upserts[0]
    assert point.id == str(memory.id)
    assert set(point.vector) == {"body", "trigger"}
    assert point.payload["tags"] == ["tag-one"]
    assert point.payload["has_trigger_description"] is True


@pytest.mark.asyncio
async def test_get_vectors_fetches_named_vectors_in_one_call_and_marks_missing() -> None:
    env_id = uuid4()
    present_id = uuid4()
    missing_id = uuid4()
    client = _FakeRetrieveClient(
        [
            SimpleNamespace(id=str(present_id), vector={"body": [1.0, 2.0], "trigger": [3.0, 4.0]}),
        ]
    )
    store = QdrantVectorStore(Settings(_env_file=None))  # type: ignore[arg-type]
    store._client = client  # noqa: SLF001

    out = await store.get_vectors(env_id=env_id, ids=[present_id, missing_id], vector_name="body")

    assert out == {present_id: [1.0, 2.0], missing_id: None}
    assert len(client.retrieve_calls) == 1
    assert client.retrieve_calls[0]["ids"] == [str(present_id), str(missing_id)]
