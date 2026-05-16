from __future__ import annotations

import math
import datetime as dt
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import pytest

from memory_mcp.db.models import Memory
from memory_mcp.errors import InvalidInputError
from memory_mcp.memories import MemoryUpdatePatch, MemoryWriteRequest, _to_response
from memory_mcp.search import auto_context as auto_mod
from memory_mcp.search import api as search_api
from memory_mcp.search.auto_context import memory_auto_context


class _FakeEmbedder:
    model_id = "all-MiniLM-L6-v2"
    dimension = 6

    _groups = {
        "deploy": 0,
        "deployment": 0,
        "ship": 0,
        "release": 0,
        "prod": 0,
        "production": 0,
        "database": 1,
        "postgres": 1,
        "schema": 1,
        "migration": 1,
        "test": 2,
        "pytest": 2,
        "build": 3,
        "cache": 4,
        "auth": 5,
    }

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for raw in text.lower().replace("-", " ").split():
            word = raw.strip(".,:;!?()[]{}")
            idx = self._groups.get(word)
            if idx is not None:
                vector[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector


class _FakeVectorStore:
    def __init__(self, embedder: _FakeEmbedder) -> None:
        self.embedder = embedder
        self.trigger_vectors: dict[UUID, dict[UUID, list[float]]] = {}
        self.forced_scores: dict[UUID, float] = {}
        self.last_hit_ids: list[UUID] = []

    def add_trigger(self, env_id: UUID, memory_id: UUID, text: str) -> None:
        self.trigger_vectors.setdefault(env_id, {})[memory_id] = self.embedder.embed_texts([text])[0]

    async def ensure_env_collection(self, *, env_id: UUID, dimension: int) -> None:
        pass

    async def upsert(
        self,
        *,
        env_id: UUID,
        point_id: UUID,
        vector: Sequence[float] | Mapping[str, Sequence[float]],
        payload: Mapping[str, Any],
    ) -> None:
        if isinstance(vector, Mapping) and "trigger" in vector:
            self.trigger_vectors.setdefault(env_id, {})[point_id] = list(vector["trigger"])

    async def delete(self, *, env_id: UUID, point_id: UUID) -> None:
        self.trigger_vectors.get(env_id, {}).pop(point_id, None)

    async def search(
        self,
        *,
        env_id: UUID,
        query_vector: Sequence[float],
        limit: int,
        filters: Mapping[str, Any] | None = None,
        vector_name: str = "body",
    ) -> list[dict[str, Any]]:
        assert vector_name == "trigger"
        hits: list[dict[str, Any]] = []
        for memory_id, vector in self.trigger_vectors.get(env_id, {}).items():
            score = self.forced_scores.get(memory_id, sum(a * b for a, b in zip(query_vector, vector, strict=True)))
            hits.append({"id": str(memory_id), "score": score, "payload": {}})
        hits.sort(key=lambda h: h["score"], reverse=True)
        self.last_hit_ids = [UUID(str(h["id"])) for h in hits[:limit]]
        return hits[:limit]

    async def get_vector(self, *, env_id: UUID, id: str, vector_name: str = "body") -> list[float] | None:
        return self.trigger_vectors.get(env_id, {}).get(UUID(id))

    async def close(self) -> None:
        pass


class _Result:
    def __init__(self, *, scalar: str | None = None, rows: list[Memory] | None = None) -> None:
        self._scalar = scalar
        self._rows = rows or []

    def scalar_one_or_none(self) -> str | None:
        return self._scalar

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[Memory]:
        return self._rows


class _FakeSession:
    def __init__(self, env_id: UUID, memories: list[Memory], store: _FakeVectorStore) -> None:
        self.env_id = env_id
        self.memories = memories
        self.store = store
        self.calls = 0

    async def execute(self, _stmt: Any) -> _Result:
        self.calls += 1
        if (self.calls - 1) % 3 == 0:
            return _Result(scalar=_FakeEmbedder.model_id)
        ids = set(self.store.last_hit_ids)
        rows = [
            m for m in self.memories
            if m.env_id == self.env_id
            and (not ids or m.id in ids)
            and m.trigger_description is not None
        ]
        return _Result(rows=rows)


def _patch_sessions(
    monkeypatch: pytest.MonkeyPatch,
    env_id: UUID,
    memories: list[Memory],
    store: _FakeVectorStore,
) -> None:
    session = _FakeSession(env_id, memories, store)

    @asynccontextmanager
    async def fake_session_scope():
        yield session

    monkeypatch.setattr(search_api, "session_scope", fake_session_scope)
    monkeypatch.setattr(auto_mod, "session_scope", fake_session_scope)


def _memory(
    *,
    env_id: UUID,
    title: str,
    body: str = "body",
    trigger_description: str | None,
    salience: float = 0.5,
) -> Memory:
    return Memory(
        id=uuid4(),
        env_id=env_id,
        kind="fact",
        status="active",
        title=title,
        body=body,
        trigger_description=trigger_description,
        salience=salience,
        confidence=0.9,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        metadata_={},
        version=1,
        created_at=dt.datetime.now(dt.UTC),
        updated_at=dt.datetime.now(dt.UTC),
    )


@pytest.mark.asyncio
async def test_empty_trigger_pool_returns_empty_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    embedder = _FakeEmbedder()
    store = _FakeVectorStore(embedder)
    _patch_sessions(monkeypatch, env_id, [], store)

    resp = await memory_auto_context(task_desc="deploy the app", env_id=env_id, vector_store=store, embedder=embedder)

    assert resp.hits == []


@pytest.mark.asyncio
async def test_exact_text_match_returns_highest_score(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    embedder = _FakeEmbedder()
    store = _FakeVectorStore(embedder)
    exact = _memory(env_id=env_id, title="exact", trigger_description="deploy production release")
    other = _memory(env_id=env_id, title="other", trigger_description="postgres schema migration")
    store.add_trigger(env_id, exact.id, exact.trigger_description or "")
    store.add_trigger(env_id, other.id, other.trigger_description or "")
    _patch_sessions(monkeypatch, env_id, [exact, other], store)

    resp = await memory_auto_context(
        task_desc="deploy production release",
        env_id=env_id,
        vector_store=store,
        embedder=embedder,
    )

    assert resp.hits[0].memory_id == exact.id
    assert resp.hits[0].score >= resp.hits[-1].score


@pytest.mark.asyncio
async def test_semantic_match_returns_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    embedder = _FakeEmbedder()
    store = _FakeVectorStore(embedder)
    memory = _memory(env_id=env_id, title="deploy", trigger_description="deploy service to production")
    store.add_trigger(env_id, memory.id, memory.trigger_description or "")
    _patch_sessions(monkeypatch, env_id, [memory], store)

    resp = await memory_auto_context(
        task_desc="ship a release to prod",
        env_id=env_id,
        vector_store=store,
        embedder=embedder,
    )

    assert [hit.memory_id for hit in resp.hits] == [memory.id]


@pytest.mark.asyncio
async def test_env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    env_a, env_b = uuid4(), uuid4()
    embedder = _FakeEmbedder()
    store = _FakeVectorStore(embedder)
    a = _memory(env_id=env_a, title="a", trigger_description="deploy production release")
    b = _memory(env_id=env_b, title="b", trigger_description="deploy production release")
    store.add_trigger(env_a, a.id, a.trigger_description or "")
    store.add_trigger(env_b, b.id, b.trigger_description or "")
    _patch_sessions(monkeypatch, env_b, [a, b], store)

    resp = await memory_auto_context(task_desc="deploy prod", env_id=env_b, vector_store=store, embedder=embedder)

    assert [hit.memory_id for hit in resp.hits] == [b.id]


@pytest.mark.asyncio
async def test_salience_tiebreak_when_scores_are_tied(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    embedder = _FakeEmbedder()
    store = _FakeVectorStore(embedder)
    low = _memory(env_id=env_id, title="low", trigger_description="deploy prod", salience=0.1)
    high = _memory(env_id=env_id, title="high", trigger_description="deploy prod", salience=0.9)
    store.add_trigger(env_id, low.id, low.trigger_description or "")
    store.add_trigger(env_id, high.id, high.trigger_description or "")
    store.forced_scores = {low.id: 0.5, high.id: 0.5}
    _patch_sessions(monkeypatch, env_id, [low, high], store)

    resp = await memory_auto_context(task_desc="deploy prod", env_id=env_id, vector_store=store, embedder=embedder)

    assert [hit.memory_id for hit in resp.hits] == [high.id, low.id]


@pytest.mark.asyncio
async def test_top_k_one_returns_one_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    embedder = _FakeEmbedder()
    store = _FakeVectorStore(embedder)
    memories = [
        _memory(env_id=env_id, title="a", trigger_description="deploy prod"),
        _memory(env_id=env_id, title="b", trigger_description="database migration"),
    ]
    for memory in memories:
        store.add_trigger(env_id, memory.id, memory.trigger_description or "")
    _patch_sessions(monkeypatch, env_id, memories, store)

    resp = await memory_auto_context(
        task_desc="deploy prod",
        env_id=env_id,
        top_k=1,
        vector_store=store,
        embedder=embedder,
    )

    assert len(resp.hits) == 1


@pytest.mark.asyncio
async def test_top_k_zero_raises_invalid_input() -> None:
    with pytest.raises(InvalidInputError, match="top_k"):
        await memory_auto_context(task_desc="deploy", env_id=uuid4(), top_k=0)


@pytest.mark.asyncio
async def test_empty_task_desc_raises_invalid_input() -> None:
    with pytest.raises(InvalidInputError, match="task_desc"):
        await memory_auto_context(task_desc="   ", env_id=uuid4())


def test_mem_write_response_round_trips_trigger_description() -> None:
    env_id = uuid4()
    req = MemoryWriteRequest(kind="fact", body="body", env_id=env_id, trigger_description="deploy prod")
    memory = _memory(env_id=env_id, title="round-trip", body=req.body, trigger_description=req.trigger_description)

    resp = _to_response(memory, [])

    assert req.trigger_description == "deploy prod"
    assert resp.trigger_description == "deploy prod"


@pytest.mark.asyncio
async def test_mem_update_trigger_change_reflected_by_auto_context(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    embedder = _FakeEmbedder()
    store = _FakeVectorStore(embedder)
    memory = _memory(env_id=env_id, title="mutable", trigger_description="database migration")
    store.add_trigger(env_id, memory.id, memory.trigger_description or "")
    _patch_sessions(monkeypatch, env_id, [memory], store)

    before = await memory_auto_context(task_desc="deploy prod", env_id=env_id, vector_store=store, embedder=embedder)
    patch = MemoryUpdatePatch(expected_version=1, trigger_description="deploy production release")
    memory.trigger_description = patch.trigger_description
    store.add_trigger(env_id, memory.id, memory.trigger_description or "")
    after = await memory_auto_context(task_desc="deploy prod", env_id=env_id, vector_store=store, embedder=embedder)

    assert before.hits[0].trigger_description == "database migration"
    assert [hit.memory_id for hit in after.hits] == [memory.id]
    assert after.hits[0].trigger_description == "deploy production release"
