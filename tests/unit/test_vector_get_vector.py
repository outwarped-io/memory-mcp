from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID, uuid4

import pytest

from memory_mcp.db.vector.base import VectorStore


class _MinimalVectorStore:
    def __init__(self) -> None:
        self._vectors: dict[tuple[UUID, str], list[float]] = {}

    async def ensure_env_collection(self, *, env_id: UUID, dimension: int) -> None:
        pass

    async def upsert(
        self,
        *,
        env_id: UUID,
        point_id: UUID,
        vector: Sequence[float],
        payload: Mapping[str, Any],
    ) -> None:
        self._vectors[(env_id, str(point_id))] = list(vector)

    async def delete(self, *, env_id: UUID, point_id: UUID) -> None:
        self._vectors.pop((env_id, str(point_id)), None)

    async def search(
        self,
        *,
        env_id: UUID,
        query_vector: Sequence[float],
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    async def get_vector(self, *, env_id: UUID, id: str) -> list[float] | None:
        return self._vectors.get((env_id, id))

    async def get_vectors(
        self,
        *,
        env_id: UUID,
        ids: list[UUID],
        vector_name: str = "body",
    ) -> dict[UUID, list[float] | None]:
        return {point_id: self._vectors.get((env_id, str(point_id))) for point_id in ids}

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_vector_store_protocol_can_implement_get_vector() -> None:
    env_id = uuid4()
    point_id = uuid4()
    store = _MinimalVectorStore()

    assert isinstance(store, VectorStore)

    await store.upsert(env_id=env_id, point_id=point_id, vector=[0.1, 0.2], payload={})

    assert await store.get_vector(env_id=env_id, id=str(point_id)) == [0.1, 0.2]


@pytest.mark.asyncio
async def test_get_vector_returns_none_for_missing_point() -> None:
    store = _MinimalVectorStore()

    assert await store.get_vector(env_id=uuid4(), id=str(uuid4())) is None
