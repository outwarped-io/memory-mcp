"""Entity (people/services/repos/...) API namespace."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from memory_mcp_client._batch import BatchResult, run_bounded
from memory_mcp_client.api._base import _BaseAPI
from memory_mcp_schemas.entities import (
    EntityBrowseRequest,
    EntityBrowseResponse,
    EntityMergeRequest,
    EntityResolveRequest,
    EntityResponse,
    EntityUpsertRequest,
)
from memory_mcp_schemas.graph import (
    EntityNeighborsRequest,
    EntityNeighborsResponse,
)


class EntitiesAPI(_BaseAPI):
    """Memory-mcp entities namespace."""

    async def upsert(
        self,
        request: EntityUpsertRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EntityResponse:
        if request is None:
            request = EntityUpsertRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("ent_upsert", payload, model=EntityResponse)

    async def upsert_many(
        self,
        items: list[EntityUpsertRequest],
        *,
        max_concurrency: int = 8,
    ) -> BatchResult[EntityUpsertRequest, EntityResponse]:
        """Upsert many entities with bounded client-side concurrency."""

        return await run_bounded(items, self.upsert, max_concurrency=max_concurrency)

    async def resolve(
        self,
        request: EntityResolveRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> list[EntityResponse]:
        if request is None:
            request = EntityResolveRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("ent_resolve", payload, model=EntityResponse)

    async def merge(
        self,
        request: EntityMergeRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EntityResponse:
        if request is None:
            request = EntityMergeRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("ent_merge", payload, model=EntityResponse)

    async def neighbors(
        self,
        request: EntityNeighborsRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EntityNeighborsResponse:
        if request is None:
            request = EntityNeighborsRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call(
            "ent_neighbors", payload, model=EntityNeighborsResponse
        )

    async def browse(
        self,
        request: EntityBrowseRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EntityBrowseResponse:
        if request is None:
            request = EntityBrowseRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("ent_browse", payload, model=EntityBrowseResponse)
