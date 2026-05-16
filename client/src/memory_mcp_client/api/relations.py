"""Inter-entity relation API namespace."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from memory_mcp_client.api._base import _BaseAPI
from memory_mcp_schemas.relations import (
    RelationBrowseRequest,
    RelationBrowseResponse,
    RelationLinkRequest,
    RelationResponse,
)


class RelationsAPI(_BaseAPI):
    """Memory-mcp relations namespace."""

    async def link(
        self,
        request: RelationLinkRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> RelationResponse:
        if request is None:
            request = RelationLinkRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("rel_link", payload, model=RelationResponse)

    async def browse(
        self,
        request: RelationBrowseRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> RelationBrowseResponse:
        if request is None:
            request = RelationBrowseRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("rel_browse", payload, model=RelationBrowseResponse)
