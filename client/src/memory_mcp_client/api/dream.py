"""Background dream-cycle API namespace."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from memory_mcp_schemas.dream import (
    DreamProposalsListRequest,
    DreamProposalsListResponse,
    DreamReviewRequest,
    DreamReviewResponse,
    DreamRunRequest,
    DreamRunResponse,
    DreamStatusRequest,
    DreamStatusResponse,
)

from memory_mcp_client.api._base import _BaseAPI


def _payload(
    request: Any,
    *,
    agent_id: UUID | str | None,
    attached_env_ids: list[UUID | str] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
    if agent_id is not None:
        payload["agent_id"] = str(agent_id)
    if attached_env_ids is not None:
        payload["attached_env_ids"] = [str(env_id) for env_id in attached_env_ids]
    return payload


class DreamAPI(_BaseAPI):
    """Memory-mcp dream namespace."""

    async def run(
        self,
        request: DreamRunRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> DreamRunResponse:
        if request is None:
            request = DreamRunRequest(**kwargs)
        return await self._call(
            "dream_run_",
            _payload(request, agent_id=agent_id, attached_env_ids=attached_env_ids),
            model=DreamRunResponse,
        )

    async def status(
        self,
        request: DreamStatusRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> DreamStatusResponse:
        if request is None:
            request = DreamStatusRequest(**kwargs)
        return await self._call(
            "dream_status_",
            _payload(request, agent_id=agent_id, attached_env_ids=attached_env_ids),
            model=DreamStatusResponse,
        )

    async def proposals_list(
        self,
        request: DreamProposalsListRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> DreamProposalsListResponse:
        if request is None:
            request = DreamProposalsListRequest(**kwargs)
        return await self._call(
            "dream_proposals_list_",
            _payload(request, agent_id=agent_id, attached_env_ids=attached_env_ids),
            model=DreamProposalsListResponse,
        )

    async def review(
        self,
        request: DreamReviewRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> DreamReviewResponse:
        if request is None:
            request = DreamReviewRequest(**kwargs)
        return await self._call(
            "dream_review_",
            _payload(request, agent_id=agent_id, attached_env_ids=attached_env_ids),
            model=DreamReviewResponse,
        )
