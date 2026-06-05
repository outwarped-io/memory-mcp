"""Environment management API namespace."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from memory_mcp_schemas.envs import (
    AttachedEnvsResponse,
    EnvCreateRequest,
    EnvResponse,
)

from memory_mcp_client.api._base import _BaseAPI


class EnvsAPI(_BaseAPI):
    """Environment management namespace."""

    async def create(
        self,
        request: EnvCreateRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvResponse:
        if request is None:
            request = EnvCreateRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("env_create_", payload, model=EnvResponse)

    async def list_(
        self,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> list[EnvResponse]:
        payload: dict[str, Any] = {}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("env_list_", payload, model=EnvResponse)

    async def get(
        self,
        *,
        name: str | None = None,
        env_id: UUID | str | None = None,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> EnvResponse:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if env_id is not None:
            payload["env_id"] = str(env_id)
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("env_get_", payload, model=EnvResponse)

    async def attach(
        self,
        *,
        name: str,
        session_id: UUID | str,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> AttachedEnvsResponse:
        payload: dict[str, Any] = {
            "name": name,
            "session_id": str(session_id),
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("env_attach_", payload, model=AttachedEnvsResponse)

    async def detach(
        self,
        *,
        name: str,
        session_id: UUID | str,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> AttachedEnvsResponse:
        payload: dict[str, Any] = {
            "name": name,
            "session_id": str(session_id),
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("env_detach_", payload, model=AttachedEnvsResponse)
