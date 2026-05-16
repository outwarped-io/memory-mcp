"""Playbook macro invocation API namespace."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from memory_mcp_client.api._base import _BaseAPI
from memory_mcp_schemas.playbooks import PlaybookInvokeResponse


class PlaybooksAPI(_BaseAPI):
    """Memory-mcp playbooks namespace."""

    async def invoke(
        self,
        macro: str,
        env_id: UUID | str,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> PlaybookInvokeResponse:
        """Invoke a playbook macro in an environment."""
        payload: dict[str, Any] = {"macro": macro, "env_id": str(env_id)}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("playbook_invoke", payload, model=PlaybookInvokeResponse)
