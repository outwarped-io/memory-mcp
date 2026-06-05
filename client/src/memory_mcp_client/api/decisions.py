"""Decision/ADR export API namespace."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from memory_mcp_schemas.decisions import AdrExportResponse

from memory_mcp_client.api._base import _BaseAPI


class DecisionsAPI(_BaseAPI):
    """Memory-mcp decisions namespace."""

    async def adr_export(
        self,
        memory_id: UUID | str,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> AdrExportResponse:
        """Export an ADR/decision memory as markdown."""
        payload: dict[str, Any] = {"memory_id": str(memory_id)}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(env_id) for env_id in attached_env_ids]
        return await self._call("adr_export", payload, model=AdrExportResponse)
