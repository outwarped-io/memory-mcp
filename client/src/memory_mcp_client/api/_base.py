"""Shared base class for namespace APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_mcp_client.client import MemoryClient


class _BaseAPI:
    """Holds a reference to the parent :class:`MemoryClient`.

    Subclasses call ``self._call("tool_name", payload, model=ResponseModel)``
    to dispatch a single MCP tool invocation.
    """

    __slots__ = ("_client",)

    def __init__(self, client: "MemoryClient") -> None:
        self._client = client

    async def _call(
        self,
        tool: str,
        payload: dict[str, Any] | None = None,
        *,
        model: Any | None = None,
    ) -> Any:
        return await self._client._call(tool, payload, model=model)
