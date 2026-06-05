"""Memory CRUD + search + graph + browse + provenance API namespace."""

from __future__ import annotations

import datetime as dt
from typing import Any
from uuid import UUID

from memory_mcp_schemas.browse import (
    MemBrowseRequest,
    MemBrowseResponse,
    MemFacetsRequest,
    MemFacetsResponse,
)
from memory_mcp_schemas.context_pack import ContextPackResponse
from memory_mcp_schemas.digest import DigestResponse, ResumeResponse
from memory_mcp_schemas.env_ops import (
    MemCopyRequest,
    MemCopyResponse,
    MemMoveRequest,
    MemMoveResponse,
)
from memory_mcp_schemas.graph import (
    MemNeighborsRequest,
    MemNeighborsResponse,
    MemRelatedRequest,
    MemRelatedResponse,
)
from memory_mcp_schemas.journal import JournalRequest
from memory_mcp_schemas.memories import (
    JournalResponse,
    MemoryHardDeleteRequest,
    MemoryHardDeleteResponse,
    MemoryResponse,
    MemorySupersedeRequest,
    MemorySupersedeResponse,
    MemoryUpdatePatch,
    MemoryWriteRequest,
)
from memory_mcp_schemas.provenance import (
    MemLineageRequest,
    MemLineageResponse,
    MemSourcesBrowseRequest,
    MemSourcesBrowseResponse,
)
from memory_mcp_schemas.search import (
    AutoContextResponse,
    MemorySearchRequest,
    MemorySearchResponse,
)

from memory_mcp_client._batch import BatchResult, run_bounded
from memory_mcp_client.api._base import _BaseAPI


class MemoriesAPI(_BaseAPI):
    """Memory-mcp memories namespace."""

    async def write(
        self,
        request: MemoryWriteRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemoryResponse:
        """Create a new memory."""
        if request is None:
            request = MemoryWriteRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_write", payload, model=MemoryResponse)

    async def write_many(
        self,
        items: list[MemoryWriteRequest],
        *,
        max_concurrency: int = 8,
    ) -> BatchResult[MemoryWriteRequest, MemoryResponse]:
        """Create many memories with bounded client-side concurrency."""

        return await run_bounded(items, self.write, max_concurrency=max_concurrency)

    async def get(
        self,
        memory_id: UUID | str,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> MemoryResponse:
        """Fetch a single memory by id."""
        payload: dict[str, Any] = {"memory_id": str(memory_id)}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_get", payload, model=MemoryResponse)

    async def get_many(
        self,
        memory_ids: list[UUID | str],
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> list[MemoryResponse]:
        """Bulk fetch memories by id."""
        payload: dict[str, Any] = {"memory_ids": [str(m) for m in memory_ids]}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_get_many", payload, model=MemoryResponse)

    async def update(
        self,
        memory_id: UUID | str,
        patch: MemoryUpdatePatch | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemoryResponse:
        """Patch a memory using optimistic concurrency control."""
        if patch is None:
            patch = MemoryUpdatePatch(**kwargs)
        payload: dict[str, Any] = {
            "memory_id": str(memory_id),
            "patch": patch.model_dump(mode="json", exclude_unset=True),
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_update", payload, model=MemoryResponse)

    async def archive(
        self,
        memory_id: UUID | str,
        expected_version: int,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> MemoryResponse:
        """Move a memory to the archived lifecycle state."""
        payload: dict[str, Any] = {
            "memory_id": str(memory_id),
            "expected_version": expected_version,
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_archive", payload, model=MemoryResponse)

    async def retire(
        self,
        memory_id: UUID | str,
        expected_version: int,
        reason: str,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> MemoryResponse:
        """Move a memory to the retired lifecycle state."""
        payload: dict[str, Any] = {
            "memory_id": str(memory_id),
            "expected_version": expected_version,
            "reason": reason,
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_retire", payload, model=MemoryResponse)

    async def supersede(
        self,
        old_memory_id: UUID | str,
        request: MemorySupersedeRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        attached_env_names: list[str] | None = None,
        **kwargs: Any,
    ) -> MemorySupersedeResponse:
        """Replace a memory with a successor; returns a typed ``{old, new}``.

        Returns :class:`MemorySupersedeResponse` so callers don't have to
        index a raw dict. ``response.old`` and ``response.new`` are
        :class:`MemoryResponse` instances.
        """
        if request is None:
            request = MemorySupersedeRequest(**kwargs)
        payload: dict[str, Any] = {
            "old_memory_id": str(old_memory_id),
            "request": request.model_dump(mode="json"),
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        if attached_env_names is not None:
            payload["attached_env_names"] = list(attached_env_names)
        raw = await self._call("mem_supersede", payload, model=None)
        return MemorySupersedeResponse.model_validate(raw)

    async def hard_delete(
        self,
        memory_id: UUID | str,
        request: MemoryHardDeleteRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        attached_env_names: list[str] | None = None,
        **kwargs: Any,
    ) -> MemoryHardDeleteResponse:
        """Permanently destroy a memory's canonical row, body, and projections.

        Required for the sensitive-write recovery protocol; ``mem_retire``
        is soft-delete only. Refs-guarded — caller must clean up
        dependents (lineage / superseded_by / graph nodes) first.

        Set ``request.confirm_destroy=True`` and provide a non-empty
        ``request.reason``. ``request.expected_version`` carries the
        optimistic-lock version; mismatch raises
        :class:`VersionConflictError`.
        """
        if request is None:
            request = MemoryHardDeleteRequest(**kwargs)
        payload: dict[str, Any] = {
            "memory_id": str(memory_id),
            "request": request.model_dump(mode="json"),
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        if attached_env_names is not None:
            payload["attached_env_names"] = list(attached_env_names)
        return await self._call(
            "mem_hard_delete",
            payload,
            model=MemoryHardDeleteResponse,
        )

    async def copy(
        self,
        request: MemCopyRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemCopyResponse:
        """Call mem_copy tool. See server-side docstring."""
        if request is None:
            request = MemCopyRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_copy_", payload, model=MemCopyResponse)

    async def move(
        self,
        request: MemMoveRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemMoveResponse:
        """Call mem_move tool. See server-side docstring."""
        if request is None:
            request = MemMoveRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_move_", payload, model=MemMoveResponse)

    async def journal(
        self,
        request: JournalRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        attached_env_names: list[str] | None = None,
        **kwargs: Any,
    ) -> JournalResponse:
        """Append a short-form observation and return the created memory.

        Returns :class:`JournalResponse` — a thin subclass of
        :class:`MemoryResponse` for SDK type-hint clarity. No new fields.
        """
        if request is None:
            request = JournalRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        if attached_env_names is not None:
            payload["attached_env_names"] = list(attached_env_names)
        return await self._call("mem_journal", payload, model=JournalResponse)

    async def digest(
        self,
        env_id: UUID | str,
        since_ts: dt.datetime | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> DigestResponse:
        """Summarize an environment into a persisted six-section digest."""
        payload: dict[str, Any] = {"env_id": str(env_id)}
        if since_ts is not None:
            payload["since_ts"] = since_ts.isoformat()
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_digest", payload, model=DigestResponse)

    async def resume(
        self,
        env_id: UUID | str,
        journal_tail: int = 20,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> ResumeResponse:
        """Return the latest digest, recent journal entries, and counts."""
        payload: dict[str, Any] = {
            "env_id": str(env_id),
            "journal_tail": journal_tail,
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_resume", payload, model=ResumeResponse)

    async def search(
        self,
        request: MemorySearchRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemorySearchResponse:
        """Search memories across attached environments."""
        if request is None:
            request = MemorySearchRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_search", payload, model=MemorySearchResponse)

    async def auto_context(
        self,
        task_desc: str,
        env_id: UUID | str,
        top_k: int = 8,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> AutoContextResponse:
        """Find memories whose trigger descriptions match the current task."""
        payload: dict[str, Any] = {
            "task_desc": task_desc,
            "env_id": str(env_id),
            "top_k": top_k,
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_auto_context", payload, model=AutoContextResponse)

    async def neighbors(
        self,
        request: MemNeighborsRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemNeighborsResponse:
        """Walk the projected graph from a starting memory."""
        if request is None:
            request = MemNeighborsRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_neighbors", payload, model=MemNeighborsResponse)

    async def related(
        self,
        request: MemRelatedRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemRelatedResponse:
        """Find memories related by shared entities or semantic similarity."""
        if request is None:
            request = MemRelatedRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_related", payload, model=MemRelatedResponse)

    async def lineage(
        self,
        request: MemLineageRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemLineageResponse:
        """Trace provenance lineage around a seed memory."""
        if request is None:
            request = MemLineageRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_lineage", payload, model=MemLineageResponse)

    async def sources_browse(
        self,
        request: MemSourcesBrowseRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemSourcesBrowseResponse:
        """Browse memory provenance sources."""
        if request is None:
            request = MemSourcesBrowseRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_sources_browse", payload, model=MemSourcesBrowseResponse)

    async def browse(
        self,
        request: MemBrowseRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemBrowseResponse:
        """Browse memories without relevance ranking."""
        if request is None:
            request = MemBrowseRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_browse", payload, model=MemBrowseResponse)

    async def facets(
        self,
        request: MemFacetsRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> MemFacetsResponse:
        """Aggregate memory facet counts."""
        if request is None:
            request = MemFacetsRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_facets", payload, model=MemFacetsResponse)

    async def context_pack(
        self,
        task_desc: str,
        env_id: UUID | str,
        token_budget: int = 4000,
        include_core: bool = True,
        include_journal: bool = True,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> ContextPackResponse:
        """Build a token-budgeted prompt bundle for a task."""
        payload: dict[str, Any] = {
            "task_desc": task_desc,
            "env_id": str(env_id),
            "token_budget": token_budget,
            "include_core": include_core,
            "include_journal": include_journal,
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("mem_context_pack", payload, model=ContextPackResponse)
