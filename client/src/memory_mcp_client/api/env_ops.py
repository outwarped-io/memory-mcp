"""Environment operations API namespace."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from memory_mcp_client.api._base import _BaseAPI
from memory_mcp_schemas.env_ops import (
    EnvCloneRequest,
    EnvCloneResponse,
    EnvDeleteRequest,
    EnvDeleteResponse,
    EnvDiffRequest,
    EnvDiffResponse,
    EnvExportRequest,
    EnvExportResponse,
    EnvImportReport,
    EnvImportRequest,
    EnvMergeRequest,
    EnvMergeResponse,
    EnvMigrateRequest,
    EnvMigrateResponse,
    EnvRenameRequest,
    EnvRenameResponse,
    EnvRestoreRequest,
    EnvRestoreResponse,
    EnvSnapshotRequest,
    EnvSnapshotResponse,
)


class EnvOpsAPI(_BaseAPI):
    """Client API for env_ops tools."""

    def _request_payload(
        self,
        request: Any,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return payload

    async def export(
        self,
        request: EnvExportRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvExportResponse:
        """Call env_export tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvExportRequest(**kwargs)
        return await self._call(
            "env_export_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvExportResponse,
        )

    async def import_(
        self,
        request: EnvImportRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvImportReport:
        """Call env_import tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvImportRequest(**kwargs)
        return await self._call(
            "env_import_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvImportReport,
        )

    async def diff(
        self,
        request: EnvDiffRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvDiffResponse:
        """Call env_diff tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvDiffRequest(**kwargs)
        return await self._call(
            "env_diff_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvDiffResponse,
        )

    async def clone(
        self,
        request: EnvCloneRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvCloneResponse:
        """Call env_clone tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvCloneRequest(**kwargs)
        return await self._call(
            "env_clone_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvCloneResponse,
        )

    async def merge(
        self,
        request: EnvMergeRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvMergeResponse:
        """Call env_merge tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvMergeRequest(**kwargs)
        return await self._call(
            "env_merge_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvMergeResponse,
        )

    async def migrate(
        self,
        request: EnvMigrateRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvMigrateResponse:
        """Call env_migrate tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvMigrateRequest(**kwargs)
        return await self._call(
            "env_migrate_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvMigrateResponse,
        )

    async def snapshot(
        self,
        request: EnvSnapshotRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvSnapshotResponse:
        """Call env_snapshot tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvSnapshotRequest(**kwargs)
        return await self._call(
            "env_snapshot_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvSnapshotResponse,
        )

    async def restore(
        self,
        request: EnvRestoreRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvRestoreResponse:
        """Call env_restore tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvRestoreRequest(**kwargs)
        return await self._call(
            "env_restore_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvRestoreResponse,
        )

    async def delete(
        self,
        request: EnvDeleteRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvDeleteResponse:
        """Call env_delete tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvDeleteRequest(**kwargs)
        return await self._call(
            "env_delete_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvDeleteResponse,
        )

    async def rename(
        self,
        request: EnvRenameRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> EnvRenameResponse:
        """Call env_rename tool. See server-side docstring for semantics."""
        if request is None:
            request = EnvRenameRequest(**kwargs)
        return await self._call(
            "env_rename_",
            self._request_payload(
                request, agent_id=agent_id, attached_env_ids=attached_env_ids
            ),
            model=EnvRenameResponse,
        )
