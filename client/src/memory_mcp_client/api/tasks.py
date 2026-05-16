"""Task tree + dependency + status API namespace."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from memory_mcp_client._batch import BatchResult, run_bounded
from memory_mcp_client.api._base import _BaseAPI
from memory_mcp_schemas.enums import TaskRelationKind, TaskStatus
from memory_mcp_schemas.tasks import (
    TaskCreateRequest,
    TaskLinkMemoryRequest,
    TaskLinkMemoryResponse,
    TaskListRequest,
    TaskListResponse,
    TaskRelationResponse,
    TaskResponse,
    TaskTreeResponse,
)


class TasksAPI(_BaseAPI):
    """Memory-mcp tasks namespace."""

    async def create(
        self,
        request: TaskCreateRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> TaskResponse:
        if request is None:
            request = TaskCreateRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("task_create", payload, model=TaskResponse)

    async def create_many(
        self,
        items: list[TaskCreateRequest],
        *,
        max_concurrency: int = 8,
    ) -> BatchResult[TaskCreateRequest, TaskResponse]:
        """Create many tasks with bounded client-side concurrency."""

        return await run_bounded(items, self.create, max_concurrency=max_concurrency)

    async def substep(
        self,
        parent_task_id: UUID | str,
        title: str,
        *,
        description: str | None = None,
        priority: int = 50,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> TaskResponse:
        payload: dict[str, Any] = {
            "parent_task_id": str(parent_task_id),
            "title": title,
            "description": description,
            "priority": priority,
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("task_substep", payload, model=TaskResponse)

    async def dep_link(
        self,
        src_task_id: UUID | str,
        dst_task_id: UUID | str,
        *,
        type: TaskRelationKind | str = TaskRelationKind.depends_on,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> TaskRelationResponse:
        payload: dict[str, Any] = {
            "src_task_id": str(src_task_id),
            "dst_task_id": str(dst_task_id),
            "type": type.value if isinstance(type, TaskRelationKind) else type,
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("task_dep_link", payload, model=TaskRelationResponse)

    async def status_set(
        self,
        task_id: UUID | str,
        status: TaskStatus | str,
        expected_version: int,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> TaskResponse:
        payload: dict[str, Any] = {
            "task_id": str(task_id),
            "status": status.value if isinstance(status, TaskStatus) else status,
            "expected_version": expected_version,
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("task_status_set", payload, model=TaskResponse)

    async def list(
        self,
        request: TaskListRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> TaskListResponse:
        if request is None:
            request = TaskListRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("task_list", payload, model=TaskListResponse)

    async def next(
        self,
        env_id: UUID | str,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> TaskResponse | None:
        payload: dict[str, Any] = {"env_id": str(env_id)}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        raw = await self._call("task_next", payload, model=None)
        if not raw:
            return None
        return TaskResponse.model_validate(raw)

    async def tree(
        self,
        task_id: UUID | str,
        *,
        max_depth: int = 10,
        max_nodes: int = 200,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
    ) -> TaskTreeResponse:
        payload: dict[str, Any] = {
            "task_id": str(task_id),
            "max_depth": max_depth,
            "max_nodes": max_nodes,
        }
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("task_tree", payload, model=TaskTreeResponse)

    async def link_memory(
        self,
        request: TaskLinkMemoryRequest | None = None,
        *,
        agent_id: UUID | str | None = None,
        attached_env_ids: list[UUID | str] | None = None,
        **kwargs: Any,
    ) -> TaskLinkMemoryResponse:
        if request is None:
            request = TaskLinkMemoryRequest(**kwargs)
        payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if attached_env_ids is not None:
            payload["attached_env_ids"] = [str(e) for e in attached_env_ids]
        return await self._call("task_link_memory", payload, model=TaskLinkMemoryResponse)
