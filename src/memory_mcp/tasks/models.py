"""Backward-compat shim: re-export DTOs from memory_mcp_schemas.tasks."""

from __future__ import annotations

from memory_mcp_schemas.tasks import *  # noqa: F401,F403
from memory_mcp_schemas.tasks import (
    TaskCreateRequest,
    TaskResponse,
    TaskListRequest,
    TaskListResponse,
    TaskTreeLine,
    TaskTreeResponse,
    TaskRelationRequest,
    TaskRelationResponse,
    TaskLinkMemoryRequest,
    TaskLinkMemoryResponse,
)

__all__ = ['TaskCreateRequest', 'TaskResponse', 'TaskListRequest', 'TaskListResponse', 'TaskTreeLine', 'TaskTreeResponse', 'TaskRelationRequest', 'TaskRelationResponse', 'TaskLinkMemoryRequest', 'TaskLinkMemoryResponse']
