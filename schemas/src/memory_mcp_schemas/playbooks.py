"""Response models for playbook invocation."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from memory_mcp_schemas.memories import MemoryResponse


class PlaybookInvokeResponse(BaseModel):
    """Resolved playbook invocation payload."""

    model_config = ConfigDict(from_attributes=True)

    playbook: MemoryResponse
    steps: list[str]
    referenced_memories: list[MemoryResponse]
    missing_refs: list[UUID]
    missing_task_refs: list[UUID] = Field(default_factory=list)
