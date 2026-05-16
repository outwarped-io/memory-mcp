"""Pydantic schemas for task tree tools."""

from __future__ import annotations

import datetime as dt
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memory_mcp_schemas._env_refs import validate_required_env_ref_pair

from memory_mcp_schemas.enums import TaskRelationKind, TaskStatus


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    title: str = Field(min_length=1)
    description: str | None = None
    priority: int = Field(default=50, ge=1, le=100)
    playbook_id: UUID | None = None

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "TaskCreateRequest":
        return validate_required_env_ref_pair(self)


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    env_id: UUID
    title: str
    description: str | None
    status: TaskStatus
    priority: int
    playbook_id: UUID | None
    version: int
    created_at: dt.datetime
    updated_at: dt.datetime


class TaskListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    status: TaskStatus | None = None
    priority_max: int | None = Field(default=None, ge=1, le=100)
    cursor: str | None = None
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "TaskListRequest":
        return validate_required_env_ref_pair(self)


class TaskListResponse(BaseModel):
    hits: list[TaskResponse]
    next_cursor: str | None


class TaskTreeLine(BaseModel):
    depth: int
    task_id: UUID
    status: str
    desc: str
    version: int


class TaskTreeResponse(BaseModel):
    root_id: UUID
    lines: list[TaskTreeLine]
    truncated: bool
    total_visited: int


class TaskRelationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src_task_id: UUID
    dst_task_id: UUID
    type: TaskRelationKind = TaskRelationKind.depends_on


class TaskRelationResponse(BaseModel):
    src_task_id: UUID
    dst_task_id: UUID
    type: TaskRelationKind
    created_at: dt.datetime


class TaskLinkMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    memory_id: UUID
    relation: TaskRelationKind

    @field_validator("relation")
    @classmethod
    def _validate_relation(cls, v: TaskRelationKind) -> TaskRelationKind:
        if v not in {
            TaskRelationKind.motivated_by,
            TaskRelationKind.produces,
            TaskRelationKind.references,
        }:
            raise ValueError("relation must be motivated_by, produces, or references")
        return v


class TaskLinkMemoryResponse(BaseModel):
    relation_id: UUID
    task_id: UUID
    memory_id: UUID
    relation: TaskRelationKind
    created_at: dt.datetime
