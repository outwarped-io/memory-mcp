"""Pydantic schemas for the provenance tool surface (extracted from server module provenance)."""

from __future__ import annotations

import datetime as dt
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_mcp_schemas._env_refs import validate_optional_env_ref_list_pair, validate_optional_env_ref_pair

from memory_mcp_schemas.memories import MemoryResponse


class MemLineageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: UUID
    direction: Literal["ancestors", "descendants", "both"] = "both"
    relations: list[str] | None = Field(
        default=None,
        max_length=20,
        description="Filter to specific lineage relations (e.g. ['supersedes', 'promoted_from']). None = all.",
    )
    max_depth: int = Field(default=10, ge=1, le=50)
    max_edges: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Max total edges returned across ancestors + descendants combined before truncation.",
    )
    env_id: UUID | None = Field(
        default=None,
        description="Optional sanity check; mismatch raises NotFound (env never leaked).",
    )
    env_name: str | None = None

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "MemLineageRequest":
        return validate_optional_env_ref_pair(self)


class MemLineageEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_memory_id: UUID
    child_memory_id: UUID
    relation: str
    created_at: dt.datetime
    depth: int = Field(ge=1)


class MemLineageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: MemoryResponse
    ancestors: list[MemLineageEdge]
    descendants: list[MemLineageEdge]
    nodes: dict[UUID, MemoryResponse]
    truncated: bool


class MemSourcesBrowseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None
    memory_ids: list[UUID] | None = Field(default=None, max_length=100)
    source_types: list[str] | None = Field(default=None, max_length=20)
    source_refs: list[str] | None = Field(default=None, max_length=100)
    agent_ids: list[UUID] | None = Field(default=None, max_length=50)
    created_after: dt.datetime | None = None
    created_before: dt.datetime | None = None
    hydrate_memories: bool = False
    descending: bool = True
    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "MemSourcesBrowseRequest":
        return validate_optional_env_ref_list_pair(self)


class MemSourceHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    memory_id: UUID
    env_id: UUID
    source_type: str
    source_ref: str | None
    agent_id: UUID | None
    created_at: dt.datetime
    evidence_span: str | None


class MemSourcesBrowseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[MemSourceHit]
    next_cursor: str | None = None
    nodes: dict[UUID, MemoryResponse] | None = None
