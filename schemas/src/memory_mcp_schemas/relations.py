"""Pydantic schemas for the relations tool surface (extracted from server module relations)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memory_mcp_schemas._env_refs import validate_optional_env_ref_list_pair


_MAX_TYPE_FILTER_VALUES = 20


class RelationEndpoint(BaseModel):
    """A typed endpoint of a relation: ``(kind, id)`` pair.

    ``kind`` selects the source table (``entity`` or ``memory``); ``id``
    is the canonical record id. The matching ``graph_nodes`` row is
    looked up (and created if absent) on the fly.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["entity", "memory", "task"]
    id: UUID


class RelationLinkRequest(BaseModel):
    """Create or update a relation edge.

    Idempotent on ``(src_node_id, dst_node_id, type)``; passing the
    same ``properties`` twice is a no-op.
    """

    model_config = ConfigDict(extra="forbid")

    src: RelationEndpoint
    dst: RelationEndpoint
    type: str = Field(min_length=1, max_length=200)
    properties: dict[str, Any] = Field(default_factory=dict)
    env_id: UUID | None = None
    env_name: str | None = None
    expected_version: int | None = Field(default=None, ge=1)

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("type cannot be blank")
        return v


class RelationResponse(BaseModel):
    """Wire shape returned by relation tools."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    env_id: UUID
    src: RelationEndpoint
    dst: RelationEndpoint
    src_node_id: UUID
    dst_node_id: UUID
    type: str
    properties: dict[str, Any]
    version: int
    created_at: dt.datetime
    updated_at: dt.datetime


class RelationBrowseRequest(BaseModel):
    """Input schema for :func:`relation_browse`.

    Endpoint pinning is optional: pass ``src_id`` and/or ``dst_id`` to
    restrict to relations touching a specific entity/memory; pass
    ``src_kind`` / ``dst_kind`` to restrict by endpoint kind without
    pinning a specific id. All filters compose (AND).
    """

    model_config = ConfigDict(extra="forbid")

    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None
    types: list[str] | None = Field(
        default=None,
        max_length=_MAX_TYPE_FILTER_VALUES,
        description="Edge-type filter (max 20 distinct values).",
    )
    src_kind: Literal["entity", "memory", "task"] | None = None
    dst_kind: Literal["entity", "memory", "task"] | None = None
    src_id: UUID | None = None
    dst_id: UUID | None = None
    created_after: dt.datetime | None = None

    descending: bool = True
    limit: int = Field(default=100, ge=1, le=500)
    cursor: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "RelationBrowseRequest":
        return validate_optional_env_ref_list_pair(self)

    @model_validator(mode="after")
    def _require_kind_with_id(self) -> "RelationBrowseRequest":
        # ``graph_nodes`` enforces uniqueness on ``(node_type, record_id)``
        # — entity_ids and memory_ids share the UUID namespace but a given
        # id only collides across kinds in pathological cases. Even so,
        # callers who pin an id without a kind are very likely confused
        # about *which* node they meant. Fail fast.
        if self.src_id is not None and self.src_kind is None:
            raise ValueError("src_kind is required when src_id is provided")
        if self.dst_id is not None and self.dst_kind is None:
            raise ValueError("dst_kind is required when dst_id is provided")
        return self


class RelationBrowseHit(BaseModel):
    """A single relation row in :class:`RelationBrowseResponse`.

    ``src_id`` / ``dst_id`` carry the canonical record id (entity.id or
    memory.id) — NOT the graph_node id — so callers can chain into
    ``ent_neighbors`` or ``mem_get`` without a second lookup.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    env_id: UUID
    type: str
    src_kind: Literal["entity", "memory", "task"]
    src_id: UUID
    dst_kind: Literal["entity", "memory", "task"]
    dst_id: UUID
    properties: dict[str, Any]
    created_at: dt.datetime
    updated_at: dt.datetime


class RelationBrowseResponse(BaseModel):
    hits: list[RelationBrowseHit]
    next_cursor: str | None
    has_more: bool
    schema_version: int = 1
