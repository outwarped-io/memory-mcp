"""Pydantic schemas for the envs tool surface (extracted from server module envs)."""

from __future__ import annotations

import datetime as dt
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EnvCreateRequest(BaseModel):
    """Input schema for ``env_create``.

    ``default_embedding_model_id`` is required by the schema but the tool
    layer fills it from settings if absent.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    kind: str | None = Field(default=None, max_length=64)
    retention_policy: dict[str, Any] = Field(default_factory=dict)
    default_embedding_model_id: str | None = Field(default=None, max_length=200)


class EnvResponse(BaseModel):
    """Wire shape for an environment row."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    kind: str | None
    retention_policy: dict[str, Any]
    default_embedding_model_id: str
    created_at: Any  # datetime — kept as Any to avoid pydantic tz coercion in v1
    status: str = "active"
    deleted_at: dt.datetime | None = None


class AttachedEnvsResponse(BaseModel):
    """Wire shape returned by attach / detach / list-attached."""

    session_id: UUID
    attached: list[EnvResponse]
