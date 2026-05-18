"""Pydantic schemas for the ``mem_top`` tool surface (Phase 1, v0.14)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from memory_mcp_schemas.enums import MemoryKind, MemoryStatus
from memory_mcp_schemas.memories import MemoryResponse


MemTopBy = Literal["salience", "access_count", "reference_count", "reference_velocity"]
MemTopTagMatch = Literal["any", "all"]


class MemTopRequest(BaseModel):
    """Input schema for :func:`memory_top`.

    ``by`` selects the ranking metric. All metrics share the same stable
    tie-breaker: ``(metric DESC, created_at DESC, id DESC)`` — so two
    callers seeing the same data always get the same top-N.

    ``tag_match`` defaults to ``"any"`` (OR semantics) for parity with
    ``mem_search`` / ``mem_browse``; ``"all"`` (AND) is opt-in.

    ``velocity_window_days`` is only consulted when ``by="reference_velocity"``;
    for other metrics it is ignored.
    """

    model_config = ConfigDict(extra="forbid")

    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None

    by: MemTopBy = Field(
        default="salience",
        description="Ranking metric.",
    )
    kinds: list[MemoryKind] | None = None
    tags: list[str] | None = Field(
        default=None,
        description="Tag filter; see ``tag_match`` for semantics.",
    )
    tag_match: MemTopTagMatch = Field(
        default="any",
        description="OR (any) is the default; AND (all) requires every tag to be present.",
    )
    statuses: list[MemoryStatus] | None = Field(
        default=None,
        description="Status filter. Default: ``[active]`` — top-of-the-board is a live-only signal.",
    )

    velocity_window_days: int = Field(
        default=30, ge=1, le=365,
        description="Only used when ``by='reference_velocity'``.",
    )

    limit: int = Field(default=10, ge=1, le=100)


class MemTopItem(BaseModel):
    """One ranked memory + the metric value that placed it."""

    model_config = ConfigDict(extra="forbid")

    memory: MemoryResponse
    metric_value: float


class MemTopResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[MemTopItem]
    by: MemTopBy
    total_examined: int
    schema_version: int = 1
