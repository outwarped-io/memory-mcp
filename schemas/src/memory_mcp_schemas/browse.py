"""Pydantic schemas for the browse tool surface (extracted from server module browse)."""

from __future__ import annotations

import datetime as dt
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from memory_mcp_schemas.enums import MemoryKind, MemoryStatus
from memory_mcp_schemas.memories import MemoryResponse


BrowseOrderField = Literal["updated_at", "created_at"]


FacetField = Literal["kind", "status", "tag", "month"]


class MemBrowseRequest(BaseModel):
    """Input schema for :func:`memory_browse`.

    Filter parity with :class:`MemorySearchRequest`. NO ``query`` field —
    browse is deterministic-ordered listing, not relevance-ranked search.
    """

    model_config = ConfigDict(extra="forbid")

    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None
    kinds: list[MemoryKind] | None = None
    tags: list[str] | None = Field(
        default=None,
        description="ANY of these tags must be present (OR semantics; mirrors mem_search).",
    )
    statuses: list[MemoryStatus] | None = Field(
        default=None,
        description="Status filter. Default: ``[proposed, active]``.",
    )
    created_after: dt.datetime | None = None
    created_before: dt.datetime | None = None
    updated_after: dt.datetime | None = None

    order_by: BrowseOrderField = "updated_at"
    descending: bool = True

    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = Field(default=None, max_length=4096)
    include_expired: bool = Field(
        default=False,
        description=(
            "v0.17 default-tightening. When False (default), memories whose "
            "``expires_at`` has passed are excluded from results. Included "
            "in the cursor fingerprint, so paging across different flag "
            "values raises ``cursor mismatch``."
        ),
    )


class MemBrowseResponse(BaseModel):
    hits: list[MemoryResponse]
    next_cursor: str | None
    has_more: bool
    schema_version: int = 1


class FacetBucket(BaseModel):
    value: str
    count: int


class MemFacetsRequest(BaseModel):
    """Input schema for :func:`memory_facets`.

    The default facet set (``kind``, ``status``, ``tag``) covers the
    cold-start "what's in this env?" pre-flight. ``month`` is opt-in
    because it adds a date_trunc grouping that scales with row count
    more than the others.
    """

    model_config = ConfigDict(extra="forbid")

    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None
    facets: list[FacetField] = Field(
        default_factory=lambda: ["kind", "status", "tag"],
        description="Which facets to compute.",
    )
    tag_limit: int = Field(default=50, ge=1, le=500)
    statuses: list[MemoryStatus] | None = None
    kinds: list[MemoryKind] | None = None
    tags: list[str] | None = None
    created_after: dt.datetime | None = None
    created_before: dt.datetime | None = None
    updated_after: dt.datetime | None = None

    accuracy: Literal["exact", "approximate"] = "exact"
    max_rows: int = Field(default=100_000, ge=1_000)
    include_expired: bool = Field(
        default=False,
        description=(
            "v0.17 default-tightening. When False (default), memories whose "
            "``expires_at`` has passed are excluded from facet counts."
        ),
    )


class MemFacetsResponse(BaseModel):
    total: int
    by_env: dict[UUID, int]
    facets: dict[str, list[FacetBucket]]
    approximate: bool = False
    sampled_rows: int = 0
    schema_version: int = 1
