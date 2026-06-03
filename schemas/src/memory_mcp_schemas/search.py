"""Pydantic schemas for the search tool surface.

``MemorySearchRequest.env_names`` is resolved server-side to ``env_ids``;
callers may provide either UUIDs or friendly names, not both.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_mcp_schemas.enums import MemoryKind
from memory_mcp_schemas.memories import MemoryResponse


class ExpansionPreset(str, Enum):
    narrow = "narrow"
    default = "default"
    broad = "broad"


SearchMode = Literal["auto", "hybrid", "lex", "sem", "graph", "id"]


ConsistencyMode = Literal["default", "fresh", "canonical"]


_EXPANSION_MUTEX_FIELDS: tuple[str, ...] = (
    "min_score",
    "fallback",
    "follow_superseded",
    "include_stale",
    "include_archived",
    "include_retired",
)


class MemorySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None
    kinds: list[MemoryKind] | None = None
    tags: list[str] | None = None
    created_after: dt.datetime | None = None
    created_before: dt.datetime | None = None
    updated_after: dt.datetime | None = None
    mode: SearchMode = "hybrid"
    limit: int = Field(default=10, ge=1, le=200)
    expansion: ExpansionPreset | None = None
    include_stale: bool = False
    include_archived: bool = False
    include_retired: bool = False
    include_expired: bool = Field(
        default=False,
        description=(
            "v0.17 default-tightening. When False (default), memories whose "
            "``expires_at`` has passed are excluded from results. Set True "
            "to include them (admin/debug). Independent of the lifecycle "
            "flags above and intentionally NOT part of the ``expansion`` "
            "preset bundle — ``include_expired`` may be combined with any "
            "``expansion`` value."
        ),
    )
    follow_superseded: bool = True
    consistency: ConsistencyMode = "default"
    ids: list[UUID] | None = None  # used by mode=id
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Post-fusion score threshold (the *tighten* lever). Hits with "
            "``score < min_score`` are dropped before truncation. Useful "
            "values are typically in ``0.005..0.05`` for the default RRF "
            "+ salience boost; the empirical 50th-percentile fused score "
            "on real traffic is ~0.016 and the 90th-percentile is ~0.035. "
            "Combines with ``fallback`` — if the threshold empties the "
            "result set, the fallback ladder treats it as 0 hits and "
            "continues broadening."
        ),
    )
    fallback: bool = Field(
        default=False,
        description=(
            "Auto-broaden cascade on empty results (the *loosen* lever). "
            "When True and the initial search returns 0 hits (or all hits "
            "are filtered out by ``min_score``), the server re-runs the "
            "query with progressively broader scope and returns the first "
            "non-empty pass. Steps in order: "
            "(1) widen ``mode`` from ``lex`` to ``hybrid`` (no-op if mode "
            "is already broader); "
            "(2) drop optional filters ``kinds`` / ``tags`` / time bounds; "
            "(3) widen lifecycle (``include_stale`` + ``include_archived``); "
            "(4) drop ``follow_superseded`` and boost ``limit`` to "
            "``min(limit*5, 100)``. "
            "Each step is gated on the prior pass returning 0 hits. The "
            "response field ``fallback_used`` lists the steps that fired."
        ),
    )

    @model_validator(mode="after")
    def _validate_expansion_mutex(self) -> "MemorySearchRequest":
        if self.expansion is None:
            return self
        offending = [field for field in _EXPANSION_MUTEX_FIELDS if field in self.model_fields_set]
        if offending:
            fields = ", ".join(offending)
            raise ValueError(
                f"expansion cannot be combined with explicit override(s) for {fields}; "
                "expansion is a bundle preset",
            )
        return self


class MemorySearchHit(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    memory: MemoryResponse
    score: float
    sources: list[str]
    raw_scores: dict[str, float]


class ProjectionStatusEntry(BaseModel):
    env_id: UUID
    sink: str
    last_event_id: int | None
    lag_seconds: float | None
    status: str | None


class MemorySearchResponse(BaseModel):
    hits: list[MemorySearchHit]
    mode: SearchMode
    # Mode actually executed; differs from `mode` when auto dispatch or
    # canonical fallback rewrites the request (e.g. auto→hybrid, sem→lex).
    effective_mode: SearchMode
    consistency_used: ConsistencyMode  # may differ from requested (e.g. fresh→canonical)
    projection_status: list[ProjectionStatusEntry]
    truncated: bool = False
    fallback_used: list[str] = Field(
        default_factory=list,
        description=(
            "Names of broadening steps that fired when ``fallback=True``. "
            "Empty when the original query already returned hits or "
            "``fallback`` was not requested. Step names: "
            "``mode->hybrid``, ``drop_filters``, ``widen_lifecycle``, "
            "``boost_limit``."
        ),
    )
    expansion_resolved: dict[str, Any] | None = None


class AutoContextHit(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    memory_id: UUID
    title: str
    body: str
    trigger_description: str
    score: float
    salience: float
    kind: str


class AutoContextResponse(BaseModel):
    hits: list[AutoContextHit]
    task_desc_used: str
