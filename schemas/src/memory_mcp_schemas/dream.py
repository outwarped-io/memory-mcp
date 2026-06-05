"""Pydantic schemas for the dream tool surface (extracted from server module dream.api)."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_mcp_schemas._env_refs import validate_optional_env_ref_pair

from memory_mcp_schemas.memories import MemoryResponse


class DreamMode(StrEnum):
    """Pass identifiers — the ``dream_runs.mode`` column stores these."""

    decay = "decay"
    dedupe = "dedupe"
    promote = "promote"
    decision_conflicts = "decision_conflicts"
    recount = "recount"


class DreamPassOutcome(StrEnum):
    done = "done"
    failed = "failed"
    skipped = "skipped"
    cancelled = "cancelled"


DreamReviewAction = Literal["accept", "reject", "defer", "amend"]


DreamProposalKind = Literal[
    "merge_candidate",
    "promotion_candidate",
    "decay_candidate",
    "decision_conflict_candidate",
]


DreamProposalStatus = Literal[
    "open",
    "accepted",
    "rejected",
    "amended",
    "deferred",
    "expired",
]


class DreamRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    modes: list[DreamMode] | None = None
    wait: bool = False
    triggered_by: Literal["tool", "test"] = "tool"

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "DreamRunRequest":
        return validate_optional_env_ref_pair(self)


class DreamRunScheduledItem(BaseModel):
    env_id: UUID
    mode: DreamMode


class DreamRunReport(BaseModel):
    """Wire-friendly mirror of :class:`DreamPassReport`."""

    env_id: UUID
    mode: DreamMode
    outcome: DreamPassOutcome
    dream_run_id: UUID | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    last_error: str | None = None
    duration_seconds: float = 0.0


class DreamRunResponse(BaseModel):
    scheduled: list[DreamRunScheduledItem] = Field(default_factory=list)
    reports: list[DreamRunReport] = Field(default_factory=list)


class DreamRunSummaryEntry(BaseModel):
    """Slim view of a ``dream_runs`` row for status responses."""

    id: UUID
    env_id: UUID
    mode: DreamMode
    status: str
    started_at: dt.datetime
    ended_at: dt.datetime | None
    triggered_by: str
    summarizer_kind: str | None
    summary: dict[str, Any]
    last_error: str | None


class DreamHeartbeatEntry(BaseModel):
    sink: str
    env_id: UUID
    last_success_at: dt.datetime | None
    lag_seconds: float | None
    status: str | None
    last_error: str | None


class DreamStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    runs_per_mode: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "DreamStatusRequest":
        return validate_optional_env_ref_pair(self)


class DreamStatusResponse(BaseModel):
    last_runs: list[DreamRunSummaryEntry]
    open_proposal_counts: dict[str, int]
    summarizer_kind: str
    llm_backend: str
    llm_status: dict[str, Any]
    heartbeats: list[DreamHeartbeatEntry]


class DreamProposalsListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    status: DreamProposalStatus | None = None
    kind: DreamProposalKind | None = None
    summarizer_kind: Literal["llm", "template"] | None = None
    limit: int = Field(default=20, ge=1, le=200)
    cursor: str | None = None

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "DreamProposalsListRequest":
        return validate_optional_env_ref_pair(self)


class DreamProposalEntry(BaseModel):
    id: UUID
    env_id: UUID
    kind: DreamProposalKind
    status: DreamProposalStatus
    summarizer_kind: str | None
    llm_failed: bool
    payload: dict[str, Any]
    dream_run_id: UUID | None
    created_at: dt.datetime
    updated_at: dt.datetime
    reviewed_at: dt.datetime | None
    reviewed_by_agent_id: UUID | None
    review_action: str | None
    review_notes: str | None


class DreamProposalsListResponse(BaseModel):
    items: list[DreamProposalEntry]
    next_cursor: str | None = None


class DreamReviewPatch(BaseModel):
    """Optional caller overrides applied during ``accept``."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    body: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class DreamReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: UUID
    action: DreamReviewAction
    notes: str | None = None
    patch: DreamReviewPatch | None = None
    # Optional concurrency-safety override: maps memory_id → expected version.
    # When provided, the accept handler verifies each row's current version
    # before mutating it; mismatch raises ``VERSION_CONFLICT``. When omitted,
    # the handler proceeds with whatever version the FOR UPDATE lock returned
    # — race-prone if a concurrent writer slipped in between proposal
    # emission and review.
    expected_versions: dict[UUID, int] | None = None


class DreamReviewResponse(BaseModel):
    proposal: DreamProposalEntry
    accepted_memory: MemoryResponse | None = None
    superseded_memory_ids: list[UUID] = Field(default_factory=list)
