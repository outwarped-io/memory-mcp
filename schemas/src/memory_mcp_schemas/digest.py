"""Pydantic models for session digest and resume tools."""

from __future__ import annotations

import datetime as dt
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memory_mcp_schemas._env_refs import validate_optional_env_ref_pair, validate_required_env_ref_pair


class DigestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    since_ts: dt.datetime | None = None

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "DigestRequest":
        return validate_required_env_ref_pair(self)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    journal_tail: int = 20

    @field_validator("journal_tail")
    @classmethod
    def _clamp_tail(cls, value: int) -> int:
        return min(200, max(0, value))

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "ResumeRequest":
        return validate_required_env_ref_pair(self)


class DigestSections(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: str = ""
    active_context: str = ""
    system_patterns: str = ""
    tech_context: str = ""
    progress: str = ""
    open_questions: str = ""


class DigestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: UUID
    sections: DigestSections
    summarizer_kind: str
    source_type: str


class DigestMemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    env_id: UUID
    kind: str
    title: str | None = None
    body: str
    salience: float
    created_at: dt.datetime
    updated_at: dt.datetime


class ResumeStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_count: int
    entity_count: int
    last_journal_ts: dt.datetime | None


class ResumeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latest_digest: DigestSections | None
    recent_journal: list[DigestMemoryEntry] = Field(default_factory=list)
    summary_stats: ResumeStats
