"""Pydantic schemas for ADR-lite decisions."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from memory_mcp_schemas.enums import DecisionStatus


class DecisionMeta(BaseModel):
    """Structured metadata attached to ``kind=decision`` memories."""

    model_config = ConfigDict(extra="forbid")

    status: DecisionStatus
    rationale: str
    constraints: list[str]
    consequences: list[str] | None = None
    superseded_by: UUID | None

    @field_validator("rationale")
    @classmethod
    def _rationale_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("rationale must be a non-empty string")
        return stripped

    @field_validator("constraints")
    @classmethod
    def _constraints_stripped(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("constraints entries must be non-empty strings")
            out.append(stripped)
        return out

    @field_validator("consequences")
    @classmethod
    def _consequences_stripped(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        out: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("consequences entries must be non-empty strings")
            out.append(stripped)
        return out

    @model_validator(mode="after")
    def _status_superseded_by_coupling(self) -> "DecisionMeta":
        if self.status == DecisionStatus.superseded:
            if self.superseded_by is None:
                raise ValueError("status='superseded' requires superseded_by")
        elif self.superseded_by is not None:
            raise ValueError("superseded_by is only valid when status='superseded'")
        return self


class AdrExportResponse(BaseModel):
    """Markdown export for a decision memory."""

    markdown: str
    status: str | None
    memory_id: UUID
