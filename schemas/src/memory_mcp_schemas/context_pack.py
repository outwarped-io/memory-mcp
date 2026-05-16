"""Pydantic response models for ``mem_context_pack``."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


ContextPackSectionName = Literal[
    "digest",
    "trigger_matched",
    "recent_journal",
    "tasks",
    "decisions",
    "playbooks",
    "archival",
]


class ContextPackHit(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    memory_id: UUID
    title: str
    body: str
    kind: str
    salience: float
    tokens_used: int
    body_truncated: bool


class ContextPackSection(BaseModel):
    name: ContextPackSectionName
    items: list[ContextPackHit]
    tokens_used: int
    cap_tokens: int
    truncation_count: int


class ContextPackResponse(BaseModel):
    sections: list[ContextPackSection]
    total_tokens: int
    budget_used_pct: float = Field(ge=0.0)
    sections_skipped: list[str]
    task_desc_used: str
