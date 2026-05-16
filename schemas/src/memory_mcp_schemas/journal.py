"""Pydantic schemas for the journal tool surface (extracted from server module journal)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class JournalRequest(BaseModel):
    """Input schema for :func:`memory_journal`.

    ``env_id`` is optional and resolved by the underlying
    :func:`memory_write`: explicit wins, sole attached env is inferred,
    multiple-without-explicit raises :class:`EnvAmbiguousError`.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="The journal entry body.")
    env_id: UUID | None = None
    env_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    salience: float | None = Field(default=None, ge=0.0, le=1.0)
