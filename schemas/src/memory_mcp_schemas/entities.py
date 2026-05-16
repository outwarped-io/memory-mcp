"""Pydantic schemas for the entities tool surface (extracted from server module entities)."""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memory_mcp_schemas._env_refs import validate_optional_env_ref_list_pair


_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


_WS_RE = re.compile(r"\s+")


def _normalize_name(name: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace.

    Used for both canonical names and aliases so a name matches across
    case, punctuation, and whitespace variations. Returns the empty
    string only if input is purely whitespace/punctuation — caller
    should reject that upstream.
    """
    s = unicodedata.normalize("NFKC", name).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _validate_name(value: str, field: str) -> str:
    if not value or not _normalize_name(value):
        raise ValueError(f"{field} cannot be empty after normalization")
    return value


EntityOrderField = Literal["canonical_name", "created_at"]


class EntityUpsertRequest(BaseModel):
    """Create or update an entity by ``(env_id, normalized_canonical_name)``.

    ``expected_version`` is required when updating an existing row;
    omitted (``None``) means "create-or-no-op-update". A mismatch raises
    :class:`VersionConflictError`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1, max_length=64)
    canonical_name: str = Field(min_length=1, max_length=400)
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    env_id: UUID | None = None
    env_name: str | None = None
    expected_version: int | None = Field(default=None, ge=1)

    @field_validator("canonical_name")
    @classmethod
    def _check_canonical(cls, v: str) -> str:
        return _validate_name(v, "canonical_name")

    @field_validator("aliases")
    @classmethod
    def _check_aliases(cls, v: list[str]) -> list[str]:
        for a in v:
            if not isinstance(a, str):
                raise ValueError("aliases must be strings")
            if not a or not _normalize_name(a):
                raise ValueError("alias cannot be empty after normalization")
            if len(a) > 400:
                raise ValueError("alias too long (max 400)")
        return v


class EntityResolveRequest(BaseModel):
    """Look up entities by name (matches canonical OR any alias)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None
    kinds: list[str] | None = None
    limit: int = Field(default=20, ge=1, le=200)


class EntityMergeRequest(BaseModel):
    """Merge ``merge_ids`` entities into ``keep_id``.

    ``expected_versions`` carries the expected version for *every* entity
    involved (keep + each merge id). Mismatch on any → :class:`VersionConflictError`.
    """

    model_config = ConfigDict(extra="forbid")

    keep_id: UUID
    merge_ids: list[UUID] = Field(min_length=1)
    expected_versions: dict[UUID, int]

    @field_validator("merge_ids")
    @classmethod
    def _check_no_duplicates(cls, v: list[UUID]) -> list[UUID]:
        if len(v) != len(set(v)):
            raise ValueError("merge_ids must be unique")
        return v


class EntityResponse(BaseModel):
    """Wire shape returned by entity tools."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    env_id: UUID
    kind: str
    canonical_name: str
    normalized_name: str
    aliases: list[str]
    metadata: dict[str, Any]
    version: int
    created_at: dt.datetime
    updated_at: dt.datetime


class EntityBrowseRequest(BaseModel):
    """Input schema for :func:`entity_browse`.

    Filter + ordering parity with :class:`EntityResolveRequest` where it
    overlaps; adds keyset pagination + optional ``name_prefix`` filter.
    """

    model_config = ConfigDict(extra="forbid")

    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None
    kinds: list[str] | None = None
    name_prefix: str | None = Field(
        default=None,
        min_length=1,
        max_length=400,
        description=(
            "Prefix match against the entity's normalized canonical name "
            "OR any normalized alias. Case/punctuation-insensitive."
        ),
    )

    order_by: EntityOrderField = "canonical_name"
    descending: bool = False

    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "EntityBrowseRequest":
        return validate_optional_env_ref_list_pair(self)


class EntityBrowseResponse(BaseModel):
    hits: list[EntityResponse]
    next_cursor: str | None
    has_more: bool
    schema_version: int = 1
