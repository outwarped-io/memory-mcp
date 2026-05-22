"""Pydantic schemas for ``mem_compose`` (N→1 caller-driven aggregation).

Phase 2 of the popularity / compose / decompose track (v0.15.0). The
surface mirrors the internal dream handlers ``_accept_merge`` and
``_accept_promotion`` but bypasses the ``DreamProposal`` envelope so a
caller (agent, SDK user, automation) can compose directly.

Two modes:

* ``promote`` (default, non-destructive) — sources stay ``active``; the
  new memory cites them via ``LineageRelation.promoted_from``. Used for
  summarisation / abstraction.
* ``merge`` (destructive) — sources transition to ``superseded`` with
  ``superseded_by`` pointing at the new memory; the new memory cites
  them via ``LineageRelation.supersedes``. Used for true aggregation.

The narrow :class:`MemComposeTarget` model is **not** a thin alias for
:class:`MemoryWriteRequest`: the latter carries fields (``env_id`` /
``env_name`` / ``source_type`` / ``source_ref`` / ``entity_links`` /
``macro`` / ``steps``) whose semantics either don't make sense for a
composed memory or would silently override server-side bookkeeping.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memory_mcp_schemas._env_refs import validate_optional_env_ref_pair
from memory_mcp_schemas.enums import MemoryKind
from memory_mcp_schemas.memories import MemoryResponse


# ---------------------------------------------------------------------------
# Tag-policy literal
# ---------------------------------------------------------------------------

ComposeMode = Literal["promote", "merge"]
ComposeTagPolicy = Literal["target", "union", "target_plus_union"]


# ---------------------------------------------------------------------------
# Target payload
# ---------------------------------------------------------------------------

class MemComposeTarget(BaseModel):
    """Narrow target payload accepted by ``mem_compose``.

    Distinct from :class:`MemoryWriteRequest` to avoid silent acceptance
    of fields that compose deliberately controls server-side (env,
    source_type, entity_links). Provenance of the merged row is always
    ``MemorySourceType.agent`` with ``source_ref`` derived from the
    dedupe key.
    """

    model_config = ConfigDict(extra="forbid")

    kind: MemoryKind
    title: str | None = Field(default=None, max_length=400)
    body: str = Field(min_length=1)
    trigger_description: str | None = None
    tags: list[str] | None = Field(
        default=None,
        description=(
            "Tag set for the merged memory. ``None`` defers to ``tag_policy`` "
            "(per-mode default). ``[]`` (empty list) means *no target tags*; "
            "the policy still resolves source tags (e.g. union) on top."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    salience: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    pinned: bool = False
    expires_at: dt.datetime | None = None
    decision_meta: dict[str, Any] | None = None

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if not all(isinstance(t, str) for t in v):
            raise ValueError("tags must be strings")
        return v


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class MemComposeRequest(BaseModel):
    """Combine N source memories (≥2) into a single new memory.

    The dedupe key (server-computed sha256 over a canonical-JSON
    envelope; or caller-provided ``idempotency_key``) makes retries
    idempotent: a replay returns the same composed memory with
    ``idempotency_replay=true`` and performs no further mutation.
    """

    model_config = ConfigDict(extra="forbid")

    source_ids: list[UUID] = Field(
        min_length=2,
        max_length=20,
        description=(
            "IDs of memories to compose. Must contain at least 2 distinct "
            "ids and at most 20 (cap on transaction size + MCP response "
            "size). All sources must belong to the same env."
        ),
    )
    target: MemComposeTarget
    mode: ComposeMode = Field(
        default="promote",
        description=(
            "``promote`` (default, non-destructive) → sources stay active, "
            "lineage relation ``promoted_from``. ``merge`` (destructive) → "
            "sources transition to ``superseded`` with ``superseded_by`` "
            "set to the new memory, lineage relation ``supersedes``."
        ),
    )
    expected_versions: dict[UUID, int] | None = Field(
        default=None,
        description=(
            "Optimistic-lock check. Map of source id → expected version. "
            "Mismatch raises ``VERSION_CONFLICT``. Bypassed on idempotent "
            "replay (the original call already succeeded)."
        ),
    )
    tag_policy: ComposeTagPolicy | None = Field(
        default=None,
        description=(
            "How to resolve effective tags on the merged memory. Defaults: "
            "``merge``→``target_plus_union``, ``promote``→``target``. "
            "``target`` = only ``target.tags``; ``union`` = union of source "
            "tags (ignoring target); ``target_plus_union`` = ``target.tags`` "
            "∪ union(source tags)."
        ),
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Caller-supplied dedupe handle. When present, used verbatim as "
            "the server's dedupe key instead of the sha256 hash. Useful for "
            "distributed retries where the caller has its own request id."
        ),
    )
    env_id: UUID | None = Field(
        default=None,
        description=(
            "Optional env scope assertion. When supplied, server checks "
            "that all sources belong to this env; mismatch raises "
            "``CROSS_ENV_NOT_ALLOWED``."
        ),
    )
    env_name: str | None = None

    @field_validator("source_ids")
    @classmethod
    def _no_duplicates(cls, v: list[UUID]) -> list[UUID]:
        if len(v) != len(set(v)):
            raise ValueError("source_ids contains duplicates")
        return v

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "MemComposeRequest":
        return validate_optional_env_ref_pair(self)

    @model_validator(mode="after")
    def _validate_expected_versions(self) -> "MemComposeRequest":
        if self.expected_versions:
            unknown = set(self.expected_versions) - set(self.source_ids)
            if unknown:
                raise ValueError(
                    "expected_versions contains ids not in source_ids: "
                    + ", ".join(str(u) for u in sorted(unknown))
                )
        return self


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class ComposeLineageRow(BaseModel):
    """One lineage edge emitted by ``mem_compose`` (parent=source, child=new)."""

    model_config = ConfigDict(extra="forbid")

    parent_memory_id: UUID
    child_memory_id: UUID
    relation: Literal["promoted_from", "supersedes"]


class MemComposeResponse(BaseModel):
    """Result of ``mem_compose``."""

    model_config = ConfigDict(extra="forbid")

    memory: MemoryResponse
    mode: ComposeMode
    source_ids: list[UUID] = Field(
        description="Echo of the sorted source ids the server actually composed."
    )
    lineage_rows: list[ComposeLineageRow] = Field(
        description=(
            "One row per source (parent=source, child=new memory) with the "
            "mode-appropriate relation."
        ),
    )
    retired_source_ids: list[UUID] = Field(
        default_factory=list,
        description=(
            "Populated only when ``mode='merge'``. Sources whose status "
            "transitioned to ``superseded`` with ``superseded_by=memory.id``."
        ),
    )
    auto_wired: list[UUID] = Field(
        default_factory=list,
        description=(
            "Populated only when Phase 4 auto-wire is enabled. Memory ids "
            "the server linked via ``rel_link(type='related_to_popular')`` "
            "in the same transaction. Always empty in v0.15.0 Phase 2."
        ),
    )
    idempotency_replay: bool = Field(
        default=False,
        description=(
            "True when the call matched an existing dedupe key and returned "
            "the previously-composed memory without further mutation."
        ),
    )
    tag_policy_applied: ComposeTagPolicy = Field(
        description="Effective tag policy (caller override or per-mode default)."
    )
    dedupe_key: str = Field(
        description=(
            "Stable dedupe key for this composition. Either the sha256-hex "
            "head (32 chars) computed by the server or the caller-supplied "
            "``idempotency_key``. Persisted on the merged memory row."
        ),
    )
