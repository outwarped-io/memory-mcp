"""Pydantic schemas for the memories tool surface (extracted from server module memories)."""

from __future__ import annotations

import datetime as dt
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memory_mcp_schemas.enums import MemoryKind, MemorySourceType, MemoryStatus


class MemoryWriteRequest(BaseModel):
    """Create a new memory.

    ``env_id`` is optional: if the caller has exactly one attached env, it
    is inferred. Multiple attached envs without an explicit choice raises
    :class:`EnvAmbiguousError`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: MemoryKind
    title: str | None = Field(default=None, max_length=400)
    body: str = Field(min_length=1)
    trigger_description: str | None = None
    steps: list[str] | None = None
    macro: str | None = None
    env_id: UUID | None = None
    env_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    salience: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    expires_at: dt.datetime | None = None
    pinned: bool = False
    entity_links: list[UUID] = Field(default_factory=list)
    source_type: MemorySourceType = MemorySourceType.agent
    source_ref: str | None = Field(default=None, max_length=2000)
    evidence_span: str | None = Field(default=None, max_length=4000)
    decision_meta: dict[str, Any] | None = None

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, v: list[str]) -> list[str]:
        # Schema-level shape only — :func:`_normalize_tags` does the real cleanup
        # since it's also called by ``memory_update``.
        if not all(isinstance(t, str) for t in v):
            raise ValueError("tags must be strings")
        return v


class MemorySupersedeRequest(BaseModel):
    """Old memory + new memory body in a single atomic call.

    ``new`` carries the same shape as :class:`MemoryWriteRequest` but
    ``env_id`` is forced to match the old memory's env (cross-env
    supersede is admin-only and deferred to v1.5).
    """

    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    new: MemoryWriteRequest


class MemoryUpdatePatch(BaseModel):
    """Patch payload for ``mem_update``.

    Field absence means no change; explicit ``None`` clears nullable fields.
    """

    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    title: str | None = Field(default=None, max_length=400)
    body: str | None = Field(default=None, min_length=1)
    trigger_description: str | None = None
    steps: list[str] | None = None
    macro: str | None = None
    kind: MemoryKind | None = None
    status: MemoryStatus | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    salience: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    pinned: bool | None = None
    expires_at: dt.datetime | None = None
    verified_at: dt.datetime | None = None
    decision_meta: dict[str, Any] | None = None


class MemoryResponse(BaseModel):
    """Wire shape returned by all memory tools."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    env_id: UUID
    kind: MemoryKind
    status: MemoryStatus
    title: str | None
    body: str
    trigger_description: str | None = None
    steps: list[str] | None = None
    macro: str | None = None
    tags: list[str]
    metadata: dict[str, Any]
    salience: float
    confidence: float
    pinned: bool
    access_count: int
    last_accessed_at: dt.datetime | None
    negative_feedback_count: int
    verified_at: dt.datetime | None
    expires_at: dt.datetime | None
    superseded_by: UUID | None
    decision_meta: dict[str, Any] | None = None
    version: int
    created_at: dt.datetime
    updated_at: dt.datetime

    reference_count: int = 0
    reference_breakdown: dict[str, int] = Field(
        default_factory=lambda: {"rel_link": 0, "lineage": 0, "task": 0, "playbook": 0}
    )
    reference_authority: float = 0.0
    reference_velocity: int | None = None


class MemorySupersedeResponse(BaseModel):
    """Wire shape returned by ``mem_supersede``.

    Returns the *old* memory (now ``status="superseded"``, ``superseded_by``
    pointing at the new id) and the *new* memory (freshly inserted in
    either the same env, or the destination env when ``cross_env=True``).

    The MCP tool returns this as a plain dict
    ``{"old": <MemoryResponse>, "new": <MemoryResponse>}`` over the wire;
    callers that want a typed handle can validate against this model on
    the receiving side.
    """

    model_config = ConfigDict(extra="forbid")

    old: MemoryResponse
    new: MemoryResponse


class JournalResponse(MemoryResponse):
    """Wire shape returned by ``mem_journal``.

    ``memory_journal`` is a thin wrapper around ``memory_write`` that
    forces ``kind=MemoryKind.observation`` and stamps a journal-entry
    source-type. The response is byte-for-byte identical to
    :class:`MemoryResponse`; the dedicated subclass exists so SDK callers
    can name the return type without re-using the more general
    :class:`MemoryResponse`. No new fields — the type difference is
    purely documentation.
    """

    model_config = ConfigDict(from_attributes=True)


class MemoryHardDeleteRequest(BaseModel):
    """Permanently delete a memory's canonical row, body, and projections.

    Hard delete is a saga, not an atomic operation. The Postgres canonical
    commit happens synchronously; projection eviction (Qdrant, Neo4j)
    is enqueued via the outbox and observed asynchronously by the
    projection worker. The response surfaces the projection-eviction
    state so callers know whether to retry, wait, or treat the call as
    complete.

    A tombstone row (``memory_tombstones``) is written in the same
    transaction. Callers cannot recover the deleted body afterwards —
    rotate any leaked secrets before relying on this call as a
    mitigation. See ``memory-mcp.instructions.md §14``.

    Caller must pass ``confirm_destroy=True`` to acknowledge the
    irreversibility; otherwise the request raises ``INVALID_INPUT``.
    """

    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(
        ...,
        description=(
            "Optimistic-lock version of the canonical memory row. "
            "Read it from ``MemoryResponse.version`` of the most recent "
            "fetch. Mismatch raises ``VERSION_CONFLICT``."
        ),
    )
    reason: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=("Free-text justification recorded on the tombstone for audit. Required."),
    )
    confirm_destroy: bool = Field(
        ...,
        description=(
            "Must be ``true``. The server refuses to hard-delete without "
            "this acknowledgement so accidental destruction is impossible."
        ),
    )
    wait_for_projection: bool = Field(
        default=False,
        description=(
            "When true, the call blocks (up to a server-configured ceiling) "
            "until Qdrant and Neo4j projection eviction completes. Default "
            "false — the call returns as soon as the canonical commit "
            "lands, and ``projection_eviction`` reports the pending state."
        ),
    )
    cascade: bool = Field(
        default=False,
        description="Opt in to deleting forward-lineage dependents in the same operation.",
    )
    max_cascade_depth: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum forward-lineage depth allowed when ``cascade=true``.",
    )
    max_cascade_count: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum total rows affected when ``cascade=true``.",
    )
    dry_run: bool = Field(
        default=False,
        description="When true, report the rows that would be deleted without mutating state.",
    )


class HardDeleteProjectionStatus(BaseModel):
    """Per-sink eviction state at response time.

    ``completed`` — the projection worker has confirmed the sink no
    longer holds the row.
    ``pending`` — the outbox event is enqueued but not yet observed by
    the worker. Re-poll with ``projection_status`` if needed.
    ``failed`` — the worker hit a non-retryable error. Operator
    intervention required; check ``projection_status`` for details.
    """

    qdrant: str = Field(..., description="One of: completed, pending, failed.")
    neo4j: str = Field(..., description="One of: completed, pending, failed.")


class MemoryHardDeleteAffected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    lifecycle_before: str
    edge_reason: str
    version: int
    depth: int


class MemoryHardDeleteResponse(BaseModel):
    """Result of ``mem_hard_delete``."""

    model_config = ConfigDict(from_attributes=True)

    deleted_id: UUID = Field(..., description="The memory id that was targeted for hard-delete.")
    deleted_at: dt.datetime | None = Field(
        default=None,
        description=(
            "Canonical Postgres commit time. Same row appears in "
            "``memory_tombstones.deleted_at``. Absent on ``dry_run=true``."
        ),
    )
    canonical_deleted: bool = Field(
        ...,
        description=("True when the canonical rows were deleted. ``false`` on dry-run responses."),
    )
    projection_eviction: HardDeleteProjectionStatus | None = Field(
        default=None,
        description=(
            "Per-sink eviction state. When ``wait_for_projection=True`` "
            "was passed and the server completed the wait, both sinks "
            "should report ``completed``. Absent on ``dry_run=true``."
        ),
    )
    tombstone_id: UUID | None = Field(
        default=None,
        description=(
            "Id of the persisted root ``memory_tombstones`` row. Useful for "
            "audit queries and for the ``see tombstone <id>`` hint that "
            "future ``mem_get`` calls will return. Absent on ``dry_run=true``."
        ),
    )
    cascade_root: UUID | None = Field(
        default=None,
        description="Correlation id shared by all rows in a cascade operation.",
    )
    affected: list[MemoryHardDeleteAffected] = Field(
        default_factory=list,
        description="Ordered list of rows affected by the delete (leaves first, root last).",
    )


MemHardDeleteRequest = MemoryHardDeleteRequest
MemHardDeleteResponse = MemoryHardDeleteResponse
