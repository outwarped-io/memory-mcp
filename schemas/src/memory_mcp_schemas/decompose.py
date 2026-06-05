"""Pydantic schemas for ``mem_decompose`` (1→N caller-driven decomposition).

Phase 3 of the popularity / compose / decompose track (v0.15.0). Surface
mirrors :mod:`memory_mcp_schemas.compose` so an agent that already knows
``mem_compose`` can reach for ``mem_decompose`` without learning a new
shape vocabulary.

Two modes:

* ``derive`` (default, non-destructive) — source stays ``active``; each
  new child cites the source via ``LineageRelation.derived_from``. Used
  for atomic-fact extraction, evidence-leaf splitting, summary expansion.
* ``split`` (destructive) — source transitions to ``retired``; each new
  child cites the source via ``LineageRelation.split_from``. Used when
  the source was a wrong granularity from the start and should disappear
  from search after the decomposition.

The narrow :class:`MemDecomposeChild` model is **not** a thin alias for
:class:`MemoryWriteRequest`: the latter carries fields (``env_id`` /
``env_name`` / ``source_type`` / ``source_ref`` / ``entity_links`` /
``macro`` / ``steps``) whose semantics either don't make sense for a
decomposed child or would silently override server-side bookkeeping.

Lineage relations (``split_from`` / ``derived_from``) are NOT in the
load-bearing popularity whitelist for ``split_from`` (migration 0021,
Stage C1.5 redirect E.11) — splitting a retired source should not bump
its analytics. ``derived_from`` IS in the whitelist — a derive-mode
parent is a conceptual originator of its atomic derivatives.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memory_mcp_schemas.enums import MemoryKind
from memory_mcp_schemas.memories import MemoryResponse


# ---------------------------------------------------------------------------
# Mode literal
# ---------------------------------------------------------------------------

DecomposeMode = Literal["split", "derive"]


# ---------------------------------------------------------------------------
# Child payload
# ---------------------------------------------------------------------------


class MemDecomposeChild(BaseModel):
    """Narrow child payload accepted by ``mem_decompose``.

    Distinct from :class:`MemoryWriteRequest` to avoid silent acceptance
    of fields that decompose deliberately controls server-side (env,
    source_type, entity_links). Provenance of each child is always
    ``MemorySourceType.agent`` with ``source_ref`` derived from the
    decompose operation id (C0 source map, RD G note).
    """

    model_config = ConfigDict(extra="forbid")

    kind: MemoryKind
    title: str | None = Field(default=None, max_length=400)
    body: str = Field(min_length=1)
    trigger_description: str | None = None
    tags: list[str] | None = Field(
        default=None,
        description=(
            "Tag set for this child memory. ``None`` defers to server "
            "defaults; ``[]`` means intentionally no tags."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    salience: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    pinned: bool = False
    expires_at: dt.datetime | None = None
    decision_meta: dict[str, Any] | None = None

    @field_validator("kind")
    @classmethod
    def _no_playbook(cls, v: MemoryKind) -> MemoryKind:
        # Playbook is a structured kind with its own ``steps`` field on
        # MemoryWriteRequest; the narrow child schema does not expose
        # ``steps``, so a playbook child would land malformed. Reject
        # at schema time (RD G; Stage C5 wiring).
        if v == MemoryKind.playbook:
            raise ValueError(
                "kind=playbook is not allowed for decompose children "
                "(no ``steps`` field in MemDecomposeChild)"
            )
        return v

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


class MemDecomposeRequest(BaseModel):
    """Decompose one source memory into N children (2 ≤ N ≤ 20).

    The dedupe key (server-computed sha256 over a canonical-JSON
    envelope; or caller-provided ``idempotency_key``) makes retries
    idempotent: a replay returns the same children with
    ``idempotency_replay=true`` and performs no further mutation.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: UUID = Field(
        description=(
            "ID of the memory to decompose. Must resolve to a memory in "
            "an env the caller has visibility into."
        ),
    )
    children: list[MemDecomposeChild] = Field(
        min_length=2,
        max_length=20,
        description=(
            "Target children. Must contain at least 2 entries (a 1→1 "
            "transform is ``mem_supersede``) and at most 20 (cap on "
            "transaction size + MCP response size)."
        ),
    )
    mode: DecomposeMode = Field(
        default="derive",
        description=(
            "``derive`` (default, non-destructive) → source stays active, "
            "lineage relation ``derived_from``. ``split`` (destructive) → "
            "source transitions to ``retired``, lineage relation "
            "``split_from``. Mirrors compose's promote/merge polarity."
        ),
    )
    expected_version: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optimistic-lock check against ``source.version``. Mismatch "
            "raises ``VERSION_CONFLICT``. Bypassed on idempotent replay "
            "(the original call already succeeded)."
        ),
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Caller-supplied dedupe handle. When present, used verbatim "
            "as the server's dedupe key instead of the sha256 hash. "
            "Useful for distributed retries where the caller has its own "
            "request id."
        ),
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class DecomposeLineageRow(BaseModel):
    """One lineage edge emitted by ``mem_decompose`` (parent=source, child=new)."""

    model_config = ConfigDict(extra="forbid")

    parent_memory_id: UUID
    child_memory_id: UUID
    relation: Literal["split_from", "derived_from"]


class MemDecomposeResponse(BaseModel):
    """Result of ``mem_decompose``."""

    model_config = ConfigDict(extra="forbid")

    source: MemoryResponse = Field(
        description=(
            "Post-decomposition source state. For ``mode='split'`` the "
            "status is ``retired``; for ``mode='derive'`` the source is "
            "unchanged. Always reflects what's in the DB after the "
            "transaction commits (or what was in the DB at the original "
            "call's commit, on replay)."
        ),
    )
    children: list[MemoryResponse] = Field(
        description=(
            "Newly-inserted children in the order they were provided in "
            "the request. On replay, the same children are returned "
            "(reconstructed from the decompose_operations row)."
        ),
    )
    mode: DecomposeMode
    lineage_rows: list[DecomposeLineageRow] = Field(
        description=(
            "One row per child (parent=source, child=new memory) with "
            "the mode-appropriate relation (``split_from`` or "
            "``derived_from``)."
        ),
    )
    auto_wired: list[UUID] = Field(
        default_factory=list,
        description=(
            "Flat ordered-unique union of all auto-wired dst memory "
            "ids across every child. Built by iterating children in "
            "insertion order and de-duplicating dst ids on first "
            "occurrence. Empty when the feature is disabled or every "
            "child's candidate fan-out was empty. For per-child mapping "
            "see ``auto_wired_by_child``. Kept as a flat list for "
            "backward-compatibility with v0.15.x callers."
        ),
    )
    auto_wired_by_child: dict[UUID, list[UUID]] | None = Field(
        default=None,
        description=(
            "v0.16+ per-child auto-wire mapping.\n"
            "\n"
            "* ``None`` — feature OFF on **first write only** (master "
            "switch or per-decompose switch disabled). Replay NEVER "
            "returns ``None`` — replay always populates as a per-child "
            "dict reflecting current relations state.\n"
            "* ``{child_id: []}`` — feature ON but that child had no "
            "edges. Covers: candidate fan-out empty above threshold, "
            "Stage-A failure (degraded silently to per-child empties), "
            "Stage-B per-child savepoint rollback on insert failure, OR "
            "replay of a row that had no wired edges.\n"
            "* ``{child_id: [dst_id, ...]}`` — populated mapping. The "
            "per-child list contains the actually-inserted dst memory "
            "ids in deterministic order; ``ON CONFLICT DO NOTHING`` may "
            "have absorbed duplicate edges from concurrent operations.\n"
            "\n"
            "Replay reconstructs from current ``relations`` table state "
            "(matches ``mem_compose``'s state-current semantic) — a "
            "manual ``rel_link(type='related_to_popular')`` issued from "
            "any child after the original decompose WILL surface here on "
            "replay. The flat ``auto_wired`` field is always the "
            "ordered-unique union of ``auto_wired_by_child.values()`` "
            "when this field is populated."
        ),
    )
    idempotency_replay: bool = Field(
        default=False,
        description=(
            "True when the call matched an existing dedupe key and "
            "returned the previously-decomposed children without "
            "further mutation."
        ),
    )
    dedupe_key: str = Field(
        description=(
            "Stable dedupe key for this decomposition. Either the "
            "sha256-hex head (32 chars) computed by the server or the "
            "caller-supplied ``idempotency_key``. Persisted on the "
            "decompose_operations row."
        ),
    )
    operation_id: UUID = Field(
        description=(
            "ID of the ``decompose_operations`` row that records this "
            "decomposition. Stable across replays. Used as the "
            "``source_ref`` for each child's ``MemorySource(source_type="
            "'agent')`` provenance row."
        ),
    )
