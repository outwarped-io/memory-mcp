"""Pydantic schemas for the graph tool surface (extracted from server module graph)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from memory_mcp_schemas._env_refs import validate_optional_env_ref_pair

from memory_mcp_schemas.memories import MemoryResponse


class EntityNeighborsRequest(BaseModel):
    """Input schema for :func:`entity_neighbors`.

    Fields support the **plan's documented public names** as input
    aliases (``id`` → ``entity_id``, ``types`` → ``edge_types``,
    ``kind`` → singular form translated to internal ``kinds``) so MCP
    clients can use either spelling.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    entity_id: UUID = Field(
        validation_alias=AliasChoices("entity_id", "id"),
        description="Canonical entity id to traverse from.",
    )
    hops: int = Field(default=1, ge=1, le=3)
    edge_types: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("edge_types", "types"),
        max_length=20,
        description="Restrict traversal to these relation types. None = all.",
    )
    kind: Literal["entity", "memory", "both"] = Field(
        default="both",
        description=(
            "Filter the **terminal** node kind. ``both`` (default) returns "
            "either. Path-transit nodes are NOT filtered by ``kind``."
        ),
    )
    direction: Literal["out", "in", "both"] = Field(
        default="both",
        description="Edge traversal direction relative to the start entity.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Pre-filter cap on backend results. Lifecycle filtering may "
            "leave the response with fewer hits — paginate via "
            "``next_cursor``."
        ),
    )
    cursor: str | None = Field(
        default=None,
        max_length=4096,
        description=(
            "Opaque pagination cursor returned by a previous call. Bound "
            "to the original query shape; mismatches raise INVALID_CURSOR."
        ),
    )
    env_id: UUID | None = Field(
        default=None,
        description=(
            "Optional sanity check: must match the entity's env. A "
            "mismatch raises NOT_FOUND (env is never leaked through the "
            "error)."
        ),
    )
    env_name: str | None = None
    consistency: Literal["default", "fresh"] = Field(
        default="default",
        description=(
            "Read-after-write semantics. ``default`` reads the projected "
            "graph as-is (eventually consistent). ``fresh`` waits up to "
            "``settings.search_fresh_max_wait_seconds`` for the graph "
            "projection sink to catch up to the env's outbox watermark "
            "before traversing; on timeout the response is still served "
            "but may miss recent writes. Only meaningful when the graph "
            "backend is the projected Neo4j store; with the Postgres "
            "recursive-CTE fallback the canonical truth is read directly "
            "and ``consistency`` has no effect."
        ),
    )

    @field_validator("edge_types")
    @classmethod
    def _check_edge_types(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        out: list[str] = []
        for raw in v:
            t = raw.strip()
            if not t:
                raise ValueError("edge_types entries must be non-empty")
            if len(t) > 200:
                raise ValueError(
                    "edge_types entries must be <= 200 characters"
                )
            out.append(t)
        return out

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "EntityNeighborsRequest":
        return validate_optional_env_ref_pair(self)


class NeighborNodeResponse(BaseModel):
    """A single graph node materialized for the wire response."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["entity", "memory"]
    id: UUID
    name: str | None = Field(
        description=(
            "``canonical_name`` for entity nodes; ``title`` for memory "
            "nodes. ``None`` when the canonical row is missing (rare; "
            "would indicate eventual-consistency drift)."
        ),
    )
    env_id: UUID


class NeighborPathStepResponse(BaseModel):
    """One real-edge step in a neighbor's path.

    ``src`` and ``dst`` are the actual relation endpoints, **not** the
    traversal direction. To render a human-readable walk, clients can
    chain steps by matching successive nodes; for ``direction="in"``
    the chain reads from terminal back to start.
    """

    model_config = ConfigDict(extra="forbid")

    src_kind: Literal["entity", "memory"]
    src_id: UUID
    dst_kind: Literal["entity", "memory"]
    dst_id: UUID
    edge_type: str


class NeighborHitResponse(BaseModel):
    """A neighbor returned by :func:`entity_neighbors`."""

    model_config = ConfigDict(extra="forbid")

    node: NeighborNodeResponse
    path_length: int = Field(ge=1)
    path: list[NeighborPathStepResponse]
    score: float | None = None


class EntityNeighborsResponse(BaseModel):
    """Wire shape returned by :func:`entity_neighbors`."""

    model_config = ConfigDict(extra="forbid")

    hits: list[NeighborHitResponse]
    next_cursor: str | None = None


class MemNeighborsRequest(BaseModel):
    """Input schema for :func:`memory_neighbors`.

    Memory-rooted mirror of :class:`EntityNeighborsRequest`. Walks the
    projected graph starting from a ``memory_id`` instead of an entity
    id; everything else (hops, edge_types, direction, terminal kind
    filter, cursor protocol, consistency semantics) matches
    ``entity_neighbors`` exactly.

    NotFoundError is raised if the memory has no ``graph_nodes`` row
    yet — that means it was never registered as an endpoint of any
    relation. Use ``mem_sources_browse`` / ``mem_lineage`` for that
    memory's provenance instead.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    memory_id: UUID = Field(
        validation_alias=AliasChoices("memory_id", "id"),
        description="Canonical memory id to traverse from.",
    )
    hops: int = Field(default=1, ge=1, le=3)
    edge_types: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("edge_types", "types"),
        max_length=20,
        description="Restrict traversal to these relation types. None = all.",
    )
    kind: Literal["entity", "memory", "both"] = Field(
        default="both",
        description="Filter the **terminal** node kind. Path-transit nodes are NOT filtered.",
    )
    direction: Literal["out", "in", "both"] = Field(default="both")
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=4096)
    env_id: UUID | None = Field(
        default=None,
        description="Optional sanity check: must match the memory's env. Mismatch raises NotFound.",
    )
    env_name: str | None = None
    consistency: Literal["default", "fresh"] = Field(default="default")
    fallback: bool = Field(
        default=False,
        description=(
            "Retry an empty traversal with progressively looser graph constraints. "
            "Steps in order: ``widen_hops`` (up to 3), ``drop_predicate``, "
            "then ``include_retired``. The response field ``fallback_used`` "
            "lists the steps that fired."
        ),
    )

    @field_validator("edge_types")
    @classmethod
    def _check_edge_types(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        out: list[str] = []
        for raw in v:
            t = raw.strip()
            if not t:
                raise ValueError("edge_types entries must be non-empty")
            if len(t) > 200:
                raise ValueError(
                    "edge_types entries must be <= 200 characters"
                )
            out.append(t)
        return out

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "MemNeighborsRequest":
        return validate_optional_env_ref_pair(self)


class MemNeighborsResponse(BaseModel):
    """Wire shape returned by :func:`memory_neighbors`.

    Same shape as :class:`EntityNeighborsResponse` for caller-side symmetry.
    """

    model_config = ConfigDict(extra="forbid")

    hits: list[NeighborHitResponse]
    next_cursor: str | None = None
    fallback_used: list[str] = Field(default_factory=list)


class MemRelatedRequest(BaseModel):
    """Input schema for :func:`memory_related`.

    Two modes:

    * ``shared_entity`` — memories that reference at least one entity in common
      with the seed, ranked by overlap count descending. Ties use
      ``memory.updated_at DESC, memory.id DESC``. Cursor stability has the
      documented keyset caveat: overlap counts can drift between page calls, so
      agents should dedupe by memory id.

    * ``semantic`` — top-K nearest vectors in Qdrant using the seed memory's
       stored embedding. This costs zero embedder calls. Missing embeddings or
       vector-store outages return an explanatory ``note`` rather than falling
       back to fresh embedding. v1 does not support cursor pagination for this
       mode; increase ``limit`` (max 500) to retrieve more results.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: UUID
    relation: Literal["shared_entity", "semantic"] = "shared_entity"
    limit: int = Field(default=20, ge=1, le=500)
    cursor: str | None = Field(default=None, max_length=4096)
    min_score: float | None = Field(default=None, ge=0, le=1)
    fallback: bool = Field(
        default=False,
        description=(
            "Retry an empty result with progressively looser graph constraints. "
            "Steps in order: ``widen_hops``, ``drop_predicate``, then "
            "``include_retired``; steps that actually fired are echoed in "
            "``fallback_used``. In the current relation set only ``include_retired`` "
            "can change the query."
        ),
    )
    env_id: UUID | None = None
    env_name: str | None = None

    @model_validator(mode="after")
    def _check_min_score_relation(self) -> MemRelatedRequest:
        if self.min_score is not None and self.relation != "semantic":
            raise ValueError(
                f"min_score is only valid when relation='semantic'; got relation='{self.relation}'"
            )
        return self

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "MemRelatedRequest":
        return validate_optional_env_ref_pair(self)


class MemRelatedHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: UUID
    score: float
    shared_entity_ids: list[UUID] | None = None
    memory: MemoryResponse


class MemRelatedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[MemRelatedHit]
    next_cursor: str | None = None
    note: Literal["ok", "no_embedding", "vector_store_unavailable"] = "ok"
    fallback_used: list[str] = Field(default_factory=list)
