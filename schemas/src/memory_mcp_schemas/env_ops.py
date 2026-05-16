"""Pydantic schemas for the v0.8 environment operations tool surface."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_mcp_schemas._env_refs import validate_required_env_ref_pair
from memory_mcp_schemas.browse import MemBrowseRequest
from memory_mcp_schemas.envs import EnvResponse
from memory_mcp_schemas.memories import MemoryResponse

BrowseFilter = MemBrowseRequest


class ExportFormat(str, Enum):
    """Export container format."""

    archive = "archive"
    directory = "directory"


class SourceMetadata(BaseModel):
    """Source environment metadata captured in an export manifest."""

    model_config = ConfigDict(extra="forbid")

    env_id: UUID
    env_name: str
    default_embedding_model_id: str
    instance_fingerprint: str


class IncludeFlags(BaseModel):
    """Feature flags included in an export."""

    model_config = ConfigDict(extra="forbid")

    embeddings: bool
    provenance: bool
    dream_history: bool
    grants: bool


class ExportManifest(BaseModel):
    """Manifest written alongside an environment export."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="0.8.0", pattern=r"^0\.8\.0$")
    memory_mcp_version: str
    source: SourceMetadata
    exported_at: dt.datetime
    exported_by_agent: str | None
    include_flags: IncludeFlags
    counts: dict[str, int]
    checksums: dict[str, str]


class MemoryVectorRecord(BaseModel):
    """One named-vector record in embeddings/memory_vectors.jsonl.

    Backend-neutral. Qdrant stores named vectors (body, optional trigger);
    pgvector future-backend can write a single ``body`` record. Row order
    in the file is NOT significant — records are keyed by ``memory_id`` +
    ``vector_name`` and consumers call ``VectorStore.upsert`` per record.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: UUID
    memory_version: int = Field(ge=1)
    model_id: str = Field(min_length=1)
    vector_name: Literal["body", "trigger"]
    dimension: int = Field(ge=1)
    vector: list[float]

    @model_validator(mode="after")
    def _vector_matches_dimension(self) -> MemoryVectorRecord:
        if len(self.vector) != self.dimension:
            raise ValueError(f"vector length {len(self.vector)} != dimension {self.dimension}")
        return self


class RemapTable(BaseModel):
    """Maps source-archive UUIDs to destination-env UUIDs during import/merge.

    Every table that an exported memory or related row references by UUID
    must appear here. Two-pass import sets ``memories`` first (with
    ``superseded_by=NULL``), then UPDATEs ``superseded_by`` using the
    remap in a second pass.
    """

    model_config = ConfigDict(extra="forbid")

    memories: dict[UUID, UUID] = Field(default_factory=dict)
    entities: dict[UUID, UUID] = Field(default_factory=dict)
    entity_aliases: dict[UUID, UUID] = Field(default_factory=dict)
    tasks: dict[UUID, UUID] = Field(default_factory=dict)
    tags: dict[UUID, UUID] = Field(default_factory=dict)
    graph_nodes: dict[UUID, UUID] = Field(default_factory=dict)
    relations: dict[UUID, UUID] = Field(default_factory=dict)
    memory_sources: dict[UUID, UUID] = Field(default_factory=dict)
    memory_lineage: dict[UUID, UUID] = Field(default_factory=dict)
    dream_runs: dict[UUID, UUID] = Field(default_factory=dict)
    dream_proposals: dict[UUID, UUID] = Field(default_factory=dict)


class ImportMode(str, Enum):
    """Import conflict behavior."""

    fail = "fail"
    skip = "skip"
    overwrite = "overwrite"
    merge = "merge"


class EntityMergeStrategy(str, Enum):
    """Entity matching strategy for env merge."""

    by_canonical_key = "by_canonical_key"
    by_id = "by_id"
    keep_both = "keep_both"


class TagMergeStrategy(str, Enum):
    """Tag conflict strategy for env merge."""

    union = "union"
    src_wins = "src_wins"
    dst_wins = "dst_wins"


class DiffGranularity(str, Enum):
    """Environment diff detail level."""

    counts = "counts"
    entity_keys = "entity_keys"
    memory_hashes = "memory_hashes"
    full = "full"


class RestoreMode(str, Enum):
    """Snapshot restore target mode."""

    replace_env_in_place = "replace_env_in_place"
    restore_to_new_env = "restore_to_new_env"


class MigrationMode(str, Enum):
    """Bulk migration operation mode."""

    copy = "copy"
    move = "move"


class ArchiveVersionDecision(str, Enum):
    """Outcome of the archive version compatibility check."""

    accept = "accept"
    accept_with_migration = "accept_with_migration"
    reject_too_old = "reject_too_old"
    reject_too_new = "reject_too_new"


class BatchFailure(BaseModel):
    """Single failed item in a batch operation."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    error_code: str = ""
    message: str
    memory_id: UUID | None = None
    code: str | None = None


T = TypeVar("T")


class BatchResult(BaseModel, Generic[T]):
    """Generic batch result envelope."""

    model_config = ConfigDict(extra="forbid")

    successes: list[T]
    failures: list[BatchFailure]
    dry_run: bool


class EnvExportRequest(BaseModel):
    """Request schema for ``env_export``.

    Decisions and playbooks are exported as memories with
    ``kind ∈ {decision, playbook}``, never as separate archive files.
    """

    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    format: ExportFormat
    target_path: str
    include_embeddings: bool = True
    include_provenance: bool = True
    include_grants: bool = False
    include_dream_history: bool = False
    chunk_size: int = Field(default=5000, ge=1)

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "EnvExportRequest":
        return validate_required_env_ref_pair(self)


class EnvExportResponse(BaseModel):
    """Response schema for ``env_export``."""

    model_config = ConfigDict(extra="forbid")

    manifest: ExportManifest
    output_path: str
    byte_size: int = Field(ge=0)
    counts_by_kind: dict[str, int]


class EnvImportRequest(BaseModel):
    """Request schema for ``env_import``."""

    model_config = ConfigDict(extra="forbid")

    source_path: str
    target_env_name: str | None = None
    target_env_id: UUID | None = None
    mode: ImportMode = ImportMode.fail
    dry_run: bool = True
    re_embed_if_model_mismatch: bool = True
    allow_bulk_reembed: bool = False

    @model_validator(mode="after")
    def _validate_target(self) -> EnvImportRequest:
        if (self.target_env_name is None) == (self.target_env_id is None):
            raise ValueError("exactly one of target_env_name or target_env_id must be set")
        return self


class EnvImportReport(BaseModel):
    """Report returned by ``env_import``."""

    model_config = ConfigDict(extra="forbid")

    target_env_id: UUID
    dry_run: bool
    mode: ImportMode
    counts: dict[str, int]
    conflicts: dict[str, int]
    sample_conflicts: dict[str, list[str]]
    remap_table_size: int = Field(ge=0)
    pending_vector_rebuild: int = Field(default=0, ge=0)
    re_embed_count: int = Field(default=0, ge=0)
    entity_merges_performed: int = Field(default=0, ge=0)
    archive_version_decision: ArchiveVersionDecision


class EnvMergeRequest(BaseModel):
    """Request schema for ``env_merge``."""

    model_config = ConfigDict(extra="forbid")

    src_env_id: UUID
    dst_env_id: UUID
    entity_strategy: EntityMergeStrategy = EntityMergeStrategy.by_canonical_key
    tag_strategy: TagMergeStrategy = TagMergeStrategy.union
    dry_run: bool = False
    delete_src_after: bool = True
    allow_embedding_mismatch: bool = False
    allow_external_ref_rewrite: bool = False


class EnvMergeReport(BaseModel):
    """Report returned by ``env_merge``."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool
    counts_merged: dict[str, int]
    entity_collapses: int = Field(ge=0)
    tag_unions: int = Field(ge=0)
    cross_env_lineage_rewrites: int = Field(ge=0)
    src_archived: bool


class EnvMergeResponse(BaseModel):
    """Response schema for ``env_merge``."""

    model_config = ConfigDict(extra="forbid")

    dst_env_id: UUID
    src_env_id: UUID
    delete_src_after: bool
    counts: dict[str, int]
    entity_merges_performed: int = Field(ge=0)
    external_refs_rewritten: int = Field(ge=0)
    pending_vector_rebuild: int = Field(default=0, ge=0)
    remap_table_size: int = Field(ge=0)


class EnvCloneRequest(BaseModel):
    """Request schema for ``env_clone``."""

    model_config = ConfigDict(extra="forbid")

    src_env_id: UUID
    new_name: str
    include_embeddings: bool = True
    filter: BrowseFilter | None = None
    lineage_depth: int = Field(default=1, ge=0, le=5)
    include_referenced_entities: bool = True


class EnvCloneResponse(BaseModel):
    """Response schema for ``env_clone``."""

    model_config = ConfigDict(extra="forbid")

    dst_env_id: UUID
    dst_env_name: str
    new_env_id: UUID
    counts: dict[str, int]
    closure_inclusions: dict[str, int] = Field(default_factory=dict)
    pending_vector_rebuild: int = Field(default=0, ge=0)
    remap_table_size: int = Field(ge=0)


class EnvDiffRequest(BaseModel):
    """Request schema for ``env_diff``."""

    model_config = ConfigDict(extra="forbid")

    env_a_id: UUID
    env_b_id: UUID
    granularity: DiffGranularity = DiffGranularity.counts


class EnvDiffReport(BaseModel):
    """Report returned by ``env_diff``."""

    model_config = ConfigDict(extra="forbid")

    granularity: DiffGranularity
    counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    entity_keys: dict[str, list[str]] | None = None
    memory_hashes: dict[str, Any] | None = None
    full: dict[str, Any] | None = None
    truncated: bool = False
    per_table_a_only: dict[str, int] = Field(default_factory=dict)
    per_table_b_only: dict[str, int] = Field(default_factory=dict)
    per_table_both: dict[str, int] = Field(default_factory=dict)
    entity_keys_a_only: list[str] | None = None
    entity_keys_b_only: list[str] | None = None
    memory_hashes_a_only: list[UUID] | None = None
    memory_hashes_b_only: list[UUID] | None = None
    memory_hashes_changed: list[UUID] | None = None


class EnvDiffResponse(EnvDiffReport):
    """Response schema for ``env_diff``."""


class EnvSnapshotRequest(BaseModel):
    """Request schema for ``env_snapshot``."""

    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    label: str
    include_embeddings: bool = True

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "EnvSnapshotRequest":
        return validate_required_env_ref_pair(self)


class EnvSnapshotResponse(BaseModel):
    """Response schema for ``env_snapshot``."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: UUID
    env_id: UUID
    label: str
    created_at: dt.datetime
    path: str
    size_bytes: int = Field(ge=0)
    checksum: str


class SnapshotResponse(EnvSnapshotResponse):
    """Backward-compatible alias for the env snapshot response."""


class EnvRestoreRequest(BaseModel):
    """Request schema for ``env_restore``."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: UUID
    mode: RestoreMode
    confirm_destroy: bool = False
    new_env_name: str | None = None

    @model_validator(mode="after")
    def _validate_restore_target(self) -> EnvRestoreRequest:
        if self.mode == RestoreMode.restore_to_new_env and not self.new_env_name:
            raise ValueError("new_env_name is required for restore_to_new_env")
        return self


class EnvRestoreReport(BaseModel):
    """Report returned by ``env_restore``."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: UUID
    mode: RestoreMode
    target_env_id: UUID
    counts_restored: dict[str, int]


class EnvRestoreResponse(EnvRestoreReport):
    """Response schema for ``env_restore``."""

    import_report: EnvImportReport | None = None
    pending_vector_rebuild: int = Field(default=0, ge=0)
    re_embed_count: int = Field(default=0, ge=0)


class EnvDeleteRequest(BaseModel):
    """Request schema for ``env_delete``."""

    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    confirm_destroy: bool
    cascade_external_refs: bool = False

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "EnvDeleteRequest":
        return validate_required_env_ref_pair(self)


class EnvDeleteReport(BaseModel):
    """Report returned by ``env_delete``."""

    model_config = ConfigDict(extra="forbid")

    env_id: UUID
    counts_deleted: dict[str, int]
    external_refs_tombstoned: int = Field(ge=0)


class EnvDeleteResponse(BaseModel):
    """Response schema for ``env_delete``."""

    model_config = ConfigDict(extra="forbid")

    env_id: UUID
    confirm_destroy: bool
    cascade_external_refs: bool
    counts: dict[str, int]
    external_lineage_exit_dropped: int = Field(ge=0)
    external_lineage_entry_dropped: int = Field(ge=0)


class EnvRenameRequest(BaseModel):
    """Request schema for ``env_rename``."""

    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None
    new_name: str | None = None
    new_default_embedding_model_id: str | None = None
    new_retention_policy: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_env_refs(self) -> "EnvRenameRequest":
        return validate_required_env_ref_pair(self)


class EnvRenameResponse(BaseModel):
    """Response schema for ``env_rename``."""

    model_config = ConfigDict(extra="forbid")

    env_id: UUID
    name: str
    default_embedding_model_id: str
    retention_policy: dict[str, Any]
    changed_fields: list[str]
    warning: str | None = None


class MemCopyRequest(BaseModel):
    """Request schema for ``mem_copy``."""

    model_config = ConfigDict(extra="forbid")

    memory_id: UUID
    dst_env_id: UUID
    copy_tags: bool = True
    copy_provenance: bool = True
    create_lineage_edge: bool = True
    preserve_timestamps: bool = False
    re_embed_if_model_mismatch: bool = False
    copy_lineage: bool = False
    copy_entities: Literal["never", "if_present_in_dst", "always_create"] = "if_present_in_dst"


class MemCopyResponse(BaseModel):
    """Response schema for ``mem_copy``."""

    model_config = ConfigDict(extra="forbid")

    dst_memory_id: UUID
    dst_env_id: UUID
    lineage_edge_id: str | None = None
    pending_vector_rebuild: int = Field(default=0, ge=0)


class MemMoveRequest(BaseModel):
    """Request schema for ``mem_move``.

    Implemented as ``mem_copy`` + supersede source. The source memory
    transitions to ``status='superseded'`` with ``superseded_by`` pointing
    at the new dst-env memory.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: UUID
    dst_env_id: UUID
    redirect_source: bool = True
    copy_tags: bool = True
    copy_provenance: bool = True
    create_lineage_edge: bool = True
    preserve_timestamps: bool = False
    re_embed_if_model_mismatch: bool = False
    copy_lineage: bool = False


class MemMoveResponse(MemCopyResponse):
    """Response schema for ``mem_move``."""

    source_memory_status: Literal["superseded", "deleted"]


class EnvMigrateRequest(BaseModel):
    """Request schema for ``env_migrate``."""

    model_config = ConfigDict(extra="forbid")

    src_env_id: UUID
    dst_env_id: UUID
    filter: BrowseFilter | None = None
    mode: MigrationMode = MigrationMode.copy
    copy_tags: bool = True
    copy_provenance: bool = True
    create_lineage_edges: bool = True
    preserve_timestamps: bool = False
    re_embed_if_model_mismatch: bool = False
    preserve_supersession_chain: bool = True
    include_superseded: bool = False
    fail_fast: bool = False
    dry_run: bool = False


class EnvMigrateReport(BaseModel):
    """Report returned by ``env_migrate``."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool
    mode: MigrationMode
    processed: int = Field(ge=0)
    migrated: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failed: int = Field(ge=0)
    sample_failures: list[BatchFailure]


class EnvMigrateResponse(BaseModel):
    """Response schema for ``env_migrate``."""

    model_config = ConfigDict(extra="forbid")

    src_env_id: UUID
    dst_env_id: UUID
    mode: MigrationMode
    attempted: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    remap: dict[UUID, UUID] = Field(default_factory=dict)
    failures: list[BatchFailure] = Field(default_factory=list)
    truncated: bool = False
    pending_vector_rebuild: int = Field(default=0, ge=0)
    closure_inclusions: int = Field(default=0, ge=0)


__all__ = [
    "ArchiveVersionDecision",
    "BatchFailure",
    "BatchResult",
    "BrowseFilter",
    "DiffGranularity",
    "EntityMergeStrategy",
    "EnvCloneRequest",
    "EnvCloneResponse",
    "EnvDeleteReport",
    "EnvDeleteRequest",
    "EnvDeleteResponse",
    "EnvDiffReport",
    "EnvDiffRequest",
    "EnvDiffResponse",
    "EnvExportRequest",
    "EnvExportResponse",
    "EnvImportReport",
    "EnvImportRequest",
    "EnvMergeReport",
    "EnvMergeRequest",
    "EnvMergeResponse",
    "EnvMigrateReport",
    "EnvMigrateRequest",
    "EnvMigrateResponse",
    "EnvRenameRequest",
    "EnvRenameResponse",
    "EnvResponse",
    "EnvRestoreReport",
    "EnvRestoreRequest",
    "EnvRestoreResponse",
    "EnvSnapshotRequest",
    "EnvSnapshotResponse",
    "ExportFormat",
    "ExportManifest",
    "ImportMode",
    "IncludeFlags",
    "MemCopyRequest",
    "MemCopyResponse",
    "MemMoveRequest",
    "MemMoveResponse",
    "MemoryVectorRecord",
    "MemoryResponse",
    "MigrationMode",
    "RemapTable",
    "RestoreMode",
    "SnapshotResponse",
    "SourceMetadata",
    "TagMergeStrategy",
]
