"""SQLAlchemy 2.0 ORM models for the v1 schema.

These mirror ``migrations/versions/0001_v1_initial.py`` exactly. Server-side
constraints (CHECKs, the monotonic-version trigger, composite cross-env FKs)
are enforced by Postgres; the ORM layer only needs to allow correct CRUD.

Naming convention applied via ``SQLAlchemy_naming_convention`` so future
``alembic revision --autogenerate`` runs produce predictable constraint
names.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# ---------------------------------------------------------------------------
# environments / agents / sessions / tokens / env_grants
# ---------------------------------------------------------------------------

class Environment(Base):
    __tablename__ = "environments"
    __table_args__ = (
        Index(
            "environments_active_idx",
            "name",
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    retention_policy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    default_embedding_model_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="active",
        server_default=text("'active'"),
    )
    deleted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
    )
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Token(Base):
    __tablename__ = "tokens"

    token_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
    )
    hashed_secret: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default=text("ARRAY[]::text[]"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EnvGrant(Base):
    __tablename__ = "env_grants"

    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="CASCADE"),
        primary_key=True,
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    granted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Snapshot(Base):
    __tablename__ = "snapshots"
    __table_args__ = (
        UniqueConstraint("env_id", "label", name="snapshots_env_label_uniq"),
        Index("snapshots_env_created_idx", "env_id", text("created_at DESC")),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_by_agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class MemoryTombstone(Base):
    """Audit marker left behind by ``memory_hard_delete``.

    The canonical Memory row is fully gone (body, embeddings, tags),
    but a tombstone row survives so:

    * ``mem_get`` on the deleted id can return a recognisable
      ``NOT_FOUND`` with hint ``see tombstone <id>``.
    * Operators can audit hard-delete activity weeks later.
    * Leak-recovery procedures (memory-mcp.instructions.md §14) have a
      stable record to reference when filing the API gap.

    The tombstone never carries the deleted content.
    """

    __tablename__ = "memory_tombstones"
    __table_args__ = (
        Index("memory_tombstones_env_deleted_at_idx", "env_id", text("deleted_at DESC")),
        Index("memory_tombstones_deleted_by_idx", "deleted_by_agent_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    deleted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_by_agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    cascade_root: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    original_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_status: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# memories
# ---------------------------------------------------------------------------

class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        UniqueConstraint("id", "env_id", name="memories_id_env_uniq"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active", server_default=text("'active'"))
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    trigger_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    steps: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    macro: Mapped[str | None] = mapped_column(Text, nullable=True)
    body_tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(title,'')), 'A') || "
            "setweight(to_tsvector('english', coalesce(body,'')), 'B')",
            persisted=True,
        ),
        nullable=True,
    )
    salience: Mapped[float] = mapped_column(Numeric, nullable=False, default=0.5, server_default=text("0.5"))
    confidence: Mapped[float] = mapped_column(Numeric, nullable=False, default=0.5, server_default=text("0.5"))
    access_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default=text("0"))
    last_accessed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    negative_feedback_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    reference_count_rel_link: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0"),
    )
    reference_count_lineage: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0"),
    )
    reference_count_task: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0"),
    )
    reference_count_playbook: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0"),
    )
    reference_count: Mapped[int] = mapped_column(
        Integer,
        Computed(
            "reference_count_rel_link "
            "+ reference_count_lineage "
            "+ reference_count_task "
            "+ reference_count_playbook",
            persisted=True,
        ),
        nullable=False,
    )
    # Phase 1e — authority-weighted citations (recount-pass-only writes).
    # NUMERIC(18,6) accommodates per-citation-occurrence worst case; recount
    # walks canonical edge state each cycle when DREAM_POPULARITY_AUTHORITY_WEIGHTED
    # is enabled.
    ref_authority_rel_link: Mapped[float] = mapped_column(
        Numeric(18, 6), nullable=False, default=0, server_default=text("0"),
    )
    ref_authority_lineage: Mapped[float] = mapped_column(
        Numeric(18, 6), nullable=False, default=0, server_default=text("0"),
    )
    ref_authority_task: Mapped[float] = mapped_column(
        Numeric(18, 6), nullable=False, default=0, server_default=text("0"),
    )
    ref_authority_playbook: Mapped[float] = mapped_column(
        Numeric(18, 6), nullable=False, default=0, server_default=text("0"),
    )
    reference_authority: Mapped[float] = mapped_column(
        Numeric(19, 6),
        Computed(
            "ref_authority_rel_link "
            "+ ref_authority_lineage "
            "+ ref_authority_task "
            "+ ref_authority_playbook",
            persisted=True,
        ),
        nullable=False,
    )
    # Stamped by the recount pass each cycle it touches a memory; NULL means
    # "never recomputed" (knob has never been on for this env).
    authority_last_recount_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    # Phase 1e-d (Migration 0019) — formula version this row's stored
    # ``salience`` was computed under. ``0`` = pre-1e-d / unstamped;
    # recount pass compares against ``Settings.dream_salience_formula_version``
    # and re-stamps + re-computes any row that's behind. **Any change to
    # ``compute_salience`` math MUST bump the settings value** so existing
    # rows re-stamp on the next recount cycle (see ``salience.py`` docstring).
    salience_formula_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0"),
    )
    verified_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="RESTRICT"),
        nullable=True,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"),
    )
    decision_meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1, server_default=text("1"))


# ---------------------------------------------------------------------------
# entities / entity_aliases
# ---------------------------------------------------------------------------

class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("env_id", "normalized_name", name="entities_env_id_normalized_name_key"),
        UniqueConstraint("id", "env_id", name="entities_id_env_uniq"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1, server_default=text("1"))


class EntityAlias(Base):
    __tablename__ = "entity_aliases"
    __table_args__ = (
        PrimaryKeyConstraint("entity_id", "normalized_alias"),
        UniqueConstraint("env_id", "normalized_alias", name="entity_aliases_env_id_normalized_alias_key"),
        ForeignKeyConstraint(
            ("entity_id", "env_id"),
            ("entities.id", "entities.env_id"),
            ondelete="CASCADE",
            name="entity_aliases_entity_env_fk",
        ),
    )

    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    env_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_alias: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        UniqueConstraint("id", "env_id", name="tasks_id_env_uniq"),
        Index("tasks_env_status_priority_created_idx", "env_id", "status", "priority", "created_at"),
        Index(
            "tasks_env_playbook_idx",
            "env_id",
            "playbook_id",
            postgresql_where=text("playbook_id IS NOT NULL"),
        ),
        Index("tasks_updated_desc_idx", text("updated_at DESC")),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending", server_default=text("'pending'"))
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=50, server_default=text("50"))
    playbook_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="SET NULL"),
        nullable=True,
    )
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1, server_default=text("1"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_by_agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )


# ---------------------------------------------------------------------------
# graph_nodes / relations
# ---------------------------------------------------------------------------

class GraphNode(Base):
    __tablename__ = "graph_nodes"
    __table_args__ = (
        UniqueConstraint("id", "env_id", name="graph_nodes_id_env_uniq"),
        CheckConstraint(
            "(node_type='memory' AND memory_id IS NOT NULL AND entity_id IS NULL AND task_id IS NULL) "
            "OR (node_type='entity' AND entity_id IS NOT NULL AND memory_id IS NULL AND task_id IS NULL) "
            "OR (node_type='task' AND task_id IS NOT NULL AND entity_id IS NULL AND memory_id IS NULL)",
            name="graph_nodes_exactly_one_target_chk",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    node_type: Mapped[str] = mapped_column(Text, nullable=False)
    memory_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=True,
    )
    entity_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=True,
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=True,
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Relation(Base):
    __tablename__ = "relations"
    __table_args__ = (
        UniqueConstraint("src_node_id", "dst_node_id", "type", name="relations_src_dst_type_uniq"),
        ForeignKeyConstraint(
            ("src_node_id", "env_id"),
            ("graph_nodes.id", "graph_nodes.env_id"),
            ondelete="CASCADE",
            name="relations_src_env_fk",
        ),
        ForeignKeyConstraint(
            ("dst_node_id", "env_id"),
            ("graph_nodes.id", "graph_nodes.env_id"),
            ondelete="CASCADE",
            name="relations_dst_env_fk",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    src_node_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    dst_node_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    properties: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1, server_default=text("1"))


# ---------------------------------------------------------------------------
# tags / memory_tags
# ---------------------------------------------------------------------------

class Tag(Base):
    __tablename__ = "tags"
    __table_args__ = (
        UniqueConstraint("env_id", "name", name="tags_env_id_name_key"),
        UniqueConstraint("id", "env_id", name="tags_id_env_uniq"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)


class MemoryTag(Base):
    __tablename__ = "memory_tags"
    __table_args__ = (
        PrimaryKeyConstraint("memory_id", "tag_id"),
        ForeignKeyConstraint(
            ("memory_id", "env_id"),
            ("memories.id", "memories.env_id"),
            ondelete="CASCADE",
            name="memory_tags_memory_env_fk",
        ),
        ForeignKeyConstraint(
            ("tag_id", "env_id"),
            ("tags.id", "tags.env_id"),
            ondelete="CASCADE",
            name="memory_tags_tag_env_fk",
        ),
    )

    memory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    tag_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    env_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)


# ---------------------------------------------------------------------------
# audit_log / memory_sources / memory_lineage
# ---------------------------------------------------------------------------

class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    record_type: Mapped[str] = mapped_column(Text, nullable=False)
    record_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    env_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="SET NULL"),
        nullable=True,
    )
    op: Mapped[str] = mapped_column(Text, nullable=False)
    at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    by_agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    subject_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    redacted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    redaction_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class MemorySource(Base):
    __tablename__ = "memory_sources"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    evidence_span: Mapped[str | None] = mapped_column(Text, nullable=True)


class MemoryLineage(Base):
    __tablename__ = "memory_lineage"
    __table_args__ = (
        PrimaryKeyConstraint("parent_memory_id", "child_memory_id", "relation"),
    )

    parent_memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
    )
    child_memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
    )
    relation: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


# ---------------------------------------------------------------------------
# outbox / outbox_delivery / projection_state
# ---------------------------------------------------------------------------

class Outbox(Base):
    __tablename__ = "outbox"
    __table_args__ = (
        UniqueConstraint(
            "aggregate_type", "aggregate_id", "aggregate_version",
            name="outbox_aggregate_type_aggregate_id_aggregate_version_key",
        ),
        Index("ix_outbox_aggregate", "aggregate_id", "aggregate_version"),
        Index("ix_outbox_available", "available_at", "event_id"),
        Index("ix_outbox_env_event", "env_id", "event_id"),
    )

    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    aggregate_type: Mapped[str] = mapped_column(Text, nullable=False)
    aggregate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    aggregate_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="CASCADE"),
        nullable=False,
    )
    op: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    available_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class OutboxDelivery(Base):
    __tablename__ = "outbox_delivery"
    __table_args__ = (
        PrimaryKeyConstraint("event_id", "sink"),
    )

    event_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("outbox.event_id", ondelete="CASCADE"),
        nullable=False,
    )
    sink: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending", server_default=text("'pending'"))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    locked_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    locked_until: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    done_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ProjectionState(Base):
    __tablename__ = "projection_state"
    __table_args__ = (
        PrimaryKeyConstraint("sink", "env_id"),
    )

    sink: Mapped[str] = mapped_column(Text, nullable=False)
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="CASCADE"),
        nullable=False,
    )
    last_event_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    last_success_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lag_seconds: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    status: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# dream_runs / dream_proposals (Phase 2.2)
# ---------------------------------------------------------------------------

class DreamRun(Base):
    __tablename__ = "dream_runs"
    __table_args__ = (
        Index("dream_runs_env_started_idx", "env_id", "started_at"),
        Index("dream_runs_mode_started_idx", "mode", "started_at"),
        Index(
            "dream_runs_running_idx", "env_id", "mode",
            postgresql_where=text("status = 'running'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"),
    )
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="CASCADE"),
        nullable=False,
    )
    mode: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="running", server_default=text("'running'"),
    )
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    triggered_by: Mapped[str] = mapped_column(
        Text, nullable=False, default="scheduler", server_default=text("'scheduler'"),
    )
    summarizer_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"),
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DreamProposal(Base):
    __tablename__ = "dream_proposals"
    __table_args__ = (
        Index("dream_proposals_env_status_idx", "env_id", "status"),
        Index("dream_proposals_env_kind_status_idx", "env_id", "kind", "status"),
        Index(
            "dream_proposals_run_idx", "dream_run_id",
            postgresql_where=text("dream_run_id IS NOT NULL"),
        ),
        Index("dream_proposals_created_idx", "env_id", "created_at"),
        Index(
            "dream_proposals_open_dedupe_key_uniq",
            "env_id", "kind", "dedupe_key",
            unique=True,
            postgresql_where=text("status = 'open' AND dedupe_key IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"),
    )
    env_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="open", server_default=text("'open'"),
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"),
    )
    summarizer_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_failed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false"),
    )
    dedupe_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    dream_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dream_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    reviewed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by_agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )
    review_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
