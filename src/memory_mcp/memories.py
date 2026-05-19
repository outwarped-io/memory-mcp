"""Canonical memory CRUD for v1 (local-only build).

Surface
-------

Tool-facing async functions that the MCP transport will call:

* :func:`memory_write` — create a new memory in ``active`` status.
* :func:`memory_get` / :func:`memory_get_many` — read by id, with side-effect access bump.
* :func:`memory_update` — patch fields with optimistic-lock check.
* :func:`memory_archive` / :func:`memory_retire` — convenience lifecycle wrappers.
* :func:`memory_supersede` — atomic "create new + mark old as superseded".

Each tool calls :func:`rbac.require` *after* loading the memory (so v1.5 RBAC
sees the row's actual ``env_id``) and emits one outbox event per state
change via :func:`memory_mcp.db.outbox.enqueue_event`.

Optimistic concurrency
----------------------

Every write that mutates a row uses the pattern::

    UPDATE memories
    SET <patch>, version = version + 1, updated_at = now()
    WHERE id = :id AND version = :expected_version
    RETURNING *

If ``RETURNING`` is empty, we re-read by id to disambiguate
:class:`NotFoundError` vs :class:`VersionConflictError`. The
``require_version_monotonic`` trigger (migration 0001) guarantees no
concurrent writer can sneak a non-monotonic update past us.

Audit log content policy (GDPR-aware)
-------------------------------------

``audit_log.before`` / ``audit_log.after`` deliberately store **hashed body
content + structural metadata only** — never the raw ``body`` or
``metadata_`` fields. This means a future ``memory_delete_hard`` does not
need to walk the audit table to redact memory content; the canonical row
is the only place full text lives.

Outbox payload policy
---------------------

The outbox payload is a self-contained snapshot of everything the
projection worker needs to materialize the row in the sink (Qdrant,
Neo4j, …). Workers never re-read canonical Postgres for projection,
which avoids races between the worker and concurrent updates.

Embedding generation happens in the worker (writer stays light).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import (
    AuditLog,
    Entity,
    Environment,
    GraphNode,
    Memory,
    MemoryLineage,
    MemorySource,
    MemoryTag,
    MemoryTombstone,
    Relation,
    Tag,
)
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.decisions.api import validate_decision_meta
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import (
    LineageRelation,
    MemoryKind,
    MemorySourceType,
    MemoryStatus,
    OutboxAggregateType,
    OutboxOp,
    is_valid_transition,
)
from memory_mcp.dream.salience import (
    SalienceInputs,
    compute_salience,
    salience_weights_from_settings,
)
from memory_mcp.errors import (
    BlastRadiusExceededError,
    EmbeddingModelMismatchError,
    EnvAmbiguousError,
    InvalidInputError,
    InvalidTransitionError,
    MemoryMCPError,
    NotFoundError,
    VersionConflictError,
)
from memory_mcp.identity import AgentContext

from memory_mcp_schemas.env_ops import (
    MemCopyRequest,
    MemCopyResponse,
    MemMoveRequest,
    MemMoveResponse,
)
from memory_mcp_schemas.memories import (
    HardDeleteProjectionStatus,
    MemoryHardDeleteAffected,
    MemoryHardDeleteRequest,
    MemoryHardDeleteResponse,
    MemoryResponse,
    MemorySupersedeRequest,
    MemoryWriteRequest,
)

log = logging.getLogger(__name__)

__all__ = [
    "HardDeleteProjectionStatus",
    "MemoryHardDeleteAffected",
    "MemoryHardDeleteRequest",
    "MemoryHardDeleteResponse",
    "MemoryResponse",
    "MemorySupersedeRequest",
    "MemoryUpdatePatch",
    "MemoryWriteRequest",
    "mem_copy",
    "mem_move",
    "memory_archive",
    "memory_get",
    "memory_get_many",
    "memory_hard_delete",
    "memory_retire",
    "memory_supersede",
    "memory_update",
    "memory_write",
]


# Statuses for which the canonical memory should appear in projections.
_VISIBLE_IN_PROJECTION: frozenset[MemoryStatus] = frozenset({
    MemoryStatus.proposed,
    MemoryStatus.active,
    MemoryStatus.stale,
})


# ---------------------------------------------------------------------------
# Pydantic schemas (tool-facing I/O)
# ---------------------------------------------------------------------------


class MemoryUpdatePatch(BaseModel):
    """Patch payload for ``memory_update``.

    Field absence (``model_dump(exclude_unset=True)``) means *no change*;
    explicit ``None`` means *clear the field* (only meaningful for
    nullable columns: ``title``, ``expires_at``, ``verified_at``).
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
    tags: list[str] | None = None  # None=no change; []=clear; [...]=replace
    metadata: dict[str, Any] | None = None
    salience: float | None = Field(default=None, ge=0.0, le=1.0)
    # Phase 1e-d (v0.14.1) — internal/dream-pass-only. Stamped by the
    # recount pass together with a ``salience`` recompute. Not exposed
    # through user-facing ``memory_update`` callers (mcp tool wrappers do
    # not include this field). When ``None`` the column is left
    # unchanged (so direct-UPDATE paths that only touch ``salience``
    # never silently rewrite the version).
    salience_formula_version: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    pinned: bool | None = None
    expires_at: dt.datetime | None = None
    verified_at: dt.datetime | None = None
    decision_meta: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Internal helpers — env resolution, tags, audit, outbox payload
# ---------------------------------------------------------------------------


def _normalize_tags(tags: list[str]) -> list[str]:
    """Strip whitespace, drop empties, dedupe (preserve first-seen order).

    Tags are case-PRESERVING. Dedup is exact-match after strip. Tag DB
    rows are unique per ``(env_id, name)`` so different cases create
    different tags — that is intentional in v1.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in tags:
        t = raw.strip()
        if not t or t in seen:
            continue
        if len(t) > 200:
            raise ValueError(f"tag too long (max 200): {t[:50]!r}…")
        seen.add(t)
        out.append(t)
    return out


def _normalize_playbook_fields(
    *,
    kind: MemoryKind,
    steps: list[str] | None,
    macro: str | None,
) -> tuple[list[str] | None, str | None]:
    """Validate kind/steps/macro invariants and return normalized values."""
    if kind is MemoryKind.playbook:
        if steps is None:
            raise InvalidInputError("playbook steps are required")
        cleaned_steps: list[str] = []
        for step in steps:
            if not isinstance(step, str):
                raise InvalidInputError("playbook steps must be strings")
            cleaned = step.strip()
            if not cleaned:
                raise InvalidInputError("playbook steps must be non-empty strings")
            cleaned_steps.append(cleaned)
        if not cleaned_steps:
            raise InvalidInputError("playbook steps are required")
        normalized_macro = macro.strip().lower() if isinstance(macro, str) else ""
        if not normalized_macro:
            raise InvalidInputError("playbook macro is required")
        return cleaned_steps, normalized_macro

    if steps is not None or macro is not None:
        raise InvalidInputError("steps and macro are only valid for playbook memories")
    return None, None


def _is_macro_integrity_error(exc: IntegrityError) -> bool:
    orig = getattr(exc, "orig", None)
    constraint = getattr(orig, "constraint_name", None)
    if constraint == "ix_memories_macro_per_env":
        return True
    diag = getattr(orig, "diag", None)
    diag_constraint = getattr(diag, "constraint_name", None) if diag is not None else None
    if diag_constraint == "ix_memories_macro_per_env":
        return True
    return "ix_memories_macro_per_env" in str(exc)


async def _ensure_macro_available(
    session: AsyncSession,
    *,
    env_id: UUID,
    macro: str | None,
    exclude_memory_id: UUID | None = None,
) -> None:
    if macro is None:
        return
    stmt = select(Memory.id).where(
        Memory.env_id == env_id,
        Memory.macro.is_not(None),
        func.lower(Memory.macro) == macro,
    )
    if exclude_memory_id is not None:
        stmt = stmt.where(Memory.id != exclude_memory_id)
    existing = (await session.execute(stmt.limit(1))).scalar_one_or_none()
    if existing is not None:
        raise InvalidInputError("macro already in use in this env")


def _resolve_env_id(
    *,
    explicit: UUID | None,
    ctx: AgentContext,
) -> UUID:
    """Pick the env to write to.

    Order: explicit value > sole attached env. If multiple are attached
    and the caller did not specify, raise :class:`EnvAmbiguousError`
    (matches plan's documented behavior).
    """
    if explicit is not None:
        return explicit

    attached = list(dict.fromkeys(ctx.attached_env_ids))  # dedupe, preserve order
    if len(attached) == 1:
        return attached[0]
    if not attached:
        raise EnvAmbiguousError(
            "no env_id given and no env attached to the session",
            attached=[],
        )
    raise EnvAmbiguousError(
        "no env_id given and multiple envs attached; specify env_id explicitly",
        attached=[str(e) for e in attached],
    )


def _hash_body(body: str | None) -> str | None:
    if body is None:
        return None
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _audit_snapshot(memory: Memory, *, tag_names: list[str] | None = None) -> dict[str, Any]:
    """GDPR-aware audit snapshot of a memory row.

    Stores hashes for ``body`` and structural metadata for ``metadata_``;
    never embeds full text. The canonical ``memories`` row is the only
    place full content lives — ``memory_delete_hard`` (v2) needs to wipe
    only that one row to remove all sensitive content.
    """
    try:
        steps = getattr(memory, "steps")
    except AttributeError:
        steps = None
    try:
        macro = getattr(memory, "macro")
    except AttributeError:
        macro = None
    try:
        decision_meta = getattr(memory, "decision_meta")
    except AttributeError:
        decision_meta = None

    snap: dict[str, Any] = {
        "kind": memory.kind,
        "status": memory.status,
        "env_id": str(memory.env_id),
        "title": memory.title,  # title kept; conventionally short, low-PII
        "body_hash": _hash_body(memory.body),
        "body_length": len(memory.body) if memory.body else 0,
        "trigger_description_hash": _hash_body(getattr(memory, "trigger_description", None)),
        "trigger_description_length": (
            len(memory.trigger_description) if getattr(memory, "trigger_description", None) else 0
        ),
        "steps_count": len(steps) if steps else 0,
        "macro": macro,
        "salience": float(memory.salience),
        "confidence": float(memory.confidence),
        "pinned": memory.pinned,
        "version": memory.version,
        "superseded_by": str(memory.superseded_by) if memory.superseded_by else None,
        "decision_meta": (
                dict(decision_meta or {})
                if decision_meta
                else None
            ),
        "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        "verified_at": memory.verified_at.isoformat() if memory.verified_at else None,
    }
    md = memory.metadata_ or {}
    if md:
        snap["metadata_keys"] = sorted(md.keys())
    if tag_names is not None:
        snap["tags"] = list(tag_names)
    return snap


async def _record_audit(
    session: AsyncSession,
    *,
    op: str,
    memory: Memory,
    by_agent_id: UUID,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    extra_after: dict[str, Any] | None = None,
) -> None:
    """Append an ``audit_log`` row.

    ``extra_after`` lets specific operations (retire reason, supersede target)
    add structured side-info without polluting the standard snapshot.
    """
    final_after = after
    if extra_after:
        final_after = (after or {}) | extra_after
    await session.execute(
        insert(AuditLog).values(
            record_type="memory",
            record_id=memory.id,
            env_id=memory.env_id,
            op=op,
            by_agent_id=by_agent_id,
            before=before,
            after=final_after,
        )
    )


def _projection_payload(
    memory: Memory,
    *,
    tag_names: list[str],
    embedding_model_id: str,
) -> dict[str, Any]:
    """Self-contained projection payload — workers never re-read canonical."""
    return {
        "memory_id": str(memory.id),
        "env_id": str(memory.env_id),
        "kind": memory.kind,
        "status": memory.status,
        "title": memory.title,
        "body": memory.body,
        "trigger_description": memory.trigger_description,
        "steps": list(memory.steps) if memory.steps is not None else None,
        "macro": getattr(memory, "macro", None),
        "salience": float(memory.salience),
        "confidence": float(memory.confidence),
        "pinned": memory.pinned,
        "tags": list(tag_names),
        "metadata": dict(memory.metadata_ or {}),
        "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        "verified_at": memory.verified_at.isoformat() if memory.verified_at else None,
        "superseded_by": str(memory.superseded_by) if memory.superseded_by else None,
        "decision_meta": (
            dict(getattr(memory, "decision_meta", None) or {})
            if getattr(memory, "decision_meta", None)
            else None
        ),
        "version": memory.version,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
        "embedding_model_id": embedding_model_id,
    }


def _outbox_op_for(status: MemoryStatus, *, is_create: bool) -> OutboxOp:
    """Map lifecycle status to outbox op for the projection worker.

    Visible statuses (``proposed``/``active``/``stale``) → ``upsert`` on
    create, ``update`` on edit. Hidden statuses
    (``archived``/``superseded``/``retired``) → ``tombstone`` so the worker
    can deindex.
    """
    if status not in _VISIBLE_IN_PROJECTION:
        return OutboxOp.tombstone
    return OutboxOp.upsert if is_create else OutboxOp.update


async def _upsert_tags(
    session: AsyncSession,
    *,
    env_id: UUID,
    names: list[str],
) -> dict[str, UUID]:
    """Get-or-create tag rows for ``names`` in ``env_id``.

    Returns ``{name: tag_id}``. Two-phase: bulk INSERT … ON CONFLICT DO
    NOTHING then SELECT all by name — works whether the rows existed or
    were just created in this transaction.
    """
    if not names:
        return {}

    await session.execute(
        pg_insert(Tag)
        .values([{"env_id": env_id, "name": n} for n in names])
        .on_conflict_do_nothing(index_elements=["env_id", "name"])
    )
    rows = await session.execute(
        select(Tag.id, Tag.name).where(Tag.env_id == env_id, Tag.name.in_(names))
    )
    return {name: tid for tid, name in rows.all()}


async def _replace_memory_tags(
    session: AsyncSession,
    *,
    memory_id: UUID,
    env_id: UUID,
    tag_ids: list[UUID],
) -> None:
    """Replace memory_tags links with the given set."""
    await session.execute(delete(MemoryTag).where(MemoryTag.memory_id == memory_id))
    if tag_ids:
        await session.execute(
            insert(MemoryTag).values(
                [
                    {"memory_id": memory_id, "tag_id": tid, "env_id": env_id}
                    for tid in tag_ids
                ]
            )
        )


async def _load_memory_for_read(
    session: AsyncSession,
    memory_id: UUID,
) -> Memory | None:
    return await session.get(Memory, memory_id)


async def _load_tag_names(
    session: AsyncSession,
    memory_id: UUID,
) -> list[str]:
    rows = await session.execute(
        select(Tag.name)
        .join(MemoryTag, MemoryTag.tag_id == Tag.id)
        .where(MemoryTag.memory_id == memory_id)
        .order_by(Tag.name)
    )
    return [r[0] for r in rows.all()]


async def _load_env_embedding_model(
    session: AsyncSession,
    env_id: UUID,
) -> str:
    row = await session.execute(
        select(Environment.default_embedding_model_id).where(Environment.id == env_id)
    )
    val = row.scalar_one_or_none()
    if val is None:
        raise NotFoundError(f"environment {env_id} not found", env_id=str(env_id))
    return val


async def _ensure_memory_graph_node(
    session: AsyncSession,
    *,
    env_id: UUID,
    memory_id: UUID,
) -> GraphNode:
    node = (await session.execute(
        select(GraphNode).where(GraphNode.memory_id == memory_id)
    )).scalar_one_or_none()
    if node is None:
        node = GraphNode(env_id=env_id, node_type="memory", memory_id=memory_id)
        session.add(node)
        await session.flush()
        await session.refresh(node)
    return node


async def _ensure_entity_graph_node(
    session: AsyncSession,
    *,
    env_id: UUID,
    entity_id: UUID,
) -> GraphNode:
    entity = (await session.execute(select(Entity).where(Entity.id == entity_id))).scalar_one_or_none()
    if entity is None:
        raise NotFoundError(f"entity {entity_id} not found", entity_id=str(entity_id))
    if entity.env_id != env_id:
        raise ValueError(f"entity {entity_id} is in env {entity.env_id}, not memory env {env_id}")

    node = (await session.execute(
        select(GraphNode).where(GraphNode.entity_id == entity_id)
    )).scalar_one_or_none()
    if node is None:
        node = GraphNode(env_id=env_id, node_type="entity", entity_id=entity_id)
        session.add(node)
        await session.flush()
        await session.refresh(node)
    return node


async def _validate_decision_meta_for_kind(
    *,
    kind: str,
    decision_meta: dict[str, Any] | None,
    env_id: UUID,
    session: AsyncSession,
) -> dict[str, Any] | None:
    """Validate and JSON-normalize decision_meta for a resulting memory row."""
    if decision_meta is None:
        return None
    if kind != MemoryKind.decision.value:
        raise InvalidInputError("decision_meta only valid for kind=decision")
    meta = await validate_decision_meta(decision_meta, env_id, session)
    if meta is None:
        return None
    return meta.model_dump(mode="json")


def _ensure_env_visible(memory: Memory, ctx: AgentContext) -> None:
    """v1: caller can read any env they have attached.

    In v1.5 this becomes the RBAC check — for v1, ``rbac.require`` is a
    no-op so we still narrow visibility by the attached envs as a UX
    convention. Tools that have *no* attached envs see all envs (matches
    the plan's "v1 = local-only, no grants" rule).
    """
    attached = ctx.attached_env_ids
    if not attached:
        return
    if memory.env_id not in attached:
        raise NotFoundError(
            f"memory {memory.id} not visible in attached envs",
            memory_id=str(memory.id),
        )


def _to_response(
    memory: Memory,
    tag_names: list[str],
    *,
    reference_velocity: int | None = None,
) -> MemoryResponse:
    rc_rl = int(getattr(memory, "reference_count_rel_link", 0) or 0)
    rc_ln = int(getattr(memory, "reference_count_lineage", 0) or 0)
    rc_tk = int(getattr(memory, "reference_count_task", 0) or 0)
    rc_pb = int(getattr(memory, "reference_count_playbook", 0) or 0)
    rc_total = int(getattr(memory, "reference_count", rc_rl + rc_ln + rc_tk + rc_pb) or 0)
    return MemoryResponse(
        id=memory.id,
        env_id=memory.env_id,
        kind=MemoryKind(memory.kind),
        status=MemoryStatus(memory.status),
        title=memory.title,
        body=memory.body,
        trigger_description=memory.trigger_description,
        steps=list(memory.steps) if memory.steps is not None else None,
        macro=memory.macro,
        tags=list(tag_names),
        metadata=dict(memory.metadata_ or {}),
        salience=float(memory.salience),
        confidence=float(memory.confidence),
        pinned=memory.pinned,
        access_count=memory.access_count,
        last_accessed_at=memory.last_accessed_at,
        negative_feedback_count=memory.negative_feedback_count,
        verified_at=memory.verified_at,
        expires_at=memory.expires_at,
        superseded_by=memory.superseded_by,
        decision_meta=dict(memory.decision_meta or {}) if memory.decision_meta else None,
        version=memory.version,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        reference_count=rc_total,
        reference_breakdown={
            "rel_link": rc_rl,
            "lineage": rc_ln,
            "task": rc_tk,
            "playbook": rc_pb,
        },
        reference_authority=float(getattr(memory, "reference_authority", 0) or 0),
        reference_velocity=reference_velocity,
    )


class ConflictError(MemoryMCPError):
    """Operation conflicts with existing references."""

    code = "ME_REFERENCED_CANNOT_HARD_DELETE"


async def _load_env_for_cross_env_copy(session: AsyncSession, env_id: UUID) -> Environment:
    env = await session.get(Environment, env_id)
    if env is None:
        raise NotFoundError(f"environment {env_id} not found", env_id=str(env_id))
    if getattr(env, "status", "active") == "deleted":
        exc = NotFoundError(f"environment {env_id} is deleted", env_id=str(env_id))
        exc.code = "ENV_DELETED"
        raise exc
    return env


async def _copy_memory_in_session(
    session: AsyncSession,
    request: MemCopyRequest,
    *,
    ctx: AgentContext,
    settings: Settings,
) -> tuple[Memory, Memory, UUID | None, list[str], str, str]:
    source = await _load_memory_for_read(session, request.memory_id)
    if source is None:
        raise NotFoundError(f"memory {request.memory_id} not found", memory_id=str(request.memory_id))

    _ensure_env_visible(source, ctx)
    src_env = await _load_env_for_cross_env_copy(session, source.env_id)
    dst_env = await _load_env_for_cross_env_copy(session, request.dst_env_id)
    if dst_env.id == source.env_id:
        raise InvalidInputError(
            "mem_copy/mem_move require a different destination environment",
            memory_id=str(source.id),
            env_id=str(source.env_id),
        )

    rbac.require("read", source.env_id, ctx)
    rbac.require("write", dst_env.id, ctx)

    source_model_id = src_env.default_embedding_model_id
    target_model_id = dst_env.default_embedding_model_id
    if source_model_id != target_model_id and not request.re_embed_if_model_mismatch:
        raise EmbeddingModelMismatchError(expected=target_model_id, actual=source_model_id)

    await _ensure_macro_available(session, env_id=dst_env.id, macro=source.macro)

    tag_names = await _load_tag_names(session, source.id) if request.copy_tags else []
    dst_memory = Memory(
        id=uuid4(),
        env_id=dst_env.id,
        kind=source.kind,
        status=MemoryStatus.active.value,
        title=source.title,
        body=source.body,
        trigger_description=source.trigger_description,
        steps=list(source.steps) if source.steps is not None else None,
        macro=source.macro,
        salience=float(source.salience),
        confidence=float(source.confidence),
        access_count=source.access_count,
        last_accessed_at=source.last_accessed_at,
        pinned=source.pinned,
        negative_feedback_count=source.negative_feedback_count,
        verified_at=source.verified_at,
        expires_at=source.expires_at,
        superseded_by=None,
        metadata_=dict(source.metadata_ or {}),
        decision_meta=dict(source.decision_meta) if source.decision_meta is not None else None,
        version=1,
    )
    if request.preserve_timestamps:
        dst_memory.created_at = source.created_at
        dst_memory.updated_at = source.updated_at

    session.add(dst_memory)
    await session.flush()
    await session.refresh(dst_memory)

    if tag_names:
        tag_map = await _upsert_tags(session, env_id=dst_env.id, names=tag_names)
        await _replace_memory_tags(
            session,
            memory_id=dst_memory.id,
            env_id=dst_env.id,
            tag_ids=[tag_map[n] for n in tag_names],
        )

    if request.copy_provenance:
        sources = (await session.execute(
            select(MemorySource).where(MemorySource.memory_id == source.id)
        )).scalars().all()
        for row in sources:
            session.add(MemorySource(
                memory_id=dst_memory.id,
                source_type=row.source_type,
                source_ref=row.source_ref,
                agent_id=None,
                created_at=row.created_at,
                evidence_span=row.evidence_span,
            ))

    if request.copy_lineage:
        lineage_rows = (await session.execute(
            select(MemoryLineage).where(
                (MemoryLineage.parent_memory_id == source.id)
                | (MemoryLineage.child_memory_id == source.id)
            )
        )).scalars().all()
        for row in lineage_rows:
            session.add(MemoryLineage(
                parent_memory_id=dst_memory.id if row.parent_memory_id == source.id else row.parent_memory_id,
                child_memory_id=dst_memory.id if row.child_memory_id == source.id else row.child_memory_id,
                relation=row.relation,
                created_at=row.created_at,
            ))

    lineage_edge_id: UUID | None = None
    if request.create_lineage_edge:
        session.add(MemoryLineage(
            parent_memory_id=source.id,
            child_memory_id=dst_memory.id,
            relation=LineageRelation.copied_from.value,
        ))
        lineage_edge_id = source.id

    await _record_audit(
        session,
        op="copy",
        memory=dst_memory,
        by_agent_id=ctx.agent_id,
        before=None,
        after=_audit_snapshot(dst_memory, tag_names=tag_names),
        extra_after={"copied_from": str(source.id)},
    )
    await enqueue_event(
        session,
        aggregate_type=OutboxAggregateType.memory,
        aggregate_id=dst_memory.id,
        aggregate_version=dst_memory.version,
        env_id=dst_env.id,
        op=_outbox_op_for(MemoryStatus.active, is_create=True),
        payload=_projection_payload(
            dst_memory,
            tag_names=tag_names,
            embedding_model_id=target_model_id,
        ),
        settings=settings,
    )
    return source, dst_memory, lineage_edge_id, tag_names, source_model_id, target_model_id


def _default_vector_store() -> Any:
    from memory_mcp.db.vector.qdrant import QdrantVectorStore

    return QdrantVectorStore(get_settings())


async def _apply_mem_copy_vector(
    *,
    source: Memory,
    dst_memory: Memory,
    tag_names: list[str],
    source_model_id: str,
    target_model_id: str,
    re_embed_if_model_mismatch: bool,
    delete_source_vector: bool = False,
) -> int:
    store = _default_vector_store()
    try:
        payload = _projection_payload(
            dst_memory,
            tag_names=tag_names,
            embedding_model_id=target_model_id,
        )
        if source_model_id != target_model_id and re_embed_if_model_mismatch:
            from memory_mcp.env_ops._embed import maybe_re_embed

            vectors = await maybe_re_embed(
                [dst_memory],
                source_model_id,
                target_model_id,
                session=None,  # type: ignore[arg-type]
            )
            body = vectors.get(dst_memory.id)
            if body is None:
                return 1
            await store.ensure_env_collection(env_id=dst_memory.env_id, dimension=len(body))
            await store.upsert(
                env_id=dst_memory.env_id,
                point_id=dst_memory.id,
                vector={"body": body},
                payload=payload,
            )
            return 0

        body = await store.get_vector(env_id=source.env_id, id=str(source.id), vector_name="body")
        trigger = await store.get_vector(env_id=source.env_id, id=str(source.id), vector_name="trigger")
        if body is None:
            return 1
        await store.ensure_env_collection(env_id=dst_memory.env_id, dimension=len(body))
        vectors: dict[str, list[float]] = {"body": body}
        if trigger is not None:
            vectors["trigger"] = trigger
        await store.upsert(
            env_id=dst_memory.env_id,
            point_id=dst_memory.id,
            vector=vectors,
            payload=payload,
        )
        if delete_source_vector:
            await store.delete(env_id=source.env_id, point_id=source.id)
        return 0
    except Exception:
        return 1
    finally:
        await store.close()


def _lineage_edge_id(parent_id: UUID, child_id: UUID, relation: str) -> str:
    return f"{parent_id}:{child_id}:{relation}"


async def _ensure_memory_hard_delete_allowed(session: AsyncSession, memory_id: UUID) -> None:
    lineage_ref = await session.scalar(
        select(MemoryLineage.parent_memory_id)
        .where(
            (MemoryLineage.parent_memory_id == memory_id)
            | (MemoryLineage.child_memory_id == memory_id)
        )
        .limit(1)
    )
    superseded_ref = await session.scalar(select(Memory.id).where(Memory.superseded_by == memory_id).limit(1))
    graph_ref = await session.scalar(select(GraphNode.id).where(GraphNode.memory_id == memory_id).limit(1))
    if lineage_ref is not None or superseded_ref is not None or graph_ref is not None:
        raise ConflictError(
            f"memory {memory_id} has references and cannot be hard-deleted",
            memory_id=str(memory_id),
        )


async def mem_copy(request: MemCopyRequest, *, ctx: AgentContext) -> MemCopyResponse:
    """Copy a memory from source env to destination env.

    Creates a new Memory row in dst with a fresh UUID, identical body/kind/payload/embeddings.
    Tags, memory_sources, and memory_lineage are conditionally copied per request flags.
    The source memory is UNCHANGED. See plan §3.2 + §17.7.
    """
    settings = get_settings()
    async with session_scope() as session:
        source, dst_memory, lineage_parent_id, tag_names, source_model_id, target_model_id = (
            await _copy_memory_in_session(session, request, ctx=ctx, settings=settings)
        )

    pending = await _apply_mem_copy_vector(
        source=source,
        dst_memory=dst_memory,
        tag_names=tag_names,
        source_model_id=source_model_id,
        target_model_id=target_model_id,
        re_embed_if_model_mismatch=request.re_embed_if_model_mismatch,
    )
    lineage_edge_id = (
        _lineage_edge_id(lineage_parent_id, dst_memory.id, LineageRelation.copied_from.value)
        if lineage_parent_id is not None
        else None
    )
    return MemCopyResponse(
        dst_memory_id=dst_memory.id,
        dst_env_id=dst_memory.env_id,
        lineage_edge_id=lineage_edge_id,
        pending_vector_rebuild=pending,
    )


async def mem_move(request: MemMoveRequest, *, ctx: AgentContext) -> MemMoveResponse:
    """Move a memory from source env to destination env.

    Implemented as mem_copy + supersede source (§17.7). After mem_move:
    - A new memory exists in dst env with a fresh UUID.
    - The source memory has status='superseded' and superseded_by pointing at the new dst memory.
    This uses the existing Memory.superseded_by FK with no schema change.

    Behavior is identical to mem_copy plus the supersede step. The dst memory's
    body/kind/payload are byte-identical to the source's at the moment of move.
    """
    settings = get_settings()
    copy_request = MemCopyRequest(
        memory_id=request.memory_id,
        dst_env_id=request.dst_env_id,
        copy_tags=request.copy_tags,
        copy_provenance=request.copy_provenance,
        create_lineage_edge=request.create_lineage_edge,
        preserve_timestamps=request.preserve_timestamps,
        re_embed_if_model_mismatch=request.re_embed_if_model_mismatch,
        copy_lineage=request.copy_lineage,
    )
    async with session_scope() as session:
        source, dst_memory, lineage_parent_id, tag_names, source_model_id, target_model_id = (
            await _copy_memory_in_session(session, copy_request, ctx=ctx, settings=settings)
        )
        old_tag_names = await _load_tag_names(session, source.id)
        source_before = _audit_snapshot(source, tag_names=old_tag_names)
        source_status = "superseded"

        if request.redirect_source:
            await session.execute(
                update(Memory)
                .where(Memory.id == source.id)
                .values(
                    status=MemoryStatus.superseded.value,
                    superseded_by=dst_memory.id,
                    version=source.version + 1,
                    updated_at=func.now(),
                )
            )
            await session.refresh(source)
            await _record_audit(
                session,
                op="move",
                memory=source,
                by_agent_id=ctx.agent_id,
                before=source_before,
                after=_audit_snapshot(source, tag_names=old_tag_names),
                extra_after={"superseded_by": str(dst_memory.id)},
            )
            await enqueue_event(
                session,
                aggregate_type=OutboxAggregateType.memory,
                aggregate_id=source.id,
                aggregate_version=source.version,
                env_id=source.env_id,
                op=_outbox_op_for(MemoryStatus.superseded, is_create=False),
                payload=_projection_payload(
                    source,
                    tag_names=old_tag_names,
                    embedding_model_id=source_model_id,
                ),
                settings=settings,
            )
        else:
            await _ensure_memory_hard_delete_allowed(session, source.id)
            await enqueue_event(
                session,
                aggregate_type=OutboxAggregateType.memory,
                aggregate_id=source.id,
                aggregate_version=source.version,
                env_id=source.env_id,
                op=OutboxOp.tombstone,
                payload=_projection_payload(
                    source,
                    tag_names=old_tag_names,
                    embedding_model_id=source_model_id,
                ),
                settings=settings,
            )
            await _record_audit(
                session,
                op="hard_delete",
                memory=source,
                by_agent_id=ctx.agent_id,
                before=source_before,
                after={"deleted": True, "moved_to": str(dst_memory.id)},
            )
            await session.execute(delete(Memory).where(Memory.id == source.id))
            source_status = "deleted"

    pending = await _apply_mem_copy_vector(
        source=source,
        dst_memory=dst_memory,
        tag_names=tag_names,
        source_model_id=source_model_id,
        target_model_id=target_model_id,
        re_embed_if_model_mismatch=request.re_embed_if_model_mismatch,
        delete_source_vector=source_status == "deleted",
    )
    lineage_edge_id = (
        _lineage_edge_id(lineage_parent_id, dst_memory.id, LineageRelation.copied_from.value)
        if lineage_parent_id is not None
        else None
    )
    return MemMoveResponse(
        dst_memory_id=dst_memory.id,
        dst_env_id=dst_memory.env_id,
        lineage_edge_id=lineage_edge_id,
        pending_vector_rebuild=pending,
        source_memory_status=source_status,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# memory_write
# ---------------------------------------------------------------------------


async def memory_write(
    req: MemoryWriteRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemoryResponse:
    """Insert a new memory in ``active`` status with version=1.

    Side effects (single transaction):

    1. Resolve ``env_id`` (explicit or sole attached).
    2. Insert memory row.
    3. Upsert tag rows; insert ``memory_tags`` join rows.
    4. Insert ``memory_sources`` provenance row.
    5. Insert ``audit_log`` row (op=``create``).
    6. Enqueue outbox event (``op=upsert``).
    """
    settings = settings or get_settings()
    env_id = _resolve_env_id(explicit=req.env_id, ctx=ctx)
    rbac.require("write", env_id, ctx)

    tag_names = _normalize_tags(req.tags)
    steps, macro = _normalize_playbook_fields(
        kind=req.kind,
        steps=req.steps,
        macro=req.macro,
    )

    try:
        async with session_scope() as s:
            return await _memory_write_in_session(
                req,
                ctx=ctx,
                settings=settings,
                session=s,
                env_id=env_id,
                tag_names=tag_names,
                steps=steps,
                macro=macro,
            )
    except IntegrityError as exc:
        if _is_macro_integrity_error(exc):
            raise InvalidInputError("macro already in use in this env") from exc
        raise


async def _memory_write_in_session(
    req: MemoryWriteRequest,
    *,
    ctx: AgentContext,
    settings: Settings,
    session: AsyncSession,
    env_id: UUID,
    tag_names: list[str],
    steps: list[str] | None,
    macro: str | None,
) -> MemoryResponse:
    s = session
    await _ensure_macro_available(s, env_id=env_id, macro=macro)
    try:
        embedding_model_id = await _load_env_embedding_model(s, env_id)
        decision_meta = await _validate_decision_meta_for_kind(
            kind=req.kind.value,
            decision_meta=req.decision_meta,
            env_id=env_id,
            session=s,
        )

        memory = Memory(
            env_id=env_id,
            kind=req.kind.value,
            status=MemoryStatus.active.value,
            title=req.title,
            body=req.body,
            trigger_description=req.trigger_description,
            steps=steps,
            macro=macro,
            metadata_=req.metadata,
            decision_meta=decision_meta,
            pinned=req.pinned,
            expires_at=req.expires_at,
        )
        if req.salience is not None:
            memory.salience = req.salience
        if req.confidence is not None:
            memory.confidence = req.confidence

        s.add(memory)
        await s.flush()  # gen_random_uuid() runs; memory.id populated
        await s.refresh(memory)

        # Tags
        if tag_names:
            tag_map = await _upsert_tags(s, env_id=env_id, names=tag_names)
            await _replace_memory_tags(
                s,
                memory_id=memory.id,
                env_id=env_id,
                tag_ids=[tag_map[n] for n in tag_names],
            )

        # Provenance
        await s.execute(
            insert(MemorySource).values(
                memory_id=memory.id,
                source_type=req.source_type.value,
                source_ref=req.source_ref if req.source_ref is not None else str(ctx.session_id) if ctx.session_id else None,
                agent_id=ctx.agent_id,
                evidence_span=req.evidence_span,
            )
        )

        if req.entity_links:
            memory_node = await _ensure_memory_graph_node(s, env_id=env_id, memory_id=memory.id)
            for entity_id in list(dict.fromkeys(req.entity_links)):
                entity_node = await _ensure_entity_graph_node(s, env_id=env_id, entity_id=entity_id)
                existing_rel = (await s.execute(
                    select(Relation).where(
                        Relation.src_node_id == memory_node.id,
                        Relation.dst_node_id == entity_node.id,
                        Relation.type == "mentions",
                    )
                )).scalar_one_or_none()
                if existing_rel is None:
                    s.add(
                        Relation(
                            env_id=env_id,
                            src_node_id=memory_node.id,
                            dst_node_id=entity_node.id,
                            type="mentions",
                            properties={},
                        )
                    )
            await s.flush()

        # Audit
        await _record_audit(
            s,
            op="create",
            memory=memory,
            by_agent_id=ctx.agent_id,
            before=None,
            after=_audit_snapshot(memory, tag_names=tag_names),
        )

        # Outbox
        payload = _projection_payload(
            memory, tag_names=tag_names, embedding_model_id=embedding_model_id
        )
        await enqueue_event(
            s,
            aggregate_type=OutboxAggregateType.memory,
            aggregate_id=memory.id,
            aggregate_version=memory.version,
            env_id=env_id,
            op=_outbox_op_for(MemoryStatus.active, is_create=True),
            payload=payload,
            settings=settings,
        )

        return _to_response(memory, tag_names)
    except IntegrityError as exc:
        if _is_macro_integrity_error(exc):
            raise InvalidInputError("macro already in use in this env") from exc
        raise


# ---------------------------------------------------------------------------
# memory_get / memory_get_many
# ---------------------------------------------------------------------------


async def memory_get(
    memory_id: UUID,
    *,
    ctx: AgentContext,
    bump_access: bool = True,
) -> MemoryResponse:
    """Read a memory by id.

    Bumps ``access_count`` and ``last_accessed_at`` for visible
    statuses (``proposed``/``active``/``stale``) — the
    ``require_version_monotonic`` trigger explicitly allows updates that
    do not change ``version`` (see migration 0001 lines 67–70).

    Salience is recomputed in the same UPDATE so the on-read access bump,
    timestamp, and salience score all land atomically. The recompute uses
    a Python-side ``now`` (not ``func.now()``) so the value passed to
    :func:`compute_salience` matches what we write to the row.

    Retired/superseded/archived memories return without an access bump
    (still readable by id for audit/reference). The caller can disable
    bumping entirely with ``bump_access=False``.
    """
    async with session_scope() as s:
        memory = await _load_memory_for_read(s, memory_id)
        if memory is None:
            raise NotFoundError(
                f"memory {memory_id} not found",
                memory_id=str(memory_id),
            )

        rbac.require("read", memory.env_id, ctx)
        _ensure_env_visible(memory, ctx)

        if bump_access and MemoryStatus(memory.status) in _VISIBLE_IN_PROJECTION:
            now = dt.datetime.now(dt.UTC)
            weights = salience_weights_from_settings(get_settings())
            new_salience = compute_salience(
                SalienceInputs(
                    access_count=memory.access_count + 1,
                    last_accessed_at=now,
                    confidence=float(memory.confidence),
                    pinned=memory.pinned,
                    negative_feedback_count=memory.negative_feedback_count,
                    verified_at=memory.verified_at,
                    created_at=memory.created_at,
                    # Phase 1 (v0.14) — without these, the access bump
                    # erases the citation-derived component of salience
                    # for cited memories. R-B4 regression fix.
                    reference_count_rel_link=memory.reference_count_rel_link,
                    reference_count_lineage=memory.reference_count_lineage,
                    reference_count_task=memory.reference_count_task,
                    reference_count_playbook=memory.reference_count_playbook,
                    # Phase 1e-d (v0.14.1) — read-only here. The access-bump
                    # path computes salience under the current formula but
                    # intentionally does NOT stamp ``salience_formula_version``:
                    # only the recount pass stamps. Next recount picks the row
                    # up via the integer-counter / formula-version mismatch
                    # union and re-stamps then.
                    reference_authority=float(memory.reference_authority or 0),
                ),
                now=now,
                weights=weights,
            )
            await s.execute(
                update(Memory)
                .where(Memory.id == memory.id)
                .values(
                    access_count=Memory.access_count + 1,
                    last_accessed_at=now,
                    salience=new_salience,
                )
            )
            await s.refresh(memory)

        tag_names = await _load_tag_names(s, memory.id)
        return _to_response(memory, tag_names)


async def memory_get_many(
    memory_ids: list[UUID],
    *,
    ctx: AgentContext,
    bump_access: bool = False,
) -> list[MemoryResponse]:
    """Bulk-read memories. Output order matches ``memory_ids`` input order;
    missing or env-filtered ids are silently dropped.

    ``bump_access`` defaults to ``False`` here — bulk reads are typically
    used for batch hydration and shouldn't pollute access stats.
    """
    if not memory_ids:
        return []

    async with session_scope() as s:
        rows = await s.execute(select(Memory).where(Memory.id.in_(memory_ids)))
        memories: dict[UUID, Memory] = {m.id: m for m in rows.scalars().all()}

        # Visibility filter (no DB calls).
        visible: list[Memory] = []
        for mid in memory_ids:
            m = memories.get(mid)
            if m is None:
                continue
            if ctx.attached_env_ids and m.env_id not in ctx.attached_env_ids:
                continue
            rbac.require("read", m.env_id, ctx)
            visible.append(m)

        if not visible:
            return []

        visible_ids = [m.id for m in visible]

        # One JOIN-style fetch for all tags at once.
        tag_rows = await s.execute(
            select(MemoryTag.memory_id, Tag.name)
            .join(Tag, Tag.id == MemoryTag.tag_id)
            .where(MemoryTag.memory_id.in_(visible_ids))
            .order_by(MemoryTag.memory_id, Tag.name)
        )
        tags_by_id: dict[UUID, list[str]] = {mid: [] for mid in visible_ids}
        for mid, name in tag_rows.all():
            tags_by_id[mid].append(name)

        if bump_access:
            bumpable = [
                m for m in visible
                if MemoryStatus(m.status) in _VISIBLE_IN_PROJECTION
            ]
            if bumpable:
                now = dt.datetime.now(dt.UTC)
                weights = salience_weights_from_settings(get_settings())
                # Per-row UPDATE so each memory gets its own recomputed
                # salience. Loop is bounded by the caller's id list so
                # this is O(N) round-trips by design — bulk reads are
                # typically small (Phase 1 pagination caps apply at the
                # caller layer). If profiling shows hot-spotting, we can
                # switch to a single CASE/WHEN UPDATE.
                for m in bumpable:
                    new_salience = compute_salience(
                        SalienceInputs(
                            access_count=m.access_count + 1,
                            last_accessed_at=now,
                            confidence=float(m.confidence),
                            pinned=m.pinned,
                            negative_feedback_count=m.negative_feedback_count,
                            verified_at=m.verified_at,
                            created_at=m.created_at,
                            # Phase 1 (v0.14) — bulk-read bump must preserve
                            # citation contribution. R-B4 regression fix.
                            reference_count_rel_link=m.reference_count_rel_link,
                            reference_count_lineage=m.reference_count_lineage,
                            reference_count_task=m.reference_count_task,
                            reference_count_playbook=m.reference_count_playbook,
                            # Phase 1e-d — read-only (see single-read callsite
                            # above for full rationale). Access-bump never
                            # stamps ``salience_formula_version``; recount
                            # owns that stamp.
                            reference_authority=float(m.reference_authority or 0),
                        ),
                        now=now,
                        weights=weights,
                    )
                    await s.execute(
                        update(Memory)
                        .where(Memory.id == m.id)
                        .values(
                            access_count=Memory.access_count + 1,
                            last_accessed_at=now,
                            salience=new_salience,
                        )
                    )
                # Refresh to expose new access_count / last_accessed_at /
                # salience to the response.
                for m in bumpable:
                    await s.refresh(m)

        return [_to_response(m, tags_by_id.get(m.id, [])) for m in visible]


# ---------------------------------------------------------------------------
# memory_update + lifecycle wrappers
# ---------------------------------------------------------------------------


async def memory_update(
    memory_id: UUID,
    patch: MemoryUpdatePatch,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
    _audit_extra: dict[str, Any] | None = None,
) -> MemoryResponse:
    """Patch a memory with optimistic-lock semantics.

    Field absence (``model_dump(exclude_unset=True)``) means *no change*;
    explicit ``None`` clears the field (only valid for nullable columns).

    Status transitions are validated against
    :func:`is_valid_transition`; bad transitions raise
    :class:`InvalidTransitionError`. ``RETURNING``-empty after the
    UPDATE means either the row vanished (unlikely; we already loaded
    it) or a concurrent writer bumped the version → ``VersionConflictError``.
    """
    settings = settings or get_settings()
    fields_set = patch.model_fields_set

    try:
        async with session_scope() as s:
            memory = await _load_memory_for_read(s, memory_id)
            if memory is None:
                raise NotFoundError(
                    f"memory {memory_id} not found",
                    memory_id=str(memory_id),
                )

            rbac.require("write", memory.env_id, ctx)
            _ensure_env_visible(memory, ctx)

            if memory.version != patch.expected_version:
                raise VersionConflictError(
                    expected=patch.expected_version,
                    actual=memory.version,
                )

            old_tag_names = await _load_tag_names(s, memory.id)
            before = _audit_snapshot(memory, tag_names=old_tag_names)
            old_status = MemoryStatus(memory.status)
            final_kind = patch.kind if "kind" in fields_set and patch.kind is not None else MemoryKind(memory.kind)
            final_steps = patch.steps if "steps" in fields_set else memory.steps
            final_macro = patch.macro if "macro" in fields_set else memory.macro
            normalized_steps, normalized_macro = _normalize_playbook_fields(
                kind=final_kind,
                steps=list(final_steps) if final_steps is not None else None,
                macro=final_macro,
            )
            await _ensure_macro_available(
                s,
                env_id=memory.env_id,
                macro=normalized_macro,
                exclude_memory_id=memory.id,
            )

            # Build the SQL update dict from explicit fields_set ONLY
            # (so absent != cleared).
            update_values: dict[str, Any] = {
                "version": memory.version + 1,
                "updated_at": func.now(),
            }
            new_status = old_status
            if "title" in fields_set:
                update_values["title"] = patch.title
            if "body" in fields_set:
                if patch.body is None:
                    raise ValueError("body cannot be cleared (NOT NULL)")
                update_values["body"] = patch.body
            if "trigger_description" in fields_set:
                update_values["trigger_description"] = patch.trigger_description
            if "steps" in fields_set:
                update_values["steps"] = normalized_steps
            if "macro" in fields_set:
                update_values["macro"] = normalized_macro
            resulting_kind = memory.kind
            if "kind" in fields_set and patch.kind is not None:
                resulting_kind = patch.kind.value
                update_values["kind"] = patch.kind.value
            if "status" in fields_set and patch.status is not None:
                new_status = patch.status
                if not is_valid_transition(old_status, new_status):
                    raise InvalidTransitionError(src=old_status.value, dst=new_status.value)
                update_values["status"] = new_status.value
            if "metadata" in fields_set:
                # Use ORM attribute as key — string "metadata" collides with
                # SQLAlchemy's reserved MetaData attribute on the Table object.
                update_values[Memory.metadata_] = patch.metadata or {}
            if "decision_meta" in fields_set:
                update_values["decision_meta"] = await _validate_decision_meta_for_kind(
                    kind=resulting_kind,
                    decision_meta=patch.decision_meta,
                    env_id=memory.env_id,
                    session=s,
                )
            elif memory.decision_meta is not None and resulting_kind != MemoryKind.decision.value:
                raise InvalidInputError("decision_meta only valid for kind=decision")
            if "salience" in fields_set and patch.salience is not None:
                update_values["salience"] = patch.salience
            # Phase 1e-d: bundle the formula-version stamp with the
            # salience write so recount's recompute leaves a coherent
            # ``(salience, salience_formula_version)`` pair on the row.
            if (
                "salience_formula_version" in fields_set
                and patch.salience_formula_version is not None
            ):
                update_values["salience_formula_version"] = patch.salience_formula_version
            if "confidence" in fields_set and patch.confidence is not None:
                update_values["confidence"] = patch.confidence
            if "pinned" in fields_set and patch.pinned is not None:
                update_values["pinned"] = patch.pinned
            if "expires_at" in fields_set:
                update_values["expires_at"] = patch.expires_at
            if "verified_at" in fields_set:
                update_values["verified_at"] = patch.verified_at

            # Execute the optimistic-lock update.
            result = await s.execute(
                update(Memory)
                .where(Memory.id == memory.id, Memory.version == patch.expected_version)
                .values(update_values)
            )
            if result.rowcount == 0:  # type: ignore[attr-defined]
                # Row exists (we just loaded it) but version changed under us.
                # Raising aborts the transaction via session_scope's exception
                # handler — no explicit rollback needed.
                async with session_scope() as s2:
                    fresh = await _load_memory_for_read(s2, memory_id)
                actual = fresh.version if fresh else patch.expected_version + 1
                raise VersionConflictError(
                    expected=patch.expected_version, actual=actual
                )

            await s.refresh(memory)

            # Tags
            if "tags" in fields_set:
                tag_names_in = _normalize_tags(patch.tags or [])
                tag_map = await _upsert_tags(s, env_id=memory.env_id, names=tag_names_in)
                await _replace_memory_tags(
                    s,
                    memory_id=memory.id,
                    env_id=memory.env_id,
                    tag_ids=[tag_map[n] for n in tag_names_in],
                )
                new_tag_names = tag_names_in
            else:
                new_tag_names = old_tag_names

            after = _audit_snapshot(memory, tag_names=new_tag_names)
            await _record_audit(
                s,
                op="update",
                memory=memory,
                by_agent_id=ctx.agent_id,
                before=before,
                after=after,
                extra_after=_audit_extra,
            )

            embedding_model_id = await _load_env_embedding_model(s, memory.env_id)
            outbox_payload = _projection_payload(
                memory,
                tag_names=new_tag_names,
                embedding_model_id=embedding_model_id,
            )
            await enqueue_event(
                s,
                aggregate_type=OutboxAggregateType.memory,
                aggregate_id=memory.id,
                aggregate_version=memory.version,
                env_id=memory.env_id,
                op=_outbox_op_for(new_status, is_create=False),
                payload=outbox_payload,
                settings=settings,
            )

    except IntegrityError as exc:
        if _is_macro_integrity_error(exc):
            raise InvalidInputError("macro already in use in this env") from exc
        raise

    return _to_response(memory, new_tag_names)


async def memory_archive(
    memory_id: UUID,
    *,
    expected_version: int,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemoryResponse:
    """Convenience wrapper: status → archived (with tombstone projection)."""
    return await memory_update(
        memory_id,
        MemoryUpdatePatch(
            expected_version=expected_version,
            status=MemoryStatus.archived,
        ),
        ctx=ctx,
        settings=settings,
    )


async def memory_retire(
    memory_id: UUID,
    *,
    expected_version: int,
    reason: str,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemoryResponse:
    """Convenience wrapper: status → retired with reason captured in audit_log."""
    if not reason or not reason.strip():
        raise ValueError("reason is required for memory_retire")
    return await memory_update(
        memory_id,
        MemoryUpdatePatch(
            expected_version=expected_version,
            status=MemoryStatus.retired,
        ),
        ctx=ctx,
        settings=settings,
        _audit_extra={"retire_reason": reason.strip()},
    )


@dataclass(frozen=True)
class _HardDeleteCandidate:
    id: UUID
    env_id: UUID
    lifecycle_before: str
    edge_reason: str
    version: int
    depth: int

    def to_response(self) -> MemoryHardDeleteAffected:
        return MemoryHardDeleteAffected(
            id=self.id,
            lifecycle_before=self.lifecycle_before,
            edge_reason=self.edge_reason,
            version=self.version,
            depth=self.depth,
        )


def _order_hard_delete_candidates(
    candidates: dict[UUID, _HardDeleteCandidate],
) -> list[_HardDeleteCandidate]:
    return sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.depth, str(candidate.id)),
    )


async def _load_forward_hard_delete_dependents(
    session: AsyncSession,
    memory_id: UUID,
) -> dict[UUID, tuple[Memory, str]]:
    dependents: dict[UUID, tuple[Memory, str]] = {}

    lineage_rows = (
        await session.execute(
            select(MemoryLineage.relation, Memory)
            .join(Memory, Memory.id == MemoryLineage.child_memory_id)
            .where(MemoryLineage.parent_memory_id == memory_id)
        )
    ).all()
    for relation, dependent in lineage_rows:
        dependents.setdefault(dependent.id, (dependent, relation))

    superseded_rows = (
        await session.execute(select(Memory).where(Memory.superseded_by == memory_id))
    ).scalars().all()
    for dependent in superseded_rows:
        dependents.setdefault(dependent.id, (dependent, LineageRelation.supersedes.value))

    return dependents


def _blast_radius_error(
    *,
    cap_hit: Literal["depth", "count"],
    limit: int,
    affected: dict[UUID, _HardDeleteCandidate],
    offending_depth: int | None = None,
    offending_id: UUID | None = None,
) -> BlastRadiusExceededError:
    ordered = _order_hard_delete_candidates(affected)
    would_affect = [candidate.id for candidate in ordered]
    if offending_id is not None and offending_id not in would_affect:
        would_affect.append(offending_id)

    details: dict[str, Any] = {
        "affected": [candidate.to_response().model_dump(mode="json") for candidate in ordered],
    }
    if offending_depth is not None:
        details["offending_depth"] = offending_depth
        message = (
            "BLAST_RADIUS_EXCEEDED: depth "
            f"{offending_depth} exceeds max_cascade_depth={limit}"
        )
    else:
        message = (
            "BLAST_RADIUS_EXCEEDED: affected row count would exceed "
            f"max_cascade_count={limit}"
        )

    return BlastRadiusExceededError(
        cap_hit=cap_hit,
        limit=limit,
        would_affect=would_affect,
        message=message,
        **details,
    )


async def _collect_hard_delete_affected(
    session: AsyncSession,
    root: Memory,
    *,
    request: MemoryHardDeleteRequest,
    ctx: AgentContext,
) -> list[_HardDeleteCandidate]:
    visited: dict[UUID, _HardDeleteCandidate] = {
        root.id: _HardDeleteCandidate(
            id=root.id,
            env_id=root.env_id,
            lifecycle_before=root.status,
            edge_reason="target",
            version=root.version,
            depth=0,
        )
    }
    queue: deque[_HardDeleteCandidate] = deque([visited[root.id]])

    while queue:
        current = queue.popleft()
        dependents = await _load_forward_hard_delete_dependents(session, current.id)
        for dependent, edge_reason in dependents.values():
            if dependent.id in visited:
                continue

            depth = current.depth + 1
            if depth > request.max_cascade_depth:
                raise _blast_radius_error(
                    cap_hit="depth",
                    limit=request.max_cascade_depth,
                    affected=visited,
                    offending_depth=depth,
                    offending_id=dependent.id,
                )

            _ensure_env_visible(dependent, ctx)
            rbac.require("write", dependent.env_id, ctx)

            candidate = _HardDeleteCandidate(
                id=dependent.id,
                env_id=dependent.env_id,
                lifecycle_before=dependent.status,
                edge_reason=edge_reason,
                version=dependent.version,
                depth=depth,
            )
            if len(visited) + 1 > request.max_cascade_count:
                raise _blast_radius_error(
                    cap_hit="count",
                    limit=request.max_cascade_count,
                    affected=visited,
                    offending_id=dependent.id,
                )
            visited[dependent.id] = candidate
            queue.append(candidate)

    return _order_hard_delete_candidates(visited)


# ---------------------------------------------------------------------------
# memory_hard_delete  (Phase 1.1, v0.11)
# ---------------------------------------------------------------------------


async def memory_hard_delete(
    memory_id: UUID,
    request: MemoryHardDeleteRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemoryHardDeleteResponse:
    """Permanently delete a memory's canonical row, body, and projections."""
    settings = settings or get_settings()
    if not request.confirm_destroy:
        raise InvalidInputError(
            "mem_hard_delete requires confirm_destroy=true",
            memory_id=str(memory_id),
        )

    reason = request.reason.strip()
    if not reason:
        raise InvalidInputError(
            "mem_hard_delete requires a non-empty reason",
            memory_id=str(memory_id),
        )

    if not request.cascade:
        async with session_scope() as session:
            memory = await _load_memory_for_read(session, memory_id)
            if memory is None:
                raise NotFoundError(
                    f"memory {memory_id} not found",
                    memory_id=str(memory_id),
                )
            rbac.require("write", memory.env_id, ctx)
            _ensure_env_visible(memory, ctx)

            if memory.version != request.expected_version:
                raise VersionConflictError(
                    expected=request.expected_version, actual=memory.version
                )

            await _ensure_memory_hard_delete_allowed(session, memory_id)

            env_id = memory.env_id
            env_row = await session.scalar(
                select(Environment).where(Environment.id == env_id)
            )
            if env_row is None:
                raise NotFoundError(f"env {env_id} not found", env_id=str(env_id))
            embedding_model_id = env_row.default_embedding_model_id

            tag_names = await _load_tag_names(session, memory_id)
            before = _audit_snapshot(memory, tag_names=tag_names)
            original_kind = memory.kind
            original_status = memory.status

            tombstone_id = uuid4()
            await session.execute(
                insert(MemoryTombstone).values(
                    id=tombstone_id,
                    env_id=env_id,
                    deleted_by_agent_id=ctx.agent_id,
                    reason=reason,
                    original_kind=original_kind,
                    original_status=original_status,
                )
            )

            await _record_audit(
                session,
                op="hard_delete",
                memory=memory,
                by_agent_id=ctx.agent_id,
                before=before,
                after={
                    "deleted": True,
                    "tombstone_id": str(tombstone_id),
                    "reason": reason,
                },
            )

            projection_payload = _projection_payload(
                memory,
                tag_names=tag_names,
                embedding_model_id=embedding_model_id,
            )
            await enqueue_event(
                session,
                aggregate_type=OutboxAggregateType.memory,
                aggregate_id=memory.id,
                aggregate_version=memory.version + 1,
                env_id=env_id,
                op=OutboxOp.tombstone,
                payload=projection_payload,
                settings=settings,
            )

            await session.execute(delete(Memory).where(Memory.id == memory_id))
            deleted_at = dt.datetime.now(tz=dt.timezone.utc)

        projection_status = HardDeleteProjectionStatus(
            qdrant="pending",
            neo4j="pending",
        )

        return MemoryHardDeleteResponse(
            deleted_id=memory_id,
            deleted_at=deleted_at,
            canonical_deleted=True,
            projection_eviction=projection_status,
            tombstone_id=tombstone_id,
        )

    cascade_root = uuid4()
    async with session_scope() as session:
        root = await _load_memory_for_read(session, memory_id)
        if root is None:
            raise NotFoundError(
                f"memory {memory_id} not found",
                memory_id=str(memory_id),
            )
        rbac.require("write", root.env_id, ctx)
        _ensure_env_visible(root, ctx)
        if root.version != request.expected_version:
            exc = VersionConflictError(
                expected=request.expected_version,
                actual=root.version,
            )
            exc.details["memory_id"] = str(root.id)
            raise exc

        affected = await _collect_hard_delete_affected(
            session,
            root,
            request=request,
            ctx=ctx,
        )

    affected_response = [candidate.to_response() for candidate in affected]
    if request.dry_run:
        return MemoryHardDeleteResponse(
            deleted_id=memory_id,
            canonical_deleted=False,
            cascade_root=cascade_root,
            affected=affected_response,
        )

    projection_status = HardDeleteProjectionStatus(
        qdrant="pending",
        neo4j="pending",
    )
    deleted_at = dt.datetime.now(tz=dt.timezone.utc)
    root_tombstone_id: UUID | None = None

    async with session_scope() as session:
        embedding_models: dict[UUID, str] = {}
        for candidate in affected:
            locked = (
                await session.execute(
                    select(Memory)
                    .where(Memory.id == candidate.id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if locked is None:
                exc = VersionConflictError(
                    expected=candidate.version,
                    actual=candidate.version + 1,
                )
                exc.details["memory_id"] = str(candidate.id)
                raise exc

            _ensure_env_visible(locked, ctx)
            rbac.require("write", locked.env_id, ctx)
            if locked.version != candidate.version:
                exc = VersionConflictError(
                    expected=candidate.version,
                    actual=locked.version,
                )
                exc.details["memory_id"] = str(candidate.id)
                raise exc

            embedding_model_id = embedding_models.get(locked.env_id)
            if embedding_model_id is None:
                env_row = await session.scalar(
                    select(Environment).where(Environment.id == locked.env_id)
                )
                if env_row is None:
                    raise NotFoundError(
                        f"env {locked.env_id} not found",
                        env_id=str(locked.env_id),
                    )
                embedding_model_id = env_row.default_embedding_model_id
                embedding_models[locked.env_id] = embedding_model_id

            tag_names = await _load_tag_names(session, locked.id)
            before = _audit_snapshot(locked, tag_names=tag_names)
            tombstone_id = uuid4()
            if candidate.id == memory_id:
                root_tombstone_id = tombstone_id

            await session.execute(
                insert(MemoryTombstone).values(
                    id=tombstone_id,
                    env_id=locked.env_id,
                    deleted_by_agent_id=ctx.agent_id,
                    cascade_root=cascade_root,
                    reason=reason,
                    original_kind=locked.kind,
                    original_status=locked.status,
                )
            )

            await _record_audit(
                session,
                op="hard_delete",
                memory=locked,
                by_agent_id=ctx.agent_id,
                before=before,
                after={
                    "deleted": True,
                    "tombstone_id": str(tombstone_id),
                    "reason": reason,
                    "cascade_root": str(cascade_root),
                },
            )

            await enqueue_event(
                session,
                aggregate_type=OutboxAggregateType.memory,
                aggregate_id=locked.id,
                aggregate_version=locked.version + 1,
                env_id=locked.env_id,
                op=OutboxOp.tombstone,
                payload=_projection_payload(
                    locked,
                    tag_names=tag_names,
                    embedding_model_id=embedding_model_id,
                ),
                settings=settings,
            )
            await session.execute(delete(Memory).where(Memory.id == locked.id))

    if root_tombstone_id is None:
        raise NotFoundError(
            f"memory {memory_id} not found",
            memory_id=str(memory_id),
        )

    return MemoryHardDeleteResponse(
        deleted_id=memory_id,
        deleted_at=deleted_at,
        canonical_deleted=True,
        projection_eviction=projection_status,
        tombstone_id=root_tombstone_id,
        cascade_root=cascade_root,
        affected=affected_response,
    )


# ---------------------------------------------------------------------------
# memory_supersede
# ---------------------------------------------------------------------------


async def memory_supersede(
    old_id: UUID,
    req: MemorySupersedeRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> tuple[MemoryResponse, MemoryResponse]:
    """Atomically: insert new memory + mark old as superseded → new.

    Returns ``(old_after, new)``. Both memory rows, their lineage edge,
    audit entries, and outbox events commit together. If anything fails,
    nothing changes.

    v1: same-env supersede only. Cross-env is admin-only and deferred.
    """
    settings = settings or get_settings()
    new_req = req.new
    new_tag_names = _normalize_tags(new_req.tags)

    async with session_scope() as s:
        old = await _load_memory_for_read(s, old_id)
        if old is None:
            raise NotFoundError(
                f"memory {old_id} not found",
                memory_id=str(old_id),
            )

        rbac.require("write", old.env_id, ctx)
        _ensure_env_visible(old, ctx)

        # Cross-env supersede gate.
        if new_req.env_id is not None and new_req.env_id != old.env_id:
            raise InvalidTransitionError(
                src=f"env={old.env_id}",
                dst=f"env={new_req.env_id}",
            )
        env_id = old.env_id
        rbac.require("write", env_id, ctx)

        if old.version != req.expected_version:
            raise VersionConflictError(
                expected=req.expected_version, actual=old.version
            )

        old_status = MemoryStatus(old.status)
        if not is_valid_transition(old_status, MemoryStatus.superseded):
            raise InvalidTransitionError(
                src=old_status.value, dst=MemoryStatus.superseded.value
            )

        embedding_model_id = await _load_env_embedding_model(s, env_id)
        new_decision_meta = await _validate_decision_meta_for_kind(
            kind=new_req.kind.value,
            decision_meta=new_req.decision_meta,
            env_id=env_id,
            session=s,
        )

        # 1. Insert new memory (version=1, status=active)
        new_memory = Memory(
            env_id=env_id,
            kind=new_req.kind.value,
            status=MemoryStatus.active.value,
            title=new_req.title,
            body=new_req.body,
            trigger_description=new_req.trigger_description,
            metadata_=new_req.metadata,
            decision_meta=new_decision_meta,
            pinned=new_req.pinned,
            expires_at=new_req.expires_at,
        )
        if new_req.salience is not None:
            new_memory.salience = new_req.salience
        if new_req.confidence is not None:
            new_memory.confidence = new_req.confidence
        s.add(new_memory)
        await s.flush()
        await s.refresh(new_memory)

        # New memory tags
        if new_tag_names:
            tag_map = await _upsert_tags(s, env_id=env_id, names=new_tag_names)
            await _replace_memory_tags(
                s,
                memory_id=new_memory.id,
                env_id=env_id,
                tag_ids=[tag_map[n] for n in new_tag_names],
            )

        # New memory provenance
        await s.execute(
            insert(MemorySource).values(
                memory_id=new_memory.id,
                source_type=MemorySourceType.agent.value,
                source_ref=str(ctx.session_id) if ctx.session_id else None,
                agent_id=ctx.agent_id,
            )
        )

        # 2. Update old: status=superseded, superseded_by=new.id, version+=1
        old_tag_names = await _load_tag_names(s, old.id)
        old_before = _audit_snapshot(old, tag_names=old_tag_names)

        result = await s.execute(
            update(Memory)
            .where(Memory.id == old.id, Memory.version == req.expected_version)
            .values(
                status=MemoryStatus.superseded.value,
                superseded_by=new_memory.id,
                version=old.version + 1,
                updated_at=func.now(),
            )
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]
            # Concurrent writer bumped old.version after our pre-check. The
            # raise below triggers session_scope rollback — the new memory
            # we just inserted is discarded along with everything else.
            raise VersionConflictError(
                expected=req.expected_version, actual=req.expected_version + 1
            )
        await s.refresh(old)

        # 3. Lineage: parent=old, child=new, relation=supersedes
        # (reads as "new memory supersedes old memory")
        await s.execute(
            insert(MemoryLineage).values(
                parent_memory_id=old.id,
                child_memory_id=new_memory.id,
                relation=LineageRelation.supersedes.value,
            )
        )

        # 4. Audit: both rows
        await _record_audit(
            s,
            op="create",
            memory=new_memory,
            by_agent_id=ctx.agent_id,
            before=None,
            after=_audit_snapshot(new_memory, tag_names=new_tag_names),
            extra_after={"supersedes": str(old.id)},
        )
        await _record_audit(
            s,
            op="supersede",
            memory=old,
            by_agent_id=ctx.agent_id,
            before=old_before,
            after=_audit_snapshot(old, tag_names=old_tag_names),
            extra_after={"superseded_by": str(new_memory.id)},
        )

        # 5. Outbox: new=upsert, old=tombstone (superseded → not visible).
        await enqueue_event(
            s,
            aggregate_type=OutboxAggregateType.memory,
            aggregate_id=new_memory.id,
            aggregate_version=new_memory.version,
            env_id=env_id,
            op=_outbox_op_for(MemoryStatus.active, is_create=True),
            payload=_projection_payload(
                new_memory,
                tag_names=new_tag_names,
                embedding_model_id=embedding_model_id,
            ),
            settings=settings,
        )
        await enqueue_event(
            s,
            aggregate_type=OutboxAggregateType.memory,
            aggregate_id=old.id,
            aggregate_version=old.version,
            env_id=env_id,
            op=_outbox_op_for(MemoryStatus.superseded, is_create=False),
            payload=_projection_payload(
                old,
                tag_names=old_tag_names,
                embedding_model_id=embedding_model_id,
            ),
            settings=settings,
        )

    return (
        _to_response(old, old_tag_names),
        _to_response(new_memory, new_tag_names),
    )
