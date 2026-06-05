"""Compose handler: caller-driven N→1 memory aggregation (v0.15.0 Phase 2).

This module is the entry point for the ``mem_compose`` MCP tool. The
runtime contract is locked in by the Stage B1 design decision (see
``tasks/.../subtasks/.../plan.md`` Stage B). The transaction body lives
in this module so the dream worker handlers (``_accept_merge`` /
``_accept_promotion``) can eventually delegate here once parity tests
prove the refactor is safe.

The atomic transaction follows the rubber-duck #1 step order — dedupe
lookup runs **before** state validation so retries succeed even when
sources have already been superseded by the original call.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from uuid import UUID

from memory_mcp_schemas.compose import (
    ComposeLineageRow,
    ComposeMode,
    ComposeTagPolicy,
    MemComposeRequest,
    MemComposeResponse,
    MemComposeTarget,
)
from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import (
    Memory,
    MemoryLineage,
    MemorySource,
)
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import (
    LineageRelation,
    MemoryKind,
    MemorySourceType,
    MemoryStatus,
    OutboxAggregateType,
)
from memory_mcp.errors import (
    InvalidInputError,
    InvalidTransitionError,
    MemoryMCPError,
    NotFoundError,
    VersionConflictError,
)
from memory_mcp.identity import AgentContext
from memory_mcp.memories import (
    _audit_snapshot,
    _ensure_env_visible,
    _load_env_embedding_model,
    _load_tag_names,
    _lock_memories,
    _outbox_op_for,
    _projection_payload,
    _record_audit,
    _replace_memory_tags,
    _to_response,
    _upsert_tags,
    _validate_decision_meta_for_kind,
)

log = logging.getLogger(__name__)

__all__ = [
    "ComposeLineageRow",
    "ComposeMode",
    "ComposeNotImplementedError",
    "ComposeTagPolicy",
    "MemComposeRequest",
    "MemComposeResponse",
    "MemComposeTarget",
    "memory_compose",
]


class ComposeNotImplementedError(MemoryMCPError):
    """Retained for callers that imported the v0.15.0a stub.

    No longer raised by :func:`memory_compose` once B3d lands; kept so
    out-of-tree imports don't break across the upgrade.
    """

    code = "NOT_IMPLEMENTED"


# ---------------------------------------------------------------------------
# Dedupe-key helper (B3c)
# ---------------------------------------------------------------------------

# Bumped whenever the dedupe-key payload shape changes in a way that
# invalidates prior keys. Keep at 1 for the v0.15.0 release; later changes
# (e.g. adding trigger_description or expires_at to the key) must bump this
# so old + new clients don't collide on the same on-disk key.
_DEDUPE_KEY_SCHEMA_VERSION = 1


def _compute_compose_dedupe_key(
    request: MemComposeRequest,
    *,
    env_id: UUID,
) -> str:
    """Return the deterministic dedupe key for ``request``.

    Two paths:

    * If ``request.idempotency_key`` is set, return it verbatim. The schema
      already caps it at 128 chars; the server treats it as opaque.
    * Otherwise compute ``sha256(canonical_json(payload))[:32]`` where
      ``payload`` is a sorted-keys / no-whitespace JSON object containing
      every input that should disambiguate two composes:

      ``schema_version``, ``operation``, ``env_id``, ``mode``, sorted
      ``source_ids``, and the ``target`` sub-document (``kind``, ``title``,
      ``body``, sorted ``tags``, ``metadata``, ``decision_meta``,
      ``confidence``, ``salience``, ``pinned``).

    Deliberately **excluded** from the key (per rubber-duck B1):

    * ``expected_versions`` — those are an at-call-time precondition,
      not an identity signal. A retry without the version still wants
      to land on the same composed memory.
    * ``trigger_description`` — descriptive only; two composes that
      differ only in trigger description are still "the same" output.
    * ``expires_at`` — TTL is a policy hint, not identity. (Subject to
      revisit if users want different-TTL composes to coexist.)
    * ``tag_policy`` — the *effective* tag set already flows through
      ``target.tags`` after server-side resolution at B3d. Including
      the policy too would double-count.
    """
    if request.idempotency_key is not None:
        return request.idempotency_key

    target = request.target
    payload: dict[str, Any] = {
        "schema_version": _DEDUPE_KEY_SCHEMA_VERSION,
        "operation": "mem_compose",
        "env_id": str(env_id),
        "mode": request.mode,
        "source_ids": sorted(str(sid) for sid in request.source_ids),
        "target": {
            "kind": target.kind.value if hasattr(target.kind, "value") else target.kind,
            "title": target.title,
            "body": target.body,
            "tags": sorted(target.tags) if target.tags else target.tags,
            "metadata": target.metadata,
            "decision_meta": target.decision_meta,
            "confidence": target.confidence,
            "salience": target.salience,
            "pinned": target.pinned,
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def memory_compose(
    request: MemComposeRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemComposeResponse:
    """Compose N≥2 source memories into a single new memory.

    See module docstring + ``MemComposeRequest`` / ``MemComposeResponse``
    for the contract. The function dispatches to a private in-session
    helper so the dream worker can later call the same path without the
    outer ``session_scope``.

    Phase 4 — when ``settings.autowire_enabled`` is true, runs a
    read-only Stage-A candidate fetch *before* opening the transaction
    so the embedder + Qdrant round-trips don't extend the lock-hold
    window. Stage B inserts the resulting edges inside the txn just
    before the response builder.
    """
    settings = settings or get_settings()

    # Stage A — read-only pre-fetch (outside the write transaction).
    # Errors degrade silently to "no candidates"; auto-wire is a
    # best-effort feature and never blocks compose.
    autowire_candidates: list[tuple[UUID, float]] = []
    if settings.autowire_enabled:
        try:
            autowire_candidates = await _autowire_stage_a(
                request=request,
                settings=settings,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("autowire: Stage A failed (%s); skipping", exc)
            autowire_candidates = []

    async with session_scope() as s:
        return await _compose_in_session(
            s,
            request=request,
            ctx=ctx,
            settings=settings,
            autowire_candidates=autowire_candidates,
        )


async def _autowire_stage_a(
    *,
    request: MemComposeRequest,
    settings: Settings,
) -> list[tuple[UUID, float]]:
    """Open a short read-only session and call :func:`autowire_fetch_candidates`.

    Resolves ``env_id`` from the first source memory. Skips when source
    set spans multiple envs (the in-txn validator will reject the
    compose call anyway).
    """
    from memory_mcp.autowire import autowire_fetch_candidates

    async with session_scope() as s:
        rows = (await s.execute(select(Memory.id, Memory.env_id).where(Memory.id.in_(list(request.source_ids))))).all()
        if not rows:
            return []
        envs = {row[1] for row in rows}
        if len(envs) != 1:
            return []
        env_id = next(iter(envs))
        return await autowire_fetch_candidates(
            s=s,
            env_id=env_id,
            source_ids=list(request.source_ids),
            body=request.target.body,
            new_kind=request.target.kind,
            new_tags=request.target.tags,
            settings=settings,
        )


# ---------------------------------------------------------------------------
# IntegrityError classifier (mirrors _is_macro_integrity_error)
# ---------------------------------------------------------------------------


def _is_compose_dedupe_error(exc: IntegrityError) -> bool:
    """True iff ``exc`` is a unique-violation on ``ix_memories_compose_dedupe``.

    Mirrors :func:`memory_mcp.memories._is_macro_integrity_error` — we
    check ``orig.constraint_name``, ``orig.diag.constraint_name``, and
    fall back to a substring match on the rendered exception so the
    classifier is robust across psycopg driver versions.
    """
    orig = getattr(exc, "orig", None)
    constraint = getattr(orig, "constraint_name", None)
    if constraint == "ix_memories_compose_dedupe":
        return True
    diag = getattr(orig, "diag", None)
    diag_constraint = getattr(diag, "constraint_name", None) if diag is not None else None
    if diag_constraint == "ix_memories_compose_dedupe":
        return True
    return "ix_memories_compose_dedupe" in str(exc)


# ---------------------------------------------------------------------------
# Tag-policy resolution
# ---------------------------------------------------------------------------


_DEFAULT_TAG_POLICY: dict[str, ComposeTagPolicy] = {
    "promote": "target",
    "merge": "target_plus_union",
}


def _resolve_tag_policy(request: MemComposeRequest) -> ComposeTagPolicy:
    """Caller override wins; otherwise per-mode default."""
    if request.tag_policy is not None:
        return request.tag_policy
    return _DEFAULT_TAG_POLICY[request.mode]


def _resolve_effective_tags(
    *,
    target_tags: list[str] | None,
    source_tag_names: list[list[str]],
    policy: ComposeTagPolicy,
) -> list[str]:
    """Compute the final tag set per the resolved policy.

    Always returns a deterministic sorted list so concurrent compose
    calls inserting overlapping new tag rows acquire row locks in the
    same order (deadlock safety per rubber-duck #5).

    ``target_tags=None`` is treated as "no explicit target tags"; the
    policy still folds the source-tag contribution in for ``union`` and
    ``target_plus_union``. ``target_tags=[]`` is treated as an explicit
    empty target set (no functional difference here but kept for
    schema-level introspection).
    """
    explicit_target = list(target_tags or [])
    union_sources: set[str] = set()
    for ts in source_tag_names:
        union_sources.update(ts)

    if policy == "target":
        combined = set(explicit_target)
    elif policy == "union":
        combined = set(union_sources)
    else:  # target_plus_union
        combined = set(explicit_target) | union_sources

    return sorted(combined)


# ---------------------------------------------------------------------------
# Target-kind validation
# ---------------------------------------------------------------------------


def _validate_target_kind(target: MemComposeTarget) -> None:
    """Reject kinds that compose's narrow target schema can't honor.

    :class:`MemComposeTarget` deliberately omits ``steps`` and ``macro``
    (a playbook would need both). Surface that as a clean
    :class:`InvalidInputError` up front instead of letting
    ``_normalize_playbook_fields`` raise from inside the txn.
    """
    if target.kind == MemoryKind.playbook:
        raise InvalidInputError(
            "mem_compose does not support kind=playbook (narrow MemComposeTarget has no steps/macro fields)"
        )


# ---------------------------------------------------------------------------
# Lineage reconstruction for idempotent-replay responses
# ---------------------------------------------------------------------------


async def _reconstruct_replay_from_lineage(
    s: AsyncSession,
    *,
    existing: Memory,
    request_mode: ComposeMode,
) -> tuple[list[UUID], list[UUID], ComposeMode, list[ComposeLineageRow]]:
    """Derive the response shape for a dedupe-key hit from canonical lineage.

    On replay we do **not** echo ``request.source_ids`` / ``request.mode``
    — a caller that reused ``idempotency_key`` against a different source
    set would otherwise see a misleading response. Instead we walk
    :class:`MemoryLineage` for ``child_memory_id=existing.id`` and
    reconstruct the authoritative shape:

    * ``source_ids``       — sorted parent ids.
    * ``retired_source_ids`` — parents linked via ``supersedes``.
    * ``mode``             — derived from the lineage relations
      (``supersedes`` → ``merge``; ``promoted_from`` → ``promote``).
    * ``lineage_rows``     — typed rows for the response surface.

    If the caller's ``request.mode`` disagrees with the reconstructed
    mode, we raise :class:`InvalidInputError` rather than lie about what
    the server did — the original op was a different mode and the
    caller is now asking for a semantically different operation that
    happens to share an ``idempotency_key`` value.
    """
    rows = (
        await s.execute(
            select(
                MemoryLineage.parent_memory_id,
                MemoryLineage.relation,
            ).where(MemoryLineage.child_memory_id == existing.id)
        )
    ).all()

    if not rows:  # pragma: no cover — defensive; a composed memory always has lineage
        raise InvalidTransitionError(f"memory {existing.id} has compose_dedupe_key but no lineage rows")

    parent_ids = sorted({row[0] for row in rows})
    supersede_parents = sorted({row[0] for row in rows if row[1] == LineageRelation.supersedes.value})
    promote_parents = sorted({row[0] for row in rows if row[1] == LineageRelation.promoted_from.value})

    if supersede_parents and not promote_parents:
        reconstructed_mode: ComposeMode = "merge"
    elif promote_parents and not supersede_parents:
        reconstructed_mode = "promote"
    else:  # pragma: no cover — mixed-relation compositions aren't possible today
        raise InvalidTransitionError(f"memory {existing.id} has mixed lineage relations; cannot infer mode")

    if reconstructed_mode != request_mode:
        raise InvalidInputError(
            f"idempotency_key matches an existing memory but mode disagrees: "
            f"request.mode={request_mode!r} vs prior={reconstructed_mode!r}"
        )

    relation_for = (
        LineageRelation.supersedes.value if reconstructed_mode == "merge" else LineageRelation.promoted_from.value
    )
    lineage_rows = [
        ComposeLineageRow(
            parent_memory_id=parent_id,
            child_memory_id=existing.id,
            relation=("supersedes" if relation_for == LineageRelation.supersedes.value else "promoted_from"),
        )
        for parent_id in parent_ids
    ]
    retired = supersede_parents if reconstructed_mode == "merge" else []
    return parent_ids, retired, reconstructed_mode, lineage_rows


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------


async def _build_response(
    s: AsyncSession,
    *,
    memory: Memory,
    mode: ComposeMode,
    source_ids: list[UUID],
    retired_source_ids: list[UUID],
    lineage_rows: list[ComposeLineageRow],
    idempotency_replay: bool,
    tag_policy_applied: ComposeTagPolicy,
    dedupe_key: str,
    auto_wired: list[UUID] | None = None,
) -> MemComposeResponse:
    tag_names = await _load_tag_names(s, memory.id)
    return MemComposeResponse(
        memory=_to_response(memory, tag_names),
        mode=mode,
        source_ids=source_ids,
        lineage_rows=lineage_rows,
        retired_source_ids=retired_source_ids,
        auto_wired=list(auto_wired or []),
        idempotency_replay=idempotency_replay,
        tag_policy_applied=tag_policy_applied,
        dedupe_key=dedupe_key,
    )


# ---------------------------------------------------------------------------
# Main transaction body
# ---------------------------------------------------------------------------


async def _compose_in_session(
    s: AsyncSession,
    *,
    request: MemComposeRequest,
    ctx: AgentContext,
    settings: Settings,
    autowire_candidates: list[tuple[UUID, float]] | None = None,
) -> MemComposeResponse:
    """Atomic compose transaction.

    Step order matches B1 lock-in (rubber-duck #1): dedupe lookup runs
    **before** state validation so retries against superseded sources
    still succeed. All mutations live inside the single ``session_scope``
    enclosing this call.
    """

    # Step 1 — Schema validation already ran at the Pydantic boundary.
    # Defensive double-check on cardinality + uniqueness; cheap.
    if len(request.source_ids) < 2:
        raise InvalidInputError("mem_compose requires at least 2 source_ids")
    if len(set(request.source_ids)) != len(request.source_ids):
        raise InvalidInputError("mem_compose source_ids contains duplicates")

    # Step 2 — Lock source rows in sorted-UUID order to avoid deadlocks.
    sorted_ids = sorted(request.source_ids)
    locked = await _lock_memories(s, sorted_ids)
    if len(locked) != len(sorted_ids):
        found = {m.id for m in locked}
        missing = [sid for sid in sorted_ids if sid not in found]
        raise NotFoundError(f"mem_compose: source memories not found: {', '.join(str(m) for m in missing)}")
    by_id = {m.id: m for m in locked}

    # All sources must share env_id (validated below after dedupe lookup).
    env_id = locked[0].env_id

    # Step 3 — Compute dedupe key (deterministic over the request envelope).
    dedupe_key = _compute_compose_dedupe_key(request, env_id=env_id)

    # Step 4 — Dedupe lookup BEFORE state validation (RD #1).
    existing = (
        await s.execute(
            select(Memory)
            .where(
                Memory.env_id == env_id,
                Memory.compose_dedupe_key == dedupe_key,
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        source_ids_out, retired_ids, recon_mode, lineage_rows = await _reconstruct_replay_from_lineage(
            s,
            existing=existing,
            request_mode=request.mode,
        )
        from memory_mcp.autowire import reconstruct_auto_wired

        replay_auto = await reconstruct_auto_wired(s=s, memory_id=existing.id)
        return await _build_response(
            s,
            memory=existing,
            mode=recon_mode,
            source_ids=source_ids_out,
            retired_source_ids=retired_ids,
            lineage_rows=lineage_rows,
            idempotency_replay=True,
            tag_policy_applied=_resolve_tag_policy(request),
            dedupe_key=dedupe_key,
            auto_wired=replay_auto,
        )

    # Step 5 — Validate envelope (state + RBAC + kind invariants).
    for src in locked:
        if src.env_id != env_id:
            raise InvalidInputError("mem_compose: all sources must belong to the same env")
        _ensure_env_visible(src, ctx)

    # Optional caller-asserted env scope.
    if request.env_id is not None and request.env_id != env_id:
        raise InvalidInputError(f"mem_compose: request.env_id={request.env_id} does not match source env {env_id}")

    rbac.require("write", env_id, ctx)

    # Status check — sources must be visible (active or stale).
    _allowed_status = {MemoryStatus.active.value, MemoryStatus.stale.value}
    for src in locked:
        if src.status not in _allowed_status:
            raise InvalidTransitionError(src=str(src.status), dst="composed")

    # Kind invariants.
    _validate_target_kind(request.target)
    if request.mode == "merge":
        source_kinds = {src.kind for src in locked}
        if len(source_kinds) > 1:
            raise InvalidInputError(
                f"mem_compose mode=merge requires all sources to share kind; saw {sorted(source_kinds)}"
            )
        if request.target.kind.value not in source_kinds:
            raise InvalidInputError(
                f"mem_compose mode=merge requires target.kind={request.target.kind.value!r} to match source kind"
            )

    # Step 6 — Optimistic-lock check on supplied expected_versions.
    if request.expected_versions:
        for src_id, expected in request.expected_versions.items():
            actual = by_id[src_id].version
            if actual != expected:
                raise VersionConflictError(
                    expected=expected,
                    actual=actual,
                )

    # Decision-meta validation runs BEFORE the merged-row insert so the
    # txn fails cleanly without an orphan flush.
    decision_meta = await _validate_decision_meta_for_kind(
        kind=request.target.kind.value,
        decision_meta=request.target.decision_meta,
        env_id=env_id,
        session=s,
    )

    # Tag policy resolution + effective tag set (sorted for deadlock safety).
    policy = _resolve_tag_policy(request)
    source_tag_names: list[list[str]] = []
    for src in locked:
        source_tag_names.append(await _load_tag_names(s, src.id))
    effective_tags = _resolve_effective_tags(
        target_tags=request.target.tags,
        source_tag_names=source_tag_names,
        policy=policy,
    )

    # Embedding model for the env (same model across all sources).
    embedding_model_id = await _load_env_embedding_model(s, env_id)

    # Step 7 — Insert merged memory inside a savepoint so a race that lost
    # the unique-index check can be recovered as a replay rather than
    # poisoning the outer transaction.
    target = request.target
    try:
        async with s.begin_nested():
            merged = Memory(
                env_id=env_id,
                kind=target.kind.value,
                status=MemoryStatus.active.value,
                title=target.title,
                body=target.body,
                trigger_description=target.trigger_description,
                metadata_=dict(target.metadata or {}),
                decision_meta=decision_meta,
                pinned=target.pinned,
                expires_at=target.expires_at,
                compose_dedupe_key=dedupe_key,
            )
            if target.salience is not None:
                merged.salience = target.salience
            if target.confidence is not None:
                merged.confidence = target.confidence

            s.add(merged)
            await s.flush()
            await s.refresh(merged)
    except IntegrityError as exc:
        if not _is_compose_dedupe_error(exc):
            raise
        # Strip the failed in-memory instance so the next autoflush
        # doesn't try to re-insert it. ``begin_nested`` rolled the
        # savepoint back but the ORM may still hold the orphan.
        if merged in s:  # pragma: no cover — defensive
            s.expunge(merged)
        async with s.no_autoflush:
            existing = (
                await s.execute(
                    select(Memory)
                    .where(
                        Memory.env_id == env_id,
                        Memory.compose_dedupe_key == dedupe_key,
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
        if existing is None:  # pragma: no cover — race resolution must surface a row
            raise RuntimeError("mem_compose: dedupe-key race recovery found no matching row") from exc
        source_ids_out, retired_ids, recon_mode, lineage_rows = await _reconstruct_replay_from_lineage(
            s,
            existing=existing,
            request_mode=request.mode,
        )
        from memory_mcp.autowire import reconstruct_auto_wired

        race_auto = await reconstruct_auto_wired(s=s, memory_id=existing.id)
        return await _build_response(
            s,
            memory=existing,
            mode=recon_mode,
            source_ids=source_ids_out,
            retired_source_ids=retired_ids,
            lineage_rows=lineage_rows,
            idempotency_replay=True,
            tag_policy_applied=policy,
            dedupe_key=dedupe_key,
            auto_wired=race_auto,
        )

    # Step 8 — Apply tag policy (effective_tags is already sorted).
    if effective_tags:
        tag_map = await _upsert_tags(s, env_id=env_id, names=effective_tags)
        await _replace_memory_tags(
            s,
            memory_id=merged.id,
            env_id=env_id,
            tag_ids=[tag_map[n] for n in effective_tags],
        )

    # Step 9 — Provenance (agent-driven; source_ref carries dedupe-key prefix).
    await s.execute(
        insert(MemorySource).values(
            memory_id=merged.id,
            source_type=MemorySourceType.agent.value,
            source_ref=dedupe_key[:8],
            agent_id=ctx.agent_id,
        )
    )

    # Step 10 — Insert N lineage rows. Trigger from migration 0017 handles
    # parent counter bumps for whitelisted relations (promoted_from); the
    # supersedes relation is intentionally not whitelisted (RD #4 in B5).
    relation_value = (
        LineageRelation.supersedes.value if request.mode == "merge" else LineageRelation.promoted_from.value
    )
    lineage_rows: list[ComposeLineageRow] = []
    for src in locked:
        await s.execute(
            insert(MemoryLineage).values(
                parent_memory_id=src.id,
                child_memory_id=merged.id,
                relation=relation_value,
            )
        )
        lineage_rows.append(
            ComposeLineageRow(
                parent_memory_id=src.id,
                child_memory_id=merged.id,
                relation=("supersedes" if request.mode == "merge" else "promoted_from"),
            )
        )

    # Step 11 — mode='merge': transition sources to superseded.
    retired_source_ids: list[UUID] = []
    if request.mode == "merge":
        for src in locked:
            old_tag_names = source_tag_names[locked.index(src)]
            old_before = _audit_snapshot(src, tag_names=old_tag_names)
            result = await s.execute(
                update(Memory)
                .where(
                    Memory.id == src.id,
                    Memory.version == src.version,
                )
                .values(
                    status=MemoryStatus.superseded.value,
                    superseded_by=merged.id,
                    version=src.version + 1,
                    updated_at=func.now(),
                )
            )
            if result.rowcount == 0:  # type: ignore[attr-defined]
                raise VersionConflictError(
                    expected=src.version,
                    actual=src.version + 1,
                )
            await s.refresh(src)
            await _record_audit(
                s,
                op="supersede",
                memory=src,
                by_agent_id=ctx.agent_id,
                before=old_before,
                after=_audit_snapshot(src, tag_names=old_tag_names),
                extra_after={"superseded_by": str(merged.id)},
            )
            await enqueue_event(
                s,
                aggregate_type=OutboxAggregateType.memory,
                aggregate_id=src.id,
                aggregate_version=src.version,
                env_id=env_id,
                op=_outbox_op_for(MemoryStatus.superseded, is_create=False),
                payload=_projection_payload(
                    src,
                    tag_names=old_tag_names,
                    embedding_model_id=embedding_model_id,
                ),
                settings=settings,
            )
            retired_source_ids.append(src.id)

    # Step 12 — Two audit rows for merged: a baseline ``create`` row that
    # mirrors mem_write parity, plus an aggregate ``mem_compose:{mode}``
    # row that captures the source set, applied tag policy, and dedupe
    # key for cross-reference / analytics (filterable on op LIKE
    # 'mem_compose:%').
    await _record_audit(
        s,
        op="create",
        memory=merged,
        by_agent_id=ctx.agent_id,
        before=None,
        after=_audit_snapshot(merged, tag_names=effective_tags),
        extra_after={
            "compose_mode": request.mode,
            "compose_sources": [str(src.id) for src in locked],
        },
    )
    await _record_audit(
        s,
        op=f"mem_compose:{request.mode}",
        memory=merged,
        by_agent_id=ctx.agent_id,
        before=None,
        after=_audit_snapshot(merged, tag_names=effective_tags),
        extra_after={
            "source_ids": [str(src.id) for src in locked],
            "tag_policy_applied": policy,
            "dedupe_key": dedupe_key,
            "retired_source_ids": [str(rid) for rid in retired_source_ids],
        },
    )

    # Step 13 — One outbox event for the new merged row (Qdrant upsert +
    # Neo4j node create). Lineage rows do not enqueue outbox events
    # (matches dream-handler invariant; lineage stays Postgres-only).
    await enqueue_event(
        s,
        aggregate_type=OutboxAggregateType.memory,
        aggregate_id=merged.id,
        aggregate_version=merged.version,
        env_id=env_id,
        op=_outbox_op_for(MemoryStatus.active, is_create=True),
        payload=_projection_payload(
            merged,
            tag_names=effective_tags,
            embedding_model_id=embedding_model_id,
        ),
        settings=settings,
    )

    # Step 13.5 — Phase 4 auto-wire (OFF by default). Inserts up to
    # ``settings.autowire_top_k`` ``related_to_popular`` edges from the
    # new memory to its top semantic neighbours. Candidates were
    # pre-computed Stage-A in ``memory_compose`` to keep embedder /
    # Qdrant round-trips outside the lock-hold window. Errors degrade
    # silently — auto-wire never blocks compose.
    auto_wired_ids: list[UUID] = []
    if settings.autowire_enabled and autowire_candidates:
        from memory_mcp.autowire import autowire_compose_target

        try:
            auto_wired_ids = await autowire_compose_target(
                s=s,
                new_memory_id=merged.id,
                new_memory_kind=request.target.kind,
                new_memory_tags=effective_tags,
                new_memory_body=request.target.body,
                new_memory_env_id=env_id,
                candidates=autowire_candidates,
                ctx=ctx,
                settings=settings,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "autowire: Stage B failed for memory %s (%s); compose unaffected",
                merged.id,
                exc,
            )
            auto_wired_ids = []

    # Step 14 — Build the response. session_scope() commits on exit.
    return await _build_response(
        s,
        memory=merged,
        mode=request.mode,
        source_ids=sorted_ids,
        retired_source_ids=retired_source_ids,
        lineage_rows=lineage_rows,
        idempotency_replay=False,
        tag_policy_applied=policy,
        dedupe_key=dedupe_key,
        auto_wired=auto_wired_ids,
    )
