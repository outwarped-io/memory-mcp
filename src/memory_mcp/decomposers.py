"""Decompose handler: caller-driven 1→N memory decomposition (v0.15.0 Phase 3).

This module is the entry point for the ``mem_decompose`` MCP tool. The
runtime contract is locked in by the Stage C1 design decision (see
``tasks/.../subtasks/.../plan.md`` Stage C). The transaction body lives
in this module so the dream worker can eventually delegate
decompositions through here once the dream-side decompose proposal
lands.

Structure mirrors :mod:`memory_mcp.composers` deliberately. The two
modules share:

* The :func:`_compute_decompose_dedupe_key` / fingerprint pair (mirrors
  ``_compute_compose_dedupe_key``).
* The savepoint+expunge+refetch race-loss replay pattern (mirrors the
  compose dedupe-key recovery branch).
* The same in-session helpers from :mod:`memory_mcp.memories`
  (``_lock_memories``, ``_ensure_env_visible``, ``_record_audit``,
  ``_upsert_tags``, etc.) so the audit and outbox shapes stay
  consistent between aggregation and decomposition ops.

Provenance convention (RD G note + C8 yellow #4): each decomposed
child is recorded with
``MemorySource(source_type='agent', source_ref='decompose:<operation_id>')``
where ``<operation_id>`` is the ``decompose_operations`` row UUID. The
``decompose:`` prefix namespaces the ref so a downstream reader does
not mistake the UUID-shaped string for a memory id; the audit log
carries the ``op='mem_decompose:{mode}'`` distinction for cross-
reference; the operation row itself is the canonical record of
"these N children came from this 1 source via this mode".

Race-loss replay (C8 RD red flag #3): the dedupe arbiter is a
**unique index on decompose_operations(env_id, dedupe_key)**, and the
INSERT into that table runs INSIDE a ``begin_nested()`` savepoint so a
concurrent decompose that wins the race causes the loser's savepoint
to roll back cleanly. The loser then re-queries the operation row,
RBAC-checks env visibility, and returns the winner's children as a
replay response.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import (
    DecomposeOperation,
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
from memory_mcp_schemas.decompose import (
    DecomposeLineageRow,
    DecomposeMode,
    MemDecomposeChild,
    MemDecomposeRequest,
    MemDecomposeResponse,
)


log = logging.getLogger(__name__)

__all__ = [
    "DecomposeLineageRow",
    "DecomposeMode",
    "DecomposeNotImplementedError",
    "MemDecomposeChild",
    "MemDecomposeRequest",
    "MemDecomposeResponse",
    "memory_decompose",
]


class DecomposeNotImplementedError(MemoryMCPError):
    """C3 stub — raised until the C6 transaction body lands."""

    code = "NOT_IMPLEMENTED"


# ---------------------------------------------------------------------------
# Dedupe-key / fingerprint helpers (C4)
# ---------------------------------------------------------------------------

# Bumped whenever the dedupe-key payload shape changes in a way that
# invalidates prior keys. Keep at 1 for the v0.15.0 release; later changes
# (e.g. adding trigger_description or expires_at to the key) must bump this
# so old + new clients don't collide on the same on-disk key.
_DEDUPE_KEY_SCHEMA_VERSION = 1


def _canonical_child_payload(child: MemDecomposeChild) -> dict[str, Any]:
    """Identity-bearing fields of one child, sorted-key friendly.

    Excluded from the dedupe-key canonical form (see
    :func:`_compute_decompose_dedupe_key`): ``trigger_description`` and
    ``expires_at``. Those exclusions are deliberate (descriptive / policy
    hints, not identity signals); the request-fingerprint canonical form
    re-adds them via :func:`_canonical_child_payload_full`.
    """
    return {
        "kind": child.kind.value if hasattr(child.kind, "value") else child.kind,
        "title": child.title,
        "body": child.body,
        "tags": sorted(child.tags) if child.tags else child.tags,
        "metadata": child.metadata,
        "decision_meta": child.decision_meta,
        "confidence": child.confidence,
        "salience": child.salience,
        "pinned": child.pinned,
    }


def _hash_canonical_child(child: MemDecomposeChild) -> str:
    """Stable per-child sha256-hex used to sort children for dedupe-key."""
    canonical = json.dumps(
        _canonical_child_payload(child),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_decompose_dedupe_key(
    request: MemDecomposeRequest,
    *,
    env_id: UUID,
) -> str:
    """Return the deterministic dedupe key for ``request``.

    Two paths:

    * If ``request.idempotency_key`` is set, return it verbatim. The
      schema already caps it at 128 chars; the server treats it as
      opaque.
    * Otherwise compute ``sha256(canonical_json(payload))[:32]`` where
      ``payload`` is a sorted-keys / no-whitespace JSON object containing
      every input that should disambiguate two decomposes:

      ``schema_version``, ``operation``, ``env_id``, ``mode``,
      ``source_id``, and the ``children`` list — sorted by each child's
      canonical-JSON hash so re-ordering identical children does not
      produce a different key (per C1.2 lock-in).

    Deliberately **excluded** from the key (per rubber-duck C1.5):

    * ``expected_version`` — an at-call-time precondition, not an
      identity signal. A retry without the version should still land on
      the same decomposed children.
    * per-child ``trigger_description`` — descriptive only; two
      decomposes that differ only in trigger description are still "the
      same" output.
    * per-child ``expires_at`` — TTL is a policy hint, not identity.
    * ``idempotency_key`` itself when it took the override path
      (returned verbatim above).
    """
    if request.idempotency_key is not None:
        return request.idempotency_key

    children_canonical = sorted(
        (_canonical_child_payload(c) for c in request.children),
        key=lambda c: hashlib.sha256(
            json.dumps(c, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest(),
    )
    payload: dict[str, Any] = {
        "schema_version": _DEDUPE_KEY_SCHEMA_VERSION,
        "operation": "mem_decompose",
        "env_id": str(env_id),
        "mode": request.mode,
        "source_id": str(request.source_id),
        "children": children_canonical,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _compute_request_fingerprint(
    request: MemDecomposeRequest,
    *,
    env_id: UUID,
) -> str:
    """Always-canonical sha256 of the request scope (32 hex).

    Used to detect ``idempotency_key`` reuse with a different operation
    scope. The fingerprint is stored on
    ``decompose_operations.request_fingerprint``; the C6 transaction
    body looks up the operation row by dedupe-key, then compares the
    incoming fingerprint to the stored row. On mismatch:
    ``InvalidInputError("idempotency_key reused with different scope")``.

    Distinct from :func:`_compute_decompose_dedupe_key` in exactly one
    way: the fingerprint **always** sha256-computes, even when
    ``idempotency_key`` is set on the request. The dedupe-key returns
    the caller-supplied key verbatim in that case; the fingerprint
    canonicalises the scope so the two values together support
    "caller-supplied lookup token + server-verified scope hash".

    The canonical payload is **structurally identical** to the
    dedupe-key payload (sorted children by canonical hash, no
    ``expected_version``, no per-child ``trigger_description`` /
    ``expires_at``, no ``idempotency_key``) so that retry-insensitive
    changes do not trigger a false-positive scope mismatch:

    * A retry that omits ``expected_version`` (or supplies a different
      value) hashes the same — ``expected_version`` is a precondition,
      not part of operation identity.
    * A retry that re-orders identical children hashes the same —
      children are sorted by canonical hash before hashing.
    * A retry that changes ``trigger_description`` or ``expires_at``
      on a child hashes the same — those are descriptive only.

    The fingerprint differs from the dedupe-key hash by the
    ``operation`` domain-separator only (``"mem_decompose_fp"`` vs
    ``"mem_decompose"``); both 32-hex sha256 prefixes.
    """
    payload: dict[str, Any] = {
        "schema_version": _DEDUPE_KEY_SCHEMA_VERSION,
        "operation": "mem_decompose_fp",
        "env_id": str(env_id),
        "mode": request.mode,
        "source_id": str(request.source_id),
        "children": sorted(
            (_canonical_child_payload(c) for c in request.children),
            key=lambda d: hashlib.sha256(
                json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest(),
        ),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# IntegrityError classifier (mirrors _is_compose_dedupe_error)
# ---------------------------------------------------------------------------


def _is_decompose_dedupe_error(exc: IntegrityError) -> bool:
    """True iff ``exc`` is a unique-violation on ``ix_decompose_operations_dedupe``.

    Mirrors :func:`memory_mcp.composers._is_compose_dedupe_error`: check
    ``exc.orig.constraint_name``, ``exc.orig.diag.constraint_name``, and
    fall back to a substring match on the rendered exception so the
    classifier is robust across psycopg driver versions.
    """
    orig = getattr(exc, "orig", None)
    constraint = getattr(orig, "constraint_name", None)
    if constraint == "ix_decompose_operations_dedupe":
        return True
    diag = getattr(orig, "diag", None)
    diag_constraint = getattr(diag, "constraint_name", None) if diag is not None else None
    if diag_constraint == "ix_decompose_operations_dedupe":
        return True
    return "ix_decompose_operations_dedupe" in str(exc)


# ---------------------------------------------------------------------------
# Validators (C5)
# ---------------------------------------------------------------------------


# Source statuses that can be legally decomposed. Mirrors compose's
# acceptable-source set; ``active`` and ``stale`` only. ``proposed`` is
# excluded because pre-acceptance content shouldn't be split/derived (the
# accept/reject path handles those). ``archived`` / ``retired`` /
# ``superseded`` are excluded because they're already out of the active
# graph — decomposing them produces orphan lineage.
_VALID_SOURCE_STATUSES_FOR_DECOMPOSE: frozenset[MemoryStatus] = frozenset(
    {MemoryStatus.active, MemoryStatus.stale}
)


def _validate_children(request: MemDecomposeRequest) -> None:
    """Pre-lock envelope validation on the children list.

    Runs before the source is locked so cheap caller errors fail fast
    without holding a row lock. The Pydantic schema already enforces:

    * cardinality ``2 ≤ len(children) ≤ 20``
    * ``kind != playbook`` (per-field validator)
    * ``idempotency_key`` length ≤ 128
    * each child's ``body`` non-empty, ``title`` ≤ 400 chars
    * each child's ``salience`` and ``confidence`` ∈ ``[0.0, 1.0]``

    This function adds the cross-child / cross-field checks the schema
    can't express:

    * **No duplicate children** (per C1.2 B.3) — two children with the
      same canonical-JSON content (kind, title, body, tags, metadata,
      decision_meta, confidence, salience, pinned) would land as
      indistinguishable rows after the transaction; reject before lock.
      Comparison uses :func:`_canonical_child_payload` so the rule
      matches what the dedupe-key already considers identity-bearing.
    * **``decision_meta`` only valid on ``kind=decision`` children** —
      a fact / observation / procedure / etc. carrying decision_meta is
      malformed (the field has no consumer). Schema can't gate this
      because decision_meta is on every child kind for compatibility
      with ``MemoryWriteRequest``; gate it here.

    Children carrying ``decision_meta`` on ``kind=decision`` are
    accepted at this stage; the deep validation
    (``validate_decision_meta`` against env policy) runs in
    ``_decompose_in_session`` once the session is available (mirrors
    :func:`memory_mcp.memories._validate_decision_meta_for_kind`).
    """
    seen_hashes: set[str] = set()
    for idx, child in enumerate(request.children):
        if child.decision_meta is not None and child.kind != MemoryKind.decision:
            raise InvalidInputError(
                f"decision_meta only valid for kind=decision "
                f"(children[{idx}].kind={child.kind.value!r})"
            )
        canonical_hash = _hash_canonical_child(child)
        if canonical_hash in seen_hashes:
            raise InvalidInputError(
                f"duplicate child content at children[{idx}] "
                f"(canonical hash matches an earlier child)"
            )
        seen_hashes.add(canonical_hash)


def _validate_source(
    source: Memory,
    request: MemDecomposeRequest,
    ctx: AgentContext,
    *,
    is_replay: bool,
) -> None:
    """Post-lock validation on the source memory.

    Called **after** ``_lock_memories([source_id])`` and **after** the
    operation-table lookup has decided whether this call is a first-time
    write or a replay. Lifecycle and version checks are skipped on
    replay so a caller that successfully decomposed an active source
    yesterday can still replay today even after the source has been
    retired by another path (per C1.5 RD A.2):

    * **Env visibility check** — enforced on BOTH paths. Even on
      replay, the caller must be able to see the source in one of their
      attached envs; otherwise an external observer with no env grant
      could fish for operation-row ids by retrying with arbitrary
      ``idempotency_key`` values. Uses ``_ensure_env_visible``-style
      logic inline so this module doesn't import memories.py just for
      one helper.
    * **Source kind ≠ playbook** — enforced on BOTH paths. Playbook
      sources have a ``steps`` field that decompose children can't
      carry; allowing them would silently drop ``steps`` on every
      derive/split.
    * **Source status ∈ {active, stale}** — enforced ONLY on first
      write. On replay the source may now be retired / archived /
      superseded; the replay correctly returns the children that were
      created during the original transaction.
    * **``expected_version`` match** — enforced ONLY on first write
      (same reasoning: the precondition only gates the mutating call).

    The ``is_replay`` flag is passed in from the transaction body so
    the lifecycle/version checks can be uniformly skipped. The visibility
    and kind checks always run — those are *identity* invariants, not
    *transitional* ones.
    """
    if ctx.attached_env_ids and source.env_id not in ctx.attached_env_ids:
        raise NotFoundError(
            f"memory {source.id} not visible in attached envs",
            memory_id=str(source.id),
        )

    if source.kind == MemoryKind.playbook.value:
        raise InvalidInputError(
            f"cannot decompose a playbook source (source={source.id}); "
            "playbook memories carry a ``steps`` field that "
            "MemDecomposeChild does not expose"
        )

    if is_replay:
        return

    source_status = MemoryStatus(source.status)
    if source_status not in _VALID_SOURCE_STATUSES_FOR_DECOMPOSE:
        raise InvalidTransitionError(
            src=source_status.value,
            dst="decomposed",
        )

    if request.expected_version is not None and request.expected_version != source.version:
        raise VersionConflictError(
            expected=request.expected_version,
            actual=source.version,
        )


# ---------------------------------------------------------------------------
# Lineage relation per mode
# ---------------------------------------------------------------------------


_RELATION_FOR_MODE: dict[str, str] = {
    "split": LineageRelation.split_from.value,
    "derive": LineageRelation.derived_from.value,
}


# ---------------------------------------------------------------------------
# Replay reconstruction
# ---------------------------------------------------------------------------


async def _reconstruct_replay_from_operation(
    s: AsyncSession,
    *,
    operation_row: DecomposeOperation,
    ctx: AgentContext,
) -> tuple[Memory, list[Memory], list[DecomposeLineageRow]]:
    """Re-materialize the response payload for a dedupe-key hit.

    Walks ``decompose_operations.child_ids`` (preserves request order)
    to fetch the children and the source. Enforces env visibility on
    every row — replay must survive source lifecycle changes but must
    NOT bypass RBAC (C1.5 RD A.2): an external observer with no env
    grant should not be able to fish for operation-row contents by
    retrying with arbitrary ``idempotency_key`` values.
    """
    source = await s.get(Memory, operation_row.source_id)
    if source is None:  # pragma: no cover — FK RESTRICT prevents this
        raise NotFoundError(
            f"mem_decompose replay: source memory {operation_row.source_id} "
            "missing despite operation row referencing it",
            memory_id=str(operation_row.source_id),
        )
    _ensure_env_visible(source, ctx)

    child_ids = list(operation_row.child_ids)
    if not child_ids:  # pragma: no cover — schema CHECK forbids empty arrays
        raise InvalidTransitionError(
            src="empty_child_ids",
            dst="decompose_replay",
        )

    child_rows = (
        await s.execute(select(Memory).where(Memory.id.in_(child_ids)))
    ).scalars().all()
    by_id = {m.id: m for m in child_rows}
    missing = [cid for cid in child_ids if cid not in by_id]
    if missing:  # pragma: no cover — children FK CASCADE keyed off envs only
        raise NotFoundError(
            f"mem_decompose replay: child memories not found: "
            f"{', '.join(str(m) for m in missing)}",
        )
    children = [by_id[cid] for cid in child_ids]
    for child in children:
        _ensure_env_visible(child, ctx)

    relation_value = _RELATION_FOR_MODE[operation_row.mode]
    lineage_rows = [
        DecomposeLineageRow(
            parent_memory_id=source.id,
            child_memory_id=child.id,
            relation=relation_value,  # type: ignore[arg-type]
        )
        for child in children
    ]
    return source, children, lineage_rows


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------


async def _build_response(
    s: AsyncSession,
    *,
    source: Memory,
    children: list[Memory],
    mode: DecomposeMode,
    lineage_rows: list[DecomposeLineageRow],
    operation_id: UUID,
    dedupe_key: str,
    idempotency_replay: bool,
) -> MemDecomposeResponse:
    """Materialize the MCP response.

    Children are returned in the order recorded by
    ``decompose_operations.child_ids`` — which is the order they
    appeared in the original request (the transaction body inserts
    them in request order without canonical-hash re-sorting; see C8
    RF#2 resolution in plan.md).
    """
    source_tags = await _load_tag_names(s, source.id)
    children_resp = []
    for child in children:
        child_tags = await _load_tag_names(s, child.id)
        children_resp.append(_to_response(child, child_tags))
    return MemDecomposeResponse(
        source=_to_response(source, source_tags),
        children=children_resp,
        mode=mode,
        lineage_rows=lineage_rows,
        auto_wired=[],
        idempotency_replay=idempotency_replay,
        dedupe_key=dedupe_key,
        operation_id=operation_id,
    )


# ---------------------------------------------------------------------------
# Main transaction body
# ---------------------------------------------------------------------------


async def _decompose_in_session(
    s: AsyncSession,
    *,
    request: MemDecomposeRequest,
    ctx: AgentContext,
    settings: Settings,
) -> MemDecomposeResponse:
    """Atomic decompose transaction.

    Step order mirrors :func:`memory_mcp.composers._compose_in_session`
    with the polarity flipped (1 source → N children instead of N
    sources → 1 child). The locked-in design (C1.5 + C8 rubber-duck)
    drives the exact step ordering:

    1.  ``_validate_children`` — cheap envelope (cardinality already
        gated at the Pydantic boundary; this catches cross-child
        invariants: duplicate canonical hashes + ``decision_meta`` on
        non-``decision`` children).
    2.  ``_lock_memories([source_id])`` — single ``SELECT … FOR UPDATE``.
    3.  Resolve ``env_id`` from the locked row.
    4.  Compute ``dedupe_key`` + always-canonical ``request_fingerprint``.
    5.  Dedupe-table lookup BEFORE source state validation (C1.5 RD
        A.2). Three outcomes:
          * Hit + fingerprint match → return replay response (no
            mutation; RBAC still enforced by reconstruction helper).
          * Hit + fingerprint mismatch → ``InvalidInputError``
            (``idempotency_key`` reused with different scope).
          * Miss → continue.
    6.  ``_validate_source`` (env visibility + kind ≠ playbook always;
        status ∈ {active, stale} + ``expected_version`` only on first
        write).
    7.  Validate per-child ``decision_meta`` against env policy
        (deep check, async).
    8.  Pre-allocate child UUIDs + operation id (Python-side
        ``uuid4``) so the operation row carries the full
        ``child_ids`` array.
    9.  Resolve embedding-model id for the env (one round-trip).
    10. **INSIDE ``s.begin_nested()`` savepoint** (C8 RD RF#3) —
        attempt the operation-table INSERT. The
        ``(env_id, dedupe_key)`` unique index is the race arbiter;
        if a concurrent caller already claimed this key we catch
        :class:`IntegrityError`, expunge any transient ORM state,
        re-query the operation row inside ``no_autoflush``, and
        replay.
    11. Insert N children via ``add_all`` (G.13).
    12. Per-child tag normalization (one ``_upsert_tags`` per child
        for now; N ≤ 20 keeps this O(N) acceptable).
    13. Per-child provenance row — ``source_type='agent'``,
        ``source_ref='decompose:<operation_id>'`` (C8 yellow #4).
    14. Per-child lineage row (``split_from`` or ``derived_from``).
        Trigger from migration 0021 bumps
        ``source.reference_count_lineage`` for ``derived_from``
        only — ``split_from`` is excluded from the whitelist by
        design (E.11) since the source is about to retire.
    15. If ``mode='split'``: single coalesced UPDATE on source
        (status, version, retired_at), version-guarded; refresh.
        Status-flip trigger decrements counters for source's
        pre-existing whitelisted outgoing lineage — that is the
        documented behavior (RD yellow #6).
    16. Audit rows: per-child ``op='create'`` + source-side
        ``op='mem_decompose:{mode}'`` (+ ``op='retire'`` if split).
    17. Outbox events: one ``upsert`` per child; one ``tombstone``
        for the source if split; lineage and operation-table rows
        do NOT enqueue outbox events (lineage stays Postgres-only,
        matching compose / dream invariants).
    18. Build response.
    """

    # Step 1 — cheap envelope validation.
    _validate_children(request)

    # Step 2 — lock the single source row.
    locked = await _lock_memories(s, [request.source_id])
    if not locked:
        raise NotFoundError(
            f"mem_decompose: source memory {request.source_id} not found",
            memory_id=str(request.source_id),
        )
    source = locked[0]
    env_id = source.env_id

    # Step 3/4 — dedupe-key + always-canonical fingerprint.
    dedupe_key = _compute_decompose_dedupe_key(request, env_id=env_id)
    fingerprint = _compute_request_fingerprint(request, env_id=env_id)

    # Step 5 — dedupe lookup BEFORE state validation (RD A.2).
    existing_op = (await s.execute(
        select(DecomposeOperation).where(
            DecomposeOperation.env_id == env_id,
            DecomposeOperation.dedupe_key == dedupe_key,
        ).limit(1)
    )).scalar_one_or_none()
    if existing_op is not None:
        if existing_op.request_fingerprint != fingerprint:
            raise InvalidInputError(
                "idempotency_key reused with different scope "
                "(same dedupe_key, different request fingerprint); "
                "the prior decompose targeted a different source / mode / "
                "child set"
            )
        # Replay path. RBAC + env visibility enforced inside reconstructor.
        _validate_source(source, request, ctx, is_replay=True)
        source_replay, children_replay, lineage_replay = (
            await _reconstruct_replay_from_operation(
                s, operation_row=existing_op, ctx=ctx,
            )
        )
        return await _build_response(
            s,
            source=source_replay,
            children=children_replay,
            mode=existing_op.mode,  # type: ignore[arg-type]
            lineage_rows=lineage_replay,
            operation_id=existing_op.id,
            dedupe_key=dedupe_key,
            idempotency_replay=True,
        )

    # Step 6 — first-write source validation.
    _validate_source(source, request, ctx, is_replay=False)
    rbac.require("write", env_id, ctx)

    # Step 7 — per-child decision_meta deep validation (only for
    # children that carry it; _validate_children already gated presence
    # to kind=decision).
    validated_decision_meta: list[dict[str, Any] | None] = []
    for child in request.children:
        if child.decision_meta is None:
            validated_decision_meta.append(None)
        else:
            validated_decision_meta.append(
                await _validate_decision_meta_for_kind(
                    kind=child.kind.value,
                    decision_meta=child.decision_meta,
                    env_id=env_id,
                    session=s,
                )
            )

    # Step 8 — pre-allocate UUIDs so the op-row carries child_ids.
    operation_id = uuid4()
    child_uuids = [uuid4() for _ in request.children]

    # Step 9 — resolve embedding model id once.
    embedding_model_id = await _load_env_embedding_model(s, env_id)

    # Step 10 — attempt to claim the dedupe slot. Savepoint scopes the
    # race: only the op-row insert can fail on the unique-index, and
    # only that single failed INSERT rolls back. The children inserts
    # below run AFTER the savepoint commits, so a race-loss cannot
    # leave orphan children.
    try:
        async with s.begin_nested():
            await s.execute(
                insert(DecomposeOperation).values(
                    id=operation_id,
                    env_id=env_id,
                    source_id=source.id,
                    mode=request.mode,
                    dedupe_key=dedupe_key,
                    request_fingerprint=fingerprint,
                    child_ids=child_uuids,
                    created_by_agent_id=ctx.agent_id,
                )
            )
    except IntegrityError as exc:
        if not _is_decompose_dedupe_error(exc):
            raise
        async with s.no_autoflush:
            winner = (await s.execute(
                select(DecomposeOperation).where(
                    DecomposeOperation.env_id == env_id,
                    DecomposeOperation.dedupe_key == dedupe_key,
                ).limit(1)
            )).scalar_one_or_none()
        if winner is None:  # pragma: no cover — race recovery must surface a row
            raise RuntimeError(
                "mem_decompose: dedupe-key race recovery found no "
                "matching operation row"
            ) from exc
        if winner.request_fingerprint != fingerprint:
            # The winner had a different scope under the same caller
            # key — treat the same as the pre-lookup mismatch.
            raise InvalidInputError(
                "idempotency_key reused with different scope "
                "(detected after race-loss to a concurrent decompose with "
                "different request fingerprint)"
            ) from exc
        source_replay, children_replay, lineage_replay = (
            await _reconstruct_replay_from_operation(
                s, operation_row=winner, ctx=ctx,
            )
        )
        return await _build_response(
            s,
            source=source_replay,
            children=children_replay,
            mode=winner.mode,  # type: ignore[arg-type]
            lineage_rows=lineage_replay,
            operation_id=winner.id,
            dedupe_key=dedupe_key,
            idempotency_replay=True,
        )

    # We claimed the dedupe slot. From here on, every mutation is in
    # the outer txn and commits together with the op-row above.

    # Step 11 — build + insert children in request order.
    children: list[Memory] = []
    for idx, child_payload in enumerate(request.children):
        child_metadata = dict(child_payload.metadata or {})
        child_obj = Memory(
            id=child_uuids[idx],
            env_id=env_id,
            kind=child_payload.kind.value,
            status=MemoryStatus.active.value,
            title=child_payload.title,
            body=child_payload.body,
            trigger_description=child_payload.trigger_description,
            metadata_=child_metadata,
            decision_meta=validated_decision_meta[idx],
            pinned=child_payload.pinned,
            expires_at=child_payload.expires_at,
        )
        if child_payload.salience is not None:
            child_obj.salience = child_payload.salience
        if child_payload.confidence is not None:
            child_obj.confidence = child_payload.confidence
        children.append(child_obj)

    s.add_all(children)
    await s.flush()
    for child in children:
        await s.refresh(child)

    # Step 12 — per-child tags.
    child_tag_names: list[list[str]] = []
    for idx, child in enumerate(children):
        names = sorted(set(request.children[idx].tags or []))
        child_tag_names.append(names)
        if names:
            tag_map = await _upsert_tags(s, env_id=env_id, names=names)
            await _replace_memory_tags(
                s,
                memory_id=child.id,
                env_id=env_id,
                tag_ids=[tag_map[n] for n in names],
            )

    # Step 13 — provenance per child. source_ref is namespaced with the
    # ``decompose:`` prefix so a downstream reader does not confuse the
    # UUID-shaped string for a memory id (C8 yellow #4).
    provenance_ref = f"decompose:{operation_id}"
    for child in children:
        await s.execute(
            insert(MemorySource).values(
                memory_id=child.id,
                source_type=MemorySourceType.agent.value,
                source_ref=provenance_ref,
                agent_id=ctx.agent_id,
            )
        )

    # Step 14 — lineage rows. Trigger from migration 0021 bumps
    # ``source.reference_count_lineage`` ONLY for ``derived_from``
    # (``split_from`` is excluded from the popularity whitelist by
    # design — E.11).
    relation_value = _RELATION_FOR_MODE[request.mode]
    lineage_rows: list[DecomposeLineageRow] = []
    for child in children:
        await s.execute(
            insert(MemoryLineage).values(
                parent_memory_id=source.id,
                child_memory_id=child.id,
                relation=relation_value,
            )
        )
        lineage_rows.append(
            DecomposeLineageRow(
                parent_memory_id=source.id,
                child_memory_id=child.id,
                relation=relation_value,  # type: ignore[arg-type]
            )
        )

    # Capture pre-retire source snapshot for audit consistency.
    source_tag_names = await _load_tag_names(s, source.id)
    source_before = _audit_snapshot(source, tag_names=source_tag_names)

    # Step 15 — split mode retires the source in a single coalesced UPDATE.
    if request.mode == "split":
        result = await s.execute(
            update(Memory)
            .where(
                Memory.id == source.id,
                Memory.version == source.version,
            )
            .values(
                status=MemoryStatus.retired.value,
                version=source.version + 1,
                retired_at=func.now(),
                updated_at=func.now(),
            )
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]
            raise VersionConflictError(
                expected=source.version,
                actual=source.version + 1,
            )
        await s.refresh(source)

    # Step 16 — audit rows.
    # 16a — per-child create rows (mirror mem_write parity).
    for idx, child in enumerate(children):
        await _record_audit(
            s,
            op="create",
            memory=child,
            by_agent_id=ctx.agent_id,
            before=None,
            after=_audit_snapshot(child, tag_names=child_tag_names[idx]),
            extra_after={
                "decompose_mode": request.mode,
                "decompose_source": str(source.id),
                "decompose_operation_id": str(operation_id),
            },
        )
    # 16b — aggregate mem_decompose row on the source. Mirrors compose's
    # ``mem_compose:{mode}`` row; filterable on ``op LIKE 'mem_decompose:%'``.
    await _record_audit(
        s,
        op=f"mem_decompose:{request.mode}",
        memory=source,
        by_agent_id=ctx.agent_id,
        before=source_before,
        after=_audit_snapshot(source, tag_names=source_tag_names),
        extra_after={
            "child_ids": [str(c.id) for c in children],
            "dedupe_key": dedupe_key,
            "operation_id": str(operation_id),
            "decompose_mode": request.mode,
        },
    )
    # 16c — explicit retire audit row when split (mirrors mem_supersede /
    # compose-merge per-source audit shape).
    if request.mode == "split":
        await _record_audit(
            s,
            op="retire",
            memory=source,
            by_agent_id=ctx.agent_id,
            before=source_before,
            after=_audit_snapshot(source, tag_names=source_tag_names),
            extra_after={
                "retired_via": "mem_decompose:split",
                "operation_id": str(operation_id),
            },
        )

    # Step 17 — outbox events.
    # 17a — per-child upsert (Qdrant vector + Neo4j node create).
    for idx, child in enumerate(children):
        await enqueue_event(
            s,
            aggregate_type=OutboxAggregateType.memory,
            aggregate_id=child.id,
            aggregate_version=child.version,
            env_id=env_id,
            op=_outbox_op_for(MemoryStatus.active, is_create=True),
            payload=_projection_payload(
                child,
                tag_names=child_tag_names[idx],
                embedding_model_id=embedding_model_id,
            ),
            settings=settings,
        )
    # 17b — tombstone for retired source under split (Qdrant drop).
    if request.mode == "split":
        await enqueue_event(
            s,
            aggregate_type=OutboxAggregateType.memory,
            aggregate_id=source.id,
            aggregate_version=source.version,
            env_id=env_id,
            op=_outbox_op_for(MemoryStatus.retired, is_create=False),
            payload=_projection_payload(
                source,
                tag_names=source_tag_names,
                embedding_model_id=embedding_model_id,
            ),
            settings=settings,
        )

    # Step 18 — build response. session_scope() commits on exit.
    return await _build_response(
        s,
        source=source,
        children=children,
        mode=request.mode,
        lineage_rows=lineage_rows,
        operation_id=operation_id,
        dedupe_key=dedupe_key,
        idempotency_replay=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def memory_decompose(
    request: MemDecomposeRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemDecomposeResponse:
    """Decompose a source memory into N≥2 children.

    See module docstring + :class:`MemDecomposeRequest` /
    :class:`MemDecomposeResponse` for the contract. Dispatches to a
    private in-session helper so the dream worker can eventually call
    the same path without the outer ``session_scope``.
    """
    settings = settings or get_settings()
    async with session_scope() as s:
        return await _decompose_in_session(
            s, request=request, ctx=ctx, settings=settings,
        )
