"""Decompose handler: caller-driven 1→N memory decomposition (v0.15.0 Phase 3).

This module is the entry point for the ``mem_decompose`` MCP tool. The
runtime contract is locked in by the Stage C1 design decision (see
``tasks/.../subtasks/.../plan.md`` Stage C). The transaction body lands
in C6 once C5 (validation) ships; this module (post-C4) carries the
shared idempotency primitives that both the transaction body and its
tests consume.

Structure mirrors :mod:`memory_mcp.composers` so the dream worker can
eventually delegate decompositions through here once the semantics line
up with whatever dream-side decompose proposal lands.

Provenance convention (RD G note): each decomposed child is recorded
with ``MemorySource(source_type='agent', source_ref=str(operation.id))``
where ``operation.id`` is the ``decompose_operations`` row UUID. The
audit log carries the ``op='mem_decompose:{mode}'`` distinction;
``source_ref`` opaquely points back to the operation row so a downstream
tool that wants "this memory came from decompose op X" can query
``decompose_operations`` directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from memory_mcp.db.models import Memory
from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.errors import (
    InvalidInputError,
    InvalidTransitionError,
    NotFoundError,
    VersionConflictError,
    MemoryMCPError,
)
from memory_mcp.identity import AgentContext
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


def _canonical_child_payload_full(child: MemDecomposeChild) -> dict[str, Any]:
    """Full identity + policy payload for the request fingerprint.

    Adds the fields the dedupe-key intentionally drops:
    ``trigger_description`` and ``expires_at``. Two requests that share a
    dedupe-key but differ in fingerprint mean the caller reused
    ``idempotency_key`` (or hit a sha256 collision — astronomically
    unlikely) with a different scope, and the server raises
    ``InvalidInputError`` rather than silently substituting different
    content on replay.
    """
    payload = _canonical_child_payload(child)
    payload["trigger_description"] = child.trigger_description
    payload["expires_at"] = (
        child.expires_at.isoformat() if child.expires_at is not None else None
    )
    return payload


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
    """Always-canonical sha256 of the full request envelope (32 hex).

    Distinct from :func:`_compute_decompose_dedupe_key`:

    * The dedupe key respects ``idempotency_key`` (returns it verbatim);
      the fingerprint does NOT — it always reflects the actual request
      scope.
    * Stored on ``decompose_operations.request_fingerprint``. The C6
      transaction body compares the incoming fingerprint to the stored
      row; on mismatch the call raises
      ``InvalidInputError("idempotency_key reused with different scope")``
      so callers who reuse a key with a different source / different
      children / different mode are detected rather than receiving a
      silently-substituted replay.

    Payload differs from the dedupe-key payload by:

    * ``operation`` is ``"mem_decompose_fp"`` (domain separator so the
      key and fingerprint hash spaces don't accidentally collide).
    * ``expected_version`` is included.
    * Each child is hashed via
      :func:`_canonical_child_payload_full`, which restores the
      ``trigger_description`` and ``expires_at`` fields excluded from
      the dedupe key.
    * Children list is NOT re-sorted at fingerprint time. The request
      order matters for the fingerprint (a caller resubmitting children
      in a different order is a different request envelope; the dedupe
      key absorbs that re-ordering deliberately, but the fingerprint
      should not).
    """
    payload: dict[str, Any] = {
        "schema_version": _DEDUPE_KEY_SCHEMA_VERSION,
        "operation": "mem_decompose_fp",
        "env_id": str(env_id),
        "mode": request.mode,
        "source_id": str(request.source_id),
        "expected_version": request.expected_version,
        "children": [_canonical_child_payload_full(c) for c in request.children],
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
# Entry point (C3 stub — C6 lands the real transaction body)
# ---------------------------------------------------------------------------


async def memory_decompose(
    request: MemDecomposeRequest,
    *,
    ctx: AgentContext,
) -> MemDecomposeResponse:
    """Decompose a source memory into N≥2 children.

    C3 stub — the surface is wired (request validation runs via Pydantic;
    a real call still raises so callers can detect the missing handler
    cleanly). The transaction body lands in v0.15.0 Phase 3 C6.
    """
    raise DecomposeNotImplementedError(
        "mem_decompose handler not yet implemented in this build. "
        "Schema validation succeeded; transaction body lands in v0.15.0 Phase 3 C6."
    )
