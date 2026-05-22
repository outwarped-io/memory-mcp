"""Compose handler: caller-driven N→1 memory aggregation (v0.15.0 Phase 2).

This module is the entry point for the ``mem_compose`` MCP tool. The
runtime contract is locked in by the Stage B1 design decision (see
``tasks/.../subtasks/.../plan.md`` Stage B). The transaction body lives
in this module so the dream worker handlers (``_accept_merge`` /
``_accept_promotion``) can eventually delegate here once parity tests
prove the refactor is safe.

B3c (this commit) adds the deterministic dedupe-key helper. The atomic
transaction body lands at B3d.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from uuid import UUID

from memory_mcp.errors import MemoryMCPError
from memory_mcp.identity import AgentContext
from memory_mcp_schemas.compose import (
    ComposeLineageRow,
    ComposeMode,
    ComposeTagPolicy,
    MemComposeRequest,
    MemComposeResponse,
    MemComposeTarget,
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
    """B2 stub — raised until the B3d transaction body lands."""

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
) -> MemComposeResponse:
    """Compose N≥2 source memories into a single new memory.

    B3c stub — schema validation runs via Pydantic; a real call still
    raises so callers can detect the missing handler cleanly. B3d lands
    the transaction body that resolves env_id (from the locked source
    rows) and persists the dedupe key returned by
    :func:`_compute_compose_dedupe_key`.
    """
    raise ComposeNotImplementedError(
        "mem_compose handler not yet implemented in this build. "
        "Schema validation succeeded; transaction body lands in v0.15.0 Phase 2 B3d."
    )
